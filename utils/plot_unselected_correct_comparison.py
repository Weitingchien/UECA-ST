#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
根據 unselected_correct_summary.txt 繪製多實驗組的比較柱狀圖

可繪製指標:
  - correct_rate:   all_doc_correct / total_docs     (模型全對的比例)
  - selection_rate:  selected / total_docs            (被選中的比例)
  - miss_rate:       correct_but_not_selected / all_doc_correct  (正確但被遺漏的比例)
  - unselected_correct_density:
                     correct_but_not_selected / (total_docs - selected)
                     (未被選中的文件中，其實是正確的比例)

計算方式:
  1) 每個 seed 的 unselected_correct_summary.txt → 每 fold 一組數值 → 10 折平均
  2) 跨 seed: 3 個 seed 的平均 ± std

用法範例:
  python utils/plot_unselected_correct_comparison.py \
    --summary-files \
      results_ep.../..._consistency_CE_summary.txt \
      results_ep.../..._CE_summary.txt \
    --experiments-root ep_split10_t1te1v1_u7_disjoint

  # 手動指定標籤
  python utils/plot_unselected_correct_comparison.py \
    --summary-files summary1.txt summary2.txt \
    --labels "Consistency" "Baseline" \
    --experiments-root ep_split10_t1te1v1_u7_disjoint

  # 指定繪製的指標 (可多選)
  python utils/plot_unselected_correct_comparison.py \
    --summary-files summary1.txt summary2.txt \
    --experiments-root ep_split10_t1te1v1_u7_disjoint \
    --metrics correct_rate miss_rate
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from dataclasses import dataclass

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# ── 字體設定：與 plot_val_pair_f1_per_round.py 一致 ─────────
plt.rcParams['font.family'] = ['Calisto MT', 'DFKai-SB', 'DejaVu Serif', 'Noto Serif CJK JP']
plt.rcParams['axes.unicode_minus'] = False

FONT_CHINESE = 'DFKai-SB'   # 標楷體
FONT_ENGLISH = 'Calisto MT'


# ══════════════════════════════════════════════════════════════
#  指標定義
# ══════════════════════════════════════════════════════════════

METRIC_CHOICES = [
    'correct_rate',
    'selection_rate',
    'miss_rate',
    'unselected_correct_density',
]

METRIC_DISPLAY = {
    'correct_rate':               '文檔準確率\n(全對文檔數 / 未標註文檔總數)',
    'selection_rate':             '選中率\n(選中文檔數 / 未標註文檔總數)',
    'miss_rate':                  '遺漏率\n(未選中正確數 / 全對文檔數)',
    'unselected_correct_density': '未選中正確密度\n(未選中正確數 / 未選中總數)',
}

METRIC_DISPLAY_SHORT = {
    'correct_rate':               '文檔準確率',
    'selection_rate':             '選中率',
    'miss_rate':                  '遺漏率',
    'unselected_correct_density': '未選中正確密度',
}


# ══════════════════════════════════════════════════════════════
#  解析工具 (與 plot_val_pair_f1_per_round.py 一致)
# ══════════════════════════════════════════════════════════════

def read_experiment_names_from_summary(summary_file: Path) -> list[str]:
    """從 summary txt 解析所有 prompt_ECPE_few_shot_ST_... 實驗目錄名稱"""
    text = summary_file.read_text(encoding="utf-8", errors="ignore")
    names: list[str] = []
    for line in text.splitlines():
        m = re.match(r"\s*-\s*(prompt_ECPE_few_shot_ST_[^\s]+)", line)
        if m:
            names.append(m.group(1).strip())
    return names


def extract_seed(dirname: str) -> str:
    """從實驗目錄名稱中抽取 seed 值"""
    m = re.search(r"seed(\d+)", dirname)
    return m.group(1) if m else dirname


def get_short_label(name: str) -> str:
    """
    從資料夾名稱或 summary 檔名擷取可辨識的簡短標籤。
    (與 plot_val_pair_f1_per_round.py 完全一致)

    規則：
      1) 去掉 _summary.txt 後綴
      2) 從 'th' 開始擷取；若無 th 則從 'nest_k5_' 開始擷取
      3) 移除共用參數片段:
         _gamma[\d.]+, _st\d+, _ste\d+, _seed\d+, _retain_pseudo,
         _nbeta[\d.]+, _nm[\d.]+
      4) 移除 knncause 前面的冗餘 cause(_clause)?_
    """
    s = re.sub(r'_summary\.txt$', '', name, flags=re.IGNORECASE)
    # 特殊情況：_st0_ 表示僅 few-shot 無自訓練
    if '_st0_' in s or s.endswith('_st0'):
        suffix_match = re.search(r'_(CE|EC)$', s)
        suffix = f"_{suffix_match.group(1)}" if suffix_match else ""
        return f"st0{suffix}"
    m = re.search(r'(th[\d.]+_.*)', s)
    if m:
        s = m.group(1)
    else:
        m2 = re.search(r'(nest_k\d+_.*)', s)
        if m2:
            s = m2.group(1)
    s = re.sub(r'_seed\d+', '', s)
    s = re.sub(r'_gamma[\d.]+', '', s)
    s = re.sub(r'_st\d+', '', s)
    s = re.sub(r'_ste\d+', '', s)
    s = re.sub(r'_retain_pseudo', '', s)
    s = re.sub(r'_nbeta[\d.]+', '', s)
    s = re.sub(r'_nm[\d.]+', '', s)
    s = re.sub(r'cause(_clause)?_(?=knncause)', '', s)
    s = re.sub(r'__+', '_', s)
    s = s.strip('_')
    return s


def find_pseudo_results_dir(exp_dir: Path) -> Path:
    """找出最新的 pseudo_results_* 子資料夾"""
    candidates = sorted([p for p in exp_dir.glob("pseudo_results_*") if p.is_dir()])
    if not candidates:
        raise FileNotFoundError(f"{exp_dir} 底下找不到 pseudo_results_* 資料夾")
    return candidates[-1]


def resolve_experiments_roots(args: argparse.Namespace) -> list[Path]:
    """解析並驗證實驗根目錄 (支援單一路徑與多路徑)"""
    if getattr(args, "experiments_roots", None):
        roots = [Path(p) for p in args.experiments_roots]
    elif getattr(args, "experiments_root", None):
        roots = [Path(args.experiments_root)]
    else:
        roots = []

    missing = [str(p) for p in roots if not p.is_dir()]
    if missing:
        raise FileNotFoundError(
            "以下 experiments root 不存在:\n" + "\n".join(missing)
        )
    return roots


def resolve_experiment_dir(exp_name: str, roots: list[Path]) -> Path:
    """在多個根目錄中解析單一實驗資料夾。"""
    hits = [root / exp_name for root in roots if (root / exp_name).is_dir()]
    if not hits:
        searched = "\n".join(str(root / exp_name) for root in roots)
        raise FileNotFoundError(
            f"找不到實驗目錄: {exp_name}\n已搜尋:\n{searched}"
        )

    if len(hits) > 1:
        print(
            f"[警告] 實驗目錄 {exp_name} 在多個 roots 都存在，"
            f"將使用第一個: {hits[0]}"
        )
    return hits[0]


# ══════════════════════════════════════════════════════════════
#  讀取 unselected_correct_summary.txt
# ══════════════════════════════════════════════════════════════

@dataclass
class FoldData:
    fold: int
    total_docs: int
    all_doc_correct: int
    selected: int
    correct_but_not_selected: int


def parse_unselected_correct_summary(path: Path) -> list[FoldData]:
    """解析 unselected_correct_summary.txt，回傳各 fold 的數據"""
    if not path.exists():
        raise FileNotFoundError(f"找不到: {path}")

    text = path.read_text(encoding="utf-8", errors="ignore")
    folds: list[FoldData] = []

    # 匹配: fold1: total_docs=1389, all_doc_correct=283, selected=1128, correct_but_not_selected=25
    pat = re.compile(
        r"fold(\d+):\s*total_docs=(\d+),\s*all_doc_correct=(\d+),\s*"
        r"selected=(\d+),\s*correct_but_not_selected=(\d+)"
    )
    for line in text.splitlines():
        m = pat.match(line.strip())
        if m:
            folds.append(FoldData(
                fold=int(m.group(1)),
                total_docs=int(m.group(2)),
                all_doc_correct=int(m.group(3)),
                selected=int(m.group(4)),
                correct_but_not_selected=int(m.group(5)),
            ))
    return folds


def compute_fold_metric(fold: FoldData, metric: str) -> float:
    """計算單一 fold 的指定指標"""
    if metric == 'correct_rate':
        return fold.all_doc_correct / fold.total_docs if fold.total_docs else 0
    elif metric == 'selection_rate':
        return fold.selected / fold.total_docs if fold.total_docs else 0
    elif metric == 'miss_rate':
        return fold.correct_but_not_selected / fold.all_doc_correct if fold.all_doc_correct else 0
    elif metric == 'unselected_correct_density':
        unselected = fold.total_docs - fold.selected
        return fold.correct_but_not_selected / unselected if unselected else 0
    else:
        raise ValueError(f"未知的指標: {metric}")


def get_metric_fraction(fold: FoldData, metric: str) -> tuple[int, int]:
    """回傳指定指標在單一 fold 的 (分子, 分母)。"""
    if metric == 'correct_rate':
        return fold.all_doc_correct, fold.total_docs
    if metric == 'selection_rate':
        return fold.selected, fold.total_docs
    if metric == 'miss_rate':
        return fold.correct_but_not_selected, fold.all_doc_correct
    if metric == 'unselected_correct_density':
        return fold.correct_but_not_selected, (fold.total_docs - fold.selected)
    raise ValueError(f"未知的指標: {metric}")


# ══════════════════════════════════════════════════════════════
#  資料收集
# ══════════════════════════════════════════════════════════════

def collect_group_metrics(
    exp_dirs: list[Path],
    metrics: list[str],
) -> dict[str, dict[str, float]]:
    """
    回傳結構:
    {
      seed_label: {
        metric_name: 10折平均值,
        ...
      },
      ...
    }
    """
    result: dict[str, dict[str, float]] = {}

    for exp_dir in exp_dirs:
        seed_label = f"seed{extract_seed(exp_dir.name)}"
        pseudo_dir = find_pseudo_results_dir(exp_dir)
        summary_path = pseudo_dir / "unselected_correct_summary.txt"

        folds = parse_unselected_correct_summary(summary_path)
        if not folds:
            print(f"  [警告] {summary_path} 未解析到任何 fold 數據")
            continue

        seed_metrics: dict[str, float] = {}
        for metric in metrics:
            vals = [compute_fold_metric(f, metric) for f in folds]
            seed_metrics[metric] = float(np.mean(vals))

        result[seed_label] = seed_metrics

    return result


def collect_group_fold_series(
    exp_dirs: list[Path],
    metric: str,
) -> dict:
    """收集單一實驗組在每 fold 的指標序列與分子/分母統計。"""
    per_seed: dict[str, dict[int, dict[str, float]]] = {}
    all_folds: set[int] = set()

    for exp_dir in exp_dirs:
        seed_label = f"seed{extract_seed(exp_dir.name)}"
        pseudo_dir = find_pseudo_results_dir(exp_dir)
        summary_path = pseudo_dir / "unselected_correct_summary.txt"
        folds = parse_unselected_correct_summary(summary_path)
        if not folds:
            print(f"  [警告] {summary_path} 未解析到任何 fold 數據")
            continue

        fold_map: dict[int, dict[str, float]] = {}
        for f in folds:
            num, den = get_metric_fraction(f, metric)
            rate = (num / den) if den else 0.0
            fold_map[f.fold] = {
                "num": int(num),
                "den": int(den),
                "rate": float(rate),
            }
            all_folds.add(f.fold)

        per_seed[seed_label] = fold_map

    fold_ids = sorted(all_folds)
    pooled_num: list[int] = []
    pooled_den: list[int] = []
    pooled_rate: list[float] = []
    mean_rate: list[float] = []
    std_rate: list[float] = []

    for fold in fold_ids:
        seed_rates: list[float] = []
        fold_num_sum = 0
        fold_den_sum = 0
        for seed in sorted(per_seed.keys()):
            one = per_seed[seed].get(fold)
            if not one:
                continue
            seed_rates.append(float(one["rate"]))
            fold_num_sum += int(one["num"])
            fold_den_sum += int(one["den"])

        pooled_num.append(fold_num_sum)
        pooled_den.append(fold_den_sum)
        pooled_rate.append((fold_num_sum / fold_den_sum) if fold_den_sum else 0.0)
        mean_rate.append(float(np.mean(seed_rates)) if seed_rates else 0.0)
        std_rate.append(float(np.std(seed_rates)) if seed_rates else 0.0)

    return {
        "folds": fold_ids,
        "per_seed": per_seed,
        "pooled_num": pooled_num,
        "pooled_den": pooled_den,
        "pooled_rate": pooled_rate,
        "mean_rate": mean_rate,
        "std_rate": std_rate,
    }


def collect_group_metric_fraction(
    exp_dirs: list[Path],
    metric: str,
) -> tuple[int, int]:
    """收集單一實驗組在所有 seed 與所有 fold 的加總分子/分母。"""
    num_sum = 0
    den_sum = 0

    for exp_dir in exp_dirs:
        pseudo_dir = find_pseudo_results_dir(exp_dir)
        summary_path = pseudo_dir / "unselected_correct_summary.txt"
        folds = parse_unselected_correct_summary(summary_path)
        if not folds:
            print(f"  [警告] {summary_path} 未解析到任何 fold 數據")
            continue

        for f in folds:
            num, den = get_metric_fraction(f, metric)
            num_sum += int(num)
            den_sum += int(den)

    return num_sum, den_sum


# ══════════════════════════════════════════════════════════════
#  繪圖
# ══════════════════════════════════════════════════════════════

# 與 plot_val_pair_f1_per_round.py 一致的配色
# 前 9 色 (index 0-8) 對應原有 9 組實驗，index 9 保留給 consistency
DISTINCT_COLORS = [
    "#d62728",  # 紅       (nest_k5_nestmul3)
    "#1f77b4",  # 藍       (nest_k5_nestmul5)
    "#2ca02c",  # 綠       (hybrid_alt)
    "#ff7f0e",  # 橙       (hybrid_rev_switch5)
    "#9467bd",  # 紫       (hybrid_switch5)
    "#e377c2",  # 粉紅     (maskall)
    "#17becf",  # 青       (maskboth)
    "#8c564b",  # 棕       (maskcause)
    "#bcbd22",  # 黃綠     (maskemotion)
    "#000000",  # 黑       (consistency)
    "#393b79",  # 深藍紫
    "#e7969c",  # 淡珊瑚
]

# 與 pair_m1_multi_seed_comparison_precision.png 對齊的固定顏色映射
LABEL_FIXED_COLORS = {
    'st0_CE': '#d62728',
    'th0.9_maskemotion_consistency_CE': '#1f77b4',
    'th0.9_maskemotion_CE': '#2ca02c',
    'th0.9_maskemotion_somc_CE': '#4A148C',
    'nest_k5_knncause_clause_mask_nestmul5_somc_CE': '#9467bd',
    'nest_k5_knncause_clause_mask_nestmul5_CE': '#e377c2',
    'consistency_only_all_equal_CE': '#1f77b4',
    'nest_k5_knncause_clause_mask_nestmul7_consistency_CE': '#2ca02c',
    'nest_k5_knncause_clause_mask_nestmul7_CE': '#ff7f0e',
}


def normalize_label_for_color(label: str) -> str:
    """將標籤正規化後再做固定配色比對。"""
    s = label.strip()
    s = re.sub(r'_consistency_somc_', '_somc_', s)
    return s


def get_color_by_label(label: str, idx: int) -> str:
    """優先使用固定顏色映射，未命中時退回循序色盤。"""
    normalized = normalize_label_for_color(label)
    return LABEL_FIXED_COLORS.get(normalized, DISTINCT_COLORS[idx % len(DISTINCT_COLORS)])


def assign_distinct_colors(labels: list[str]) -> list[str]:
    """為當前圖上的標籤分配顏色，盡量避免同圖撞色。"""
    assigned: list[str | None] = []
    used: set[str] = set()

    for label in labels:
        normalized = normalize_label_for_color(label)
        fixed = LABEL_FIXED_COLORS.get(normalized)
        if fixed is not None:
            assigned.append(fixed)
            used.add(fixed.lower())
        else:
            assigned.append(None)

    for i, color in enumerate(assigned):
        if color is not None:
            continue

        chosen = None
        for candidate in DISTINCT_COLORS:
            if candidate.lower() not in used:
                chosen = candidate
                break

        if chosen is None:
            chosen = DISTINCT_COLORS[i % len(DISTINCT_COLORS)]

        assigned[i] = chosen
        used.add(chosen.lower())

    return [c if c is not None else DISTINCT_COLORS[i % len(DISTINCT_COLORS)] for i, c in enumerate(assigned)]


def plot_comparison(
    all_group_metrics: list[dict[str, dict[str, float]]],
    labels: list[str],
    metrics: list[str],
    output_path: Path,
    title: str | None = None,
    no_std_bar: bool = False,
    fraction_labels: list[str] | None = None,
) -> None:
    """
    繪製分組柱狀圖比較多個實驗組

    all_group_metrics: [group1_data, group2_data, ...]
      每個 group_data = { seed_label: { metric: value, ... }, ... }
    """
    n_metrics = len(metrics)
    n_groups = len(labels)

    # 計算每組每個指標的 mean ± std
    group_means: list[list[float]] = []   # [group_idx][metric_idx]
    group_stds: list[list[float]] = []

    for group_data in all_group_metrics:
        means = []
        stds = []
        seed_labels = sorted(group_data.keys())
        for metric in metrics:
            vals = [group_data[s][metric] for s in seed_labels]
            means.append(float(np.mean(vals)) * 100)
            stds.append(float(np.std(vals)) * 100)
        group_means.append(means)
        group_stds.append(stds)

    # 單一 metric 時採用與 pair_m1_multi_seed_comparison_f1.png 相同視覺樣式
    if n_metrics == 1:
        x = np.arange(n_groups)
        values = [group_means[g][0] for g in range(n_groups)]
        stds = [group_stds[g][0] for g in range(n_groups)]
        colors = assign_distinct_colors(labels)

        fig, ax = plt.subplots(figsize=(14, 7))
        fig.patch.set_facecolor('#DDDDDD')
        ax.set_facecolor('#DDDDDD')

        bar_kwargs = dict(color=colors, edgecolor='white', linewidth=1.5, width=0.5)
        if not no_std_bar:
            bar_kwargs["yerr"] = stds
            bar_kwargs["capsize"] = 5
        bars = ax.bar(x, values, **bar_kwargs)

        for i, (bar, mean_v, std_v) in enumerate(zip(bars, values, stds)):
            top_offset = 0.5 if no_std_bar else std_v + 0.5
            text_lines = [f'{mean_v:.2f}%']
            if fraction_labels is not None and i < len(fraction_labels):
                text_lines.append(f"({fraction_labels[i]})")
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + top_offset,
                "\n".join(text_lines),
                ha='center', va='bottom',
                fontsize=11, fontname=FONT_ENGLISH,
                fontweight='bold',
                color=colors[i],
            )

        if title is None:
            title = f"{METRIC_DISPLAY_SHORT[metrics[0]]}"
        ax.set_title(title, fontsize=22, fontweight='bold', fontname=FONT_ENGLISH)
        ax.set_ylabel('Score (%)', fontsize=18, fontname=FONT_ENGLISH)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=12, fontname=FONT_ENGLISH, rotation=15, ha='right')

        y_max = max(values) + (max(stds) if stds else 0)
        ax.set_ylim(0, y_max + 12)

        for lbl in ax.get_yticklabels():
            lbl.set_fontsize(16)
            lbl.set_fontname(FONT_ENGLISH)

        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        fig.savefig(str(output_path), dpi=150, bbox_inches='tight')
        print(f"圖片已儲存: {output_path}")
        plt.close(fig)
        return

    # 多 metric 保留原本分組柱狀圖樣式
    bar_width = 0.8 / n_groups
    x = np.arange(n_metrics)

    fig, ax = plt.subplots(figsize=(max(10, 4 * n_metrics), 8))
    fig.patch.set_facecolor('#DDDDDD')
    ax.set_facecolor('#DDDDDD')

    colors = assign_distinct_colors(labels)
    for g_idx, label in enumerate(labels):
        offset = (g_idx - (n_groups - 1) / 2) * bar_width
        group_color = colors[g_idx]
        bar_kwargs = dict(
            height=group_means[g_idx],
            width=bar_width * 0.9,
            label=f"{label}",
            color=group_color,
            alpha=0.85,
            edgecolor='white',
            linewidth=0.5,
        )
        if not no_std_bar:
            bar_kwargs["yerr"] = group_stds[g_idx]
            bar_kwargs["capsize"] = 5
        bars = ax.bar(x + offset, **bar_kwargs)
        for bar, mean_v, std_v in zip(bars, group_means[g_idx], group_stds[g_idx]):
            top_offset = 0.3 if no_std_bar else std_v + 0.3
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + top_offset,
                f'{mean_v:.1f}%',
                ha='center', va='bottom',
                fontsize=11, fontname=FONT_ENGLISH,
                fontweight='bold',
                color=group_color,
            )

    if title is None:
        title = "未標註文檔全對率"
    ax.set_title(title, fontsize=24, fontweight='bold', fontname=FONT_CHINESE)
    ax.set_ylabel('比例 (%)', fontsize=24, fontname=FONT_CHINESE)
    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_DISPLAY[m] for m in metrics], fontsize=20, fontname=FONT_CHINESE)

    for lbl in ax.get_yticklabels():
        lbl.set_fontsize(20)
        lbl.set_fontname(FONT_ENGLISH)

    legend = ax.legend(fontsize=14, loc='lower left')
    legend.get_frame().set_facecolor('#DDDDDD')
    for text in legend.get_texts():
        has_cjk = any('\u4e00' <= ch <= '\u9fff' for ch in text.get_text())
        text.set_fontname(FONT_CHINESE if has_cjk else FONT_ENGLISH)

    ax.grid(True, axis='y', linestyle='--', alpha=0.7)

    fig.subplots_adjust(left=0.10, bottom=0.15, right=0.97, top=0.93)
    fig.savefig(str(output_path), dpi=150, bbox_inches='tight', pad_inches=0.3)
    print(f"圖片已儲存: {output_path}")
    plt.close(fig)


def plot_fold_line_comparison(
    fold_series_list: list[dict],
    labels: list[str],
    metric: str,
    output_path: Path,
    title: str | None = None,
) -> None:
    """繪製每 fold 的折線圖（預設使用 pooled 分子/分母換算比例）。"""
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor('#DDDDDD')
    ax.set_facecolor('#DDDDDD')

    for idx, (label, series) in enumerate(zip(labels, fold_series_list)):
        x = series["folds"]
        y = [v * 100 for v in series["pooled_rate"]]
        color = get_color_by_label(label, idx)
        ax.plot(
            x,
            y,
            marker='o',
            linewidth=2.2,
            markersize=6,
            color=color,
            label=label,
        )

    if title is None:
        title = f"每 fold 比例折線圖 ({METRIC_DISPLAY_SHORT[metric]})"
    ax.set_title(title, fontsize=24, fontweight='bold', fontname=FONT_CHINESE)
    ax.set_xlabel('Fold', fontsize=22, fontname=FONT_ENGLISH)
    ax.set_ylabel('比例 (%)', fontsize=24, fontname=FONT_CHINESE)

    for lbl in ax.get_xticklabels():
        lbl.set_fontsize(16)
        lbl.set_fontname(FONT_ENGLISH)
    for lbl in ax.get_yticklabels():
        lbl.set_fontsize(18)
        lbl.set_fontname(FONT_ENGLISH)

    ax.grid(True, linestyle='--', alpha=0.7)
    legend = ax.legend(fontsize=13, loc='best')
    legend.get_frame().set_facecolor('#DDDDDD')
    for text in legend.get_texts():
        has_cjk = any('\u4e00' <= ch <= '\u9fff' for ch in text.get_text())
        text.set_fontname(FONT_CHINESE if has_cjk else FONT_ENGLISH)

    fig.subplots_adjust(left=0.10, bottom=0.12, right=0.97, top=0.93)
    fig.savefig(str(output_path), dpi=150, bbox_inches='tight', pad_inches=0.3)
    print(f"圖片已儲存: {output_path}")
    plt.close(fig)


def plot_group_line_comparison(
    all_group_metrics: list[dict[str, dict[str, float]]],
    labels: list[str],
    metric: str,
    output_path: Path,
    title: str | None = None,
    no_std_bar: bool = False,
    fraction_labels: list[str] | None = None,
) -> None:
    """繪製各組平均值的折線圖（不展開每 fold）。"""
    x = np.arange(len(labels))
    y_mean: list[float] = []
    y_std: list[float] = []

    for group_data in all_group_metrics:
        seed_labels = sorted(group_data.keys())
        vals = [group_data[s][metric] * 100 for s in seed_labels]
        y_mean.append(float(np.mean(vals)) if vals else 0.0)
        y_std.append(float(np.std(vals)) if vals else 0.0)

    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor('#DDDDDD')
    ax.set_facecolor('#DDDDDD')

    colors = [get_color_by_label(lbl, i) for i, lbl in enumerate(labels)]

    # 主折線
    ax.plot(x, y_mean, color='#333333', linewidth=1.8, alpha=0.9, zorder=2)

    # 各組點
    for i, (label, mean_v) in enumerate(zip(labels, y_mean)):
        ax.scatter(x[i], mean_v, color=colors[i], s=90, zorder=3)
        text_lines = [f"{mean_v:.1f}%"]
        if fraction_labels is not None and i < len(fraction_labels):
            text_lines.append(f"({fraction_labels[i]})")
        ax.text(
            x[i],
            mean_v + (0.2 if no_std_bar else y_std[i] + 0.2),
            "\n".join(text_lines),
            ha='center', va='bottom',
            fontsize=11, fontname=FONT_ENGLISH, fontweight='bold',
            color=colors[i],
        )

    if not no_std_bar:
        ax.errorbar(
            x,
            y_mean,
            yerr=y_std,
            fmt='none',
            ecolor='#666666',
            elinewidth=1.2,
            capsize=4,
            zorder=1,
        )

    if title is None:
        title = f"{METRIC_DISPLAY_SHORT[metric]} (組別平均折線圖)"
    ax.set_title(title, fontsize=24, fontweight='bold', fontname=FONT_CHINESE)
    ax.set_ylabel('比例 (%)', fontsize=24, fontname=FONT_CHINESE)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=14, fontname=FONT_ENGLISH, rotation=15, ha='right')

    for lbl in ax.get_yticklabels():
        lbl.set_fontsize(18)
        lbl.set_fontname(FONT_ENGLISH)

    ax.grid(True, linestyle='--', alpha=0.7)
    fig.subplots_adjust(left=0.10, bottom=0.20, right=0.97, top=0.93)
    fig.savefig(str(output_path), dpi=150, bbox_inches='tight', pad_inches=0.3)
    print(f"圖片已儲存: {output_path}")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════
#  文字報告
# ══════════════════════════════════════════════════════════════

def write_text_report(
    all_group_metrics: list[dict[str, dict[str, float]]],
    labels: list[str],
    metrics: list[str],
    output_path: Path,
) -> None:
    """輸出人類可讀的文字摘要"""
    lines: list[str] = []
    lines.append("=" * 70)
    lines.append("未標註文檔全對率 比較報告")
    lines.append("=" * 70)

    for g_idx, (group_data, label) in enumerate(zip(all_group_metrics, labels)):
        lines.append(f"\n{'─' * 70}")
        lines.append(f"  {label}")
        lines.append(f"{'─' * 70}")

        seed_labels = sorted(group_data.keys())

        # 表頭
        seed_header = "  ".join(f"{s:>10s}" for s in seed_labels)
        lines.append(f"  {'Metric':>35s}  {seed_header}  {'Mean':>8s}  {'Std':>7s}")
        lines.append(f"  {'─' * (37 + 12 * len(seed_labels) + 20)}")

        for metric in metrics:
            vals = np.array([group_data[s][metric] for s in seed_labels]) * 100
            mean_v = np.mean(vals)
            std_v = np.std(vals)
            val_str = "  ".join(f"{v:10.2f}%" for v in vals)
            lines.append(
                f"  {METRIC_DISPLAY_SHORT[metric]:>35s}  {val_str}  {mean_v:7.2f}%  {std_v:6.2f}%"
            )

    lines.append(f"\n{'=' * 70}")
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"文字報告已儲存: {output_path}")


def write_fold_fraction_report(
    fold_series_list: list[dict],
    labels: list[str],
    metric: str,
    output_path: Path,
) -> None:
    """輸出每 fold 的分子/分母/比例明細。"""
    lines: list[str] = []
    lines.append("=" * 90)
    lines.append(f"每 fold 分子分母明細 ({METRIC_DISPLAY_SHORT[metric]})")
    lines.append("=" * 90)

    for label, series in zip(labels, fold_series_list):
        lines.append(f"\n{'─' * 90}")
        lines.append(f"  {label}")
        lines.append(f"{'─' * 90}")
        lines.append("  fold | pooled_num | pooled_den | pooled_rate")
        lines.append("  " + "-" * 66)
        for fold, num, den, rate in zip(
            series["folds"],
            series["pooled_num"],
            series["pooled_den"],
            series["pooled_rate"],
        ):
            lines.append(f"  {fold:>4d} | {num:>10d} | {den:>10d} | {rate*100:>9.2f}%")

    lines.append(f"\n{'=' * 90}")
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"每折明細已儲存: {output_path}")


# ══════════════════════════════════════════════════════════════
#  main
# ══════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="根據 unselected_correct_summary.txt 繪製多實驗組比較柱狀圖",
    )
    p.add_argument(
        "--summary-files",
        nargs="+",
        required=True,
        help="一或多個 summary txt 路徑 (multi-seed summary)",
    )
    p.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="每個 summary 對應的顯示名稱。"
             "若不提供則自動從 summary 檔名擷取特徵片段",
    )
    exp_root_group = p.add_mutually_exclusive_group(required=True)
    exp_root_group.add_argument(
        "--experiments-root",
        help="單一實驗根目錄，例如 ep_split10_t1te1v1_u7_disjoint",
    )
    exp_root_group.add_argument(
        "--experiments-roots",
        nargs="+",
        help="多個實驗根目錄，會依序搜尋 summary 內的實驗資料夾",
    )
    p.add_argument(
        "--metrics",
        nargs="+",
        choices=METRIC_CHOICES,
        default=METRIC_CHOICES,
        help=f"要繪製的指標 (預設全部: {METRIC_CHOICES})",
    )
    p.add_argument(
        "--output-dir",
        default="png",
        help="輸出目錄",
    )
    p.add_argument(
        "--output-name",
        default="unselected_correct_comparison",
        help="輸出檔名前綴 (不含副檔名)",
    )
    p.add_argument(
        "--title",
        default=None,
        help="圖表標題 (預設自動產生)",
    )
    p.add_argument(
        "--no-std-bar",
        action="store_true",
        default=False,
        help="不顯示標準差 error bar",
    )
    p.add_argument(
        "--plot-style",
        choices=["bar", "line-group", "line-fold"],
        default="bar",
        help="繪圖樣式: bar=柱狀圖, line-group=各組平均折線圖, line-fold=每 fold 折線圖",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── 自動產生 labels ──
    if args.labels is not None:
        if len(args.summary_files) != len(args.labels):
            raise ValueError(
                f"--summary-files ({len(args.summary_files)}) 與 "
                f"--labels ({len(args.labels)}) 數量不一致"
            )
        labels = args.labels
    else:
        labels = [get_short_label(Path(sf).name) for sf in args.summary_files]
        print(f"自動產生標籤: {labels}")

    roots = resolve_experiments_roots(args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # ── 收集每組實驗的資料 ──
    all_group_metrics: list[dict[str, dict[str, float]]] = []
    all_group_fold_series: list[dict] = []
    group_fraction_labels: list[str] = []

    for sf_str, label in zip(args.summary_files, labels):
        summary_path = Path(sf_str)
        if not summary_path.exists():
            raise FileNotFoundError(f"找不到 summary 檔案: {summary_path}")

        exp_names = read_experiment_names_from_summary(summary_path)
        if not exp_names:
            raise RuntimeError(f"在 summary 中解析不到任何實驗目錄: {summary_path}")

        exp_dirs: list[Path] = []
        for n in exp_names:
            p = resolve_experiment_dir(n, roots)
            exp_dirs.append(p)

        print(f"\n[{label}] 讀取 {len(exp_dirs)} 個 seed 的 unselected_correct_summary...")
        group_data = collect_group_metrics(exp_dirs, args.metrics)
        if args.plot_style == "line-fold":
            if len(args.metrics) != 1:
                raise ValueError("line-fold 模式目前只支援單一 metric，請只傳一個 --metrics")
            group_series = collect_group_fold_series(exp_dirs, args.metrics[0])
            all_group_fold_series.append(group_series)
        elif len(args.metrics) == 1 and args.plot_style in {"line-group", "bar"}:
            num, den = collect_group_metric_fraction(exp_dirs, args.metrics[0])
            group_fraction_labels.append(f"{num}/{den}")

        # 印出每 seed 的數值
        seed_labels = sorted(group_data.keys())
        for metric in args.metrics:
            vals = [group_data[s][metric] * 100 for s in seed_labels]
            mean_v = np.mean(vals)
            std_v = np.std(vals)
            seed_str = ", ".join(f"{s}={v:.2f}%" for s, v in zip(seed_labels, vals))
            print(f"  {METRIC_DISPLAY_SHORT[metric]:>30s}: mean={mean_v:.2f}% ± {std_v:.2f}%  ({seed_str})")

        all_group_metrics.append(group_data)

    # ── 輸出 ──
    base = output_dir / args.output_name
    if args.plot_style == "line-fold":
        metric = args.metrics[0]
        plot_fold_line_comparison(
            fold_series_list=all_group_fold_series,
            labels=labels,
            metric=metric,
            output_path=base.with_suffix(".png"),
            title=args.title,
        )
        write_fold_fraction_report(
            fold_series_list=all_group_fold_series,
            labels=labels,
            metric=metric,
            output_path=base.with_name(base.name + "_fold_details.txt"),
        )
    elif args.plot_style == "line-group":
        if len(args.metrics) != 1:
            raise ValueError("line-group 模式目前只支援單一 metric，請只傳一個 --metrics")
        metric = args.metrics[0]
        plot_group_line_comparison(
            all_group_metrics=all_group_metrics,
            labels=labels,
            metric=metric,
            output_path=base.with_suffix(".png"),
            title=args.title,
            no_std_bar=args.no_std_bar,
            fraction_labels=group_fraction_labels,
        )
        write_text_report(
            all_group_metrics=all_group_metrics,
            labels=labels,
            metrics=args.metrics,
            output_path=base.with_suffix(".txt"),
        )
    else:
        plot_comparison(
            all_group_metrics=all_group_metrics,
            labels=labels,
            metrics=args.metrics,
            output_path=base.with_suffix(".png"),
            title=args.title,
            no_std_bar=args.no_std_bar,
            fraction_labels=group_fraction_labels if len(args.metrics) == 1 else None,
        )
        write_text_report(
            all_group_metrics=all_group_metrics,
            labels=labels,
            metrics=args.metrics,
            output_path=base.with_suffix(".txt"),
        )


if __name__ == "__main__":
    main()
