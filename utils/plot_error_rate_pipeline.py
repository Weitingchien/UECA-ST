#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
偽標籤錯誤率分析管線工具

整合三步驟：
  Step 1: analyze_pseudo_label_accuracy.py  → 對每個 seed 目錄產生 summary
  Step 2: aggregate_pseudo_label_error_rate.py → 跨 seed 彙整 Mean ± Std
  Step 3: 繪製各組錯誤率折線圖

參數接口與 plot_val_pair_f1_per_round.py 一致：
  --summary-files  : 一或多個 summary txt 路徑
  --experiments-root: 實驗根目錄

用法範例：
  python utils/plot_error_rate_pipeline.py \
    --summary-files \
        $(ls results_ep_split10_t1te1v1_u7_disjoint/*_CE_summary.txt | grep -v consistency | grep -v '_EC_summary' | grep -v '_st0_') \
        results_ep_split10_t1te1v1_u7_disjoint/UECA_Prompt_f1-10_i70_lr1e-5_bs8_wd0.01_th0.9_maskemotion_gamma0.5_st10_ste3_retain_pseudo_consistency_CE_summary.txt \
    --experiments-root ep_split10_t1te1v1_u7_disjoint \
    --output pseudo_label_error_rate_comparison.png
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# ── 字體設定：與 plot_val_pair_f1_per_round.py 一致 ─────────
plt.rcParams['font.family'] = ['Calisto MT', 'DFKai-SB', 'DejaVu Serif', 'Noto Serif CJK JP']
plt.rcParams['axes.unicode_minus'] = False

FONT_CHINESE = 'DFKai-SB'
FONT_ENGLISH = 'Calisto MT'


# ══════════════════════════════════════════════════════════════
#  從 plot_val_pair_f1_per_round.py 複用的解析工具
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


def get_short_label(name: str) -> str:
    """
    從資料夾名稱或 summary 檔名擷取可辨識的簡短標籤。
    邏輯與 plot_val_pair_f1_per_round.py 的 get_short_label() 完全一致。
    """
    s = re.sub(r'_summary\.txt$', '', name, flags=re.IGNORECASE)
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


# ══════════════════════════════════════════════════════════════
#  從 analyze_pseudo_label_accuracy.py 複用的分析函式
# ══════════════════════════════════════════════════════════════

def _find_pseudo_results_dir(pred_dir: str) -> str | None:
    """尋找 pseudo_results_* 資料夾"""
    import glob
    dirs = glob.glob(os.path.join(pred_dir, "pseudo_results_*"))
    return dirs[0] if dirs else None


def _count_perfect_docs(filepath: str) -> tuple[int, int, list[int]]:
    """計算檔案中 3 個 [MASK] 全對的文檔數量"""
    if not os.path.exists(filepath):
        return 0, 0, []
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    doc_matches = re.findall(r'^doc_id: \d+', content, re.MULTILINE)
    total_docs = len(doc_matches)
    lines = content.split('\n')
    perfect_count = 0
    perfect_doc_ids = []
    current_doc_id = None
    for line in lines:
        doc_match = re.match(r'^doc_id: (\d+)', line)
        if doc_match:
            current_doc_id = int(doc_match.group(1))
        if '正確/總數:' in line and '錯誤率 0.0000' in line and '排除' not in line:
            perfect_count += 1
            if current_doc_id is not None:
                perfect_doc_ids.append(current_doc_id)
    return total_docs, perfect_count, perfect_doc_ids


def analyze_folder(pred_dir: str) -> dict | None:
    """分析單一資料夾，統計所有 fold 和 round 的錯誤數量"""
    pseudo_dir = _find_pseudo_results_dir(pred_dir)
    if not pseudo_dir:
        print(f"  警告：找不到 pseudo_results_* 資料夾於 {pred_dir}")
        return None
    folder_name = os.path.basename(pred_dir.rstrip('/\\'))
    fold_round_results = {}
    total_docs = 0
    total_perfect = 0
    for fold in range(1, 11):
        fold_round_results[fold] = {}
        for round_idx in range(1, 11):
            filename = f"pseudo_label_evaluation_fold{fold}_round{round_idx}.txt"
            filepath = os.path.join(pseudo_dir, filename)
            docs, perfect, perfect_ids = _count_perfect_docs(filepath)
            fold_round_results[fold][round_idx] = {
                'total': docs, 'perfect': perfect,
                'error': docs - perfect, 'perfect_doc_ids': perfect_ids,
            }
            total_docs += docs
            total_perfect += perfect
    total_error = total_docs - total_perfect
    return {
        'folder_name': folder_name,
        'fold_round_results': fold_round_results,
        'total_docs': total_docs, 'perfect_docs': total_perfect,
        'error_docs': total_error,
        'error_rate': total_error / total_docs if total_docs > 0 else 0,
    }


def save_analysis_to_file(pred_dir: str, result: dict) -> str:
    """將 analyze 結果儲存到 pseudo_label_accuracy_summary.txt"""
    output_path = os.path.join(pred_dir, "pseudo_label_accuracy_summary.txt")
    lines = []
    lines.append("=" * 100)
    lines.append(f"偽標籤錯誤率分析報告")
    lines.append(f"資料夾: {result['folder_name']}")
    lines.append("=" * 100)
    # 詳細表格
    header = f"{'Fold':<6}"
    for r in range(1, 11):
        header += f"{'R'+str(r):<10}"
    header += f"{'折總計':<12}"
    lines.append(header)
    lines.append("-" * 100)
    for fold in range(1, 11):
        line = f"{fold:<6}"
        fold_total = 0
        fold_error = 0
        for round_idx in range(1, 11):
            r = result['fold_round_results'][fold][round_idx]
            line += f"{r['error']}/{r['total']:<7}"
            fold_total += r['total']
            fold_error += r['error']
        line += f"{fold_error}/{fold_total}"
        lines.append(line)
    lines.append("-" * 100)
    round_line = f"{'輪總計':<6}"
    for round_idx in range(1, 11):
        round_total = sum(result['fold_round_results'][f][round_idx]['total'] for f in range(1, 11))
        round_error = sum(result['fold_round_results'][f][round_idx]['error'] for f in range(1, 11))
        round_line += f"{round_error}/{round_total:<7}"
    lines.append(round_line)
    # 總結
    lines.append("")
    lines.append("=" * 100)
    lines.append("總結")
    lines.append("=" * 100)
    lines.append(f"總選中文檔數: {result['total_docs']}")
    lines.append(f"全對文檔數: {result['perfect_docs']}")
    lines.append(f"錯誤文檔數: {result['error_docs']}")
    lines.append(f"錯誤率: {result['error_rate']*100:.2f}%")
    lines.append(f"正確率: {(1 - result['error_rate'])*100:.2f}%")
    lines.append("")
    # 全對文檔 ID
    lines.append("=" * 100)
    lines.append("全對文檔 ID 詳細資訊 (3 個 [MASK] 全部預測正確)")
    lines.append("=" * 100)
    for fold in range(1, 11):
        lines.append(f"\nFold {fold}:")
        lines.append("-" * 80)
        for round_idx in range(1, 11):
            r = result['fold_round_results'][fold][round_idx]
            perfect_ids = r.get('perfect_doc_ids', [])
            if perfect_ids:
                ids_str = ', '.join(map(str, sorted(perfect_ids)))
                lines.append(f"  R{round_idx}: [{len(perfect_ids)}筆] {ids_str}")
            else:
                lines.append(f"  R{round_idx}: [無]")
    lines.append("")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    return output_path


# ══════════════════════════════════════════════════════════════
#  從 aggregate_pseudo_label_error_rate.py 複用的彙整函式
# ══════════════════════════════════════════════════════════════

def _parse_error_rates(filepath: str) -> dict | None:
    """解析 pseudo_label_accuracy_summary.txt，提取每輪的錯誤率"""
    if not os.path.isfile(filepath):
        return None
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    match = re.search(r'輪總計\s+([\d/\s]+)', content)
    if not match:
        return None
    pairs = re.findall(r'(\d+)/(\d+)', match.group(1))
    if len(pairs) < 10:
        return None
    result = {}
    for i, (error, total) in enumerate(pairs[:10]):
        error_i, total_i = int(error), int(total)
        result[f'R{i+1}'] = (error_i / total_i * 100) if total_i > 0 else 0
    total_match = re.search(r'錯誤率:\s*([\d.]+)%', content)
    if total_match:
        result['total'] = float(total_match.group(1))
    return result


def aggregate_group(exp_dirs: list[str], label: str, output_dir: str) -> dict | None:
    """
    彙整一組（多 seed）實驗的錯誤率，回傳每輪 Mean（用於繪圖）。
    同時將彙整報告寫入 output_dir。
    """
    all_results = []
    seed_labels = []

    for d in exp_dirs:
        summary_file = os.path.join(d, 'pseudo_label_accuracy_summary.txt')
        result = _parse_error_rates(summary_file)
        if result:
            all_results.append(result)
            seed_match = re.search(r'seed(\d+)', d)
            seed_labels.append(f"seed{seed_match.group(1)}" if seed_match else "?")

    if not all_results:
        return None

    rounds = [f'R{i}' for i in range(1, 11)]
    aggregated = {}
    for r in rounds:
        values = [res[r] for res in all_results if r in res]
        if values:
            aggregated[r] = {
                'mean': np.mean(values),
                'std': np.std(values, ddof=1) if len(values) > 1 else 0.0,
                'values': values,
            }

    total_values = [res['total'] for res in all_results if 'total' in res]
    if total_values:
        aggregated['total'] = {
            'mean': np.mean(total_values),
            'std': np.std(total_values, ddof=1) if len(total_values) > 1 else 0.0,
            'values': total_values,
        }

    # 寫入彙整報告
    _save_aggregate_report(aggregated, seed_labels, exp_dirs, label, output_dir)
    return aggregated


def _save_aggregate_report(
    aggregated: dict, seed_labels: list[str], exp_dirs: list[str],
    label: str, output_dir: str,
) -> None:
    """儲存彙整報告文字檔"""
    os.makedirs(output_dir, exist_ok=True)
    safe_label = re.sub(r'[^\w\-.]', '_', label)
    output_path = os.path.join(output_dir, f"{safe_label}_error_rate.txt")

    lines = []
    lines.append("=" * 70)
    lines.append(f"多 Seed 偽標籤錯誤率彙整 — {label}")
    lines.append("=" * 70)
    lines.append(f"\n彙整的實驗目錄 (共 {len(exp_dirs)} 個):")
    for d in exp_dirs:
        lines.append(f"  - {os.path.basename(d)}")

    lines.append("\n" + "-" * 70)
    lines.append("各 Seed 每輪錯誤率 (%):")
    lines.append("-" * 70)

    header = f"{'輪次':<8}" + "".join(f"{s:>12}" for s in seed_labels) + f"{'Mean':>12}{'± Std':>10}"
    lines.append(header)
    lines.append("-" * (8 + 12 * len(seed_labels) + 22))

    rounds = [f'R{i}' for i in range(1, 11)]
    for r in rounds:
        if r in aggregated:
            a = aggregated[r]
            row = f"{r:<8}"
            row += "".join(f"{v:>11.2f}%" for v in a['values'])
            row += f"{a['mean']:>11.2f}%{a['std']:>9.2f}%"
            lines.append(row)

    lines.append("-" * (8 + 12 * len(seed_labels) + 22))
    if 'total' in aggregated:
        a = aggregated['total']
        row = f"{'總計':<8}"
        row += "".join(f"{v:>11.2f}%" for v in a['values'])
        row += f"{a['mean']:>11.2f}%{a['std']:>9.2f}%"
        lines.append(row)

    lines.append("\n" + "=" * 70)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"  ✓ 彙整報告已儲存: {output_path}")


# ══════════════════════════════════════════════════════════════
#  繪圖 — 與 plot_val_pair_f1_per_round.py 完全一致的視覺風格
# ══════════════════════════════════════════════════════════════

# 顏色 (與 plot_val_pair_f1_per_round.py 的 DISTINCT_COLORS 一致)
DISTINCT_COLORS = [
    "#d62728",  # 紅
    "#1f77b4",  # 藍
    "#2ca02c",  # 綠
    "#ff7f0e",  # 橙
    "#9467bd",  # 紫
    "#e377c2",  # 粉紅
    "#17becf",  # 青
    "#8c564b",  # 棕
    "#bcbd22",  # 黃綠
    "#000000",  # 黑
    "#393b79",  # 深藍紫
    "#e7969c",  # 淡珊瑚
]

# Marker 分組 (與 plot_val_pair_f1_per_round.py 一致)
GROUP_MARKERS = {
    "mask":         "o",   # ● 圓形
    "nest":         "s",   # ■ 方形
    "hybrid":       "^",   # ▲ 三角形
    "consistency":  "D",   # ◆ 菱形
    "_other":       "X",   # ✕
}


def classify_label(label: str) -> str:
    """將 label 歸類為 consistency / hybrid / nest / mask / _other"""
    if "consistency" in label:
        return "consistency"
    if "hybrid" in label:
        return "hybrid"
    if label.startswith("nest_k"):
        return "nest"
    if "mask" in label and "hybrid" not in label:
        return "mask"
    return "_other"


def _make_color_set(main_hex: str) -> dict:
    """從 main 色自動衍生 light / fill 色"""
    r, g, b, _ = mcolors.to_rgba(main_hex)
    lr, lg, lb = 0.50 * r + 0.50, 0.50 * g + 0.50, 0.50 * b + 0.50
    fr, fg, fb = 0.25 * r + 0.75, 0.25 * g + 0.75, 0.25 * b + 0.75
    return {
        "main": main_hex,
        "light": mcolors.to_hex((lr, lg, lb)),
        "fill": mcolors.to_hex((fr, fg, fb)),
    }


def plot_error_rates(
    all_aggregated: list[dict],
    labels: list[str],
    output_path: str,
    title: str = "各輪偽標籤錯誤率比較 (CE)",
) -> None:
    """繪製錯誤率折線圖，視覺風格與 plot_val_pair_f1_per_round.py 一致"""
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor('#DDDDDD')
    ax.set_facecolor('#DDDDDD')

    rounds = list(range(1, 11))

    # 分配顏色 + marker
    assigned: list[tuple[dict, str]] = []
    for g_idx, label in enumerate(labels):
        grp = classify_label(label)
        marker = GROUP_MARKERS.get(grp, "X")
        color_hex = DISTINCT_COLORS[g_idx % len(DISTINCT_COLORS)]
        assigned.append((_make_color_set(color_hex), marker))

    for g_idx, (aggregated, label) in enumerate(zip(all_aggregated, labels)):
        colors, marker = assigned[g_idx]
        means = [aggregated[f'R{r}']['mean'] for r in rounds if f'R{r}' in aggregated]

        if not means:
            continue

        ax.plot(rounds, means,
                label=label,
                color=colors["main"],
                marker=marker,
                linewidth=2.5,
                markersize=10)

        # 標註最高點和最低點
        max_idx = means.index(max(means))
        min_idx = means.index(min(means))

        ax.annotate(f'{means[max_idx]:.1f}%',
                    xy=(rounds[max_idx], means[max_idx]),
                    xytext=(0, 8),
                    textcoords='offset points',
                    ha='center', va='bottom',
                    fontsize=11, fontname=FONT_ENGLISH,
                    color=colors["main"], fontweight='bold')

        ax.annotate(f'{means[min_idx]:.1f}%',
                    xy=(rounds[min_idx], means[min_idx]),
                    xytext=(0, -8),
                    textcoords='offset points',
                    ha='center', va='top',
                    fontsize=11, fontname=FONT_ENGLISH,
                    color=colors["main"], fontweight='bold')

        # 印出數值
        print(f"\n[{label}] mean error rate per round:")
        for r_idx, r in enumerate(rounds):
            key = f'R{r}'
            if key in aggregated:
                a = aggregated[key]
                vals_str = ", ".join(f"{v:.2f}%" for v in a['values'])
                print(f"  R{r}: mean={a['mean']:.2f}%  ({vals_str})")

    # 標題、軸標籤 (字型大小與 plot_val_pair_f1_per_round.py 一致)
    ax.set_title(title, fontsize=24, fontweight='bold', fontname=FONT_CHINESE)
    ax.set_xlabel('自訓練輪次 (Round)', fontsize=24, fontname=FONT_CHINESE)
    ax.set_ylabel('平均錯誤率 (%)', fontsize=24, fontname=FONT_CHINESE)

    # X 軸刻度
    ax.set_xticks(rounds)
    ax.set_xticklabels([f'R{r}' for r in rounds], fontsize=20, fontname=FONT_ENGLISH)

    # Y 軸刻度
    for lbl in ax.get_yticklabels():
        lbl.set_fontsize(20)
        lbl.set_fontname(FONT_ENGLISH)

    # 圖例
    legend = ax.legend(fontsize=14, loc='best', ncol=2)
    legend.get_frame().set_facecolor('#DDDDDD')
    for text in legend.get_texts():
        has_cjk = any('\u4e00' <= ch <= '\u9fff' for ch in text.get_text())
        text.set_fontname(FONT_CHINESE if has_cjk else FONT_ENGLISH)

    ax.grid(True, linestyle='--', alpha=0.7)

    fig.subplots_adjust(left=0.12, bottom=0.10, right=0.97, top=0.93)
    fig.savefig(output_path, dpi=150, bbox_inches='tight', pad_inches=0.3)
    print(f"\n圖片已儲存: {output_path}")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════
#  main
# ══════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="偽標籤錯誤率分析管線 (analyze → aggregate → plot)",
    )
    p.add_argument(
        "--summary-files", nargs="+", required=True,
        help="一或多個 summary txt 路徑",
    )
    p.add_argument(
        "--labels", nargs="+", default=None,
        help="每個 summary 對應的顯示名稱 (數量需與 --summary-files 一致)"
             "若不提供則自動從 summary 檔名擷取特徵片段",
    )
    p.add_argument(
        "--experiments-root", required=True,
        help="實驗根目錄，例如 ep_split10_t1te1v1_u7_disjoint",
    )
    p.add_argument(
        "--output", default="pseudo_label_error_rate_comparison.png",
        help="輸出圖片路徑 (預設: pseudo_label_error_rate_comparison.png)",
    )
    p.add_argument(
        "--aggregate-dir", default="results_error_rate",
        help="彙整報告輸出目錄 (預設: results_error_rate)",
    )
    p.add_argument(
        "--title", default="各輪偽標籤錯誤率比較 (CE)",
        help="圖表標題",
    )
    p.add_argument(
        "--skip-analyze", action="store_true",
        help="跳過 Step 1 (若已有 pseudo_label_accuracy_summary.txt)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── 自動產生 labels ──────────────────────────────────
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

    root = Path(args.experiments_root)
    if not root.is_dir():
        raise FileNotFoundError(f"找不到 experiments-root: {root}")

    all_aggregated: list[dict] = []
    valid_labels: list[str] = []

    for sf_str, label in zip(args.summary_files, labels):
        summary_path = Path(sf_str)
        if not summary_path.exists():
            print(f"⚠ 找不到 summary 檔案: {summary_path}，跳過")
            continue

        exp_names = read_experiment_names_from_summary(summary_path)
        if not exp_names:
            print(f"⚠ 在 summary 中解析不到實驗目錄: {summary_path}，跳過")
            continue

        exp_dirs: list[str] = []
        for n in exp_names:
            p = root / n
            if not p.is_dir():
                print(f"  ⚠ 找不到實驗目錄: {p}，跳過")
                continue
            exp_dirs.append(str(p))

        if not exp_dirs:
            continue

        print(f"\n{'='*70}")
        print(f"處理: {label} ({len(exp_dirs)} 個 seed)")
        print(f"{'='*70}")

        # ── Step 1: Analyze ──────────────────────────────
        if not args.skip_analyze:
            print(f"\n[Step 1] 分析偽標籤錯誤率...")
            for d in exp_dirs:
                summary_exists = os.path.isfile(
                    os.path.join(d, 'pseudo_label_accuracy_summary.txt')
                )
                if summary_exists:
                    print(f"  ✓ 已存在 summary: {os.path.basename(d)}")
                    continue
                result = analyze_folder(d)
                if result:
                    out = save_analysis_to_file(d, result)
                    print(f"  ✓ 已產生: {out}")
                else:
                    print(f"  ✗ 分析失敗: {os.path.basename(d)}")

        # ── Step 2: Aggregate ────────────────────────────
        print(f"\n[Step 2] 跨 seed 彙整...")
        aggregated = aggregate_group(exp_dirs, label, args.aggregate_dir)
        if aggregated:
            all_aggregated.append(aggregated)
            valid_labels.append(label)
        else:
            print(f"  ✗ 彙整失敗: {label}")

    # ── Step 3: Plot ─────────────────────────────────────
    if not all_aggregated:
        print("\n✗ 沒有有效的數據可繪製")
        return

    print(f"\n{'='*70}")
    print(f"[Step 3] 繪製錯誤率折線圖 ({len(valid_labels)} 組)...")
    print(f"{'='*70}")
    plot_error_rates(all_aggregated, valid_labels, args.output, title=args.title)


if __name__ == "__main__":
    main()
