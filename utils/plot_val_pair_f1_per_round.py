#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
繪製「不同 seed 在各自訓練輪次的驗證集 Emotion / Cause / Pair 指標」曲線圖

資料來源:
    - init_supervised_metrics/fold{fold}_init_supervised_metrics.txt  → Round 0
    - self_training_results_*/self_training_val_results_fold{fold}_round{r}.txt → Round 1~N

繪圖邏輯:
    1) 每個 seed、每個 round: 讀取 10 折驗證 Emotion / Cause / Pair 指標 → 計算 10 折平均
    2) 每個 experiment group (如 baseline / consistency):
     - 各 seed 畫一條淡色線
     - 3 seed 的平均畫一條粗線 ± std 帶

用法範例 (比較 consistency vs baseline):
  python utils/plot_val_pair_f1_per_round.py \
    --summary-files \
      results_ep_split10_t1te1v1_u7_disjoint/UECA_Prompt_f1-10_i70_lr1e-5_bs8_wd0.01_th0.9_maskemotion_gamma0.5_st10_ste3_retain_pseudo_consistency_CE_summary.txt \
      results_ep_split10_t1te1v1_u7_disjoint/UECA_Prompt_f1-10_i70_lr1e-5_bs8_wd0.01_th0.9_maskemotion_gamma0.5_st10_ste3_retain_pseudo_CE_summary.txt \
    --labels "Consistency" "Baseline" \
    --experiments-root ep_split10_t1te1v1_u7_disjoint \
    --fold-start 1 --fold-end 10 \
    --rounds 10

也支援單一 summary:
  python utils/plot_val_pair_f1_per_round.py \
    --summary-files results_ep.../xxx_summary.txt \
    --labels "Consistency" \
    --experiments-root ep_split10_t1te1v1_u7_disjoint

若想畫 Emotion F1，可再加上:
    --target emotion --metric f1

若想畫 Cause F1，可再加上:
    --target cause --metric f1
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

# ── 字體設定：與 plot_pseudo_label_error_rate.py 一致 ─────────
# 注意：需先在 WSL 中連結 Windows 字體：sudo ln -s /mnt/c/Windows/Fonts /usr/share/fonts/windows
plt.rcParams['font.family'] = ['Calisto MT', 'DFKai-SB', 'DejaVu Serif', 'Noto Serif CJK JP']
plt.rcParams['axes.unicode_minus'] = False  # 解決負號顯示問題

# 定義字體物件，供需要精確控制時使用
FONT_CHINESE = 'DFKai-SB'   # 標楷體
FONT_ENGLISH = 'Calisto MT'  # Calisto MT 英文字體

TARGET_INDEX = {
    "emotion": 0,
    "cause": 1,
    "pair": 2,
}

TARGET_LABELS = {
    "emotion": "Emotion",
    "cause": "Cause",
    "pair": "Pair",
}

METRIC_LABELS = {
    "f1": "F1",
    "precision": "Precision",
    "recall": "Recall",
}

INIT_METRIC_LABELS = {
    "f1": "F1",
    "precision": "P",
    "recall": "R",
}


# ══════════════════════════════════════════════════════════════
#  解析工具
# ══════════════════════════════════════════════════════════════

def read_experiment_names_from_summary(summary_file: Path) -> list[str]:
    """
    從 summary 文字檔中解析「實驗資料夾名稱」清單

    參數:
      summary_file: 單一 summary 檔案路徑

    回傳:
      names: 例如 ["prompt_ECPE_few_shot_ST_...", ...]

    動作說明:
      1) 讀全文
      2) 逐行用正則抓出以 '- prompt_ECPE_few_shot_ST_' 開頭的項目
      3) 收集成 list 回傳
    """
    # 讀取 summary 文字內容 (忽略無法解碼字元，避免中斷)
    text = summary_file.read_text(encoding="utf-8", errors="ignore")
    # 存放解析出的實驗資料夾名稱
    names: list[str] = []
    # 逐行檢查，因為 summary 通常是條列格式
    for line in text.splitlines():
        # 範例可匹配: "- prompt_ECPE_few_shot_ST_2026_..."
        m = re.match(r"\s*-\s*(prompt_ECPE_few_shot_ST_[^\s]+)", line)
        if m:
            # m.group(1) 是正則第一個括號抓到的字串
            names.append(m.group(1).strip())
    return names


def extract_seed(dirname: str) -> str:
    """
    從實驗資料夾名稱中擷取 seed 數字

    例如:
      '...seed42_...' -> '42'

    若找不到 seed{數字}，就回傳原字串，避免流程中斷
    """
    m = re.search(r"seed(\d+)", dirname)
    return m.group(1) if m else dirname


def find_self_training_results_dir(exp_dir: Path) -> Path:
    """找到 self_training_results_* 子目錄（排序後取最後一個，視為最新)"""
    # 收集所有符合名稱模式且確實為資料夾的路徑
    candidates = sorted([p for p in exp_dir.glob("self_training_results_*") if p.is_dir()])
    if not candidates:
        raise FileNotFoundError(f"{exp_dir} 底下找不到 self_training_results_* 資料夾")
    # 命名通常包含時間戳，排序後最後一個常是最新結果
    return candidates[-1]


def get_short_label(name: str) -> str:
    """
    從資料夾名稱或 summary 檔名擷取可辨識的簡短標籤

    規則:
      1) 去掉 _summary.txt 後綴
      2) 從 'th' 開始擷取；若無 th 則從 'nest_k5_' 開始擷取
      3) 移除共用參數片段:
         _gamma[\d.]+, _st\d+, _ste\d+, _seed\d+, _retain_pseudo,
         _nbeta[\d.]+, _nm[\d.]+
      4) 移除 knncause 前面的冗餘 cause(_clause)?_
         (cause_clause_knncause → knncause, cause_knncause → knncause)

    範例:
      ..._th0.9_maskemotion_gamma0.5_st10_ste3_retain_pseudo_consistency_CE_summary.txt
      → th0.9_maskemotion_consistency_CE

      ..._th0.9_hybrid_switch5_nest_k5_cause_knncause_clause_mask_nestmul3_nbeta0.1_nm0.6_..._CE_summary.txt
      → th0.9_hybrid_switch5_nest_k5_knncause_clause_mask_nestmul3_CE

      ..._nest_k5_cause_clause_knncause_clause_mask_nestmul3_nbeta0.1_nm0.6_..._CE_summary.txt
      → nest_k5_knncause_clause_mask_nestmul3_CE
    """
    # 去掉 _summary.txt 後綴
    s = re.sub(r'_summary\.txt$', '', name, flags=re.IGNORECASE)
    # 特殊情況：_st0_ 表示僅 few-shot 無自訓練
    if '_st0_' in s or s.endswith('_st0'):
        suffix_match = re.search(r'_(CE|EC)$', s)
        suffix = f"_{suffix_match.group(1)}" if suffix_match else ""
        return f"st0{suffix}"
    # 從 th 開始擷取；若無 th 則從 nest_k5_ 開始
    m = re.search(r'(th[\d.]+_.*)', s)
    if m:
        s = m.group(1)
    else:
        m2 = re.search(r'(nest_k\d+_.*)', s)
        if m2:
            s = m2.group(1)
    # 移除共用參數片段
    s = re.sub(r'_seed\d+', '', s)
    s = re.sub(r'_gamma[\d.]+', '', s)
    s = re.sub(r'_st\d+', '', s)
    s = re.sub(r'_ste\d+', '', s)
    s = re.sub(r'_retain_pseudo', '', s)
    s = re.sub(r'_nbeta[\d.]+', '', s)
    s = re.sub(r'_nm[\d.]+', '', s)
    # 移除 knncause 前面的冗餘 cause(_clause)?_
    s = re.sub(r'cause(_clause)?_(?=knncause)', '', s)
    # 與 pair_m1 圖一致：將 consistency_somc 簡化為 somc
    s = re.sub(r'_consistency_somc_', '_somc_', s)
    # 清理可能產生的連續底線
    s = re.sub(r'__+', '_', s)
    s = s.strip('_')
    return s


# ── 讀取初始監督指標 (Round 0) ─────────────────────────────────

def extract_metric_values(metric_blob: str) -> list[float]:
    """從 [tensor(0.1234), 0.5678] 這類片段中擷取所有數值。"""
    number_strings = re.findall(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)", metric_blob)
    return [float(v) for v in number_strings]


def parse_init_metric(path: Path, target: str = "pair", metric: str = "f1") -> float | None:
    """
        從 init_supervised_metrics.txt 讀 Emotion / Cause / Pair 指標
        target: 'emotion' / 'cause' / 'pair'
    metric: 'f1' / 'precision' / 'recall'
    格式:
            Emotion F1: [tensor(0.7958)]
            Cause F1: [tensor(0.5810)]
      Pair F1: [tensor(0.5160)]
      Pair P: [0.5271739130148275]
      Pair R: [tensor(0.5052)]
    """
    # 檔案不存在就回傳 None，交由上層決定如何處理（例如跳過該 fold）
    if not path.exists():
        return None

    # 讀取 Round 0 指標檔內容
    text = path.read_text(encoding="utf-8", errors="ignore")

    target_label = TARGET_LABELS.get(target)
    metric_label = INIT_METRIC_LABELS.get(metric)
    if target_label is None or metric_label is None:
        return None

    m = re.search(rf"{target_label}\s+{metric_label}:\s*\[(.*?)\]", text)
    if not m:
        return None

    values = extract_metric_values(m.group(1))
    return values[-1] if values else None


# ── 讀取自訓練 round N 驗證指標 ────────────────────────────────

def parse_val_metric(path: Path, target: str = "pair", metric: str = "f1") -> float | None:
    """
        從 self_training_val_results_fold{f}_round{r}.txt 讀 Emotion / Cause / Pair 指標
        target: 'emotion' / 'cause' / 'pair'
    metric: 'f1' / 'precision' / 'recall'
    格式:
      Precision (Emotion, Cause, Pair): 0.7865, 0.6053, 0.5272
      Recall (Emotion, Cause, Pair): 0.7407, 0.4792, 0.5052
      F1 Score (Emotion, Cause, Pair): 0.7629, 0.5349, 0.5160
        取最後一行對應指標的 emotion / cause / pair 值
    """
    # 檔案不存在表示該 fold/round 無結果
    if not path.exists():
        return None

    # 讀取自訓練 round 的驗證結果檔
    text = path.read_text(encoding="utf-8", errors="ignore")

    # 根據 metric 選擇要抓哪一行（F1 / Precision / Recall）
    if metric == "f1":
        pattern = r"F1 Score \(Emotion, Cause, Pair\):\s*([\d.]+),\s*([\d.]+),\s*([\d.]+)"
    elif metric == "precision":
        pattern = r"Precision \(Emotion, Cause, Pair\):\s*([\d.]+),\s*([\d.]+),\s*([\d.]+)"
    elif metric == "recall":
        pattern = r"Recall \(Emotion, Cause, Pair\):\s*([\d.]+),\s*([\d.]+),\s*([\d.]+)"
    else:
        return None
    target_idx = TARGET_INDEX.get(target)
    if target_idx is None:
        return None

    # 可能同檔案中有多次輸出，先全部找出
    matches = re.findall(pattern, text)
    if not matches:
        return None
    # 最後一筆 = 該 round 最終結果
    return float(matches[-1][target_idx])


# ══════════════════════════════════════════════════════════════
#  核心: 收集一整組實驗 (多 seed) 的每 round 資料
# ══════════════════════════════════════════════════════════════

def collect_group_data(
    exp_dirs: list[Path],
    fold_start: int,
    fold_end: int,
    num_rounds: int,
    target: str = "pair",
    metric: str = "f1",
) -> dict:
    """
    回傳結構:
    {
      seed_label: {
        "rounds": [0, 1, ..., num_rounds],
        "fold_avg_f1": [val_r0, val_r1, ..., val_rN],   # 10 折平均
        "fold_f1s": [[fold1, fold2, ...], ...],         # 每 round 的各 fold 值
      },
      ...
    }
    target: 'emotion' / 'cause' / 'pair'
    metric: 'f1' / 'precision' / 'recall'
    """
    # num_folds 目前主要供語意說明，實際平均以讀到的 fold_vals 為準
    num_folds = fold_end - fold_start + 1
    result: dict = {}

    # 每個 exp_dir 通常可視為一個 seed 的實驗結果資料夾
    for exp_dir in exp_dirs:
        # 產生 seed 標籤，例: seed42
        seed_label = f"seed{extract_seed(exp_dir.name)}"
        # 自訓練結果資料夾 (Round 1~N)
        st_dir = find_self_training_results_dir(exp_dir)
        # 初始監督結果資料夾（Round 0）
        init_dir = exp_dir / "init_supervised_metrics"

        # rounds_list 會是 [0, 1, ..., N]
        rounds_list = list(range(0, num_rounds + 1))  # [0, 1, ..., N]
        # 每個 round 的「fold 平均值」
        fold_avg_f1: list[float] = []
        # 每個 round 原始 fold 值（方便後續除錯/分析）
        fold_f1s_all: list[list[float]] = []

        # 逐 round 計算
        for r in rounds_list:
            # 收集此 round 的所有 fold 指標
            fold_vals: list[float] = []
            # 逐 fold 讀取
            for fold in range(fold_start, fold_end + 1):
                if r == 0:
                    # Round 0 = 初始監督
                    val = parse_init_metric(
                        init_dir / f"fold{fold}_init_supervised_metrics.txt",
                        target=target,
                        metric=metric,
                    )
                else:
                    # Round 1~N = 自訓練
                    val = parse_val_metric(
                        st_dir / f"self_training_val_results_fold{fold}_round{r}.txt",
                        target=target,
                        metric=metric,
                    )
                # 只收集成功解析的數值
                if val is not None:
                    fold_vals.append(val) # 第 r round 的 fold_vals list 會有 fold_start..fold_end 的值(缺值則少一個元素)

            # 同一 seed、同一 round，對 fold 做平均
            if fold_vals:
                fold_avg_f1.append(np.mean(fold_vals))
            else:
                # 全部缺值時用 NaN，後續可用 nanmean / nanstd 忽略
                fold_avg_f1.append(np.nan)
            fold_f1s_all.append(fold_vals) # 每一 round對應一個fold_vals list

        result[seed_label] = {
            "rounds": rounds_list,
            "fold_avg_f1": fold_avg_f1,
            "fold_f1s": fold_f1s_all,
        }

    return result


# ══════════════════════════════════════════════════════════════
#  繪圖
# ══════════════════════════════════════════════════════════════

# ── 按實驗類型分組 marker；每條線用獨立鮮明顏色 ──────────
#  3 組: mask (th0.9_mask*), hybrid (th0.9_hybrid*), nest (nest_k5_*)
#  ▸ 同組共用 marker 形狀 → 一眼辨識組別
#  ▸ 每條線用不同的鮮明顏色 → 輕鬆區分個別實驗

import matplotlib.colors as mcolors

# 10+ 種鮮明且在 #DDDDDD 背景上高辨識度的顏色
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

# 與 pair_m1_multi_seed_comparison.png 對齊的固定顏色映射
# 重點: 顏色由標籤決定，而非由出現順序決定，避免隱藏 st0_CE 後整體洗牌
LABEL_FIXED_COLORS = {
    "st0_CE": "#d62728",
    "th0.9_maskemotion_consistency_CE": "#1f77b4",
    "th0.9_maskemotion_CE": "#2ca02c",
    "th0.9_maskemotion_somc_CE": "#4A148C",
    "th0.9_maskemotion_consistency_somc_CE": "#4A148C",
    "nest_k5_knncause_clause_mask_nestmul5_somc_CE": "#9467bd",
    "nest_k5_knncause_clause_mask_nestmul5_consistency_somc_CE": "#9467bd",
    "nest_k5_knncause_clause_mask_nestmul5_CE": "#e377c2",
}


def _make_color_set(main_hex: str) -> dict:
    """從 main 色自動衍生 light / fill 色."""
    r, g, b, _ = mcolors.to_rgba(main_hex)
    lr, lg, lb = 0.50 * r + 0.50, 0.50 * g + 0.50, 0.50 * b + 0.50
    fr, fg, fb = 0.25 * r + 0.75, 0.25 * g + 0.75, 0.25 * b + 0.75
    return {
        "main": main_hex,
        "light": mcolors.to_hex((lr, lg, lb)),
        "fill": mcolors.to_hex((fr, fg, fb)),
    }


def get_line_color_by_label(label: str, fallback_idx: int) -> str:
    """優先使用固定映射；未命中時退回循序配色。"""
    return LABEL_FIXED_COLORS.get(label, DISTINCT_COLORS[fallback_idx % len(DISTINCT_COLORS)])


# 每組的 marker 形狀 (4 組)
GROUP_MARKERS = {
    "mask":         "o",   # ● 圓形
    "nest":         "s",   # ■ 方形
    "hybrid":       "^",   # ▲ 三角形
    "consistency":  "D",   # ◆ 菱形
    "_other":       "X",   # ✕
}


def classify_label(label: str) -> str:
    """將 label 歸類為 consistency / hybrid / nest / mask / _other."""
    if "consistency" in label:
        return "consistency"
    if "hybrid" in label:
        return "hybrid"
    if label.startswith("nest_k"):
        return "nest"
    # th0.9_mask*, th0.9_maskall, th0.9_maskemotion 等
    if "mask" in label and "hybrid" not in label:
        return "mask"
    return "_other"


def safe_nanmean_axis0(matrix: np.ndarray) -> np.ndarray:
    """沿 axis=0 計算 nanmean；若某欄全是 NaN，該欄回傳 NaN 且不觸發警告。"""
    counts = np.sum(np.isfinite(matrix), axis=0)
    sums = np.nansum(matrix, axis=0)
    out = np.full(matrix.shape[1], np.nan, dtype=np.float64)
    np.divide(sums, counts, out=out, where=counts > 0)
    return out


def safe_nanstd_axis0(matrix: np.ndarray) -> np.ndarray:
    """沿 axis=0 計算 nanstd；若某欄全是 NaN，該欄回傳 NaN 且不觸發警告。"""
    out = np.full(matrix.shape[1], np.nan, dtype=np.float64)
    for col_idx in range(matrix.shape[1]):
        vals = matrix[:, col_idx]
        vals = vals[np.isfinite(vals)]
        if vals.size > 0:
            out[col_idx] = float(np.std(vals, ddof=0))
    return out


def plot_groups(
    all_groups: list[dict],
    labels: list[str],
    num_rounds: int,
    output_path: Path,
    title: str = "驗證集 Pair F1",
    show_individual_seeds: bool = True,
    show_std_band: bool = True,
) -> None:
    """
    all_groups: list of collect_group_data() 回傳值
    labels: 每組的顯示名稱
    """
    # 建立畫布與座標軸
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor('#DDDDDD')   # 圖片背景顏色
    ax.set_facecolor('#DDDDDD')          # 繪圖區域背景顏色
    # round 座標: 0..N
    rounds_arr = np.arange(0, num_rounds + 1)

    # ── 為每個 label 分配獨立顏色 + 同組 marker ──────────
    # assigned 會記錄每個群組對應的顏色組與 marker
    assigned: list[tuple[dict, str]] = []  # (color_set, marker)
    for g_idx, label in enumerate(labels):
        # 依 label 判斷屬於哪一類方法（mask/hybrid/nest/consistency）
        grp = classify_label(label)
        # 同類方法可共用 marker 形狀，利於辨識
        marker = GROUP_MARKERS.get(grp, "D")
        # 每條主線仍給獨立顏色
        color_hex = get_line_color_by_label(label, g_idx)
        assigned.append((_make_color_set(color_hex), marker))

    # 逐群組畫線
    for g_idx, (group_data, label) in enumerate(zip(all_groups, labels)):
        colors, marker = assigned[g_idx]

        # 收集每 seed 的 fold_avg_f1，組成 (n_seeds, n_rounds) 矩陣
        # seed_labels 例: ['seed20', 'seed42', 'seed60']
        seed_labels = sorted(group_data.keys())
        # matrix shape = (seed數, round數)
        matrix = np.array([group_data[s]["fold_avg_f1"] for s in seed_labels])  # (n_seeds, n_rounds+1)

        # 個別 seed 淡色線 (不加入圖例)
        if show_individual_seeds:
            for i, s in enumerate(seed_labels):
                # 個別 seed 的淡色虛線（輔助觀察離散程度）
                ax.plot(
                    rounds_arr,
                    matrix[i] * 100,
                    color=colors["light"],
                    linewidth=1,
                    alpha=0.6,
                    linestyle="--",
                    marker=".",
                    markersize=4,
                )

        # 跨 seed 平均 ± std
        # 主線：跨 seed 的平均（忽略 NaN）
        mean_f1 = safe_nanmean_axis0(matrix) * 100
        # 帶狀區：跨 seed 的標準差（忽略 NaN）
        std_f1 = safe_nanstd_axis0(matrix) * 100

        ax.plot(
            rounds_arr,
            mean_f1,
            color=colors["main"],
            linewidth=2.5,
            label=f"{label}",
            marker=marker,
            markersize=10,
        )
        if show_std_band:
            # 平均 ± 標準差帶狀區
            ax.fill_between(
                rounds_arr,
                mean_f1 - std_f1,
                mean_f1 + std_f1,
                color=colors["fill"],
                alpha=0.35,
            )

        # ── 標註最高點與最低點 ──────────────────────────────
        # 找主線的最高點與最低點（忽略 NaN）
        max_idx = int(np.nanargmax(mean_f1))
        min_idx = int(np.nanargmin(mean_f1))
        max_val = mean_f1[max_idx]
        min_val = mean_f1[min_idx]
        max_round_label = "Init" if rounds_arr[max_idx] == 0 else f"R{rounds_arr[max_idx]}"
        min_round_label = "Init" if rounds_arr[min_idx] == 0 else f"R{rounds_arr[min_idx]}"

        # 印出計算細節
        print(f"\n[{label}] mean_f1 per round (3 seeds 的 10 折平均):")
        for r_idx, r in enumerate(rounds_arr):
            r_label = "Init" if r == 0 else f"R{r}"
            seed_vals_str = ", ".join(
                f"{s}={matrix[i, r_idx]*100:.2f}%" for i, s in enumerate(seed_labels)
            )
            print(f"  {r_label:>5s}: mean={mean_f1[r_idx]:.2f}%  ({seed_vals_str})")
        # 第二高點 (排除最高點後的最大值)
        # 由大到小排序後取第二名
        sorted_indices = np.argsort(mean_f1)[::-1]  # 由大到小排序的 index
        sec_idx = int(sorted_indices[1]) if len(sorted_indices) > 1 else max_idx
        sec_val = mean_f1[sec_idx]
        sec_round_label = "Init" if rounds_arr[sec_idx] == 0 else f"R{rounds_arr[sec_idx]}"

        print(f"  → 最高點: {max_round_label} = {max_val:.2f}%")
        print(f"  → 第二高: {sec_round_label} = {sec_val:.2f}%")
        print(f"  → 最低點: {min_round_label} = {min_val:.2f}%")

        # 最高點標註 (往上偏移)
        ax.annotate(
            f'{max_val:.1f}%',
            xy=(rounds_arr[max_idx], max_val),
            xytext=(0, 10),
            textcoords='offset points',
            ha='center', va='bottom',
            fontsize=11, fontname=FONT_ENGLISH,
            color=colors["main"], fontweight='bold',
        )
        # 第二高點標註 (往上偏移)
        ax.annotate(
            f'{sec_val:.1f}%',
            xy=(rounds_arr[sec_idx], sec_val),
            xytext=(0, 10),
            textcoords='offset points',
            ha='center', va='bottom',
            fontsize=11, fontname=FONT_ENGLISH,
            color=colors["main"], fontweight='bold',
        )
        # 最低點標註 (往下偏移)
        ax.annotate(
            f'{min_val:.1f}%',
            xy=(rounds_arr[min_idx], min_val),
            xytext=(0, -10),
            textcoords='offset points',
            ha='center', va='top',
            fontsize=11, fontname=FONT_ENGLISH,
            color=colors["main"], fontweight='bold',
        )

    # ── 標題、軸標籤 ──────────────────────────────────────
    # 座標軸標題與字型設定
    ax.set_title(title, fontsize=24, fontweight='bold', fontname=FONT_CHINESE)
    ax.set_xlabel('自訓練輪次 (Round) ', fontsize=24, fontname=FONT_CHINESE)
    ax.set_ylabel(f'{title} (%)', fontsize=24, fontname=FONT_CHINESE)

    # X 軸刻度
    # X 軸顯示 Init, R1...RN
    ax.set_xticks(rounds_arr)
    tick_labels = ["Init" if r == 0 else f"R{r}" for r in rounds_arr]
    ax.set_xticklabels(tick_labels, fontsize=20, fontname=FONT_ENGLISH)

    # Y 軸刻度
    for lbl in ax.get_yticklabels():
        lbl.set_fontsize(20)
        lbl.set_fontname(FONT_ENGLISH)

    # 圖例：含中文的文字用中文字體，純英文用英文字體
    # 圖例自動放置，2 欄排版
    legend = ax.legend(fontsize=14, loc='best', ncol=2)
    legend.get_frame().set_facecolor('#DDDDDD')
    for text in legend.get_texts():
        # 檢查是否含中文字元 (Unicode CJK 範圍)
        has_cjk = any('\u4e00' <= ch <= '\u9fff' for ch in text.get_text())
        text.set_fontname(FONT_CHINESE if has_cjk else FONT_ENGLISH)

    ax.grid(True, linestyle='--', alpha=0.7)

    fig.subplots_adjust(left=0.12, bottom=0.10, right=0.97, top=0.93)
    # 儲存圖片
    fig.savefig(str(output_path), dpi=150, bbox_inches='tight', pad_inches=0.3)
    print(f"圖片已儲存: {output_path}")
    plt.close(fig)


def write_csv_report(
    all_groups: list[dict],
    labels: list[str],
    num_rounds: int,
    output_path: Path,
) -> None:
    """輸出 CSV 以便進一步用 Excel 等工具分析"""
    # 注意: 欄名歷史沿用 fold_avg_pair_f1，實際可代表 f1 / precision / recall
    header = ["group", "seed", "round", "fold_avg_pair_f1"]
    rows: list[str] = [",".join(header)]

    for g_idx, (group_data, label) in enumerate(zip(all_groups, labels)):
        for seed_label in sorted(group_data.keys()):
            data = group_data[seed_label]
            for r, f1 in zip(data["rounds"], data["fold_avg_f1"]):
                rows.append(f"{label},{seed_label},{r},{f1:.6f}")

    output_path.write_text("\n".join(rows), encoding="utf-8")
    print(f"CSV 已儲存: {output_path}")


def write_text_report(
    all_groups: list[dict],
    labels: list[str],
    num_rounds: int,
    output_path: Path,
    report_title: str,
) -> None:
    """輸出人類可讀的文字摘要"""
    # lines: 最終要寫入 .txt 報告的每一行文字
    lines: list[str] = []
    lines.append("=" * 70)
    lines.append(f"{report_title} 逐輪分析報告")
    lines.append("=" * 70)

    report_name = report_title.replace("驗證集 ", "", 1)

    rounds_arr = np.arange(0, num_rounds + 1)

    for g_idx, (group_data, label) in enumerate(zip(all_groups, labels)):
        lines.append(f"\n{'─' * 70}")
        lines.append(f"  {label}")
        lines.append(f"{'─' * 70}")

        # 按 seed 排序，讓報告輸出順序穩定
        seed_labels = sorted(group_data.keys())
        matrix = np.array([group_data[s]["fold_avg_f1"] for s in seed_labels])

        # 表頭
        seed_header = "  ".join(f"{s:>10s}" for s in seed_labels)
        lines.append(f"  {'Round':>6s}  {seed_header}  {'Mean':>8s}  {'Std':>7s}")
        lines.append(f"  {'─' * (8 + 12 * len(seed_labels) + 20)}")

        for r_idx, r in enumerate(rounds_arr):
            # 取某一個 round 的所有 seed 值
            vals = matrix[:, r_idx] * 100
            valid_vals = vals[np.isfinite(vals)]
            mean_v = float(np.mean(valid_vals)) if valid_vals.size > 0 else float("nan")
            std_v = float(np.std(valid_vals, ddof=0)) if valid_vals.size > 0 else float("nan")
            r_label = "Init" if r == 0 else f"R{r}"
            val_str = "  ".join(f"{v:10.2f}%" for v in vals)
            lines.append(f"  {r_label:>6s}  {val_str}  {mean_v:7.2f}%  {std_v:6.2f}%")

        # 標出最佳 round
        # 跨 seed 平均後找最佳 round
        mean_arr = safe_nanmean_axis0(matrix) * 100
        best_r = int(np.nanargmax(mean_arr))
        best_label = "Init" if best_r == 0 else f"R{best_r}"
        lines.append(f"\n  ★ 最佳平均 {report_name} 出現在 {best_label}: {mean_arr[best_r]:.2f}%")

    lines.append(f"\n{'=' * 70}")
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"文字報告已儲存: {output_path}")


# ══════════════════════════════════════════════════════════════
#  main
# ══════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="繪製驗證集 Emotion / Cause / Pair 指標隨自訓練輪次變化的曲線圖",
    )
    p.add_argument(
        "--summary-files",
        nargs="+",
        required=True,
        help="一或多個 summary txt 路徑",
    )
    p.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="每個 summary 對應的顯示名稱 (數量需與 --summary-files 一致)"
             "若不提供則自動從 summary 檔名擷取特徵片段",
    )
    p.add_argument(
        "--experiments-root",
        required=False,
        help="實驗根目錄，例如 ep_split10_t1te1v1_u7_disjoint",
    )
    p.add_argument(
        "--experiments-roots",
        nargs="+",
        default=None,
        help="可同時指定多個實驗根目錄，腳本會依序查找實驗資料夾",
    )
    p.add_argument("--fold-start", type=int, default=1)
    p.add_argument("--fold-end", type=int, default=10)
    p.add_argument("--rounds", type=int, default=10, help="自訓練總輪數")
    p.add_argument(
        "--output-dir",
        default="png",
        help="輸出目錄 (圖片 + CSV + 文字報告)",
    )
    p.add_argument(
        "--output-name",
        default="val_pair_f1_per_round",
        help="輸出檔名前綴 (不含副檔名)",
    )
    p.add_argument(
        "--metric",
        default="f1",
        choices=["f1", "precision", "recall", "all"],
        help="要繪製的指標: f1 / precision / recall / all (一次畫三張)",
    )
    p.add_argument(
        "--target",
        default="pair",
        choices=["emotion", "cause", "pair", "all"],
        help="要繪製的目標: emotion / cause / pair / all (一次畫三類目標)",
    )
    p.add_argument(
        "--title",
        default=None,
        help="圖表標題 (預設依 target + metric 自動產生)",
    )
    p.add_argument(
        "--no-individual-seeds",
        action="store_true",
        help="不繪製個別 seed 的淡色線",
    )
    p.add_argument(
        "--no-std-band",
        action="store_true",
        help="不繪製 ±std 帶狀面積",
    )
    # 回傳 argparse 解析後的參數物件
    return p.parse_args()


def get_metric_display(target: str, metric: str) -> dict[str, str]:
    """回傳指定 target + metric 對應的標題資訊。"""
    target_label = TARGET_LABELS[target]
    metric_label = METRIC_LABELS[metric]
    title = f"驗證集 {target_label} {metric_label}"
    return {
        "title": title,
        "report_name": f"{target_label} {metric_label}",
    }


def build_output_base_name(output_name: str, target: str, metric: str) -> str:
    """組合輸出檔名前綴，並保留舊版 pair 命名相容性。"""
    suffix_parts: list[str] = []
    if target != "pair":
        suffix_parts.append(target)
    if metric != "f1":
        suffix_parts.append(metric)

    suffix = "" if not suffix_parts else f"_{'_'.join(suffix_parts)}"
    return f"{output_name}{suffix}"


def run_for_metric(
    args,
    labels: list[str],
    root: Path,
    output_dir: Path,
    target: str,
    metric: str,
) -> None:
    """針對單一 target + metric 執行收集 + 繪圖 + 報告。"""
    # 依 target + metric 取對應顯示名稱
    disp = get_metric_display(target, metric)
    title = args.title if args.title else disp["title"]

    # all_groups: 每個 summary 對應一個 group_data
    all_groups: list[dict] = []

    root_candidates: list[Path] = []
    if args.experiments_roots:
        root_candidates.extend(Path(p) for p in args.experiments_roots)
    if args.experiments_root:
        root_from_arg = Path(args.experiments_root)
        if root_from_arg not in root_candidates:
            root_candidates.append(root_from_arg)
    if root not in root_candidates:
        root_candidates.insert(0, root)
    valid_roots = [p for p in root_candidates if p.is_dir()]
    for summary_path_str in args.summary_files:
        summary_path = Path(summary_path_str)
        if not summary_path.exists():
            raise FileNotFoundError(f"找不到 summary 檔案: {summary_path}")

        # 從 summary 解析實驗名稱 (通常是不同 seed)
        exp_names = read_experiment_names_from_summary(summary_path)
        if not exp_names:
            raise RuntimeError(f"在 summary 中解析不到任何實驗目錄: {summary_path}")

        # 將實驗名稱轉成完整路徑 (支援多根目錄依序查找)
        exp_dirs: list[Path] = []
        for n in exp_names:
            resolved = None
            for candidate_root in valid_roots:
                p = candidate_root / n
                if p.is_dir():
                    resolved = p
                    break
            if resolved is None:
                raise FileNotFoundError(f"找不到實驗目錄: {valid_roots[0] / n}")
            exp_dirs.append(resolved)

        # 收集這個群組(這個 summary) 在所有 round 的指標
        group_data = collect_group_data(
            exp_dirs=exp_dirs,
            fold_start=args.fold_start,
            fold_end=args.fold_end,
            num_rounds=args.rounds,
            target=target,
            metric=metric,
        )
        all_groups.append(group_data)

    # 輸出檔名: 若 metric != f1, 檔名加上 _precision / _recall
    # 命名規則:
    #   pair + f1        -> output_name
    #   pair + recall    -> output_name_recall
    #   emotion + f1     -> output_name_emotion
    #   cause + f1       -> output_name_cause
    #   emotion + recall -> output_name_emotion_recall
    base_name = build_output_base_name(args.output_name, target, metric)
    base = output_dir / base_name

    plot_groups(
        all_groups=all_groups,
        labels=labels,
        num_rounds=args.rounds,
        output_path=base.with_suffix(".png"),
        title=title,
        show_individual_seeds=not args.no_individual_seeds,
        show_std_band=not args.no_std_band,
    )
    write_csv_report(
        all_groups=all_groups,
        labels=labels,
        num_rounds=args.rounds,
        output_path=base.with_suffix(".csv"),
    )
    write_text_report(
        all_groups=all_groups,
        labels=labels,
        num_rounds=args.rounds,
        output_path=base.with_suffix(".txt"),
        report_title=title,
    )


def main() -> None:
    # 1) 解析命令列參數
    args = parse_args()

    # ── 自動產生 labels ──────────────────────────────────
    # 2) 決定圖例 labels: 使用者自訂或自動擷取
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

    # 3) 檢查 experiments-root(s) 是否存在
    root_candidates: list[Path] = []
    if args.experiments_roots:
        root_candidates.extend(Path(p) for p in args.experiments_roots)
    if args.experiments_root:
        root_path = Path(args.experiments_root)
        if root_path not in root_candidates:
            root_candidates.append(root_path)

    if not root_candidates:
        raise ValueError("請指定 --experiments-root 或 --experiments-roots")

    valid_roots = [p for p in root_candidates if p.is_dir()]
    invalid_roots = [p for p in root_candidates if not p.is_dir()]
    for p in invalid_roots:
        print(f"⚠ 找不到 experiments-root: {p}")
    if not valid_roots:
        raise FileNotFoundError("沒有可用的 experiments-root")
    print(f"可用實驗根目錄: {[str(p) for p in valid_roots]}")

    # 相容既有流程: 先用第一個 root 跑主流程，缺檔時再由 run_for_metric 中多根解析
    root = valid_roots[0]

    # 4) 準備輸出目錄
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # ── 依 --metric / --target 決定要跑哪些輸出 ───────────
    if args.metric == "all":
        metrics_to_run = ["f1", "precision", "recall"]
    else:
        metrics_to_run = [args.metric]

    if args.target == "all":
        targets_to_run = ["emotion", "cause", "pair"]
    else:
        targets_to_run = [args.target]

    # 5) 逐個 target / metric 執行完整流程 (讀取、計算、繪圖、輸出報告)
    for target in targets_to_run:
        for metric in metrics_to_run:
            print(f"\n{'='*60}")
            print(f"  正在處理 target={target}, metric={metric}")
            print(f"{'='*60}")
            run_for_metric(args, labels, root, output_dir, target, metric)


if __name__ == "__main__":
    main()
