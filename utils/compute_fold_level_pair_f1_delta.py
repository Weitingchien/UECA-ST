#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
根據兩份 multi-seed summary 檔，計算逐 fold 的 Pair F1 差值。

用途:
1. 讀取兩份 summary 檔中的實驗目錄清單。
2. 依 seed 對齊兩組實驗。
3. 讀取每個實驗的 fold1_test_evaluation.txt ~ fold10_test_evaluation.txt。
4. 擷取每折的 Pair F1，計算 delta = left - right。
5. 輸出逐 fold、逐 seed 的差值表與整體摘要。

使用方式範例:
    python utils/compute_fold_level_pair_f1_delta.py \
      --left-summary results_ep_split10_t1v1te1_u7_aligned_disjoint_2019/UECA_Prompt_f1-10_i70_lr1e-5_bs8_wd0.01_nest_k5_cause_clause_knncause_clause_nestmul1_nbeta0.1_nm0.6_gamma0.5_nlmnest_st5_ste20_remove_pseudo_v2_CE_summary.txt \
      --right-summary results_ep_split10_t1v1te1_u7_aligned_disjoint_2019/UECA_Prompt_f1-10_i70_lr1e-5_bs8_wd0.01_st0_CE_summary.txt \
      --experiments-root /mnt/h/ep_split10_t1v1te1_u7_aligned_disjoint_2019 \
      --left-label ST \
      --right-label ST0
"""

from __future__ import annotations  # 啟用較新的型別標註寫法，讓回傳型別更清楚。

import argparse  # 解析命令列參數
import re  # 用正則擷取實驗目錄名稱、seed 與 Pair F1 數值
from pathlib import Path  # 用 Path 處理路徑，讓程式碼更易讀

import matplotlib.pyplot as plt  # 繪製 mean delta 長條圖

from label_utils import configure_fonts  # 套用現有專案的字型設定，避免中文顯示異常

try:  # 優先嘗試匯入 scipy，讓腳本可直接做 Wilcoxon 檢定
    from scipy.stats import wilcoxon  # 使用 Wilcoxon signed-rank test 計算 p 值
except ImportError:  # 若環境沒有 scipy，就保留為 None，稍後以友善訊息提示
    wilcoxon = None


# 這個正則用來從 summary 檔中抓出每一個實驗資料夾名稱
SUMMARY_EXPERIMENT_PATTERN = re.compile(r"^\s*-\s*(prompt_ECPE_few_shot_ST_[^\s]+)\s*$")

# 這個正則用來從實驗資料夾名稱中抓出 seed 編號
SEED_PATTERN = re.compile(r"seed(\d+)")

# 這個正則用來從 fold*_test_evaluation.txt 中抓出 F1 三元組
PAIR_F1_PATTERN = re.compile(
    r"F1 Score \(Emotion, Cause, Pair\):\s*([\d.]+),\s*([\d.]+),\s*([\d.]+)"
)

# 這個正則用來從 summary 檔的「10 折累計統計結果」區塊抓出 Pair (m1) 的 micro F1
SUMMARY_PAIR_MICRO_PATTERN = re.compile(
    r"10 折累計統計結果.*?Pair \(m1\):.*?F1 \(micro\):\s*([\d.]+)%",
    re.DOTALL,
)

PAIR_M1_MATCH_BG_COLOR = "#DDDDDD"


def parse_args() -> argparse.Namespace:
    """解析命令列參數。"""
    # 建立 argparse 解析器，描述這支工具的用途。
    parser = argparse.ArgumentParser(description="計算兩組實驗的逐 fold Pair F1 差值。")

    # left-summary 代表被減號左邊的那一組，例如 self-training。
    parser.add_argument("--left-summary", required=True, help="左側 summary 檔路徑。")

    # right-summary 代表被減號右邊的那一組，例如 st0 baseline。
    parser.add_argument("--right-summary", required=True, help="右側 summary 檔路徑。")

    # experiments-root 是所有 prompt_ECPE... 實驗目錄的父目錄。
    parser.add_argument("--experiments-root", required=True, help="實驗父目錄，例如 /mnt/h/ep_split10_...。")

    # 可自訂左側方法的顯示名稱，方便輸出報告閱讀。
    parser.add_argument("--left-label", default="left", help="左側方法顯示名稱。")

    # 可自訂右側方法的顯示名稱，方便輸出報告閱讀。
    parser.add_argument("--right-label", default="right", help="右側方法顯示名稱。")

    # 可指定 fold 起點，預設從 1 開始。
    parser.add_argument("--fold-start", type=int, default=1, help="起始 fold 編號。")

    # 可指定 fold 終點，預設到 10 結束。
    parser.add_argument("--fold-end", type=int, default=10, help="結束 fold 編號。")

    # 可指定 Wilcoxon 的對立假設，預設採用較保守的雙尾檢定。
    parser.add_argument(
        "--wilcoxon-alternative",
        default="two-sided",
        choices=["two-sided", "greater", "less"],
        help="Wilcoxon signed-rank test 的對立假設。",
    )

    # 若提供這個參數，就把每個 fold 的 mean delta 另外畫成一張長條圖。
    parser.add_argument(
        "--mean-delta-plot",
        default=None,
        help="可選的 mean delta 長條圖輸出路徑，例如 png/mean_delta_bar.png。",
    )

    parser.add_argument(
        "--zero-based-y-axis",
        action="store_true",
        help="若所有 mean delta 皆非負，將長條圖 Y 軸下限設為 0。",
    )

    # 若開啟，mean delta 圖會套用與 pair_m1_f1_vs_k_emotion_clause_nestmul1_no_errorbar.png 相同的灰色背景樣式。
    parser.add_argument(
        "--match-pair-m1-background",
        action="store_true",
        help="若提供，mean delta 長條圖將套用與 pair_m1_f1_vs_k_emotion_clause_nestmul1_no_errorbar.png 相同的背景顏色與水平格線樣式。",
    )

    # 若提供 output，就把報告也寫到檔案。
    parser.add_argument("--output", default=None, help="可選的輸出報告檔案路徑。")

    # 回傳解析完的參數物件。
    return parser.parse_args()


def read_text(path: Path) -> str:
    """用統一設定讀文字檔。"""
    # 使用 utf-8 讀取，若有少數異常字元也不讓程式中斷。
    return path.read_text(encoding="utf-8", errors="ignore")


def extract_experiment_names(summary_path: Path) -> list[str]:
    """從 summary 檔中擷取實驗資料夾名稱。"""
    # 先讀入整份 summary 文字內容。
    text = read_text(summary_path)

    # 準備一個 list 存放抓到的實驗目錄名稱。
    experiment_names: list[str] = []

    # 逐行掃描 summary 檔，找出「  - prompt_ECPE...」格式的行。
    for line in text.splitlines():
        # 若這一行符合實驗目錄的格式，就把名稱取出來。
        match = SUMMARY_EXPERIMENT_PATTERN.match(line)
        if match:
            experiment_names.append(match.group(1))

    # 如果一個都抓不到，代表輸入的 summary 不是預期格式，直接報錯。
    if not experiment_names:
        raise ValueError(f"無法從 summary 中解析任何實驗目錄: {summary_path}")

    # 回傳實驗目錄名稱清單。
    return experiment_names


def extract_seed(experiment_name: str) -> str:
    """從實驗目錄名稱中擷取 seed，並回傳像 seed20 這種格式。"""
    # 用正則尋找 seed 後面的數字。
    match = SEED_PATTERN.search(experiment_name)

    # 若找不到 seed，就表示這個資料夾名稱不符合預期格式。
    if match is None:
        raise ValueError(f"無法從實驗目錄名稱擷取 seed: {experiment_name}")

    # 回傳標準化後的 seed 名稱，例如 seed20。
    return f"seed{match.group(1)}"


def seed_sort_key(seed_name: str) -> tuple[int, str]:
    """讓 seed20、seed42、seed60 可以按照數字排序。"""
    # 再次用正則抓出數字部分，供排序使用。
    match = SEED_PATTERN.search(seed_name)

    # 若格式正常，就用數字排序。
    if match:
        return (int(match.group(1)), seed_name)

    # 若格式異常，就退回字串排序，避免整體流程崩潰。
    return (10**9, seed_name)


def resolve_experiment_map(summary_path: Path, experiments_root: Path) -> dict[str, Path]:
    """把 summary 裡的實驗目錄解析成 seed -> Path 的對照表。"""
    # 先從 summary 中抓出所有實驗目錄名稱。
    experiment_names = extract_experiment_names(summary_path)

    # 建立一個字典，把每個 seed 對應到實際的實驗資料夾路徑。
    experiment_map: dict[str, Path] = {}

    # 逐一處理每個實驗目錄名稱。
    for experiment_name in experiment_names:
        # 從資料夾名稱中擷取 seed，例如 seed20。
        seed_name = extract_seed(experiment_name)

        # 把 experiments_root 和目錄名稱接成完整路徑。
        experiment_dir = experiments_root / experiment_name

        # 若資料夾不存在，就直接報錯，避免後面讀檔時更難追問題。
        if not experiment_dir.is_dir():
            raise FileNotFoundError(f"找不到實驗資料夾: {experiment_dir}")

        # 若同一個 summary 裡出現重複 seed，代表配對會有歧義，直接報錯。
        if seed_name in experiment_map:
            raise ValueError(f"同一份 summary 中出現重複 seed: {seed_name}")

        # 把這個 seed 對應到它的實驗資料夾。
        experiment_map[seed_name] = experiment_dir

    # 回傳 seed 到實驗路徑的對照表。
    return experiment_map


def parse_pair_f1(test_evaluation_path: Path) -> float:
    """從單一 fold 的 test_evaluation.txt 擷取 Pair F1。"""
    # 讀入 test_evaluation 文字內容。
    text = read_text(test_evaluation_path)

    # 找出所有 F1 Score 三元組，理論上通常只會有一筆，但取最後一筆最保險。
    matches = PAIR_F1_PATTERN.findall(text)

    # 若找不到任何 F1 Score 行，就代表檔案格式不符預期。
    if not matches:
        raise ValueError(f"無法從檔案擷取 Pair F1: {test_evaluation_path}")

    # 取最後一筆 match 的第 3 個值，也就是 Pair F1。
    return float(matches[-1][2])


def parse_summary_pair_m1_micro(summary_path: Path) -> float | None:
    """從 summary 檔的 10 折累計統計結果區塊擷取 Pair (m1) 的 micro F1。"""
    # 讀入整份 summary 文字。
    text = read_text(summary_path)

    # 嘗試用正則找到 Pair (m1) 的 micro F1 百分比。
    match = SUMMARY_PAIR_MICRO_PATTERN.search(text)

    # 若找不到就回傳 None，表示這個欄位不影響主流程，只是少了交叉檢查資訊。
    if match is None:
        return None

    # 找得到就轉成浮點數後回傳。
    return float(match.group(1))


def ensure_same_seed_set(left_map: dict[str, Path], right_map: dict[str, Path]) -> list[str]:
    """確認左右兩組 summary 的 seed 集合完全一致。"""
    # 取出左右兩邊的 seed 集合。
    left_seeds = set(left_map.keys())
    right_seeds = set(right_map.keys())

    # 若 seed 集合不同，就無法做成對比較，直接報錯。
    if left_seeds != right_seeds:
        raise ValueError(
            "左右兩組 summary 的 seed 集合不一致，無法逐 seed 對齊比較。\n"
            f"left: {sorted(left_seeds, key=seed_sort_key)}\n"
            f"right: {sorted(right_seeds, key=seed_sort_key)}"
        )

    # 回傳排序後的 seed 清單，供後續固定欄位順序使用。
    return sorted(left_seeds, key=seed_sort_key)


def format_signed(value: float) -> str:
    """把差值格式化成帶正負號的小數字串。"""
    # 例如 +1.23 或 -0.45。
    return f"{value:+.2f}"


def build_fold_rows(
    left_map: dict[str, Path],
    right_map: dict[str, Path],
    seed_names: list[str],
    fold_start: int,
    fold_end: int,
) -> tuple[list[dict], dict[str, list[float]]]:
    """建立逐 fold 的 Pair F1 差值資料。"""
    # 準備一個 list 存放每一折的結果列。
    fold_rows: list[dict] = []

    # 準備一個 dict 累積每個 seed 的所有 delta，方便最後算平均。
    seed_deltas: dict[str, list[float]] = {seed_name: [] for seed_name in seed_names}

    # 從 fold_start 一路處理到 fold_end。
    for fold_id in range(fold_start, fold_end + 1):
        # 每一折用一個 dict 保存所有 seed 的原值與差值。
        row = {
            "fold": fold_id,
            "values": {},
            "mean_delta": 0.0,
        }

        # 這個 list 用來暫存本折三個 seed 的 delta，最後算本折平均。
        current_fold_deltas: list[float] = []

        # 逐一處理每個 seed。
        for seed_name in seed_names:
            # 組出左側實驗的 fold test 檔路徑。
            left_test_path = left_map[seed_name] / f"fold{fold_id}_test_evaluation.txt"

            # 組出右側實驗的 fold test 檔路徑。
            right_test_path = right_map[seed_name] / f"fold{fold_id}_test_evaluation.txt"

            # 從左側檔案讀出 Pair F1，並乘以 100 轉成百分比表示。
            left_pair_f1 = parse_pair_f1(left_test_path) * 100

            # 從右側檔案讀出 Pair F1，並乘以 100 轉成百分比表示。
            right_pair_f1 = parse_pair_f1(right_test_path) * 100

            # 差值定義為 left - right，單位是百分點。
            delta = left_pair_f1 - right_pair_f1

            # 把這個 seed 的詳細數值記到 row 中。
            row["values"][seed_name] = {
                "left": left_pair_f1,
                "right": right_pair_f1,
                "delta": delta,
            }

            # 把 delta 加到本折暫存 list。
            current_fold_deltas.append(delta)

            # 同時把 delta 累積到該 seed 的歷史清單。
            seed_deltas[seed_name].append(delta)

        # 計算本折三個 seed 的平均 delta。
        row["mean_delta"] = sum(current_fold_deltas) / len(current_fold_deltas)

        # 把這一折的資料加入 fold_rows。
        fold_rows.append(row)

    # 回傳逐 fold 結果與逐 seed 的 delta 清單。
    return fold_rows, seed_deltas


def build_fold_mean_deltas(fold_rows: list[dict]) -> list[float]:
    """把逐 fold 結果轉成 Wilcoxon 需要的 fold 平均 delta 清單。"""
    # 逐一取出每一折的 mean_delta，形成一個長度等於 fold 數的 list。
    return [row["mean_delta"] for row in fold_rows]


def compute_wilcoxon_result(fold_mean_deltas: list[float], alternative: str) -> dict | None:
    """對逐 fold 平均 delta 做 Wilcoxon signed-rank test。"""
    # 若環境沒有 scipy，就回傳 None，稍後在報告中說明原因。
    if wilcoxon is None:
        return None

    # 移除完全等於 0 的差值，因為 Wilcoxon 的 wilcox 模式會忽略這些點。
    non_zero_deltas = [delta for delta in fold_mean_deltas if delta != 0]

    # 若全部都是 0，就表示兩邊完全沒有差異，無法做有意義的檢定。
    if not non_zero_deltas:
        return {
            "available": False,
            "reason": "所有 fold 平均 delta 都等於 0，無法進行 Wilcoxon 檢定。",
        }

    # 呼叫 scipy 的 Wilcoxon signed-rank test，直接對差值是否偏離 0 做檢定。
    result = wilcoxon(non_zero_deltas, alternative=alternative, zero_method="wilcox", method="auto")

    # 統計正、負、零差值的數量，方便解讀 p 值來源。
    positive_count = sum(1 for delta in fold_mean_deltas if delta > 0)
    negative_count = sum(1 for delta in fold_mean_deltas if delta < 0)
    zero_count = sum(1 for delta in fold_mean_deltas if delta == 0)

    # 回傳整理好的檢定資訊，供報告統一輸出。
    return {
        "available": True,
        "statistic": float(result.statistic),
        "pvalue": float(result.pvalue),
        "sample_size": len(non_zero_deltas),
        "total_folds": len(fold_mean_deltas),
        "positive_count": positive_count,
        "negative_count": negative_count,
        "zero_count": zero_count,
        "alternative": alternative,
    }


def build_mean_delta_plot_title(left_label: str, right_label: str) -> str:
    """建立 mean delta 長條圖標題。"""
    # 標題直接說明差值方向，讀圖時比較不容易搞混。
    return f"各折 Pair F1 差值({left_label} - {right_label})"


def apply_pair_m1_background_style(fig, ax) -> None:
    """套用與 pair_m1_f1_vs_k_emotion_clause_nestmul1_no_errorbar.png 相同的灰底與格線風格。"""
    fig.patch.set_facecolor(PAIR_M1_MATCH_BG_COLOR)
    ax.set_facecolor(PAIR_M1_MATCH_BG_COLOR)
    ax.set_axisbelow(False)
    ax.grid(True, axis="y", linestyle="--", linewidth=1.1, alpha=0.95, color="#000000")
    for gridline in ax.get_ygridlines():
        gridline.set_zorder(10)


def plot_mean_delta_barchart(
    fold_rows: list[dict],
    left_label: str,
    right_label: str,
    output_path: Path,
    match_pair_m1_background: bool = False,
    zero_based_y_axis: bool = False,
) -> None:
    """把每個 fold 的 mean delta 畫成以 0 為基準線的長條圖。"""
    # 套用現有專案的字型設定，讓中文與負號都能正常顯示。
    configure_fonts(font_size=16)

    # 準備 X 軸標籤，例如 fold1、fold2 ...。
    fold_labels = [f"fold{row['fold']}" for row in fold_rows]

    # 準備 Y 軸資料，也就是每一折的 mean delta。
    mean_deltas = [row["mean_delta"] for row in fold_rows]

    # 從 0 起始時不可包含負值，避免把任何長條靜默裁掉。
    if zero_based_y_axis and any(delta < 0 for delta in mean_deltas):
        raise ValueError("--zero-based-y-axis 只能用於所有 mean delta 皆非負的圖。")

    # 依正負值給不同顏色，方便一眼看出哪幾折是正增益、哪幾折是負增益。
    colors = ["#2E8B57" if delta >= 0 else "#C44E52" for delta in mean_deltas]

    # 建立圖與座標軸。
    fig, ax = plt.subplots(figsize=(12, 6.5))

    # 依使用者選項決定是否套用與 pair_m1 圖一致的背景樣式。
    if match_pair_m1_background:
        apply_pair_m1_background_style(fig, ax)

    # 畫出每一折的長條。
    bars = ax.bar(fold_labels, mean_deltas, color=colors, edgecolor="white", linewidth=1.2)

    # 畫出 y=0 的基準線，方便判斷正負差值。
    ax.axhline(0, color="#333333", linewidth=1.5)

    # 設定圖表標題與座標軸標籤。
    ax.set_title(build_mean_delta_plot_title(left_label, right_label), pad=14)
    ax.set_xlabel("Fold")
    ax.set_ylabel("Pair F1 差值")

    # 預設沿用原本較淡的格線；若已套用 pair_m1 樣式則維持該樣式設定。
    if not match_pair_m1_background:
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)
        ax.set_axisbelow(True)

    # 依據最大絕對值自動保留上下空間，避免數值標註貼邊。
    max_abs_delta = max(abs(delta) for delta in mean_deltas) if mean_deltas else 1.0
    margin = max(0.6, max_abs_delta * 0.18)
    if zero_based_y_axis:
        ax.set_ylim(0, max_abs_delta + margin)
    else:
        ax.set_ylim(-max_abs_delta - margin, max_abs_delta + margin)

    # 在每個長條上方或下方標註數值。
    for bar, delta in zip(bars, mean_deltas):
        # 依正負決定文字應放在長條上方還是下方。
        vertical_offset = 0.12 if delta >= 0 else -0.12
        vertical_align = "bottom" if delta >= 0 else "top"

        # 在長條中心位置寫上帶正負號的數值。
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            delta + vertical_offset,
            format_signed(delta),
            ha="center",
            va=vertical_align,
            fontsize=12,
        )

    # 讓版面自動調整，避免標籤被裁切。
    fig.tight_layout()

    # 確保輸出目錄存在，再把圖片寫到指定路徑。
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())

    # 關閉 figure，避免腳本多次呼叫時累積記憶體與圖表狀態。
    plt.close(fig)


def build_report(
    left_summary: Path,
    right_summary: Path,
    experiments_root: Path,
    left_label: str,
    right_label: str,
    left_map: dict[str, Path],
    right_map: dict[str, Path],
    seed_names: list[str],
    fold_rows: list[dict],
    seed_deltas: dict[str, list[float]],
    wilcoxon_result: dict | None,
) -> str:
    """把結果整理成容易閱讀的純文字報告。"""
    # 建立一個 list，逐行累積最後要輸出的文字。
    lines: list[str] = []

    # 報告標題。
    lines.append("=" * 78)
    lines.append("逐 Fold Pair F1 差值報告")
    lines.append("=" * 78)

    # 說明差值的定義，避免之後看報告時忘記正負方向。
    lines.append(f"差值定義: delta = {left_label} - {right_label} (單位: 百分點)")
    lines.append(f"left_summary:  {left_summary}")
    lines.append(f"right_summary: {right_summary}")
    lines.append(f"experiments_root: {experiments_root}")
    lines.append("")

    # 顯示左右兩組如何按 seed 對齊，方便人工檢查有沒有配錯。
    lines.append("Seed 對齊結果:")
    for seed_name in seed_names:
        lines.append(f"  - {seed_name}")
        lines.append(f"    {left_label}:  {left_map[seed_name].name}")
        lines.append(f"    {right_label}: {right_map[seed_name].name}")
    lines.append("")

    # 建立逐 fold 的差值表表頭。
    header_cells = ["fold"] + seed_names + ["mean_delta"]
    lines.append("逐 fold 差值表:")
    lines.append("  " + "\t".join(header_cells))

    # 逐折輸出每個 seed 的 delta 與本折平均 delta。
    for row in fold_rows:
        cells = [str(row["fold"])]
        for seed_name in seed_names:
            cells.append(format_signed(row["values"][seed_name]["delta"]))
        cells.append(format_signed(row["mean_delta"]))
        lines.append("  " + "\t".join(cells))
    lines.append("")

    # 輸出每個 seed 的平均差值。
    lines.append("各 seed 差值:")
    for seed_name in seed_names:
        mean_delta = sum(seed_deltas[seed_name]) / len(seed_deltas[seed_name])
        lines.append(f"  - {seed_name}: {format_signed(mean_delta)}")

    # 把 30 個 fold-seed delta 全部展平後，再算一個總平均。
    all_deltas = [delta for deltas in seed_deltas.values() for delta in deltas]
    overall_mean_delta = sum(all_deltas) / len(all_deltas)
    lines.append(f"  - 全部 fold-seed delta 的平均: {format_signed(overall_mean_delta)}")
    lines.append("")

    # 額外把每個 fold 的 seed 平均 delta 列出，這也是 Wilcoxon 的檢定輸入。
    fold_mean_deltas = build_fold_mean_deltas(fold_rows)
    lines.append("Wilcoxon 檢定輸入（每 fold 的 seed 平均 delta）:")
    lines.append("  - " + ", ".join(f"fold{index}: {format_signed(delta)}" for index, delta in enumerate(fold_mean_deltas, start=1)))
    lines.append("")

    # 若有 Wilcoxon 結果，就把檢定統計量與 p 值輸出。
    if wilcoxon_result is not None:
        lines.append("Wilcoxon signed-rank test:")

        # 若 available 為 False，表示檢定條件不成立或環境缺少依賴。
        if not wilcoxon_result.get("available", False):
            lines.append(f"  - 無法計算: {wilcoxon_result['reason']}")
        else:
            lines.append("  - 檢定單位: 每個 fold 的 seed 平均 delta")
            lines.append(f"  - 對立假設: {wilcoxon_result['alternative']}")
            lines.append(
                "  - 正 / 負 / 零差值 fold 數: "
                f"{wilcoxon_result['positive_count']} / "
                f"{wilcoxon_result['negative_count']} / "
                f"{wilcoxon_result['zero_count']}"
            )
            lines.append(f"  - 有效樣本數: {wilcoxon_result['sample_size']} folds")
            lines.append(f"  - W 統計量: {wilcoxon_result['statistic']:.4f}")
            lines.append(f"  - p 值: {wilcoxon_result['pvalue']:.6f}")
            lines.append("  - p 值越小，代表在『其實沒有穩定差異』這個前提下，觀察到目前這組差值的機率越低。")

            # 這裡用最常見的 0.05 門檻做直觀提醒，但不把它寫成唯一標準。
            if wilcoxon_result["pvalue"] < 0.05:
                lines.append("  - 以 0.05 為常見門檻時，可視為差值具有統計顯著性。")
            else:
                lines.append("  - 以 0.05 為常見門檻時，現有證據不足以宣稱差值具有統計顯著性。")

        lines.append("")

    # 額外從 summary 檔中抓 Pair (m1) micro F1，作為交叉檢查。
    left_summary_pair = parse_summary_pair_m1_micro(left_summary)
    right_summary_pair = parse_summary_pair_m1_micro(right_summary)

    # 若兩邊都能抓到，就把 summary micro delta 也列出來。
    if left_summary_pair is not None and right_summary_pair is not None:
        summary_micro_delta = left_summary_pair - right_summary_pair
        lines.append("Summary 交叉檢查:")
        lines.append(f"  - {left_label} Pair (m1) micro F1:  {left_summary_pair:.2f}%")
        lines.append(f"  - {right_label} Pair (m1) micro F1: {right_summary_pair:.2f}%")
        lines.append(f"  - summary micro delta: {format_signed(summary_micro_delta)}")
        lines.append("")

    # 補充說明，提醒使用者 summary micro delta 和 fold-level 平均 delta 不一定完全相同。
    lines.append("說明:")
    lines.append("  - fold-level delta 平均 = 先算每折差值，再做平均。")
    lines.append("  - Wilcoxon 的 p 值 = 用每 fold 平均 delta 檢查差值是否系統性偏離 0。")
    lines.append("  - summary micro delta = 先合併 10 折預測，再算 micro F1，最後做相減。")
    lines.append("  - 因為計算順序不同，兩者可能接近但不會保證完全相同。")
    lines.append("")

    # 最後附上每個 seed 在每一折的原始 Pair F1 與差值，方便人工核對。
    lines.append("逐 seed 詳細原值:")
    for seed_name in seed_names:
        lines.append(f"  [{seed_name}]")
        for row in fold_rows:
            value = row["values"][seed_name]
            lines.append(
                "    "
                f"fold{row['fold']}: "
                f"{left_label}={value['left']:.2f}%  "
                f"{right_label}={value['right']:.2f}%  "
                f"delta={format_signed(value['delta'])}"
            )
        lines.append("")

    # 把所有行用換行符接起來，形成最終報告文字。
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    """主流程。"""
    # 先解析命令列參數。
    args = parse_args()

    # 把字串路徑轉成 Path，後續處理會比較一致。
    left_summary = Path(args.left_summary)
    right_summary = Path(args.right_summary)
    experiments_root = Path(args.experiments_root)

    # 基本存在性檢查，避免用錯路徑時得到模糊錯誤訊息。
    if not left_summary.is_file():
        raise FileNotFoundError(f"找不到 left summary: {left_summary}")
    if not right_summary.is_file():
        raise FileNotFoundError(f"找不到 right summary: {right_summary}")
    if not experiments_root.is_dir():
        raise FileNotFoundError(f"找不到 experiments_root: {experiments_root}")

    # 依 summary 解析出左右兩組的 seed -> 實驗目錄對照表。
    left_map = resolve_experiment_map(left_summary, experiments_root)
    right_map = resolve_experiment_map(right_summary, experiments_root)

    # 確保兩組 summary 的 seed 可一一對齊。
    seed_names = ensure_same_seed_set(left_map, right_map)

    # 建立逐 fold 的 Pair F1 差值資料。
    fold_rows, seed_deltas = build_fold_rows(
        left_map=left_map,
        right_map=right_map,
        seed_names=seed_names,
        fold_start=args.fold_start,
        fold_end=args.fold_end,
    )

    # 取出每個 fold 的 seed 平均 delta，準備做 Wilcoxon signed-rank test。
    fold_mean_deltas = build_fold_mean_deltas(fold_rows)

    # 以逐 fold 平均 delta 為配對單位，直接計算 Wilcoxon 的統計量與 p 值。
    wilcoxon_result = compute_wilcoxon_result(
        fold_mean_deltas=fold_mean_deltas,
        alternative=args.wilcoxon_alternative,
    )

    # 若 scipy 不存在，就把原因包成統一格式，讓報告仍能順利輸出。
    if wilcoxon_result is None:
        wilcoxon_result = {
            "available": False,
            "reason": "目前環境缺少 scipy，無法計算 Wilcoxon 檢定。",
        }

    # 把計算結果組成純文字報告。
    report = build_report(
        left_summary=left_summary,
        right_summary=right_summary,
        experiments_root=experiments_root,
        left_label=args.left_label,
        right_label=args.right_label,
        left_map=left_map,
        right_map=right_map,
        seed_names=seed_names,
        fold_rows=fold_rows,
        seed_deltas=seed_deltas,
        wilcoxon_result=wilcoxon_result,
    )

    # 若有指定圖片輸出路徑，就另外生成一張 mean delta 長條圖。
    if args.mean_delta_plot:
        plot_mean_delta_barchart(
            fold_rows=fold_rows,
            left_label=args.left_label,
            right_label=args.right_label,
            output_path=Path(args.mean_delta_plot),
            match_pair_m1_background=args.match_pair_m1_background,
            zero_based_y_axis=args.zero_based_y_axis,
        )

    # 先把報告印到終端，方便直接查看。
    print(report, end="")

    # 若使用者指定了輸出檔路徑，就另外寫檔保存。
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report, encoding="utf-8")
        print(f"\n報告已寫入: {output_path}")

    # 若有輸出圖檔，也在終端補一行提示。
    if args.mean_delta_plot:
        print(f"\nmean delta 長條圖已寫入: {args.mean_delta_plot}")


# 只有直接執行此檔案時，才進入主流程。
if __name__ == "__main__":
    main()
