"""統計三個 seed、十折與五輪的 S0／S1／S2 情緒類別散度關係。"""

# 本腳本是訓練完成後的離線診斷分析：
# 1. 真實情緒子句與真實 emotion_category 只用來建立 S0/S1/S2，不曾回饋模型或 NeST 選樣。
# 2. 主要分析量是當輪 raw divergence；EMA、抽樣結果與納入率只作選樣機制的輔助描述。
# 3. Macro 依 round→seed→fold 逐層等權；pooled 則按 document-round observations 加權。
# 4. 結果只能說明條件式關聯，不能證明情緒類別一致性造成散度下降或效能提升。

import argparse
import csv
import itertools
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import fmean

import matplotlib.pyplot as plt
import numpy as np

from label_utils import configure_fonts


# 以目前腳本所在位置向上兩層取得專案根目錄，避免依賴執行指令時的工作目錄。
REPO_ROOT = Path(__file__).resolve().parents[1]
# summary 只負責列出三個 seed 的實驗目錄名稱；逐輪資料仍從各實驗目錄讀取。
DEFAULT_SUMMARY = REPO_ROOT / (
    "results_ep_split10_t1v1te1_u7_aligned_disjoint_2019/"
    "UECA_Prompt_f1-10_i70_lr1e-5_bs8_wd0.01_nest_k5_emotion_clause_"
    "knnemotion_clause_nestmul1_nbeta0.1_nm0.6_gamma0.5_nlmnest_st5_"
    "ste20_remove_pseudo_v2_CE_summary.txt"
)
# 原始切分檔提供 doc_id、pair、子句位置及真實 emotion_category。
DEFAULT_DATASET = REPO_ROOT / "split10_train1_val1_test1_unlabeled7_aligned_disjoint_2019"
# 所有分析結果集中寫入獨立目錄，不修改原始實驗與資料集檔案。
DEFAULT_OUTPUT = REPO_ROOT / "analysis/multiseed_round_emotion_category_divergence"
# 論文圖與統計表分開保存，方便直接引用。
DEFAULT_PNG_DIR = REPO_ROOT / "png/multiseed_round_emotion_category_divergence"
BAR_COLORS = ("#4C78A8", "#D68C45", "#C44E52", "#59A14F", "#B279A2")
GRAY_BACKGROUND = "#DDDDDD"
# 同一個 H 槽在 WSL 與 Windows 中的路徑表示不同，因此各保留一種表示法。
WSL_EXPERIMENTS_ROOT = Path("/mnt/h/ep_split10_t1v1te1_u7_aligned_disjoint_2019")
WINDOWS_EXPERIMENTS_ROOT = Path("H:/ep_split10_t1v1te1_u7_aligned_disjoint_2019")
# 若目前在 WSL 且 /mnt/h 存在就使用 WSL 路徑，否則使用 Windows 的 H 槽路徑。
DEFAULT_EXPERIMENTS_ROOT = (
    WSL_EXPERIMENTS_ROOT if WSL_EXPERIMENTS_ROOT.exists() else WINDOWS_EXPERIMENTS_ROOT
)


def parse_args():
    # 所有參數都有符合目前實驗的預設值，完整分析時不必額外輸入參數。
    parser = argparse.ArgumentParser(
        description="依老師定義統計多 seed、逐輪的 S0／S1／S2 與 NeST 散度。"
    )
    # 指定列出三個 seed 實驗目錄的彙整文字檔。
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    # 指定三個實驗目錄共同所在的根目錄。
    parser.add_argument("--experiments_root", type=Path, default=DEFAULT_EXPERIMENTS_ROOT)
    # 指定 fold{n}_train.json 與 fold{n}_unlabeled.json 所在目錄。
    parser.add_argument("--dataset_dir", type=Path, default=DEFAULT_DATASET)
    # 指定 CSV 與 JSON 統計結果的輸出目錄。
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    # fold 範圍可用於小範圍檢查；正式實驗預設分析 Fold 1 至 Fold 10。
    parser.add_argument("--start_fold", type=int, default=1)
    parser.add_argument("--end_fold", type=int, default=10)
    # 每折預設分析五輪 self-training。
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument(
        "--analysis_k",
        type=int,
        help="只取來源實驗已保存的前 k 位最近鄰，進行事後敏感度分析",
    )
    parser.add_argument("--plot_output", type=Path)
    parser.add_argument("--count_plot_output", type=Path)
    parser.add_argument("--selected_count_plot_output", type=Path)
    parser.add_argument("--selected_count_f1_plot_output", type=Path)
    parser.add_argument("--plot_selected_count_with_emotion_f1", action="store_true")
    parser.add_argument(
        "--plot_each_seed",
        action="store_true",
        help="分別輸出各 seed 的五輪累積 S₁／S₂／S₃ 與五輪平均驗證集 Emotion F1 圖。",
    )
    parser.add_argument(
        "--plot_round_quality_vs_validation_f1_change",
        action="store_true",
        help="以 seed×fold×round 配對實際納入樣本組成與驗證 Emotion F1 變化。",
    )
    parser.add_argument(
        "--emotion_f1_split",
        choices=("test", "validation"),
        default="test",
        help="下方 Emotion F1 使用測試集或驗證集結果。",
    )
    parser.add_argument(
        "--validation_f1_mode",
        choices=("best", "round_mean"),
        default="best",
        help=(
            "驗證集模式的聚合方式：best 取各 seed 的最佳輪次；"
            "round_mean 平均第 1 輪至 --rounds 指定輪次。"
        ),
    )
    parser.add_argument("--background", choices=("gray", "white"), default="white")
    parser.add_argument(
        "--plot_aggregation",
        choices=("macro", "micro", "all_micro"),
        default="macro",
    )
    return parser.parse_args()


def summary_name(summary_path):
    # 圖檔以前綴識別實驗設定；只移除固定的 _summary.txt，不改動其餘名稱。
    filename = Path(summary_path).name
    suffix = "_summary.txt"
    return filename[: -len(suffix)] if filename.endswith(suffix) else Path(filename).stem


def posthoc_tag(analysis_k):
    """為事後鄰居敏感度分析的輸出檔名加入可辨識標記。"""

    return f"_posthoc_k{analysis_k}" if analysis_k is not None else ""


def load_json(path):
    # 原始中文內容使用 UTF-8 儲存，直接回傳反序列化後的 list 或 dict。
    with Path(path).open(encoding="utf-8") as file:
        return json.load(file)


def write_csv(path, rows):
    # utf-8-sig 會加入 BOM，方便在 Windows Excel 中正確顯示繁體中文。
    with Path(path).open("w", newline="", encoding="utf-8-sig") as file:
        # 每種輸出表的所有 row 都使用固定欄位順序，因此以第一列的 key 建立表頭。
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def mean(values):
    # None 代表該組沒有可計算值；排除 None 後才計算平均，不能把缺值當成 0。
    values = [value for value in values if value is not None]
    return fmean(values) if values else None


def ratio(numerator, denominator):
    # 分母為 0 時回傳 None，讓 CSV 留白、JSON 寫入 null，避免產生誤導性的 0。
    return numerator / denominator if denominator else None


def format_divergence(value, decimal_places=4):
    # 小到會被指定小數位數四捨五入成 0 的非零散度改用科學記號，避免圖上誤顯示為 0。
    scientific_threshold = max(0.001, 0.5 * 10 ** (-decimal_places))
    return (
        f"{value:.2e}"
        if 0 < value < scientific_threshold
        else f"{value:.{decimal_places}f}"
    )


def format_count(value):
    """整數計數不顯示小數；跨 seed 平均後的計數保留兩位小數。"""

    return str(round(value)) if math.isclose(value, round(value)) else f"{value:.2f}"


def plot_fold_divergence(
    fold_rows,
    output_path,
    background,
    aggregation,
    title_suffix="",
):
    """依指定聚合方式將每折 A1、A2 畫成論文用分組長條圖。"""

    configure_fonts(font_size=16)
    if aggregation == "all_micro":
        # 原論文字型不含 Unicode 下標數字，加入 DejaVu Serif 作為下標字形的後備字型。
        plt.rcParams["font.family"] = [*plt.rcParams["font.family"], "DejaVu Serif"]
    positions = list(range(len(fold_rows)))
    width = 0.32
    fields = {
        "macro": ("raw_A1_macro", "raw_A2_macro"),
        "micro": ("raw_A1_paired_valid_micro", "raw_A2_paired_valid_micro"),
        "all_micro": ("raw_A1_observation_pooled", "raw_A2_observation_pooled"),
    }
    a1_field, a2_field = fields[aggregation]
    a1_values = [row[a1_field] for row in fold_rows]
    a2_values = [row[a2_field] for row in fold_rows]
    label_decimal_places = 2 if aggregation == "all_micro" else 4
    a1_legend = "A₁（S₁）" if aggregation == "all_micro" else "A1（S1）"
    a2_legend = "A₂（S₂）" if aggregation == "all_micro" else "A2（S2）"
    assert all(value is not None and math.isfinite(value) for value in a1_values + a2_values)

    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=180)
    facecolor = GRAY_BACKGROUND if background == "gray" else "white"
    fig.patch.set_facecolor(facecolor)
    ax.set_facecolor(facecolor)

    a1_bars = ax.bar(
        [position - width / 2 for position in positions],
        a1_values,
        width,
        label=a1_legend,
        color=BAR_COLORS[0],
        edgecolor="white",
        linewidth=1.2,
        zorder=2,
    )
    a2_bars = ax.bar(
        [position + width / 2 for position in positions],
        a2_values,
        width,
        label=a2_legend,
        color=BAR_COLORS[1],
        edgecolor="white",
        linewidth=1.2,
        zorder=2,
    )
    ax.bar_label(
        a1_bars,
        labels=[format_divergence(value, label_decimal_places) for value in a1_values],
        padding=3,
        fontsize=12,
    )
    ax.bar_label(
        a2_bars,
        labels=[format_divergence(value, label_decimal_places) for value in a2_values],
        padding=3,
        fontsize=12,
    )

    ax.set_xticks(positions)
    ax.set_xticklabels([str(row["fold"]) for row in fold_rows])
    aggregation_labels = {
        "macro": "Macro",
        "micro": "Paired-valid Micro",
    }
    title = (
        "S₁ 與 S₂ 的平均散度"
        if aggregation == "all_micro"
        else f"各折 S1 與 S2 的平均 NeST 散度（{aggregation_labels[aggregation]}）"
    )
    ax.set_title(f"{title}{title_suffix}", pad=14)
    ax.set_xlabel("Fold")
    ax.set_ylabel("散度" if aggregation == "all_micro" else "平均 NeST 散度")
    ax.set_ylim(0, max(a1_values + a2_values) * 1.28)
    grid_alpha = 0.95 if background == "gray" else 0.35
    grid_width = 1.1 if background == "gray" else 1.0
    ax.grid(
        True,
        axis="y",
        linestyle="--",
        linewidth=grid_width,
        alpha=grid_alpha,
        color="black",
        zorder=10,
    )
    ax.set_axisbelow(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=True)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_fold_sample_counts(
    fold_rows,
    output_path,
    background,
    series=(("s1_n", "S₁"), ("s2_n", "S₂")),
    title="S₁ 與 S₂ 的樣本數量",
):
    """將每折指定群組的樣本觀察筆數繪成並列長條圖。"""

    configure_fonts(font_size=16)
    plt.rcParams["font.family"] = [*plt.rcParams["font.family"], "DejaVu Serif"]
    positions = list(range(len(fold_rows)))
    width = min(0.9, 0.32 * len(series)) / len(series)
    all_counts = []

    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=180)
    facecolor = GRAY_BACKGROUND if background == "gray" else "white"
    fig.patch.set_facecolor(facecolor)
    ax.set_facecolor(facecolor)

    for index, (field, label) in enumerate(series):
        counts = [row[field] for row in fold_rows]
        offset = (index - (len(series) - 1) / 2) * width
        bars = ax.bar(
            [position + offset for position in positions],
            counts,
            width,
            label=label,
            color=BAR_COLORS[index],
            edgecolor="white",
            linewidth=1.2,
            zorder=2,
        )
        ax.bar_label(
            bars,
            labels=[str(value) for value in counts],
            padding=3,
            fontsize=12,
        )
        all_counts.extend(counts)

    ax.set_xticks(positions)
    ax.set_xticklabels([str(row["fold"]) for row in fold_rows])
    ax.set_title(title, pad=14)
    ax.set_xlabel("Fold")
    ax.set_ylabel("樣本數量（筆）")
    ax.set_ylim(0, max(all_counts) * 1.22)
    grid_alpha = 0.95 if background == "gray" else 0.35
    grid_width = 1.1 if background == "gray" else 1.0
    ax.grid(
        True,
        axis="y",
        linestyle="--",
        linewidth=grid_width,
        alpha=grid_alpha,
        color="black",
        zorder=10,
    )
    ax.set_axisbelow(False)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=len(series),
        frameon=True,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def load_fold_test_emotion_metrics(experiments, start_fold, end_fold):
    """計算同一折各 seed 的測試 Emotion F1，再跨 seed 取平均。"""

    fold_metrics = []
    for fold in range(start_fold, end_fold + 1):
        true_positive = predicted_positive = gold_positive = 0
        seed_f1_values = []
        row = {"fold": fold}
        for experiment in experiments:
            result_path = experiment["experiment_dir"] / f"fold{fold}_test_evaluation.txt"
            text = result_path.read_text(encoding="utf-8")
            match = re.search(
                r"Emotion:\s*預測正確的數量:(\d+(?:\.\d+)?)\s+"
                r"預測出情緒的數量:(\d+(?:\.\d+)?)\s+"
                r"實際正確情緒的數量:(\d+(?:\.\d+)?)",
                text,
            )
            assert match, f"找不到 Emotion 評估計數: {result_path}"
            tp, predicted, gold = (int(float(value)) for value in match.groups())
            true_positive += tp
            predicted_positive += predicted
            gold_positive += gold
            seed_precision = tp / predicted
            seed_recall = tp / gold
            seed_f1 = 2 * seed_precision * seed_recall / (seed_precision + seed_recall)
            seed_f1_values.append(seed_f1)
            row[f"seed{experiment['seed']}_emotion_f1"] = seed_f1

        precision = true_positive / predicted_positive
        recall = true_positive / gold_positive
        row.update(
            {
                "emotion_true_positive": true_positive,
                "emotion_predicted_positive": predicted_positive,
                "emotion_gold_positive": gold_positive,
                "emotion_precision": precision,
                "emotion_recall": recall,
                "emotion_f1": mean(seed_f1_values),
            }
        )
        fold_metrics.append(row)
    return fold_metrics


def load_fold_best_validation_emotion_f1(experiments, start_fold, end_fold):
    """讀取各 seed 自訓練期間的最佳驗證 Emotion F1，再依折取平均。"""

    fold_metrics = []
    for fold in range(start_fold, end_fold + 1):
        seed_metrics = {}
        for experiment in experiments:
            checkpoint_path = (
                experiment["experiment_dir"] / f"fold{fold}_best_val_checkpoint_emo.txt"
            )
            checkpoint_text = checkpoint_path.read_text(encoding="utf-8")
            best_round = int(
                re.search(r"stage:\s*self_training_round_(\d+)", checkpoint_text).group(1)
            )
            metrics_path = (
                experiment["experiment_dir"] / f"fold{fold}_self_training_round_metrics.csv"
            )
            with metrics_path.open(encoding="utf-8") as file:
                round_rows = list(csv.DictReader(file))
            best_row = next(
                row
                for row in round_rows
                if int(row["self_training_round"]) == best_round
            )
            seed_metrics[experiment["seed"]] = {
                "round": best_round,
                "f1": float(best_row["val_f1_emotion"]),
            }

        row = {
            "fold": fold,
            "emotion_f1": mean([item["f1"] for item in seed_metrics.values()]),
        }
        for seed, item in sorted(seed_metrics.items()):
            row[f"seed{seed}_best_round"] = item["round"]
            row[f"seed{seed}_emotion_f1"] = item["f1"]
        fold_metrics.append(row)
    return fold_metrics


def load_fold_round_mean_validation_emotion_f1(
    experiments,
    start_fold,
    end_fold,
    rounds,
):
    """平均各折三個 seed 在指定自訓練輪次內的驗證 Emotion F1。"""

    fold_metrics = []
    for fold in range(start_fold, end_fold + 1):
        all_f1_values = []
        row = {"fold": fold}
        for experiment in experiments:
            seed = experiment["seed"]
            metrics_path = (
                experiment["experiment_dir"]
                / f"fold{fold}_self_training_round_metrics.csv"
            )
            with metrics_path.open(encoding="utf-8") as file:
                round_rows = [
                    item
                    for item in csv.DictReader(file)
                    if 1 <= int(item["self_training_round"]) <= rounds
                ]
            assert len(round_rows) == rounds, (
                f"Fold {fold} seed {seed} 的驗證紀錄輪數不是 {rounds}: "
                f"{metrics_path}"
            )
            seed_f1_values = [float(item["val_f1_emotion"]) for item in round_rows]
            all_f1_values.extend(seed_f1_values)
            row[f"seed{seed}_round_mean_emotion_f1"] = mean(seed_f1_values)

        row["emotion_f1"] = mean(all_f1_values)
        fold_metrics.append(row)
    return fold_metrics


def load_fold_round_pooled_validation_emotion_f1(
    experiments,
    start_fold,
    end_fold,
    rounds,
):
    """合併每折所有 seed 與指定輪次的驗證計數，計算 pooled micro F1。"""

    fold_metrics = []
    count_pattern = re.compile(
        r"Emotion:\s*預測正確的數量:(\d+(?:\.\d+)?)\s+"
        r"預測出情緒的數量:(\d+(?:\.\d+)?)\s+"
        r"實際正確情緒的數量:(\d+(?:\.\d+)?)"
    )
    for fold in range(start_fold, end_fold + 1):
        true_positive = predicted_positive = gold_positive = 0
        for experiment in experiments:
            result_dirs = sorted(
                experiment["experiment_dir"].glob("self_training_results_*")
            )
            assert len(result_dirs) == 1, (
                f"self_training_results 目錄數量不是 1: {experiment['experiment_dir']}"
            )
            for round_index in range(1, rounds + 1):
                result_path = result_dirs[0] / (
                    f"self_training_val_results_fold{fold}_round{round_index}.txt"
                )
                match = count_pattern.search(result_path.read_text(encoding="utf-8"))
                assert match, f"找不到驗證 Emotion 評估計數: {result_path}"
                tp, predicted, gold = (int(float(value)) for value in match.groups())
                true_positive += tp
                predicted_positive += predicted
                gold_positive += gold

        precision = true_positive / predicted_positive
        recall = true_positive / gold_positive
        emotion_f1 = 2 * precision * recall / (precision + recall)
        fold_metrics.append(
            {
                "fold": fold,
                "emotion_true_positive": true_positive,
                "emotion_predicted_positive": predicted_positive,
                "emotion_gold_positive": gold_positive,
                "emotion_precision": precision,
                "emotion_recall": recall,
                "emotion_f1": emotion_f1,
            }
        )
    return fold_metrics


def load_fold_emotion_metrics(
    experiments,
    start_fold,
    end_fold,
    split,
    validation_f1_mode,
    rounds,
):
    """依參數讀取測試結果、最佳驗證結果或逐輪平均驗證結果。"""

    if split == "validation":
        if validation_f1_mode == "round_mean":
            return load_fold_round_mean_validation_emotion_f1(
                experiments,
                start_fold,
                end_fold,
                rounds,
            )
        return load_fold_best_validation_emotion_f1(experiments, start_fold, end_fold)
    return load_fold_test_emotion_metrics(experiments, start_fold, end_fold)


def plot_selected_counts_with_emotion_f1(
    count_rows,
    metric_rows,
    output_path,
    background,
    split,
    validation_f1_mode="best",
    rounds=5,
    title_suffix="",
    count_ylim_max=None,
    split_label_override=None,
    figure_title_override=None,
):
    """上圖顯示各折 S₁ 至 S₅ 數量，下圖顯示同折 Emotion F1。"""

    assert [row["fold"] for row in count_rows] == [row["fold"] for row in metric_rows]
    configure_fonts(font_size=16)
    plt.rcParams["font.family"] = [*plt.rcParams["font.family"], "DejaVu Serif"]
    positions = list(range(len(count_rows)))
    facecolor = GRAY_BACKGROUND if background == "gray" else "white"
    fig, (count_ax, f1_ax) = plt.subplots(
        2,
        1,
        figsize=(12, 9),
        dpi=180,
        sharex=True,
        gridspec_kw={"height_ratios": (2, 1)},
    )
    fig.patch.set_facecolor(facecolor)
    count_ax.set_facecolor(facecolor)
    f1_ax.set_facecolor(facecolor)

    series = (
        ("S₁", lambda row: row["s1_n"]),
        ("S₂", lambda row: row["s2_n"]),
        ("S₃", lambda row: row["s3_n"]),
        ("S₄=S₁+S₂", lambda row: row["s1_n"] + row["s2_n"]),
        ("S₅=總數", lambda row: row["s0_n"]),
    )
    width = 0.16
    all_counts = []
    center_offset = (len(series) - 1) / 2
    for index, (label, value_from_row) in enumerate(series):
        counts = [value_from_row(row) for row in count_rows]
        bars = count_ax.bar(
            [position + (index - center_offset) * width for position in positions],
            counts,
            width,
            label=label,
            color=BAR_COLORS[index],
            edgecolor="white",
            linewidth=1.2,
            zorder=2,
        )
        count_ax.bar_label(
            bars,
            labels=[format_count(value) for value in counts],
            padding=3,
            fontsize=9,
        )
        all_counts.extend(counts)

    if split == "validation" and validation_f1_mode == "best":
        count_ylabel = "平均樣本數量（筆）"
    elif split == "validation":
        count_ylabel = "累積樣本數量（筆）"
    else:
        count_ylabel = "樣本數量（筆）"
    count_ax.set_ylabel(count_ylabel)
    count_ax.set_ylim(
        0,
        count_ylim_max
        if count_ylim_max is not None
        else max(all_counts) * 1.24,
    )
    count_ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=5, frameon=True)

    f1_values = [100 * row["emotion_f1"] for row in metric_rows]
    f1_ax.plot(
        positions,
        f1_values,
        color=BAR_COLORS[0],
        marker="o",
        linewidth=2.2,
        markersize=7,
        zorder=3,
    )
    for position, value in zip(positions, f1_values):
        f1_ax.annotate(f"{value:.2f}", (position, value), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=11)
    f1_ax.set_ylim(0, 100)
    if split_label_override is not None:
        split_label = split_label_override
    elif split == "validation" and validation_f1_mode == "best":
        split_label = "自訓練最佳驗證集"
    elif split == "validation":
        split_label = f"第 1 至第 {rounds} 輪平均驗證集"
    else:
        split_label = "測試集"
    f1_ax.set_ylabel(f"{split_label} Emotion F1（%）")
    f1_ax.set_xlabel("Fold")
    f1_ax.set_xticks(positions)
    f1_ax.set_xticklabels([str(row["fold"]) for row in count_rows])

    grid_alpha = 0.95 if background == "gray" else 0.35
    grid_width = 1.1 if background == "gray" else 1.0
    for axis in (count_ax, f1_ax):
        axis.grid(
            True,
            axis="y",
            linestyle="--",
            linewidth=grid_width,
            alpha=grid_alpha,
            color="black",
            zorder=10,
        )
        axis.set_axisbelow(False)

    if figure_title_override is not None:
        figure_title = figure_title_override
    elif split == "validation" and validation_f1_mode == "best":
        figure_title = (
            "最佳驗證 Emotion F1 輪次的 S₁ 至 S₅ 平均樣本數量與 Emotion F1"
        )
    elif split == "validation":
        figure_title = (
            f"S₁ 至 S₅ 累積樣本數量與第 1 至第 {rounds} 輪"
            "平均驗證集 Emotion F1"
        )
    else:
        figure_title = "S₁ 至 S₅ 樣本數量與測試集 Emotion F1"
    fig.suptitle(f"{figure_title}{title_suffix}", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97), h_pad=2.0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def count_selected_s0_groups(rows):
    """計算指定分析單位內被 NeST 選中的 S₀、S₁、S₂、S₃ 數量。"""

    # S₀ 只保留被 NeST 選中，且 query 與全部 KNN 都具可比較情緒類別的文檔。
    s0_rows = [row for row in rows if row["nest_selected"] and row["category_valid"]]
    # S₁、S₂ 的情緒子句預測正確；差別在於 KNN 情緒類別是否全部相同。
    s1_n = sum(
        row["query_emotion_exact_s0"] and row["category_mismatch_count"] == 0
        for row in s0_rows
    )
    s2_n = sum(
        row["query_emotion_exact_s0"] and row["category_mismatch_count"] > 0
        for row in s0_rows
    )
    # S₃ 收納 S₀ 中情緒子句預測與真實答案不一致的文檔。
    s3_n = sum(not row["query_emotion_exact_s0"] for row in s0_rows)
    assert len(s0_rows) == s1_n + s2_n + s3_n
    return {"s0_n": len(s0_rows), "s1_n": s1_n, "s2_n": s2_n, "s3_n": s3_n}


def count_training_s0_groups(rows):
    """統計實際納入自訓練且可比較情緒類別的 S₀、S₁、S₂、S₃。"""

    admitted_rows = [row for row in rows if row["admitted_for_training"]]
    # S₀ 要求 query 與全部 KNN 都是單一情緒原因組合且具有有效情緒類別。
    s0_rows = [row for row in admitted_rows if row["category_valid"]]
    s1_n = sum(
        row["query_emotion_exact_s0"] and row["category_mismatch_count"] == 0
        for row in s0_rows
    )
    s2_n = sum(
        row["query_emotion_exact_s0"] and row["category_mismatch_count"] > 0
        for row in s0_rows
    )
    s3_n = sum(not row["query_emotion_exact_s0"] for row in s0_rows)
    assert len(s0_rows) == s1_n + s2_n + s3_n
    return {
        "candidate_n": len(rows),
        "nest_selected_n": sum(row["nest_selected"] for row in rows),
        "admitted_n": len(admitted_rows),
        "excluded_admitted_n": len(admitted_rows) - len(s0_rows),
        "s0_n": len(s0_rows),
        "s1_n": s1_n,
        "s2_n": s2_n,
        "s3_n": s3_n,
        "emotion_exact_ratio": ratio(s1_n + s2_n, len(s0_rows)),
        "category_consistency_ratio": ratio(s1_n, s1_n + s2_n),
    }


def build_round_quality_f1_change_rows(
    document_rows,
    experiments,
    start_fold,
    end_fold,
    rounds,
):
    """將 R2–R5 的訓練樣本組成配對到真正父 checkpoint 至本輪的 F1 變化。"""

    rows_by_unit = defaultdict(list)
    for row in document_rows:
        rows_by_unit[(row["seed"], row["fold"], row["round"])].append(row)

    result = []
    for experiment in experiments:
        seed = experiment["seed"]
        for fold in range(start_fold, end_fold + 1):
            metrics_path = (
                experiment["experiment_dir"]
                / f"fold{fold}_self_training_round_metrics.csv"
            )
            with metrics_path.open(encoding="utf-8") as file:
                metrics = sorted(
                    csv.DictReader(file),
                    key=lambda row: int(row["self_training_round"]),
                )
            assert [int(row["self_training_round"]) for row in metrics] == list(
                range(1, rounds + 1)
            )

            # self-training 的 Pair-F1 global best 從 Round 1 開始建立，strict > 保留較早同分輪。
            best_pair_row = None
            for metric in metrics:
                round_index = int(metric["self_training_round"])
                current_pair_f1 = float(metric["val_f1_pair"])
                current_emotion_f1 = float(metric["val_f1_emotion"])

                # Round 1 的實際父模型是 supervised Pair-best；其精確 Emotion F1 不在此 CSV。
                if round_index >= 2:
                    assert best_pair_row is not None
                    parent_round = int(best_pair_row["self_training_round"])
                    parent_pair_f1 = float(best_pair_row["val_f1_pair"])
                    parent_emotion_f1 = float(best_pair_row["val_f1_emotion"])
                    counts = count_training_s0_groups(
                        rows_by_unit[(seed, fold, round_index)]
                    )
                    result.append(
                        {
                            "seed": seed,
                            "fold": fold,
                            "round": round_index,
                            "parent_type": "prior_self_training_global_best_pair",
                            "parent_round": parent_round,
                            **counts,
                            "parent_val_f1_pair": parent_pair_f1,
                            "current_val_f1_pair": current_pair_f1,
                            "parent_val_f1_emotion": parent_emotion_f1,
                            "current_val_f1_emotion": current_emotion_f1,
                            "delta_val_f1_emotion": current_emotion_f1
                            - parent_emotion_f1,
                            "delta_val_f1_emotion_percentage_points": 100
                            * (current_emotion_f1 - parent_emotion_f1),
                        }
                    )

                if (
                    best_pair_row is None
                    or current_pair_f1 > float(best_pair_row["val_f1_pair"])
                ):
                    best_pair_row = metric

    expected_n = len(experiments) * (end_fold - start_fold + 1) * (rounds - 1)
    assert len(result) == expected_n
    return result


def average_ranks(values):
    """以相同值共享平均名次的方式建立 Spearman 所需 ranks。"""

    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2 + 1
        start = end
    return ranks


def spearman_rho(x_values, y_values):
    """不依賴 SciPy，直接計算含 ties 的 Spearman 等級相關係數。"""

    x_ranks = average_ranks(x_values)
    y_ranks = average_ranks(y_values)
    if np.std(x_ranks) == 0 or np.std(y_ranks) == 0:
        return None
    return float(np.corrcoef(x_ranks, y_ranks)[0, 1])


def fold_cluster_bootstrap_spearman(
    rows,
    x_field,
    y_field,
    repeats=10000,
    random_seed=42,
):
    """以 fold 為抽樣區塊估計 Spearman rho 的 95% bootstrap 信賴區間。"""

    valid_rows = [
        row for row in rows if row[x_field] is not None and row[y_field] is not None
    ]
    fold_ids = sorted({row["fold"] for row in valid_rows})
    rows_by_fold = {
        fold: [row for row in valid_rows if row["fold"] == fold] for fold in fold_ids
    }
    observed = spearman_rho(
        [row[x_field] for row in valid_rows],
        [row[y_field] for row in valid_rows],
    )

    generator = np.random.default_rng(random_seed)
    bootstrap_rhos = []
    for _ in range(repeats):
        sampled_rows = []
        for fold in generator.choice(fold_ids, size=len(fold_ids), replace=True):
            sampled_rows.extend(rows_by_fold[int(fold)])
        rho = spearman_rho(
            [row[x_field] for row in sampled_rows],
            [row[y_field] for row in sampled_rows],
        )
        if rho is not None:
            bootstrap_rhos.append(rho)

    ci_low, ci_high = np.quantile(bootstrap_rhos, [0.025, 0.975])
    return {
        "metric": x_field,
        "n": len(valid_rows),
        "cluster_unit": "fold",
        "cluster_n": len(fold_ids),
        "spearman_rho": observed,
        "bootstrap_repeats": repeats,
        "bootstrap_valid_n": len(bootstrap_rhos),
        "ci_level": 0.95,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "random_seed": random_seed,
    }


def plot_round_ratio_vs_f1_change(
    rows,
    x_field,
    x_label,
    title,
    output_path,
    background,
    statistics,
    color,
):
    """繪製逐輪比例與父模型至本輪驗證 Emotion F1 變化的散點圖。"""

    valid_rows = [row for row in rows if row[x_field] is not None]
    x_values = [100 * row[x_field] for row in valid_rows]
    y_values = [row["delta_val_f1_emotion_percentage_points"] for row in valid_rows]
    configure_fonts(font_size=16)
    plt.rcParams["font.family"] = [*plt.rcParams["font.family"], "DejaVu Serif"]
    facecolor = GRAY_BACKGROUND if background == "gray" else "white"
    fig, ax = plt.subplots(figsize=(8.5, 6), dpi=180)
    fig.patch.set_facecolor(facecolor)
    ax.set_facecolor(facecolor)
    ax.scatter(
        x_values,
        y_values,
        s=62,
        alpha=0.68,
        color=color,
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
    )
    ax.axhline(0, color="black", linestyle="--", linewidth=1.1, zorder=2)
    ax.set_title(title, pad=14)
    ax.set_xlabel(x_label)
    ax.set_ylabel("驗證集 Emotion F1 變化（百分點）")
    ax.grid(True, axis="y", linestyle="--", color="black", alpha=0.28, zorder=1)
    ax.text(
        0.03,
        0.97,
        (
            f"Spearman ρ = {statistics['spearman_rho']:.3f}\n"
            f"fold-cluster bootstrap 95% CI = "
            f"[{statistics['ci_low']:.3f}, {statistics['ci_high']:.3f}]\n"
            f"n = {statistics['n']}"
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        bbox={"facecolor": facecolor, "edgecolor": "#777777", "alpha": 0.9},
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def aggregate_selected_s0_group_counts(
    document_rows,
    start_fold,
    end_fold,
    seed=None,
):
    """彙總每折被選中的 S₁、S₂、S₃；指定 seed 時只統計該次實驗。"""

    return [
        {
            "fold": fold,
            **count_selected_s0_groups(
                [
                    row
                    for row in document_rows
                    if row["fold"] == fold
                    and (seed is None or row["seed"] == seed)
                ]
            ),
        }
        for fold in range(start_fold, end_fold + 1)
    ]


def select_seed_round_mean_emotion_f1(metric_rows, seed):
    """從逐折彙整結果取出單一 seed 的五輪平均驗證 Emotion F1。"""

    field = f"seed{seed}_round_mean_emotion_f1"
    return [
        {"fold": row["fold"], "emotion_f1": row[field]}
        for row in metric_rows
    ]


def aggregate_best_validation_round_group_counts(
    document_rows,
    fold_metric_rows,
    experiments,
):
    """每個 seed × fold 只取 Emotion F1 最佳輪，再依 fold 平均三個 seed 的數量。"""

    detail_rows = []
    fold_counts = []
    for metric_row in fold_metric_rows:
        fold = metric_row["fold"]
        seed_counts = []
        for experiment in experiments:
            seed = experiment["seed"]
            best_round = int(metric_row[f"seed{seed}_best_round"])
            counts = count_selected_s0_groups(
                [
                    row
                    for row in document_rows
                    if row["seed"] == seed
                    and row["fold"] == fold
                    and row["round"] == best_round
                ]
            )
            seed_row = {
                "seed": seed,
                "fold": fold,
                "best_round": best_round,
                "best_validation_emotion_f1": metric_row[f"seed{seed}_emotion_f1"],
                **counts,
            }
            detail_rows.append(seed_row)
            seed_counts.append(seed_row)

        averaged = {
            field: mean([row[field] for row in seed_counts])
            for field in ("s0_n", "s1_n", "s2_n", "s3_n")
        }
        assert math.isclose(
            averaged["s0_n"],
            averaged["s1_n"] + averaged["s2_n"] + averaged["s3_n"],
        )
        fold_counts.append({"fold": fold, **averaged})
    return fold_counts, detail_rows


def normalize_category(category):
    # None、空字串與字面上的 null 都統一表示為無有效情緒類別。
    value = "null" if category is None else str(category).strip().lower()
    if not value or value == "null":
        return "null"
    # 複合類別依 & 拆開並排序，例如 disgust&anger 與 anger&disgust 視為相同。
    return "&".join(sorted(part.strip() for part in value.split("&")))


def single_pair_info(document):
    # 老師的情緒類別分析只處理恰好一組 emotion-cause pair 的文檔。
    if len(document["pairs"]) != 1:
        return None
    # pair 格式為 [emotion_clause_id, cause_clause_id]，索引 0 是情緒子句編號。
    emotion_clause = int(document["pairs"][0][0])
    # 依情緒子句編號找到該子句的真實 emotion_category。
    clause = next(
        clause
        for clause in document["clauses"]
        if int(clause["clause_id"]) == emotion_clause
    )
    # 回傳情緒子句編號及正規化後的情緒類別；多組 pair 則已在前面回傳 None。
    return emotion_clause, normalize_category(clause["emotion_category"])


def index_documents(path):
    # 所有跨檔案對齊都以 doc_id 為準，不能使用原始 JSON 的列位置。
    documents = load_json(path)
    indexed = {str(document["doc_id"]): document for document in documents}
    # dict 長度變短代表原始資料中存在重複 doc_id，後續 join 將失去唯一性。
    assert len(indexed) == len(documents), f"doc_id 重複: {path}"
    return indexed


def parse_experiments(summary_path, experiments_root):
    # 只擷取以「- prompt_ECPE_few_shot_ST_」開頭的實驗目錄名稱。
    pattern = re.compile(r"^\s*-\s*(prompt_ECPE_few_shot_ST_\S+)\s*$", re.MULTILINE)
    names = pattern.findall(Path(summary_path).read_text(encoding="utf-8"))
    experiments = []
    for name in names:
        # 從目錄名稱的 _seed20_、_seed42_、_seed60_ 片段取得 seed。
        seed = int(re.search(r"_seed(\d+)_", name).group(1))
        experiment_dir = Path(experiments_root) / name
        # 不用實驗時間硬組 pseudo_results 名稱，因 seed60 的時間實際晚了一秒。
        pseudo_dirs = sorted(experiment_dir.glob("pseudo_results_*"))
        assert len(pseudo_dirs) == 1, f"pseudo_results 目錄數量不是 1: {experiment_dir}"
        experiments.append(
            {
                "seed": seed,
                "name": name,
                "experiment_dir": experiment_dir,
                "pseudo_dir": pseudo_dirs[0],
            }
        )
    # 本次 summary 應恰好包含三次獨立 seed 實驗，且 seed 不可重複。
    assert len(experiments) == 3, "summary 應包含三個 seed 實驗目錄"
    assert len({item["seed"] for item in experiments}) == 3, "summary 的 seed 重複"
    # 固定依 seed 數值排序，使輸出順序可重現。
    return sorted(experiments, key=lambda item: item["seed"])


def emotion_clause_ids(tokens):
    # emotion MASK 每個 token 對應一個子句；值為「是」的 1-based 位置就是預測情緒子句。
    # 此函式只產生方便人工閱讀的欄位；真正 S0 仍比較完整 emotion token tuple。
    return "|".join(str(index) for index, token in enumerate(tokens, start=1) if token == "是")


def group_statistics(rows):
    # 此函式一次只接收同一 seed-fold-round 內的 S1 或 S2 文檔。
    raw_values = [row["raw_divergence"] for row in rows]
    return {
        "n": len(rows),
        # raw_sum 供後續 observation-pooled 敏感度分析；raw_mean 才是逐單位 macro 的來源。
        "raw_sum": math.fsum(raw_values),
        "raw_mean": mean(raw_values),
        # D_u、D_l 分開保留，方便追查 raw divergence 的兩個組成部分。
        "D_u_mean": mean([row["D_u"] for row in rows]),
        "D_l_mean": mean([row["D_l"] for row in rows]),
        # ema_score 是實際 NeST 選樣使用的跨輪平滑分數，不是本輪原始散度。
        "ema_mean": mean([row["ema_score"] for row in rows]),
        # nest_selected 表示被 NeST 抽中，不必然等同最後實際加入自訓練資料。
        "nest_selected_n": sum(row["nest_selected"] for row in rows),
        "nest_selection_rate": ratio(sum(row["nest_selected"] for row in rows), len(rows)),
        # admitted_for_training 才表示該文檔實際進入本輪 pseudo-labeled 訓練集；
        # 但後續 token-level loss threshold 仍可能只讓部分偽標籤 token 產生梯度。
        "admitted_n": sum(row["admitted_for_training"] for row in rows),
        "admission_rate": ratio(sum(row["admitted_for_training"] for row in rows), len(rows)),
        # N×p 將不同候選池大小的初始抽樣權重換到「均勻抽樣平均值為 1」的尺度。
        "relative_weight_mean": mean([row["relative_sampling_weight"] for row in rows]),
    }


def recompute_prefix_divergence(sample, analysis_k, beta):
    """以已保存且由近至遠排列的前 k 位鄰居重新計算原始 NeST 散度。"""

    neighbors = np.asarray(sample["D_u_numerator"][:analysis_k], dtype=np.float32)
    query_distribution = np.asarray(sample["D_u_denominator"], dtype=np.float32)
    neighbor_mean = neighbors.mean(axis=0)
    score_u = (
        np.log((1e-10 + neighbors) / (1e-10 + query_distribution[None, :]))
        * (1e-10 + neighbors)
    )
    score_l = (
        np.log((1e-10 + neighbor_mean[None, :]) / (1e-10 + neighbors))
        * (1e-10 + neighbor_mean[None, :])
    )
    D_u = float(np.sum(score_u))
    D_l = float(np.sum(score_l))
    return D_u, D_l, D_u + beta * D_l


def analyze_unit(
    seed,
    fold,
    round_index,
    pseudo_dir,
    train_documents,
    unlabeled_documents,
    analysis_k=None,
):
    # 每個分析單位固定由「一個 seed × 一個 fold × 一個 self-training round」構成。
    # divergence JSON 保存當輪 KNN、散度、EMA 與 NeST 抽樣結果。
    divergence_path = pseudo_dir / f"nest_divergence_scores_fold{fold}_round{round_index}.json"
    # prediction JSON 保存當輪所有候選未標註文檔的偽標籤與真值對照。
    prediction_path = pseudo_dir / (
        f"all_unlabeled_pseudo_predictions_fold{fold}_round{round_index}.json"
    )
    # pseudo_labeled JSON 保存最後實際納入本輪自訓練的文檔。
    admitted_path = pseudo_dir / f"pseudo_labeled_samples_fold{fold}_round{round_index}.json"

    # 三份逐輪檔案必須一起讀取，才能區分「候選、抽中、實際納入訓練」。
    divergence = load_json(divergence_path)
    predictions = load_json(prediction_path)
    admitted_records = load_json(admitted_path)
    samples = divergence["samples"]
    # prediction JSON 與 divergence JSON 即使列順序不同，也能透過 doc_id 正確 join。
    prediction_by_id = {str(record["doc_id"]): record for record in predictions}
    sample_ids = [str(sample["doc_id"]) for sample in samples]
    admitted_ids = {str(record["doc_id"]) for record in admitted_records}

    # 以下 assert 是輸入資料的不變量檢查；任何一項失敗都停止分析，不會默默略過資料。
    assert int(divergence["fold"]) == fold
    assert int(divergence["round"]) == round_index
    # 本次研究只分析 emotion_clause 模式產生的 KNN 與散度。
    assert divergence["divergence_mode"] == "emotion_clause"
    # divergence 與 prediction 必須各自沒有重複 doc_id，且兩邊候選集合完全相同。
    assert len(sample_ids) == len(set(sample_ids)) == len(prediction_by_id)
    assert set(sample_ids) == set(prediction_by_id)
    assert int(divergence["statistics"]["total_unlabeled"]) == len(samples)

    # k、beta、m 直接讀取實驗紀錄，避免在分析程式中另外硬編一套參數。
    source_k = int(divergence["params"]["k"])
    effective_k = source_k if analysis_k is None else analysis_k
    assert 1 <= effective_k <= source_k
    beta = float(divergence["params"]["beta"])
    m = float(divergence["params"]["m"])
    # labeled_doc_ids 是經模型長度過濾後真正可供 KNN 搜尋的原始 fold-train anchor 集合；
    # 它不包含歷輪累積 pseudo dataset，也不一定等於原始 train JSON 的全部文檔數。
    labeled_ids = {str(doc_id) for doc_id in divergence["labeled_doc_ids"]}
    # selected=True 表示 NeST 在加權無放回抽樣階段抽到該文檔。
    nest_selected_ids = {
        str(sample["doc_id"]) for sample in samples if bool(sample["selected"])
    }
    # selected_for_training 表示後續流程結束後，該文檔確實進入 pseudo-labeled 訓練集。
    prediction_admitted_ids = {
        str(record["doc_id"])
        for record in predictions
        if bool(record["selected_for_training"])
    }
    # prediction 內的納入標記必須與 pseudo_labeled_samples 實體清單完全相同。
    assert admitted_ids == prediction_admitted_ids
    # 實際納入訓練者必須先被 NeST 抽中；若有後處理，納入集合可能只是抽中集合的子集。
    assert admitted_ids <= nest_selected_ids
    assert int(divergence["statistics"]["num_selected"]) == len(nest_selected_ids)

    # rows 保存本分析單位中每一篇候選 query 的完整 gate 與散度明細。
    rows = []
    # formula_errors 用來記錄 JSON raw divergence 與重算公式間的最大浮點誤差。
    formula_errors = []
    for sample in samples:
        # 所有 JSON 的 doc_id 先轉成字串，避免某些檔案使用整數、某些檔案使用字串。
        doc_id = str(sample["doc_id"])
        prediction = prediction_by_id[doc_id]
        # query 的 GT pair 與 emotion_category 由原始 fold unlabeled split 取得。
        query_document = unlabeled_documents[doc_id]
        # 直接使用 divergence JSON 保存的 neighbor_doc_ids，不用 neighbor_indices 索引原始 train。
        # 原因是超過模型長度的 train 文檔會被過濾，原始 JSON 列位置可能已經錯位。
        source_neighbor_ids = [str(value) for value in sample["neighbor_doc_ids"]]
        source_neighbor_distances = [float(value) for value in sample["neighbor_l2_distances"]]
        # 先驗證來源實驗保存的完整鄰居，再取由近至遠的前 effective_k 位進行事後分析。
        assert len(source_neighbor_ids) == len(set(source_neighbor_ids)) == source_k
        assert len(source_neighbor_distances) == source_k
        assert all(
            left <= right + 1e-6
            for left, right in zip(source_neighbor_distances, source_neighbor_distances[1:])
        )
        assert set(source_neighbor_ids) <= labeled_ids
        assert all(doc_id in train_documents for doc_id in source_neighbor_ids)
        neighbor_ids = source_neighbor_ids[:effective_k]
        neighbor_documents = [train_documents[neighbor_id] for neighbor_id in neighbor_ids]

        # prompt 的每個子句依序有 emotion／cause／pair 三個 MASK，因此長度為 3×子句數。
        pseudo_tokens = prediction["pseudo_label_tokens"]
        ground_truth_tokens = prediction["gt_label_tokens"]
        assert prediction["ground_truth_available"]
        assert len(pseudo_tokens) == len(ground_truth_tokens) == 3 * len(query_document["clauses"])
        # 從 e/c/p 序列擷取索引 0、3、6……，只保留所有 emotion MASK 的預測與真值。
        pseudo_emotion_tokens = tuple(pseudo_tokens[0::3])
        ground_truth_emotion_tokens = tuple(ground_truth_tokens[0::3])
        # 老師定義的原始 S0：query 的每一個 emotion MASK 都必須逐位置完全相同。
        # 這會同時檢查漏掉、多預測及預測出「是／非」以外 token 的情況。
        # 此 gate 不要求 cause／pair MASK 正確，也不要求五個 KNN 的模型情緒預測正確。
        # 現有逐輪 JSON 沒保存 labeled anchors 的逐子句預測，因此本分析只採老師目前的 query-only S0。
        query_emotion_exact = pseudo_emotion_tokens == ground_truth_emotion_tokens

        # single_pair_info 回傳 (情緒子句編號, 正規化 category)，多組 pair 則回傳 None。
        query_info = single_pair_info(query_document)
        neighbor_info = [single_pair_info(document) for document in neighbor_documents]
        query_single_pair = query_info is not None
        neighbors_all_single_pair = all(info is not None for info in neighbor_info)
        query_category = query_info[1] if query_info else ""
        # 不符合 single-pair 的鄰居仍以文字標記寫入明細，方便追查被排除原因。
        neighbor_categories = [info[1] if info else "not_single_pair" for info in neighbor_info]
        # category_valid 要求 query 與全部 k 個鄰居均為 single-pair 且類別不是 null。
        category_valid = (
            query_single_pair
            and neighbors_all_single_pair
            and query_category != "null"
            and all(category != "null" for category in neighbor_categories)
        )
        # analysis_s0 是真正能進入情緒類別 S1/S2 比較的 S0 子集合。
        # 它比原始 S0 多了 query/KNN single-pair 及有效 category 的限制。
        analysis_s0 = query_emotion_exact and category_valid
        # mismatch_count 計算 k 個鄰居中，有幾個 category 與 query 不同。
        mismatch_count = (
            sum(category != query_category for category in neighbor_categories)
            if category_valid
            else None
        )
        # S1/S2 使用原始 split 的真實 emotion_category 作事後分組；GT 類別沒有參與 NeST 選樣。
        # S1：全部鄰居相同；S2：至少一個不同；未通過 analysis_s0 者不進任一組。
        group = "S1" if analysis_s0 and mismatch_count == 0 else "S2" if analysis_s0 else "excluded"

        # 依 gate 的實際先後順序記錄第一個排除原因，讓 S0 漏斗可以被逐筆稽核。
        if not query_emotion_exact:
            exclusion_reason = "query_emotion_not_exact"
        elif not query_single_pair:
            exclusion_reason = "query_not_single_pair"
        elif not neighbors_all_single_pair:
            exclusion_reason = "neighbor_not_single_pair"
        elif not category_valid:
            exclusion_reason = "invalid_category"
        else:
            exclusion_reason = ""

        # raw divergence 是老師假設的主要分析值：本輪 D_u + beta×D_l，不含 EMA。
        if effective_k == source_k:
            D_u = float(sample["D_u"])
            D_l = float(sample["D_l"])
            raw_divergence = float(sample["raw_divergence"])
        else:
            D_u, D_l, raw_divergence = recompute_prefix_divergence(
                sample,
                effective_k,
                beta,
            )
        if effective_k == 1:
            assert abs(D_l) <= 1e-6
        # 重新計算公式並容許合理的 float32/float64 誤差。
        formula_error = abs(raw_divergence - (D_u + beta * D_l))
        formula_errors.append(formula_error)
        assert math.isclose(raw_divergence, D_u + beta * D_l, rel_tol=1e-5, abs_tol=1e-5)
        # 無限大與 NaN 都不能進入平均數與統計檢定。
        assert all(math.isfinite(value) for value in (D_u, D_l, raw_divergence))

        # probability 是無放回抽樣前的初始正規化權重，不是最終邊際納入機率。
        probability = float(sample["probability"])
        rows.append(
            {
                # 識別欄位：同一 doc_id 可跨 seed 或在尚未被選走前跨 round 重複出現。
                "seed": seed,
                "fold": fold,
                "round": round_index,
                "source_k": source_k,
                "analysis_k": effective_k,
                "query_doc_id": doc_id,
                # 各 gate 的布林結果保留於逐文檔表，便於重算 S0/S1/S2。
                "query_single_pair": query_single_pair,
                "neighbors_all_single_pair": neighbors_all_single_pair,
                "query_emotion_exact_s0": query_emotion_exact,
                "category_valid": category_valid,
                "analysis_s0": analysis_s0,
                "group": group,
                "exclusion_reason": exclusion_reason,
                # query 的單一 GT 情緒子句與 category；非 single-pair 時留白。
                "query_emotion_clause": query_info[0] if query_info else "",
                "query_category": query_category,
                # 以「|」串接 1-based 子句編號，讓人可直接查看預測與 GT 情緒子句集合。
                "predicted_emotion_clauses": emotion_clause_ids(pseudo_emotion_tokens),
                "ground_truth_emotion_clauses": emotion_clause_ids(ground_truth_emotion_tokens),
                # 記錄 emotion MASK 中出現多少個不是「是／非」的非法二元分類 token。
                "invalid_predicted_emotion_token_n": sum(
                    token not in {"是", "非"} for token in pseudo_emotion_tokens
                ),
                # k 個鄰居資訊以「|」串接，順序與當輪 KNN 排名一致。
                "source_neighbor_doc_ids": "|".join(source_neighbor_ids),
                "neighbor_doc_ids": "|".join(neighbor_ids),
                "neighbor_emotion_clauses": "|".join(
                    str(info[0]) if info else "" for info in neighbor_info
                ),
                "neighbor_categories": "|".join(neighbor_categories),
                "category_mismatch_count": mismatch_count,
                # 主要散度及其兩個組成部分。
                "D_u": D_u,
                "D_l": D_l,
                "raw_divergence": raw_divergence,
                # divergence_score 是含跨輪 EMA 的實際選樣分數，只作輔助分析。
                "ema_score": float(sample["divergence_score"]),
                "sampling_probability": probability,
                "selection_source_k": source_k,
                # 候選數 N×p：數值 1 表示與該輪均勻抽樣的平均權重相同。
                "relative_sampling_weight": len(samples) * probability,
                # 分開記錄「被 NeST 抽中」與「最後實際進入訓練」。
                "nest_selected": bool(sample["selected"]),
                "admitted_for_training": doc_id in admitted_ids,
            }
        )

    # 只有 analysis_s0 中的文檔才會出現在 S1 或 S2。
    s1_rows = [row for row in rows if row["group"] == "S1"]
    s2_rows = [row for row in rows if row["group"] == "S2"]
    s1 = group_statistics(s1_rows)
    s2 = group_statistics(s2_rows)
    # A1/A2 配對比較必須同時存在 S1 與 S2；任一組為空就不是有效比較單位。
    comparison_valid = bool(s1["n"] and s2["n"])
    analysis_s0_n = sum(row["analysis_s0"] for row in rows)
    # S1 與 S2 必須互斥且完整涵蓋 analysis_s0。
    assert analysis_s0_n == s1["n"] + s2["n"]

    # unit 是固定一列的 seed-fold-round 摘要；所有 150 個單位都保留。
    unit = {
        "seed": seed,
        "fold": fold,
        "round": round_index,
        # source_valid=True 表示來源檔、doc_id join 與公式 assert 已通過；
        # 它不表示 S1/S2 可比較，後者必須查看 comparison_valid。
        "source_valid": True,
        "source_k": source_k,
        "k": effective_k,
        "beta": beta,
        "m": m,
        # candidate_n 是該輪尚未被前輪移除的模型可用候選數。
        "candidate_n": len(rows),
        # 以下欄位構成逐步縮小的 S0 分析漏斗，便於呈現樣本代表性。
        "query_single_pair_n": sum(row["query_single_pair"] for row in rows),
        "query_exact_s0_n": sum(row["query_emotion_exact_s0"] for row in rows),
        "single_pair_query_exact_n": sum(
            row["query_single_pair"] and row["query_emotion_exact_s0"] for row in rows
        ),
        "neighbors_single_pair_s0_n": sum(
            row["query_single_pair"]
            and row["query_emotion_exact_s0"]
            and row["neighbors_all_single_pair"]
            for row in rows
        ),
        "analysis_s0_n": analysis_s0_n,
        "query_exact_s0_coverage": ratio(
            sum(row["query_emotion_exact_s0"] for row in rows), len(rows)
        ),
        "analysis_s0_coverage": ratio(analysis_s0_n, len(rows)),
        # S1/S2 是可分析 S0 中的兩個互斥組別。
        "s1_n": s1["n"],
        "s2_n": s2["n"],
        "s1_share_in_analysis_s0": ratio(s1["n"], analysis_s0_n),
        "comparison_valid": comparison_valid,
        # small_group 只提醒任一組少於 5 筆，不會因此排除或修改結果。
        "small_group": comparison_valid and min(s1["n"], s2["n"]) < 5,
        # 空組的平均數會是 None；不可填 0，因 0 會被誤解為真實零散度。
        "invalid_reason": "" if comparison_valid else "S1_empty" if not s1["n"] else "S2_empty",
        # raw_sum 供 pooled 敏感度結果；raw_mean 供主要階層 macro 聚合。
        "s1_raw_sum": s1["raw_sum"],
        "s2_raw_sum": s2["raw_sum"],
        "s1_raw_mean": s1["raw_mean"],
        "s2_raw_mean": s2["raw_mean"],
        # 散度差固定定義為 S2-S1；正值表示 S1 平均散度較低。
        "raw_delta_s2_minus_s1": (
            s2["raw_mean"] - s1["raw_mean"] if comparison_valid else None
        ),
        "s1_D_u_mean": s1["D_u_mean"],
        "s2_D_u_mean": s2["D_u_mean"],
        "s1_D_l_mean": s1["D_l_mean"],
        "s2_D_l_mean": s2["D_l_mean"],
        "s1_ema_mean": s1["ema_mean"],
        "s2_ema_mean": s2["ema_mean"],
        # EMA 差只作實際選樣機制的敏感度描述，不取代 raw 主結果。
        "ema_delta_s2_minus_s1": (
            s2["ema_mean"] - s1["ema_mean"] if comparison_valid else None
        ),
        "s1_nest_selected_n": s1["nest_selected_n"],
        "s2_nest_selected_n": s2["nest_selected_n"],
        "s1_nest_selection_rate": s1["nest_selection_rate"],
        "s2_nest_selection_rate": s2["nest_selection_rate"],
        # 選取率差固定定義為 S1-S2；正值表示 S1 實際較常被 NeST 抽中。
        "nest_selection_rate_delta_s1_minus_s2": (
            s1["nest_selection_rate"] - s2["nest_selection_rate"]
            if comparison_valid
            else None
        ),
        "s1_admitted_n": s1["admitted_n"],
        "s2_admitted_n": s2["admitted_n"],
        "s1_admission_rate": s1["admission_rate"],
        "s2_admission_rate": s2["admission_rate"],
        # 納入率同樣採 S1-S2；這才對應「實際參與自訓練」的差異。
        "admission_rate_delta_s1_minus_s2": (
            s1["admission_rate"] - s2["admission_rate"]
            if comparison_valid
            else None
        ),
        "s1_relative_weight_mean": s1["relative_weight_mean"],
        "s2_relative_weight_mean": s2["relative_weight_mean"],
        # 以下欄位只用於輸入品質與選樣流程稽核。
        # 這裡計算「至少出現一個非法 emotion token 的文檔數」，不是非法 token 總數。
        "invalid_emotion_prediction_n": sum(
            row["invalid_predicted_emotion_token_n"] > 0 for row in rows
        ),
        # 由於 admitted 必為 selected 子集合，此值實際代表「抽中但最後未納入」的文檔數。
        "selection_admission_mismatch_n": sum(
            row["nest_selected"] != row["admitted_for_training"] for row in rows
        ),
        "probability_sum": math.fsum(row["sampling_probability"] for row in rows),
        "raw_formula_max_error": max(formula_errors),
    }
    # 回傳候選與實際納入集合，讓 main 驗證下一輪候選池 lineage。
    # config 用來確認三個 seed 的 k、beta、m 完全相同。
    return rows, unit, set(sample_ids), admitted_ids, (source_k, beta, m)


COUNT_FIELDS = [
    # 這些都是可直接加總的「文檔－輪次觀察筆數」或事件筆數。
    # 同一 doc_id 跨 round 或 seed 再次出現時，會再次計數。
    "candidate_n",
    "query_single_pair_n",
    "query_exact_s0_n",
    "single_pair_query_exact_n",
    "neighbors_single_pair_s0_n",
    "analysis_s0_n",
    "s1_n",
    "s2_n",
    "s1_nest_selected_n",
    "s2_nest_selected_n",
    "s1_admitted_n",
    "s2_admitted_n",
    "invalid_emotion_prediction_n",
    "selection_admission_mismatch_n",
]

# 此 mapping 定義從 seed-fold-round 單位表中，取哪些欄位進行等權 macro 平均。
# 只有 S1 與 S2 都非空的 comparison_valid 單位才會進入這些配對指標。
MACRO_METRICS = {
    # A1 對應 S1，A2 對應 S2；主要結果使用 raw divergence。
    "raw_A1_macro": "s1_raw_mean",
    "raw_A2_macro": "s2_raw_mean",
    # EMA 是實際選樣分數的敏感度分析。
    "ema_A1_macro": "s1_ema_mean",
    "ema_A2_macro": "s2_ema_mean",
    # 選取率與納入率分開，避免把抽中直接等同真正參與訓練。
    "s1_nest_selection_rate_macro": "s1_nest_selection_rate",
    "s2_nest_selection_rate_macro": "s2_nest_selection_rate",
    "s1_admission_rate_macro": "s1_admission_rate",
    "s2_admission_rate_macro": "s2_admission_rate",
    "s1_relative_weight_macro": "s1_relative_weight_mean",
    "s2_relative_weight_macro": "s2_relative_weight_mean",
}


def add_deltas(row):
    # 散度差採 A2-A1；正值支持「類別完全一致的 S1 散度較低」。
    row["raw_delta_macro_s2_minus_s1"] = (
        row["raw_A2_macro"] - row["raw_A1_macro"]
        if row["raw_A1_macro"] is not None and row["raw_A2_macro"] is not None
        else None
    )
    # EMA 差沿用 S2-S1 的方向，但只作選樣分數敏感度描述。
    row["ema_delta_macro_s2_minus_s1"] = (
        row["ema_A2_macro"] - row["ema_A1_macro"]
        if row["ema_A1_macro"] is not None and row["ema_A2_macro"] is not None
        else None
    )
    # 選取率與納入率差改採 S1-S2；正值才表示 S1 較常被抽中／納入。
    row["nest_selection_rate_delta_macro_s1_minus_s2"] = (
        row["s1_nest_selection_rate_macro"] - row["s2_nest_selection_rate_macro"]
        if row["s1_nest_selection_rate_macro"] is not None
        and row["s2_nest_selection_rate_macro"] is not None
        else None
    )
    row["admission_rate_delta_macro_s1_minus_s2"] = (
        row["s1_admission_rate_macro"] - row["s2_admission_rate_macro"]
        if row["s1_admission_rate_macro"] is not None
        and row["s2_admission_rate_macro"] is not None
        else None
    )


def add_pooled_statistics(row):
    # pooled 平均讓每一筆 document-round observation 等權，樣本較多的組別／輪次權重較大。
    # 目前 pooled 使用所有有該組資料的單位；即使另一組為空，該單位仍會貢獻單組 sum/count。
    # 因此它是 all-available observation-pooled 敏感度結果，不是主要配對 macro 結果。
    row["raw_A1_observation_pooled"] = ratio(row["s1_raw_sum"], row["s1_n"])
    row["raw_A2_observation_pooled"] = ratio(row["s2_raw_sum"], row["s2_n"])
    row["raw_delta_observation_pooled_s2_minus_s1"] = (
        row["raw_A2_observation_pooled"] - row["raw_A1_observation_pooled"]
        if row["raw_A1_observation_pooled"] is not None
        and row["raw_A2_observation_pooled"] is not None
        else None
    )
    # coverage 回答實際候選觀察值中有多少通過 gate，因此使用全部來源有效單位的 pooled 比例。
    row["query_exact_s0_coverage"] = ratio(row["query_exact_s0_n"], row["candidate_n"])
    row["analysis_s0_coverage"] = ratio(row["analysis_s0_n"], row["candidate_n"])


def add_paired_valid_micro_statistics(row, units):
    # 配對有效單位是同一 seed-fold-round 中 S1 與 S2 都至少有一筆文檔的單位。
    valid_units = [unit for unit in units if unit["comparison_valid"]]
    row["paired_valid_units"] = len(valid_units)

    # 分母：配對有效單位內，S1 與 S2 各自的文檔－輪次觀察值總數。
    row["paired_valid_s1_observation_n"] = sum(
        unit["s1_n"] for unit in valid_units
    )
    row["paired_valid_s2_observation_n"] = sum(
        unit["s2_n"] for unit in valid_units
    )

    # 分子：配對有效單位內，S1 與 S2 各自的原始散度總和。
    row["paired_valid_s1_raw_sum"] = math.fsum(
        unit["s1_raw_sum"] for unit in valid_units
    )
    row["paired_valid_s2_raw_sum"] = math.fsum(
        unit["s2_raw_sum"] for unit in valid_units
    )

    # Micro 讓每筆文檔－輪次觀察值等權，不先平均 round 或 seed。
    row["raw_A1_paired_valid_micro"] = ratio(
        row["paired_valid_s1_raw_sum"],
        row["paired_valid_s1_observation_n"],
    )
    row["raw_A2_paired_valid_micro"] = ratio(
        row["paired_valid_s2_raw_sum"],
        row["paired_valid_s2_observation_n"],
    )
    row["raw_delta_paired_valid_micro_s2_minus_s1"] = (
        row["raw_A2_paired_valid_micro"] - row["raw_A1_paired_valid_micro"]
        if row["raw_A1_paired_valid_micro"] is not None
        and row["raw_A2_paired_valid_micro"] is not None
        else None
    )


def aggregate_seed_fold(seed, fold, units):
    # 同一 seed-fold 的五輪中，只有 S1/S2 都非空的 round 可進行 A1/A2 配對比較。
    valid = [unit for unit in units if unit["comparison_valid"]]
    row = {
        "seed": seed,
        "fold": fold,
        "source_valid_units": len(units),
        "comparison_valid_units": len(valid),
        "valid_rounds": "|".join(str(unit["round"]) for unit in valid),
    }
    # 代表性與事件筆數使用全部五輪直接加總，不因某輪缺少 S1 或 S2 而消失。
    for field in COUNT_FIELDS:
        row[field] = sum(unit[field] for unit in units)
    # pooled sum 同樣保留所有輪次的可用單組觀察值。
    row["s1_raw_sum"] = math.fsum(unit["s1_raw_sum"] for unit in units)
    row["s2_raw_sum"] = math.fsum(unit["s2_raw_sum"] for unit in units)
    # 主要 macro 先在此讓同一 seed-fold 內的有效 rounds 等權。
    for output_field, unit_field in MACRO_METRICS.items():
        row[output_field] = mean([unit[unit_field] for unit in valid])
    add_deltas(row)
    add_pooled_statistics(row)
    add_paired_valid_micro_statistics(row, units)
    return row


def aggregate_fold(fold, seed_fold_rows, units, unique_seed_docs):
    # seed 至少有一個 comparison-valid round，才有可納入該 fold 配對 macro 的 seed-level 結果。
    valid = [row for row in seed_fold_rows if row["comparison_valid_units"] > 0]
    row = {
        "fold": fold,
        "source_valid_units": len(units),
        "comparison_valid_units": sum(item["comparison_valid"] for item in units),
        "valid_seed_n": len(valid),
        # complete_three_seed 只表示三個 seed 各至少有一個有效 round，不代表 15/15 單位全有效。
        "complete_three_seed": len(valid) == 3,
        # 例如 20:1|2|3;42:1|2 可清楚看出各 seed 實際納入 macro 的 rounds。
        "valid_rounds_by_seed": ";".join(
            f"{item['seed']}:{item['valid_rounds']}" for item in seed_fold_rows
        ),
    }
    # Fold 的 counts 是三個 seed、五輪的 document-round observations 總和。
    for field in COUNT_FIELDS:
        row[field] = sum(unit[field] for unit in units)
    row["s1_raw_sum"] = math.fsum(unit["s1_raw_sum"] for unit in units)
    row["s2_raw_sum"] = math.fsum(unit["s2_raw_sum"] for unit in units)
    # unique key 使用 (seed, doc_id)：同一 seed 內跨 round 去重，但同 doc 跨 seed 仍視為不同實驗觀察。
    # 同一文檔若不同 round 曾落入不同組，也可能同時出現在 S1 與 S2 的各自 unique set。
    row["s1_unique_seed_doc_n"] = len(unique_seed_docs[(fold, "S1")])
    row["s2_unique_seed_doc_n"] = len(unique_seed_docs[(fold, "S2")])
    # 第二層 macro：先完成各 seed 內 round 平均，再讓有效 seeds 於 fold 內等權。
    for output_field in MACRO_METRICS:
        row[output_field] = mean([item[output_field] for item in valid])
    add_deltas(row)
    add_pooled_statistics(row)
    add_paired_valid_micro_statistics(row, units)
    return row


def exact_sign_flip_test(differences):
    # differences 是最多十個 fold-level (A2-A1)，統計單位是 fold，不是單篇文檔。
    # 此檢定只評估跨折差值方向／大小，不構成 category 一致性造成散度變化的因果證明。
    observed = mean(differences)
    # 虛無假設下，每折差值的正負號可互換；F 折會完整列舉 2^F 種符號組合。
    signed_means = [
        mean([sign * value for sign, value in zip(signs, differences)])
        for signs in itertools.product((-1, 1), repeat=len(differences))
    ]
    return {
        "fold_n": len(differences),
        "observed_mean_delta": observed,
        # 雙尾 p-value 檢驗平均差是否偏離 0，不預先限定方向。
        "two_sided_p": ratio(
            sum(abs(value) >= abs(observed) - 1e-15 for value in signed_means),
            len(signed_means),
        ),
        # 單尾 p-value 對應預先指定的 A2>A1，也就是 S1 散度較低假設。
        "one_sided_A2_greater_p": ratio(
            sum(value >= observed - 1e-15 for value in signed_means),
            len(signed_means),
        ),
    }


def main():
    # 解析命令列參數，並由 summary 找出三個 seed 的實驗與 pseudo_results 目錄。
    args = parse_args()
    experiments = parse_experiments(args.summary, args.experiments_root)
    # 僅建立分析輸出目錄；不會修改任何來源實驗檔。
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 三個 seed 共用同一組資料切分，先依 fold 載入並建立 doc_id 索引，避免重複讀檔。
    split_cache = {}
    for fold in range(args.start_fold, args.end_fold + 1):
        train_documents = index_documents(args.dataset_dir / f"fold{fold}_train.json")
        unlabeled_documents = index_documents(args.dataset_dir / f"fold{fold}_unlabeled.json")
        # 同一 fold 的原始 train 與 unlabeled 文檔必須互斥。
        assert set(train_documents).isdisjoint(unlabeled_documents)
        split_cache[fold] = train_documents, unlabeled_documents

    # document_rows：每篇候選文檔在每輪的明細；同一文檔可跨 round 重複出現。
    document_rows = []
    # unit_rows：固定一列代表一個 seed-fold-round，完整執行時應有 3×10×5=150 列。
    unit_rows = []
    # 用 (fold, group) 收集唯一 (seed, doc_id)，供 Fold 表的去重描述性計數。
    unique_seed_docs = defaultdict(set)
    # 第一個單位決定共同 (k, beta, m)，其餘 149 個單位都必須與它一致。
    expected_config = None

    # 第一層迴圈依序處理三個獨立 seed 實驗。
    for experiment in experiments:
        seed = experiment["seed"]
        print(f"seed={seed}, experiment={experiment['name']}")
        # 第二層迴圈處理該 seed 的 Fold 1 至 Fold 10。
        for fold in range(args.start_fold, args.end_fold + 1):
            train_documents, unlabeled_documents = split_cache[fold]
            # 保存前一輪候選與實際納入集合，用來驗證 remove_pseudo 的狀態轉移。
            previous_candidates = None
            previous_admitted = None
            fold_units = []
            # 第三層迴圈依時間順序處理 Round 1 至 Round 5。
            for round_index in range(1, args.rounds + 1):
                rows, unit, candidates, admitted, config = analyze_unit(
                    seed,
                    fold,
                    round_index,
                    experiment["pseudo_dir"],
                    train_documents,
                    unlabeled_documents,
                    args.analysis_k,
                )
                # remove_pseudo 實驗的正確未標註資料池 lineage：
                # 下一輪候選 = 上一輪候選 - 上一輪「實際納入訓練」文檔。
                # 這裡不是直接扣除所有 nest_selected，因後處理可能使抽中者未真正納入。
                # 此集合檢查只驗證候選資料血緣，不驗證模型 checkpoint／global-best 血緣；
                # 若改分析 retain_pseudo 實驗，也不能直接沿用此等式。
                if previous_candidates is not None:
                    assert candidates == previous_candidates - previous_admitted
                previous_candidates, previous_admitted = candidates, admitted
                expected_config = config if expected_config is None else expected_config
                # 三個 seed 的 k、beta、m 必須一致，才能合併成同一組實驗分析。
                assert config == expected_config, "三個實驗的 k、beta、m 不一致"
                # 累積逐文檔與逐單位結果；統計值尚未在不同 seed 或 fold 間混合。
                document_rows.extend(rows)
                unit_rows.append(unit)
                fold_units.append(unit)
                for row in rows:
                    # excluded 文檔不屬於 S1/S2，因此不進 unique group 計數。
                    if row["group"] in {"S1", "S2"}:
                        unique_seed_docs[(fold, row["group"])].add((seed, row["query_doc_id"]))
            # 主控台每折只印漏斗末端的 counts，完整細節寫入輸出 CSV。
            print(
                f"  fold{fold}: candidates={sum(item['candidate_n'] for item in fold_units)}, "
                f"analysis S0={sum(item['analysis_s0_n'] for item in fold_units)}, "
                f"S1={sum(item['s1_n'] for item in fold_units)}, "
                f"S2={sum(item['s2_n'] for item in fold_units)}"
            )

    # 第一層聚合：同一 seed-fold 內，將 comparison-valid rounds 等權平均。
    seed_fold_rows = []
    for experiment in experiments:
        seed = experiment["seed"]
        for fold in range(args.start_fold, args.end_fold + 1):
            units = [
                unit for unit in unit_rows if unit["seed"] == seed and unit["fold"] == fold
            ]
            seed_fold_rows.append(aggregate_seed_fold(seed, fold, units))

    # 第二層聚合：同一 fold 內，將具有有效結果的 seeds 等權平均，得到每折 A1/A2。
    fold_rows = []
    for fold in range(args.start_fold, args.end_fold + 1):
        units = [unit for unit in unit_rows if unit["fold"] == fold]
        per_seed = [row for row in seed_fold_rows if row["fold"] == fold]
        fold_rows.append(aggregate_fold(fold, per_seed, units, unique_seed_docs))

    # Fold 只要至少一個 seed 有 comparison-valid round，就會進目前的 overall 聚合。
    # 是否三個 seed 都有效另由 fold_summary 的 complete_three_seed 欄位呈現。
    valid_folds = [row for row in fold_rows if row["valid_seed_n"] > 0]
    overall = {
        # 保存來源與參數，讓輸出可追溯至指定 summary、實驗根目錄與資料切分。
        "analysis": "multi-seed round-level emotion-category divergence",
        "summary": str(args.summary.resolve()),
        "experiments_root": str(args.experiments_root.resolve()),
        "dataset_dir": str(args.dataset_dir.resolve()),
        "seeds": [item["seed"] for item in experiments],
        "experiments": {str(item["seed"]): item["name"] for item in experiments},
        "folds": list(range(args.start_fold, args.end_fold + 1)),
        "rounds": args.rounds,
        "analysis_type": (
            "source_experiment"
            if args.analysis_k is None
            else "posthoc_nearest_neighbor_prefix"
        ),
        "source_k": expected_config[0],
        "k": expected_config[0] if args.analysis_k is None else args.analysis_k,
        "selection_source_k": expected_config[0],
        "beta": expected_config[1],
        "m": expected_config[2],
        # 明確寫入 S0/S1/S2 及主要聚合定義，避免只看數值時誤解母集合。
        "definitions": {
            "S0": "query 的所有 emotion MASK 預測逐位置等於真值",
            "analysis_S0": "S0 且 query/KNN 均為 single-pair 且 category 有效",
            "S1": "analysis S0 中 query 與全部 KNN category 相同",
            "S2": "analysis S0 中至少一個 KNN category 不同",
            "primary_divergence": "raw_divergence = D_u + beta * D_l",
            "selection_semantics": "selected、EMA、sampling probability、admitted 與模型均沿用 source_k 實驗；事後 analysis_k 不會重新選樣或訓練",
            "primary_aggregation": "round 先在 seed-fold 內等權，再將 seeds 於 fold 內等權，最後 folds 等權",
            "paired_valid_micro": "只合併 S1/S2 都非空之 seed-fold-round 內的文檔－輪次觀察值，再以散度總和除以觀察值總數",
            "all_available_micro": "S1 與 S2 各自保留所有可用文檔－輪次觀察值，再以各組散度總和除以各組觀察值總數",
        },
        # 下列 observation 數是 document-round observations，不是互異文檔數。
        "candidate_document_round_observations": sum(row["candidate_n"] for row in unit_rows),
        "query_exact_s0_observations": sum(row["query_exact_s0_n"] for row in unit_rows),
        "analysis_s0_observations": sum(row["analysis_s0_n"] for row in unit_rows),
        "S1_observations": sum(row["s1_n"] for row in unit_rows),
        "S2_observations": sum(row["s2_n"] for row in unit_rows),
        "valid_fold_n": len(valid_folds),
        "expected_fold_n": args.end_fold - args.start_fold + 1,
        # 第三層 macro：最後讓有效 folds 等權，得到全體 A1、A2 與 A2-A1。
        "raw_A1_overall_macro": mean([row["raw_A1_macro"] for row in valid_folds]),
        "raw_A2_overall_macro": mean([row["raw_A2_macro"] for row in valid_folds]),
        "raw_delta_overall_macro_s2_minus_s1": mean(
            [row["raw_delta_macro_s2_minus_s1"] for row in valid_folds]
        ),
        # 正差 fold 數表示有幾折符合 A1<A2，但不能單獨取代正式檢定。
        "positive_delta_fold_n": sum(
            row["raw_delta_macro_s2_minus_s1"] > 0 for row in valid_folds
        ),
        "ema_delta_overall_macro_s2_minus_s1": mean(
            [row["ema_delta_macro_s2_minus_s1"] for row in valid_folds]
        ),
        "nest_selection_rate_delta_overall_macro_s1_minus_s2": mean(
            [row["nest_selection_rate_delta_macro_s1_minus_s2"] for row in valid_folds]
        ),
        "admission_rate_delta_overall_macro_s1_minus_s2": mean(
            [row["admission_rate_delta_macro_s1_minus_s2"] for row in valid_folds]
        ),
        # sign-flip 只使用 fold-level raw delta，不把大量重複 document-round 列當獨立樣本。
        "exact_fold_sign_flip_test": exact_sign_flip_test(
            [row["raw_delta_macro_s2_minus_s1"] for row in valid_folds]
        ),
    }
    # All-available micro 不要求同一單位同時存在 S1 與 S2，兩組各自保留全部樣本。
    overall["all_available_s1_raw_sum"] = math.fsum(
        unit["s1_raw_sum"] for unit in unit_rows
    )
    overall["all_available_s2_raw_sum"] = math.fsum(
        unit["s2_raw_sum"] for unit in unit_rows
    )
    overall["raw_A1_all_available_micro"] = ratio(
        overall["all_available_s1_raw_sum"],
        overall["S1_observations"],
    )
    overall["raw_A2_all_available_micro"] = ratio(
        overall["all_available_s2_raw_sum"],
        overall["S2_observations"],
    )
    overall["raw_delta_all_available_micro_s2_minus_s1"] = (
        overall["raw_A2_all_available_micro"]
        - overall["raw_A1_all_available_micro"]
    )
    # 十折整體 micro 直接合併所有配對有效單位，不先平均各折結果。
    add_paired_valid_micro_statistics(overall, unit_rows)

    tag = posthoc_tag(args.analysis_k)
    title_suffix = (
        ""
        if args.analysis_k is None
        else f"（來源 k={expected_config[0]}，事後 k={args.analysis_k}）"
    )
    # 五份輸出的分析層級由細到粗：document → seed-fold-round → seed-fold → fold → overall。
    # document_level 包含全部候選與 excluded，不是只保存 S0、S1、S2。
    write_csv(args.output_dir / f"document_level{tag}.csv", document_rows)
    write_csv(args.output_dir / f"unit_seed_fold_round_summary{tag}.csv", unit_rows)
    write_csv(args.output_dir / f"seed_fold_summary{tag}.csv", seed_fold_rows)
    write_csv(args.output_dir / f"fold_summary{tag}.csv", fold_rows)
    # JSON 保留巢狀定義、來源與檢定結果，使用 UTF-8 直接保存中文。
    with (args.output_dir / f"overall_summary{tag}.json").open("w", encoding="utf-8") as file:
        json.dump(overall, file, ensure_ascii=False, indent=2)

    aggregation_name = {
        "macro": "macro",
        "micro": "paired_valid_micro",
        "all_micro": "all_available_micro",
    }[args.plot_aggregation]
    plot_output = args.plot_output or DEFAULT_PNG_DIR / (
        f"fold_{aggregation_name}_A1_A2_mean_divergence{tag}_{args.background}.png"
    )
    plot_fold_divergence(
        fold_rows,
        plot_output,
        args.background,
        args.plot_aggregation,
        title_suffix,
    )
    count_plot_output = args.count_plot_output or DEFAULT_PNG_DIR / (
        f"fold_all_available_micro_S1_S2_sample_count{tag}_{args.background}.png"
    )
    plot_fold_sample_counts(
        fold_rows,
        count_plot_output,
        args.background,
        title=f"S₁ 與 S₂ 的樣本數量{title_suffix}",
    )
    selected_count_rows = aggregate_selected_s0_group_counts(
        document_rows,
        args.start_fold,
        args.end_fold,
    )
    selected_count_plot_output = args.selected_count_plot_output or DEFAULT_PNG_DIR / (
        f"{summary_name(args.summary)}{tag}_fold_selected_S1_S2_S3_"
        f"sample_count_{args.background}.png"
    )
    plot_fold_sample_counts(
        selected_count_rows,
        selected_count_plot_output,
        args.background,
        series=(("s1_n", "S₁"), ("s2_n", "S₂"), ("s3_n", "S₃")),
        title=f"S₁、S₂ 與 S₃ 的樣本數量{title_suffix}",
    )
    selected_count_metric_csv = None
    best_round_count_csv = None
    selected_count_f1_plot_output = None
    per_seed_plot_outputs = []
    round_quality_detail_csv = None
    round_quality_statistics_csv = None
    round_emotion_ratio_plot = None
    round_category_ratio_plot = None
    if args.plot_selected_count_with_emotion_f1 or args.selected_count_f1_plot_output:
        emotion_metric_rows = load_fold_emotion_metrics(
            experiments,
            args.start_fold,
            args.end_fold,
            args.emotion_f1_split,
            args.validation_f1_mode,
            args.rounds,
        )
        plot_count_rows = selected_count_rows
        if (
            args.emotion_f1_split == "validation"
            and args.validation_f1_mode == "best"
        ):
            plot_count_rows, best_round_detail_rows = (
                aggregate_best_validation_round_group_counts(
                    document_rows,
                    emotion_metric_rows,
                    experiments,
                )
            )
            best_round_count_csv = args.output_dir / (
                f"{summary_name(args.summary)}{tag}_seed_fold_best_validation_emotion_"
                "round_S1_S2_S3.csv"
            )
            write_csv(best_round_count_csv, best_round_detail_rows)
        selected_count_metric_rows = [
            {**count_row, **metric_row}
            for count_row, metric_row in zip(plot_count_rows, emotion_metric_rows)
        ]
        if args.emotion_f1_split == "test":
            metric_filename = "emotion_metrics.csv"
            plot_metric_name = "emotion_F1"
        elif args.validation_f1_mode == "round_mean":
            metric_filename = "round_mean_validation_emotion_metrics.csv"
            plot_metric_name = "round_mean_validation_emotion_F1"
        else:
            metric_filename = "validation_emotion_metrics.csv"
            plot_metric_name = "validation_emotion_F1"
        selected_count_metric_csv = args.output_dir / (
            f"{summary_name(args.summary)}{tag}_fold_selected_S1_S2_S3_{metric_filename}"
        )
        write_csv(selected_count_metric_csv, selected_count_metric_rows)
        selected_count_f1_plot_output = (
            args.selected_count_f1_plot_output
            or DEFAULT_PNG_DIR
            / (
                f"{summary_name(args.summary)}{tag}_fold_selected_S1_S2_S3_"
                f"sample_count_and_{plot_metric_name}_{args.background}.png"
            )
        )
        plot_selected_counts_with_emotion_f1(
            plot_count_rows,
            emotion_metric_rows,
            selected_count_f1_plot_output,
            args.background,
            args.emotion_f1_split,
            args.validation_f1_mode,
            args.rounds,
            title_suffix,
        )

    if args.plot_each_seed:
        round_mean_metric_rows = load_fold_round_mean_validation_emotion_f1(
            experiments,
            args.start_fold,
            args.end_fold,
            args.rounds,
        )
        per_seed_plot_data = []
        for experiment in experiments:
            seed = experiment["seed"]
            count_rows = aggregate_selected_s0_group_counts(
                document_rows,
                args.start_fold,
                args.end_fold,
                seed=seed,
            )
            metric_rows = select_seed_round_mean_emotion_f1(
                round_mean_metric_rows,
                seed,
            )
            output_path = DEFAULT_PNG_DIR / (
                f"UECA_NeST{tag}_seed{seed}_S123_round_mean_val_"
                f"emotion_F1_{args.background}.png"
            )
            per_seed_plot_data.append((seed, count_rows, metric_rows, output_path))

        shared_count_ylim = 1.24 * max(
            row[field]
            for _, count_rows, _, _ in per_seed_plot_data
            for row in count_rows
            for field in ("s1_n", "s2_n", "s3_n", "s0_n")
        )
        for seed, count_rows, metric_rows, output_path in per_seed_plot_data:
            plot_selected_counts_with_emotion_f1(
                count_rows,
                metric_rows,
                output_path,
                args.background,
                "validation",
                "round_mean",
                args.rounds,
                f"{title_suffix}（seed={seed}）",
                shared_count_ylim,
            )
            per_seed_plot_outputs.append((seed, output_path))

    if args.plot_round_quality_vs_validation_f1_change:
        round_quality_rows = build_round_quality_f1_change_rows(
            document_rows,
            experiments,
            args.start_fold,
            args.end_fold,
            args.rounds,
        )
        prefix = f"{summary_name(args.summary)}{tag}"
        round_quality_detail_csv = args.output_dir / (
            f"{prefix}_seed_fold_round_training_S1_S2_S3_"
            "validation_emotion_F1_change.csv"
        )
        write_csv(round_quality_detail_csv, round_quality_rows)

        statistic_specs = (
            (
                "emotion_exact_ratio",
                "情緒偽標籤正確比例（%）",
                "情緒偽標籤正確比例與驗證集 Emotion F1 變化的關聯",
                BAR_COLORS[0],
                "round_emotion_exact_ratio_vs_validation_emotion_F1_change",
            ),
            (
                "category_consistency_ratio",
                "KNN 情緒類別完全一致比例（%）",
                "KNN 情緒類別完全一致比例與驗證集 Emotion F1 變化的關聯",
                BAR_COLORS[1],
                "round_KNN_category_consistency_ratio_vs_validation_emotion_F1_change",
            ),
        )
        round_statistics = []
        round_plots = []
        for x_field, x_label, title, color, filename in statistic_specs:
            statistics = fold_cluster_bootstrap_spearman(
                round_quality_rows,
                x_field,
                "delta_val_f1_emotion_percentage_points",
            )
            round_statistics.append(statistics)
            output_path = DEFAULT_PNG_DIR / f"{prefix}_{filename}_{args.background}.png"
            plot_round_ratio_vs_f1_change(
                round_quality_rows,
                x_field,
                x_label,
                f"{title}{title_suffix}",
                output_path,
                args.background,
                statistics,
                color,
            )
            round_plots.append(output_path)

        round_quality_statistics_csv = args.output_dir / (
            f"{prefix}_round_quality_validation_emotion_F1_change_spearman.csv"
        )
        write_csv(round_quality_statistics_csv, round_statistics)
        round_emotion_ratio_plot, round_category_ratio_plot = round_plots

    # 主控台回報主要 raw macro 與 paired-valid micro；其他完整指標請查看輸出表。
    print(f"\n分析完成: {args.output_dir}")
    if args.analysis_k is not None:
        print(
            f"事後鄰居分析: source k={expected_config[0]} → analysis k={args.analysis_k}; "
            "selected／EMA／抽樣機率／模型仍沿用 source k 實驗"
        )
    print(
        f"overall raw A1={overall['raw_A1_overall_macro']:.8f}, "
        f"A2={overall['raw_A2_overall_macro']:.8f}, "
        f"A2-A1={overall['raw_delta_overall_macro_s2_minus_s1']:.8f}, "
        f"positive folds={overall['positive_delta_fold_n']}/{overall['valid_fold_n']}"
    )
    print(
        f"paired-valid micro A1={overall['raw_A1_paired_valid_micro']:.8f}, "
        f"A2={overall['raw_A2_paired_valid_micro']:.8f}, "
        f"A2-A1={overall['raw_delta_paired_valid_micro_s2_minus_s1']:.8f}"
    )
    print(
        f"all-available micro A1={overall['raw_A1_all_available_micro']:.8f}, "
        f"A2={overall['raw_A2_all_available_micro']:.8f}, "
        f"A2-A1={overall['raw_delta_all_available_micro_s2_minus_s1']:.8f}"
    )
    print(f"平均散度圖檔: {plot_output}")
    print(f"樣本數量圖檔: {count_plot_output}")
    print(f"被選中 S1／S2／S3 數量圖檔: {selected_count_plot_output}")
    if selected_count_f1_plot_output:
        if best_round_count_csv:
            print(f"各 seed × fold 最佳驗證輪次的 S1／S2／S3: {best_round_count_csv}")
        print(f"各折 S1／S2／S3 與 Emotion 指標: {selected_count_metric_csv}")
        print(f"S1／S2／S3 數量與 Emotion F1 上下圖: {selected_count_f1_plot_output}")
    for seed, output_path in per_seed_plot_outputs:
        print(f"seed {seed} 的 S1／S2／S3 與五輪平均驗證 Emotion F1 圖檔: {output_path}")
    if round_quality_detail_csv:
        print(f"逐輪訓練樣本組成與 Emotion F1 變化: {round_quality_detail_csv}")
        print(f"逐輪 Spearman 與 fold-cluster bootstrap: {round_quality_statistics_csv}")
        print(f"情緒偽標籤正確比例散點圖: {round_emotion_ratio_plot}")
        print(f"KNN 情緒類別一致比例散點圖: {round_category_ratio_plot}")


if __name__ == "__main__":
    # 只有直接執行此檔案時才啟動分析；被其他模組 import 時不會自動讀寫檔案。
    main()
