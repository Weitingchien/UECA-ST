"""逐 Fold 繪製每輪 S₁／S₂／S₃ 與驗證集三任務 pooled-micro F1。"""

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

from analyze_multiseed_round_emotion_category_divergence import (
    DEFAULT_EXPERIMENTS_ROOT,
    DEFAULT_OUTPUT,
    DEFAULT_PNG_DIR,
    DEFAULT_SUMMARY,
    parse_experiments,
)
from label_utils import configure_fonts


COUNT_COLORS = {
    "s1_n": "#4C78A8",
    "s2_n": "#D68C45",
    "s3_n": "#C44E52",
    "s4_n": "#59A14F",
    "s5_n": "#B279A2",
}
COUNT_LABELS = {
    "s1_n": "S₁",
    "s2_n": "S₂",
    "s3_n": "S₃",
    "s4_n": "S₄=S₁+S₂",
    "s5_n": "S₅=S₁+S₂+S₃",
}
RATIO_COLOR = "#2F6B3F"
F1_COLORS = {
    "emotion_f1": "#4C78A8",
    "cause_f1": "#F28E2B",
    "pair_f1": "#59A14F",
}
TASK_LABELS = {
    "emotion": "Emotion",
    "cause": "Cause",
    "pair": "Pair",
}
TASK_PATTERNS = {
    "emotion": re.compile(
        r"Emotion:\s*預測正確的數量:(\d+(?:\.\d+)?)\s+"
        r"預測出情緒的數量:(\d+(?:\.\d+)?)\s+"
        r"實際正確情緒的數量:(\d+(?:\.\d+)?)"
    ),
    "cause": re.compile(
        r"Cause:\s*預測正確的數量:(\d+(?:\.\d+)?)\s+"
        r"預測出原因的數量:(\d+(?:\.\d+)?)\s+"
        r"實際正確原因的數量:(\d+(?:\.\d+)?)"
    ),
    "pair": re.compile(
        r"Pair:\s*預測正確的數量:(\d+(?:\.\d+)?)\s+"
        r"預測出組合的數量:(\d+(?:\.\d+)?)\s+"
        r"實際正確組合的數量:(\d+(?:\.\d+)?)"
    ),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="每折分別繪製 Round 1～5 的 S₁～S₃ 與驗證集三任務 F1。"
    )
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--experiments_root", type=Path, default=DEFAULT_EXPERIMENTS_ROOT)
    parser.add_argument("--analysis_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--png_dir", type=Path, default=DEFAULT_PNG_DIR)
    parser.add_argument("--analysis_k", type=int, default=3)
    parser.add_argument("--start_fold", type=int, default=1)
    parser.add_argument("--end_fold", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--background", choices=("white", "gray"), default="white")
    return parser.parse_args()


def read_round_group_counts(document_csv, start_fold, end_fold, rounds):
    """串流彙總三個 seed 在每折、每輪被選中的 S₁／S₂／S₃。"""

    counts = defaultdict(lambda: {"s1_n": 0, "s2_n": 0, "s3_n": 0})
    source_k_values = set()
    with document_csv.open(encoding="utf-8-sig") as file:
        for row in csv.DictReader(file):
            fold = int(row["fold"])
            round_index = int(row["round"])
            if not (
                start_fold <= fold <= end_fold
                and 1 <= round_index <= rounds
                and row["nest_selected"] == "True"
                and row["category_valid"] == "True"
            ):
                continue

            source_k_values.add(int(row["source_k"]))
            key = (fold, round_index)
            if row["query_emotion_exact_s0"] == "True":
                mismatch_count = int(row["category_mismatch_count"])
                field = "s1_n" if mismatch_count == 0 else "s2_n"
            else:
                field = "s3_n"
            counts[key][field] += 1

    assert len(source_k_values) == 1, f"source k 不唯一: {source_k_values}"
    result = []
    for fold in range(start_fold, end_fold + 1):
        for round_index in range(1, rounds + 1):
            row = counts[(fold, round_index)]
            s4_n = row["s1_n"] + row["s2_n"]
            s5_n = s4_n + row["s3_n"]
            assert s5_n > 0, f"Fold {fold} Round {round_index} 沒有有效 S₅ 樣本"
            result.append(
                {
                    "fold": fold,
                    "round": round_index,
                    **row,
                    "s4_n": s4_n,
                    "s5_n": s5_n,
                    "s4_s5_ratio": s4_n / s5_n,
                }
            )
    return result, source_k_values.pop()


def load_round_pooled_validation_f1(
    experiments,
    start_fold,
    end_fold,
    rounds,
):
    """合併同折、同輪三個 seed 的 TP／預測數／實際數後計算三任務 F1。"""

    result_dirs = {}
    for experiment in experiments:
        candidates = sorted(
            experiment["experiment_dir"].glob("self_training_results_*")
        )
        assert len(candidates) == 1, (
            f"self_training_results 目錄數量不是 1: {experiment['experiment_dir']}"
        )
        result_dirs[experiment["seed"]] = candidates[0]

    rows = []
    for fold in range(start_fold, end_fold + 1):
        for round_index in range(1, rounds + 1):
            task_counts = {
                task: {"tp": 0, "predicted": 0, "gold": 0}
                for task in TASK_LABELS
            }
            for experiment in experiments:
                result_path = result_dirs[experiment["seed"]] / (
                    f"self_training_val_results_fold{fold}_round{round_index}.txt"
                )
                text = result_path.read_text(encoding="utf-8")
                for task, pattern in TASK_PATTERNS.items():
                    match = pattern.search(text)
                    assert match, f"找不到 {task} 驗證計數: {result_path}"
                    tp, predicted, gold = (int(float(value)) for value in match.groups())
                    task_counts[task]["tp"] += tp
                    task_counts[task]["predicted"] += predicted
                    task_counts[task]["gold"] += gold

            row = {"fold": fold, "round": round_index}
            for task, counts in task_counts.items():
                assert counts["predicted"] > 0 and counts["gold"] > 0
                row[f"{task}_tp"] = counts["tp"]
                row[f"{task}_predicted"] = counts["predicted"]
                row[f"{task}_gold"] = counts["gold"]
                row[f"{task}_f1"] = 2 * counts["tp"] / (
                    counts["predicted"] + counts["gold"]
                )
            rows.append(row)
    return rows


def merge_rows(count_rows, metric_rows):
    metrics_by_key = {
        (row["fold"], row["round"]): row for row in metric_rows
    }
    assert len(metrics_by_key) == len(metric_rows)
    merged = []
    for count_row in count_rows:
        key = (count_row["fold"], count_row["round"])
        metric_row = metrics_by_key[key]
        merged.append({**count_row, **metric_row})
    return merged


def aggregate_folds(rows, rounds):
    """依 Round 加總所有 Fold 的數量與驗證計數，重新計算比例及 micro F1。"""

    aggregated = []
    for round_index in range(1, rounds + 1):
        round_rows = [row for row in rows if row["round"] == round_index]
        assert round_rows, f"Round {round_index} 沒有資料"
        row = {"round": round_index}
        for field in ("s1_n", "s2_n", "s3_n", "s4_n", "s5_n"):
            row[field] = sum(item[field] for item in round_rows)
        assert row["s4_n"] == row["s1_n"] + row["s2_n"]
        assert row["s5_n"] == row["s4_n"] + row["s3_n"]
        row["s4_s5_ratio"] = row["s4_n"] / row["s5_n"]

        for task in TASK_LABELS:
            for suffix in ("tp", "predicted", "gold"):
                field = f"{task}_{suffix}"
                row[field] = sum(item[field] for item in round_rows)
            row[f"{task}_f1"] = 2 * row[f"{task}_tp"] / (
                row[f"{task}_predicted"] + row[f"{task}_gold"]
            )
        aggregated.append(row)
    return aggregated


def plot_rounds(
    rows,
    output_path,
    source_k,
    analysis_k,
    background,
    scope_title,
    aggregation_description,
    *,
    show_s4_count=False,
    grouped_s1_s5=False,
    show_ratio=True,
    f1_fields=("emotion_f1", "cause_f1", "pair_f1"),
    f1_legend_loc="upper center",
    f1_legend_bbox_to_anchor=None,
    figure_title=None,
    f1_ylabel="驗證集 pooled micro F1（%）",
    suptitle_y=0.995,
    layout_rect_top=0.95,
    layout_pad=1.08,
):
    configure_fonts(font_size=15)
    plt.rcParams["font.family"] = [*plt.rcParams["font.family"], "DejaVu Serif"]
    facecolor = "#DDDDDD" if background == "gray" else "white"
    if show_ratio:
        fig, (count_ax, ratio_ax, f1_ax) = plt.subplots(
            3,
            1,
            figsize=(12, 11),
            dpi=180,
            sharex=True,
            gridspec_kw={"height_ratios": (1.25, 0.55, 1)},
        )
        axes = (count_ax, ratio_ax, f1_ax)
    else:
        fig, (count_ax, f1_ax) = plt.subplots(
            2,
            1,
            figsize=(12, 8),
            dpi=180,
            sharex=True,
            gridspec_kw={"height_ratios": (1.35, 1)},
        )
        ratio_ax = None
        axes = (count_ax, f1_ax)
    fig.patch.set_facecolor(facecolor)
    for axis in axes:
        axis.set_facecolor(facecolor)

    rounds = [row["round"] for row in rows]
    if grouped_s1_s5:
        count_fields = ("s1_n", "s2_n", "s3_n", "s4_n", "s5_n")
        bar_width = 0.14
        center = (len(count_fields) - 1) / 2
        for index, field in enumerate(count_fields):
            values = [row[field] for row in rows]
            positions = [
                round_index + (index - center) * bar_width
                for round_index in rounds
            ]
            bars = count_ax.bar(
                positions,
                values,
                width=bar_width,
                label=COUNT_LABELS[field],
                color=COUNT_COLORS[field],
                edgecolor="white",
                linewidth=0.8,
                zorder=2,
            )
            count_ax.bar_label(
                bars,
                labels=[str(value) for value in values],
                padding=2,
                fontsize=9,
            )
        count_max = max(row["s5_n"] for row in rows)
        count_ax.legend(loc="upper center", ncol=5, frameon=True)
    else:
        bottoms = [0] * len(rows)
        for field in ("s1_n", "s2_n", "s3_n"):
            values = [row[field] for row in rows]
            bars = count_ax.bar(
                rounds,
                values,
                width=0.62,
                bottom=bottoms,
                label=COUNT_LABELS[field],
                color=COUNT_COLORS[field],
                edgecolor="white",
                linewidth=1,
                zorder=2,
            )
            count_ax.bar_label(
                bars,
                labels=[str(value) for value in values],
                label_type="center",
                fontsize=10,
                color="white",
            )
            bottoms = [bottom + value for bottom, value in zip(bottoms, values)]

        for row, stacked_total in zip(rows, bottoms):
            round_index = row["round"]
            assert stacked_total == row["s5_n"]
            if show_s4_count:
                count_ax.annotate(
                    f"S₄={row['s4_n']}",
                    (round_index, row["s4_n"]),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    color="white",
                )
            count_ax.annotate(
                f"S₅={row['s5_n']}",
                (round_index, row["s5_n"]),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                fontsize=10,
            )
        count_max = max(bottoms)
        count_ax.legend(loc="upper center", ncol=3, frameon=True)

    if ratio_ax is not None:
        ratio_values = [100 * row["s4_s5_ratio"] for row in rows]
        ratio_ax.plot(
            rounds,
            ratio_values,
            color=RATIO_COLOR,
            marker="D",
            linewidth=2.2,
            markersize=6,
            label="S₄/S₅",
            zorder=4,
        )
        for round_index, value in zip(rounds, ratio_values):
            ratio_ax.annotate(
                f"{value:.1f}%",
                (round_index, value),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                fontsize=10,
                color=RATIO_COLOR,
            )
        ratio_ax.set_ylim(0, 100)
        ratio_ax.set_yticks((0, 25, 50, 75, 100))
        ratio_ax.set_ylabel("S₄/S₅（%）", color=RATIO_COLOR)
        ratio_ax.tick_params(axis="y", colors=RATIO_COLOR)
        ratio_ax.legend(loc="upper center", frameon=True)

    count_ax.set_ylabel("當輪樣本數量（筆）")
    count_ax.set_ylim(0, count_max * 1.23)

    f1_series = {
        "emotion_f1": ("Emotion F1", 7),
        "cause_f1": ("Cause F1", 7),
        "pair_f1": ("Pair F1", -14),
    }
    assert set(f1_fields) <= set(f1_series)
    for field in f1_fields:
        label, label_offset = f1_series[field]
        values = [100 * row[field] for row in rows]
        f1_ax.plot(
            rounds,
            values,
            color=F1_COLORS[field],
            marker="o",
            linewidth=2.2,
            markersize=7,
            label=label,
            zorder=3,
        )
        for round_index, value in zip(rounds, values):
            f1_ax.annotate(
                f"{value:.2f}",
                (round_index, value),
                xytext=(0, label_offset),
                textcoords="offset points",
                ha="center",
                fontsize=9,
                color=F1_COLORS[field],
            )

    f1_ax.set_ylim(0, 100)
    f1_ax.set_ylabel(f1_ylabel)
    f1_ax.set_xlabel("Round")
    f1_ax.set_xticks(rounds)
    f1_ax.legend(
        loc=f1_legend_loc,
        bbox_to_anchor=f1_legend_bbox_to_anchor,
        ncol=len(f1_fields),
        frameon=True,
    )

    for axis in axes:
        axis.grid(
            True,
            axis="y",
            linestyle="--",
            linewidth=1,
            alpha=0.35,
            color="black",
            zorder=1,
        )
        axis.set_axisbelow(True)

    if figure_title is None:
        figure_title = (
            f"{scope_title}：Round 1 至 Round {len(rows)} 當輪 S₁～S₅與驗證集三任務 F1\n"
            f"（{aggregation_description}；來源 k={source_k}，事後 k={analysis_k}）"
        )
    fig.suptitle(figure_title, y=suptitle_y)
    fig.tight_layout(
        rect=(0, 0, 1, layout_rect_top),
        h_pad=1.7,
        pad=layout_pad,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor=facecolor)
    plt.close(fig)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    tag = f"posthoc_k{args.analysis_k}"
    document_csv = args.analysis_dir / f"document_level_{tag}.csv"
    assert document_csv.exists(), f"找不到逐文檔分析檔: {document_csv}"

    count_rows, source_k = read_round_group_counts(
        document_csv,
        args.start_fold,
        args.end_fold,
        args.rounds,
    )
    experiments = parse_experiments(args.summary, args.experiments_root)
    metric_rows = load_round_pooled_validation_f1(
        experiments,
        args.start_fold,
        args.end_fold,
        args.rounds,
    )
    rows = merge_rows(count_rows, metric_rows)

    output_dir = args.png_dir / f"fold_round_validation_f1_{tag}_{args.background}"
    outputs = []
    for fold in range(args.start_fold, args.end_fold + 1):
        fold_rows = [row for row in rows if row["fold"] == fold]
        output_path = output_dir / (
            f"UECA_NeST_{tag}_fold{fold}_round_S123_"
            f"validation_ECP_F1_pooled_micro_{args.background}.png"
        )
        plot_rounds(
            fold_rows,
            output_path,
            source_k,
            args.analysis_k,
            args.background,
            f"Fold {fold}",
            "三個 seed 數量加總／驗證計數 pooled micro",
        )
        outputs.append(output_path)

    all_fold_rows = aggregate_folds(rows, args.rounds)
    all_fold_output = output_dir / (
        f"UECA_NeST_{tag}_fold{args.start_fold}-{args.end_fold}_sum_round_S123_"
        f"validation_ECP_F1_pooled_micro_{args.background}.png"
    )
    plot_rounds(
        all_fold_rows,
        all_fold_output,
        source_k,
        args.analysis_k,
        args.background,
        f"Fold {args.start_fold} 至 Fold {args.end_fold} 彙總",
        "三個 seed × 十折數量加總／驗證計數 pooled micro",
        grouped_s1_s5=True,
        show_ratio=False,
        f1_fields=("emotion_f1",),
        f1_legend_loc="lower center",
        f1_legend_bbox_to_anchor=(0.5, 1.02),
        figure_title="UECA-NeST-Semo-Tpair-Fbest 五輪自訓練的分組數量與驗證集 Emotion F1",
        f1_ylabel="驗證集 Emotion F1（%）",
        suptitle_y=0.985,
        layout_rect_top=0.975,
        layout_pad=0.8,
    )

    csv_path = args.analysis_dir / (
        f"UECA_NeST_{tag}_fold_round_S123_"
        "validation_ECP_F1_pooled_micro.csv"
    )
    write_csv(csv_path, rows)
    all_fold_csv_path = args.analysis_dir / (
        f"UECA_NeST_{tag}_fold{args.start_fold}-{args.end_fold}_sum_round_S123_"
        "validation_ECP_F1_pooled_micro.csv"
    )
    write_csv(all_fold_csv_path, all_fold_rows)
    print(f"逐輪明細: {csv_path}")
    print(f"十折彙總明細: {all_fold_csv_path}")
    print(f"十折彙總圖: {all_fold_output}")
    for output in outputs:
        print(f"Fold 圖: {output}")


if __name__ == "__main__":
    main()
