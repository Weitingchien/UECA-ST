"""Plot single-seed, per-fold self-training trends and Spearman ranks.

This is an offline diagnostic.  Ground-truth-based S groups are used only for
post-hoc analysis and never feed back into NeST selection or model training.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from analyze_multiseed_round_emotion_category_divergence import (
    DEFAULT_EXPERIMENTS_ROOT,
    DEFAULT_OUTPUT,
    DEFAULT_PNG_DIR,
    DEFAULT_SUMMARY,
    average_ranks,
    parse_experiments,
    spearman_rho,
)
from label_utils import configure_fonts
from plot_fold_round_s123_validation_f1 import (
    load_round_pooled_validation_f1,
    merge_rows,
)


COLORS = {
    "s4_n": "#009E73",
    "s3_n": "#D55E00",
    "s4_s5_ratio": "#CC79A7",
    "emotion_f1": "#0072B2",
}
LABELS = {
    "s4_n": "S₄",
    "s3_n": "S₃",
    "s4_s5_ratio": "S₄/S₅",
    "emotion_f1": "驗證集 Emotion F1",
}
RANK_FIELDS = {
    "s4_n": "s4_rank",
    "s3_n": "s3_rank",
    "s4_s5_ratio": "s4_s5_ratio_rank",
    "emotion_f1": "emotion_f1_rank",
}
SPEARMAN_FIELDS = ("s4_n", "s3_n", "s4_s5_ratio")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "以單一 seed、逐 Fold 繪製 S₄、S₃、S₄/S₅、驗證集 Emotion F1 "
            "及折內 Spearman 排名。"
        )
    )
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument(
        "--experiments_root", type=Path, default=DEFAULT_EXPERIMENTS_ROOT
    )
    parser.add_argument("--analysis_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--png_dir", type=Path, default=DEFAULT_PNG_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--analysis_k", type=int, default=3)
    parser.add_argument("--start_fold", type=int, default=1)
    parser.add_argument("--end_fold", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--background", choices=("white", "gray"), default="white")
    return parser.parse_args()


def read_seed_round_group_counts(
    document_csv,
    seed,
    analysis_k,
    start_fold,
    end_fold,
    rounds,
):
    """Count S₁–S₆ for one seed while retaining Fold×Round keys."""

    grouped = defaultdict(
        lambda: {"s1_n": 0, "s2_n": 0, "s3_n": 0, "s6_n": 0}
    )
    source_k_values = set()
    matched_analysis_k = set()

    with document_csv.open(encoding="utf-8-sig") as file:
        for row in csv.DictReader(file):
            if int(row["seed"]) != seed:
                continue
            fold = int(row["fold"])
            round_index = int(row["round"])
            if not (
                start_fold <= fold <= end_fold
                and 1 <= round_index <= rounds
                and row["nest_selected"] == "True"
            ):
                continue

            source_k_values.add(int(row["source_k"]))
            matched_analysis_k.add(int(row["analysis_k"]))
            counts = grouped[(fold, round_index)]
            counts["s6_n"] += 1

            if row["category_valid"] != "True":
                continue
            if row["query_emotion_exact_s0"] == "True":
                field = (
                    "s1_n" if int(row["category_mismatch_count"]) == 0 else "s2_n"
                )
            else:
                field = "s3_n"
            counts[field] += 1

    assert len(source_k_values) == 1, f"source k 不唯一: {source_k_values}"
    assert matched_analysis_k == {analysis_k}, (
        f"CSV analysis k 與參數不一致: {matched_analysis_k} != {{{analysis_k}}}"
    )

    rows = []
    for fold in range(start_fold, end_fold + 1):
        for round_index in range(1, rounds + 1):
            counts = grouped[(fold, round_index)]
            s4_n = counts["s1_n"] + counts["s2_n"]
            s5_n = s4_n + counts["s3_n"]
            assert s5_n > 0, f"Fold {fold} Round {round_index} 的 S₅ 為 0"
            assert s5_n <= counts["s6_n"]
            rows.append(
                {
                    "seed": seed,
                    "fold": fold,
                    "round": round_index,
                    "source_k": next(iter(source_k_values)),
                    "analysis_k": analysis_k,
                    **counts,
                    "s4_n": s4_n,
                    "s5_n": s5_n,
                    "s4_s5_ratio": s4_n / s5_n,
                }
            )
    return rows, next(iter(source_k_values))


def add_descending_ranks(rows, start_fold, end_fold, rounds):
    """Add within-fold average ranks where rank 1 means the largest value."""

    ranked_rows = [dict(row) for row in rows]
    for fold in range(start_fold, end_fold + 1):
        fold_rows = sorted(
            (row for row in ranked_rows if row["fold"] == fold),
            key=lambda row: row["round"],
        )
        assert len(fold_rows) == rounds
        for value_field, rank_field in RANK_FIELDS.items():
            ranks = average_ranks([-float(row[value_field]) for row in fold_rows])
            assert math.isclose(
                float(np.sum(ranks)), rounds * (rounds + 1) / 2
            )
            for row, rank in zip(fold_rows, ranks):
                row[rank_field] = float(rank)
    return ranked_rows


def build_fold_spearman_rows(rows, start_fold, end_fold, rounds):
    result = []
    for fold in range(start_fold, end_fold + 1):
        fold_rows = sorted(
            (row for row in rows if row["fold"] == fold),
            key=lambda row: row["round"],
        )
        assert len(fold_rows) == rounds
        f1_values = [float(row["emotion_f1"]) for row in fold_rows]
        for field in SPEARMAN_FIELDS:
            rho = spearman_rho(
                [float(row[field]) for row in fold_rows],
                f1_values,
            )
            result.append(
                {
                    "seed": fold_rows[0]["seed"],
                    "fold": fold,
                    "x_field": field,
                    "x_label": LABELS[field],
                    "n_rounds": rounds,
                    "spearman_rho": rho,
                    "emotion_f1_min": min(f1_values),
                    "emotion_f1_max": max(f1_values),
                    "emotion_f1_range_percentage_points": 100
                    * (max(f1_values) - min(f1_values)),
                }
            )
    return result


def summarize_spearman(rows, seed):
    summary = []
    for field in SPEARMAN_FIELDS:
        values = [
            float(row["spearman_rho"])
            for row in rows
            if row["x_field"] == field and row["spearman_rho"] is not None
        ]
        assert values
        q1, median, q3 = np.percentile(values, (25, 50, 75))
        tolerance = 1e-12
        summary.append(
            {
                "seed": seed,
                "x_field": field,
                "x_label": LABELS[field],
                "valid_fold_n": len(values),
                "median_rho": float(median),
                "q1_rho": float(q1),
                "q3_rho": float(q3),
                "min_rho": min(values),
                "max_rho": max(values),
                "positive_fold_n": sum(value > tolerance for value in values),
                "zero_fold_n": sum(abs(value) <= tolerance for value in values),
                "negative_fold_n": sum(value < -tolerance for value in values),
            }
        )
    return summary


def nice_count_upper(rows):
    maximum = max(max(row["s4_n"], row["s3_n"]) for row in rows)
    return max(10, int(math.ceil(maximum * 1.22 / 10) * 10))


def plot_fold(rows, spearman_rows, output_path, count_upper, background):
    configure_fonts(font_size=13)
    plt.rcParams["font.family"] = [*plt.rcParams["font.family"], "DejaVu Serif"]
    facecolor = "#DDDDDD" if background == "gray" else "white"
    fold = rows[0]["fold"]
    seed = rows[0]["seed"]
    source_k = rows[0]["source_k"]
    analysis_k = rows[0]["analysis_k"]
    rounds = [row["round"] for row in rows]

    fig, (count_ax, percentage_ax, rank_ax) = plt.subplots(
        3,
        1,
        figsize=(10.8, 10.2),
        dpi=180,
        sharex=True,
        gridspec_kw={"height_ratios": (1.25, 1, 1.05)},
    )
    fig.patch.set_facecolor(facecolor)
    for axis in (count_ax, percentage_ax, rank_ax):
        axis.set_facecolor(facecolor)
        axis.grid(
            True,
            axis="y",
            linestyle="--",
            linewidth=1,
            alpha=0.32,
            color="black",
        )
        axis.set_axisbelow(True)

    plot_specs = {
        "s4_n": {"marker": "s", "linestyle": "-"},
        "s3_n": {"marker": "^", "linestyle": "--"},
        "s4_s5_ratio": {"marker": "D", "linestyle": "-."},
        "emotion_f1": {"marker": "o", "linestyle": "-"},
    }

    for field in ("s4_n", "s3_n"):
        values = [row[field] for row in rows]
        count_ax.plot(
            rounds,
            values,
            color=COLORS[field],
            linewidth=2.2,
            markersize=7,
            label=LABELS[field],
            **plot_specs[field],
        )
        label_offset = -8 if field == "s4_n" else 8
        horizontal_offset = -7 if field == "s4_n" else 7
        horizontal_alignment = "right" if field == "s4_n" else "left"
        for round_index, value in zip(rounds, values):
            count_ax.annotate(
                str(value),
                (round_index, value),
                xytext=(horizontal_offset, label_offset),
                textcoords="offset points",
                ha=horizontal_alignment,
                va="center",
                color=COLORS[field],
                fontsize=10,
            )
    count_ax.set_ylim(0, count_upper)
    count_ax.set_ylabel("當輪文件數量（筆）")
    count_ax.set_title("(a) S₄ 與 S₃ 的當輪數量", loc="left", fontsize=14)

    percentage_values = {
        "s4_s5_ratio": [100 * row["s4_s5_ratio"] for row in rows],
        "emotion_f1": [100 * row["emotion_f1"] for row in rows],
    }
    for field in ("s4_s5_ratio", "emotion_f1"):
        values = percentage_values[field]
        percentage_ax.plot(
            rounds,
            values,
            color=COLORS[field],
            linewidth=2.2,
            markersize=7,
            label=LABELS[field],
            **plot_specs[field],
        )
        label_offset = 9 if field == "s4_s5_ratio" else -16
        decimals = 1 if field == "s4_s5_ratio" else 2
        for round_index, value in zip(rounds, values):
            percentage_ax.annotate(
                f"{value:.{decimals}f}%",
                (round_index, value),
                xytext=(0, label_offset),
                textcoords="offset points",
                ha="center",
                va="center",
                color=COLORS[field],
                fontsize=9,
            )
    percentage_ax.set_ylim(0, 100)
    percentage_ax.set_yticks((0, 25, 50, 75, 100))
    percentage_ax.set_ylabel("百分比（%）")
    percentage_ax.set_title(
        "(b) 當輪 S₄/S₅ 與累積自訓練後的驗證集 Emotion F1",
        loc="left",
        fontsize=14,
    )

    for field in ("s4_n", "s3_n", "s4_s5_ratio", "emotion_f1"):
        rank_ax.plot(
            rounds,
            [row[RANK_FIELDS[field]] for row in rows],
            color=COLORS[field],
            linewidth=2.0,
            markersize=6,
            label=LABELS[field],
            **plot_specs[field],
        )
    rho_by_field = {
        row["x_field"]: row["spearman_rho"] for row in spearman_rows
    }
    rho_parts = []
    for field in SPEARMAN_FIELDS:
        rho = rho_by_field[field]
        formatted_rho = "NA" if rho is None else f"{rho:+.3f}"
        rho_parts.append(f"ρ({LABELS[field]}, F1)={formatted_rho}")
    rho_text = "，".join(rho_parts)
    rank_ax.set_ylim(5.35, 0.65)
    rank_ax.set_yticks((1, 2, 3, 4, 5))
    rank_ax.set_ylabel("數值排名\n（1＝數值最高）")
    rank_ax.set_xlabel("Round")
    rank_ax.set_xticks(rounds)
    rank_ax.set_title(f"(c) 折內排名；{rho_text}", loc="left", fontsize=13)

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=COLORS[field],
            linewidth=2.2,
            markersize=7,
            label=LABELS[field],
            **plot_specs[field],
        )
        for field in ("s4_n", "s3_n", "s4_s5_ratio", "emotion_f1")
    ]
    fig.suptitle(
        f"UECA-NeST-Semo-Tpair-Fbest（seed={seed}）— Fold {fold} 五輪分析",
        y=0.982,
        fontsize=18,
    )
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.947),
        ncol=4,
        frameon=True,
    )
    fig.text(
        0.5,
        0.018,
        (
            f"來源 k={source_k}、事後 k={analysis_k}；S₄=S₁+S₂，S₅=S₄+S₃；"
            "排名 1 表示數值最高（S₃ 排名 1 表示錯誤數最多）"
        ),
        ha="center",
        fontsize=10,
    )
    fig.subplots_adjust(
        left=0.105,
        right=0.98,
        bottom=0.085,
        top=0.885,
        hspace=0.32,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, facecolor=facecolor)
    plt.close(fig)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    assert args.start_fold <= args.end_fold
    assert args.rounds >= 2
    tag = f"posthoc_k{args.analysis_k}"
    document_csv = args.analysis_dir / f"document_level_{tag}.csv"
    assert document_csv.exists(), f"找不到逐文檔分析檔: {document_csv}"

    experiments = parse_experiments(args.summary, args.experiments_root)
    selected_experiments = [item for item in experiments if item["seed"] == args.seed]
    assert len(selected_experiments) == 1, (
        f"summary 中 seed={args.seed} 的實驗數量不是 1"
    )
    experiment = selected_experiments[0]

    count_rows, source_k = read_seed_round_group_counts(
        document_csv,
        args.seed,
        args.analysis_k,
        args.start_fold,
        args.end_fold,
        args.rounds,
    )
    metric_rows = load_round_pooled_validation_f1(
        selected_experiments,
        args.start_fold,
        args.end_fold,
        args.rounds,
    )
    count_keys = {(row["fold"], row["round"]) for row in count_rows}
    metric_keys = {(row["fold"], row["round"]) for row in metric_rows}
    assert count_keys == metric_keys
    rows = add_descending_ranks(
        merge_rows(count_rows, metric_rows),
        args.start_fold,
        args.end_fold,
        args.rounds,
    )

    expected_n = (args.end_fold - args.start_fold + 1) * args.rounds
    assert len(rows) == expected_n
    for fold in range(args.start_fold, args.end_fold + 1):
        fold_rows = [row for row in rows if row["fold"] == fold]
        assert len(fold_rows) == args.rounds
        assert len({row["emotion_gold"] for row in fold_rows}) == 1
        assert len({row["s6_n"] for row in fold_rows}) == 1
        for row in fold_rows:
            assert row["s4_n"] == row["s1_n"] + row["s2_n"]
            assert row["s5_n"] == row["s4_n"] + row["s3_n"]
            assert math.isclose(row["s4_s5_ratio"], row["s4_n"] / row["s5_n"])
            assert math.isclose(
                row["emotion_f1"],
                2
                * row["emotion_tp"]
                / (row["emotion_predicted"] + row["emotion_gold"]),
            )

    spearman_rows = build_fold_spearman_rows(
        rows, args.start_fold, args.end_fold, args.rounds
    )
    spearman_summary = summarize_spearman(spearman_rows, args.seed)

    output_dir = args.png_dir / (
        f"seed{args.seed}_fold_round_validation_emotion_spearman_"
        f"{tag}_{args.background}"
    )
    count_upper = nice_count_upper(rows)
    output_paths = []
    for fold in range(args.start_fold, args.end_fold + 1):
        fold_rows = sorted(
            (row for row in rows if row["fold"] == fold),
            key=lambda row: row["round"],
        )
        fold_spearman = [row for row in spearman_rows if row["fold"] == fold]
        output_path = output_dir / (
            f"UECA_NeST_{tag}_seed{args.seed}_fold{fold}_round_"
            f"S4_S3_validation_emotion_F1_micro_spearman_{args.background}.png"
        )
        plot_fold(
            fold_rows,
            fold_spearman,
            output_path,
            count_upper,
            args.background,
        )
        output_paths.append(output_path)

    prefix = f"UECA_NeST_{tag}_seed{args.seed}"
    detail_csv = args.analysis_dir / (
        f"{prefix}_fold_round_S4_S3_validation_emotion_F1_micro.csv"
    )
    spearman_csv = args.analysis_dir / (
        f"{prefix}_fold_S4_S3_ratio_validation_emotion_F1_spearman.csv"
    )
    summary_csv = args.analysis_dir / (
        f"{prefix}_fold1-10_S4_S3_ratio_validation_emotion_F1_"
        "spearman_summary.csv"
    )
    metadata_path = args.analysis_dir / (
        f"{prefix}_fold_round_validation_emotion_spearman_metadata.json"
    )
    write_csv(detail_csv, rows)
    write_csv(spearman_csv, spearman_rows)
    write_csv(summary_csv, spearman_summary)
    metadata_path.write_text(
        json.dumps(
            {
                "summary": str(args.summary),
                "experiments_root": str(args.experiments_root),
                "experiment": experiment["name"],
                "seed": args.seed,
                "source_k": source_k,
                "analysis_k": args.analysis_k,
                "folds": [args.start_fold, args.end_fold],
                "rounds": args.rounds,
                "definitions": {
                    "S4": "S1 + S2; query emotion MASK exact-correct within category-valid selected subset",
                    "S3": "query emotion MASK not exact-correct within category-valid selected subset",
                    "S5": "S4 + S3; category-valid selected subset",
                    "S6": "all documents selected by NeST",
                    "emotion_f1": "2 * TP / (predicted + gold)",
                    "rank": "within-fold descending average rank; rank 1 is the largest value",
                    "spearman": "Pearson correlation of average ranks; ties retained",
                },
                "interpretation_notes": [
                    "S counts describe the newly selected cohort in the current round.",
                    "Validation F1 is measured after training on labeled data plus pseudo data accumulated through the current round.",
                    "From round 2 onward, training reloads the historical Pair-validation-best checkpoint, which may not be the immediately preceding round.",
                    "This is a descriptive association analysis and does not establish causality.",
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"單位明細: {detail_csv}")
    print(f"逐 Fold Spearman: {spearman_csv}")
    print(f"Spearman 摘要: {summary_csv}")
    print(f"資料來源說明: {metadata_path}")
    print(f"圖檔目錄: {output_dir}")
    for output_path in output_paths:
        print(f"Fold 圖: {output_path}")


if __name__ == "__main__":
    main()
