"""繪製各折 Pair-best 累積 S₁～S₅ 與最終測試 Emotion pooled-micro F1。"""

import argparse
import csv
import hashlib
import json
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
from plot_fold_round_s123_validation_f1 import (
    COUNT_COLORS,
    COUNT_LABELS,
    F1_COLORS,
    TASK_PATTERNS,
    write_csv,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "上圖繪製各 Fold 至 Pair-best 回合的累積 S₁～S₅，"
            "下圖繪製相同 Fold 的最終測試集 Emotion pooled-micro F1。"
        )
    )
    parser.add_argument("--count_summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--test_summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--experiments_root", type=Path, default=DEFAULT_EXPERIMENTS_ROOT)
    parser.add_argument("--analysis_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--png_dir", type=Path, default=DEFAULT_PNG_DIR)
    parser.add_argument("--analysis_k", type=int, default=3)
    parser.add_argument("--start_fold", type=int, default=1)
    parser.add_argument("--end_fold", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--background", choices=("white", "gray"), default="white")
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_matching_training_trajectory(
    count_experiments,
    test_experiments,
    start_fold,
    end_fold,
    rounds,
):
    """確認上下圖來源實驗的逐輪選樣、預測與納入資料完全相同。"""

    count_by_seed = {item["seed"]: item for item in count_experiments}
    test_by_seed = {item["seed"]: item for item in test_experiments}
    assert count_by_seed.keys() == test_by_seed.keys(), (
        "S₁～S₅ 與測試 F1 的 seed 集合不同: "
        f"{sorted(count_by_seed)} != {sorted(test_by_seed)}"
    )

    filename_templates = (
        "nest_divergence_scores_fold{fold}_round{round}.json",
        "all_unlabeled_pseudo_predictions_fold{fold}_round{round}.json",
        "pseudo_labeled_samples_fold{fold}_round{round}.json",
    )
    matched_file_pairs = 0
    identical_path_pairs = 0
    sha256_matched_pairs = 0
    for seed in sorted(count_by_seed):
        count_pseudo_dir = count_by_seed[seed]["pseudo_dir"]
        test_pseudo_dir = test_by_seed[seed]["pseudo_dir"]
        for fold in range(start_fold, end_fold + 1):
            for round_index in range(1, rounds + 1):
                for template in filename_templates:
                    filename = template.format(fold=fold, round=round_index)
                    count_path = count_pseudo_dir / filename
                    test_path = test_pseudo_dir / filename
                    assert count_path.exists(), f"找不到分組數量來源檔: {count_path}"
                    assert test_path.exists(), f"找不到測試軌跡來源檔: {test_path}"
                    if count_path.resolve() == test_path.resolve():
                        matched_file_pairs += 1
                        identical_path_pairs += 1
                        continue
                    assert count_path.stat().st_size == test_path.stat().st_size, (
                        f"上下圖來源實驗的檔案大小不同: {count_path} != {test_path}"
                    )
                    assert sha256_file(count_path) == sha256_file(test_path), (
                        f"上下圖來源實驗的內容不同: {count_path} != {test_path}"
                    )
                    matched_file_pairs += 1
                    sha256_matched_pairs += 1

    return {
        "method": (
            "identical source paths"
            if sha256_matched_pairs == 0
            else "identical source paths or SHA-256 exact match"
        ),
        "matched_file_pairs": matched_file_pairs,
        "identical_path_pairs": identical_path_pairs,
        "sha256_matched_pairs": sha256_matched_pairs,
        "filename_templates": list(filename_templates),
        "seeds": sorted(count_by_seed),
        "folds": [start_fold, end_fold],
        "rounds": rounds,
    }


def read_seed_fold_round_group_counts(
    document_csv,
    start_fold,
    end_fold,
    rounds,
):
    """保留 seed 維度，統計每個 seed × Fold × Round 的 S₁～S₅。"""

    counts = defaultdict(lambda: {"s1_n": 0, "s2_n": 0, "s3_n": 0})
    source_k_values = set()
    with document_csv.open(encoding="utf-8-sig") as file:
        for row in csv.DictReader(file):
            seed = int(row["seed"])
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
            key = (seed, fold, round_index)
            if row["query_emotion_exact_s0"] == "True":
                field = (
                    "s1_n"
                    if int(row["category_mismatch_count"]) == 0
                    else "s2_n"
                )
            else:
                field = "s3_n"
            counts[key][field] += 1

    assert len(source_k_values) == 1, f"source k 不唯一: {source_k_values}"
    return counts, source_k_values.pop()


def load_pair_best_rounds(experiments, start_fold, end_fold, rounds):
    """讀取最終測試 Pair-best checkpoint 在每個 seed × Fold 的來源輪次。"""

    pattern = re.compile(r"stage:\s*self_training_round_(\d+)")
    best_round_by_unit = {}
    rows = []
    for experiment in experiments:
        seed = experiment["seed"]
        for fold in range(start_fold, end_fold + 1):
            checkpoint_path = (
                experiment["experiment_dir"]
                / f"fold{fold}_best_val_checkpoint_pair.txt"
            )
            match = pattern.search(checkpoint_path.read_text(encoding="utf-8"))
            assert match, f"找不到 Pair-best round: {checkpoint_path}"
            best_round = int(match.group(1))
            assert 1 <= best_round <= rounds, (
                f"Pair-best round 超出 1～{rounds}: {checkpoint_path}"
            )
            best_round_by_unit[(seed, fold)] = best_round
            rows.append(
                {
                    "seed": seed,
                    "fold": fold,
                    "pair_best_round": best_round,
                    "checkpoint_metadata": str(checkpoint_path),
                }
            )
    return best_round_by_unit, rows


def aggregate_cumulative_counts_by_fold(
    counts,
    best_round_by_unit,
    experiments,
    start_fold,
    end_fold,
):
    """累積 R1 至各 seed 的 Pair-best round，再於同一 Fold 合併三個 seed。"""

    fold_rows = []
    detail_rows = []
    for fold in range(start_fold, end_fold + 1):
        fold_counts = {"s1_n": 0, "s2_n": 0, "s3_n": 0}
        for experiment in experiments:
            seed = experiment["seed"]
            best_round = best_round_by_unit[(seed, fold)]
            seed_counts = {"s1_n": 0, "s2_n": 0, "s3_n": 0}
            for round_index in range(1, best_round + 1):
                unit_counts = counts[(seed, fold, round_index)]
                assert sum(unit_counts.values()) > 0, (
                    f"seed {seed} Fold {fold} Round {round_index} 沒有 S₁～S₅ 資料"
                )
                for field in seed_counts:
                    seed_counts[field] += unit_counts[field]
                    fold_counts[field] += unit_counts[field]

            seed_s4_n = seed_counts["s1_n"] + seed_counts["s2_n"]
            seed_s5_n = seed_s4_n + seed_counts["s3_n"]
            detail_rows.append(
                {
                    "seed": seed,
                    "fold": fold,
                    "pair_best_round": best_round,
                    **seed_counts,
                    "s4_n": seed_s4_n,
                    "s5_n": seed_s5_n,
                }
            )

        s4_n = fold_counts["s1_n"] + fold_counts["s2_n"]
        s5_n = s4_n + fold_counts["s3_n"]
        fold_rows.append(
            {
                "fold": fold,
                **fold_counts,
                "s4_n": s4_n,
                "s5_n": s5_n,
            }
        )
    return fold_rows, detail_rows


def load_fold_pooled_test_emotion_f1(experiments, start_fold, end_fold):
    """先合併同折三個 seed 的測試計數，再重算 Emotion micro F1。"""

    rows = []
    pattern = TASK_PATTERNS["emotion"]
    for fold in range(start_fold, end_fold + 1):
        true_positive = 0
        predicted_positive = 0
        gold_positive = 0
        row = {"fold": fold}

        for experiment in experiments:
            result_path = (
                experiment["experiment_dir"] / f"fold{fold}_test_evaluation.txt"
            )
            match = pattern.search(result_path.read_text(encoding="utf-8"))
            assert match, f"找不到測試集 Emotion 評估計數: {result_path}"
            tp, predicted, gold = (int(float(value)) for value in match.groups())
            true_positive += tp
            predicted_positive += predicted
            gold_positive += gold
            row[f"seed{experiment['seed']}_emotion_tp"] = tp
            row[f"seed{experiment['seed']}_emotion_predicted"] = predicted
            row[f"seed{experiment['seed']}_emotion_gold"] = gold

        assert predicted_positive > 0 and gold_positive > 0
        row.update(
            {
                "emotion_tp": true_positive,
                "emotion_predicted": predicted_positive,
                "emotion_gold": gold_positive,
                "emotion_f1": 2
                * true_positive
                / (predicted_positive + gold_positive),
            }
        )
        rows.append(row)
    return rows


def plot_fold_counts_and_test_f1(
    count_rows,
    test_rows,
    output_path,
    background,
):
    """上下圖皆以 Fold 為橫軸，配對累積分組數量與最終測試 F1。"""

    configure_fonts(font_size=15)
    plt.rcParams["font.family"] = [*plt.rcParams["font.family"], "DejaVu Serif"]
    facecolor = "#DDDDDD" if background == "gray" else "white"
    fig, (count_ax, f1_ax) = plt.subplots(
        2,
        1,
        figsize=(12, 8),
        dpi=180,
        sharex=True,
        gridspec_kw={"height_ratios": (1.35, 1)},
    )
    fig.patch.set_facecolor(facecolor)
    count_ax.set_facecolor(facecolor)
    f1_ax.set_facecolor(facecolor)

    folds = [row["fold"] for row in count_rows]
    assert folds == [row["fold"] for row in test_rows]
    count_fields = ("s1_n", "s2_n", "s3_n", "s4_n", "s5_n")
    bar_width = 0.14
    center = (len(count_fields) - 1) / 2
    for index, field in enumerate(count_fields):
        values = [row[field] for row in count_rows]
        positions = [
            fold + (index - center) * bar_width
            for fold in folds
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
        label_padding = (2, 2, 12, 22, 2)[index]
        annotations = count_ax.bar_label(
            bars,
            labels=[str(value) for value in values],
            padding=label_padding,
            fontsize=8.5,
        )
        label_dx = (index - center) * 2.0
        for annotation in annotations:
            annotation.set_position((label_dx, label_padding))

    count_ax.set_ylim(0, max(row["s5_n"] for row in count_rows) * 1.23)
    count_ax.set_ylabel("各 seed 分別累積至其 Pair-best 回合後加總（筆次）")
    count_ax.set_xticks(folds)
    count_ax.legend(loc="upper center", ncol=5, frameon=True)

    f1_values = [100 * row["emotion_f1"] for row in test_rows]
    f1_ax.plot(
        folds,
        f1_values,
        color=F1_COLORS["emotion_f1"],
        marker="o",
        linewidth=2.2,
        markersize=7,
        label="Emotion F1",
        zorder=3,
    )
    for fold, value in zip(folds, f1_values):
        f1_ax.annotate(
            f"{value:.2f}",
            (fold, value),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            color=F1_COLORS["emotion_f1"],
        )
    f1_ax.set_ylim(0, 100)
    f1_ax.set_ylabel("測試集 Emotion F1（%）")
    f1_ax.set_xlabel("Fold")
    f1_ax.set_xticks(folds)
    f1_ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=1,
        frameon=True,
    )

    for axis in (count_ax, f1_ax):
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

    fig.suptitle(
        "UECA-NeST-Semo-Tpair-Fbest 各 Fold 至 Pair-best 回合的累積分組數量與測試集 Emotion F1",
        y=0.985,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.975), h_pad=1.7, pad=0.8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor=facecolor)
    plt.close(fig)


def main():
    args = parse_args()
    tag = f"posthoc_k{args.analysis_k}"
    document_csv = args.analysis_dir / f"document_level_{tag}.csv"
    assert document_csv.exists(), f"找不到逐文檔分析檔: {document_csv}"
    assert args.count_summary.exists(), f"找不到分組數量 summary: {args.count_summary}"
    assert args.test_summary.exists(), f"找不到測試 summary: {args.test_summary}"

    counts, source_k = read_seed_fold_round_group_counts(
        document_csv,
        args.start_fold,
        args.end_fold,
        args.rounds,
    )

    count_experiments = parse_experiments(
        args.count_summary,
        args.experiments_root,
    )
    experiments = parse_experiments(args.test_summary, args.experiments_root)
    trajectory_verification = verify_matching_training_trajectory(
        count_experiments,
        experiments,
        args.start_fold,
        args.end_fold,
        args.rounds,
    )
    pair_best_rounds, pair_best_round_rows = load_pair_best_rounds(
        experiments,
        args.start_fold,
        args.end_fold,
        args.rounds,
    )
    fold_count_rows, seed_fold_count_rows = aggregate_cumulative_counts_by_fold(
        counts,
        pair_best_rounds,
        experiments,
        args.start_fold,
        args.end_fold,
    )
    test_rows = load_fold_pooled_test_emotion_f1(
        experiments,
        args.start_fold,
        args.end_fold,
    )

    output_dir = args.png_dir / f"fold_round_test_f1_{tag}_{args.background}"
    output_path = output_dir / (
        f"UECA_NeST_{tag}_fold{args.start_fold}-{args.end_fold}_round_S123_"
        f"test_emotion_F1_pooled_micro_{args.background}.png"
    )
    plot_fold_counts_and_test_f1(
        fold_count_rows,
        test_rows,
        output_path,
        args.background,
    )

    count_csv_path = args.analysis_dir / (
        f"UECA_NeST_{tag}_fold{args.start_fold}-{args.end_fold}_"
        "pair_best_cumulative_S123_counts.csv"
    )
    detail_count_csv_path = args.analysis_dir / (
        f"UECA_NeST_{tag}_seed_fold{args.start_fold}-{args.end_fold}_"
        "pair_best_cumulative_S123_counts.csv"
    )
    pair_best_round_csv_path = args.analysis_dir / (
        f"UECA_NeST_{tag}_seed_fold{args.start_fold}-{args.end_fold}_"
        "pair_best_rounds.csv"
    )
    test_csv_path = args.analysis_dir / (
        f"UECA_NeST_{tag}_fold{args.start_fold}-{args.end_fold}_"
        "test_emotion_F1_pooled_micro.csv"
    )
    write_csv(count_csv_path, fold_count_rows)
    write_csv(detail_count_csv_path, seed_fold_count_rows)
    write_csv(pair_best_round_csv_path, pair_best_round_rows)
    write_csv(test_csv_path, test_rows)

    metadata_path = args.analysis_dir / (
        f"UECA_NeST_{tag}_fold{args.start_fold}-{args.end_fold}_"
        "round_S123_test_emotion_F1_pooled_micro_metadata.json"
    )
    metadata = {
        "figure": str(output_path.resolve()),
        "count_document_csv": str(document_csv.resolve()),
        "count_summary": str(args.count_summary.resolve()),
        "test_summary": str(args.test_summary.resolve()),
        "experiments_root": str(args.experiments_root.resolve()),
        "source_k": source_k,
        "analysis_k": args.analysis_k,
        "count_scope": (
            "for each seed and Fold, sum category-valid NeST-selected document "
            "observations from Round 1 through that seed-fold's Pair-best round; "
            "then sum the three seeds within each Fold"
        ),
        "count_csv": str(count_csv_path.resolve()),
        "seed_fold_count_csv": str(detail_count_csv_path.resolve()),
        "pair_best_round_csv": str(pair_best_round_csv_path.resolve()),
        "pair_best_round_source": "fold{fold}_best_val_checkpoint_pair.txt",
        "test_scope": (
            "each Fold pools final self_training Pair-best test Emotion counts "
            "over three seeds"
        ),
        "emotion_f1_formula": "2 * sum(TP) / (sum(predicted) + sum(gold))",
        "training_trajectory_verification": trajectory_verification,
    }
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)

    total_tp = sum(row["emotion_tp"] for row in test_rows)
    total_predicted = sum(row["emotion_predicted"] for row in test_rows)
    total_gold = sum(row["emotion_gold"] for row in test_rows)
    overall_f1 = 2 * total_tp / (total_predicted + total_gold)
    print(f"各折至 Pair-best 回合累積 S₁～S₅: {count_csv_path}")
    print(f"各 seed × Fold 累積明細: {detail_count_csv_path}")
    print(f"各 seed × Fold Pair-best round: {pair_best_round_csv_path}")
    print(f"逐折測試 Emotion pooled-micro F1: {test_csv_path}")
    print(f"來源與軌跡核對: {metadata_path}")
    print(f"測試圖: {output_path}")
    print(
        "Fold 1～10 整體測試 Emotion pooled-micro F1: "
        f"{100 * overall_f1:.2f}% "
        f"(TP={total_tp}, predicted={total_predicted}, gold={total_gold})"
    )


if __name__ == "__main__":
    main()
