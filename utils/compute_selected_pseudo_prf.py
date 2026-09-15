#!/usr/bin/env python3
"""統計 selected pseudo labels 的 Emotion / Cause / Pair PRF。"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev


SUMMARY_EXPERIMENT_PATTERN = re.compile(r"^\s*-\s*(prompt_ECPE_few_shot_ST_[^\s]+)\s*$")
SEED_PATTERN = re.compile(r"seed(\d+)")


@dataclass
class MetricCounts:
    tp: int = 0
    pred_total: int = 0
    gt_total: int = 0

    def precision(self) -> float:
        return 0.0 if self.pred_total == 0 else self.tp / self.pred_total

    def recall(self) -> float:
        return 0.0 if self.gt_total == 0 else self.tp / self.gt_total

    def f1(self) -> float:
        precision = self.precision()
        recall = self.recall()
        return 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)


@dataclass
class RoundSeedResult:
    seed_name: str
    round_id: int
    emotion: MetricCounts
    cause: MetricCounts
    pair_m1: MetricCounts
    pair_m2: MetricCounts
    pair_m3: MetricCounts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="統計 selected pseudo labels 在各輪自訓練中的 Emotion / Cause / Pair PRF"
    )
    parser.add_argument("--summary", required=True, help="multi-seed summary 檔案路徑")
    parser.add_argument("--experiments-root", required=True, help="實驗資料夾根目錄")
    parser.add_argument("--fold-start", type=int, default=1, help="起始 fold（包含）")
    parser.add_argument("--fold-end", type=int, default=10, help="結束 fold（包含）")
    parser.add_argument("--round-start", type=int, default=1, help="起始 round（包含）")
    parser.add_argument("--round-end", type=int, default=5, help="結束 round（包含）")
    parser.add_argument("--output", required=True, help="輸出報告檔案路徑")
    return parser.parse_args()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def read_json(path: Path):
    return json.loads(read_text(path))


def extract_experiment_names(summary_path: Path) -> list[str]:
    experiment_names: list[str] = []
    for line in read_text(summary_path).splitlines():
        match = SUMMARY_EXPERIMENT_PATTERN.match(line)
        if match:
            experiment_names.append(match.group(1))
    if not experiment_names:
        raise ValueError(f"在 summary 檔案中找不到任何實驗資料夾名稱：{summary_path}")
    return experiment_names


def extract_seed(experiment_name: str) -> str:
    match = SEED_PATTERN.search(experiment_name)
    if match is None:
        raise ValueError(f"無法從實驗名稱中解析 seed：{experiment_name}")
    return f"seed{match.group(1)}"


def seed_sort_key(seed_name: str) -> tuple[int, str]:
    match = SEED_PATTERN.search(seed_name)
    if match:
        return (int(match.group(1)), seed_name)
    return (10**9, seed_name)


def resolve_experiment_map(summary_path: Path, experiments_root: Path) -> dict[str, Path]:
    experiment_map: dict[str, Path] = {}
    for experiment_name in extract_experiment_names(summary_path):
        seed_name = extract_seed(experiment_name)
        experiment_dir = experiments_root / experiment_name
        if not experiment_dir.is_dir():
            raise FileNotFoundError(f"找不到實驗資料夾：{experiment_dir}")
        if seed_name in experiment_map:
            raise ValueError(f"summary 中出現重複 seed：{seed_name}")
        experiment_map[seed_name] = experiment_dir
    return experiment_map


def find_single_pseudo_results_dir(experiment_dir: Path) -> Path:
    pseudo_dirs = sorted(path for path in experiment_dir.glob("pseudo_results_*") if path.is_dir())
    if not pseudo_dirs:
        raise FileNotFoundError(f"在實驗資料夾中找不到 pseudo_results_* 目錄：{experiment_dir}")
    if len(pseudo_dirs) > 1:
        raise ValueError(
            f"實驗資料夾中存在多個 pseudo_results_* 目錄：{experiment_dir}，"
            f"找到 {[path.name for path in pseudo_dirs]}"
        )
    return pseudo_dirs[0]


def ensure_label_lists(record: dict, json_path: Path) -> tuple[list[str], list[str]]:
    pseudo_label_tokens = record.get("pseudo_label_tokens")
    gt_label_tokens = record.get("gt_label_tokens")
    if not isinstance(pseudo_label_tokens, list) or not isinstance(gt_label_tokens, list):
        raise ValueError(
            f"selected pseudo JSON 缺少 pseudo_label_tokens 或 gt_label_tokens："
            f"{json_path} doc_id={record.get('doc_id')}"
        )
    if len(pseudo_label_tokens) != len(gt_label_tokens):
        raise ValueError(
            f"標籤長度不一致：{json_path} doc_id={record.get('doc_id')} "
            f"pseudo={len(pseudo_label_tokens)} gt={len(gt_label_tokens)}"
        )
    if len(pseudo_label_tokens) % 3 != 0:
        raise ValueError(
            f"標籤長度無法被 3 整除：{json_path} doc_id={record.get('doc_id')} len={len(pseudo_label_tokens)}"
        )
    return pseudo_label_tokens, gt_label_tokens


def build_doc_rows(record: dict, json_path: Path) -> tuple[list[tuple[str, str, str]], list[tuple[str, str, str]]]:
    pseudo_label_tokens, gt_label_tokens = ensure_label_lists(record, json_path)
    pseudo_rows = []
    gt_rows = []
    for index in range(0, len(pseudo_label_tokens), 3):
        pseudo_rows.append(tuple(pseudo_label_tokens[index:index + 3]))
        gt_rows.append(tuple(gt_label_tokens[index:index + 3]))
    return gt_rows, pseudo_rows


def accumulate_metric(counts: MetricCounts, tp: int, pred_total: int, gt_total: int) -> None:
    counts.tp += tp
    counts.pred_total += pred_total
    counts.gt_total += gt_total


def accumulate_record_metrics(result: RoundSeedResult, record: dict, json_path: Path) -> None:
    doc_id = str(record.get("doc_id"))
    gt_rows, pseudo_rows = build_doc_rows(record, json_path)

    pred_emo_map: dict[tuple[str, str], str] = {}
    gt_emo_map: dict[tuple[str, str], str] = {}

    for clause_idx, (gt_row, pred_row) in enumerate(zip(gt_rows, pseudo_rows), start=1):
        clause_key = str(clause_idx)
        pred_emo_map[(doc_id, clause_key)] = pred_row[0]
        gt_emo_map[(doc_id, clause_key)] = gt_row[0]

    for gt_row, pred_row in zip(gt_rows, pseudo_rows):
        gt_emo, gt_cause, gt_pair = gt_row
        pred_emo, pred_cause, pred_pair = pred_row

        accumulate_metric(
            result.emotion,
            tp=1 if (pred_emo == "是" and gt_emo == "是") else 0,
            pred_total=1 if pred_emo == "是" else 0,
            gt_total=1 if gt_emo == "是" else 0,
        )
        accumulate_metric(
            result.cause,
            tp=1 if (pred_cause == "是" and gt_cause == "是") else 0,
            pred_total=1 if pred_cause == "是" else 0,
            gt_total=1 if gt_cause == "是" else 0,
        )

        pair_gt_total = 1 if gt_pair != "无" else 0

        accumulate_metric(
            result.pair_m1,
            tp=1 if (gt_pair != "无" and pred_pair == gt_pair) else 0,
            pred_total=1 if pred_pair != "无" else 0,
            gt_total=pair_gt_total,
        )
        accumulate_metric(
            result.pair_m2,
            tp=1 if (pred_pair != "无" and pred_cause == "是" and pred_pair == gt_pair and gt_cause == "是") else 0,
            pred_total=1 if (pred_pair != "无" and pred_cause == "是") else 0,
            gt_total=pair_gt_total,
        )

        pair3_pred_positive = (
            pred_cause == "是"
            and pred_pair != "无"
            and pred_emo_map.get((doc_id, pred_pair), "非") == "是"
        )
        pair3_tp = (
            pair3_pred_positive
            and gt_cause == "是"
            and pred_pair == gt_pair
            and gt_emo_map.get((doc_id, gt_pair), "非") == "是"
        )
        accumulate_metric(
            result.pair_m3,
            tp=1 if pair3_tp else 0,
            pred_total=1 if pair3_pred_positive else 0,
            gt_total=pair_gt_total,
        )


def collect_round_seed_results(
    summary_path: Path,
    experiments_root: Path,
    fold_start: int,
    fold_end: int,
    round_start: int,
    round_end: int,
) -> dict[int, list[RoundSeedResult]]:
    experiment_map = resolve_experiment_map(summary_path, experiments_root)
    round_results: dict[int, list[RoundSeedResult]] = {}

    for seed_name in sorted(experiment_map.keys(), key=seed_sort_key):
        pseudo_dir = find_single_pseudo_results_dir(experiment_map[seed_name])
        for round_id in range(round_start, round_end + 1):
            result = RoundSeedResult(
                seed_name=seed_name,
                round_id=round_id,
                emotion=MetricCounts(),
                cause=MetricCounts(),
                pair_m1=MetricCounts(),
                pair_m2=MetricCounts(),
                pair_m3=MetricCounts(),
            )
            found_any = False
            for fold in range(fold_start, fold_end + 1):
                json_path = pseudo_dir / f"pseudo_labeled_samples_fold{fold}_round{round_id}.json"
                if not json_path.is_file():
                    continue
                found_any = True
                records = read_json(json_path)
                if not isinstance(records, list):
                    raise ValueError(f"selected pseudo JSON 最外層不是列表：{json_path}")
                for record in records:
                    accumulate_record_metrics(result, record, json_path)
            if found_any:
                round_results.setdefault(round_id, []).append(result)

    return round_results


def format_percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def format_metric_line(metric_name: str, counts: MetricCounts) -> list[str]:
    return [
        f"{metric_name}:",
        f"  TP={counts.tp}, Pred={counts.pred_total}, GT={counts.gt_total}",
        f"  Precision={format_percent(counts.precision())}  Recall={format_percent(counts.recall())}  F1={format_percent(counts.f1())}",
    ]


def build_average_counts(results: list[RoundSeedResult], field_name: str) -> MetricCounts:
    average_counts = MetricCounts()
    for result in results:
        metric_counts = getattr(result, field_name)
        average_counts.tp += metric_counts.tp
        average_counts.pred_total += metric_counts.pred_total
        average_counts.gt_total += metric_counts.gt_total
    return average_counts


def build_mean_std_line(results: list[RoundSeedResult], field_name: str) -> str:
    precisions = [getattr(result, field_name).precision() for result in results]
    recalls = [getattr(result, field_name).recall() for result in results]
    f1_scores = [getattr(result, field_name).f1() for result in results]

    def summary(values: list[float]) -> str:
        if len(values) == 1:
            return format_percent(values[0])
        return f"{format_percent(mean(values))} ± {format_percent(stdev(values))}"

    return (
        f"  Seed 平均 Precision={summary(precisions)}  "
        f"Recall={summary(recalls)}  F1={summary(f1_scores)}"
    )


def build_report(
    summary_path: Path,
    experiments_root: Path,
    fold_start: int,
    fold_end: int,
    round_start: int,
    round_end: int,
    round_results: dict[int, list[RoundSeedResult]],
) -> str:
    lines: list[str] = []
    lines.append("=" * 70)
    lines.append("Selected Pseudo Labels PRF 統計")
    lines.append("=" * 70)
    lines.append(f"Summary 檔案: {summary_path}")
    lines.append(f"實驗根目錄: {experiments_root}")
    lines.append(f"統計 fold 範圍: fold{fold_start} ~ fold{fold_end}")
    lines.append(f"統計 round 範圍: round{round_start} ~ round{round_end}")
    lines.append("")
    lines.append("實驗目錄:")
    for experiment_name in extract_experiment_names(summary_path):
        lines.append(f"- {experiment_name}")

    metric_fields = [
        ("emotion", "Emotion"),
        ("cause", "Cause"),
        ("pair_m1", "Pair (m1)"),
        ("pair_m2", "Pair (m2)"),
        ("pair_m3", "Pair (m3)"),
    ]

    for round_id in range(round_start, round_end + 1):
        results = sorted(round_results.get(round_id, []), key=lambda item: seed_sort_key(item.seed_name))
        lines.append("")
        lines.append("-" * 70)
        lines.append(f"Round {round_id}")
        lines.append("-" * 70)

        if not results:
            lines.append("找不到任何 selected pseudo labels 資料。")
            continue

        for result in results:
            lines.append("")
            lines.append(f"[{result.seed_name}]")
            for field_name, label in metric_fields:
                lines.extend(format_metric_line(label, getattr(result, field_name)))

        lines.append("")
        lines.append("[三個 seed 平均]")
        for field_name, label in metric_fields:
            average_counts = build_average_counts(results, field_name)
            lines.extend(format_metric_line(label, average_counts))
            lines.append(build_mean_std_line(results, field_name))

    lines.append("")
    lines.append("=" * 70)
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    summary_path = Path(args.summary)
    experiments_root = Path(args.experiments_root)
    output_path = Path(args.output)

    round_results = collect_round_seed_results(
        summary_path=summary_path,
        experiments_root=experiments_root,
        fold_start=args.fold_start,
        fold_end=args.fold_end,
        round_start=args.round_start,
        round_end=args.round_end,
    )

    report = build_report(
        summary_path=summary_path,
        experiments_root=experiments_root,
        fold_start=args.fold_start,
        fold_end=args.fold_end,
        round_start=args.round_start,
        round_end=args.round_end,
        round_results=round_results,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"已輸出報告：{output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())