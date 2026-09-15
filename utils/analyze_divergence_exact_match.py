#!/usr/bin/env python3
"""Analyze low-vs-high divergence exact-match rates on unlabeled documents."""

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
class DocRecord:
    doc_id: str
    divergence_score: float
    exact_match: bool
    selected: bool


@dataclass
class GroupStats:
    exact_docs: int
    total_docs: int
    selected_docs: int
    mean_score: float

    @property
    def exact_rate(self) -> float:
        return 0.0 if self.total_docs == 0 else self.exact_docs / self.total_docs


@dataclass
class UnitResult:
    summary_label: str
    seed_name: str
    fold_id: int
    round_id: int
    low: GroupStats
    high: GroupStats

    @property
    def rate_gap(self) -> float:
        return self.low.exact_rate - self.high.exact_rate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare low-vs-high divergence groups by full-document pseudo exact match rate."
    )
    parser.add_argument("--summaries", nargs="+", default=None, help="One or more multi-seed summary files")
    parser.add_argument(
        "--summaries-file",
        default=None,
        help="Text file containing summary paths, one per line",
    )
    parser.add_argument("--experiments-root", required=True, help="Experiments root directory")
    parser.add_argument("--fold-start", type=int, default=1, help="Start fold (inclusive)")
    parser.add_argument("--fold-end", type=int, default=10, help="End fold (inclusive)")
    parser.add_argument("--round-start", type=int, default=1, help="Start round (inclusive)")
    parser.add_argument("--round-end", type=int, default=5, help="End round (inclusive)")
    parser.add_argument(
        "--quantile",
        type=float,
        default=0.25,
        help="Bottom/top quantile used to define low/high divergence groups (default: 0.25)",
    )
    parser.add_argument("--output", required=True, help="Output report path")
    return parser.parse_args()


def resolve_summary_paths(args: argparse.Namespace) -> list[Path]:
    summary_paths: list[Path] = []

    if args.summaries:
        summary_paths.extend(Path(path) for path in args.summaries)

    if args.summaries_file:
        summaries_file = Path(args.summaries_file)
        for line in read_text(summaries_file).splitlines():
            stripped = line.strip()
            if stripped:
                summary_paths.append(Path(stripped))

    if not summary_paths:
        raise ValueError("Please provide --summaries or --summaries-file")

    return summary_paths


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def read_json(path: Path):
    return json.loads(read_text(path))


def print_progress(message: str) -> None:
    print(message, flush=True)


def extract_experiment_names(summary_path: Path) -> list[str]:
    experiment_names: list[str] = []
    for line in read_text(summary_path).splitlines():
        match = SUMMARY_EXPERIMENT_PATTERN.match(line)
        if match:
            experiment_names.append(match.group(1))
    if not experiment_names:
        raise ValueError(f"No experiment directories found in summary: {summary_path}")
    return experiment_names


def extract_seed(experiment_name: str) -> str:
    match = SEED_PATTERN.search(experiment_name)
    if match is None:
        raise ValueError(f"Cannot parse seed from experiment name: {experiment_name}")
    return f"seed{match.group(1)}"


def seed_sort_key(seed_name: str) -> tuple[int, str]:
    match = SEED_PATTERN.search(seed_name)
    if match:
        return (int(match.group(1)), seed_name)
    return (10**9, seed_name)


def summary_label(summary_path: Path) -> str:
    name = summary_path.name
    for suffix in ("_CE_summary.txt", "_EC_summary.txt", "_summary.txt"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return summary_path.stem


def resolve_experiment_map(summary_path: Path, experiments_root: Path) -> dict[str, Path]:
    experiment_map: dict[str, Path] = {}
    for experiment_name in extract_experiment_names(summary_path):
        seed_name = extract_seed(experiment_name)
        experiment_dir = experiments_root / experiment_name
        if not experiment_dir.is_dir():
            raise FileNotFoundError(f"Experiment directory not found: {experiment_dir}")
        if seed_name in experiment_map:
            raise ValueError(f"Duplicate seed in summary: {seed_name}")
        experiment_map[seed_name] = experiment_dir
    return experiment_map


def find_single_pseudo_results_dir(experiment_dir: Path) -> Path:
    pseudo_dirs = sorted(path for path in experiment_dir.glob("pseudo_results_*") if path.is_dir())
    if not pseudo_dirs:
        raise FileNotFoundError(f"No pseudo_results_* directory found under: {experiment_dir}")
    if len(pseudo_dirs) > 1:
        raise ValueError(
            f"Multiple pseudo_results_* directories found under {experiment_dir}: "
            f"{[path.name for path in pseudo_dirs]}"
        )
    return pseudo_dirs[0]


def exact_match_from_prediction(record: dict, path: Path) -> bool:
    if not record.get("ground_truth_available", False):
        raise ValueError(f"Ground truth unavailable in file: {path} doc_id={record.get('doc_id')}")

    pseudo_label_tokens = record.get("pseudo_label_tokens")
    gt_label_tokens = record.get("gt_label_tokens")
    if not isinstance(pseudo_label_tokens, list) or not isinstance(gt_label_tokens, list):
        raise ValueError(f"Missing token lists in file: {path} doc_id={record.get('doc_id')}")
    return pseudo_label_tokens == gt_label_tokens


def load_doc_records(divergence_path: Path, prediction_path: Path) -> list[DocRecord]:
    divergence_data = read_json(divergence_path)
    prediction_data = read_json(prediction_path)

    prediction_map: dict[str, dict] = {}
    for record in prediction_data:
        doc_id = str(record.get("doc_id"))
        if doc_id in prediction_map:
            raise ValueError(f"Duplicate doc_id in prediction file: {prediction_path} doc_id={doc_id}")
        prediction_map[doc_id] = record

    records: list[DocRecord] = []
    for sample in divergence_data.get("samples", []):
        doc_id = str(sample.get("doc_id"))
        if doc_id not in prediction_map:
            raise ValueError(f"doc_id {doc_id} from divergence file not found in prediction file: {prediction_path}")
        records.append(
            DocRecord(
                doc_id=doc_id,
                divergence_score=float(sample["divergence_score"]),
                exact_match=exact_match_from_prediction(prediction_map[doc_id], prediction_path),
                selected=bool(sample.get("selected", False)),
            )
        )

    if not records:
        raise ValueError(f"No samples found in divergence file: {divergence_path}")
    return records


def summarize_group(records: list[DocRecord]) -> GroupStats:
    exact_docs = sum(1 for record in records if record.exact_match)
    selected_docs = sum(1 for record in records if record.selected)
    mean_score = mean(record.divergence_score for record in records)
    return GroupStats(
        exact_docs=exact_docs,
        total_docs=len(records),
        selected_docs=selected_docs,
        mean_score=mean_score,
    )


def analyze_unit(
    summary_name: str,
    seed_name: str,
    pseudo_dir: Path,
    fold_id: int,
    round_id: int,
    quantile: float,
) -> UnitResult:
    divergence_path = pseudo_dir / f"nest_divergence_scores_fold{fold_id}_round{round_id}.json"
    prediction_path = pseudo_dir / f"all_unlabeled_pseudo_predictions_fold{fold_id}_round{round_id}.json"

    if not divergence_path.is_file():
        raise FileNotFoundError(f"Missing divergence file: {divergence_path}")
    if not prediction_path.is_file():
        raise FileNotFoundError(f"Missing prediction file: {prediction_path}")

    records = sorted(load_doc_records(divergence_path, prediction_path), key=lambda item: item.divergence_score)
    group_size = int(len(records) * quantile)
    if group_size <= 0:
        raise ValueError(f"Quantile {quantile} is too small for {len(records)} records in {divergence_path}")
    if group_size * 2 > len(records):
        raise ValueError(f"Quantile {quantile} is too large for {len(records)} records in {divergence_path}")

    low_group = records[:group_size]
    high_group = records[-group_size:]
    return UnitResult(
        summary_label=summary_name,
        seed_name=seed_name,
        fold_id=fold_id,
        round_id=round_id,
        low=summarize_group(low_group),
        high=summarize_group(high_group),
    )


def collect_results_for_summary(
    summary_path: Path,
    experiments_root: Path,
    fold_start: int,
    fold_end: int,
    round_start: int,
    round_end: int,
    quantile: float,
) -> list[UnitResult]:
    experiment_map = resolve_experiment_map(summary_path, experiments_root)
    name = summary_label(summary_path)
    unit_results: list[UnitResult] = []
    total_units = len(experiment_map) * (fold_end - fold_start + 1) * (round_end - round_start + 1)
    current_unit = 0

    print_progress(f"[{name}] start: {len(experiment_map)} seeds, {total_units} units")

    for seed_name in sorted(experiment_map.keys(), key=seed_sort_key):
        pseudo_dir = find_single_pseudo_results_dir(experiment_map[seed_name])
        print_progress(f"[{name}] seed={seed_name} pseudo_dir={pseudo_dir.name}")
        for fold_id in range(fold_start, fold_end + 1):
            for round_id in range(round_start, round_end + 1):
                current_unit += 1
                print_progress(
                    f"[{name}] {current_unit}/{total_units} -> {seed_name} fold{fold_id} round{round_id}"
                )
                unit_results.append(analyze_unit(name, seed_name, pseudo_dir, fold_id, round_id, quantile))

    print_progress(f"[{name}] done: collected {len(unit_results)} units")

    return unit_results


def safe_mean(values: list[float]) -> float:
    return 0.0 if not values else mean(values)


def safe_std(values: list[float]) -> float:
    return 0.0 if len(values) <= 1 else stdev(values)


def format_percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def build_overall_lines(results: list[UnitResult]) -> list[str]:
    low_exact_total = sum(result.low.exact_docs for result in results)
    low_doc_total = sum(result.low.total_docs for result in results)
    high_exact_total = sum(result.high.exact_docs for result in results)
    high_doc_total = sum(result.high.total_docs for result in results)

    low_rate_pooled = 0.0 if low_doc_total == 0 else low_exact_total / low_doc_total
    high_rate_pooled = 0.0 if high_doc_total == 0 else high_exact_total / high_doc_total

    low_rates = [result.low.exact_rate for result in results]
    high_rates = [result.high.exact_rate for result in results]
    gaps = [result.rate_gap for result in results]

    low_better = sum(1 for gap in gaps if gap > 0)
    tied = sum(1 for gap in gaps if gap == 0)
    high_better = sum(1 for gap in gaps if gap < 0)

    lines = []
    lines.append("整體摘要:")
    lines.append(
        f"  pooled low:  exact={low_exact_total}/{low_doc_total}  rate={format_percent(low_rate_pooled)}"
    )
    lines.append(
        f"  pooled high: exact={high_exact_total}/{high_doc_total}  rate={format_percent(high_rate_pooled)}"
    )
    lines.append(f"  pooled rate gap (low-high) = {format_percent(low_rate_pooled - high_rate_pooled)}")
    lines.append(
        "  macro average low rate  = "
        f"{format_percent(safe_mean(low_rates))} ± {format_percent(safe_std(low_rates))}"
    )
    lines.append(
        "  macro average high rate = "
        f"{format_percent(safe_mean(high_rates))} ± {format_percent(safe_std(high_rates))}"
    )
    lines.append(
        "  macro average gap       = "
        f"{format_percent(safe_mean(gaps))} ± {format_percent(safe_std(gaps))}"
    )
    lines.append(
        f"  units with low > high: {low_better}/{len(results)}; tie: {tied}; low < high: {high_better}/{len(results)}"
    )
    return lines


def build_round_summary_lines(results: list[UnitResult]) -> list[str]:
    round_ids = sorted({result.round_id for result in results})
    lines = []
    lines.append("按 round 的 macro 平均:")
    header = "  round | low_rate | high_rate | gap"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for round_id in round_ids:
        round_results = [result for result in results if result.round_id == round_id]
        low_rate = safe_mean([result.low.exact_rate for result in round_results])
        high_rate = safe_mean([result.high.exact_rate for result in round_results])
        gap = safe_mean([result.rate_gap for result in round_results])
        lines.append(
            f"  {round_id:>5} | {format_percent(low_rate):>8} | {format_percent(high_rate):>9} | {format_percent(gap):>8}"
        )
    return lines


def build_thesis_lines(results: list[UnitResult], label: str) -> list[str]:
    low_rates = [result.low.exact_rate for result in results]
    high_rates = [result.high.exact_rate for result in results]
    gaps = [result.rate_gap for result in results]
    low_better = sum(1 for gap in gaps if gap > 0)
    lines = []
    lines.append("論文可用摘要:")
    lines.append(
        f"  {label} 在 {len(results)} 個 seed-fold-round 單位中，"
        f"低散度文檔的全文偽標籤全對率平均為 {format_percent(safe_mean(low_rates))}，"
        f"高散度文檔平均為 {format_percent(safe_mean(high_rates))}，"
        f"平均差距為 {format_percent(safe_mean(gaps))}。"
    )
    lines.append(
        f"  其中共有 {low_better}/{len(results)} 個單位呈現 low > high，"
        "支持『低散度文檔的偽標籤全文全對比例高於高散度文檔』的觀察。"
    )
    return lines


def build_detail_lines(results: list[UnitResult]) -> list[str]:
    lines = []
    lines.append("詳細結果 (每個 seed / fold / round):")
    lines.append(
        "  seed   fold round | low_exact/total rate  low_selected mean_score | "
        "high_exact/total rate high_selected mean_score | gap"
    )
    lines.append("  " + "-" * 115)
    for result in sorted(results, key=lambda item: (seed_sort_key(item.seed_name), item.fold_id, item.round_id)):
        lines.append(
            f"  {result.seed_name:<6} {result.fold_id:>4} {result.round_id:>5} | "
            f"{result.low.exact_docs:>3}/{result.low.total_docs:<3} {format_percent(result.low.exact_rate):>8} "
            f"{result.low.selected_docs:>12} {result.low.mean_score:>10.4f} | "
            f"{result.high.exact_docs:>3}/{result.high.total_docs:<3} {format_percent(result.high.exact_rate):>8} "
            f"{result.high.selected_docs:>13} {result.high.mean_score:>10.4f} | "
            f"{format_percent(result.rate_gap):>8}"
        )
    return lines


def build_summary_section(summary_path: Path, experiments_root: Path, results: list[UnitResult]) -> list[str]:
    label = summary_label(summary_path)
    lines: list[str] = []
    lines.append("=" * 90)
    lines.append(label)
    lines.append("=" * 90)
    lines.append(f"summary file: {summary_path}")
    lines.append(f"experiments root: {experiments_root}")
    lines.append("experiment directories:")
    for experiment_name in extract_experiment_names(summary_path):
        lines.append(f"  - {experiment_name}")
    lines.append("")
    lines.extend(build_overall_lines(results))
    lines.append("")
    lines.extend(build_round_summary_lines(results))
    lines.append("")
    lines.extend(build_thesis_lines(results, label))
    lines.append("")
    lines.extend(build_detail_lines(results))
    lines.append("")
    return lines


def build_report(
    summary_paths: list[Path],
    experiments_root: Path,
    fold_start: int,
    fold_end: int,
    round_start: int,
    round_end: int,
    quantile: float,
    results_by_summary: dict[str, list[UnitResult]],
) -> str:
    lines: list[str] = []
    lines.append("低散度 vs 高散度 文檔的全文偽標籤全對率分析")
    lines.append("=" * 90)
    lines.append("分析設定:")
    lines.append(f"  - 實驗根目錄: {experiments_root}")
    lines.append(f"  - fold 範圍: fold{fold_start} ~ fold{fold_end}")
    lines.append(f"  - round 範圍: round{round_start} ~ round{round_end}")
    lines.append(f"  - low/high 定義: 各 seed-fold-round 內，最低/最高 {format_percent(quantile)} 散度文檔")
    lines.append("  - 文檔全對定義: pseudo_label_tokens 與 gt_label_tokens 完全一致")
    lines.append("  - 分析母體: all_unlabeled_pseudo_predictions 中所有可比較文檔")
    lines.append("  - low/high 表中的 selected_docs 僅作為補充背景，不改變分組方式")
    lines.append("")
    lines.append("summary files:")
    for summary_path in summary_paths:
        lines.append(f"  - {summary_path}")
    lines.append("")

    for summary_path in summary_paths:
        label = summary_label(summary_path)
        lines.extend(build_summary_section(summary_path, experiments_root, results_by_summary[label]))

    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()

    if not 0 < args.quantile < 0.5:
        raise ValueError("--quantile must be between 0 and 0.5")

    summary_paths = resolve_summary_paths(args)
    experiments_root = Path(args.experiments_root)
    output_path = Path(args.output)

    print_progress(
        f"Start analysis: {len(summary_paths)} summaries, folds {args.fold_start}-{args.fold_end}, "
        f"rounds {args.round_start}-{args.round_end}, quantile={args.quantile}"
    )

    results_by_summary: dict[str, list[UnitResult]] = {}
    for summary_path in summary_paths:
        print_progress(f"Load summary: {summary_path}")
        results_by_summary[summary_label(summary_path)] = collect_results_for_summary(
            summary_path=summary_path,
            experiments_root=experiments_root,
            fold_start=args.fold_start,
            fold_end=args.fold_end,
            round_start=args.round_start,
            round_end=args.round_end,
            quantile=args.quantile,
        )

    report = build_report(
        summary_paths=summary_paths,
        experiments_root=experiments_root,
        fold_start=args.fold_start,
        fold_end=args.fold_end,
        round_start=args.round_start,
        round_end=args.round_end,
        quantile=args.quantile,
        results_by_summary=results_by_summary,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    print_progress(f"Report written to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())