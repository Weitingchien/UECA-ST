"""統計 prompt_ECPE_few_shot_ST_* 實驗中每折每輪的 filtered_error_rate

依序執行下列步驟：
1. 尋找符合指定模式的實驗資料夾（預設為 prompt_ECPE_few_shot_ST_*)
2. 掃描其中的 pseudo_results_* 目錄，讀取 pseudo_labeled_samples_fold{i}_round{j}.json
3. 對每個檔案計算 (filtered_error_rate 加總 / 文檔數量) 作為該輪錯誤率
4. 進一步計算各折的平均錯誤率（將十輪錯誤率加總後除以 10)

輸出內容會列出各折各輪的錯誤率、文檔數量，以及每折的平均錯誤率，
並在每個實驗資料夾下寫入 pseudo_error_report.txt 方便後續查閱。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional


@dataclass
class RoundStats:
    """描述單一 fold/round 的錯誤率統計"""

    avg_error: float
    doc_count: int
    missing_rate_fields: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="統計 pseudo_labeled_samples_fold{i}_round{j}.json 的 filtered_error_rate"
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("."),
        help="搜尋 prompt_ECPE_few_shot_ST_* 的起始目錄 (預設：當前目錄)",
    )
    parser.add_argument(
        "--pattern",
        default="prompt_ECPE_few_shot_ST_*",
        help="實驗資料夾的 glob 模式 (預設: prompt_ECPE_few_shot_ST_*)",
    )
    parser.add_argument(
        "--results-pattern",
        default="pseudo_results_*",
        help="實驗內 pseudo_results 目錄的 glob 模式 (預設: pseudo_results_*)",
    )
    parser.add_argument(
        "--fold-start",
        type=int,
        default=1,
        help="要統計的 fold 起始編號 (預設: 1)",
    )
    parser.add_argument(
        "--fold-end",
        type=int,
        default=10,
        help="要統計的 fold 結束編號 (預設: 10)",
    )
    parser.add_argument(
        "--round-start",
        type=int,
        default=1,
        help="要統計的 round 起始編號 (預設: 1)",
    )
    parser.add_argument(
        "--round-end",
        type=int,
        default=10,
        help="要統計的 round 結束編號 (預設: 10)",
    )
    return parser.parse_args()


def load_round_stats(json_path: Path) -> Optional[RoundStats]:
    """讀取單一 JSON 檔並回傳 round 統計"""

    try:
        records = json.loads(json_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        print(f"  [警告] 無法讀取 {json_path}: {exc}")
        return None

    if not isinstance(records, list) or not records:
        print(f"  [警告] {json_path.name} 格式不是非空列表，跳過")
        return None

    doc_count = len(records)
    missing_rate_fields = 0
    total_error = 0.0

    for record in records:
        rate = record.get("filtered_error_rate") if isinstance(record, dict) else None
        if isinstance(rate, (int, float)):
            total_error += float(rate)
        else:
            missing_rate_fields += 1

    avg_error = total_error / doc_count
    return RoundStats(avg_error=avg_error, doc_count=doc_count, missing_rate_fields=missing_rate_fields)


def analyze_pseudo_dir(
    pseudo_dir: Path, fold_ids: Iterable[int], round_ids: Iterable[int]
) -> Dict[int, Dict[int, RoundStats]]:
    """回傳 {fold: {round: RoundStats}} 結構。"""

    results: Dict[int, Dict[int, RoundStats]] = {}
    for fold in fold_ids:
        round_map: Dict[int, RoundStats] = {}
        for round_id in round_ids:
            json_path = pseudo_dir / f"pseudo_labeled_samples_fold{fold}_round{round_id}.json"
            if not json_path.exists():
                continue
            stats = load_round_stats(json_path)
            if stats is not None:
                round_map[round_id] = stats
        if round_map:
            results[fold] = round_map
    return results


def render_summary(
    experiment: Path,
    pseudo_dir: Path,
    data: Dict[int, Dict[int, RoundStats]],
    fold_ids: List[int],
    round_ids: List[int],
) -> str:
    lines: List[str] = []
    rel_dir = pseudo_dir.relative_to(experiment.parent)
    lines.append(f"=== {experiment.name} | {rel_dir.name} ===")
    fold_normalized: List[float] = []
    fold_actual: List[float] = []
    for fold in fold_ids:
        round_stats = data.get(fold, {})
        if not round_stats:
            lines.append(f"fold {fold:02d}: 找不到任何輪次資料")
            continue

        available_rounds = [round_stats[r].avg_error for r in round_ids if r in round_stats]
        available_count = len(available_rounds)
        normalized_avg = sum(available_rounds) / len(round_ids) if round_ids else 0.0
        actual_avg = sum(available_rounds) / available_count if available_count else 0.0
        missing_rounds = [str(r) for r in round_ids if r not in round_stats]

        lines.append(
            f"fold {fold:02d}: avg_error(normalized)={normalized_avg:.4f}, "
            f"avg_error(actual)={actual_avg:.4f} | rounds {available_count}/{len(round_ids)}"
        )
        if missing_rounds:
            lines.append(f"  缺少輪次: {', '.join(missing_rounds)} (視為 0 納入 normalized 平均)")

        fold_normalized.append(normalized_avg)
        fold_actual.append(actual_avg)

        for round_id in round_ids:
            stats = round_stats.get(round_id)
            if stats is None:
                lines.append(f"  round {round_id:02d}: --")
                continue
            missing_note = (
                f", 缺少 filtered_error_rate 欄位 {stats.missing_rate_fields} 筆"
                if stats.missing_rate_fields
                else ""
            )
            lines.append(
                f"  round {round_id:02d}: err={stats.avg_error:.4f}, docs={stats.doc_count}{missing_note}"
            )

    if fold_normalized:
        overall_normalized = sum(fold_normalized) / len(fold_normalized)
        overall_actual = sum(fold_actual) / len(fold_actual)
        lines.append(
            f"overall folds: avg_error(normalized)={overall_normalized:.4f}, "
            f"avg_error(actual)={overall_actual:.4f} | folds {len(fold_normalized)}/{len(fold_ids)}"
        )

    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    fold_ids = list(range(args.fold_start, args.fold_end + 1))
    round_ids = list(range(args.round_start, args.round_end + 1))

    experiments = [
        path
        for path in sorted(args.base_dir.glob(args.pattern))
        if path.is_dir()
    ]

    if not experiments:
        print("找不到任何符合的實驗資料夾")
        return

    for exp_dir in experiments:
        pseudo_dirs = [
            path
            for path in sorted(exp_dir.glob(args.results_pattern))
            if path.is_dir()
        ]
        if not pseudo_dirs:
            print(f"\n=== {exp_dir.name} ===\n  (無 pseudo_results 目錄)")
            continue

        exp_summaries: List[str] = []
        for pseudo_dir in pseudo_dirs:
            data = analyze_pseudo_dir(pseudo_dir, fold_ids, round_ids)
            if not data:
                text = (
                    f"=== {exp_dir.name} | {pseudo_dir.name} ===\n"
                    f"  (無對應的 pseudo_labeled_samples 檔案)"
                )
                print("\n" + text)
                exp_summaries.append(text)
                continue
            summary_text = render_summary(exp_dir, pseudo_dir, data, fold_ids, round_ids)
            print("\n" + summary_text)
            exp_summaries.append(summary_text)

        if exp_summaries:
            report_path = exp_dir / "pseudo_error_report.txt"
            report_content = "\n\n".join(exp_summaries).rstrip() + "\n"
            report_path.write_text(report_content, encoding="utf-8")
            print(f"\n[訊息] 已寫入 {report_path}")


if __name__ == "__main__":
    main()
