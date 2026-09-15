#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""將 avg3 checkpoint source CSV 彙整成精簡表"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
import math
from pathlib import Path
import re


def parse_args() -> argparse.Namespace:
    """解析命令列參數。"""
    parser = argparse.ArgumentParser(description="將 avg3 checkpoint source CSV 轉成碩論用精簡表。")
    parser.add_argument("--input-csv", required=True, help="來源 CSV，例如 png/avg3_checkpoint_sources_cause.csv")
    parser.add_argument("--output-base", required=True, help="輸出檔案前綴，例如 png/avg3_checkpoint_sources_cause_thesis_table")
    parser.add_argument("--table-title", default="Avg3 來源 checkpoint 輪次分布", help="表格標題。")
    parser.add_argument("--count-column", default="distinct_round_count", help="要彙整的欄位名稱，預設為 distinct_round_count。")
    parser.add_argument(
        "--experiments-roots",
        nargs="+",
        default=None,
        help="可選：一或多個實驗根目錄。提供後會額外計算各分組的平均驗證集 Pair F1。",
    )
    parser.add_argument(
        "--round-column",
        default="pair_round",
        help="要對應到哪個 round 欄位來抓驗證指標，預設為 pair_round。",
    )
    parser.add_argument(
        "--metric-column-name",
        default="平均驗證集Pair F1",
        help="輸出表格中的額外指標欄名稱。",
    )
    parser.add_argument(
        "--final-val-dirs",
        nargs="+",
        default=None,
        help=(
            "可選：一或多個最終 Avg3 validation 輸出目錄，例如 "
            "debug/avg3_final_val_eval_seed20_full/out。提供後會直接讀 val_evaluation_fold*.txt。"
        ),
    )
    return parser.parse_args()


def read_rows(csv_path: Path) -> list[dict[str, str]]:
    """讀取來源 CSV。"""
    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def get_description(value: int) -> str:
    """把 distinct_round_count 轉成較容易寫進論文的中文描述。"""
    if value == 1:
        return "三個最佳模型來自同一輪"
    if value == 2:
        return "三個最佳模型來自兩個不同輪次"
    if value == 3:
        return "三個最佳模型來自三個不同輪次"
    return f"三個最佳模型來自 {value} 個不同輪次"


def parse_init_metric(init_metrics_path: Path) -> float:
    """從初始監督 metrics 檔中擷取 Pair F1。"""
    text = init_metrics_path.read_text(encoding="utf-8", errors="ignore")
    match = re.search(r"Pair\s+F1:\s*\[(.*?)\]", text)
    if match is None:
        raise ValueError(f"無法從初始 metrics 擷取 Pair F1: {init_metrics_path}")

    number_matches = re.findall(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)", match.group(1))
    if not number_matches:
        raise ValueError(f"初始 metrics 沒有可用的 Pair F1 數值: {init_metrics_path}")

    return float(number_matches[-1])


def resolve_experiment_dir(experiment_name: str, experiments_roots: list[Path]) -> Path:
    """從多個 experiments root 中找到對應的實驗資料夾。"""
    for root in experiments_roots:
        candidate = root / experiment_name
        if candidate.is_dir():
            return candidate
    joined_roots = ", ".join(str(path) for path in experiments_roots)
    raise FileNotFoundError(f"找不到實驗資料夾 {experiment_name}，已搜尋: {joined_roots}")


def parse_pair_f1_from_final_eval(eval_path: Path) -> float:
    """從 final validation 評估檔擷取 Pair F1。"""
    text = eval_path.read_text(encoding="utf-8", errors="ignore")
    match = re.search(r"F1 Score \(Emotion, Cause, Pair\):\s*[-+]?\d+(?:\.\d+)?,\s*[-+]?\d+(?:\.\d+)?,\s*([-+]?\d+(?:\.\d+)?)", text)
    if match is None:
        raise ValueError(f"無法從 final eval 檔擷取 Pair F1: {eval_path}")
    return float(match.group(1))


def build_seed_to_final_val_dir(final_val_dirs: list[Path]) -> dict[str, Path]:
    """從輸出目錄名稱建立 seed 對應表。"""
    seed_to_dir: dict[str, Path] = {}
    for path in final_val_dirs:
        match = re.search(r"seed(\d+)", path.as_posix())
        if match is None:
            raise ValueError(f"無法從 final val 目錄辨識 seed 編號: {path}")
        seed_key = f"seed{match.group(1)}"
        if seed_key in seed_to_dir:
            raise ValueError(f"重複的 seed 輸出目錄: {seed_key} -> {path}")
        seed_to_dir[seed_key] = path
    return seed_to_dir


def read_final_val_pair_f1_for_row(
    row: dict[str, str],
    seed_to_final_val_dir: dict[str, Path],
    metrics_cache: dict[Path, float],
) -> float:
    """讀取單一 seed-fold 對應的最終 Avg3 validation Pair F1。"""
    seed = row.get("seed")
    fold_text = row.get("fold")
    if not seed or not fold_text:
        raise ValueError("來源 CSV 缺少 seed 或 fold 欄位，無法讀取 final Avg3 validation 結果。")

    if seed not in seed_to_final_val_dir:
        available = ", ".join(sorted(seed_to_final_val_dir))
        raise KeyError(f"找不到 {seed} 對應的 final validation 輸出目錄，可用 seed: {available}")

    eval_path = seed_to_final_val_dir[seed] / f"val_evaluation_fold{int(fold_text)}.txt"
    if not eval_path.is_file():
        raise FileNotFoundError(f"找不到 final validation 評估檔: {eval_path}")

    if eval_path not in metrics_cache:
        metrics_cache[eval_path] = parse_pair_f1_from_final_eval(eval_path)
    return metrics_cache[eval_path]


def read_pair_f1_for_row(
    row: dict[str, str],
    experiments_roots: list[Path],
    round_column: str,
    metrics_cache: dict[Path, dict[int, float]],
) -> float:
    """讀取單一 seed-fold 資料列對應 round 的驗證集 Pair F1"""
    experiment_name = row.get("experiment_dir")
    fold_text = row.get("fold")
    round_text = row.get(round_column)

    if not experiment_name or not fold_text or not round_text:
        raise ValueError("來源 CSV 缺少 experiment_dir、fold 或 round 欄位，無法計算平均驗證集 Pair F1。")

    fold = int(fold_text)
    round_value = int(round_text)
    experiment_dir = resolve_experiment_dir(experiment_name, experiments_roots)

    if round_value == 0:
        init_metrics_path = experiment_dir / "init_supervised_metrics" / f"fold{fold}_init_supervised_metrics.txt"
        if not init_metrics_path.is_file():
            raise FileNotFoundError(f"找不到初始監督 metrics 檔案: {init_metrics_path}")
        return parse_init_metric(init_metrics_path)

    metrics_csv_path = experiment_dir / f"fold{fold}_self_training_round_metrics.csv"
    if not metrics_csv_path.is_file():
        raise FileNotFoundError(f"找不到 round metrics CSV: {metrics_csv_path}")

    if metrics_csv_path not in metrics_cache:
        round_to_pair_f1: dict[int, float] = {}
        with metrics_csv_path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            for metrics_row in reader:
                row_round = int(metrics_row["self_training_round"])
                round_to_pair_f1[row_round] = float(metrics_row["val_f1_pair"])
        metrics_cache[metrics_csv_path] = round_to_pair_f1

    round_to_pair_f1 = metrics_cache[metrics_csv_path]
    if round_value not in round_to_pair_f1:
        raise ValueError(f"{metrics_csv_path} 找不到 self_training_round={round_value} 的 val_f1_pair")

    return round_to_pair_f1[round_value]


def build_summary_rows(
    rows: list[dict[str, str]],
    count_column: str,
    experiments_roots: list[Path] | None = None,
    round_column: str = "pair_round",
    final_val_dirs: list[Path] | None = None,
) -> tuple[list[dict[str, str]], int, int, float]:
    """依 distinct_round_count 之類的欄位做分組統計。"""
    if not rows:
        raise ValueError("來源 CSV 沒有資料列，無法建立精簡表。")

    if count_column not in rows[0]:
        raise ValueError(f"來源 CSV 找不到欄位: {count_column}")

    counter = Counter(int(row[count_column]) for row in rows)
    total = len(rows)
    summary_rows: list[dict[str, str]] = []
    metric_means: dict[int, float] = {}
    metric_stds: dict[int, float] = {}

    if experiments_roots or final_val_dirs:
        grouped_metric_values: dict[int, list[float]] = {value: [] for value in counter}
        if final_val_dirs:
            metrics_cache: dict[Path, float] = {}
            seed_to_final_val_dir = build_seed_to_final_val_dir(final_val_dirs)
            for row in rows:
                count_value = int(row[count_column])
                grouped_metric_values[count_value].append(
                    read_final_val_pair_f1_for_row(
                        row=row,
                        seed_to_final_val_dir=seed_to_final_val_dir,
                        metrics_cache=metrics_cache,
                    )
                )
        else:
            metrics_cache = {}
            for row in rows:
                count_value = int(row[count_column])
                grouped_metric_values[count_value].append(
                    read_pair_f1_for_row(
                        row=row,
                        experiments_roots=experiments_roots,
                        round_column=round_column,
                        metrics_cache=metrics_cache,
                    )
                )
        metric_means = {
            value: sum(values) / len(values)
            for value, values in grouped_metric_values.items()
            if values
        }
        metric_stds = {
            value: math.sqrt(sum((item - metric_means[value]) ** 2 for item in values) / len(values))
            for value, values in grouped_metric_values.items()
            if values
        }

    for value in sorted(counter.keys()):
        count = counter[value]
        percent = count * 100 / total
        row_data = {
            "distinct_round_count": str(value),
            "count": str(count),
            "percent": f"{percent:.1f}%",
            "description": get_description(value),
        }
        if metric_means:
            row_data["avg_val_pair_f1"] = f"{metric_means[value] * 100:.2f}%"
            row_data["std_val_pair_f1"] = f"{metric_stds[value] * 100:.2f}%"
        summary_rows.append(row_data)

    multi_round_count = sum(counter[value] for value in counter if value > 1)
    multi_round_percent = multi_round_count * 100 / total
    return summary_rows, total, multi_round_count, multi_round_percent


def build_markdown(
    title: str,
    summary_rows: list[dict[str, str]],
    total: int,
    multi_round_count: int,
    multi_round_percent: float,
    metric_column_name: str | None = None,
    metric_std_column_name: str | None = None,
) -> str:
    """建立 Markdown 表格內容。"""
    lines: list[str] = []
    lines.append(title)
    lines.append("")
    if metric_column_name and metric_std_column_name:
        lines.append(f"| 不同來源輪次數 | 組合數 | 比例 | {metric_column_name} | {metric_std_column_name} | 說明 |")
        lines.append("| --- | ---: | ---: | ---: | ---: | --- |")
    elif metric_column_name:
        lines.append(f"| 不同來源輪次數 | 組合數 | 比例 | {metric_column_name} | 說明 |")
        lines.append("| --- | ---: | ---: | ---: | --- |")
    else:
        lines.append("| 不同來源輪次數 | 組合數 | 比例 | 說明 |")
        lines.append("| --- | ---: | ---: | --- |")

    for row in summary_rows:
        if metric_column_name and metric_std_column_name:
            lines.append(
                f"| {row['distinct_round_count']} | {row['count']} | {row['percent']} | {row['avg_val_pair_f1']} | {row['std_val_pair_f1']} | {row['description']} |"
            )
        elif metric_column_name:
            lines.append(
                f"| {row['distinct_round_count']} | {row['count']} | {row['percent']} | {row['avg_val_pair_f1']} | {row['description']} |"
            )
        else:
            lines.append(
                f"| {row['distinct_round_count']} | {row['count']} | {row['percent']} | {row['description']} |"
            )

    if metric_column_name and metric_std_column_name:
        lines.append(f"| 合計 | {total} | 100.0% | - | - | - |")
    elif metric_column_name:
        lines.append(f"| 合計 | {total} | 100.0% | - | - |")
    else:
        lines.append(f"| 合計 | {total} | 100.0% | - |")
    lines.append("")
    lines.append(
        f"補充說明：distinct_round_count > 1 的組合共有 {multi_round_count}/{total} ({multi_round_percent:.1f}%)，"
        "表示 Avg3 多數情況下融合了至少兩個不同輪次的 task-specific 最佳模型。"
    )
    lines.append("")
    return "\n".join(lines)


def write_slim_csv(
    summary_rows: list[dict[str, str]],
    total: int,
    csv_path: Path,
    metric_column_name: str | None = None,
    metric_std_column_name: str | None = None,
) -> None:
    """輸出精簡版 CSV，方便貼到 Excel 或 Word 表格。"""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["不同來源輪次數", "組合數", "比例"]
    if metric_column_name:
        fieldnames.append(metric_column_name)
    if metric_std_column_name:
        fieldnames.append(metric_std_column_name)
    fieldnames.append("說明")

    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            output_row = {
                "不同來源輪次數": row["distinct_round_count"],
                "組合數": row["count"],
                "比例": row["percent"],
                "說明": row["description"],
            }
            if metric_column_name:
                output_row[metric_column_name] = row["avg_val_pair_f1"]
            if metric_std_column_name:
                output_row[metric_std_column_name] = row["std_val_pair_f1"]
            writer.writerow(output_row)
        total_row = {"不同來源輪次數": "合計", "組合數": str(total), "比例": "100.0%", "說明": "-"}
        if metric_column_name:
            total_row[metric_column_name] = "-"
        if metric_std_column_name:
            total_row[metric_std_column_name] = "-"
        writer.writerow(total_row)


def main() -> None:
    """主流程。"""
    args = parse_args()

    input_csv = Path(args.input_csv)
    output_base = Path(args.output_base)
    experiments_roots = [Path(path) for path in args.experiments_roots] if args.experiments_roots else None
    final_val_dirs = [Path(path) for path in args.final_val_dirs] if args.final_val_dirs else None

    if not input_csv.is_file():
        raise FileNotFoundError(f"找不到來源 CSV: {input_csv}")

    if experiments_roots and final_val_dirs:
        raise ValueError("--experiments-roots 與 --final-val-dirs 只能擇一使用。")

    if experiments_roots:
        for path in experiments_roots:
            if not path.is_dir():
                raise FileNotFoundError(f"找不到 experiments root: {path}")

    if final_val_dirs:
        for path in final_val_dirs:
            if not path.is_dir():
                raise FileNotFoundError(f"找不到 final val 輸出目錄: {path}")

    rows = read_rows(input_csv)
    summary_rows, total, multi_round_count, multi_round_percent = build_summary_rows(
        rows,
        args.count_column,
        experiments_roots=experiments_roots,
        round_column=args.round_column,
        final_val_dirs=final_val_dirs,
    )

    metric_column_name = args.metric_column_name if (experiments_roots or final_val_dirs) else None
    metric_std_column_name = f"{args.metric_column_name}標準差" if metric_column_name else None

    markdown_text = build_markdown(
        title=args.table_title,
        summary_rows=summary_rows,
        total=total,
        multi_round_count=multi_round_count,
        multi_round_percent=multi_round_percent,
        metric_column_name=metric_column_name,
        metric_std_column_name=metric_std_column_name,
    )

    md_path = output_base.with_suffix(".md")
    slim_csv_path = output_base.with_suffix(".csv")

    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(markdown_text, encoding="utf-8")
    write_slim_csv(
        summary_rows,
        total,
        slim_csv_path,
        metric_column_name=metric_column_name,
        metric_std_column_name=metric_std_column_name,
    )

    print(f"Markdown 表格已寫入: {md_path}")
    print(f"精簡 CSV 已寫入: {slim_csv_path}")


if __name__ == "__main__":
    main()