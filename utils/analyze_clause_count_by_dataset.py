#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""統計資料集每篇文檔的子句數量。"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable


DEFAULT_DATASET_DIRS = [
    "split10_train1_val1_test1_unlabeled7_aligned_disjoint_2019",
    "split10_enecpe_reccon_merged",
    "split10_enecpe",
    "split10_reccon",
]


def parse_args() -> argparse.Namespace:
    """解析命令列參數。"""

    parser = argparse.ArgumentParser(description="統計每篇文檔的 clauses 數量平均。")
    parser.add_argument("--dataset-dirs", nargs="*", default=DEFAULT_DATASET_DIRS, help="要統計的資料集資料夾。")
    parser.add_argument("--output-csv", default="analysis/clause_count_by_dataset.csv", help="輸出 CSV 路徑。")
    parser.add_argument("--include-duplicates", action="store_true", help="不依 doc_id 去重，直接統計所有 fold JSON 中出現的文檔。")
    return parser.parse_args()


def load_documents(path: Path) -> list[dict]:
    """讀取單一 JSON 檔中的文檔列表。"""

    with path.open("r", encoding="utf-8") as file_obj:
        data = json.load(file_obj)
    if not isinstance(data, list):
        return []
    return [doc for doc in data if isinstance(doc, dict)]


def clause_count(doc: dict) -> int:
    """取得單篇文檔的子句數量，優先使用 clauses 長度。"""

    clauses = doc.get("clauses")
    if isinstance(clauses, list):
        return len(clauses)
    try:
        return int(doc.get("doc_len", 0))
    except (TypeError, ValueError):
        return 0


def mean(values: Iterable[int]) -> float:
    """計算平均；沒有資料時回傳 0。"""

    values = list(values)
    return sum(values) / len(values) if values else 0.0


def summarize_counts(*, label: str, counts: list[int]) -> dict[str, object]:
    """把一組子句數量轉成 CSV row。"""

    sorted_counts = sorted(counts)
    return {
        "dataset": label,
        "doc_count": len(sorted_counts),
        "avg_clause_count": f"{mean(sorted_counts):.4f}",
        "min_clause_count": sorted_counts[0] if sorted_counts else 0,
        "max_clause_count": sorted_counts[-1] if sorted_counts else 0,
    }


def json_files(dataset_dir: Path) -> list[Path]:
    """取得資料夾中的 fold JSON 檔，排除 metadata.json。"""

    return sorted(path for path in dataset_dir.glob("*.json") if path.name != "metadata.json")


def analyze_dataset(dataset_dir: Path, *, deduplicate: bool) -> list[dict[str, object]]:
    """統計單一資料夾的整體與各 JSON 檔子句數量。"""

    rows: list[dict[str, object]] = []
    unique_doc_counts: dict[str, int] = {}
    duplicate_counts: list[int] = []

    for path in json_files(dataset_dir):
        docs = load_documents(path)
        file_counts = [clause_count(doc) for doc in docs]
        rows.append(summarize_counts(label=f"{dataset_dir.name}/{path.name}", counts=file_counts))

        for index, doc in enumerate(docs):
            count = clause_count(doc)
            duplicate_counts.append(count)
            doc_id = str(doc.get("doc_id", f"{path.name}:{index}"))
            unique_doc_counts.setdefault(doc_id, count)

    overall_counts = list(unique_doc_counts.values()) if deduplicate else duplicate_counts
    overall_label = dataset_dir.name if deduplicate else f"{dataset_dir.name}__with_duplicates"
    rows.insert(0, summarize_counts(label=overall_label, counts=overall_counts))
    return rows


def write_csv(rows: list[dict[str, object]], output_csv: Path) -> None:
    """寫出統計結果 CSV。"""

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8-sig", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    """主流程。"""

    args = parse_args()
    rows: list[dict[str, object]] = []
    for raw_dir in args.dataset_dirs:
        dataset_dir = Path(raw_dir)
        if not dataset_dir.is_dir():
            print(f"[skip] 找不到資料夾: {dataset_dir}")
            continue
        rows.extend(analyze_dataset(dataset_dir, deduplicate=not args.include_duplicates))

    if not rows:
        raise SystemExit("沒有任何可輸出的統計結果。")
    write_csv(rows, Path(args.output_csv))
    print(f"已輸出子句數量統計: {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())