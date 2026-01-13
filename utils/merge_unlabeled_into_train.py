#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Merge few-shot train and unlabeled splits back into full supervised folds.

Example usage:
    python merge_unlabeled_into_train.py \
        --input-dir split10_home_few_shot_with_val \
        --output-dir split10_home_supervised_full
"""

import argparse
import json
from pathlib import Path
from typing import List


def load_json(path: Path) -> List[dict]:
    """Load a JSON file containing a list of documents."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: List[dict]) -> None:
    """Persist a list of documents as JSON with UTF-8 encoding."""
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def merge_fold(
    src_dir: Path,
    dst_dir: Path,
    fold: int,
    copy_val: bool,
    copy_test: bool,
    merge_val_into_train: bool,
) -> None:
    """Merge one fold's train/unlabeled (optionally val) splits, then copy val/test if requested."""
    train_path = src_dir / f"fold{fold}_train.json"
    unlabeled_path = src_dir / f"fold{fold}_unlabeled.json"

    if not train_path.exists():
        raise FileNotFoundError(f"找不到訓練資料檔案: {train_path}")
    if not unlabeled_path.exists():
        raise FileNotFoundError(f"找不到未標籤資料檔案: {unlabeled_path}")

    train_docs = load_json(train_path)
    unlabeled_docs = load_json(unlabeled_path)

    val_docs: List[dict] = []
    if merge_val_into_train:
        val_path = src_dir / f"fold{fold}_val.json"
        if not val_path.exists():
            raise FileNotFoundError(f"找不到驗證資料檔案: {val_path}")
        val_docs = load_json(val_path)

    combined_docs = train_docs + val_docs + unlabeled_docs

    save_json(dst_dir / f"fold{fold}_train.json", combined_docs)

    parts = [f"train={len(train_docs)}"]
    if merge_val_into_train:
        parts.append(f"val={len(val_docs)}")
    parts.append(f"unlabeled={len(unlabeled_docs)}")

    print(f"fold{fold}: {' + '.join(parts)} => combined={len(combined_docs)}")

    if copy_val:
        val_path = src_dir / f"fold{fold}_val.json"
        if not val_path.exists():
            raise FileNotFoundError(f"找不到驗證資料檔案: {val_path}")
        save_json(dst_dir / f"fold{fold}_val.json", load_json(val_path))

    if copy_test:
        test_path = src_dir / f"fold{fold}_test.json"
        if not test_path.exists():
            raise FileNotFoundError(f"找不到測試資料檔案: {test_path}")
        save_json(dst_dir / f"fold{fold}_test.json", load_json(test_path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="將 few-shot 訓練集與未標籤資料合併，重建完整監督式訓練資料。"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="來源資料夾 (包含 fold*_train.json / fold*_unlabeled.json 等)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="輸出資料夾 (若不存在會自動建立)"
    )
    parser.add_argument(
        "--num-folds",
        type=int,
        default=10,
        help="折數 (預設: 10)"
    )
    parser.add_argument(
        "--copy-val",
        action="store_true",
        help="同步複製 fold*_val.json 至輸出資料夾"
    )
    parser.add_argument(
        "--copy-test",
        action="store_true",
        help="同步複製 fold*_test.json 至輸出資料夾"
    )
    parser.add_argument(
        "--merge-val-into-train",
        action="store_true",
        help="將 fold*_val.json 併入輸出訓練檔案 (用於重建 90/10 train-test)"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input_dir.exists():
        raise FileNotFoundError(f"來源資料夾不存在: {args.input_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for fold in range(1, args.num_folds + 1):
        merge_fold(
            src_dir=args.input_dir,
            dst_dir=args.output_dir,
            fold=fold,
            copy_val=args.copy_val,
            copy_test=args.copy_test,
            merge_val_into_train=args.merge_val_into_train,
        )

    print(f"\n合併完成！輸出路徑: {args.output_dir}")


if __name__ == "__main__":
    main()
