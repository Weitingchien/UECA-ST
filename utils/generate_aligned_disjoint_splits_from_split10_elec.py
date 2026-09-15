#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate few-label aligned, disjoint Elec splits from the official train set.

For each label budget, ten class-balanced train sets and ten class-balanced
validation sets are sampled without reuse across folds or roles.  In each
fold, all other official-training examples form the unlabeled set.

Unlabeled JSON files deliberately retain their ground-truth labels.  Training
code must ignore those labels when constructing unlabeled model inputs; they
remain available only for auditing or pseudo-label quality evaluation.

The official 25K Elec test set is never repartitioned.  It is written once as
``elec_canonical_test.json`` and is the fixed final evaluation set for all
folds.
"""

import argparse
import json
import os
import random
from collections import Counter


LABEL_MAP = {
    "1": 0,  # negative
    "2": 1,  # positive
}


def save_json(data, file_path):
    # 取得輸出檔案的父資料夾路徑。
    parent_dir = os.path.dirname(file_path)
    # 只有在父資料夾路徑不為空字串時才建立資料夾。
    if parent_dir:
        # 遞迴建立父資料夾；若資料夾已存在則不視為錯誤。
        os.makedirs(parent_dir, exist_ok=True)
    # 以 UTF-8 編碼及覆寫模式開啟目標 JSON 檔案。
    with open(file_path, "w", encoding="utf-8") as file_obj:
        # 保留非 ASCII 字元，並使用兩個空白縮排輸出易讀的 JSON。
        json.dump(data, file_obj, ensure_ascii=False, indent=2)


def load_elec_documents(text_path, label_path, id_prefix):
    """Load aligned Elec text/label files into a generic JSON representation."""
    # 先確認評論文字檔存在，避免稍後得到較難理解的開檔錯誤。
    if not os.path.isfile(text_path):
        # 文字檔不存在時，回報實際找不到的路徑。
        raise FileNotFoundError(f"Elec text file not found: {text_path}")
    # 再確認與文字檔逐行對齊的標籤檔存在。
    if not os.path.isfile(label_path):
        # 標籤檔不存在時，回報實際找不到的路徑。
        raise FileNotFoundError(f"Elec label file not found: {label_path}")

    # 以 UTF-8 編碼讀取文字檔。
    with open(text_path, "r", encoding="utf-8") as text_file:
        # 每一行是一筆評論；只移除行尾換行符號，不移除正文空白。
        texts = [line.rstrip("\r\n") for line in text_file]
    # 以 UTF-8 編碼讀取標籤檔。
    with open(label_path, "r", encoding="utf-8") as label_file:
        # 每一行是一筆標籤；移除標籤前後不具意義的空白。
        raw_labels = [line.strip() for line in label_file]

    # 文字與標籤必須逐行對齊，因此兩個檔案的筆數必須完全相同。
    if len(texts) != len(raw_labels):
        # 筆數不一致時立即停止，避免評論被配到錯誤標籤。
        raise ValueError(
            f"Text/label count mismatch: {text_path} has {len(texts)} rows, "
            f"but {label_path} has {len(raw_labels)} rows."
        )
    # 空資料集無法進行後續切分，因此視為輸入錯誤。
    if not texts:
        # 指出實際為空的文字檔路徑。
        raise ValueError(f"Elec input is empty: {text_path}")

    # 找出不在預期標籤映射表中的原始標籤值。
    unknown_labels = sorted(set(raw_labels) - set(LABEL_MAP))
    # 若存在未知標籤，拒絕默默猜測其類別意義。
    if unknown_labels:
        # 列出所有未知值，方便檢查資料版本或標籤格式。
        raise ValueError(f"Unexpected Elec labels in {label_path}: {unknown_labels}")

    # 建立轉換後文件的容器。
    documents = []
    # 同步走訪每一筆評論及其對應的原始標籤。
    for row_index, (text, raw_label) in enumerate(zip(texts, raw_labels)):
        # 評論若只有空白，代表該筆資料沒有可用文字。
        if not text.strip():
            # 使用從 1 開始的列號回報空白評論所在位置。
            raise ValueError(f"Empty review at row {row_index + 1}: {text_path}")
        # 將目前評論轉換為後續切分程式統一使用的字典格式。
        documents.append(
            {
                # 使用資料來源前綴及固定寬度列號產生穩定且唯一的文件 ID。
                "doc_id": f"{id_prefix}_{row_index:05d}",
                # 原樣保留評論文字。
                "text": text,
                # 將官方的 1/2 標籤轉換為模型常用的 0/1 標籤。
                "label": LABEL_MAP[raw_label],
            }
        )
    # 回傳完成格式轉換的全部文件。
    return documents


def build_few_label_role_blocks(documents, n_splits, labels_per_class, seed):
    """Create globally disjoint, class-balanced train and validation blocks."""
    # 至少需要兩個 fold，跨 fold 不重複的設定才有意義。
    if n_splits < 2:
        # fold 數量不合法時立即停止。
        raise ValueError("n_splits must be at least 2.")
    # 每類標註樣本數必須是正整數。
    if labels_per_class <= 0:
        # 不允許產生空白或負數大小的標註集合。
        raise ValueError("labels_per_class must be positive.")

    # 建立區域亂數產生器，使相同 seed 可重現完全相同的切分。
    rng = random.Random(seed)
    # 為每個 fold 建立一個獨立的訓練文件容器。
    train_blocks = [[] for _ in range(n_splits)]
    # 為每個 fold 建立一個獨立的驗證文件容器。
    val_blocks = [[] for _ in range(n_splits)]
    # 取得資料中所有類別並排序，確保遍歷順序固定。
    labels = sorted({doc["label"] for doc in documents})
    # 每個類別分開抽樣，藉此維持所有集合的類別平衡。
    for label in labels:
        # 收集目前類別在原始文件串列中的索引。
        label_indices = [
            index for index, doc in enumerate(documents) if doc["label"] == label
        ]
        # 以固定 seed 打亂目前類別的索引。
        rng.shuffle(label_indices)
        # 每個 fold 的 train 與 val 都需要指定數量，因此總需求乘以 2。
        required = 2 * n_splits * labels_per_class
        # 確認目前類別有足夠樣本支援所有不重複的 train/val 集合。
        if len(label_indices) < required:
            # 樣本不足時說明現有數量與所需數量。
            raise ValueError(
                f"Label {label} has {len(label_indices)} examples, but "
                f"{required} are required for {n_splits} disjoint train/val "
                f"folds with {labels_per_class} examples per class per role."
            )
        # 逐一配置每個 fold 的訓練及驗證樣本。
        for fold_index in range(n_splits):
            # 計算目前 fold 在訓練抽樣區段中的起點。
            train_start = fold_index * labels_per_class
            # 計算目前 fold 在訓練抽樣區段中的終點（不包含終點）。
            train_end = train_start + labels_per_class
            # 驗證區段接在全部訓練區段之後，避免 train/val 樣本重疊。
            val_start = n_splits * labels_per_class + train_start
            # 計算目前 fold 在驗證抽樣區段中的終點（不包含終點）。
            val_end = val_start + labels_per_class
            # 將目前類別指定範圍的文件加入目前 fold 訓練集合。
            train_blocks[fold_index].extend(
                documents[index] for index in label_indices[train_start:train_end]
            )
            # 將目前類別另一個指定範圍的文件加入目前 fold 驗證集合。
            val_blocks[fold_index].extend(
                documents[index] for index in label_indices[val_start:val_end]
            )

    # 完成所有類別配置後，逐 fold 打亂文件順序。
    for fold_index in range(n_splits):
        # 打亂目前 fold 的訓練文件，避免類別按順序集中排列。
        rng.shuffle(train_blocks[fold_index])
        # 打亂目前 fold 的驗證文件，避免類別按順序集中排列。
        rng.shuffle(val_blocks[fold_index])
    # 回傳十組互不重複的訓練集合與十組互不重複的驗證集合。
    return train_blocks, val_blocks


def get_doc_ids(documents):
    # 以集合回傳所有文件 ID，供重複、交集與聯集檢查使用。
    return {document["doc_id"] for document in documents}


def validate_source(train_documents, canonical_test_documents, n_splits):
    # 取得官方訓練集的所有文件 ID。
    train_ids = get_doc_ids(train_documents)
    # 取得官方固定測試集的所有文件 ID。
    test_ids = get_doc_ids(canonical_test_documents)
    # 集合大小若小於文件數量，代表訓練集內存在重複 ID。
    if len(train_ids) != len(train_documents):
        # 重複 ID 會破壞互斥性判斷，因此立即停止。
        raise ValueError("Duplicate doc_id values exist in the Elec train set.")
    # 同樣確認固定測試集內沒有重複 ID。
    if len(test_ids) != len(canonical_test_documents):
        # 回報固定測試集的文件 ID 重複問題。
        raise ValueError("Duplicate doc_id values exist in the Elec canonical test set.")
    # 官方訓練集與測試集不得共用任何文件 ID。
    if train_ids & test_ids:
        # 若有交集，代表資料洩漏或 ID 產生方式有誤。
        raise ValueError("Elec train and canonical test doc_id values overlap.")

    # 對官方訓練集與固定測試集逐一檢查類別結構。
    for split_name, documents in (
        ("train", train_documents),
        ("canonical test", canonical_test_documents),
    ):
        # 計算目前資料集每個標籤的樣本數。
        label_counts = Counter(doc["label"] for doc in documents)
        # Elec 二元分類資料必須恰好包含 0 與 1 兩種標籤。
        if set(label_counts) != {0, 1}:
            # 回報實際出現的標籤與數量。
            raise ValueError(f"{split_name} must contain labels 0 and 1: {label_counts}")
        # 論文使用的 Elec 資料應為正負類別平衡。
        if label_counts[0] != label_counts[1]:
            # 類別數量不相等時停止，避免後續實驗設定偏離預期。
            raise ValueError(f"{split_name} is not class-balanced: {label_counts}")
        # 每個類別至少要有與 fold 數相同的樣本數。
        if any(count < n_splits for count in label_counts.values()):
            # 樣本不足時無法為每個 fold 分配該類別。
            raise ValueError(
                f"{split_name} has too few examples per class for {n_splits} blocks."
            )


def validate_fold(source_ids, fold_num, train_docs, val_docs, unlabeled_docs):
    # 將目前 fold 三種角色的文件 ID 分別轉為集合。
    role_ids = {
        # 目前 fold 的有標註訓練樣本 ID。
        "train": get_doc_ids(train_docs),
        # 目前 fold 的有標註驗證樣本 ID。
        "val": get_doc_ids(val_docs),
        # 目前 fold 的未標註訓練樣本 ID（檔案仍保留真值）。
        "unlabeled": get_doc_ids(unlabeled_docs),
    }

    # 保留角色名稱順序，供兩兩交集檢查使用。
    role_names = list(role_ids)
    # 依序選擇左側角色。
    for left_index, left_name in enumerate(role_names):
        # 只與其後角色比較，避免同一對角色重複檢查。
        for right_name in role_names[left_index + 1 :]:
            # 計算兩個角色的文件 ID 交集。
            overlap = role_ids[left_name] & role_ids[right_name]
            # 任一交集都代表同一 fold 發生資料角色重疊。
            if overlap:
                # 顯示前五個重疊 ID，方便定位問題又避免訊息過長。
                raise ValueError(
                    f"fold{fold_num} {left_name}/{right_name} overlap: "
                    f"{sorted(overlap)[:5]}"
                )

    # 合併目前 fold 的 train、val 與 unlabeled 文件 ID。
    union_ids = set().union(*role_ids.values())
    # 聯集必須與完整官方訓練集完全一致。
    if union_ids != source_ids:
        # 找出原始訓練集中遺漏的前五個文件 ID。
        missing = sorted(source_ids - union_ids)[:5]
        # 找出不應出現在官方訓練集切分中的前五個額外 ID。
        extra = sorted(union_ids - source_ids)[:5]
        # 回報覆蓋不完整或混入額外樣本的錯誤。
        raise ValueError(
            f"fold{fold_num} does not exactly cover the Elec train set; "
            f"missing={missing}, extra={extra}"
        )


def write_self_training_fold(
    output_dir,
    fold_num,
    train_docs,
    val_docs,
    unlabeled_docs,
):
    # 寫出目前 fold 的有標註訓練資料。
    save_json(train_docs, os.path.join(output_dir, f"fold{fold_num}_train.json"))
    # 寫出目前 fold 的有標註驗證資料。
    save_json(val_docs, os.path.join(output_dir, f"fold{fold_num}_val.json"))
    # 寫出目前 fold 的 unlabeled 資料，並刻意保留每筆樣本的 label。
    save_json(
        unlabeled_docs,
        os.path.join(output_dir, f"fold{fold_num}_unlabeled.json"),
    )


def validate_cross_fold_roles(train_blocks, val_blocks):
    """Require every labeled sample to occur in exactly one fold and one role."""
    # 記錄所有先前已配置為 train 或 val 的文件 ID。
    seen_ids = set()
    # 依序檢查所有訓練 blocks，再檢查所有驗證 blocks。
    for role_name, blocks in (("train", train_blocks), ("val", val_blocks)):
        # 以從 1 開始的 fold 編號逐一檢查各 block。
        for fold_num, documents in enumerate(blocks, start=1):
            # 取得目前角色及 fold 中的所有文件 ID。
            document_ids = get_doc_ids(documents)
            # 計算目前 block 與先前全部 train/val blocks 的交集。
            overlap = seen_ids & document_ids
            # 若有交集，表示標註樣本被跨 fold 或跨角色重複使用。
            if overlap:
                # 顯示前五個重複 ID 以協助除錯。
                raise ValueError(
                    f"{role_name} fold{fold_num} reuses labeled samples: "
                    f"{sorted(overlap)[:5]}"
                )
            # 將目前 block 的 ID 加入已使用集合，供後續 block 檢查。
            seen_ids.update(document_ids)


def generate_aligned_splits(
    train_documents,
    canonical_test_documents,
    output_dir,
    n_splits,
    labels_per_class,
    seed,
):
    # 依指定標註預算建立所有 fold 的互斥且類別平衡 train/val blocks。
    train_blocks, val_blocks = build_few_label_role_blocks(
        train_documents,
        n_splits,
        labels_per_class,
        seed,
    )
    # 再次確認任何有標註樣本都沒有跨 fold 或跨角色重複出現。
    validate_cross_fold_roles(train_blocks, val_blocks)
    # 保存完整官方訓練集的 ID 集合，供每個 fold 驗證資料覆蓋率。
    source_ids = get_doc_ids(train_documents)

    # 將官方 25,000 筆測試資料完整寫出，不參與任何 fold 的重新切分。
    save_json(
        canonical_test_documents,
        os.path.join(output_dir, "elec_canonical_test.json"),
    )

    # 顯示目前標註預算以及每個 fold 實際產生的 train/val 大小。
    print(
        f"labels_per_class={labels_per_class}; "
        f"train sizes={[len(block) for block in train_blocks]}; "
        f"val sizes={[len(block) for block in val_blocks]}"
    )
    # 逐一產生每個 fold 的 train、val 與 unlabeled 檔案。
    for fold_index in range(n_splits):
        # 將從 0 開始的串列索引轉為從 1 開始的輸出 fold 編號。
        fold_num = fold_index + 1
        # 取得目前 fold 已預先配置完成的訓練文件。
        self_train_docs = train_blocks[fold_index]
        # 取得目前 fold 已預先配置完成的驗證文件。
        self_val_docs = val_blocks[fold_index]
        # 合併目前 fold 的 train/val ID，兩者都不能再進入該 fold 的 unlabeled。
        labeled_ids = get_doc_ids(self_train_docs) | get_doc_ids(self_val_docs)
        # 官方訓練集中除目前 fold train/val 以外的所有文件都是 unlabeled。
        self_unlabeled_docs = [
            document
            for document in train_documents
            if document["doc_id"] not in labeled_ids
        ]
        # 使用每個 fold 不同但可重現的 seed 打亂 unlabeled 文件順序。
        random.Random(seed + fold_num).shuffle(self_unlabeled_docs)

        # 驗證目前 fold 的三種角色互斥，且完整覆蓋官方訓練集。
        validate_fold(
            source_ids,
            fold_num,
            self_train_docs,
            self_val_docs,
            self_unlabeled_docs,
        )
        # 將驗證通過的目前 fold 寫入指定輸出資料夾。
        write_self_training_fold(
            output_dir,
            fold_num,
            self_train_docs,
            self_val_docs,
            self_unlabeled_docs,
        )

        # 防止未來修改程式時意外把 unlabeled 的真值標籤刪除。
        if any("label" not in document for document in self_unlabeled_docs):
            # 只要有一筆缺少 label 就中止，避免輸出格式悄悄改變。
            raise ValueError(f"fold{fold_num} unlabeled samples lost ground-truth labels.")
        # 顯示目前 fold 三種角色的實際樣本數量。
        print(
            f"Fold {fold_num}: train/val/unlabeled="
            f"{len(self_train_docs)}/{len(self_val_docs)}/"
            f"{len(self_unlabeled_docs)}"
        )


def main():
    # 建立命令列參數解析器並描述此工具的用途。
    parser = argparse.ArgumentParser(
        description="Generate aligned, disjoint 10-fold splits for Elec."
    )
    # 設定原始 Elec 文字檔及標籤檔所在資料夾。
    parser.add_argument(
        "--input-dir",
        default="data/elec/elec",
        help="Directory containing elec-25k-train.* and elec-test.*.",
    )
    # 設定各標註預算輸出資料夾的共同父資料夾。
    parser.add_argument(
        "--output-root",
        default="data/elec",
        help="Parent directory for labels_per_class_30/50/100 outputs.",
    )
    # 設定要產生的 fold 數量，預設為 10。
    parser.add_argument("--n-splits", type=int, default=10)
    # 設定所有隨機抽樣使用的基礎 seed，確保結果可重現。
    parser.add_argument("--seed", type=int, default=42)
    # 設定要一次產生的一個或多個「每類標註數」實驗預算。
    parser.add_argument(
        "--labels-per-class",
        type=int,
        nargs="+",
        default=[30, 50, 100],
        help="Few-label budgets to generate (default: 30 50 100).",
    )
    # 解析使用者從命令列傳入的全部參數。
    args = parser.parse_args()

    # 載入官方 25,000 筆訓練評論及其標籤，並使用 train 作為 ID 前綴。
    train_documents = load_elec_documents(
        os.path.join(args.input_dir, "elec-25k-train.txt"),
        os.path.join(args.input_dir, "elec-25k-train.cat"),
        id_prefix="train",
    )
    # 載入官方 25,000 筆測試評論及其標籤，並使用 test 作為 ID 前綴。
    canonical_test_documents = load_elec_documents(
        os.path.join(args.input_dir, "elec-test.txt"),
        os.path.join(args.input_dir, "elec-test.cat"),
        id_prefix="test",
    )
    # 在切分前檢查來源資料的 ID、類別及 train/test 互斥性。
    validate_source(train_documents, canonical_test_documents, args.n_splits)

    # 顯示成功載入的官方訓練集與固定測試集筆數。
    print(
        f"Loaded Elec: train={len(train_documents)}, "
        f"canonical test={len(canonical_test_documents)}"
    )
    # 不允許同一標註預算重複出現，以免重複覆寫同一輸出資料夾。
    if len(set(args.labels_per_class)) != len(args.labels_per_class):
        # 透過 argparse 顯示清楚的參數錯誤並結束程式。
        parser.error("--labels-per-class contains duplicate values.")

    # 逐一處理使用者要求的每類標註樣本數。
    for labels_per_class in args.labels_per_class:
        # 依標註預算建立獨立輸出資料夾名稱。
        output_dir = os.path.join(
            args.output_root,
            f"labels_per_class_{labels_per_class}",
        )
        # 顯示目前即將產生的輸出資料夾。
        print(f"\nGenerating: {output_dir}")
        # 產生目前標註預算的十組切分及固定官方測試集。
        generate_aligned_splits(
            train_documents=train_documents,
            canonical_test_documents=canonical_test_documents,
            output_dir=output_dir,
            n_splits=args.n_splits,
            labels_per_class=labels_per_class,
            seed=args.seed,
        )
    # 所有標註預算皆完成後顯示結束訊息。
    print("Elec aligned-disjoint split generation completed.")


# 此檔案被直接執行時才呼叫 main；被匯入時不會自動產生資料。
if __name__ == "__main__":
    # 進入命令列程式的主要執行流程。
    main()
