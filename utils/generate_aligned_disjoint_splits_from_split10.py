#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
從 split10 任一折還原完整資料集，並同步產生兩套使用相同 test blocks 的資料：

1. 監督式資料：90% train + 10% test
2. 自訓練資料：10% train + 10% val + 10% test + 70% unlabeled

設計重點：
- 只需要 split10 任一折的 train/test，就能還原完整全集
- 先把還原後的全集存成 JSON，再從該 JSON 讀回來切分
- supervised 與 self-training 的同 fold test 完全一致
- self-training 的 train、val、test 各自都在不同 folds 間不重疊
"""

import argparse
import json
import os

from generate_disjoint_splits import (
    build_disjoint_blocks,
    convert_to_unlabeled,
    flatten_blocks,
    get_rotated_blocks,
)


def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file_obj:
        return json.load(file_obj)


def save_json(data, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as file_obj:
        json.dump(data, file_obj, ensure_ascii=False, indent=2)


def doc_id_key(doc):
    doc_id = str(doc.get('doc_id', ''))
    if doc_id.isdigit():
        return (0, int(doc_id))
    return (1, doc_id)


def get_doc_ids(documents):
    doc_ids = []
    for doc in documents:
        if 'doc_id' not in doc:
            raise ValueError('發現缺少 doc_id 的文件，無法進行切分驗證')
        doc_ids.append(str(doc['doc_id']))
    return doc_ids


def reconstruct_full_documents(split10_dir, source_fold):
    train_path = os.path.join(split10_dir, f'fold{source_fold}_train.json')
    test_path = os.path.join(split10_dir, f'fold{source_fold}_test.json')

    if not os.path.exists(train_path):
        raise FileNotFoundError(f'找不到檔案: {train_path}')
    if not os.path.exists(test_path):
        raise FileNotFoundError(f'找不到檔案: {test_path}')

    merged_documents = load_json(train_path) + load_json(test_path)

    unique_by_doc_id = {}
    for document in merged_documents:
        doc_id = str(document.get('doc_id', ''))
        if not doc_id:
            raise ValueError('發現缺少 doc_id 的文件，無法還原完整資料集。')
        if doc_id in unique_by_doc_id:
            raise ValueError(
                f'在 fold{source_fold} 還原全集時發現重複 doc_id={doc_id}，'
                '代表 train/test 不是乾淨的不重疊切分。'
            )
        unique_by_doc_id[doc_id] = document

    return sorted(unique_by_doc_id.values(), key=doc_id_key)


def save_reconstructed_source(documents, reconstructed_dir, source_fold):
    os.makedirs(reconstructed_dir, exist_ok=True)
    source_json_path = os.path.join(reconstructed_dir, f'full_documents_from_fold{source_fold}.json')
    save_json(documents, source_json_path)
    return source_json_path


def write_supervised_fold(output_dir, fold_num, train_docs, test_docs):
    save_json(train_docs, os.path.join(output_dir, f'fold{fold_num}_train.json'))
    save_json(test_docs, os.path.join(output_dir, f'fold{fold_num}_test.json'))


def write_self_training_fold(output_dir, fold_num, train_docs, val_docs, test_docs, unlabeled_docs):
    save_json(train_docs, os.path.join(output_dir, f'fold{fold_num}_train.json'))
    save_json(val_docs, os.path.join(output_dir, f'fold{fold_num}_val.json'))
    save_json(test_docs, os.path.join(output_dir, f'fold{fold_num}_test.json'))
    save_json(
        [convert_to_unlabeled(doc) for doc in unlabeled_docs],
        os.path.join(output_dir, f'fold{fold_num}_unlabeled.json'),
    )


def ensure_pairwise_disjoint(role_name, fold_to_doc_ids):
    seen_doc_ids = set()
    for fold_num in sorted(fold_to_doc_ids):
        current_doc_ids = fold_to_doc_ids[fold_num]
        overlap_doc_ids = seen_doc_ids & current_doc_ids
        if overlap_doc_ids:
            sample_doc_ids = ', '.join(sorted(overlap_doc_ids)[:5])
            raise ValueError(
                f'{role_name} 在不同 folds 之間發生重疊。'
                f' fold{fold_num} 至少包含這些重複 doc_id: {sample_doc_ids}'
            )
        seen_doc_ids.update(current_doc_ids)
        
        
# total_doc_ids: 所有文件的 doc_id 集合
# fold_records: 前面切分時整理好的每一折資料，裡面放了 supervised_train、supervised_test、self_train、self_val、self_test、self_unlabeled
def validate_outputs(total_doc_ids, fold_records):
    self_train_doc_ids = {}
    self_val_doc_ids = {}
    self_test_doc_ids = {}
    # fold_num 如果為1，record 就是「這一折的所有資料內容」
    for fold_num, record in sorted(fold_records.items()):
        print(f'\n驗證 fold{fold_num} 的切分結果...')
        supervised_train_ids = set(get_doc_ids(record['supervised_train']))
        supervised_test_ids = set(get_doc_ids(record['supervised_test']))
        self_train_ids = set(get_doc_ids(record['self_train']))
        self_val_ids = set(get_doc_ids(record['self_val']))
        self_test_ids = set(get_doc_ids(record['self_test']))
        self_unlabeled_ids = set(get_doc_ids(record['self_unlabeled']))
        
        if fold_num == 1:
            print(f'supervised_train_ids: {[doc_id for doc_id in supervised_train_ids]}')
            print(f'supervised_test_ids: {[doc_id for doc_id in supervised_test_ids]}')
            print(f'self_train_ids: {[doc_id for doc_id in self_train_ids]}')
            print(f'self_val_ids: {[doc_id for doc_id in self_val_ids]}')
            print(f'self_test_ids: {[doc_id for doc_id in self_test_ids]}')
            

        if supervised_test_ids != self_test_ids:
            raise ValueError(f'fold{fold_num} 的 supervised test 與 self-training test 不一致')
        # & 是集合交集
        if supervised_train_ids & supervised_test_ids:
            raise ValueError(f'fold{fold_num} 的 supervised train/test 發生重疊')
        if supervised_train_ids | supervised_test_ids != total_doc_ids:
            raise ValueError(f'fold{fold_num} 的 supervised train/test 無法完整覆蓋全集')
        # | 是集合聯集
        self_partition_ids = self_train_ids | self_val_ids | self_test_ids | self_unlabeled_ids
        if self_partition_ids != total_doc_ids:
            raise ValueError(f'fold{fold_num} 的 self-training 四個子集合無法完整覆蓋全集')

        if self_train_ids & self_val_ids:
            raise ValueError(f'fold{fold_num} 的 self-training train/val 發生重疊')
        if self_train_ids & self_test_ids:
            raise ValueError(f'fold{fold_num} 的 self-training train/test 發生重疊')
        if self_train_ids & self_unlabeled_ids:
            raise ValueError(f'fold{fold_num} 的 self-training train/unlabeled 發生重疊')
        if self_val_ids & self_test_ids:
            raise ValueError(f'fold{fold_num} 的 self-training val/test 發生重疊')
        if self_val_ids & self_unlabeled_ids:
            raise ValueError(f'fold{fold_num} 的 self-training val/unlabeled 發生重疊')
        if self_test_ids & self_unlabeled_ids:
            raise ValueError(f'fold{fold_num} 的 self-training test/unlabeled 發生重疊')

        self_train_doc_ids[fold_num] = self_train_ids
        self_val_doc_ids[fold_num] = self_val_ids
        self_test_doc_ids[fold_num] = self_test_ids

    ensure_pairwise_disjoint('self-training 的 train', self_train_doc_ids)
    ensure_pairwise_disjoint('self-training 的 val', self_val_doc_ids)
    ensure_pairwise_disjoint('self-training 的 test', self_test_doc_ids)


def generate_aligned_splits(documents, supervised_output_dir, self_training_output_dir, n_splits, seed):
    all_blocks = build_disjoint_blocks(
        documents,
        n_splits=n_splits,
        random_state=seed,
        verbose=True,
    )
    total_doc_ids = set(get_doc_ids(documents))
    fold_records = {}

    for fold_index in range(n_splits):
        fold_num = fold_index + 1
        # 依照目前是第幾折，將 all_blocks 做「輪替」(rotation)
        # 例如：
        # fold1 -> [B0, B1, B2, B3, ..., B9]
        # fold2 -> [B1, B2, B3, ..., B9, B0]
        # fold3 -> [B2, B3, B4, ..., B9, B0, B1]
        # 這樣每一折都能用不同的 block 來扮演 train / val / test / unlabeled
        rotated_blocks = get_rotated_blocks(all_blocks, fold_index, verbose=True)

        # self-training 資料切分規則：
        # rotated_blocks[0] -> train (10%)
        # rotated_blocks[1] -> val (10%)
        # rotated_blocks[2] -> test (10%)
        # rotated_blocks[3:] -> unlabeled (剩下 70%)
        # flatten_blocks 的作用是把「block 陣列」展平成單一文件列表
        self_train_docs = flatten_blocks(rotated_blocks[0:1])
        # print(f"self_train_docs: {[doc['doc_id'] for doc in self_train_docs]}")
        # print(f"self_train_docs count: {len(self_train_docs)}")
        self_val_docs = flatten_blocks(rotated_blocks[1:2])
        self_test_docs = flatten_blocks(rotated_blocks[2:3])
        self_unlabeled_docs = flatten_blocks(rotated_blocks[3:])

        # supervised 資料切分規則：
        # 目標是 90% train + 10% test，且 test 要和 self-training 的 test 完全一致
        # 因此 supervised test 直接使用上面同一個 self_test_docs
        # supervised train 則把除了 test block 以外的其餘 block 全部合併：
        #   - self-training 的 train block
        #   - self-training 的 val block
        #   - self-training 的 unlabeled blocks
        # 也就是 [0:2] + [3:]，唯獨不包含 [2] 這個 test block
        supervised_train_docs = flatten_blocks(list(rotated_blocks[0:2]) + list(rotated_blocks[3:]))
        supervised_test_docs = self_test_docs

        write_supervised_fold(
            supervised_output_dir,
            fold_num,
            supervised_train_docs,
            supervised_test_docs,
        )
        write_self_training_fold(
            self_training_output_dir,
            fold_num,
            self_train_docs,
            self_val_docs,
            self_test_docs,
            self_unlabeled_docs,
        )

        fold_records[fold_num] = {
            'supervised_train': supervised_train_docs,
            'supervised_test': supervised_test_docs,
            'self_train': self_train_docs,
            'self_val': self_val_docs,
            'self_test': self_test_docs,
            'self_unlabeled': self_unlabeled_docs,
        }

        print(f'Fold {fold_num} 統計:')
        print(f'  supervised train/test = {len(supervised_train_docs)}/{len(supervised_test_docs)}')
        print(
            '  self-training train/val/test/unlabeled = '
            f'{len(self_train_docs)}/{len(self_val_docs)}/{len(self_test_docs)}/{len(self_unlabeled_docs)}'
        )

    validate_outputs(total_doc_ids, fold_records)
    print('\n驗證通過：supervised 與 self-training 的 test 已對齊，且 self-training 的 train/val/test 在不同 folds 間皆不重疊')


def main():
    parser = argparse.ArgumentParser(
        description='從 split10 任一折還原完整資料集，並產生對齊的 supervised/self-training disjoint splits'
    )
    parser.add_argument('--input-dir', type=str, default='split10', help='split10 資料夾路徑')
    parser.add_argument('--source-fold', type=int, default=1, help='要拿哪一折來還原完整資料集')
    parser.add_argument(
        '--reconstructed-dir',
        type=str,
        default='split10_full_2019',
        help='還原後完整 JSON 的輸出資料夾',
    )
    parser.add_argument(
        '--supervised-output-dir',
        type=str,
        default='split10_train9_test1_aligned_disjoint_2019',
        help='90% train / 10% test 監督式資料輸出資料夾',
    )
    parser.add_argument(
        '--self-training-output-dir',
        type=str,
        default='split10_train1_val1_test1_unlabeled7_aligned_disjoint_2019',
        help='10% train / 10% val / 10% test / 70% unlabeled 自訓練資料輸出資料夾',
    )
    parser.add_argument('--n_splits', type=int, default=10, help='折數，預設 10')
    parser.add_argument('--seed', type=int, default=42, help='隨機種子，預設 42')
    args = parser.parse_args()

    documents = reconstruct_full_documents(args.input_dir, args.source_fold)
    print(f'已由 fold{args.source_fold} 還原完整資料集，共 {len(documents)} 篇文件')

    source_json_path = save_reconstructed_source(documents, args.reconstructed_dir, args.source_fold)
    print(f'完整資料集已寫出至: {source_json_path}')

    reconstructed_documents = load_json(source_json_path)
    print(f'從還原資料夾重新讀回 JSON，共 {len(reconstructed_documents)} 篇文件，開始切分')

    generate_aligned_splits(
        documents=reconstructed_documents,
        supervised_output_dir=args.supervised_output_dir,
        self_training_output_dir=args.self_training_output_dir,
        n_splits=args.n_splits,
        seed=args.seed,
    )

    print(f'\n監督式資料已輸出至: {args.supervised_output_dir}')
    print(f'自訓練資料已輸出至: {args.self_training_output_dir}')


if __name__ == '__main__':
    main()