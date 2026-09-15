#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
將 CAREL-VAE 中文 5 個 domain 合併，並輸出與 split10 aligned-disjoint 資料集
相同 JSON 格式的 10 折監督式 / 自訓練資料。

預設來源檔案：
- data/ECPE_new_dataset/society.txt
- data/ECPE_new_dataset/entertainment.txt
- data/ECPE_new_dataset/home.txt
- data/ECPE_new_dataset/education.txt
- data/ECPE_new_dataset/finance.txt

這 5 個檔案的文件數量與論文 Table 4 一致：
- home: 746
- society: 659
- finance: 263
- education: 153
- entertainment: 52

輸出：
1. carel_zh5_train9_test1_aligned_disjoint
2. carel_zh5_train1_val1_test1_unlabeled7_aligned_disjoint

設計目標：
- 先合併 5 個 domain 成單一完整 JSON
- JSON schema 與既有 split10 資料完全一致
- supervised 與 self-training 的同 fold test 完全一致
- self-training 的 train / val / test 在不同 folds 間互不重疊
"""

import argparse
import json
import os
import re

from generate_disjoint_splits import (
    build_disjoint_blocks,
    convert_to_unlabeled,
    flatten_blocks,
    get_rotated_blocks,
)


VALID_EMOTIONS = {
    'happiness',
    'sadness',
    'disgust',
    'surprise',
    'fear',
    'anger',
    'null',
}


EMOTION_ID_TO_NAME = {
    0: 'happiness',
    1: 'sadness',
    2: 'disgust',
    3: 'surprise',
    4: 'fear',
    5: 'anger',
    6: 'null',
}


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


def parse_pair_line(pair_line):
    # 先找出原始 pair 行中的所有 (emotion, cause) 配對。
    pair_matches = re.findall(r'\((\d+)\s*,\s*(\d+)\)', pair_line)

    # 用 seen_pairs 記錄已出現過的 pair，避免同一文件內重複寫入相同配對。
    seen_pairs = set()

    # 用 unique_pairs 保留去重後、且維持原始出現順序的結果。
    unique_pairs = []

    # 逐一處理每個 pair。
    for emo, cause in pair_matches:
        # 將 pair 統一轉成整數 tuple，方便放進 set 判斷是否重複。
        pair_key = (int(emo), int(cause))

        # 若這個 pair 已經出現過，就跳過，不再重複加入。
        if pair_key in seen_pairs:
            continue

        # 記錄這個 pair 已出現，並保留到輸出列表中。
        seen_pairs.add(pair_key)
        unique_pairs.append([pair_key[0], pair_key[1]])

    # 回傳去重後的 pairs。
    return unique_pairs


def normalize_emotion_field(value):
    normalized = value.strip()
    if '&' in normalized:
        normalized = normalized.split('&')[-1].strip()
    return normalized or 'null'


def normalize_emotion_category(value, doc_id, clause_id):
    normalized = normalize_emotion_field(value)
    if normalized.isdigit():
        emotion_id = int(normalized)
        if emotion_id not in EMOTION_ID_TO_NAME:
            raise ValueError(
                f'doc_id={doc_id} clause_id={clause_id} 出現未知 emotion_category_id: {normalized}'
            )
        return EMOTION_ID_TO_NAME[emotion_id]

    if normalized not in VALID_EMOTIONS:
        raise ValueError(
            f'doc_id={doc_id} clause_id={clause_id} 出現未知 emotion_category: {normalized}'
        )
    return normalized


def normalize_emotion_token(value):
    normalized = normalize_emotion_field(value)
    if normalized == '-1' or normalized.isdigit():
        return 'null'
    return normalized


def parse_clause_line(clause_line, doc_id, expected_clause_id):
    parts = clause_line.strip().split(',', 3)
    if len(parts) != 4:
        raise ValueError(
            f'doc_id={doc_id} 的子句格式異常，預期為 4 欄，實際為 {len(parts)} 欄: {clause_line.rstrip()}'
        )

    clause_id = parts[0].strip()
    emotion_category = normalize_emotion_category(parts[1], doc_id, clause_id)
    emotion_token = normalize_emotion_token(parts[2])
    clause_text = parts[3].strip().replace(' ', '')

    if not clause_id.isdigit():
        raise ValueError(f'doc_id={doc_id} 的 clause_id 不是整數: {clause_id}')
    if int(clause_id) != expected_clause_id:
        raise ValueError(
            f'doc_id={doc_id} 的子句編號不連續，預期 {expected_clause_id}，實際 {clause_id}'
        )
    return {
        'clause_id': clause_id,
        'emotion_category': emotion_category,
        'emotion_token': emotion_token,
        'clause': clause_text,
    }


def parse_domain_file(file_path):
    documents = []

    with open(file_path, 'r', encoding='utf-8') as file_obj:
        lines = file_obj.readlines()

    line_index = 0
    while line_index < len(lines):
        header_line = lines[line_index].strip()
        if not header_line:
            line_index += 1
            continue

        header_parts = header_line.split()
        if len(header_parts) != 2 or not header_parts[0].isdigit() or not header_parts[1].isdigit():
            raise ValueError(f'無法解析文件開頭: {lines[line_index].rstrip()}')

        doc_id = header_parts[0]
        doc_len = int(header_parts[1])
        line_index += 1

        if line_index >= len(lines):
            raise ValueError(f'doc_id={doc_id} 缺少 pair 行')

        pair_line = lines[line_index].strip()
        pairs = parse_pair_line(pair_line)
        line_index += 1

        clauses = []
        for expected_clause_id in range(1, doc_len + 1):
            if line_index >= len(lines):
                raise ValueError(f'doc_id={doc_id} 在讀取子句時提前結束')
            clause_line = lines[line_index]
            clauses.append(parse_clause_line(clause_line, doc_id, expected_clause_id))
            line_index += 1

        pair_clause_ids = {clause_id for pair in pairs for clause_id in pair}
        invalid_pair_ids = [clause_id for clause_id in pair_clause_ids if clause_id < 1 or clause_id > doc_len]
        if invalid_pair_ids:
            raise ValueError(
                f'doc_id={doc_id} 的 pair 超出 doc_len={doc_len}: {sorted(invalid_pair_ids)}'
            )

        documents.append(
            {
                'doc_id': doc_id,
                'doc_len': doc_len,
                'pairs': pairs,
                'clauses': clauses,
            }
        )

    return documents


def load_carel_domains(input_dir, domains):
    merged_documents = []
    per_domain_counts = {}
    seen_doc_ids = set()

    for domain in domains:
        file_path = os.path.join(input_dir, f'{domain}.txt')
        if not os.path.exists(file_path):
            raise FileNotFoundError(f'找不到 domain 檔案: {file_path}')

        domain_documents = parse_domain_file(file_path)
        per_domain_counts[domain] = len(domain_documents)

        for doc in domain_documents:
            doc_id = str(doc['doc_id'])
            if doc_id in seen_doc_ids:
                raise ValueError(
                    f'合併 domain 時發現重複 doc_id={doc_id}，請先確認原始資料是否互斥'
                )
            seen_doc_ids.add(doc_id)
            merged_documents.append(doc)

    merged_documents.sort(key=doc_id_key)
    return merged_documents, per_domain_counts


def validate_reference_schema(reference_json_path, generated_documents):
    if not reference_json_path:
        return

    reference_documents = load_json(reference_json_path)
    if not reference_documents:
        raise ValueError(f'參考檔案沒有任何文件: {reference_json_path}')
    if not generated_documents:
        raise ValueError('合併後沒有任何文件，無法驗證 schema')

    reference_doc_keys = list(reference_documents[0].keys())
    generated_doc_keys = list(generated_documents[0].keys())
    if generated_doc_keys != reference_doc_keys:
        raise ValueError(
            '生成文件的頂層 key 順序與參考資料不一致：'
            f'{generated_doc_keys} != {reference_doc_keys}'
        )

    reference_clause_keys = list(reference_documents[0]['clauses'][0].keys())
    generated_clause_keys = list(generated_documents[0]['clauses'][0].keys())
    if generated_clause_keys != reference_clause_keys:
        raise ValueError(
            '生成文件的 clause key 順序與參考資料不一致：'
            f'{generated_clause_keys} != {reference_clause_keys}'
        )


def save_merged_source(documents, output_dir, file_name):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, file_name)
    save_json(documents, output_path)
    return output_path


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


def validate_outputs(total_doc_ids, fold_records):
    self_train_doc_ids = {}
    self_val_doc_ids = {}
    self_test_doc_ids = {}

    for fold_num, record in sorted(fold_records.items()):
        supervised_train_ids = set(get_doc_ids(record['supervised_train']))
        supervised_test_ids = set(get_doc_ids(record['supervised_test']))
        self_train_ids = set(get_doc_ids(record['self_train']))
        self_val_ids = set(get_doc_ids(record['self_val']))
        self_test_ids = set(get_doc_ids(record['self_test']))
        self_unlabeled_ids = set(get_doc_ids(record['self_unlabeled']))

        if supervised_test_ids != self_test_ids:
            raise ValueError(f'fold{fold_num} 的 supervised test 與 self-training test 不一致')
        if supervised_train_ids & supervised_test_ids:
            raise ValueError(f'fold{fold_num} 的 supervised train/test 發生重疊')
        if supervised_train_ids | supervised_test_ids != total_doc_ids:
            raise ValueError(f'fold{fold_num} 的 supervised train/test 無法完整覆蓋全集')

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
        rotated_blocks = get_rotated_blocks(all_blocks, fold_index, verbose=True)

        self_train_docs = flatten_blocks(rotated_blocks[0:1])
        self_val_docs = flatten_blocks(rotated_blocks[1:2])
        self_test_docs = flatten_blocks(rotated_blocks[2:3])
        self_unlabeled_docs = flatten_blocks(rotated_blocks[3:])

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
        description='合併 CAREL 中文 5 個 domain，並產生 aligned-disjoint 監督式 / 自訓練資料集'
    )
    parser.add_argument('--input-dir', type=str, default='data/ECPE_new_dataset', help='CAREL 中文 5-domain txt 所在資料夾')
    parser.add_argument(
        '--domains',
        nargs='+',
        default=['society', 'entertainment', 'home', 'education', 'finance'],
        help='要合併的中文 domain 名稱，不需附加 .txt',
    )
    parser.add_argument(
        '--merged-dir',
        type=str,
        default='carel_zh5_full',
        help='合併後完整 JSON 的輸出資料夾',
    )
    parser.add_argument(
        '--merged-file-name',
        type=str,
        default='carel_zh5_documents.json',
        help='合併後完整 JSON 的檔名',
    )
    parser.add_argument(
        '--supervised-output-dir',
        type=str,
        default='carel_zh5_train9_test1_aligned_disjoint',
        help='90% train / 10% test 監督式資料輸出資料夾',
    )
    parser.add_argument(
        '--self-training-output-dir',
        type=str,
        default='carel_zh5_train1_val1_test1_unlabeled7_aligned_disjoint',
        help='10% train / 10% val / 10% test / 70% unlabeled 自訓練資料輸出資料夾',
    )
    parser.add_argument(
        '--reference-json',
        type=str,
        default='split10_train1_val1_test1_unlabeled7_aligned_disjoint_2019/fold1_train.json',
        help='用來驗證 schema 與 key 順序的參考 JSON',
    )
    parser.add_argument('--n_splits', type=int, default=10, help='折數，預設 10')
    parser.add_argument('--seed', type=int, default=42, help='隨機種子，預設 42')
    args = parser.parse_args()

    merged_documents, per_domain_counts = load_carel_domains(args.input_dir, args.domains)
    print('已讀取 CAREL 中文 domains：')
    for domain in args.domains:
        print(f'  {domain}: {per_domain_counts[domain]} 篇文件')
    print(f'合併後總文件數: {len(merged_documents)}')

    validate_reference_schema(args.reference_json, merged_documents)
    print(f'已確認 schema 與 key 順序對齊參考檔案: {args.reference_json}')

    merged_json_path = save_merged_source(merged_documents, args.merged_dir, args.merged_file_name)
    print(f'完整合併資料已寫出至: {merged_json_path}')

    merged_documents_reloaded = load_json(merged_json_path)
    print(f'從合併 JSON 重新讀回 {len(merged_documents_reloaded)} 篇文件，開始切分')

    generate_aligned_splits(
        documents=merged_documents_reloaded,
        supervised_output_dir=args.supervised_output_dir,
        self_training_output_dir=args.self_training_output_dir,
        n_splits=args.n_splits,
        seed=args.seed,
    )

    print(f'\n監督式資料已輸出至: {args.supervised_output_dir}')
    print(f'自訓練資料已輸出至: {args.self_training_output_dir}')


if __name__ == '__main__':
    main()