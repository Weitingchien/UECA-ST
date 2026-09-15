#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
計算兩份 JSON 資料集的重複比例

重複定義：
1. 文件全文內容完全相同
2. pairs 完全相同
3. 不比較 emotion_token
4. 不比較 emotion_category

其中 pairs 會先去除重複，再排序後比較，
所以像 [[14, 13], [14, 13]] 會視為 [[14, 13]]
"""

import argparse
import json
import os


# 讀取 JSON 檔案
def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file_obj:
        return json.load(file_obj)


# 把資料寫成 JSON 檔案
def save_json(data, file_path):
    output_dir = os.path.dirname(file_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(file_path, 'w', encoding='utf-8') as file_obj:
        json.dump(data, file_obj, ensure_ascii=False, indent=2)


# 將 doc_id 轉成適合排序的 key
def doc_id_sort_key(doc_id):
    text = str(doc_id)
    if text.isdigit():
        return (0, int(text))
    return (1, text)


# 將 pairs 正規化成整數，並去除重複 pair
def normalize_unique_pairs(pairs):
    seen_pairs = set()
    unique_pairs = []

    for pair in pairs or []:
        left_value = int(pair[0]) if str(pair[0]).isdigit() else pair[0]
        right_value = int(pair[1]) if str(pair[1]).isdigit() else pair[1]
        pair_key = (left_value, right_value)

        if pair_key in seen_pairs:
            continue

        seen_pairs.add(pair_key)
        unique_pairs.append([pair_key[0], pair_key[1]])

    unique_pairs.sort(key=lambda item: (item[0], item[1]))
    return unique_pairs


# 取出文件全文內容，保留子句順序
def get_clause_texts(document):
    clause_texts = []
    for clause in document.get('clauses', []):
        clause_texts.append(str(clause.get('clause', '')))
    return tuple(clause_texts)


# 用來判斷重複的文件signature
def build_document_signature(document):
    clause_texts = get_clause_texts(document)
    pairs = tuple((pair[0], pair[1]) for pair in normalize_unique_pairs(document.get('pairs', [])))
    return clause_texts, pairs


# 建立 signature -> doc_id 清單的對照表
def build_signature_map(documents):
    signature_map = {}

    for document in documents:
        doc_id = str(document.get('doc_id', ''))
        if not doc_id:
            raise ValueError('發現缺少 doc_id 的文件，無法比較。')

        signature = build_document_signature(document)
        if signature not in signature_map:
            signature_map[signature] = []
        signature_map[signature].append(doc_id)

    for signature in signature_map:
        signature_map[signature].sort(key=doc_id_sort_key)

    return signature_map


# 建立 doc_id -> document 的對照表，方便同 doc_id 比較
def build_doc_id_map(documents, source_name):
    doc_map = {}

    for document in documents:
        doc_id = str(document.get('doc_id', ''))
        if not doc_id:
            raise ValueError(f'{source_name} 發現缺少 doc_id 的文件，無法比較。')

        if doc_id in doc_map:
            raise ValueError(f'{source_name} 內出現重複 doc_id={doc_id}。')

        doc_map[doc_id] = document

    return doc_map


# 依照 signature 配對兩份資料集中的重複文件
def match_duplicate_documents(left_map, right_map):
    matched_doc_pairs = []
    matched_left_ids = []
    matched_right_ids = []

    common_signatures = set(left_map.keys()) & set(right_map.keys())

    for signature in common_signatures:
        left_ids = left_map[signature]
        right_ids = right_map[signature]
        match_count = min(len(left_ids), len(right_ids))

        for index in range(match_count):
            matched_doc_pairs.append({
                'left_doc_id': left_ids[index],
                'right_doc_id': right_ids[index],
                'pairs': [list(pair) for pair in signature[1]],
            })
            matched_left_ids.append(left_ids[index])
            matched_right_ids.append(right_ids[index])

    matched_left_ids.sort(key=doc_id_sort_key)
    matched_right_ids.sort(key=doc_id_sort_key)
    matched_doc_pairs.sort(
        key=lambda item: (doc_id_sort_key(item['left_doc_id']), doc_id_sort_key(item['right_doc_id']))
    )

    return matched_doc_pairs, matched_left_ids, matched_right_ids


# 找出沒有被配對到的 doc_id
def find_unmatched_doc_ids(all_doc_ids, matched_doc_ids):
    matched_set = set(matched_doc_ids)
    unmatched_doc_ids = []

    for doc_id in all_doc_ids:
        if doc_id not in matched_set:
            unmatched_doc_ids.append(doc_id)

    unmatched_doc_ids.sort(key=doc_id_sort_key)
    return unmatched_doc_ids


# 取得某篇文件的 clauses
def get_clauses(document):
    clauses = document.get('clauses', [])
    if not isinstance(clauses, list):
        raise ValueError(f'doc_id={document.get("doc_id")} 的 clauses 不是 list。')
    return clauses


# 比較同 doc_id 文件的全文內容，列出差異句位與完整句子
def compare_document_text(left_doc, right_doc):
    left_clauses = get_clauses(left_doc)
    right_clauses = get_clauses(right_doc)
    differences = []

    total = max(len(left_clauses), len(right_clauses))
    for index in range(total):
        left_clause = left_clauses[index] if index < len(left_clauses) else None
        right_clause = right_clauses[index] if index < len(right_clauses) else None

        left_clause_id = '' if left_clause is None else str(left_clause.get('clause_id', ''))
        right_clause_id = '' if right_clause is None else str(right_clause.get('clause_id', ''))
        left_text = '' if left_clause is None else str(left_clause.get('clause', ''))
        right_text = '' if right_clause is None else str(right_clause.get('clause', ''))

        # 只要有缺句、clause_id 不同、或句子內容不同，就視為文字差異
        if left_clause is None or right_clause is None or left_clause_id != right_clause_id or left_text != right_text:
            differences.append({
                'position': index + 1,
                'left_clause_id': left_clause_id,
                'right_clause_id': right_clause_id,
                'left_text': left_text,
                'right_text': right_text,
            })

    return differences


# 比較同 doc_id 文件的 pairs 是否不同
def compare_document_pairs(left_doc, right_doc):
    left_pairs = normalize_unique_pairs(left_doc.get('pairs', []))
    right_pairs = normalize_unique_pairs(right_doc.get('pairs', []))
    return left_pairs, right_pairs, left_pairs != right_pairs


# 收集同 doc_id 的內容與 pairs 差異文檔
def collect_doc_id_mismatches(left_doc_map, right_doc_map):
    mismatch_records = []
    overlap_doc_ids = sorted(set(left_doc_map.keys()) & set(right_doc_map.keys()), key=doc_id_sort_key)

    for doc_id in overlap_doc_ids:
        left_doc = left_doc_map[doc_id]
        right_doc = right_doc_map[doc_id]
        text_differences = compare_document_text(left_doc, right_doc)
        left_pairs, right_pairs, pair_changed = compare_document_pairs(left_doc, right_doc)

        if not text_differences and not pair_changed:
            continue

        mismatch_records.append({
            'doc_id': doc_id,
            'has_text_difference': bool(text_differences),
            'has_pair_difference': pair_changed,
            'left_pairs': left_pairs,
            'right_pairs': right_pairs,
            'text_differences': text_differences,
        })

    return mismatch_records


# 把 doc_id 分段，避免一行太長
def chunk_list(items, chunk_size):
    chunks = []
    start = 0

    while start < len(items):
        end = start + chunk_size
        chunks.append(items[start:end])
        start = end

    return chunks


# 輸出摘要項目與對應 doc_id
def append_summary_item(lines, label, doc_ids):
    lines.append(f'- {label}: {len(doc_ids)}')

    if not doc_ids:
        lines.append('  doc_id: 無')
        return

    for index, group in enumerate(chunk_list(doc_ids, 30)):
        if index == 0:
            lines.append(f'  doc_id: {", ".join(group)}')
        else:
            lines.append(f'          {", ".join(group)}')


# 建立同 doc_id 的內容與 pairs 差異報告
def build_doc_id_mismatch_report(args, overlap_doc_ids, left_only_doc_ids, right_only_doc_ids, mismatch_records):
    lines = []

    text_mismatch_doc_ids = [record['doc_id'] for record in mismatch_records if record['has_text_difference']]
    pair_mismatch_doc_ids = [record['doc_id'] for record in mismatch_records if record['has_pair_difference']]
    both_mismatch_doc_ids = [
        record['doc_id']
        for record in mismatch_records
        if record['has_text_difference'] and record['has_pair_difference']
    ]

    lines.append('兩份資料集同 doc_id 的內容與 pairs 差異報告')
    lines.append('')
    lines.append('比較規則：比較 clauses 的完整文字內容與句子位置，並比較去重後的 pairs。')
    lines.append('不比較 emotion_token，也不比較 emotion_category。')
    lines.append('pairs 會先去除重複再比較，所以 [[14, 13], [14, 13]] 視為 [[14, 13]]。')
    lines.append('')
    lines.append(f'左側資料集: {args.left_path}')
    lines.append(f'右側資料集: {args.right_path}')
    lines.append('')
    lines.append('摘要')
    lines.append(f'- 兩份資料集共有的 doc_id 數量: {len(overlap_doc_ids)}')
    append_summary_item(lines, '左側資料集獨有、無法做逐句比較的 doc_id', left_only_doc_ids)
    append_summary_item(lines, '右側資料集獨有、無法做逐句比較的 doc_id', right_only_doc_ids)
    append_summary_item(lines, '同 doc_id 但內容或 pairs 不完全相同的文檔數量', [record['doc_id'] for record in mismatch_records])
    append_summary_item(lines, '同 doc_id 但全文內容不完全相同的文檔數量', text_mismatch_doc_ids)
    append_summary_item(lines, '同 doc_id 但 pairs 不完全相同的文檔數量', pair_mismatch_doc_ids)
    append_summary_item(lines, '同 doc_id 且內容與 pairs 都不同的文檔數量', both_mismatch_doc_ids)
    lines.append('')
    lines.append('逐篇差異明細')

    if not mismatch_records:
        lines.append('無')
        return '\n'.join(lines)

    for record in mismatch_records:
        lines.append('')
        lines.append(f"doc_id={record['doc_id']}")
        lines.append(f"has_text_difference: {record['has_text_difference']}")
        lines.append(f"has_pair_difference: {record['has_pair_difference']}")

        if record['has_pair_difference']:
            lines.append(f"left_pairs: {record['left_pairs']}")
            lines.append(f"right_pairs: {record['right_pairs']}")

        if record['has_text_difference']:
            for difference in record['text_differences']:
                lines.append(f"- 第 {difference['position']} 句")
                lines.append(f"  左側 clause_id: {difference['left_clause_id'] or '缺少'}")
                lines.append(f"  左側完整內容: {difference['left_text'] or '缺少這一句'}")
                lines.append(f"  右側 clause_id: {difference['right_clause_id'] or '缺少'}")
                lines.append(f"  右側完整內容: {difference['right_text'] or '缺少這一句'}")
        else:
            lines.append('全文內容相同。')

    return '\n'.join(lines)


# 建立文字報告。
def build_text_report(args, matched_left_ids, matched_right_ids, unmatched_left_ids, unmatched_right_ids, matched_doc_pairs):
    lines = []

    left_total = args.left_count
    right_total = args.right_count
    duplicate_count = len(matched_doc_pairs)

    left_ratio = duplicate_count / left_total if left_total else 0.0
    right_ratio = duplicate_count / right_total if right_total else 0.0

    lines.append('兩份資料集的重複比例報告')
    lines.append('')
    lines.append('重複判定規則：全文內容完全相同，且 pairs 完全相同。')
    lines.append('比較時忽略 emotion_token，亦不比較 emotion_category。')
    lines.append('pairs 會先去除重複再比較，所以 [[14, 13], [14, 13]] 視為 [[14, 13]]。')
    lines.append('')
    lines.append(f'左側資料集: {args.left_path}')
    lines.append(f'右側資料集: {args.right_path}')
    lines.append('')
    lines.append('摘要')
    lines.append(f'- 左側資料集文檔總數: {left_total}')
    lines.append(f'- 右側資料集文檔總數: {right_total}')
    lines.append(f'- 完全重複的文檔數量: {duplicate_count}')
    lines.append(f'- 左側資料集的重複比例: {left_ratio:.4%}')
    lines.append(f'- 右側資料集的重複比例: {right_ratio:.4%}')
    append_summary_item(lines, '左側資料集中可在右側找到完全重複的 doc_id', matched_left_ids)
    append_summary_item(lines, '右側資料集中可在左側找到完全重複的 doc_id', matched_right_ids)
    append_summary_item(lines, '左側資料集中找不到完全重複對象的 doc_id', unmatched_left_ids)
    append_summary_item(lines, '右側資料集中找不到完全重複對象的 doc_id', unmatched_right_ids)
    lines.append('')
    lines.append('完全重複的文件配對')

    if not matched_doc_pairs:
        lines.append('無')
    else:
        for pair in matched_doc_pairs:
            lines.append(
                f"left_doc_id={pair['left_doc_id']}  right_doc_id={pair['right_doc_id']}  pairs={pair['pairs']}"
            )

    return '\n'.join(lines)


# 主程式
def main():
    parser = argparse.ArgumentParser(description='計算兩份資料集以全文與 pairs 為準的重複比例')
    parser.add_argument(
        '--left_path',
        default='carel_zh5_full/carel_zh5_documents.json',
        help='左側資料集 JSON 路徑',
    )
    parser.add_argument(
        '--right_path',
        default='split10_full_2019/full_documents_from_fold1.json',
        help='右側資料集 JSON 路徑',
    )
    parser.add_argument(
        '--output_path',
        default='debug/carel_vs_split10_text_pairs_duplicate_report.txt',
        help='文字報告輸出路徑',
    )
    parser.add_argument(
        '--json_output_path',
        default='debug/carel_vs_split10_text_pairs_duplicate_summary.json',
        help='JSON 摘要輸出路徑',
    )
    parser.add_argument(
        '--mismatch_output_path',
        default='debug/carel_vs_split10_text_mismatch_by_doc_id_report.txt',
        help='同 doc_id 內容與 pairs 差異文字報告輸出路徑',
    )
    parser.add_argument(
        '--mismatch_json_output_path',
        default='debug/carel_vs_split10_text_mismatch_by_doc_id_summary.json',
        help='同 doc_id 內容與 pairs 差異 JSON 摘要輸出路徑',
    )
    args = parser.parse_args()

    left_documents = load_json(args.left_path)
    right_documents = load_json(args.right_path)

    left_doc_ids = sorted([str(doc.get('doc_id', '')) for doc in left_documents], key=doc_id_sort_key)
    right_doc_ids = sorted([str(doc.get('doc_id', '')) for doc in right_documents], key=doc_id_sort_key)

    left_map = build_signature_map(left_documents)
    right_map = build_signature_map(right_documents)
    left_doc_map = build_doc_id_map(left_documents, '左側資料集')
    right_doc_map = build_doc_id_map(right_documents, '右側資料集')

    matched_doc_pairs, matched_left_ids, matched_right_ids = match_duplicate_documents(left_map, right_map)
    unmatched_left_ids = find_unmatched_doc_ids(left_doc_ids, matched_left_ids)
    unmatched_right_ids = find_unmatched_doc_ids(right_doc_ids, matched_right_ids)

    overlap_doc_ids = sorted(set(left_doc_map.keys()) & set(right_doc_map.keys()), key=doc_id_sort_key)
    left_only_doc_ids = sorted(set(left_doc_map.keys()) - set(right_doc_map.keys()), key=doc_id_sort_key)
    right_only_doc_ids = sorted(set(right_doc_map.keys()) - set(left_doc_map.keys()), key=doc_id_sort_key)
    doc_id_mismatch_records = collect_doc_id_mismatches(left_doc_map, right_doc_map)

    args.left_count = len(left_documents)
    args.right_count = len(right_documents)

    report_text = build_text_report(
        args,
        matched_left_ids,
        matched_right_ids,
        unmatched_left_ids,
        unmatched_right_ids,
        matched_doc_pairs,
    )

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, 'w', encoding='utf-8') as file_obj:
        file_obj.write(report_text)

    mismatch_report_text = build_doc_id_mismatch_report(
        args,
        overlap_doc_ids,
        left_only_doc_ids,
        right_only_doc_ids,
        doc_id_mismatch_records,
    )
    os.makedirs(os.path.dirname(args.mismatch_output_path), exist_ok=True)
    with open(args.mismatch_output_path, 'w', encoding='utf-8') as file_obj:
        file_obj.write(mismatch_report_text)

    duplicate_count = len(matched_doc_pairs)
    left_ratio = duplicate_count / len(left_documents) if left_documents else 0.0
    right_ratio = duplicate_count / len(right_documents) if right_documents else 0.0

    summary = {
        'left_path': args.left_path,
        'right_path': args.right_path,
        'left_count': len(left_documents),
        'right_count': len(right_documents),
        'duplicate_count': duplicate_count,
        'left_duplicate_ratio': left_ratio,
        'right_duplicate_ratio': right_ratio,
        'matched_left_doc_ids': matched_left_ids,
        'matched_right_doc_ids': matched_right_ids,
        'unmatched_left_doc_ids': unmatched_left_ids,
        'unmatched_right_doc_ids': unmatched_right_ids,
        'matched_doc_pairs': matched_doc_pairs,
    }
    save_json(summary, args.json_output_path)

    text_mismatch_doc_ids = [record['doc_id'] for record in doc_id_mismatch_records if record['has_text_difference']]
    pair_mismatch_doc_ids = [record['doc_id'] for record in doc_id_mismatch_records if record['has_pair_difference']]
    both_mismatch_doc_ids = [
        record['doc_id']
        for record in doc_id_mismatch_records
        if record['has_text_difference'] and record['has_pair_difference']
    ]

    mismatch_summary = {
        'left_path': args.left_path,
        'right_path': args.right_path,
        'overlap_doc_id_count': len(overlap_doc_ids),
        'left_only_doc_ids': left_only_doc_ids,
        'right_only_doc_ids': right_only_doc_ids,
        'any_mismatch_count': len(doc_id_mismatch_records),
        'any_mismatch_doc_ids': [record['doc_id'] for record in doc_id_mismatch_records],
        'text_mismatch_count': len(text_mismatch_doc_ids),
        'text_mismatch_doc_ids': text_mismatch_doc_ids,
        'pair_mismatch_count': len(pair_mismatch_doc_ids),
        'pair_mismatch_doc_ids': pair_mismatch_doc_ids,
        'both_mismatch_count': len(both_mismatch_doc_ids),
        'both_mismatch_doc_ids': both_mismatch_doc_ids,
        'mismatch_records': doc_id_mismatch_records,
    }
    save_json(mismatch_summary, args.mismatch_json_output_path)

    print(f'左側資料集文檔總數: {len(left_documents)}')
    print(f'右側資料集文檔總數: {len(right_documents)}')
    print(f'完全重複的文檔數量: {duplicate_count}')
    print(f'左側資料集的重複比例: {left_ratio:.4%}')
    print(f'右側資料集的重複比例: {right_ratio:.4%}')
    print(f'同 doc_id 但內容或 pairs 不完全相同的文檔數量: {len(doc_id_mismatch_records)}')
    print(f'同 doc_id 但全文內容不完全相同的文檔數量: {len(text_mismatch_doc_ids)}')
    print(f'同 doc_id 但 pairs 不完全相同的文檔數量: {len(pair_mismatch_doc_ids)}')
    print(f'文字報告已輸出到: {args.output_path}')
    print(f'JSON 摘要已輸出到: {args.json_output_path}')
    print(f'全文差異報告已輸出到: {args.mismatch_output_path}')
    print(f'全文差異 JSON 摘要已輸出到: {args.mismatch_json_output_path}')


if __name__ == '__main__':
    main()