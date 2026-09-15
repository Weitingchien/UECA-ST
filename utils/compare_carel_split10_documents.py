#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
比較 CAREL 與 split10 還原全集 JSON 的重疊文檔與逐句差異。

功能重點：
1. 找出兩份資料中 doc_id 重疊的文檔。
2. 判斷重疊文檔是完全相同，還是只有部分內容不同。
3. 若內容不同，列出是第幾句不同，並說明差在哪裡。
4. 額外比較 pairs、emotion_category、emotion_token 是否不同。

預設輸入：
- carel_zh5_full/carel_zh5_documents.json
- split10_full_2019/full_documents_from_fold1.json

預設輸出：
- debug/carel_vs_split10_doc_diff_report.txt
"""

import argparse
import json
import os


# 讀取 JSON 檔案，並回傳 Python 的資料結構。
def load_json(file_path):
    # 用 UTF-8 開啟檔案，避免中文內容出現亂碼。
    with open(file_path, 'r', encoding='utf-8') as file_obj:
        # 把 JSON 內容轉成 list / dict。
        return json.load(file_obj)


# 把資料寫回 JSON 檔案。
def save_json(data, file_path):
    # 先確保輸出資料夾存在。
    output_dir = os.path.dirname(file_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # 以 UTF-8 寫檔，並保留中文原文。
    with open(file_path, 'w', encoding='utf-8') as file_obj:
        # indent=2 讓輸出的 JSON 比較好閱讀。
        json.dump(data, file_obj, ensure_ascii=False, indent=2)


# 將句子位置統一轉成可比較、可排序的型別。
def normalize_position(value):
    # 若原值是純數字，就轉成整數，方便後續排序與比較。
    if str(value).isdigit():
        return int(value)

    # 若不是純數字，就保留成字串。
    return str(value)


# 判斷欄位是否為真正有值，而不是 null / None / 空字串。
def has_meaningful_value(value):
    # None 視為沒有值。
    if value is None:
        return False

    # 統一轉成字串後再判斷。
    text = str(value).strip().lower()

    # 常見的空值表示都視為沒有值。
    if text in ('', 'null', 'none'):
        return False

    # 其餘情況視為有值。
    return True


# 將 doc_id 轉成適合排序的 key。
def doc_id_sort_key(doc_id):
    # 先把 doc_id 統一轉成字串，避免 int / str 混用。
    doc_id_text = str(doc_id)

    # 若 doc_id 是純數字，就用整數排序，避免 1, 10, 2 這種情況。
    if doc_id_text.isdigit():
        return (0, int(doc_id_text))

    # 若不是純數字，就退回字串排序。
    return (1, doc_id_text)


# 將句子位置轉成適合排序的 key。
def position_sort_key(position):
    # 若是整數句位，直接排在前面。
    if isinstance(position, int):
        return (0, position)

    # 其餘句位用字串排序。
    return (1, str(position))


# 把文件列表轉成 doc_id -> document 的對照表。
def build_doc_map(documents, source_name):
    # 建立空字典，後續用來快速依 doc_id 查詢文件。
    doc_map = {}

    # 逐篇處理文件。
    for document in documents:
        # 讀出 doc_id，並統一轉成字串。
        doc_id = str(document.get('doc_id', ''))

        # 若缺少 doc_id，直接報錯，因為後面無法對齊文件。
        if not doc_id:
            raise ValueError(f'{source_name} 發現缺少 doc_id 的文件，無法比較。')

        # 若同一份檔案內出現重複 doc_id，也直接報錯。
        if doc_id in doc_map:
            raise ValueError(f'{source_name} 內出現重複 doc_id={doc_id}。')

        # 將文件放入字典，方便後續快速查找。
        doc_map[doc_id] = document

    # 回傳建立好的對照表。
    return doc_map


# 將 clauses 取出成列表。
def get_clauses(document):
    # 若沒有 clauses 欄位，就回傳空列表，避免後續程式壞掉。
    clauses = document.get('clauses', [])

    # 若 clauses 不是列表，代表資料格式不符預期。
    if not isinstance(clauses, list):
        raise ValueError(f'doc_id={document.get("doc_id")} 的 clauses 不是 list。')

    # 回傳 clauses 列表。
    return clauses


# 將 pairs 正規化，避免 int / str 混用造成看起來不同、實際相同的情況。
def normalize_pairs(pairs):
    # 建立新的 pairs 列表。
    normalized = []

    # 逐一處理每個 pair。
    for pair in pairs:
        # 預設先保留原值，若能轉成整數再轉。
        left_value = pair[0]
        right_value = pair[1]

        # 若左邊是純數字字串，就轉成整數。
        if str(left_value).isdigit():
            left_value = int(left_value)

        # 若右邊是純數字字串，就轉成整數。
        if str(right_value).isdigit():
            right_value = int(right_value)

        # 把正規化後的 pair 放進新列表。
        normalized.append([left_value, right_value])

    # 回傳正規化後的 pairs。
    return normalized


# 將 pairs 排序，避免因順序不同但內容相同而誤判。
def sort_pairs(pairs):
    # 依 pair 的左值、右值排序，讓比較結果固定。
    return sorted(
        pairs,
        key=lambda pair: (position_sort_key(pair[0]), position_sort_key(pair[1])),
    )


# 將 pairs 去重後再排序，讓 [[14, 13], [14, 13]] 視為 [[14, 13]]。
def normalize_unique_pairs(pairs):
    # 先建立集合記錄已出現過的 pair，避免重複保留。
    seen_pairs = set()

    # 建立去重後的 pair 列表。
    unique_pairs = []

    # 先做型別正規化，再逐一去重。
    for pair in normalize_pairs(pairs):
        pair_key = (pair[0], pair[1])
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)
        unique_pairs.append([pair_key[0], pair_key[1]])

    # 回傳排序後的唯一 pairs。
    return sort_pairs(unique_pairs)


# 取得文件中的情緒子句句位。
def get_emotion_clause_positions(document):
    # 建立句位集合，避免重複。
    positions = set()

    # 逐句檢查是否有情緒標註。
    for clause in get_clauses(document):
        emotion_category = clause.get('emotion_category')
        emotion_token = clause.get('emotion_token')

        # 只要 emotion_category 或 emotion_token 任一個有值，就視為情緒子句。
        if has_meaningful_value(emotion_category) or has_meaningful_value(emotion_token):
            positions.add(normalize_position(clause.get('clause_id', '')))

    # 回傳排序後的句位列表。
    return sorted(positions, key=position_sort_key)


# 取得文件中的原因子句句位。
def get_cause_clause_positions(document):
    # 建立句位集合，避免重複。
    positions = set()

    # pairs 的第二個值就是原因子句的位置。
    for pair in normalize_pairs(document.get('pairs', [])):
        positions.add(normalize_position(pair[1]))

    # 回傳排序後的句位列表。
    return sorted(positions, key=position_sort_key)


# 取得只出現在某一邊的句位。
def diff_positions(list_a, list_b):
    # 先轉成集合，方便做差集。
    set_a = set(list_a)
    set_b = set(list_b)

    # 回傳只在 A、只在 B 的句位清單。
    only_in_a = sorted(set_a - set_b, key=position_sort_key)
    only_in_b = sorted(set_b - set_a, key=position_sort_key)
    return only_in_a, only_in_b


# 建立文件前幾句的預覽文字，方便在報告中快速辨識文件內容。
def build_preview(document, max_clauses=6):
    # 先取出 clauses。
    clauses = get_clauses(document)

    # 建立預覽文字列表。
    preview_parts = []

    # 只取前 max_clauses 句，避免預覽過長。
    for clause in clauses[:max_clauses]:
        # 加入 clause 文字；若缺少 clause 欄位，就放空字串。
        preview_parts.append(str(clause.get('clause', '')))

    # 用 | 串起來，讓預覽比較好讀。
    return ' | '.join(preview_parts)


# 找出兩段文字第一個不同的位置。
def find_first_difference(text_a, text_b):
    # 先取較短字串的長度，避免索引超出範圍。
    limit = min(len(text_a), len(text_b))

    # 從前往後逐字比較。
    index = 0
    while index < limit:
        # 若發現不同字元，直接回傳位置。
        if text_a[index] != text_b[index]:
            return index
        index += 1

    # 若前面都相同，但長度不同，差異點就是較短字串的結尾。
    if len(text_a) != len(text_b):
        return limit

    # 若長度也相同，代表兩段文字完全一致。
    return -1


# 取出差異點附近的片段，方便看出哪裡不一樣。
def extract_snippet(text, diff_index, radius=12):
    # 若索引小於 0，代表沒有差異，直接回傳全文。
    if diff_index < 0:
        return text

    # 計算片段起點，避免小於 0。
    start = diff_index - radius
    if start < 0:
        start = 0

    # 計算片段終點，避免超出字串長度。
    end = diff_index + radius
    if end > len(text):
        end = len(text)

    # 截出片段。
    snippet = text[start:end]

    # 若不是從字串開頭開始，就補上省略號。
    if start > 0:
        snippet = '...' + snippet

    # 若不是到字串結尾，就補上省略號。
    if end < len(text):
        snippet = snippet + '...'

    # 回傳可閱讀的片段。
    return snippet


# 將文字差異整理成一句容易理解的描述。
def describe_text_difference(text_a, text_b):
    # 先找出第一個差異位置。
    diff_index = find_first_difference(text_a, text_b)

    # 若沒有差異，直接回傳說明文字。
    if diff_index == -1:
        return '句子文字相同。'

    # 擷取 CAREL 的差異片段。
    snippet_a = extract_snippet(text_a, diff_index)

    # 擷取 split10 的差異片段。
    snippet_b = extract_snippet(text_b, diff_index)

    # 回傳簡單清楚的差異描述。
    return (
        f'第 {diff_index + 1} 個字開始不同；'
        f'CAREL 片段：{snippet_a}；'
        f'split10 片段：{snippet_b}'
    )


# 比較兩篇文件的 clauses，找出逐句差異。
def compare_clauses(carel_doc, split_doc):
    # 先取出兩邊的 clauses。
    carel_clauses = get_clauses(carel_doc)
    split_clauses = get_clauses(split_doc)

    # 建立差異列表。
    differences = []

    # 取兩邊較大的句子數，確保不會漏掉多出來的句子。
    total = max(len(carel_clauses), len(split_clauses))

    # 依句子位置逐句比較。
    index = 0
    while index < total:
        # 預設先視為缺句，若存在再取值。
        carel_clause = None
        split_clause = None

        # 若 CAREL 這邊有這一句，就取出來。
        if index < len(carel_clauses):
            carel_clause = carel_clauses[index]

        # 若 split10 這邊有這一句，就取出來。
        if index < len(split_clauses):
            split_clause = split_clauses[index]

        # 建立單一句子的差異資訊。
        clause_difference = {
            'position': index + 1,
            'carel_clause': carel_clause,
            'split_clause': split_clause,
            'text_changed': False,
            'clause_id_changed': False,
            'emotion_category_changed': False,
            'emotion_token_changed': False,
            'text_difference_note': '',
        }

        # 若其中一邊缺句，直接視為差異。
        if carel_clause is None or split_clause is None:
            clause_difference['text_changed'] = True
            clause_difference['text_difference_note'] = '其中一邊缺少這一句。'
            differences.append(clause_difference)
            index += 1
            continue

        # 取出 clause_id，方便比較對齊是否一致。
        carel_clause_id = str(carel_clause.get('clause_id', ''))
        split_clause_id = str(split_clause.get('clause_id', ''))

        # 若 clause_id 不同，也記錄下來。
        if carel_clause_id != split_clause_id:
            clause_difference['clause_id_changed'] = True

        # 取出兩邊句子文字。
        carel_text = str(carel_clause.get('clause', ''))
        split_text = str(split_clause.get('clause', ''))

        # 若句子文字不同，就記錄差異說明。
        if carel_text != split_text:
            clause_difference['text_changed'] = True
            clause_difference['text_difference_note'] = describe_text_difference(carel_text, split_text)

        # 讀出 emotion_category。
        carel_emotion_category = str(carel_clause.get('emotion_category', ''))
        split_emotion_category = str(split_clause.get('emotion_category', ''))

        # 若 emotion_category 不同，就記錄。
        if carel_emotion_category != split_emotion_category:
            clause_difference['emotion_category_changed'] = True

        # 讀出 emotion_token。
        carel_emotion_token = str(carel_clause.get('emotion_token', ''))
        split_emotion_token = str(split_clause.get('emotion_token', ''))

        # 若 emotion_token 不同，就記錄。
        if carel_emotion_token != split_emotion_token:
            clause_difference['emotion_token_changed'] = True

        # 只要這一句有任一種差異，就收進結果中。
        has_difference = False
        if clause_difference['text_changed']:
            has_difference = True
        if clause_difference['clause_id_changed']:
            has_difference = True
        if clause_difference['emotion_category_changed']:
            has_difference = True
        if clause_difference['emotion_token_changed']:
            has_difference = True

        if has_difference:
            differences.append(clause_difference)

        # 繼續比較下一句。
        index += 1

    # 回傳所有逐句差異。
    return differences


# 比較兩篇文件，整理出文件層級的差異摘要。
def compare_documents(carel_doc, split_doc):
    # 先建立回傳用的結果字典。
    result = {
        'doc_id': str(carel_doc.get('doc_id', '')),
        'preview': build_preview(carel_doc),
        'doc_len_changed': False,
        'pairs_changed': False,
        'emotion_clause_positions_changed': False,
        'cause_clause_positions_changed': False,
        'has_text_difference': False,
        'has_annotation_position_difference': False,
        'pair_positions_carel': [],
        'pair_positions_split10': [],
        'emotion_clause_positions_carel': [],
        'emotion_clause_positions_split10': [],
        'cause_clause_positions_carel': [],
        'cause_clause_positions_split10': [],
        'clause_differences': [],
    }

    # 取出 doc_len，若沒有就以 clauses 長度補上。
    carel_doc_len = carel_doc.get('doc_len', len(get_clauses(carel_doc)))
    split_doc_len = split_doc.get('doc_len', len(get_clauses(split_doc)))

    # 若 doc_len 不同，就標記成文件層級差異。
    if carel_doc_len != split_doc_len:
        result['doc_len_changed'] = True
        result['has_text_difference'] = True

    # 取出並正規化 pairs，避免 int / str 混用。
    carel_pairs = normalize_unique_pairs(carel_doc.get('pairs', []))
    split_pairs = normalize_unique_pairs(split_doc.get('pairs', []))

    # 保存排序後的 pairs，方便報告直接輸出。
    result['pair_positions_carel'] = carel_pairs
    result['pair_positions_split10'] = split_pairs

    # 若 pairs 不同，就標記成標註差異。
    if carel_pairs != split_pairs:
        result['pairs_changed'] = True
        result['has_annotation_position_difference'] = True

    # 取出兩邊的情緒子句句位。
    carel_emotion_positions = get_emotion_clause_positions(carel_doc)
    split_emotion_positions = get_emotion_clause_positions(split_doc)
    result['emotion_clause_positions_carel'] = carel_emotion_positions
    result['emotion_clause_positions_split10'] = split_emotion_positions

    # 若情緒子句句位不同，就記錄下來。
    if carel_emotion_positions != split_emotion_positions:
        result['emotion_clause_positions_changed'] = True
        result['has_annotation_position_difference'] = True

    # 取出兩邊的原因子句句位。
    carel_cause_positions = get_cause_clause_positions(carel_doc)
    split_cause_positions = get_cause_clause_positions(split_doc)
    result['cause_clause_positions_carel'] = carel_cause_positions
    result['cause_clause_positions_split10'] = split_cause_positions

    # 若原因子句句位不同，就記錄下來。
    if carel_cause_positions != split_cause_positions:
        result['cause_clause_positions_changed'] = True
        result['has_annotation_position_difference'] = True

    # 逐句比較兩篇文件。
    clause_differences = compare_clauses(carel_doc, split_doc)
    result['clause_differences'] = clause_differences

    # 逐一檢查每一句差異屬於哪種類型。
    for clause_difference in clause_differences:
        # 只要句子文字不同、缺句或 clause_id 對不上，就視為內容差異。
        if clause_difference['text_changed'] or clause_difference['clause_id_changed']:
            result['has_text_difference'] = True

    # 回傳整篇文件的比較結果。
    return result


# 把清單分段，避免一行放太多 doc_id 看不清楚。
def chunk_list(items, chunk_size):
    # 建立分段後的新列表。
    chunks = []

    # 從頭開始，每次取 chunk_size 個元素。
    start = 0
    while start < len(items):
        end = start + chunk_size
        chunks.append(items[start:end])
        start = end

    # 回傳分段結果。
    return chunks


# 收集摘要各分類對應的 doc_id，方便同時輸出數量與清單。
def collect_summary_groups(comparison_results):
    groups = {
        'identical_ids': [],
        'different_ids': [],
        'text_different_ids': [],
        'pure_text_different_ids': [],
        'annotation_position_different_ids': [],
        'pair_different_ids': [],
        'emotion_clause_position_different_ids': [],
        'cause_clause_position_different_ids': [],
        'pair_and_emotion_overlap_ids': [],
    }

    # 逐篇判斷屬於哪一個摘要分類。
    for result in comparison_results:
        has_difference = result['has_text_difference'] or result['has_annotation_position_difference']

        if not has_difference:
            groups['identical_ids'].append(result['doc_id'])
            continue

        groups['different_ids'].append(result['doc_id'])

        if result['has_text_difference']:
            groups['text_different_ids'].append(result['doc_id'])

        if result['has_text_difference'] and not result['has_annotation_position_difference']:
            groups['pure_text_different_ids'].append(result['doc_id'])

        if result['has_annotation_position_difference']:
            groups['annotation_position_different_ids'].append(result['doc_id'])

        if result['pairs_changed']:
            groups['pair_different_ids'].append(result['doc_id'])

        if result['emotion_clause_positions_changed']:
            groups['emotion_clause_position_different_ids'].append(result['doc_id'])

        if result['cause_clause_positions_changed']:
            groups['cause_clause_position_different_ids'].append(result['doc_id'])

        if result['pairs_changed'] and result['emotion_clause_positions_changed']:
            groups['pair_and_emotion_overlap_ids'].append(result['doc_id'])

    return groups


# 在摘要區塊中輸出某一類統計與其對應的 doc_id。
def append_summary_item(lines, label, doc_ids):
    lines.append(f'- {label}: {len(doc_ids)}')

    # 若這一類沒有任何文件，仍補一行方便驗證。
    if not doc_ids:
        lines.append('  doc_id: 無')
        return

    # 分段輸出，避免單行過長難以閱讀。
    for chunk_index, group in enumerate(chunk_list(doc_ids, 30)):
        if chunk_index == 0:
            lines.append(f'  doc_id: {", ".join(group)}')
        else:
            lines.append(f'          {", ".join(group)}')


# 依照比較結果建立文字報告。
def build_text_report(carel_path, split_path, carel_map, split_map, overlap_ids, comparison_results):
    # 建立報告內容列表，最後再用換行串起來。
    lines = []

    # 寫入基本資訊。
    lines.append('CAREL 與 split10 文檔差異報告')
    lines.append('')
    lines.append(f'CAREL 檔案: {carel_path}')
    lines.append(f'split10 檔案: {split_path}')
    lines.append('')

    # 先收集各摘要分類的 doc_id 清單。
    summary_groups = collect_summary_groups(comparison_results)

    # 寫入摘要區塊。
    lines.append('摘要')
    lines.append(f'- CAREL 文檔總數: {len(carel_map)}')
    lines.append(f'- split10 文檔總數: {len(split_map)}')
    lines.append(f'- 重疊 doc_id 數量: {len(overlap_ids)}')
    lines.append(f'- 只有 CAREL 有的 doc_id 數量: {len(carel_map) - len(overlap_ids)}')
    lines.append(f'- 只有 split10 有的 doc_id 數量: {len(split_map) - len(overlap_ids)}')
    lines.append(f'- 完全相同的重疊文檔數量: {len(summary_groups["identical_ids"])}')
    append_summary_item(lines, '有任一差異的重疊文檔數量', summary_groups['different_ids'])
    append_summary_item(lines, '純文字差異、不含標註差異的文檔數量', summary_groups['pure_text_different_ids'])
    append_summary_item(lines, '標註句位不同的重疊文檔數量', summary_groups['annotation_position_different_ids'])
    lines.append('- 以下 3 項是標註句位不同的子類別統計，彼此可重疊，不可直接相加')
    append_summary_item(lines, 'pair 不同的重疊文檔數量', summary_groups['pair_different_ids'])
    append_summary_item(lines, '情緒子句句位不同的重疊文檔數量', summary_groups['emotion_clause_position_different_ids'])
    append_summary_item(lines, '原因子句句位不同的重疊文檔數量', summary_groups['cause_clause_position_different_ids'])
    append_summary_item(lines, 'pair 與情緒子句句位同時不同的文檔數量', summary_groups['pair_and_emotion_overlap_ids'])
    lines.append('')

    # 寫入重疊 doc_id 清單。
    lines.append('重疊 doc_id 清單')
    for group in chunk_list(overlap_ids, 30):
        lines.append(', '.join(group))
    lines.append('')

    # 寫入有差異的 doc_id 清單。
    lines.append('有差異的重疊 doc_id 清單')
    for group in chunk_list(summary_groups['different_ids'], 30):
        lines.append(', '.join(group))
    lines.append('')

    # 只列出真正有差異的文件詳情。
    lines.append('逐篇差異明細')

    # 逐篇輸出差異內容。
    for result in comparison_results:
        # 沒差異的文件不展開明細，避免報告太長。
        if not (result['has_text_difference'] or result['has_annotation_position_difference']):
            continue

        # 取出對應的原始文件，方便補充 pairs 與 doc_len。
        doc_id = result['doc_id']
        carel_doc = carel_map[doc_id]
        split_doc = split_map[doc_id]

        # 取出 doc_len。
        carel_doc_len = carel_doc.get('doc_len', len(get_clauses(carel_doc)))
        split_doc_len = split_doc.get('doc_len', len(get_clauses(split_doc)))

        # 寫入文件標題與預覽。
        lines.append('')
        lines.append(f'doc_id={doc_id}')
        lines.append(f'preview: {result["preview"]}')
        lines.append(f'has_text_difference: {result["has_text_difference"]}')
        lines.append(f'has_annotation_position_difference: {result["has_annotation_position_difference"]}')
        lines.append(f'doc_len_carel: {carel_doc_len}')
        lines.append(f'doc_len_split10: {split_doc_len}')

        # 若 pairs 不同，就把兩邊 pairs 都印出來。
        if result['pairs_changed']:
            lines.append(f'pairs_carel: {result["pair_positions_carel"]}')
            lines.append(f'pairs_split10: {result["pair_positions_split10"]}')

        # 若情緒子句句位不同，就印出兩邊句位。
        if result['emotion_clause_positions_changed']:
            emotion_only_in_carel, emotion_only_in_split10 = diff_positions(
                result['emotion_clause_positions_carel'],
                result['emotion_clause_positions_split10'],
            )
            lines.append(f'emotion_clause_positions_carel: {result["emotion_clause_positions_carel"]}')
            lines.append(f'emotion_clause_positions_split10: {result["emotion_clause_positions_split10"]}')
            lines.append(f'emotion_only_in_carel: {emotion_only_in_carel}')
            lines.append(f'emotion_only_in_split10: {emotion_only_in_split10}')

        # 若原因子句句位不同，就印出兩邊句位。
        if result['cause_clause_positions_changed']:
            cause_only_in_carel, cause_only_in_split10 = diff_positions(
                result['cause_clause_positions_carel'],
                result['cause_clause_positions_split10'],
            )
            lines.append(f'cause_clause_positions_carel: {result["cause_clause_positions_carel"]}')
            lines.append(f'cause_clause_positions_split10: {result["cause_clause_positions_split10"]}')
            lines.append(f'cause_only_in_carel: {cause_only_in_carel}')
            lines.append(f'cause_only_in_split10: {cause_only_in_split10}')

        # 若沒有逐句差異，補一句說明。
        if not result['clause_differences']:
            lines.append('逐句內容相同，差異只出現在文件層級。')
            continue

        # 逐句輸出差異。
        for clause_difference in result['clause_differences']:
            # 取出兩邊句子資料。
            carel_clause = clause_difference['carel_clause']
            split_clause = clause_difference['split_clause']

            # 寫入句子位置。
            lines.append(f'- 第 {clause_difference["position"]} 句')

            # 若其中一邊缺句，就直接描述缺句情況。
            if carel_clause is None:
                lines.append('  CAREL: 缺少這一句')
            else:
                lines.append(f'  CAREL clause_id: {carel_clause.get("clause_id", "")}' )
                lines.append(f'  CAREL clause: {carel_clause.get("clause", "")}' )
                lines.append(
                    '  CAREL emotion: '
                    f'{carel_clause.get("emotion_category", "")} / '
                    f'{carel_clause.get("emotion_token", "")}'
                )

            # 若其中一邊缺句，就直接描述缺句情況。
            if split_clause is None:
                lines.append('  split10: 缺少這一句')
            else:
                lines.append(f'  split10 clause_id: {split_clause.get("clause_id", "")}' )
                lines.append(f'  split10 clause: {split_clause.get("clause", "")}' )
                lines.append(
                    '  split10 emotion: '
                    f'{split_clause.get("emotion_category", "")} / '
                    f'{split_clause.get("emotion_token", "")}'
                )

            # 若 clause_id 不同，就寫出提醒。
            if clause_difference['clause_id_changed']:
                lines.append('  差異: 兩邊的 clause_id 不一致。')

            # 若句子文字不同，就補上文字差異描述。
            if clause_difference['text_changed']:
                lines.append(f'  文字差異: {clause_difference["text_difference_note"]}')

            # 若 emotion_category 不同，就提醒是哪個欄位不同。
            if clause_difference['emotion_category_changed']:
                lines.append('  標註差異: emotion_category 不同。')

            # 若 emotion_token 不同，就提醒是哪個欄位不同。
            if clause_difference['emotion_token_changed']:
                lines.append('  標註差異: emotion_token 不同。')

    # 回傳整份文字報告。
    return '\n'.join(lines)


# 主程式：讀檔、比較、輸出報告。
def main():
    # 建立命令列參數解析器。
    parser = argparse.ArgumentParser(description='比較 CAREL 與 split10 全集 JSON 的重疊文檔差異')

    # 設定 CAREL 檔案路徑。
    parser.add_argument(
        '--carel_path',
        default='carel_zh5_full/carel_zh5_documents.json',
        help='CAREL JSON 檔案路徑',
    )

    # 設定 split10 檔案路徑。
    parser.add_argument(
        '--split10_path',
        default='split10_full_2019/full_documents_from_fold1.json',
        help='split10 JSON 檔案路徑',
    )

    # 設定文字報告輸出路徑。
    parser.add_argument(
        '--output_path',
        default='debug/carel_vs_split10_doc_diff_report.txt',
        help='文字報告輸出路徑',
    )

    # 設定 JSON 摘要輸出路徑，方便後續程式再利用。
    parser.add_argument(
        '--json_output_path',
        default='debug/carel_vs_split10_doc_diff_summary.json',
        help='JSON 摘要輸出路徑',
    )

    # 解析命令列參數。
    args = parser.parse_args()

    # 讀入兩份 JSON。
    carel_documents = load_json(args.carel_path)
    split_documents = load_json(args.split10_path)

    # 建立 doc_id 對照表。
    carel_map = build_doc_map(carel_documents, 'CAREL')
    split_map = build_doc_map(split_documents, 'split10')

    # 找出兩份資料共有的 doc_id。
    overlap_ids = []
    for doc_id in carel_map:
        if doc_id in split_map:
            overlap_ids.append(doc_id)

    # 將重疊 doc_id 排序，讓輸出順序固定。
    overlap_ids.sort(key=doc_id_sort_key)

    # 逐篇比較重疊文檔。
    comparison_results = []
    for doc_id in overlap_ids:
        carel_doc = carel_map[doc_id]
        split_doc = split_map[doc_id]
        comparison_results.append(compare_documents(carel_doc, split_doc))

    # 統計摘要數字與對應 doc_id。
    summary_groups = collect_summary_groups(comparison_results)
    identical_count = len(summary_groups['identical_ids'])
    different_count = len(summary_groups['different_ids'])
    text_difference_count = len(summary_groups['text_different_ids'])
    pure_text_difference_count = len(summary_groups['pure_text_different_ids'])
    annotation_position_difference_count = len(summary_groups['annotation_position_different_ids'])
    pair_difference_count = len(summary_groups['pair_different_ids'])
    emotion_clause_position_difference_count = len(summary_groups['emotion_clause_position_different_ids'])
    cause_clause_position_difference_count = len(summary_groups['cause_clause_position_different_ids'])
    pair_and_emotion_overlap_count = len(summary_groups['pair_and_emotion_overlap_ids'])

    # 建立文字報告內容。
    report_text = build_text_report(
        args.carel_path,
        args.split10_path,
        carel_map,
        split_map,
        overlap_ids,
        comparison_results,
    )

    # 將文字報告寫入檔案。
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, 'w', encoding='utf-8') as file_obj:
        file_obj.write(report_text)

    # 建立較精簡的 JSON 摘要，方便後續用其他程式處理。
    summary_json = {
        'carel_path': args.carel_path,
        'split10_path': args.split10_path,
        'carel_count': len(carel_map),
        'split10_count': len(split_map),
        'overlap_count': len(overlap_ids),
        'only_in_carel_count': len(carel_map) - len(overlap_ids),
        'only_in_split10_count': len(split_map) - len(overlap_ids),
        'identical_overlap_count': identical_count,
        'different_overlap_count': different_count,
        'text_difference_count': text_difference_count,
        'pure_text_difference_count': pure_text_difference_count,
        'annotation_position_difference_count': annotation_position_difference_count,
        'pair_difference_count': pair_difference_count,
        'emotion_clause_position_difference_count': emotion_clause_position_difference_count,
        'cause_clause_position_difference_count': cause_clause_position_difference_count,
        'different_doc_ids': summary_groups['different_ids'],
        'text_difference_doc_ids': summary_groups['text_different_ids'],
        'pure_text_difference_doc_ids': summary_groups['pure_text_different_ids'],
        'annotation_position_difference_doc_ids': summary_groups['annotation_position_different_ids'],
        'pair_difference_doc_ids': summary_groups['pair_different_ids'],
        'emotion_clause_position_difference_doc_ids': summary_groups['emotion_clause_position_different_ids'],
        'cause_clause_position_difference_doc_ids': summary_groups['cause_clause_position_different_ids'],
        'pair_and_emotion_overlap_doc_ids': summary_groups['pair_and_emotion_overlap_ids'],
        'overlap_doc_ids': overlap_ids,
    }

    # 把 JSON 摘要也寫入檔案。
    save_json(summary_json, args.json_output_path)

    # 在終端機印出簡短摘要，方便立刻確認結果。
    print(f'CAREL 文檔總數: {len(carel_map)}')
    print(f'split10 文檔總數: {len(split_map)}')
    print(f'重疊 doc_id 數量: {len(overlap_ids)}')
    print(f'完全相同的重疊文檔數量: {identical_count}')
    print(f'有任一差異的重疊文檔數量: {different_count}')
    print(f'純文字差異、不含標註差異的文檔數量: {pure_text_difference_count}')
    print(f'標註句位不同的重疊文檔數量: {annotation_position_difference_count}')
    print('以下 3 項是標註句位不同的子類別統計，彼此可重疊，不可直接相加')
    print(f'pair 不同的重疊文檔數量: {pair_difference_count}')
    print(f'情緒子句句位不同的重疊文檔數量: {emotion_clause_position_difference_count}')
    print(f'原因子句句位不同的重疊文檔數量: {cause_clause_position_difference_count}')
    print(f'pair 與情緒子句句位同時不同的文檔數量: {pair_and_emotion_overlap_count}')
    print(f'文字報告已輸出到: {args.output_path}')
    print(f'JSON 摘要已輸出到: {args.json_output_path}')


# 只有直接執行這支程式時，才會進入主程式。
if __name__ == '__main__':
    # 執行主程式。
    main()