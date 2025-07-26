#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析 split10/fold1_train.json 中可能導致 load_ground_truth 函數出錯的情況
"""

import json
import collections
from collections import defaultdict

def analyze_data_issues():
    """分析數據中可能的問題"""
    print('=== 分析 split10/fold1_train.json 數據格式和潛在問題 ===\n')
    
    with open('split10/fold1_train.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f'總文檔數: {len(data)}\n')
    
    # 1. 統計 pairs 的情況
    print("1. Pairs 分析:")
    empty_pairs = 0
    single_pairs = 0
    multi_pairs = 0
    max_pairs = 0
    pairs_patterns = defaultdict(int)
    
    for doc in data:
        pairs = doc.get('pairs', [])
        pairs_len = len(pairs)
        
        if pairs_len == 0:
            empty_pairs += 1
        elif pairs_len == 1:
            single_pairs += 1
        else:
            multi_pairs += 1
        
        max_pairs = max(max_pairs, pairs_len)
        pairs_patterns[pairs_len] += 1
    
    print(f'  空 pairs (可能問題): {empty_pairs}')
    print(f'  單一 pair: {single_pairs}')
    print(f'  多重 pairs: {multi_pairs}')
    print(f'  最大 pairs 數: {max_pairs}')
    print(f'  Pairs 長度分布: {dict(sorted(pairs_patterns.items()))}')
    
    # 2. 檢查 emotion_category 的值
    print("\n2. Emotion Category 分析:")
    emotion_categories = defaultdict(int)
    null_emotions = 0
    non_null_emotions = 0
    unusual_emotions = []
    
    for doc in data:
        for clause in doc.get('clauses', []):
            emotion = clause.get('emotion_category', 'missing')
            emotion_categories[emotion] += 1
            
            if emotion == 'null':
                null_emotions += 1
            elif emotion == 'missing':
                unusual_emotions.append(f"Doc {doc.get('doc_id', 'unknown')}: missing emotion_category")
            else:
                non_null_emotions += 1
    
    print(f'  Null emotions: {null_emotions}')
    print(f'  Non-null emotions: {non_null_emotions}')
    print(f'  Emotion categories: {dict(emotion_categories)}')
    if unusual_emotions:
        print(f'  異常情況: {unusual_emotions[:5]}')  # 只顯示前5個
    
    # 3. 檢查 clause_id 格式
    print("\n3. Clause ID 分析:")
    non_numeric_clause_ids = []
    duplicate_clause_ids = []
    missing_clause_ids = []
    
    for doc in data:
        doc_id = doc.get('doc_id', 'unknown')
        clause_ids = []
        
        for clause in doc.get('clauses', []):
            clause_id = clause.get('clause_id')
            
            if clause_id is None:
                missing_clause_ids.append(f"Doc {doc_id}: missing clause_id")
            else:
                clause_ids.append(clause_id)
                # 檢查是否為數字字符串
                if not clause_id.isdigit():
                    non_numeric_clause_ids.append(f"Doc {doc_id}: clause_id='{clause_id}'")
        
        # 檢查重複的clause_id
        if len(clause_ids) != len(set(clause_ids)):
            duplicate_clause_ids.append(f"Doc {doc_id}: duplicate clause_ids")
    
    print(f'  非數字 clause_id: {len(non_numeric_clause_ids)}')
    print(f'  重複 clause_id: {len(duplicate_clause_ids)}')
    print(f'  缺失 clause_id: {len(missing_clause_ids)}')
    
    if non_numeric_clause_ids:
        print(f'  非數字例子: {non_numeric_clause_ids[:3]}')
    if duplicate_clause_ids:
        print(f'  重複例子: {duplicate_clause_ids[:3]}')
    if missing_clause_ids:
        print(f'  缺失例子: {missing_clause_ids[:3]}')
    
    # 4. 檢查 pairs 中的索引是否有效
    print("\n4. Pairs 索引有效性分析:")
    invalid_pairs = []
    pairs_out_of_range = []
    pairs_wrong_format = []
    
    for doc in data:
        doc_id = doc.get('doc_id', 'unknown')
        pairs = doc.get('pairs', [])
        doc_len = doc.get('doc_len', 0)
        clauses = doc.get('clauses', [])
        actual_clause_count = len(clauses)
        
        # 獲取所有有效的clause_id
        valid_clause_ids = set()
        for clause in clauses:
            clause_id = clause.get('clause_id')
            if clause_id and clause_id.isdigit():
                valid_clause_ids.add(int(clause_id))
        
        for pair in pairs:
            # 檢查pair格式
            if not isinstance(pair, list) or len(pair) != 2:
                pairs_wrong_format.append(f"Doc {doc_id}: pair format error - {pair}")
                continue
            
            emotion_idx, cause_idx = pair
            
            # 檢查索引是否為整數
            if not isinstance(emotion_idx, int) or not isinstance(cause_idx, int):
                invalid_pairs.append(f"Doc {doc_id}: non-integer indices - {pair}")
                continue
            
            # 檢查索引是否在有效範圍內
            if emotion_idx not in valid_clause_ids or cause_idx not in valid_clause_ids:
                pairs_out_of_range.append(f"Doc {doc_id}: indices out of range - {pair}, valid_ids: {sorted(list(valid_clause_ids))}")
    
    print(f'  格式錯誤的 pairs: {len(pairs_wrong_format)}')
    print(f'  索引無效的 pairs: {len(invalid_pairs)}')
    print(f'  超出範圍的 pairs: {len(pairs_out_of_range)}')
    
    if pairs_wrong_format:
        print(f'  格式錯誤例子: {pairs_wrong_format[:2]}')
    if invalid_pairs:
        print(f'  索引無效例子: {invalid_pairs[:2]}')
    if pairs_out_of_range:
        print(f'  超出範圍例子: {pairs_out_of_range[:2]}')
    
    # 5. 檢查 doc_len 與實際 clause 數量的一致性
    print("\n5. Doc Length 一致性分析:")
    length_mismatch = []
    
    for doc in data:
        doc_id = doc.get('doc_id', 'unknown')
        doc_len = doc.get('doc_len', 0)
        actual_len = len(doc.get('clauses', []))
        
        if doc_len != actual_len:
            length_mismatch.append(f"Doc {doc_id}: declared={doc_len}, actual={actual_len}")
    
    print(f'  長度不一致的文檔: {len(length_mismatch)}')
    if length_mismatch:
        print(f'  例子: {length_mismatch[:3]}')
    
    # 6. 檢查特殊邊界情況
    print("\n6. 特殊邊界情況:")
    
    # 找一些特殊案例
    special_cases = []
    
    # 情感clause同時也是原因clause的情況
    self_reference_pairs = []
    # 一個情感對應多個原因的情況
    one_to_many_cases = []
    # 多個情感對應一個原因的情況
    many_to_one_cases = []
    
    for doc in data:
        doc_id = doc.get('doc_id', 'unknown')
        pairs = doc.get('pairs', [])
        
        if not pairs:
            continue
            
        emotions = [p[0] for p in pairs if isinstance(p, list) and len(p) == 2]
        causes = [p[1] for p in pairs if isinstance(p, list) and len(p) == 2]
        
        # 自指向pairs
        for pair in pairs:
            if isinstance(pair, list) and len(pair) == 2 and pair[0] == pair[1]:
                self_reference_pairs.append(f"Doc {doc_id}: self-reference {pair}")
        
        # 一對多情況
        emotion_counts = collections.Counter(emotions)
        for emotion, count in emotion_counts.items():
            if count > 1:
                related_causes = [p[1] for p in pairs if p[0] == emotion]
                one_to_many_cases.append(f"Doc {doc_id}: emotion {emotion} -> causes {related_causes}")
        
        # 多對一情況
        cause_counts = collections.Counter(causes)
        for cause, count in cause_counts.items():
            if count > 1:
                related_emotions = [p[0] for p in pairs if p[1] == cause]
                many_to_one_cases.append(f"Doc {doc_id}: emotions {related_emotions} -> cause {cause}")
    
    print(f'  自指向 pairs: {len(self_reference_pairs)}')
    print(f'  一對多情況: {len(one_to_many_cases)}')
    print(f'  多對一情況: {len(many_to_one_cases)}')
    
    if self_reference_pairs:
        print(f'  自指向例子: {self_reference_pairs[:2]}')
    if one_to_many_cases:
        print(f'  一對多例子: {one_to_many_cases[:2]}')
    if many_to_one_cases:
        print(f'  多對一例子: {many_to_one_cases[:2]}')
    
    # 7. 總結潛在問題
    print("\n" + "="*50)
    print("=== 潛在問題總結 ===")
    
    total_issues = 0
    
    if empty_pairs > 0:
        print(f"⚠️  空 pairs 文檔: {empty_pairs} 個")
        total_issues += empty_pairs
    
    if non_numeric_clause_ids:
        print(f"⚠️  非數字 clause_id: {len(non_numeric_clause_ids)} 個")
        total_issues += len(non_numeric_clause_ids)
    
    if duplicate_clause_ids:
        print(f"⚠️  重複 clause_id: {len(duplicate_clause_ids)} 個")
        total_issues += len(duplicate_clause_ids)
    
    if missing_clause_ids:
        print(f"⚠️  缺失 clause_id: {len(missing_clause_ids)} 個")
        total_issues += len(missing_clause_ids)
    
    if pairs_wrong_format:
        print(f"⚠️  格式錯誤 pairs: {len(pairs_wrong_format)} 個")
        total_issues += len(pairs_wrong_format)
    
    if invalid_pairs:
        print(f"⚠️  無效索引 pairs: {len(invalid_pairs)} 個")
        total_issues += len(invalid_pairs)
    
    if pairs_out_of_range:
        print(f"⚠️  超出範圍 pairs: {len(pairs_out_of_range)} 個")
        total_issues += len(pairs_out_of_range)
    
    if length_mismatch:
        print(f"⚠️  長度不一致: {len(length_mismatch)} 個")
        total_issues += len(length_mismatch)
    
    if total_issues == 0:
        print("✅ 沒有發現明顯的數據格式問題！")
    else:
        print(f"❌ 總共發現 {total_issues} 個潛在問題")
    
    print("\n=== load_ground_truth 函數兼容性分析 ===")
    
    print("當前函數可能的問題:")
    print("1. 如果 pairs 為空，函數仍可正常運行")
    print("2. 如果 clause_id 不是數字字符串，str() 比較可能出錯")
    print("3. 如果 pairs 中的索引不存在對應的 clause，會找不到匹配")
    print("4. 如果 emotion_category 缺失，會導致 KeyError")
    
    return {
        'empty_pairs': empty_pairs,
        'non_numeric_clause_ids': len(non_numeric_clause_ids),
        'duplicate_clause_ids': len(duplicate_clause_ids),
        'missing_clause_ids': len(missing_clause_ids),
        'invalid_pairs': len(invalid_pairs),
        'pairs_out_of_range': len(pairs_out_of_range),
        'length_mismatch': len(length_mismatch),
        'total_issues': total_issues
    }

if __name__ == "__main__":
    result = analyze_data_issues()
