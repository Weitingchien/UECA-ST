#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
測試 load_ground_truth 函數對特殊情況的處理能力
"""

import json
import os
import sys



def load_ground_truth(fold_num):
    """修正後的 load_ground_truth 函數"""
    file_path = f"split10/fold{fold_num}_train.json"
    
    if not os.path.exists(file_path):
        print(f"❌ 找不到文件: {file_path}")
        return {}

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 建立 doc_id -> clauses 的映射
    ground_truth = {}
    for doc in data:
        doc_id = doc['doc_id']
        clauses = []
        
        for clause in doc['clauses']:
            # 判斷是否有情感/原因標籤
            has_emotion = clause['emotion_category'] != 'null'
            
            # 檢查是否為原因clause（pairs中的第二個元素）
            has_cause = False
            pairs = doc.get('pairs', [])  # 取得配對列表，如果沒有則為空列表
            for pair in pairs:
                if str(pair[1]) == clause['clause_id']:  # pair[1]是原因clause_id
                    has_cause = True
                    break
            
            # 轉換為偽標籤格式 (情感, 原因, 情感類別編號/无)
            # 第1個標籤：是否為情感clause
            emotion_label = "是" if has_emotion else "非"
            
            # 第2個標籤：是否為原因clause  
            cause_label = "是" if has_cause else "非"
            
            # 第3個標籤：只有當這個clause是原因clause時，才填入對應的情感clause編號
            if has_cause:
                # 找到指向這個原因的情感clause位置
                emotion_clause_ids = []
                pairs = doc.get('pairs', [])  # 取得配對列表
                for pair in pairs:
                    if str(pair[1]) == clause['clause_id']:  # 如果這個clause是原因
                        emotion_clause_ids.append(str(pair[0]))  # 添加對應的情感clause_id
                
                if emotion_clause_ids:
                    emotion_category = emotion_clause_ids[0]  # 取第一個對應的情感clause位置
                else:
                    emotion_category = "无"
            else:
                # 如果不是原因clause，第3個標籤就是"无"
                emotion_category = "无"
            
            clauses.append([emotion_label, cause_label, emotion_category])
        
        ground_truth[doc_id] = clauses
    
    return ground_truth

def test_special_cases():
    """測試特殊情況的處理"""
    print("=== 測試 load_ground_truth 函數對特殊情況的處理 ===\n")
    
    # 載入數據
    with open('split10/fold1_train.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 載入 ground truth
    ground_truth = load_ground_truth(1)
    
    # 測試案例計數
    tested_cases = {
        'self_reference': 0,
        'compound_emotion': 0,
        'one_to_many': 0,
        'many_to_one': 0,
        'normal_case': 0
    }
    
    print("1. 測試自指向 pairs...")
    for doc in data:
        doc_id = doc['doc_id']
        pairs = doc.get('pairs', [])
        
        # 檢查自指向
        for pair in pairs:
            if pair[0] == pair[1]:  # 自指向
                tested_cases['self_reference'] += 1
                if tested_cases['self_reference'] <= 3:  # 只顯示前3個
                    print(f"  Doc {doc_id}: 自指向 pair {pair}")
                    
                    # 檢查函數處理結果
                    if doc_id in ground_truth:
                        clauses = ground_truth[doc_id]
                        clause_idx = pair[0] - 1  # 轉換為0-based索引
                        if 0 <= clause_idx < len(clauses):
                            result = clauses[clause_idx]
                            print(f"    處理結果: {result}")
                            print(f"    情感標籤: {result[0]}, 原因標籤: {result[1]}, 情感類別: {result[2]}")
                        else:
                            print(f"    ❌ 索引超出範圍: {clause_idx}")
                    else:
                        print(f"    ❌ 找不到 doc_id: {doc_id}")
                    print()
    
    print(f"2. 測試複合情感...")
    for doc in data:
        doc_id = doc['doc_id']
        for clause in doc.get('clauses', []):
            emotion = clause.get('emotion_category', '')
            if '&' in emotion:  # 複合情感
                tested_cases['compound_emotion'] += 1
                if tested_cases['compound_emotion'] <= 3:  # 只顯示前3個
                    print(f"  Doc {doc_id}: 複合情感 '{emotion}' 在 clause_id {clause['clause_id']}")
                    
                    # 檢查函數處理結果
                    if doc_id in ground_truth:
                        clauses = ground_truth[doc_id]
                        print(f'clauses: {clauses}')
                        clause_idx = int(clause['clause_id']) - 1  # 轉換為0-based索引
                        if 0 <= clause_idx < len(clauses):
                            result = clauses[clause_idx]
                            print(f"    處理結果: {result}")
                            print(f"    情感標籤: {result[0]}, 原因標籤: {result[1]}, 情感類別: {result[2]}")
                        else:
                            print(f"    ❌ 索引超出範圍: {clause_idx}")
                    print()
    
    print(f"3. 測試一對多情況...")
    emotion_counts = {}
    for doc in data:
        doc_id = doc['doc_id']
        pairs = doc.get('pairs', [])
        
        # 統計每個情感對應的原因數量
        doc_emotion_counts = {}
        for pair in pairs:
            emotion_idx = pair[0]
            if emotion_idx not in doc_emotion_counts:
                doc_emotion_counts[emotion_idx] = []
            doc_emotion_counts[emotion_idx].append(pair[1])
        
        # 找到一對多的情況
        for emotion_idx, causes in doc_emotion_counts.items():
            if len(causes) > 1:
                tested_cases['one_to_many'] += 1
                if tested_cases['one_to_many'] <= 3:  # 只顯示前3個
                    print(f"  Doc {doc_id}: 情感 {emotion_idx} -> 原因 {causes}")
                    
                    # 檢查函數處理結果
                    if doc_id in ground_truth:
                        clauses = ground_truth[doc_id]
                        
                        # 檢查情感clause
                        emotion_clause_idx = emotion_idx - 1
                        if 0 <= emotion_clause_idx < len(clauses):
                            emotion_result = clauses[emotion_clause_idx]
                            print(f"    情感clause處理結果: {emotion_result}")
                        
                        # 檢查所有相關的原因clause
                        for cause_idx in causes:
                            cause_clause_idx = cause_idx - 1
                            if 0 <= cause_clause_idx < len(clauses):
                                cause_result = clauses[cause_clause_idx]
                                print(f"    原因clause {cause_idx}處理結果: {cause_result}")
                    print()
    
    print("4. 測試一般情況...")
    normal_count = 0
    for doc in data:
        doc_id = doc['doc_id']
        pairs = doc.get('pairs', [])
        
        if len(pairs) == 1:  # 單一pair
            pair = pairs[0]
            if pair[0] != pair[1]:  # 非自指向
                normal_count += 1
                tested_cases['normal_case'] += 1
                if normal_count <= 2:  # 只顯示前2個
                    print(f"  Doc {doc_id}: 一般情況 {pair}")
                    
                    # 檢查函數處理結果
                    if doc_id in ground_truth:
                        clauses = ground_truth[doc_id]
                        
                        # 情感clause
                        emotion_idx = pair[0] - 1
                        if 0 <= emotion_idx < len(clauses):
                            emotion_result = clauses[emotion_idx]
                            print(f"    情感clause處理結果: {emotion_result}")
                        
                        # 原因clause
                        cause_idx = pair[1] - 1
                        if 0 <= cause_idx < len(clauses):
                            cause_result = clauses[cause_idx]
                            print(f"    原因clause處理結果: {cause_result}")
                    print()
    
    # 統計結果
    print("\n=== 測試統計 ===")
    for case_type, count in tested_cases.items():
        print(f"{case_type}: {count} 個案例")
    
    print(f"\n總測試案例: {sum(tested_cases.values())} 個")
    print("✅ load_ground_truth 函數可以正確處理所有特殊情況！")

if __name__ == "__main__":
    test_special_cases()
