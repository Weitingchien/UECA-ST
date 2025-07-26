#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
測試修正後的 load_ground_truth 函數邏輯
"""

import json
import os

def load_ground_truth_corrected(fold_num):
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
            emotion_label = "是" if has_emotion else "否"
            
            # 第2個標籤：是否為原因clause  
            cause_label = "是" if has_cause else "否"
            
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

def test_doc_1401():
    """測試文檔1401的處理結果"""
    print("=== 測試文檔1401 (複合情感案例) ===\n")
    
    # 載入數據
    with open('split10/fold1_train.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 找到文檔1401
    doc_1401 = None
    for doc in data:
        if doc['doc_id'] == '1401':
            doc_1401 = doc
            break
    
    if not doc_1401:
        print("❌ 找不到文檔1401")
        return
    
    print("原始數據:")
    print(f"doc_id: {doc_1401['doc_id']}")
    print(f"pairs: {doc_1401['pairs']}")
    print()
    
    # 顯示相關的clauses
    print("相關的clauses:")
    for clause in doc_1401['clauses']:
        clause_id = int(clause['clause_id'])
        if clause_id in [9, 10, 15, 16]:  # 只顯示與pairs相關的
            print(f"  clause {clause_id}: '{clause['clause']}' (emotion: {clause['emotion_category']})")
    print()
    
    # 載入 ground truth
    ground_truth = load_ground_truth_corrected(1)
    
    if '1401' not in ground_truth:
        print("❌ ground_truth中找不到文檔1401")
        return
    
    clauses_result = ground_truth['1401']
    
    print("修正後的處理結果:")
    print("根據UECA邏輯，第3個[MASK]只有在該clause是原因clause時才有編號")
    print()
    
    # 顯示關鍵clause的處理結果
    key_clauses = [9, 10, 15, 16]
    for clause_num in key_clauses:
        result = clauses_result[clause_num - 1]  # 轉換為0-based索引
        print(f"clause {clause_num}: {result}")
        print(f"  - 情感標籤: {result[0]} ({'是情感clause' if result[0] == '是' else '不是情感clause'})")
        print(f"  - 原因標籤: {result[1]} ({'是原因clause' if result[1] == '是' else '不是原因clause'})")
        print(f"  - 第3個標籤: {result[2]} ({'對應的情感clause編號' if result[2] != '无' else '无（因為不是原因clause）'})")
        print()
    
    print("=== 驗證邏輯正確性 ===")
    print("根據 pairs [[10,9], [16,15], [16,15]]:")
    print("✓ clause 9: 不是情感，是原因，第3個標籤應該是10")
    print("✓ clause 10: 是情感，不是原因，第3個標籤應該是无")  
    print("✓ clause 15: 不是情感，是原因，第3個標籤應該是16")
    print("✓ clause 16: 是情感（複合），不是原因，第3個標籤應該是无")
    
    # 驗證結果
    expected = {
        9: ['否', '是', '10'],   # 不是情感，是原因，對應情感clause 10
        10: ['是', '否', '无'],  # 是情感，不是原因，无編號
        15: ['否', '是', '16'],  # 不是情感，是原因，對應情感clause 16  
        16: ['是', '否', '无']   # 是情感（複合），不是原因，无編號
    }
    
    print("\n=== 結果驗證 ===")
    all_correct = True
    for clause_num, expected_result in expected.items():
        actual_result = clauses_result[clause_num - 1]
        if actual_result == expected_result:
            print(f"✅ clause {clause_num}: 正確 {actual_result}")
        else:
            print(f"❌ clause {clause_num}: 預期 {expected_result}, 實際 {actual_result}")
            all_correct = False
    
    if all_correct:
        print("\n🎉 所有測試通過！修正後的邏輯正確！")
    else:
        print("\n❌ 還有問題需要修正")

if __name__ == "__main__":
    test_doc_1401()
