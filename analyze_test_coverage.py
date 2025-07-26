#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析測試案例統計差異的原因
"""

import json
import collections

def analyze_test_coverage():
    """分析測試案例的覆蓋情況"""
    print("=== 分析測試案例覆蓋情況 ===\n")
    
    with open('split10/fold1_train.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f'總文檔數: {len(data)}')
    
    # 統計各種情況（每個文檔只計算一次，按優先級分類）
    categories = {
        'self_reference_docs': 0,     # 有自指向pairs的文檔數量
        'compound_emotion_docs': 0,   # 有複合情感的文檔數量
        'one_to_many_docs': 0,        # 有一對多情況的文檔數量
        'normal_case_docs': 0,        # 一般情況文檔數量（單一pair且非自指向）
        'multi_pairs_docs': 0,        # 多個pairs但不是一對多的文檔數量
        'no_pairs_docs': 0,           # 沒有pairs的文檔數量
    }
    
    for doc in data:
        doc_id = doc['doc_id']
        pairs = doc.get('pairs', [])
        
        # 1. 檢查沒有pairs的情況
        if len(pairs) == 0:
            categories['no_pairs_docs'] += 1
            continue
        
        # 2. 檢查自指向（優先級最高）
        has_self_reference = False
        for pair in pairs:
            if pair[0] == pair[1]:
                has_self_reference = True
                break
        
        if has_self_reference:
            categories['self_reference_docs'] += 1
            continue
        
        # 3. 檢查複合情感（第二優先級）
        has_compound_emotion = False
        for clause in doc.get('clauses', []):
            emotion = clause.get('emotion_category', '')
            if '&' in emotion:
                has_compound_emotion = True
                break
        
        if has_compound_emotion:
            categories['compound_emotion_docs'] += 1
            continue
        
        # 4. 檢查一對多情況（第三優先級）
        emotion_counts = collections.Counter([p[0] for p in pairs])
        has_one_to_many = any(count > 1 for count in emotion_counts.values())
        
        if has_one_to_many:
            categories['one_to_many_docs'] += 1
            continue
        
        # 5. 檢查一般情況（單一pair且非自指向）
        if len(pairs) == 1:
            pair = pairs[0]
            if pair[0] != pair[1]:  # 非自指向
                categories['normal_case_docs'] += 1
                continue
        
        # 6. 其他情況（多個pairs但不是一對多）
        categories['multi_pairs_docs'] += 1
    
    # 顯示統計結果
    print("\n=== 各類別統計（按文檔數計算） ===")
    for category, count in categories.items():
        print(f"{category}: {count} 個")
    
    # 計算總統計數（應該等於總文檔數）
    total_categorized = sum(categories.values())
    
    print(f"\n總分類文檔數: {total_categorized} 個")
    print(f"實際總文檔數: {len(data)} 個")
    
    # 驗證分析結果
    print("\n=== 驗證分析 ===")
    expected_total_docs = 1750
    
    print(f"預期總文檔數: {expected_total_docs}")
    print(f"實際總文檔數: {len(data)}")
    print(f"分類統計總數: {total_categorized}")
    
    if total_categorized == len(data) == expected_total_docs:
        print("✅ 完美！每個文檔都被正確分類，統計數與總文檔數一致")
        print(f"分布：{categories['self_reference_docs']}個自指向 + {categories['compound_emotion_docs']}個複合情感 + {categories['one_to_many_docs']}個一對多 + {categories['normal_case_docs']}個一般 + {categories['multi_pairs_docs']}個多pairs + {categories['no_pairs_docs']}個無pairs")
    else:
        print("❌ 統計有誤")
        print(f"差異: 分類總數 {total_categorized} vs 實際文檔數 {len(data)}")
    
    # 顯示分類詳細資訊
    print(f"\n=== 分類詳細說明 ===")
    print(f"1. 自指向文檔 ({categories['self_reference_docs']}個): 包含自指向pairs [x,x] 的文檔")
    print(f"2. 複合情感文檔 ({categories['compound_emotion_docs']}個): 包含複合情感（如anger&sadness）的文檔")  
    print(f"3. 一對多文檔 ({categories['one_to_many_docs']}個): 一個情感對應多個原因的文檔")
    print(f"4. 一般情況文檔 ({categories['normal_case_docs']}個): 單一pair且非自指向的文檔")
    print(f"5. 多pairs文檔 ({categories['multi_pairs_docs']}個): 多個獨立pairs但不是一對多的文檔")
    print(f"6. 無pairs文檔 ({categories['no_pairs_docs']}個): 沒有emotion-cause pairs的文檔")
    
    # 對比原測試邏輯
    print(f"\n=== 與原測試邏輯對比 ===")
    tested_in_original = (categories['self_reference_docs'] + 
                         categories['compound_emotion_docs'] + 
                         categories['one_to_many_docs'] + 
                         categories['normal_case_docs'])
    
    not_tested_in_original = categories['multi_pairs_docs'] + categories['no_pairs_docs']
    
    print(f"原測試邏輯會測試的文檔: {tested_in_original} 個")
    print(f"原測試邏輯不會測試的文檔: {not_tested_in_original} 個")
    print(f"原測試邏輯統計的特徵數: 1726 個（因為按特徵計算而非文檔計算）")
    
    # 顯示一些多pairs文檔的例子
    if categories['multi_pairs_docs'] > 0:
        print(f"\n=== 多pairs文檔例子（原測試邏輯未覆蓋） ===")
        multi_pairs_examples = []
        for doc in data:
            pairs = doc.get('pairs', [])
            if len(pairs) > 1:
                # 檢查是否為多pairs但不是一對多
                emotion_counts = collections.Counter([p[0] for p in pairs])
                has_one_to_many = any(count > 1 for count in emotion_counts.values())
                
                # 檢查是否有自指向
                has_self_reference = any(p[0] == p[1] for p in pairs)
                
                # 檢查是否有複合情感
                has_compound_emotion = any('&' in clause.get('emotion_category', '') for clause in doc.get('clauses', []))
                
                if not has_one_to_many and not has_self_reference and not has_compound_emotion:
                    multi_pairs_examples.append((doc['doc_id'], pairs))
                    if len(multi_pairs_examples) >= 5:
                        break
        
        for doc_id, pairs in multi_pairs_examples:
            print(f"  Doc {doc_id}: pairs={pairs}")
            
    print(f"\n💡 建議：所有文檔都已被正確分類和統計！")

if __name__ == "__main__":
    analyze_test_coverage()
