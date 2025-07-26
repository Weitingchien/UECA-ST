#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
評估偽標籤預測準確度
比較 pseudo_results 資料夾內的預測結果與 split10 資料夾內的真實標籤
"""

import json
import os
from collections import defaultdict


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


def load_pseudo_predictions(fold_num):
    """載入指定fold的偽標籤預測結果"""
    file_path = f"pseudo_results/fold{fold_num}_pseudo_text_result.txt"
    
    if not os.path.exists(file_path):
        print(f"❌ 找不到文件: {file_path}")
        return {}
    
    predictions = {}
    current_doc_id = None
    current_clauses = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            if line.startswith("doc_id :"):
                # 保存上一個文檔的數據
                if current_doc_id is not None:
                    predictions[current_doc_id] = current_clauses
                
                # 開始新文檔
                current_doc_id = line.split(":")[1].strip()
                current_clauses = []
            
            elif line and not line.startswith("doc_id"):
                # 這是預測標籤行
                current_clauses.append(line)
    
    # 保存最後一個文檔
    if current_doc_id is not None:
        predictions[current_doc_id] = current_clauses
    
    return predictions


def calculate_accuracy(ground_truth, predictions):
    """計算準確度指標"""
    stats = {
        'total_docs': 0,
        'matched_docs': 0,
        'total_clauses': 0,
        'correct_clauses': 0,
        'emotion_correct': 0,
        'emotion_total': 0,
        'cause_correct': 0,
        'cause_total': 0
    }
    
    # 找到共同的文檔ID
    common_docs = set(ground_truth.keys()) & set(predictions.keys())
    stats['total_docs'] = len(common_docs)
    
    for doc_id in common_docs:
        gt_clauses = ground_truth[doc_id]
        pred_clauses = predictions[doc_id]
        
        # 確保clause數量一致
        min_len = min(len(gt_clauses), len(pred_clauses))
        if len(gt_clauses) != len(pred_clauses):
            print(f"⚠️ Doc {doc_id}: GT有{len(gt_clauses)}個clause, 預測有{len(pred_clauses)}個")
        
        doc_correct = True
        clause_correct_count = 0
        
        for i in range(min_len):
            gt_parts = gt_clauses[i].split()
            pred_parts = pred_clauses[i].split()
            
            if len(gt_parts) >= 2 and len(pred_parts) >= 2:
                # 比較情感標籤 (第一部分)
                if gt_parts[0] == pred_parts[0]:
                    stats['emotion_correct'] += 1
                stats['emotion_total'] += 1
                
                # 比較原因標籤 (第二部分)
                if gt_parts[1] == pred_parts[1]:
                    stats['cause_correct'] += 1
                stats['cause_total'] += 1
                
                # 整體clause正確性
                if gt_clauses[i] == pred_clauses[i]:
                    clause_correct_count += 1
                else:
                    doc_correct = False
        
        stats['total_clauses'] += min_len
        stats['correct_clauses'] += clause_correct_count
        
        if doc_correct and min_len == len(gt_clauses):
            stats['matched_docs'] += 1
    
    return stats


def print_results(fold_num, stats):
    """打印結果"""
    print(f"\n📊 Fold {fold_num} 準確度評估結果:")
    print(f"  文檔總數: {stats['total_docs']}")
    print(f"  完全正確文檔: {stats['matched_docs']} ({stats['matched_docs']/max(stats['total_docs'],1)*100:.1f}%)")
    print(f"  Clause總數: {stats['total_clauses']}")
    print(f"  Clause準確度: {stats['correct_clauses']}/{stats['total_clauses']} ({stats['correct_clauses']/max(stats['total_clauses'],1)*100:.1f}%)")
    print(f"  情感標籤準確度: {stats['emotion_correct']}/{stats['emotion_total']} ({stats['emotion_correct']/max(stats['emotion_total'],1)*100:.1f}%)")
    print(f"  原因標籤準確度: {stats['cause_correct']}/{stats['cause_total']} ({stats['cause_correct']/max(stats['cause_total'],1)*100:.1f}%)")


def main():
    """主函數"""
    print("🔍 開始評估偽標籤預測準確度...")
    
    overall_stats = {
        'total_docs': 0,
        'matched_docs': 0,
        'total_clauses': 0,
        'correct_clauses': 0,
        'emotion_correct': 0,
        'emotion_total': 0,
        'cause_correct': 0,
        'cause_total': 0
    }
    
    valid_folds = 0
    
    # 處理每一折 (1-10)
    for fold_num in range(1, 11):
        print(f"\n{'='*50}")
        print(f"處理 Fold {fold_num}")
        
        # 載入數據
        ground_truth = load_ground_truth(fold_num)
        predictions = load_pseudo_predictions(fold_num)
        
        if not ground_truth or not predictions:
            print(f"⚠️ Fold {fold_num} 數據載入失敗，跳過")
            continue
        
        # 計算準確度
        stats = calculate_accuracy(ground_truth, predictions)
        
        # 打印結果
        print_results(fold_num, stats)
        
        # 累加到總體統計
        for key in overall_stats:
            overall_stats[key] += stats[key]
        valid_folds += 1
    
    # 打印總體結果
    if valid_folds > 0:
        print(f"\n{'='*50}")
        print(f"📈 總體準確度評估結果 (共{valid_folds}折):")
        print(f"  文檔總數: {overall_stats['total_docs']}")
        print(f"  完全正確文檔: {overall_stats['matched_docs']} ({overall_stats['matched_docs']/max(overall_stats['total_docs'],1)*100:.1f}%)")
        print(f"  Clause總數: {overall_stats['total_clauses']}")
        print(f"  Clause準確度: {overall_stats['correct_clauses']}/{overall_stats['total_clauses']} ({overall_stats['correct_clauses']/max(overall_stats['total_clauses'],1)*100:.1f}%)")
        print(f"  情感標籤準確度: {overall_stats['emotion_correct']}/{overall_stats['emotion_total']} ({overall_stats['emotion_correct']/max(overall_stats['emotion_total'],1)*100:.1f}%)")
        print(f"  原因標籤準確度: {overall_stats['cause_correct']}/{overall_stats['cause_total']} ({overall_stats['cause_correct']/max(overall_stats['cause_total'],1)*100:.1f}%)")
    else:
        print("❌ 沒有有效的fold數據可以處理")


if __name__ == "__main__":
    main()
