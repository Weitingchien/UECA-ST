#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
比較 pseudo-labeling 預測結果與正確答案的腳本
分析預測的準確性和差異
"""

import os
import json
import glob
import sys
from collections import defaultdict, Counter

# 添加 test 目錄到 Python 路徑，以便導入經過驗證的 load_ground_truth 函式
sys.path.append(os.path.join(os.path.dirname(__file__), 'test'))
from test_load_ground_truth import load_ground_truth

def load_ground_truth_data(train_file_path):
    """
    載入正確答案數據（使用經過驗證的 load_ground_truth 函式）
    
    Args:
        train_file_path (str): 訓練數據檔案路徑
        
    Returns:
        dict: 以 doc_id 為 key 的正確答案數據
    """
    # 從檔案路徑中提取 fold 編號
    # 例如：split10/fold1_train.json -> fold1
    fold_filename = os.path.basename(train_file_path)  # fold1_train.json
    fold_num_str = fold_filename.replace('_train.json', '').replace('fold', '')  # 1
    
    try:
        fold_num = int(fold_num_str)
    except ValueError:
        print(f"❌ 無法從檔案路徑 {train_file_path} 中提取 fold 編號")
        return {}
    
    # 使用經過驗證的 load_ground_truth 函式
    ground_truth_data = load_ground_truth(fold_num)
    
    # 轉換格式：將 [emotion_label, cause_label, emotion_category] 轉換為字串格式
    result = {}
    for doc_id, clauses in ground_truth_data.items():
        formatted_clauses = []
        for clause_labels in clauses:
            # clause_labels 是 [emotion_label, cause_label, emotion_category] 的列表
            formatted_clause = f"{clause_labels[0]} {clause_labels[1]} {clause_labels[2]}"
            formatted_clauses.append(formatted_clause)
        result[str(doc_id)] = formatted_clauses
    
    return result

def load_prediction_data(pred_file_path):
    """
    載入預測結果數據
    
    Args:
        pred_file_path (str): 預測結果檔案路徑
        
    Returns:
        dict: 以 doc_id 為 key 的預測結果數據
    """
    predictions = {}
    
    try:
        with open(pred_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        current_doc_id = None
        current_predictions = []
        
        for line in lines:
            line = line.strip()
            
            if line.startswith("doc_id :"):
                # 處理前一個文檔
                if current_doc_id is not None:
                    predictions[current_doc_id] = current_predictions
                
                # 開始新文檔
                current_doc_id = line.replace("doc_id :", "").strip()
                current_predictions = []
                
            elif line and not line.startswith("doc_id"):
                current_predictions.append(line)
        
        # 處理最後一個文檔
        if current_doc_id is not None:
            predictions[current_doc_id] = current_predictions
            
    except Exception as e:
        print(f"❌ 載入預測結果檔案 {pred_file_path} 時發生錯誤: {e}")
    
    return predictions

def compare_single_fold(fold_num, train_data_dir="split10", pred_data_dir="pseudo_results"):
    """
    比較單個 fold 的預測結果和正確答案
    
    Args:
        fold_num (int): fold 編號
        train_data_dir (str): 訓練數據目錄
        pred_data_dir (str): 預測結果目錄
        
    Returns:
        dict: 比較統計結果
    """
    # 檔案路徑
    train_file = os.path.join(train_data_dir, f"fold{fold_num}_train.json")
    pred_file = os.path.join(pred_data_dir, f"fold{fold_num}_pseudo_text_result.txt")
    
    if not os.path.exists(train_file):
        print(f"❌ 找不到訓練檔案: {train_file}")
        return None
    
    if not os.path.exists(pred_file):
        print(f"❌ 找不到預測檔案: {pred_file}")
        return None
    
    print(f"🔍 比較 fold{fold_num}...")
    
    # 載入數據
    ground_truth = load_ground_truth_data(train_file)
    predictions = load_prediction_data(pred_file)
    
    # 統計結果
    stats = {
        'fold': fold_num,
        'total_docs': 0,
        'matched_docs': 0,
        'total_sentences': 0,
        'correct_predictions': 0,
        'emotion_correct': 0,
        'cause_correct': 0,
        'pair_correct': 0,
        'emotion_total': 0,
        'cause_total': 0,
        'pair_total': 0,
        'doc_level_accuracy': {},
        'mismatched_examples': []
    }
    
    # 比較每個文檔
    for doc_id in ground_truth:
        if doc_id not in predictions:
            print(f"⚠️  預測結果中缺少 doc_id: {doc_id}")
            continue
        
        stats['total_docs'] += 1
        gt_labels = ground_truth[doc_id]
        pred_labels = predictions[doc_id]
        
        # 檢查句子數量是否一致
        if len(gt_labels) != len(pred_labels):
            print(f"⚠️  Doc {doc_id}: 句子數量不一致 (GT: {len(gt_labels)}, Pred: {len(pred_labels)})")
            continue
        
        # 逐句比較
        doc_correct = 0
        doc_total = len(gt_labels)
        
        for i, (gt_label, pred_label) in enumerate(zip(gt_labels, pred_labels)):
            stats['total_sentences'] += 1
            
            # 分解標籤
            gt_parts = gt_label.split()
            pred_parts = pred_label.split()
            
            if len(gt_parts) >= 3 and len(pred_parts) >= 3:
                gt_emotion, gt_cause, gt_pair = gt_parts[0], gt_parts[1], gt_parts[2]
                pred_emotion, pred_cause, pred_pair = pred_parts[0], pred_parts[1], pred_parts[2]
                
                # 情感標籤比較
                stats['emotion_total'] += 1
                if gt_emotion == pred_emotion:
                    stats['emotion_correct'] += 1
                
                # 原因標籤比較
                stats['cause_total'] += 1
                if gt_cause == pred_cause:
                    stats['cause_correct'] += 1
                
                # 配對標籤比較
                stats['pair_total'] += 1
                if gt_pair == pred_pair:
                    stats['pair_correct'] += 1
                
                # 完全正確比較
                if gt_label == pred_label:
                    stats['correct_predictions'] += 1
                    doc_correct += 1
                else:
                    # 記錄錯誤範例
                    if len(stats['mismatched_examples']) < 20:  # 只記錄前20個錯誤
                        stats['mismatched_examples'].append({
                            'doc_id': doc_id,
                            'sentence_id': i + 1,
                            'ground_truth': gt_label,
                            'prediction': pred_label
                        })
        
        # 文檔級別準確性
        doc_accuracy = doc_correct / doc_total if doc_total > 0 else 0
        stats['doc_level_accuracy'][doc_id] = doc_accuracy
        
        if doc_accuracy == 1.0:
            stats['matched_docs'] += 1
    
    return stats

def analyze_all_folds(max_folds=10):
    """
    分析所有 fold 的比較結果
    
    Args:
        max_folds (int): 最大 fold 數量
        
    Returns:
        dict: 全體統計結果
    """
    print("🚀 開始比較所有 fold 的預測結果與正確答案...")
    
    all_stats = {
        'total_folds': 0,
        'fold_results': {},
        'overall_stats': {
            'total_docs': 0,
            'matched_docs': 0,
            'total_sentences': 0,
            'correct_predictions': 0,
            'emotion_correct': 0,
            'cause_correct': 0,
            'pair_correct': 0,
            'emotion_total': 0,
            'cause_total': 0,
            'pair_total': 0
        }
    }
    
    # 檢查可用的 fold
    available_folds = []
    for fold_num in range(1, max_folds + 1):
        train_file = f"split10/fold{fold_num}_train.json"
        pred_file = f"pseudo_results/fold{fold_num}_pseudo_text_result.txt"
        
        if os.path.exists(train_file) and os.path.exists(pred_file):
            available_folds.append(fold_num)
    
    print(f"📁 找到 {len(available_folds)} 個可比較的 fold: {available_folds}")
    
    # 比較每個 fold
    for fold_num in available_folds:
        fold_stats = compare_single_fold(fold_num)
        
        if fold_stats:
            all_stats['total_folds'] += 1
            all_stats['fold_results'][f'fold{fold_num}'] = fold_stats
            
            # 累加統計
            overall = all_stats['overall_stats']
            overall['total_docs'] += fold_stats['total_docs']
            overall['matched_docs'] += fold_stats['matched_docs']
            overall['total_sentences'] += fold_stats['total_sentences']
            overall['correct_predictions'] += fold_stats['correct_predictions']
            overall['emotion_correct'] += fold_stats['emotion_correct']
            overall['cause_correct'] += fold_stats['cause_correct']
            overall['pair_correct'] += fold_stats['pair_correct']
            overall['emotion_total'] += fold_stats['emotion_total']
            overall['cause_total'] += fold_stats['cause_total']
            overall['pair_total'] += fold_stats['pair_total']
            
            # 顯示單個 fold 結果
            print_fold_summary(fold_stats)
    
    return all_stats

def print_fold_summary(fold_stats):
    """
    印出單個 fold 的摘要結果
    
    Args:
        fold_stats (dict): fold 統計結果
    """
    fold_num = fold_stats['fold']
    
    # 計算準確率
    sentence_accuracy = fold_stats['correct_predictions'] / max(fold_stats['total_sentences'], 1) * 100
    doc_accuracy = fold_stats['matched_docs'] / max(fold_stats['total_docs'], 1) * 100
    emotion_accuracy = fold_stats['emotion_correct'] / max(fold_stats['emotion_total'], 1) * 100
    cause_accuracy = fold_stats['cause_correct'] / max(fold_stats['cause_total'], 1) * 100
    pair_accuracy = fold_stats['pair_correct'] / max(fold_stats['pair_total'], 1) * 100
    
    print(f"  📊 fold{fold_num} 比較結果:")
    print(f"    - 總文檔數: {fold_stats['total_docs']}")
    print(f"    - 完全正確文檔: {fold_stats['matched_docs']} ({doc_accuracy:.1f}%)")
    print(f"    - 總句子數: {fold_stats['total_sentences']}")
    print(f"    - 完全正確句子: {fold_stats['correct_predictions']} ({sentence_accuracy:.1f}%)")
    print(f"    - 情感標籤準確率: {emotion_accuracy:.1f}% ({fold_stats['emotion_correct']}/{fold_stats['emotion_total']})")
    print(f"    - 原因標籤準確率: {cause_accuracy:.1f}% ({fold_stats['cause_correct']}/{fold_stats['cause_total']})")
    print(f"    - 配對標籤準確率: {pair_accuracy:.1f}% ({fold_stats['pair_correct']}/{fold_stats['pair_total']})")

def print_overall_analysis(all_stats):
    """
    印出總體分析結果
    
    Args:
        all_stats (dict): 全體統計結果
    """
    print("\n" + "="*80)
    print("📊 總體比較分析結果")
    print("="*80)
    
    overall = all_stats['overall_stats']
    
    # 計算總體準確率
    sentence_accuracy = overall['correct_predictions'] / max(overall['total_sentences'], 1) * 100
    doc_accuracy = overall['matched_docs'] / max(overall['total_docs'], 1) * 100
    emotion_accuracy = overall['emotion_correct'] / max(overall['emotion_total'], 1) * 100
    cause_accuracy = overall['cause_correct'] / max(overall['cause_total'], 1) * 100
    pair_accuracy = overall['pair_correct'] / max(overall['pair_total'], 1) * 100
    
    print(f"\n🎯 總體統計:")
    print(f"  - 比較的 fold 數: {all_stats['total_folds']}")
    print(f"  - 總文檔數: {overall['total_docs']}")
    print(f"  - 完全正確文檔: {overall['matched_docs']} ({doc_accuracy:.1f}%)")
    print(f"  - 總句子數: {overall['total_sentences']}")
    print(f"  - 完全正確句子: {overall['correct_predictions']} ({sentence_accuracy:.1f}%)")
    
    print(f"\n📋 各標籤準確率:")
    print(f"  - 情感標籤: {emotion_accuracy:.1f}% ({overall['emotion_correct']}/{overall['emotion_total']})")
    print(f"  - 原因標籤: {cause_accuracy:.1f}% ({overall['cause_correct']}/{overall['cause_total']})")
    print(f"  - 配對標籤: {pair_accuracy:.1f}% ({overall['pair_correct']}/{overall['pair_total']})")
    
    # 顯示各 fold 比較
    print(f"\n📈 各 Fold 文檔準確率比較:")
    for fold_name, fold_data in sorted(all_stats['fold_results'].items()):
        fold_doc_acc = fold_data['matched_docs'] / max(fold_data['total_docs'], 1) * 100
        fold_sent_acc = fold_data['correct_predictions'] / max(fold_data['total_sentences'], 1) * 100
        print(f"  {fold_name}: 文檔 {fold_doc_acc:.1f}%, 句子 {fold_sent_acc:.1f}%")
    
    # 顯示錯誤範例
    print(f"\n🔍 錯誤預測範例:")
    example_count = 0
    for fold_name, fold_data in sorted(all_stats['fold_results'].items()):
        for example in fold_data['mismatched_examples']:
            if example_count < 10:  # 只顯示前10個
                print(f"  {fold_name} Doc {example['doc_id']} 句子 {example['sentence_id']}:")
                print(f"    正確答案: {example['ground_truth']}")
                print(f"    預測結果: {example['prediction']}")
                example_count += 1
            else:
                break
        if example_count >= 10:
            break

def save_comparison_report(all_stats, output_file="prediction_comparison_report.txt"):
    """
    儲存比較報告
    
    Args:
        all_stats (dict): 全體統計結果
        output_file (str): 輸出檔案名稱
    """
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("Pseudo-Labeling 預測結果與正確答案比較報告\n")
            f.write("="*80 + "\n\n")
            
            overall = all_stats['overall_stats']
            
            # 總體統計
            sentence_accuracy = overall['correct_predictions'] / max(overall['total_sentences'], 1) * 100
            doc_accuracy = overall['matched_docs'] / max(overall['total_docs'], 1) * 100
            emotion_accuracy = overall['emotion_correct'] / max(overall['emotion_total'], 1) * 100
            cause_accuracy = overall['cause_correct'] / max(overall['cause_total'], 1) * 100
            pair_accuracy = overall['pair_correct'] / max(overall['pair_total'], 1) * 100
            
            f.write("總體統計:\n")
            f.write(f"比較的 fold 數: {all_stats['total_folds']}\n")
            f.write(f"總文檔數: {overall['total_docs']}\n")
            f.write(f"完全正確文檔: {overall['matched_docs']} ({doc_accuracy:.1f}%)\n")
            f.write(f"總句子數: {overall['total_sentences']}\n")
            f.write(f"完全正確句子: {overall['correct_predictions']} ({sentence_accuracy:.1f}%)\n\n")
            
            f.write("各標籤準確率:\n")
            f.write(f"情感標籤: {emotion_accuracy:.1f}% ({overall['emotion_correct']}/{overall['emotion_total']})\n")
            f.write(f"原因標籤: {cause_accuracy:.1f}% ({overall['cause_correct']}/{overall['cause_total']})\n")
            f.write(f"配對標籤: {pair_accuracy:.1f}% ({overall['pair_correct']}/{overall['pair_total']})\n\n")
            
            # 各 fold 詳細結果
            f.write("各 Fold 詳細結果:\n")
            for fold_name, fold_data in sorted(all_stats['fold_results'].items()):
                fold_doc_acc = fold_data['matched_docs'] / max(fold_data['total_docs'], 1) * 100
                fold_sent_acc = fold_data['correct_predictions'] / max(fold_data['total_sentences'], 1) * 100
                fold_emotion_acc = fold_data['emotion_correct'] / max(fold_data['emotion_total'], 1) * 100
                fold_cause_acc = fold_data['cause_correct'] / max(fold_data['cause_total'], 1) * 100
                fold_pair_acc = fold_data['pair_correct'] / max(fold_data['pair_total'], 1) * 100
                
                f.write(f"\n{fold_name}:\n")
                f.write(f"  總文檔: {fold_data['total_docs']}, 完全正確: {fold_data['matched_docs']} ({fold_doc_acc:.1f}%)\n")
                f.write(f"  總句子: {fold_data['total_sentences']}, 完全正確: {fold_data['correct_predictions']} ({fold_sent_acc:.1f}%)\n")
                f.write(f"  情感: {fold_emotion_acc:.1f}%, 原因: {fold_cause_acc:.1f}%, 配對: {fold_pair_acc:.1f}%\n")
            
            # 錯誤範例
            f.write(f"\n\n錯誤預測範例:\n")
            for fold_name, fold_data in sorted(all_stats['fold_results'].items()):
                if fold_data['mismatched_examples']:
                    f.write(f"\n{fold_name}:\n")
                    for example in fold_data['mismatched_examples'][:10]:  # 每個fold最多10個例子
                        f.write(f"  Doc {example['doc_id']} 句子 {example['sentence_id']}:\n")
                        f.write(f"    正確: {example['ground_truth']}\n")
                        f.write(f"    預測: {example['prediction']}\n")
        
        print(f"\n📄 比較報告已儲存至: {output_file}")
        
    except Exception as e:
        print(f"❌ 儲存報告時發生錯誤: {e}")

def main():
    """主函式"""
    print("開始比較 Pseudo-Labeling 預測結果與正確答案...")
    
    # 檢查必要目錄
    if not os.path.exists("split10"):
        print("找不到 split10 目錄")
        return
    
    if not os.path.exists("pseudo_results"):
        print("找不到 pseudo_results 目錄")
        return
    
    # 執行比較分析
    all_stats = analyze_all_folds()
    
    # 印出總體分析
    print_overall_analysis(all_stats)
    
    # 儲存比較報告
    save_comparison_report(all_stats)
    
    print("\n比較分析完成")

if __name__ == "__main__":
    main()
