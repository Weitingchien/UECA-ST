#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析 pseudo-labeling 預測結果的統計腳本
統計有多少個 doc_id 的預測結果完全是 "非 非 无"
"""

import os
import glob
from collections import defaultdict

def analyze_pseudo_predictions(results_dir="pseudo_results"):
    """
    分析 pseudo-labeling 預測結果
    
    Args:
        results_dir (str): 包含預測結果檔案的目錄
    
    Returns:
        dict: 統計結果
    """
    # 儲存統計結果
    stats = {
        'total_files': 0,
        'total_docs': 0,
        'all_negative_docs': 0,  # 完全是 "非 非 无" 的文檔數
        'partial_positive_docs': 0,  # 有部分正向預測的文檔數
        'fold_details': {}  # 每個 fold 的詳細統計
    }
    
    # 尋找所有 pseudo_text_result.txt 檔案
    pattern = os.path.join(results_dir, "fold*_pseudo_text_result.txt")
    files = sorted(glob.glob(pattern))
    
    if not files:
        print(f"❌ 在 {results_dir} 目錄中找不到任何 fold*_pseudo_text_result.txt 檔案")
        return stats
    
    print(f"📁 找到 {len(files)} 個預測結果檔案:")
    for file in files:
        print(f"  - {os.path.basename(file)}")
    
    stats['total_files'] = len(files)
    
    # 逐個分析每個檔案
    for file_path in files:
        fold_name = os.path.basename(file_path).replace('_pseudo_text_result.txt', '')
        print(f"\n🔍 分析 {fold_name}...")
        
        fold_stats = analyze_single_file(file_path)
        stats['fold_details'][fold_name] = fold_stats
        
        # 累加總統計
        stats['total_docs'] += fold_stats['total_docs']
        stats['all_negative_docs'] += fold_stats['all_negative_docs']
        stats['partial_positive_docs'] += fold_stats['partial_positive_docs']
        
        print(f"  📊 {fold_name} 統計:")
        print(f"    - 總文檔數: {fold_stats['total_docs']}")
        print(f"    - 完全負向預測: {fold_stats['all_negative_docs']} ({fold_stats['all_negative_docs']/max(fold_stats['total_docs'],1)*100:.1f}%)")
        print(f"    - 有正向預測: {fold_stats['partial_positive_docs']} ({fold_stats['partial_positive_docs']/max(fold_stats['total_docs'],1)*100:.1f}%)")
    
    return stats

def analyze_single_file(file_path):
    """
    分析單個預測結果檔案
    
    Args:
        file_path (str): 檔案路徑
        
    Returns:
        dict: 該檔案的統計結果
    """
    fold_stats = {
        'total_docs': 0,
        'all_negative_docs': 0,
        'partial_positive_docs': 0,
        'doc_details': {}  # 每個文檔的詳細統計
    }
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        current_doc_id = None
        current_doc_predictions = []
        
        for line in lines:
            line = line.strip()
            
            if line.startswith("doc_id :"):
                # 處理前一個文檔（如果存在）
                if current_doc_id is not None:
                    process_document(current_doc_id, current_doc_predictions, fold_stats)
                
                # 開始新文檔
                current_doc_id = line.replace("doc_id :", "").strip()
                current_doc_predictions = []
                
            elif line and not line.startswith("doc_id"):
                # 這是預測結果行
                current_doc_predictions.append(line)
        
        # 處理最後一個文檔
        if current_doc_id is not None:
            process_document(current_doc_id, current_doc_predictions, fold_stats)
            
    except Exception as e:
        print(f"❌ 讀取檔案 {file_path} 時發生錯誤: {e}")
    
    return fold_stats

def process_document(doc_id, predictions, fold_stats):
    """
    處理單個文檔的預測結果
    
    Args:
        doc_id (str): 文檔 ID
        predictions (list): 預測結果列表
        fold_stats (dict): fold 統計資料
    """
    if not predictions:
        return
    
    fold_stats['total_docs'] += 1
    
    # 統計預測結果
    all_negative = True  # 是否全部都是 "非 非 无"
    positive_count = 0   # 正向預測數量
    total_predictions = len(predictions)
    
    emotion_predictions = []  # 第一個位置的預測
    cause_predictions = []    # 第二個位置的預測
    pair_predictions = []     # 第三個位置的預測
    
    for pred_line in predictions:
        parts = pred_line.split()
        if len(parts) >= 3:
            emotion_pred = parts[0]  # 情緒預測
            cause_pred = parts[1]    # 原因預測
            pair_pred = parts[2]     # 配對預測
            
            emotion_predictions.append(emotion_pred)
            cause_predictions.append(cause_pred)
            pair_predictions.append(pair_pred)
            
            # 檢查是否為負向預測
            if not (emotion_pred == "非" and cause_pred == "非" and pair_pred == "无"):
                all_negative = False
                positive_count += 1
    
    # 更新統計
    if all_negative:
        fold_stats['all_negative_docs'] += 1
    else:
        fold_stats['partial_positive_docs'] += 1
    
    # 儲存詳細統計
    fold_stats['doc_details'][doc_id] = {
        'total_sentences': total_predictions,
        'positive_predictions': positive_count,
        'all_negative': all_negative,
        'emotion_yes_count': emotion_predictions.count("是"),
        'cause_yes_count': cause_predictions.count("是"),
        'pair_with_number': sum(1 for p in pair_predictions if p != "无" and p != "無")
    }

def print_detailed_analysis(stats):
    """
    印出詳細分析結果
    
    Args:
        stats (dict): 統計結果
    """
    print("\n" + "="*60)
    print("📊 詳細統計分析結果")
    print("="*60)
    
    if stats['total_files'] == 0:
        print("❌ 沒有找到任何檔案進行分析")
        return
    
    # 總體統計
    print(f"\n🎯 總體統計:")
    print(f"  - 分析檔案數: {stats['total_files']}")
    print(f"  - 總文檔數: {stats['total_docs']}")
    print(f"  - 完全負向預測文檔: {stats['all_negative_docs']} ({stats['all_negative_docs']/max(stats['total_docs'],1)*100:.1f}%)")
    print(f"  - 有正向預測文檔: {stats['partial_positive_docs']} ({stats['partial_positive_docs']/max(stats['total_docs'],1)*100:.1f}%)")
    
    # 各 fold 統計
    print(f"\n📋 各 Fold 詳細統計:")
    for fold_name, fold_data in sorted(stats['fold_details'].items()):
        print(f"\n  {fold_name}:")
        print(f"    總文檔: {fold_data['total_docs']}")
        print(f"    完全負向: {fold_data['all_negative_docs']} ({fold_data['all_negative_docs']/max(fold_data['total_docs'],1)*100:.1f}%)")
        print(f"    有正向: {fold_data['partial_positive_docs']} ({fold_data['partial_positive_docs']/max(fold_data['total_docs'],1)*100:.1f}%)")
    
    # 找出有正向預測的文檔範例
    print(f"\n🔍 有正向預測的文檔範例:")
    sample_count = 0
    for fold_name, fold_data in sorted(stats['fold_details'].items()):
        for doc_id, doc_detail in fold_data['doc_details'].items():
            if not doc_detail['all_negative'] and sample_count < 10:
                print(f"  {fold_name} - {doc_id}: {doc_detail['positive_predictions']}/{doc_detail['total_sentences']} 正向預測")
                print(f"    情緒='是': {doc_detail['emotion_yes_count']}, 原因='是': {doc_detail['cause_yes_count']}, 配對數字: {doc_detail['pair_with_number']}")
                sample_count += 1
                if sample_count >= 10:
                    break
        if sample_count >= 10:
            break

def save_analysis_report(stats, output_file="pseudo_prediction_analysis.txt"):
    """
    儲存分析報告到檔案
    
    Args:
        stats (dict): 統計結果
        output_file (str): 輸出檔案名稱
    """
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("Pseudo-Labeling 預測結果分析報告\n")
            f.write("="*60 + "\n\n")
            
            # 總體統計
            f.write("總體統計:\n")
            f.write(f"分析檔案數: {stats['total_files']}\n")
            f.write(f"總文檔數: {stats['total_docs']}\n")
            f.write(f"完全負向預測文檔: {stats['all_negative_docs']} ({stats['all_negative_docs']/max(stats['total_docs'],1)*100:.1f}%)\n")
            f.write(f"有正向預測文檔: {stats['partial_positive_docs']} ({stats['partial_positive_docs']/max(stats['total_docs'],1)*100:.1f}%)\n\n")
            
            # 各 fold 詳細統計
            f.write("各 Fold 詳細統計:\n")
            for fold_name, fold_data in sorted(stats['fold_details'].items()):
                f.write(f"\n{fold_name}:\n")
                f.write(f"  總文檔: {fold_data['total_docs']}\n")
                f.write(f"  完全負向: {fold_data['all_negative_docs']} ({fold_data['all_negative_docs']/max(fold_data['total_docs'],1)*100:.1f}%)\n")
                f.write(f"  有正向: {fold_data['partial_positive_docs']} ({fold_data['partial_positive_docs']/max(fold_data['total_docs'],1)*100:.1f}%)\n")
            
            # 有正向預測的文檔詳細列表
            f.write(f"\n\n有正向預測的文檔詳細列表:\n")
            for fold_name, fold_data in sorted(stats['fold_details'].items()):
                positive_docs = [(doc_id, details) for doc_id, details in fold_data['doc_details'].items() 
                               if not details['all_negative']]
                if positive_docs:
                    f.write(f"\n{fold_name} ({len(positive_docs)} 個文檔):\n")
                    for doc_id, details in positive_docs:
                        f.write(f"  {doc_id}: {details['positive_predictions']}/{details['total_sentences']} 正向, ")
                        f.write(f"情緒={details['emotion_yes_count']}, 原因={details['cause_yes_count']}, 配對={details['pair_with_number']}\n")
        
        print(f"\n📄 分析報告已儲存至: {output_file}")
        
    except Exception as e:
        print(f"❌ 儲存報告時發生錯誤: {e}")

def main():
    """主函式"""
    print("🚀 開始分析 Pseudo-Labeling 預測結果...")
    
    # 檢查 pseudo_results 目錄是否存在
    results_dir = "pseudo_results"
    if not os.path.exists(results_dir):
        print(f"❌ 目錄 {results_dir} 不存在")
        print("請確認已執行 self-training 並產生了預測結果檔案")
        return
    
    # 執行分析
    stats = analyze_pseudo_predictions(results_dir)
    
    # 印出詳細分析
    print_detailed_analysis(stats)
    
    # 儲存分析報告
    save_analysis_report(stats)
    
    print("\n✅ 分析完成！")

if __name__ == "__main__":
    main()
