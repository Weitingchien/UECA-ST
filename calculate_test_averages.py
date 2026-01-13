#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import argparse

def parse_test_results(file_path):
    """解析測試結果檔案，提取每折的指標"""
    results = {}
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 找到所有折的結果
    fold_sections = re.split(r'=== Fold \d+ ===', content)[1:]  # 跳過第一個空的部分
    print(f"總共找到 {len(fold_sections)} 個 fold 區段")
    
    for i, section in enumerate(fold_sections, 1):
        fold = i
        results[fold] = {}
        print(f"正在處理 Fold {fold}")
        
        # 提取百分比指標 - 修復正則表達式以處理空格問題
        metrics_table = re.search(r'Metric.*?\n-+\s*\n(.*?)\n-+', section, re.DOTALL)
        if metrics_table:
            lines = metrics_table.group(1).strip().split('\n')
            metrics_found = 0
            for line in lines:
                if 'Emotion' in line:
                    parts = re.findall(r'(\d+\.\d+)%', line)
                    if len(parts) >= 3:
                        results[fold]['emotion_precision'] = float(parts[0])
                        results[fold]['emotion_recall'] = float(parts[1])
                        results[fold]['emotion_f1'] = float(parts[2])
                        metrics_found += 1
                        print(f"  Emotion: {parts[0]}%, {parts[1]}%, {parts[2]}%")
                elif 'Cause' in line:
                    parts = re.findall(r'(\d+\.\d+)%', line)
                    if len(parts) >= 3:
                        results[fold]['cause_precision'] = float(parts[0])
                        results[fold]['cause_recall'] = float(parts[1])
                        results[fold]['cause_f1'] = float(parts[2])
                        metrics_found += 1
                        print(f"  Cause: {parts[0]}%, {parts[1]}%, {parts[2]}%")
                elif 'Pair (m1)' in line:
                    parts = re.findall(r'(\d+\.\d+)%', line)
                    if len(parts) >= 3:
                        results[fold]['pair_m1_precision'] = float(parts[0])
                        results[fold]['pair_m1_recall'] = float(parts[1])
                        results[fold]['pair_m1_f1'] = float(parts[2])
                        metrics_found += 1
                        print(f"  Pair (m1): {parts[0]}%, {parts[1]}%, {parts[2]}%")
                elif 'Pair (m2)' in line:
                    parts = re.findall(r'(\d+\.\d+)%', line)
                    if len(parts) >= 3:
                        results[fold]['pair_m2_precision'] = float(parts[0])
                        results[fold]['pair_m2_recall'] = float(parts[1])
                        results[fold]['pair_m2_f1'] = float(parts[2])
                        metrics_found += 1
                        print(f"  Pair (m2): {parts[0]}%, {parts[1]}%, {parts[2]}%")
                elif 'Pair (m3)' in line:
                    parts = re.findall(r'(\d+\.\d+)%', line)
                    if len(parts) >= 3:
                        results[fold]['pair_m3_precision'] = float(parts[0])
                        results[fold]['pair_m3_recall'] = float(parts[1])
                        results[fold]['pair_m3_f1'] = float(parts[2])
                        metrics_found += 1
                        print(f"  Pair (m3): {parts[0]}%, {parts[1]}%, {parts[2]}%")
            
            if metrics_found == 0:
                print(f"  [警告] Fold {fold} 沒有找到任何指標!")
            else:
                print(f"  Fold {fold} 成功解析 {metrics_found} 個指標")
        else:
            print(f"  [警告] Fold {fold} 沒有找到指標表格!")
            # 顯示區段的前 200 個字符幫助調試
            print(f"  區段內容預覽: {repr(section[:200])}")
            
        # 檢查該折是否有完整的指標
        expected_metrics = ['emotion_precision', 'emotion_recall', 'emotion_f1',
                          'cause_precision', 'cause_recall', 'cause_f1',
                          'pair_m1_precision', 'pair_m1_recall', 'pair_m1_f1',
                          'pair_m2_precision', 'pair_m2_recall', 'pair_m2_f1',
                          'pair_m3_precision', 'pair_m3_recall', 'pair_m3_f1']
        missing_metrics = [m for m in expected_metrics if m not in results[fold]]
        if missing_metrics:
            print(f"  [警告] Fold {fold} 缺少指標: {missing_metrics}")
        else:
            print(f"  Fold {fold} 所有指標完整")
    
    print(f"最終解析結果: {len(results)} 折")
    for fold_num in range(1, 11):
        if fold_num in results:
            print(f"  Fold {fold_num}: 已解析")
        else:
            print(f"  Fold {fold_num}: ❌ 未解析")
    
    return results

def calculate_averages(results):
    """計算所有指標的平均值"""
    if not results:
        return {}
    
    num_folds = len(results)
    averages = {}
    
    # 初始化累加器 (只需要百分比指標)
    sums = {}
    for metric in ['emotion_precision', 'emotion_recall', 'emotion_f1',
                   'cause_precision', 'cause_recall', 'cause_f1',
                   'pair_m1_precision', 'pair_m1_recall', 'pair_m1_f1',
                   'pair_m2_precision', 'pair_m2_recall', 'pair_m2_f1',
                   'pair_m3_precision', 'pair_m3_recall', 'pair_m3_f1']:
        sums[metric] = 0
    
    # 累加所有折的結果
    for fold, fold_results in results.items():
        for metric in sums:
            if metric in fold_results:
                sums[metric] += fold_results[metric]
    
    # 計算平均值
    for metric in sums:
        averages[metric] = sums[metric] / num_folds
    
    return averages, num_folds

def print_summary(averages, num_folds, output_file=None):
    """輸出摘要報告"""
    output = []
    
    output.append("=" * 60)
    output.append(f"測試結果摘要 (共 {num_folds} 折)")
    output.append("=" * 60)
    output.append("")
    
    # 平均指標
    output.append("平均指標 (各折平均):")
    output.append("-" * 50)
    output.append("Metric       | Precision    | Recall       | F1-Score")
    output.append("-" * 50)
    output.append(f"Emotion      | {averages.get('emotion_precision', 0):.2f}%       | {averages.get('emotion_recall', 0):.2f}%       | {averages.get('emotion_f1', 0):.2f}%")
    output.append(f"Cause        | {averages.get('cause_precision', 0):.2f}%       | {averages.get('cause_recall', 0):.2f}%       | {averages.get('cause_f1', 0):.2f}%")
    output.append(f"Pair (m1)    | {averages.get('pair_m1_precision', 0):.2f}%       | {averages.get('pair_m1_recall', 0):.2f}%       | {averages.get('pair_m1_f1', 0):.2f}%")
    output.append(f"Pair (m2)    | {averages.get('pair_m2_precision', 0):.2f}%       | {averages.get('pair_m2_recall', 0):.2f}%       | {averages.get('pair_m2_f1', 0):.2f}%")
    output.append(f"Pair (m3)    | {averages.get('pair_m3_precision', 0):.2f}%       | {averages.get('pair_m3_recall', 0):.2f}%       | {averages.get('pair_m3_f1', 0):.2f}%")
    output.append("-" * 50)
    
    # 輸出結果
    result_text = "\n".join(output)
    print(result_text)
    
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(result_text)
        print(f"\n結果已儲存至: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='計算測試結果的平均值')
    parser.add_argument('--input', '-i', required=True, help='測試結果檔案路徑')
    parser.add_argument('--output', '-o', help='輸出檔案路徑（選用）')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"錯誤: 檔案不存在 - {args.input}")
        return
    
    print(f"正在解析測試結果: {args.input}")
    
    # 解析測試結果
    results = parse_test_results(args.input)
    
    if not results:
        print("錯誤: 無法解析測試結果")
        return
    
    print(f"成功解析 {len(results)} 折的結果")
    
    # 計算平均值
    averages, num_folds = calculate_averages(results)
    
    # 輸出摘要
    print_summary(averages, num_folds, args.output)

if __name__ == "__main__":
    main()
