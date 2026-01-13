#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
統計檢定工具

功能：比較多個方法的 Pair (m1) 指標，進行成對 t-test 計算 p-value

使用方式：
    python utils/statistical_test.py --files results/file1.txt results/file2.txt
    python utils/statistical_test.py --files results/file1.txt results/file2.txt --labels "Method A" "Method B"

================================================================================
統計學名詞解釋：

1. p-value (p 值)：
   - 回答「觀察到的差異是偶然發生的機率」
   - p 值越小，越有信心說「差異是真實的」
   - 通常 p < 0.05 視為「統計顯著」

2. 成對 t-test (Paired t-test)：
   - 比較「同一組受試者」在「兩種條件」下的表現
   - 例如：同樣的 3 個 seed，分別用 Method A 和 Method B 訓練，比較結果
   - 計算每對差異的平均值和變異，判斷差異是否穩定且顯著

3. 計算公式：
   - 差異: d_i = Method_A[i] - Method_B[i] for i = 1, 2, 3 (每個 seed)
   - 平均差異: d_mean = mean(d)
   - 標準誤差: SE = std(d) / sqrt(n)
   - t 統計量: t = d_mean / SE
   - 再由 t 值和自由度 (n-1) 查表得到 p-value
================================================================================
"""

import os
import re
import argparse
import numpy as np
from scipy import stats


def parse_experiment_dirs(aggregated_file):
    """從彙整檔案中提取實驗目錄列表"""
    if not os.path.isfile(aggregated_file):
        return []
    
    with open(aggregated_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 匹配 "  - prompt_ECPE_..." 格式
    return re.findall(r'^\s*-\s*(prompt_ECPE_[^\s]+)', content, re.MULTILINE)


def get_pair_m1_metrics(directory):
    """從目錄的 counts_summary_detailed.txt 提取 Pair (m1) 的 P, R, F1"""
    filepath = os.path.join(directory, 'counts_summary_detailed.txt')
    if not os.path.isfile(filepath):
        return None
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 匹配: P (micro): 39.53% | R (micro): 28.03% | F1 (micro): 32.80%
    match = re.search(
        r'Pair \(m1\):.*?P \(micro\):\s*([\d.]+)%.*?R \(micro\):\s*([\d.]+)%.*?F1 \(micro\):\s*([\d.]+)%',
        content, re.DOTALL
    )
    
    if match:
        return {
            'P': float(match.group(1)),
            'R': float(match.group(2)),
            'F1': float(match.group(3))
        }
    return None


def get_metrics_for_file(aggregated_file):
    """讀取彙整檔案內所有 seed 的 Pair (m1) 指標"""
    dirs = parse_experiment_dirs(aggregated_file)
    
    p_vals, r_vals, f1_vals = [], [], []
    for d in dirs:
        metrics = get_pair_m1_metrics(d)
        if metrics:
            p_vals.append(metrics['P'])
            r_vals.append(metrics['R'])
            f1_vals.append(metrics['F1'])
    
    return {'P': p_vals, 'R': r_vals, 'F1': f1_vals}


def paired_ttest(values_a, values_b):
    """
    成對 t-test（Paired t-test）
    
    步驟說明：
    1. 計算每對的差異: d_i = a_i - b_i
    2. 計算差異的平均值和標準差
    3. 計算 t 統計量: t = d_mean / (d_std / sqrt(n))
    4. 根據 t 值和自由度 (n-1) 計算 p-value
    
    回傳：
        t_stat: t 統計量，越大代表差異越顯著
        p_value: p 值，越小代表差異越不可能是偶然
    """
    a = np.array(values_a)
    b = np.array(values_b)
    
    # 計算每對的差異
    differences = a - b
    
    n = len(differences)
    if n < 2:
        return None, None
    
    # 差異的平均和標準差
    d_mean = np.mean(differences)
    d_std = np.std(differences, ddof=1)  # 樣本標準差 (ddof=1)
    
    # 標準誤差 (Standard Error)
    se = d_std / np.sqrt(n)
    
    # t 統計量
    if se == 0:
        return None, None
    t_stat = d_mean / se
    
    # 自由度 = n - 1
    df = n - 1
    
    # 計算雙尾 p-value (使用 scipy)
    # p-value = 2 * P(T > |t|)，其中 T 服從自由度為 df 的 t 分布
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
    
    return t_stat, p_value


def run_all_comparisons(all_metrics, labels, metric_name='F1'):
    """執行所有方法配對的統計檢定"""
    n = len(labels)
    
    print(f"\n{'='*70}")
    print(f"{metric_name} 成對 t-test 檢定結果")
    print(f"{'='*70}")
    print(f"\n說明：p < 0.05 表示差異具有統計顯著性\n")
    
    results = []
    
    for i in range(n):
        for j in range(i+1, n):
            vals_i = all_metrics[i][metric_name]
            vals_j = all_metrics[j][metric_name]
            
            if len(vals_i) < 2 or len(vals_j) < 2:
                continue
            
            t_stat, p_value = paired_ttest(vals_i, vals_j)
            
            if t_stat is None:
                continue
            
            # 判斷顯著性
            if p_value < 0.01:
                sig = "*** (p<0.01 非常顯著)"
            elif p_value < 0.05:
                sig = "**  (p<0.05 顯著)"
            elif p_value < 0.10:
                sig = "*   (p<0.10 邊緣顯著)"
            else:
                sig = "    (無顯著差異)"
            
            # 計算 mean 差異
            mean_diff = np.mean(vals_i) - np.mean(vals_j)
            
            result = {
                'A': labels[i],
                'B': labels[j],
                'mean_A': np.mean(vals_i),
                'mean_B': np.mean(vals_j),
                'diff': mean_diff,
                't_stat': t_stat,
                'p_value': p_value,
                'sig': sig
            }
            results.append(result)
            
            print(f"{labels[i]} vs {labels[j]}:")
            print(f"  Mean A: {result['mean_A']:.2f}%  |  Mean B: {result['mean_B']:.2f}%")
            print(f"  差異: {mean_diff:+.2f}%  |  t={t_stat:.3f}  |  p={p_value:.4f}  {sig}")
            print()
    
    return results


def main():
    parser = argparse.ArgumentParser(description='統計檢定工具')
    parser.add_argument('--files', type=str, nargs='+', required=True,
                        help='彙整檔案路徑列表')
    parser.add_argument('--labels', type=str, nargs='+', default=None,
                        help='對應的標籤名稱')
    parser.add_argument('--metric', type=str, default='F1', choices=['P', 'R', 'F1'],
                        help='要檢定的指標 (預設: F1)')
    parser.add_argument('--output', type=str, default=None,
                        help='輸出檔案路徑 (可選)')
    
    args = parser.parse_args()
    
    # 預設標籤
    if args.labels is None:
        args.labels = [os.path.basename(f).replace('.txt', '').replace('_error_rate', '') 
                       for f in args.files]
    
    print("\n讀取各方法的 Pair (m1) 數據...")
    
    all_metrics = []
    for filepath, label in zip(args.files, args.labels):
        metrics = get_metrics_for_file(filepath)
        if metrics['F1']:
            all_metrics.append(metrics)
            print(f"  ✓ {label}: {len(metrics['F1'])} seeds")
            print(f"    {args.metric} values: {metrics[args.metric]}")
        else:
            print(f"  ⚠ {label}: 無法讀取")
            all_metrics.append({'P': [], 'R': [], 'F1': []})
    
    # 執行檢定
    run_all_comparisons(all_metrics, args.labels, args.metric)


if __name__ == '__main__':
    main()
