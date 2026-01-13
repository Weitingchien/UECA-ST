#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
多種子 (Multi-Seed) 結果彙整工具

功能：讀取多個不同 random seed 實驗的 counts_summary_detailed.txt，
並計算跨 seed 的 F1 平均值與標準差。

使用方式：
    python utils/aggregate_multi_seed.py --dirs dir1 dir2 dir3 [--output output.txt]

範例：
    python utils/aggregate_multi_seed.py --dirs \
        prompt_ECPE_home_..._seed20 \
        prompt_ECPE_home_..._seed42 \
        prompt_ECPE_home_..._seed60 \
        --output multi_seed_summary.txt
"""

import argparse
import os
import re
import sys
import numpy as np


def parse_counts_summary(file_path):
    """
    解析 counts_summary_detailed.txt 中「10 折累計統計結果」區塊的 F1 (micro) 數值
    
    回傳格式：
        {
            'Emotion': {'P': float, 'R': float, 'F1': float},
            'Cause': {'P': float, 'R': float, 'F1': float},
            'Pair (m1)': {'P': float, 'R': float, 'F1': float},
            'Pair (m2)': {'P': float, 'R': float, 'F1': float},
            'Pair (m3)': {'P': float, 'R': float, 'F1': float},
        }
    """
    if not os.path.isfile(file_path):
        print(f"  ⚠ 找不到檔案: {file_path}")
        return None
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 正則表達式匹配 P/R/F1 (micro) 數值
    # 格式: P (micro): 65.88% | R (micro): 69.22% | F1 (micro): 67.51%
    pattern = r'P \(micro\): ([\d.]+)% \| R \(micro\): ([\d.]+)% \| F1 \(micro\): ([\d.]+)%'
    
    # 找到所有匹配
    matches = re.findall(pattern, content)
    
    if len(matches) < 5:
        print(f"  ⚠ 無法解析 {file_path}，找到 {len(matches)} 個指標 (預期 5 個)")
        return None
    
    # 指標名稱對應順序 (依 counts_summary_detailed.txt 的順序)
    metric_names = ['Emotion', 'Cause', 'Pair (m1)', 'Pair (m2)', 'Pair (m3)']
    
    results = {}
    for i, name in enumerate(metric_names):
        p, r, f1 = matches[i]
        results[name] = {
            'P': float(p),
            'R': float(r),
            'F1': float(f1)
        }
    
    return results


def aggregate_multi_seed_results(directories, output_path=None):
    """
    彙整多個 seed 實驗的結果
    
    Args:
        directories: list of str, 實驗目錄路徑列表
        output_path: str, 輸出檔案路徑 (可選)
    
    Returns:
        dict: 彙整後的結果 (Mean ± Std)
    """
    all_results = []
    valid_dirs = []
    
    print(f"\n讀取 {len(directories)} 個實驗目錄...")
    
    for dir_path in directories:
        counts_file = os.path.join(dir_path, 'counts_summary_detailed.txt')
        result = parse_counts_summary(counts_file)
        if result is not None:
            all_results.append(result)
            valid_dirs.append(dir_path)
            seed_match = re.search(r'seed(\d+)', dir_path)
            seed_str = f"seed{seed_match.group(1)}" if seed_match else os.path.basename(dir_path)
            print(f"  ✓ 已讀取: {seed_str}")
    
    if len(all_results) == 0:
        print("✗ 沒有有效的結果可供彙整")
        return None
    
    print(f"\n成功讀取 {len(all_results)}/{len(directories)} 個實驗結果")
    
    # 計算 Mean 和 Std
    metric_names = ['Emotion', 'Cause', 'Pair (m1)', 'Pair (m2)', 'Pair (m3)']
    aggregated = {}
    individual_values = {}  # 保存個別數值供顯示
    
    for metric in metric_names:
        p_values = [r[metric]['P'] for r in all_results]
        r_values = [r[metric]['R'] for r in all_results]
        f1_values = [r[metric]['F1'] for r in all_results]
        
        individual_values[metric] = {
            'P': p_values,
            'R': r_values,
            'F1': f1_values
        }
        
        aggregated[metric] = {
            'P_mean': np.mean(p_values),
            'P_std': np.std(p_values, ddof=1) if len(p_values) > 1 else 0.0,
            'R_mean': np.mean(r_values),
            'R_std': np.std(r_values, ddof=1) if len(r_values) > 1 else 0.0,
            'F1_mean': np.mean(f1_values),
            'F1_std': np.std(f1_values, ddof=1) if len(f1_values) > 1 else 0.0,
        }
    
    # 輸出結果
    output_lines = []
    output_lines.append("=" * 70)
    output_lines.append("多種子 (Multi-Seed) 實驗結果彙整")
    output_lines.append("=" * 70)
    output_lines.append(f"\n彙整的實驗目錄 (共 {len(valid_dirs)} 個):")
    
    # 取得 seed 標籤用於表頭
    seed_labels = []
    for d in valid_dirs:
        seed_match = re.search(r'seed(\d+)', d)
        seed_label = f"seed{seed_match.group(1)}" if seed_match else os.path.basename(d)[-10:]
        seed_labels.append(seed_label)
        output_lines.append(f"  - {os.path.basename(d)}")
    
    # 顯示個別數值（用於驗證）
    output_lines.append("\n" + "-" * 70)
    output_lines.append("各 Seed 個別數值 (10 折累計 micro):")
    output_lines.append("-" * 70)
    
    for metric in metric_names:
        output_lines.append(f"\n{metric}:")
        # P row
        p_vals = individual_values[metric]['P']
        p_row = "  P:  " + "  ".join([f"{s}: {v:.2f}%" for s, v in zip(seed_labels, p_vals)])
        output_lines.append(p_row)
        # R row
        r_vals = individual_values[metric]['R']
        r_row = "  R:  " + "  ".join([f"{s}: {v:.2f}%" for s, v in zip(seed_labels, r_vals)])
        output_lines.append(r_row)
        # F1 row
        f1_vals = individual_values[metric]['F1']
        f1_row = "  F1: " + "  ".join([f"{s}: {v:.2f}%" for s, v in zip(seed_labels, f1_vals)])
        output_lines.append(f1_row)
    
    # 顯示 Mean ± Std 彙整
    output_lines.append("\n" + "-" * 70)
    output_lines.append("10 折累計統計結果 (Mean ± Std):")
    output_lines.append("-" * 70)
    
    for metric in metric_names:
        m = aggregated[metric]
        output_lines.append(f"\n{metric}:")
        output_lines.append(f"  P (micro):  {m['P_mean']:.2f}% ± {m['P_std']:.2f}%")
        output_lines.append(f"  R (micro):  {m['R_mean']:.2f}% ± {m['R_std']:.2f}%")
        output_lines.append(f"  F1 (micro): {m['F1_mean']:.2f}% ± {m['F1_std']:.2f}%")
    
    output_lines.append("\n" + "=" * 70)
    
    output_text = '\n'.join(output_lines)
    print("\n" + output_text)
    
    if output_path:
        # 確保輸出目錄存在
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            print(f"  已建立目錄: {output_dir}")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(output_text)
        print(f"\n✓ 結果已儲存至: {output_path}")
    
    return aggregated


def main():
    parser = argparse.ArgumentParser(
        description='彙整多個不同 random seed 實驗的結果',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--dirs', type=str, nargs='+', required=True,
                        help='實驗目錄路徑列表 (每個目錄內需有 counts_summary_detailed.txt)')
    parser.add_argument('--output_dir', type=str, default='multi_seed_results',
                        help='輸出目錄名稱 (預設: multi_seed_results)')
    parser.add_argument('--output', type=str, default='multi_seed_summary.txt',
                        help='輸出檔案名稱 (預設: multi_seed_summary.txt)')
    
    args = parser.parse_args()
    
    # 組合完整輸出路徑
    output_path = os.path.join(args.output_dir, args.output)
    result = aggregate_multi_seed_results(args.dirs, output_path)
    
    return 0 if result is not None else 1


if __name__ == '__main__':
    sys.exit(main())
