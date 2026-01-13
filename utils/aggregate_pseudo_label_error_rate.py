#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
多 Seed 偽標籤錯誤率彙整工具

功能：讀取多個不同 seed 實驗的 pseudo_label_accuracy_summary.txt，
      計算每輪 (R1-R10) 的平均錯誤率 ± 標準差。

使用方式：
    python utils/aggregate_pseudo_label_error_rate.py --dirs dir1 dir2 dir3
    python utils/aggregate_pseudo_label_error_rate.py --dirs dir1 dir2 dir3 --output_dir results --output error_rate_summary.txt
"""

import os
import re
import argparse
import numpy as np


def parse_error_rates(filepath):
    """
    解析 pseudo_label_accuracy_summary.txt，提取每輪的錯誤率
    
    回傳：
        dict: {'R1': 錯誤率, 'R2': 錯誤率, ..., 'R10': 錯誤率, 'total': 總錯誤率}
              若解析失敗則回傳 None
    """
    if not os.path.isfile(filepath):
        return None
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 尋找「輪總計」那一行，提取每輪的 錯誤/總數
    match = re.search(r'輪總計\s+([\d/\s]+)', content)
    if not match:
        return None
    
    # 提取所有 錯誤/總數 配對
    pairs = re.findall(r'(\d+)/(\d+)', match.group(1))
    if len(pairs) < 10:
        return None
    
    # 計算每輪錯誤率
    result = {}
    for i, (error, total) in enumerate(pairs[:10]):
        error, total = int(error), int(total)
        rate = (error / total * 100) if total > 0 else 0
        result[f'R{i+1}'] = rate
    
    # 提取總錯誤率
    total_match = re.search(r'錯誤率:\s*([\d.]+)%', content)
    if total_match:
        result['total'] = float(total_match.group(1))
    
    return result


def aggregate_error_rates(directories, output_path=None):
    """
    彙整多個 seed 實驗的錯誤率
    
    參數：
        directories: 實驗目錄路徑列表
        output_path: 輸出檔案路徑（可選）
    
    回傳：
        dict: 彙整結果 (Mean ± Std)
    """
    all_results = []
    valid_dirs = []
    
    print(f"\n讀取 {len(directories)} 個實驗目錄...")
    
    for dir_path in directories:
        summary_file = os.path.join(dir_path, 'pseudo_label_accuracy_summary.txt')
        result = parse_error_rates(summary_file)
        
        if result:
            all_results.append(result)
            valid_dirs.append(dir_path)
            # 從路徑提取 seed 標籤
            seed_match = re.search(r'seed(\d+)', dir_path)
            seed_str = f"seed{seed_match.group(1)}" if seed_match else os.path.basename(dir_path)[-15:]
            print(f"  ✓ 已讀取: {seed_str}")
        else:
            print(f"  ⚠ 找不到或無法解析: {dir_path}")
    
    if not all_results:
        print("✗ 沒有有效的結果")
        return None
    
    print(f"\n成功讀取 {len(all_results)}/{len(directories)} 個實驗結果")
    
    # 計算每輪的 Mean ± Std
    aggregated = {}
    rounds = [f'R{i}' for i in range(1, 11)]
    
    for r in rounds:
        values = [res[r] for res in all_results if r in res]
        if values:
            aggregated[r] = {
                'mean': np.mean(values),
                'std': np.std(values, ddof=1) if len(values) > 1 else 0.0,
                'values': values  # 保留個別值供驗證
            }
    
    # 計算總錯誤率的 Mean ± Std
    total_values = [res['total'] for res in all_results if 'total' in res]
    if total_values:
        aggregated['total'] = {
            'mean': np.mean(total_values),
            'std': np.std(total_values, ddof=1) if len(total_values) > 1 else 0.0,
            'values': total_values
        }
    
    # 建立輸出內容
    lines = []
    lines.append("=" * 70)
    lines.append("多 Seed 偽標籤錯誤率彙整")
    lines.append("=" * 70)
    
    # 列出實驗目錄
    lines.append(f"\n彙整的實驗目錄 (共 {len(valid_dirs)} 個):")
    seed_labels = []
    for d in valid_dirs:
        seed_match = re.search(r'seed(\d+)', d)
        seed_labels.append(f"seed{seed_match.group(1)}" if seed_match else "?")
        lines.append(f"  - {os.path.basename(d)}")
    
    # 顯示每輪的個別值
    lines.append("\n" + "-" * 70)
    lines.append("各 Seed 每輪錯誤率 (%):")
    lines.append("-" * 70)
    
    # 表頭
    header = f"{'輪次':<8}" + "".join([f"{s:>12}" for s in seed_labels]) + f"{'Mean':>12}{'± Std':>10}"
    lines.append(header)
    lines.append("-" * (8 + 12 * len(seed_labels) + 22))
    
    # 每輪數據
    for r in rounds:
        if r in aggregated:
            a = aggregated[r]
            row = f"{r:<8}"
            row += "".join([f"{v:>11.2f}%" for v in a['values']])
            row += f"{a['mean']:>11.2f}%{a['std']:>9.2f}%"
            lines.append(row)
    
    # 總計
    lines.append("-" * (8 + 12 * len(seed_labels) + 22))
    if 'total' in aggregated:
        a = aggregated['total']
        row = f"{'總計':<8}"
        row += "".join([f"{v:>11.2f}%" for v in a['values']])
        row += f"{a['mean']:>11.2f}%{a['std']:>9.2f}%"
        lines.append(row)
    
    lines.append("\n" + "=" * 70)
    
    # 輸出
    output_text = '\n'.join(lines)
    print("\n" + output_text)
    
    if output_path:
        # 確保目錄存在
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(output_text)
        print(f"\n✓ 結果已儲存至: {output_path}")
    
    return aggregated


def main():
    parser = argparse.ArgumentParser(description='彙整多個 seed 實驗的偽標籤錯誤率')
    parser.add_argument('--dirs', type=str, nargs='+', required=True,
                        help='實驗目錄路徑列表')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='輸出目錄 (預設: results)')
    parser.add_argument('--output', type=str, default='error_rate_summary.txt',
                        help='輸出檔案名稱 (預設: error_rate_summary.txt)')
    
    args = parser.parse_args()
    
    # 組合輸出路徑
    output_path = os.path.join(args.output_dir, args.output)
    
    aggregate_error_rates(args.dirs, output_path)


if __name__ == '__main__':
    main()
