#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Pair (m1) F1 長條圖繪製工具（只顯示 F1）

功能：
- 讀取多個 Multi-Seed 實驗的彙整資料
- 讀取 Supervised Full 實驗的單一結果
- 只繪製 F1 長條圖（不含 P 和 R）

使用方式：
    python utils/plot_pair_m1_f1_only.py \
        --multi_seed_files results/init.txt results/st.txt \
        --multi_seed_labels "UECA-Prompt few-shot" "ST-UECA-Prompt few-shot" \
        --supervised_dir prompt_ECPE_home_train9_test1_disjoint_UECA_CE_supervised_full \
        --supervised_label "UECA-Prompt full (90%)" \
        --output pair_m1_f1_comparison.png
"""

import os
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt

# 設定字體
plt.rcParams['font.family'] = ['Calisto MT', 'DFKai-SB', 'DejaVu Serif', 'Noto Serif CJK JP']
plt.rcParams['axes.unicode_minus'] = False

FONT_CHINESE = 'DFKai-SB'
FONT_ENGLISH = 'Calisto MT'


def parse_experiment_dirs(aggregated_file):
    """從彙整檔案中提取實驗目錄路徑列表"""
    if not os.path.isfile(aggregated_file):
        return []
    
    with open(aggregated_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    matches = re.findall(r'^\s*-\s*(prompt_ECPE_[^\s]+)', content, re.MULTILINE)
    return matches


def parse_pair_m1_metrics(directory):
    """從目錄的 counts_summary_detailed.txt 提取 Pair (m1) 的 P, R, F1"""
    filepath = os.path.join(directory, 'counts_summary_detailed.txt')
    if not os.path.isfile(filepath):
        return None
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    match = re.search(
        r'Pair \(m1\):.*?P \(micro\):\s*([\d.]+)%\s*\|\s*R \(micro\):\s*([\d.]+)%\s*\|\s*F1 \(micro\):\s*([\d.]+)%',
        content, re.DOTALL
    )
    
    if not match:
        return None
    
    return {
        'P': float(match.group(1)),
        'R': float(match.group(2)),
        'F1': float(match.group(3))
    }


def aggregate_pair_m1(aggregated_file):
    """從彙整檔案讀取各 seed 目錄，計算 Pair (m1) 的 Mean ± Std"""
    dirs = parse_experiment_dirs(aggregated_file)
    if not dirs:
        return None
    
    f1_values = []
    
    for d in dirs:
        metrics = parse_pair_m1_metrics(d)
        if metrics:
            f1_values.append(metrics['F1'])
    
    if not f1_values:
        return None
    
    return {
        'F1_mean': np.mean(f1_values),
        'F1_std': np.std(f1_values, ddof=1) if len(f1_values) > 1 else 0.0,
        'n_seeds': len(f1_values),
        'F1_values': f1_values
    }


def plot_f1_only(all_data, labels, output_path=None):
    """
    繪製 Pair (m1) F1 長條圖（只顯示 F1）
    
    參數：
        all_data: list of dicts, 每個 dict 包含 'F1_mean', 'F1_std' (可選)
        labels: 對應的標籤列表
        output_path: 輸出圖片路徑
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor('#DDDDDD')
    ax.set_facecolor('#DDDDDD')
    
    # 準備資料
    f1_means = [m['F1_mean'] for m in all_data]
    f1_stds = [m.get('F1_std', 0.0) for m in all_data]
    
    # 設定長條位置
    x = np.arange(len(labels))
    width = 0.5
    
    # 定義顏色：最後一個 (Supervised Full) 使用不同顏色
    colors = ['#4E79A7'] * (len(labels) - 1) + ['#E15759']  # 藍色 + 紅色
    
    # 繪製長條（含誤差條）
    bars = ax.bar(x, f1_means, width, yerr=f1_stds, capsize=5,
                  color=colors, edgecolor='white', linewidth=1.5,
                  error_kw={'elinewidth': 2, 'capthick': 2})
    
    # 在長條上標註數值
    for bar, mean, std in zip(bars, f1_means, f1_stds):
        height = bar.get_height()
        if std > 0:
            label_text = f'{mean:.2f}%\n±{std:.2f}'
        else:
            label_text = f'{mean:.2f}%'
        ax.annotate(label_text,
                    xy=(bar.get_x() + bar.get_width() / 2, height + std + 0.5),
                    ha='center', va='bottom',
                    fontsize=14, fontname=FONT_ENGLISH, fontweight='bold')
    
    # 設定標題和軸標籤
    ax.set_title('Pair (m1) F1 Score Comparison', fontsize=22, fontweight='bold', fontname=FONT_ENGLISH)
    ax.set_ylabel('F1 Score (%)', fontsize=18, fontname=FONT_ENGLISH)
    
    # 設定 X 軸
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=13, fontname=FONT_ENGLISH, rotation=15, ha='right')
    
    # 設定 Y 軸範圍
    max_val = max(f1_means) + max(f1_stds) if f1_stds else max(f1_means)
    ax.set_ylim(0, max_val + 12)
    
    for label in ax.get_yticklabels():
        label.set_fontsize(16)
        label.set_fontname(FONT_ENGLISH)
    
    ax.grid(True, linestyle='--', alpha=0.7, axis='y')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n圖表已儲存至: {output_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='繪製 Pair (m1) F1 長條圖（只顯示 F1）')
    parser.add_argument('--multi_seed_files', type=str, nargs='+', default=[],
                        help='Multi-Seed 彙整檔案路徑列表')
    parser.add_argument('--multi_seed_labels', type=str, nargs='+', default=[],
                        help='Multi-Seed 對應的標籤名稱')
    parser.add_argument('--supervised_dir', type=str, default=None,
                        help='Supervised Full 實驗目錄路徑')
    parser.add_argument('--supervised_label', type=str, default='Supervised Full (90%)',
                        help='Supervised Full 的標籤名稱')
    parser.add_argument('--output', type=str, default='pair_m1_f1_comparison.png',
                        help='輸出圖片檔名')
    
    args = parser.parse_args()
    
    all_data = []
    all_labels = []
    
    # 處理 Multi-Seed 檔案
    print("\n讀取 Multi-Seed 彙整檔案...")
    for filepath, label in zip(args.multi_seed_files, args.multi_seed_labels):
        print(f"\n處理: {os.path.basename(filepath)}")
        metrics = aggregate_pair_m1(filepath)
        
        if metrics:
            all_data.append(metrics)
            all_labels.append(label)
            print(f"  ✓ 成功讀取 {metrics['n_seeds']} 個 seed")
            print(f"    F1: {metrics['F1_mean']:.2f}% ± {metrics['F1_std']:.2f}%")
        else:
            print(f"  ⚠ 無法讀取")
    
    # 處理 Supervised Full 目錄
    if args.supervised_dir:
        print(f"\n處理 Supervised Full: {args.supervised_dir}")
        metrics = parse_pair_m1_metrics(args.supervised_dir)
        
        if metrics:
            all_data.append({
                'F1_mean': metrics['F1'],
                'F1_std': 0.0,  # 單一實驗沒有標準差
                'n_seeds': 1
            })
            all_labels.append(args.supervised_label)
            print(f"  ✓ F1: {metrics['F1']:.2f}%")
        else:
            print(f"  ⚠ 無法讀取")
    
    if all_data:
        plot_f1_only(all_data, all_labels, args.output)
    else:
        print("\n錯誤: 沒有有效的數據可繪製")


if __name__ == '__main__':
    main()
