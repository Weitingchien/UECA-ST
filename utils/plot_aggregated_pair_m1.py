#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
多 Seed Pair (m1) 長條圖繪製工具

讀取彙整錯誤率檔案中的實驗目錄列表，
從各目錄的 counts_summary_detailed.txt 提取 Pair (m1) 的 P, R, F1，
計算 Mean ± Std 並繪製長條圖

使用方式：
    python utils/plot_aggregated_pair_m1.py --files results/file1.txt results/file2.txt
    python utils/plot_aggregated_pair_m1.py --files results/file1.txt --labels "label1" "label2"
    python utils/plot_aggregated_pair_m1.py --files results/file1.txt --output pair_m1_comparison.png
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
    """
    從彙整檔案中提取實驗目錄路徑列表
    
    回傳：
        list: 實驗目錄名稱列表
    """
    if not os.path.isfile(aggregated_file):
        return []
    
    with open(aggregated_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 匹配 "  - prompt_ECPE_..." 格式的目錄名稱
    matches = re.findall(r'^\s*-\s*(prompt_ECPE_[^\s]+)', content, re.MULTILINE)
    return matches


def parse_pair_m1_metrics(directory):
    """
    從目錄的 counts_summary_detailed.txt 提取 Pair (m1) 的 P, R, F1
    
    回傳：
        dict: {'P': float, 'R': float, 'F1': float} 或 None
    """
    filepath = os.path.join(directory, 'counts_summary_detailed.txt')
    if not os.path.isfile(filepath):
        return None
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 匹配 Pair (m1) 區塊的 P (micro), R (micro), F1 (micro)
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
    """
    從彙整檔案讀取各 seed 目錄，計算 Pair (m1) 的 Mean ± Std
    
    回傳：
        dict: {'P_mean', 'P_std', 'R_mean', 'R_std', 'F1_mean', 'F1_std', 'n_seeds'}
    """
    dirs = parse_experiment_dirs(aggregated_file)
    if not dirs:
        return None
    
    p_values, r_values, f1_values = [], [], []
    
    for d in dirs:
        metrics = parse_pair_m1_metrics(d)
        if metrics:
            p_values.append(metrics['P'])
            r_values.append(metrics['R'])
            f1_values.append(metrics['F1'])
    
    if not p_values:
        return None
    
    return {
        'P_mean': np.mean(p_values),
        'P_std': np.std(p_values, ddof=1) if len(p_values) > 1 else 0.0,
        'R_mean': np.mean(r_values),
        'R_std': np.std(r_values, ddof=1) if len(r_values) > 1 else 0.0,
        'F1_mean': np.mean(f1_values),
        'F1_std': np.std(f1_values, ddof=1) if len(f1_values) > 1 else 0.0,
        'n_seeds': len(p_values),
        'P_values': p_values,
        'R_values': r_values,
        'F1_values': f1_values
    }


def plot_aggregated_pair_m1(all_data, labels, output_path=None):
    """
    繪製 Multi-Seed Pair (m1) 的 P, R, F1 長條圖（含誤差條）
    
    參數：
        all_data: list of aggregated metrics dicts
        labels: 對應的標籤列表
        output_path: 輸出圖片路徑
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor('#DDDDDD')
    ax.set_facecolor('#DDDDDD')
    
    # 準備資料
    p_means = [m['P_mean'] for m in all_data]
    p_stds = [m['P_std'] for m in all_data]
    r_means = [m['R_mean'] for m in all_data]
    r_stds = [m['R_std'] for m in all_data]
    f1_means = [m['F1_mean'] for m in all_data]
    f1_stds = [m['F1_std'] for m in all_data]
    
    # 設定長條位置
    x = np.arange(len(labels))
    width = 0.25
    
    # 繪製長條（不含誤差條）
    bars_p = ax.bar(x - width, p_means, width,
                    label='Precision', color='#4E79A7', edgecolor='white', 
                    linewidth=1)
    bars_r = ax.bar(x, r_means, width,
                    label='Recall', color='#F28E2B', edgecolor='white',
                    linewidth=1)
    bars_f1 = ax.bar(x + width, f1_means, width,
                     label='F1', color='#59A14F', edgecolor='white',
                     linewidth=1)
    
    # 在長條上標註數值
    for bars, means in [(bars_p, p_means), 
                        (bars_r, r_means), 
                        (bars_f1, f1_means)]:
        for bar, mean in zip(bars, means):
            height = bar.get_height()
            ax.annotate(f'{mean:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height + 0.5),
                        ha='center', va='bottom',
                        fontsize=12, fontname=FONT_ENGLISH, fontweight='bold')
    
    # 設定標題和軸標籤
    ax.set_title('Pair (m1)', fontsize=24, fontweight='bold', fontname=FONT_CHINESE)
    ax.set_ylabel('百分比 (%)', fontsize=24, fontname=FONT_CHINESE)
    
    # 設定 X 軸
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=14, fontname=FONT_ENGLISH, rotation=15, ha='right')
    
    # 設定 Y 軸範圍
    max_val = max(max(p_means), max(r_means), max(f1_means))
    ax.set_ylim(0, max_val + 15)
    
    for label in ax.get_yticklabels():
        label.set_fontsize(20)
        label.set_fontname(FONT_ENGLISH)
    
    # 設定圖例
    legend = ax.legend(fontsize=18, loc='upper right')
    legend.get_frame().set_facecolor('#DDDDDD')
    for text in legend.get_texts():
        text.set_fontname(FONT_ENGLISH)
    
    ax.grid(True, linestyle='--', alpha=0.7, axis='y')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n圖表已儲存至: {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_f1_only(all_data, labels, output_path=None):
    """
    繪製 Multi-Seed Pair (m1) 的 F1 長條圖（只顯示 F1，不顯示標準差）
    
    參數：
        all_data: list of aggregated metrics dicts
        labels: 對應的標籤列表
        output_path: 輸出圖片路徑
    """
    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor('#DDDDDD')
    ax.set_facecolor('#DDDDDD')
    
    # 準備資料
    f1_means = [m['F1_mean'] for m in all_data]
    
    # 設定長條位置
    x = np.arange(len(labels))
    width = 0.5
    
    # 繪製長條（不含誤差條）
    bars = ax.bar(x, f1_means, width,
                  color='#59A14F', edgecolor='white', linewidth=1.5)
    
    # 在長條上標註數值（只顯示 F1，不顯示標準差）
    for bar, mean in zip(bars, f1_means):
        height = bar.get_height()
        ax.annotate(f'{mean:.2f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height + 0.5),
                    ha='center', va='bottom',
                    fontsize=12, fontname=FONT_ENGLISH, fontweight='bold')
    
    # 設定標題和軸標籤
    ax.set_title('Pair (m1) F1 Score', fontsize=22, fontweight='bold', fontname=FONT_ENGLISH)
    ax.set_ylabel('F1 Score (%)', fontsize=18, fontname=FONT_ENGLISH)
    
    # 設定 X 軸
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12, fontname=FONT_ENGLISH, rotation=15, ha='right')
    
    # 設定 Y 軸範圍
    max_val = max(f1_means)
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


def aggregate_pair_m1_from_dirs(directories):
    """
    直接從目錄列表計算 Pair (m1) 的 Mean ± Std（不需要 error_rate.txt）
    
    參數：
        directories: list of str, 實驗目錄路徑列表
    
    回傳：
        dict: {'P_mean', 'P_std', 'R_mean', 'R_std', 'F1_mean', 'F1_std', 'n_seeds'} 或 None
    """
    p_values, r_values, f1_values = [], [], []
    
    for d in directories:
        metrics = parse_pair_m1_metrics(d)
        if metrics:
            p_values.append(metrics['P'])
            r_values.append(metrics['R'])
            f1_values.append(metrics['F1'])
    
    if not p_values:
        return None
    
    return {
        'P_mean': np.mean(p_values),
        'P_std': np.std(p_values, ddof=1) if len(p_values) > 1 else 0.0,
        'R_mean': np.mean(r_values),
        'R_std': np.std(r_values, ddof=1) if len(r_values) > 1 else 0.0,
        'F1_mean': np.mean(f1_values),
        'F1_std': np.std(f1_values, ddof=1) if len(f1_values) > 1 else 0.0,
        'n_seeds': len(p_values),
        'P_values': p_values,
        'R_values': r_values,
        'F1_values': f1_values
    }


def main():
    parser = argparse.ArgumentParser(
        description='繪製 Multi-Seed Pair (m1) 長條圖',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例：
  # 使用 error_rate.txt 檔案（透過自訓練產生）
  python utils/plot_aggregated_pair_m1.py --files results/file1.txt results/file2.txt

  # 直接指定目錄（不需要 error_rate.txt，適合沒有自訓練的資料夾）
  python utils/plot_aggregated_pair_m1.py \\
      --dirs prompt_ECPE_..._seed20 prompt_ECPE_..._seed42 prompt_ECPE_..._seed60 \\
      --dir_labels "UECA-Prompt few-shot"
        """
    )
    parser.add_argument('--files', type=str, nargs='+', default=[],
                        help='彙整錯誤率檔案路徑列表 (用於取得實驗目錄)')
    parser.add_argument('--labels', type=str, nargs='+', default=None,
                        help='對應 --files 的標籤名稱（不指定則使用檔名）')
    parser.add_argument('--dirs', type=str, nargs='+', action='append', default=[],
                        help='直接指定實驗目錄路徑列表（可多次使用，每組目錄對應一個標籤）')
    parser.add_argument('--dir_labels', type=str, nargs='+', default=[],
                        help='對應 --dirs 的標籤名稱')
    parser.add_argument('--output', type=str, default='pair_m1_multi_seed_comparison.png',
                        help='輸出圖片檔名')
    parser.add_argument('--f1_only', action='store_true',
                        help='只顯示 F1 分數（不顯示 Precision 和 Recall）')
    
    args = parser.parse_args()
    
    all_data = []
    valid_labels = []
    
    # 處理 --files 參數（透過 error_rate.txt）
    if args.files:
        # 如果沒有指定標籤，使用檔名
        if args.labels is None:
            args.labels = []
            for f in args.files:
                name = os.path.basename(f).replace('.txt', '').replace('_error_rate', '')
                args.labels.append(name)
        
        # 確保標籤數量與檔案數量相同
        if len(args.labels) != len(args.files):
            print("錯誤：--labels 數量必須與 --files 數量相同")
            return
        
        print("\n讀取各彙整檔案...")
        
        for filepath, label in zip(args.files, args.labels):
            print(f"\n處理: {os.path.basename(filepath)}")
            
            metrics = aggregate_pair_m1(filepath)
            
            if metrics:
                all_data.append(metrics)
                valid_labels.append(label)
                print(f"  ✓ 成功讀取 {metrics['n_seeds']} 個 seed")
                print(f"    P:  {metrics['P_mean']:.2f}% ± {metrics['P_std']:.2f}%")
                print(f"    R:  {metrics['R_mean']:.2f}% ± {metrics['R_std']:.2f}%")
                print(f"    F1: {metrics['F1_mean']:.2f}% ± {metrics['F1_std']:.2f}%")
            else:
                print(f"  ⚠ 無法讀取")
    
    # 處理 --dirs 參數（直接指定目錄）
    if args.dirs:
        # 檢查標籤數量
        if len(args.dir_labels) != len(args.dirs):
            print(f"錯誤：--dir_labels 數量 ({len(args.dir_labels)}) 必須與 --dirs 組數 ({len(args.dirs)}) 相同")
            return
        
        print("\n讀取直接指定的目錄...")
        
        for dir_group, label in zip(args.dirs, args.dir_labels):
            print(f"\n處理: {label}")
            print(f"  目錄: {dir_group}")
            
            metrics = aggregate_pair_m1_from_dirs(dir_group)
            
            if metrics:
                all_data.append(metrics)
                valid_labels.append(label)
                print(f"  ✓ 成功讀取 {metrics['n_seeds']} 個 seed")
                print(f"    P:  {metrics['P_mean']:.2f}% ± {metrics['P_std']:.2f}%")
                print(f"    R:  {metrics['R_mean']:.2f}% ± {metrics['R_std']:.2f}%")
                print(f"    F1: {metrics['F1_mean']:.2f}% ± {metrics['F1_std']:.2f}%")
            else:
                print(f"  ⚠ 無法讀取")
    
    if not args.files and not args.dirs:
        print("錯誤：請至少指定 --files 或 --dirs 參數")
        return
    
    if all_data:
        if args.f1_only:
            plot_f1_only(all_data, valid_labels, args.output)
        else:
            plot_aggregated_pair_m1(all_data, valid_labels, args.output)
    else:
        print("\n錯誤: 沒有有效的數據可繪製")


if __name__ == '__main__':
    main()
