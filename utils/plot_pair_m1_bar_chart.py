#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Pair (m1) 長條圖繪製工具

讀取多個資料夾的 counts_summary_detailed.txt，繪製 10 折累計統計結果的 Pair (m1) 長條圖

使用方式：
    python utils/plot_pair_m1_bar_chart.py --pred_dir folder1/ folder2/ folder3/
    python utils/plot_pair_m1_bar_chart.py --pred_dir folder1/ folder2/ --output pair_m1_comparison.png
"""

import os
import re
import argparse
import matplotlib.pyplot as plt

# 設定字體：中文使用標楷體 (DFKai-SB)，英文/數字使用 Calisto MT
plt.rcParams['font.family'] = ['Calisto MT', 'DFKai-SB', 'DejaVu Serif', 'Noto Serif CJK JP']
plt.rcParams['axes.unicode_minus'] = False

FONT_CHINESE = 'DFKai-SB'
FONT_ENGLISH = 'Calisto MT'


def parse_counts_summary(filepath):
    """
    解析 counts_summary_detailed.txt，提取 Pair (m1) 的 P, R, F1
    
    回傳：
        dict: {'P': float, 'R': float, 'F1': float} 或 None
    """
    if not os.path.exists(filepath):
        return None
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 尋找 Pair (m1) 區塊中的 P (micro), R (micro), F1 (micro)
    # 格式：P (micro): 43.91% | R (micro): 31.86% | F1 (micro): 36.93%
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


def get_short_name(folder_name):
    """
    從資料夾名稱中提取簡短的標籤名稱
    
    參數：
        folder_name: 完整資料夾名稱
    
    回傳：
        str: 簡短標籤 (例如: th0.9_mask_emotion 或 nest_k5_emotion_clause_nestmul3)
    """
    # 優先匹配 emotion_clause / cause_clause + nestmul: 例如 nest_k5_emotion_clause_nestmul3
    clause_mul_match = re.search(
        r'(nest_k\d+_(emotion|cause)_clause_nestmul\d+(?:p\d+)?)(?:_|$)',
        folder_name,
    )
    if clause_mul_match:
        return clause_mul_match.group(1)
    
    # 匹配 nest + nestmul (無 clause):例如 nest_k5_emotion_nestmul0p5
    nest_mul_match = re.search(
        r'(nest_k\d+_(emotion|cause)_nestmul\d+(?:p\d+)?)(?:_|$)',
        folder_name,
    )
    if nest_mul_match:
        return nest_mul_match.group(1)

    # 嘗試匹配 nest 模式: nest_k{數字}_{emotion|cause}
    nest_match = re.search(r'(nest_k\d+_(emotion|cause))', folder_name)
    if nest_match:
        return nest_match.group(1)
    
    # 嘗試匹配 mask 模式: th{數字}_mask{emotion|cause}
    mask_match = re.search(r'th[\d.]+_mask(emotion|cause)', folder_name)
    if mask_match:
        mode = mask_match.group(1)
        th_match = re.search(r'(th[\d.]+)', folder_name)
        th_str = th_match.group(1) if th_match else 'th0.9'
        return f"{th_str}_mask_{mode}"
    
    # 都找不到就回傳最後 30 個字元
    return folder_name[-30:]


def plot_pair_m1_bar_chart(all_data, output_path=None):
    """
    繪製 Pair (m1) 的 P, R, F1 長條圖
    
    參數：
        all_data: list of (folder_name, metrics_dict)
        output_path: 輸出圖片路徑
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor('#DDDDDD')
    ax.set_facecolor('#DDDDDD')
    
    # 準備資料
    labels = [get_short_name(name) for name, _ in all_data]
    p_values = [m['P'] for _, m in all_data]
    r_values = [m['R'] for _, m in all_data]
    f1_values = [m['F1'] for _, m in all_data]
    
    # 設定長條位置
    x = range(len(labels))
    width = 0.25
    
    # 繪製長條
    bars_p = ax.bar([i - width for i in x], p_values, width, label='Precision', color='#4E79A7', edgecolor='white', linewidth=1)
    bars_r = ax.bar(x, r_values, width, label='Recall', color='#F28E2B', edgecolor='white', linewidth=1)
    bars_f1 = ax.bar([i + width for i in x], f1_values, width, label='F1', color='#59A14F', edgecolor='white', linewidth=1)
    
    # 在長條上標註數值
    for bars in [bars_p, bars_r, bars_f1]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords='offset points',
                        ha='center', va='bottom',
                        fontsize=14, fontname=FONT_ENGLISH, fontweight='bold')
    
    # 設定標題和軸標籤
    ax.set_title('Pair (m1) 10 折累計統計結果比較', fontsize=24, fontweight='bold', fontname=FONT_CHINESE)
    ax.set_ylabel('百分比 (%)', fontsize=24, fontname=FONT_CHINESE)
    
    # 設定 X 軸 (標籤旋轉 45 度避免重疊)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=18, fontname=FONT_ENGLISH, rotation=5, ha='right')
    
    # 設定 Y 軸
    ax.set_ylim(0, max(max(p_values), max(r_values), max(f1_values)) + 15)
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
        print(f"圖表已儲存至: {output_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='繪製 Pair (m1) 長條圖')
    parser.add_argument('--pred_dir', type=str, nargs='+', required=True,
                        help='要分析的資料夾路徑（可指定多個）')
    parser.add_argument('--output', type=str, default='pair_m1_comparison.png',
                        help='輸出圖片檔名')
    
    args = parser.parse_args()
    
    all_data = []
    for pred_dir in args.pred_dir:
        folder_name = os.path.basename(pred_dir.rstrip('/\\'))
        summary_path = os.path.join(pred_dir, 'counts_summary_detailed.txt')
        
        metrics = parse_counts_summary(summary_path)
        if metrics:
            all_data.append((folder_name, metrics))
            print(f"已讀取: {folder_name}")
            print(f"  P={metrics['P']:.2f}% R={metrics['R']:.2f}% F1={metrics['F1']:.2f}%")
        else:
            print(f"警告：找不到或無法解析 {summary_path}")
    
    if all_data:
        plot_pair_m1_bar_chart(all_data, args.output)
    else:
        print("錯誤：沒有有效的數據可繪製")


if __name__ == '__main__':
    main()
