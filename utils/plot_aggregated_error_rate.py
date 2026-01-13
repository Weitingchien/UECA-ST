#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
彙整錯誤率折線圖繪製工具

讀取 aggregate_pseudo_label_error_rate.py 產生的彙整檔案，
繪製每輪平均錯誤率的折線圖 (支援自訂標籤)

使用方式：
    python utils/plot_aggregated_error_rate.py --files file1.txt file2.txt
    python utils/plot_aggregated_error_rate.py --files file1.txt file2.txt --labels "label1" "label2"
    python utils/plot_aggregated_error_rate.py --files file1.txt file2.txt --output comparison.png
"""

import os
import re
import argparse
import matplotlib.pyplot as plt

# 設定字體
plt.rcParams['font.family'] = ['Calisto MT', 'DFKai-SB', 'DejaVu Serif', 'Noto Serif CJK JP']
plt.rcParams['axes.unicode_minus'] = False

FONT_CHINESE = 'DFKai-SB'
FONT_ENGLISH = 'Calisto MT'


def parse_aggregated_file(filepath):
    """
    解析彙整檔案，提取每輪的 Mean 錯誤率
    
    回傳：
        list: [R1_mean, R2_mean, ..., R10_mean]，若解析失敗則回傳 None
    """
    if not os.path.isfile(filepath):
        return None
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 提取每輪的 Mean 值 (格式: R1      87.36%...  88.15%  0.84%)
    # Mean 欄位在每行倒數第二個百分比數值
    means = []
    
    for i in range(1, 11):
        # 匹配 R{i} 那行，提取 Mean 值 (倒數第二個百分比)
        pattern = rf'R{i}\s+.*?(\d+\.\d+)%\s+\d+\.\d+%\s*$'
        match = re.search(pattern, content, re.MULTILINE)
        if match:
            means.append(float(match.group(1)))
        else:
            return None
    
    return means


def plot_aggregated_error_rates(files, labels, output_path=None):
    """
    繪製彙整錯誤率折線圖
    
    參數：
        files: 彙整檔案路徑列表
        labels: 對應的標籤名稱列表
        output_path: 輸出圖片路徑（可選）
    """
    # 建立圖表
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor('#DDDDDD')
    ax.set_facecolor('#DDDDDD')
    
    rounds = list(range(1, 11))
    
    # 顏色和標記
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#17becf', '#bcbd22', '#7f7f7f']
    markers = ['o', 's', '^', 'D', 'v', 'p', 'h', '*', 'X', 'P']
    
    all_data = []
    
    # 讀取並解析每個檔案
    for i, (filepath, label) in enumerate(zip(files, labels)):
        means = parse_aggregated_file(filepath)
        
        if means is None:
            print(f"警告：無法解析 {filepath}")
            continue
        
        all_data.append((label, means))
        print(f"已讀取: {label} ({os.path.basename(filepath)})")
    
    if not all_data:
        print("錯誤：沒有有效的數據可繪製")
        return
    
    # 繪製折線
    for i, (label, means) in enumerate(all_data):
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        
        ax.plot(rounds, means,
                label=label,
                color=color,
                marker=marker,
                linewidth=2.5,
                markersize=10)
        
        # 標註最高點和最低點
        max_idx = means.index(max(means))
        min_idx = means.index(min(means))
        
        ax.annotate(f'{means[max_idx]:.1f}%',
                    xy=(rounds[max_idx], means[max_idx]),
                    xytext=(0, 8),
                    textcoords='offset points',
                    ha='center', va='bottom',
                    fontsize=11, fontname=FONT_ENGLISH,
                    color=color, fontweight='bold')
        
        ax.annotate(f'{means[min_idx]:.1f}%',
                    xy=(rounds[min_idx], means[min_idx]),
                    xytext=(0, -8),
                    textcoords='offset points',
                    ha='center', va='top',
                    fontsize=11, fontname=FONT_ENGLISH,
                    color=color, fontweight='bold')
    
    # 設定標題和軸標籤
    ax.set_title('各輪偽標籤錯誤率比較 threshold & NeST(CE)', fontsize=24, fontweight='bold', fontname=FONT_CHINESE)
    ax.set_xlabel('自訓練輪次 (Round)', fontsize=24, fontname=FONT_CHINESE)
    ax.set_ylabel('平均錯誤率 (%)', fontsize=24, fontname=FONT_CHINESE)
    
    # 設定 X 軸刻度
    ax.set_xticks(rounds)
    ax.set_xticklabels(rounds, fontsize=20, fontname=FONT_ENGLISH)
    
    # 設定 Y 軸刻度
    for label in ax.get_yticklabels():
        label.set_fontsize(20)
        label.set_fontname(FONT_ENGLISH)
    
    # 設定圖例
    legend = ax.legend(fontsize=16, loc='best')
    legend.get_frame().set_facecolor('#DDDDDD')
    for text in legend.get_texts():
        text.set_fontname(FONT_ENGLISH)
    
    # 加入網格線
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # 儲存或顯示
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n圖表已儲存至: {output_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='繪製彙整錯誤率折線圖')
    parser.add_argument('--files', type=str, nargs='+', required=True,
                        help='彙整錯誤率檔案路徑列表')
    parser.add_argument('--labels', type=str, nargs='+', default=None,
                        help='對應的標籤名稱（不指定則使用檔名）')
    parser.add_argument('--output', type=str, default='aggregated_error_rate_comparison.png',
                        help='輸出圖片檔名 (預設: aggregated_error_rate_comparison.png)')
    
    args = parser.parse_args()
    
    # 如果沒有指定標籤，使用檔名
    if args.labels is None:
        args.labels = [os.path.basename(f).replace('.txt', '') for f in args.files]
    
    # 確保標籤數量與檔案數量相同
    if len(args.labels) != len(args.files):
        print("錯誤：標籤數量必須與檔案數量相同")
        return
    
    plot_aggregated_error_rates(args.files, args.labels, args.output)


if __name__ == '__main__':
    main()
