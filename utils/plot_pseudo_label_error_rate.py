#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
偽標籤錯誤率折線圖繪製工具

讀取多個資料夾的 pseudo_label_accuracy_summary.txt，繪製每輪錯誤率的折線圖

使用方式：
    python utils/plot_pseudo_label_error_rate.py --pred_dir folder1/ folder2/ folder3/
    python utils/plot_pseudo_label_error_rate.py --pred_dir folder1/ folder2/ --output comparison.png
"""

import os
import re
import argparse
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 設定字體：中文使用標楷體 (DFKai-SB)，英文/數字使用 Calisto MT
# 注意：需先在 WSL 中連結 Windows 字體：sudo ln -s /mnt/c/Windows/Fonts /usr/share/fonts/windows
plt.rcParams['font.family'] = ['Calisto MT', 'DFKai-SB', 'DejaVu Serif', 'Noto Serif CJK JP']
plt.rcParams['axes.unicode_minus'] = False  # 解決負號顯示問題

# 定義字體物件，供需要精確控制時使用
FONT_CHINESE = 'DFKai-SB'  # 標楷體
FONT_ENGLISH = 'Calisto MT'  # Calisto MT 英文字體


def parse_summary_file(filepath):
    """
    解析 pseudo_label_accuracy_summary.txt，提取每輪的錯誤率和原始分子/分母
    
    參數：
        filepath: 檔案路徑
    
    回傳：
        tuple: (error_rates, raw_pairs)
            - error_rates: 10 個輪次的錯誤率百分比 [R1, R2, ..., R10]
            - raw_pairs: 10 個輪次的原始 (錯誤數, 總數) [(err1, tot1), ...]
    """
    # 檢查檔案是否存在
    if not os.path.exists(filepath):
        return None, None
    
    # 讀取檔案內容
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 尋找「輪總計」那一行，格式如：輪總計   341/368    339/368    ...
    match = re.search(r'輪總計\s+([\d/\s]+)', content)
    if not match:
        return None, None
    
    # 提取所有 錯誤/總數 的配對
    round_data = match.group(1)
    pairs = re.findall(r'(\d+)/(\d+)', round_data)
    
    # 計算每輪的錯誤率，並保留原始分子/分母
    error_rates = []
    raw_pairs = []
    for error, total in pairs:
        error = int(error)
        total = int(total)
        rate = (error / total * 100) if total > 0 else 0  # 轉換為百分比
        error_rates.append(rate)
        raw_pairs.append((error, total))
    
    return error_rates, raw_pairs


def get_short_name(folder_name):
    """
    從資料夾名稱中提取簡短的標籤名稱
    
    參數：
        folder_name: 完整資料夾名稱
    
    回傳：
        str: 簡短標籤 (例如: th0.9_mask_emotion 或 nest_k5_emotion)
    """
    # 優先匹配 emotion_clause / cause_clause + nestmul：例如 nest_k5_emotion_clause_nestmul3 / nest_k5_cause_clause_nestmul3
    clause_mul_match = re.search(
        r'(nest_k\d+_(emotion|cause)_clause_nestmul\d+(?:p\d+)?)(?:_|$)',
        folder_name,
    )
    if clause_mul_match:
        return clause_mul_match.group(1)

    # 匹配 nest_k5_cause_nestmul* 或 nest_k5_emotion_nestmul* (不含 _clause 的舊版格式)
    simple_mul_match = re.search(
        r'(nest_k\d+_(emotion|cause)_nestmul\d+(?:p\d+)?)(?:_|$)',
        folder_name,
    )
    if simple_mul_match:
        return simple_mul_match.group(1)

    # 嘗試匹配 nest 模式：nest_k{數字}_{emotion|cause}（不包含後面的 _gamma 等）
    nest_match = re.search(r'(nest_k\d+_(emotion|cause))', folder_name)
    if nest_match:
        return nest_match.group(1)
    
    # 嘗試匹配 mask 模式：th{數字}_mask{emotion|cause}（不包含後面的 _gamma 等）
    mask_match = re.search(r'th[\d.]+_mask(emotion|cause)', folder_name)
    if mask_match:
        # 將 maskemotion/maskcause 轉換為 mask_emotion/mask_cause
        mode = mask_match.group(1)  # emotion 或 cause
        th_match = re.search(r'(th[\d.]+)', folder_name)
        th_str = th_match.group(1) if th_match else 'th0.9'
        return f"{th_str}_mask_{mode}"
    
    # 都找不到就回傳最後 30 個字元
    return folder_name[-30:]


def plot_error_rates(all_data, output_path=None):
    """
    繪製多個資料夾的錯誤率折線圖
    
    參數：
        all_data: list of (folder_name, error_rates)
        output_path: 輸出圖片路徑（可選）
    """
    # 建立圖表，設定大小和背景顏色
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor('#DDDDDD')  # 設定圖片背景顏色
    ax.set_facecolor('#DDDDDD')  # 設定繪圖區域背景顏色
    
    # X 軸：第 1 到第 10 輪
    rounds = list(range(1, 11))
    
    # 定義不同的線條樣式和顏色（12 種顏色和標記，足夠比較多種方法）
    colors = [
        '#1f77b4',  # 藍色
        '#ff7f0e',  # 橘色
        '#2ca02c',  # 綠色
        '#d62728',  # 紅色
        '#9467bd',  # 紫色
        '#8c564b',  # 棕色
        '#e377c2',  # 粉紅色
        '#17becf',  # 青色
        '#bcbd22',  # 黃綠色
        '#7f7f7f',  # 灰色
        '#ff6b6b',  # 淺紅色
        '#4ecdc4',  # 湖水綠
    ]
    markers = ['o', 's', '^', 'D', 'v', 'p', 'h', '*', 'X', 'P', '<', '>']
    
    # 繪製每個資料夾的折線
    for i, (folder_name, error_rates, raw_pairs) in enumerate(all_data):
        label = get_short_name(folder_name)  # 取得簡短標籤
        color = colors[i % len(colors)]      # 循環使用顏色
        marker = markers[i % len(markers)]   # 循環使用標記
        
        # 繪製折線，設定線寬和標記大小
        ax.plot(rounds, error_rates, 
                label=label, 
                color=color, 
                marker=marker, 
                linewidth=2.5, 
                markersize=10)
        
        # 找出最高點和最低點
        max_idx = error_rates.index(max(error_rates))
        min_idx = error_rates.index(min(error_rates))
        max_val = error_rates[max_idx]
        min_val = error_rates[min_idx]
        max_err, max_tot = raw_pairs[max_idx]
        min_err, min_tot = raw_pairs[min_idx]
        
        # 在最高點標註數值（往上偏移），顯示 分子/分母 (百分比%)
        ax.annotate(f'{max_err}/{max_tot}\n({max_val:.1f}%)', 
                    xy=(rounds[max_idx], max_val),
                    xytext=(0, 8),  # 往上偏移 8 點
                    textcoords='offset points',
                    ha='center', va='bottom',
                    fontsize=11, fontname=FONT_ENGLISH,
                    color=color, fontweight='bold')
        
        # 在最低點標註數值（往下偏移），顯示 分子/分母 (百分比%)
        ax.annotate(f'{min_err}/{min_tot}\n({min_val:.1f}%)', 
                    xy=(rounds[min_idx], min_val),
                    xytext=(0, -8),  # 往下偏移 8 點
                    textcoords='offset points',
                    ha='center', va='top',
                    fontsize=11, fontname=FONT_ENGLISH,
                    color=color, fontweight='bold')
    
    # 設定標題和軸標籤，字體大小 24
    # 中文部分使用標楷體，英文部分使用 Calisto MT
    ax.set_title('各輪偽標籤錯誤率比較', fontsize=24, fontweight='bold', fontname=FONT_CHINESE)
    ax.set_xlabel('自訓練輪次 (Round)', fontsize=24, fontname=FONT_CHINESE)
    ax.set_ylabel('錯誤率 (%)', fontsize=24, fontname=FONT_CHINESE)
    
    # 設定 X 軸刻度為 1-10（數字使用 Calisto MT）
    ax.set_xticks(rounds)
    ax.set_xticklabels(rounds, fontsize=20, fontname=FONT_ENGLISH)
    
    # 設定 Y 軸刻度字體大小和字體
    for label in ax.get_yticklabels():
        label.set_fontsize(20)
        label.set_fontname(FONT_ENGLISH)
    
    # 設定圖例，字體大小 18（圖例中包含英文/數字，使用 Calisto MT）
    legend = ax.legend(fontsize=18, loc='best')
    legend.get_frame().set_facecolor('#DDDDDD')  # 設定圖例背景顏色
    for text in legend.get_texts():
        text.set_fontname(FONT_ENGLISH)
    
    # 加入網格線，便於閱讀
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # 調整邊距
    plt.tight_layout()
    
    # 儲存或顯示圖表
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"圖表已儲存至: {output_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    # 建立參數解析器
    parser = argparse.ArgumentParser(description='繪製偽標籤錯誤率折線圖')
    parser.add_argument('--pred_dir', type=str, nargs='+', required=True,
                        help='要分析的資料夾路徑（可指定多個）')
    parser.add_argument('--output', type=str, default='pseudo_label_error_rate_comparison.png',
                        help='輸出圖片檔名 (預設: pseudo_label_error_rate_comparison.png)')
    
    args = parser.parse_args()
    
    # 收集所有資料夾的錯誤率數據
    all_data = []
    
    for pred_dir in args.pred_dir:
        # 取得資料夾名稱
        folder_name = os.path.basename(pred_dir.rstrip('/\\'))
        
        # 組合 summary 檔案路徑
        summary_path = os.path.join(pred_dir, 'pseudo_label_accuracy_summary.txt')
        
        # 解析檔案取得錯誤率和原始分子/分母
        error_rates, raw_pairs = parse_summary_file(summary_path)
        
        if error_rates:
            all_data.append((folder_name, error_rates, raw_pairs))
            print(f"已讀取: {folder_name}")
        else:
            print(f"警告：找不到或無法解析 {summary_path}")
    
    # 繪製圖表
    if all_data:
        plot_error_rates(all_data, args.output)
    else:
        print("錯誤：沒有有效的數據可繪製")


if __name__ == '__main__':
    main()
