#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
比較 Initial Model 和 Self-Training 結果的直方圖
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.font_manager as fm

# 設定中文字體
plt.rcParams['font.family'] = ['DejaVu Sans']
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'Helvetica']
plt.rcParams['axes.unicode_minus'] = False

def parse_results_file(file_path):
    """解析結果檔案，提取指標數據"""
    data = {}
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # 找到數據行
        data_started = False
        for line in lines:
            line = line.strip()
            
            # 跳過表頭和分隔線
            if 'Metric' in line and 'Precision' in line and 'Recall' in line:
                data_started = True
                continue
            
            if not data_started or line.startswith('-') or not line:
                continue
            
            # 解析數據行
            if '|' in line:
                parts = [part.strip() for part in line.split('|')]
                if len(parts) >= 4:
                    metric_name = parts[0]
                    precision = float(parts[1].replace('%', ''))
                    recall = float(parts[2].replace('%', ''))
                    f1_score = float(parts[3].replace('%', ''))
                    
                    data[metric_name] = {
                        'Precision': precision,
                        'Recall': recall,
                        'F1-Score': f1_score
                    }
        
        return data
        
    except Exception as e:
        print(f"錯誤：無法讀取檔案 {file_path}: {e}")
        return {}

def create_comparison_plot(initial_data, self_training_data):
    """創建比較直方圖"""
    
    # 準備數據
    metrics = list(initial_data.keys())
    measures = ['Precision', 'Recall', 'F1-Score']
    
    # 創建子圖
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    # fig.suptitle('UECA-Prompt few-shot vs ST-UECA-Prompt few-shot Performance Comparison', fontsize=16, fontweight='bold')
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#592E83']
    
    for i, measure in enumerate(measures):
        ax = axes[i]
        
        # 提取數據
        initial_values = [initial_data[metric][measure] for metric in metrics]
        self_training_values = [self_training_data[metric][measure] for metric in metrics]
        
        # 繪製直方圖
        x_pos = np.arange(len(metrics))
        width = 0.35  # 柱狀圖寬度
        
        bars1 = ax.bar(x_pos - width/2, initial_values, width, 
                       label='UECA-Prompt few-shot', color='#2E86AB', alpha=0.8)
        bars2 = ax.bar(x_pos + width/2, self_training_values, width,
                       label='ST-UECA-Prompt few-shot', color='#A23B72', alpha=0.8)
        
        # 添加數值標籤（移除%符號）
        for j, (bar1, bar2, init_val, st_val) in enumerate(zip(bars1, bars2, initial_values, self_training_values)):
            ax.annotate(f'{init_val:.1f}', (bar1.get_x() + bar1.get_width()/2, bar1.get_height()),
                       textcoords="offset points", xytext=(0,5), ha='center', fontsize=10)
            ax.annotate(f'{st_val:.1f}', (bar2.get_x() + bar2.get_width()/2, bar2.get_height()),
                       textcoords="offset points", xytext=(0,5), ha='center', fontsize=10)
        
        # 設定圖表
        ax.set_xlabel('Metrics', fontsize=10)
        ax.set_ylabel(f'{measure} (%)', fontsize=12)
        ax.set_title(f'{measure}', fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(metrics, rotation=45, ha='right', fontsize=10)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 100)
        
        # 設定 Y 軸刻度標籤字型大小
        ax.tick_params(axis='y', labelsize=10)
        
        # 美化
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    return fig

def main():
    """主函數"""
    import argparse
    
    # 設定命令行參數
    parser = argparse.ArgumentParser(description='比較 Initial Model 和 Self-Training 結果')
    parser.add_argument('--initial_file', type=str, default='initial_model_test_averages.txt',
                        help='Initial Model 結果檔案路徑')
    parser.add_argument('--self_training_file', type=str, default='self_training_test_averages.txt',
                        help='Self-Training 結果檔案路徑')
    parser.add_argument('--output_prefix', type=str, default='model_comparison',
                        help='輸出檔案名稱前綴')
    
    args = parser.parse_args()
    
    print("正在生成比較直方圖...")
    print(f"讀取 Initial Model 結果: {args.initial_file}")
    print(f"讀取 Self-Training 結果: {args.self_training_file}")
    
    # 讀取數據
    initial_data = parse_results_file(args.initial_file)
    self_training_data = parse_results_file(args.self_training_file)
    
    if not initial_data or not self_training_data:
        print("無法讀取數據檔案，請檢查檔案路徑和格式")
        return
    
    print(f"Initial Model 數據: {len(initial_data)} 個指標")
    print(f"Self-Training 數據: {len(self_training_data)} 個指標")
    
    # 生成直方圖
    fig = create_comparison_plot(initial_data, self_training_data)
    fig.savefig(f'{args.output_prefix}_bar_plot.png', dpi=300, bbox_inches='tight')
    print(f"直方圖已保存: {args.output_prefix}_bar_plot.png")
    
    print("\n圖表生成完成！")
    print("直方圖：顯示兩個模型在各指標上的表現")
    
    # 打印數據摘要
    print("\n數據摘要：")
    for metric in initial_data.keys():
        init_f1 = initial_data[metric]['F1-Score']
        st_f1 = self_training_data[metric]['F1-Score']
        diff = st_f1 - init_f1
        print(f"{metric:<12}: UECA-Prompt few-shot={init_f1:6.2f}%, ST-UECA-Prompt few-shot={st_f1:6.2f}%, 差異={diff:+6.2f}%")

if __name__ == "__main__":
    main()
