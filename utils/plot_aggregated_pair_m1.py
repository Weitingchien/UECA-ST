#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
"""
多 Seed Pair (m1) 長條圖繪製工具

讀取彙整錯誤率檔案中的實驗目錄列表，
從各目錄的 counts_summary_detailed.txt 提取 Pair (m1) 的 P, R, F1，
計算 Mean ± Std 並繪製長條圖

使用方式：
    # --- 新版：使用 summary files + experiments-root ---
    python utils/plot_aggregated_pair_m1.py \
        --summary-files $(ls results_ep_split10_t1te1v1_u7_disjoint/*_CE_summary.txt) \
        --experiments-root ep_split10_t1te1v1_u7_disjoint --f1_only

    # --- 舊版：使用 error_rate.txt ---
    python utils/plot_aggregated_pair_m1.py --files results/file1.txt results/file2.txt

    # --- 舊版：直接指定目錄 ---
    python utils/plot_aggregated_pair_m1.py --dirs dir1 dir2 dir3 --dir_labels "label"
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

# 與 plot_val_pair_f1_per_round.py / plot_unselected_correct_comparison.py 一致的配色
# 前 9 色 (index 0-8) 對應原有 9 組實驗，index 9 保留給 consistency
DISTINCT_COLORS = [
    "#d62728",  # 紅       (nest_k5_nestmul3)
    "#1f77b4",  # 藍       (nest_k5_nestmul5)
    "#2ca02c",  # 綠       (hybrid_alt)
    "#ff7f0e",  # 橙       (hybrid_rev_switch5)
    "#9467bd",  # 紫       (hybrid_switch5)
    "#e377c2",  # 粉紅     (maskall)
    "#17becf",  # 青       (maskboth)
    "#8c564b",  # 棕       (maskcause)
    "#bcbd22",  # 黃綠     (maskemotion)
    "#000000",  # 黑       (consistency)
    "#393b79",  # 深藍紫
    "#e7969c",  # 淡珊瑚
]

# 與 pair_m1_multi_seed_comparison.png 對齊的固定顏色映射
LABEL_FIXED_COLORS = {
    'st0_CE': '#d62728',
    'th0.9_maskemotion_consistency_CE': '#1f77b4',
    'th0.9_maskemotion_CE': '#2ca02c',
    'th0.9_maskemotion_somc_CE': '#4A148C',
    'nest_k5_knncause_clause_mask_nestmul5_somc_CE': '#9467bd',
    'nest_k5_knncause_clause_mask_nestmul5_CE': '#e377c2',
    'consistency_only_all_equal_CE': '#1f77b4',
    'nest_k5_knncause_clause_mask_nestmul7_consistency_CE': '#2ca02c',
    'nest_k5_knncause_clause_mask_nestmul7_CE': '#ff7f0e',
    'w/o st': '#d62728',
    'nest (mult=1, subtask3)': '#1f77b4',
    '(mult=1, subtask3)': '#1f77b4',
    'nest (mult=1)': '#2ca02c',
    '(mult=1)': '#2ca02c',
    'mult=1': '#2ca02c',
    'nest (mult=1.5, subtask3)': '#ff7f0e',
    '(mult=1.5, subtask3)': '#ff7f0e',
    'nest (mult=1.5)': '#9467bd',
    '(mult=1.5)': '#9467bd',
    'mult=1.5': '#9467bd',
}


def normalize_label_for_color(label):
    return re.sub(r'\s+', ' ', label.strip().lower())


def simplify_display_label(label):
    """移除 NeST 前綴，保留括號內的實驗設定。"""
    simplified = re.sub(r'^\s*NeST\s*', '', label, flags=re.IGNORECASE).strip()
    return simplified or label


def extract_mult_group(label):
    """從標籤中抽出 mult 值，供相鄰分組用。"""
    match = re.search(r'mult\s*=\s*([\d.]+)', label, flags=re.IGNORECASE)
    if match:
        return match.group(1)
    return normalize_label_for_color(label)


def build_grouped_bar_positions(labels, intra_gap=0.78, inter_gap=1.58):
    """同一 mult 組內緊鄰，不同 mult 組之間保留較大空隙。"""
    if not labels:
        return np.array([])

    positions = [0.0]
    prev_group = extract_mult_group(labels[0])

    for label in labels[1:]:
        current_group = extract_mult_group(label)
        step = intra_gap if current_group == prev_group else inter_gap
        positions.append(positions[-1] + step)
        prev_group = current_group

    return np.array(positions, dtype=np.float32)


def style_y_grid(ax):
    ax.set_axisbelow(False)
    ax.grid(True, linestyle='--', linewidth=1.1, alpha=0.95, color='#ffffff', axis='y')
    for gridline in ax.get_ygridlines():
        gridline.set_zorder(10)


def get_color_by_label(label, idx):
    """優先使用固定顏色映射，未命中時退回循序色盤。"""
    normalized = normalize_label_for_color(label)
    return LABEL_FIXED_COLORS.get(normalized, LABEL_FIXED_COLORS.get(label, DISTINCT_COLORS[idx % len(DISTINCT_COLORS)]))


def assign_distinct_colors(labels):
    """為當前圖上的標籤分配顏色，盡量避免同圖撞色。"""
    assigned = []
    used = set()

    # 先放固定映射色，保持既有語意。
    for label in labels:
        normalized = normalize_label_for_color(label)
        fixed = LABEL_FIXED_COLORS.get(normalized, LABEL_FIXED_COLORS.get(label))
        if fixed is not None:
            assigned.append(fixed)
            used.add(fixed.lower())
        else:
            assigned.append(None)

    # 其餘標籤從色盤挑尚未使用的顏色。
    for i, color in enumerate(assigned):
        if color is not None:
            continue

        chosen = None
        for candidate in DISTINCT_COLORS:
            if candidate.lower() not in used:
                chosen = candidate
                break

        # 若顏色已用盡，才退回循環使用。
        if chosen is None:
            chosen = DISTINCT_COLORS[i % len(DISTINCT_COLORS)]

        assigned[i] = chosen
        used.add(chosen.lower())

    return assigned


def read_experiment_names_from_summary(summary_file: str) -> list[str]:
    """從 summary txt 解析所有 prompt_ECPE_few_shot_ST_... 實驗目錄名稱"""
    with open(summary_file, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()
    names: list[str] = []
    for line in text.splitlines():
        m = re.match(r"\s*-\s*(prompt_ECPE_few_shot_ST_[^\s]+)", line)
        if m:
            names.append(m.group(1).strip())
    return names


def get_short_label(name: str) -> str:
    """
    從資料夾名稱或 summary 檔名擷取可辨識的簡短標籤。

    規則：
      1) 去掉 _summary.txt 後綴
      2) 從 'th' 開始擷取；若無 th 則從 'nest_k5_' 開始擷取
      3) 移除共用參數片段:
         _gamma[\d.]+, _st\d+, _ste\d+, _seed\d+, _retain_pseudo,
         _nbeta[\d.]+, _nm[\d.]+
      4) 移除 knncause 前面的冗餘 cause(_clause)?_
    """
    s = re.sub(r'_summary\.txt$', '', name, flags=re.IGNORECASE)
    # 特殊情況：_st0_ 表示僅 few-shot 無自訓練
    if '_st0_' in s or s.endswith('_st0'):
        suffix_match = re.search(r'_(CE|EC)$', s)
        suffix = f"_{suffix_match.group(1)}" if suffix_match else ""
        return f"st0{suffix}"
    m = re.search(r'(th[\d.]+_.*)', s)
    if m:
        s = m.group(1)
    else:
        m2 = re.search(r'(nest_k\d+_.*)', s)
        if m2:
            s = m2.group(1)
    s = re.sub(r'_seed\d+', '', s)
    s = re.sub(r'_gamma[\d.]+', '', s)
    s = re.sub(r'_st\d+', '', s)
    s = re.sub(r'_ste\d+', '', s)
    s = re.sub(r'_retain_pseudo', '', s)
    s = re.sub(r'_nbeta[\d.]+', '', s)
    s = re.sub(r'_nm[\d.]+', '', s)
    s = re.sub(r'cause(_clause)?_(?=knncause)', '', s)
    # 將 consistency_somc 簡化為 somc，讓圖例更精簡
    s = re.sub(r'_consistency_somc_', '_somc_', s)
    s = re.sub(r'__+', '_', s)
    s = s.strip('_')
    return s


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


def parse_pair_m1_metrics_from_summary(summary_file: str):
    """從 summary txt 直接解析 Pair (m1) 的 Mean ± Std，供資料夾缺失時備援使用。"""
    if not os.path.isfile(summary_file):
        return None

    with open(summary_file, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    # 只在「10 折累計統計結果 (Mean ± Std)」區塊中找 Pair (m1)，
    # 避免從前面的各 seed 區塊一路跨段誤抓到 Emotion/Cause 的 micro 統計。
    stats_section_match = re.search(
        r'10 折累計統計結果 \(Mean ± Std\):\s*\n(.*?)(?:\n={6,}|\Z)',
        content,
        re.DOTALL,
    )
    if not stats_section_match:
        return None

    stats_section = stats_section_match.group(1)
    match = re.search(
        r'Pair \(m1\):\s*\n'
        r'\s*P \(micro\):\s*([\d.]+)%\s*±\s*([\d.]+)%\s*\n'
        r'\s*R \(micro\):\s*([\d.]+)%\s*±\s*([\d.]+)%\s*\n'
        r'\s*F1 \(micro\):\s*([\d.]+)%\s*±\s*([\d.]+)%',
        stats_section,
        re.DOTALL,
    )
    if not match:
        return None

    seed_count_match = re.search(r'共\s*(\d+)\s*個', content)
    n_seeds = int(seed_count_match.group(1)) if seed_count_match else 0

    return {
        'P_mean': float(match.group(1)),
        'P_std': float(match.group(2)),
        'R_mean': float(match.group(3)),
        'R_std': float(match.group(4)),
        'F1_mean': float(match.group(5)),
        'F1_std': float(match.group(6)),
        'n_seeds': n_seeds,
        'P_values': [],
        'R_values': [],
        'F1_values': [],
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


def plot_f1_only(all_data, labels, output_path=None, highlight_label=None, highlight_color=None):
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
    
    display_labels = [simplify_display_label(label) for label in labels]

    # 準備資料
    f1_means = [m['F1_mean'] for m in all_data]
    
    # 設定長條位置
    x = build_grouped_bar_positions(display_labels)
    width = 0.78
    
    # 繪製長條（不含誤差條），使用 DISTINCT_COLORS
    colors = assign_distinct_colors(display_labels)
    # 允許針對指定標籤覆蓋顏色，避免與其他方法撞色
    if highlight_label and highlight_color and highlight_label in display_labels:
        idx = display_labels.index(highlight_label)
        colors[idx] = highlight_color
    bars = ax.bar(x, f1_means, width,
                  color=colors, edgecolor='white', linewidth=1.5, zorder=2)
    
    # 在長條上標註數值（只顯示 F1，不顯示標準差）
    for i, (bar, mean) in enumerate(zip(bars, f1_means)):
        height = bar.get_height()
        ax.annotate(f'{mean:.2f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height + 0.5),
                    ha='center', va='bottom',
                    fontsize=11, fontname=FONT_ENGLISH, fontweight='bold',
                    color=colors[i])
    
    # 設定標題和軸標籤
    ax.set_title('Pair (m1) F1 Score', fontsize=22, fontweight='bold', fontname=FONT_ENGLISH)
    ax.set_ylabel('F1 Score (%)', fontsize=18, fontname=FONT_ENGLISH)
    
    # 設定 X 軸
    ax.set_xticks(x)
    ax.set_xticklabels(display_labels, fontsize=12, fontname=FONT_ENGLISH, rotation=15, ha='right')
    ax.margins(x=0.05)
    
    # 設定 Y 軸範圍
    max_val = max(f1_means)
    ax.set_ylim(0, max_val + 12)
    
    for label in ax.get_yticklabels():
        label.set_fontsize(16)
        label.set_fontname(FONT_ENGLISH)
    
    style_y_grid(ax)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n圖表已儲存至: {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_metric_only(all_data, labels, metric='f1', output_path=None, highlight_label=None, highlight_color=None):
    """繪製單一指標長條圖 (precision / recall / f1)。"""
    metric_map = {
        'precision': ('P_mean', 'Pair (m1) Precision Score', 'Precision (%)'),
        'recall': ('R_mean', 'Pair (m1) Recall Score', 'Recall (%)'),
        'f1': ('F1_mean', 'Pair (m1) F1 Score', 'F1 Score (%)'),
    }
    key, title, ylabel = metric_map.get(metric, metric_map['f1'])

    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor('#DDDDDD')
    ax.set_facecolor('#DDDDDD')

    display_labels = [simplify_display_label(label) for label in labels]

    values = [m[key] for m in all_data]
    x = build_grouped_bar_positions(display_labels)
    width = 0.78

    colors = assign_distinct_colors(display_labels)
    if highlight_label and highlight_color and highlight_label in display_labels:
        idx = display_labels.index(highlight_label)
        colors[idx] = highlight_color

    bars = ax.bar(x, values, width, color=colors, edgecolor='white', linewidth=1.5, zorder=2)

    for i, (bar, val) in enumerate(zip(bars, values)):
        height = bar.get_height()
        ax.annotate(
            f'{val:.2f}%',
            xy=(bar.get_x() + bar.get_width() / 2, height + 0.5),
            ha='center', va='bottom',
            fontsize=11, fontname=FONT_ENGLISH, fontweight='bold',
            color=colors[i],
        )

    ax.set_title(title, fontsize=22, fontweight='bold', fontname=FONT_ENGLISH)
    ax.set_ylabel(ylabel, fontsize=18, fontname=FONT_ENGLISH)
    ax.set_xticks(x)
    ax.set_xticklabels(display_labels, fontsize=12, fontname=FONT_ENGLISH, rotation=15, ha='right')
    ax.margins(x=0.05)

    max_val = max(values)
    ax.set_ylim(0, max_val + 12)

    for label in ax.get_yticklabels():
        label.set_fontsize(16)
        label.set_fontname(FONT_ENGLISH)

    style_y_grid(ax)
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
  # 新版：使用 summary files + experiments-root
  python utils/plot_aggregated_pair_m1.py \\
      --summary-files $(ls results_ep_split10_t1te1v1_u7_disjoint/*_CE_summary.txt) \\
      --experiments-root ep_split10_t1te1v1_u7_disjoint --f1_only

  # 舊版：使用 error_rate.txt 檔案
  python utils/plot_aggregated_pair_m1.py --files results/file1.txt results/file2.txt

  # 舊版：直接指定目錄
  python utils/plot_aggregated_pair_m1.py \\
      --dirs prompt_ECPE_..._seed20 prompt_ECPE_..._seed42 prompt_ECPE_..._seed60 \\
      --dir_labels "UECA-Prompt few-shot"
        """
    )
    # ── 新版：summary-files 模式 ──
    parser.add_argument('--summary-files', type=str, nargs='+', default=[],
                        help='一或多個 summary txt 路徑 (multi-seed summary)')
    parser.add_argument('--experiments-root', type=str, default=None,
                        help='實驗根目錄，例如 ep_split10_t1te1v1_u7_disjoint（相容舊參數）')
    parser.add_argument('--experiments-roots', type=str, nargs='+', default=[],
                        help='可同時指定多個實驗根目錄，腳本會依序查找實驗資料夾')
    # ── 舊版：files / dirs 模式 ──
    parser.add_argument('--files', type=str, nargs='+', default=[],
                        help='彙整錯誤率檔案路徑列表 (用於取得實驗目錄)')
    parser.add_argument('--labels', type=str, nargs='+', default=None,
                        help='對應 --files 或 --summary-files 的標籤名稱（不指定則自動擷取）')
    parser.add_argument('--dirs', type=str, nargs='+', action='append', default=[],
                        help='直接指定實驗目錄路徑列表（可多次使用，每組目錄對應一個標籤）')
    parser.add_argument('--dir_labels', type=str, nargs='+', default=[],
                        help='對應 --dirs 的標籤名稱')
    parser.add_argument('--output', type=str, default=None,
                        help='輸出圖片檔名 (預設: png/pair_m1_multi_seed_comparison.png)')
    parser.add_argument('--f1_only', action='store_true',
                        help='只顯示 F1 分數（不顯示 Precision 和 Recall）')
    parser.add_argument('--no-std-bar', action='store_true', default=False,
                        help='不顯示標準差 error bar')
    parser.add_argument('--highlight-label', type=str, default=None,
                        help='指定要覆蓋顏色的標籤名稱（搭配 --highlight-color）')
    parser.add_argument('--highlight-color', type=str, default=None,
                        help='指定覆蓋顏色（HEX，例如 #FF1493）')
    parser.add_argument('--sort-by-value', action='store_true',
                        help='依 F1 平均值由小到大排序長條圖')
    parser.add_argument('--metric-only', type=str, default=None,
                        choices=['precision', 'recall', 'f1'],
                        help='只繪製單一指標長條圖: precision / recall / f1')
    
    args = parser.parse_args()
    
    all_data = []
    valid_labels = []

    # 統一整理可用的根目錄列表（優先使用新參數 --experiments-roots）
    root_candidates = []
    if args.experiments_roots:
        root_candidates.extend(args.experiments_roots)
    if args.experiments_root and args.experiments_root not in root_candidates:
        root_candidates.append(args.experiments_root)

    # ── 新版：--summary-files + --experiments-root ──
    if args.summary_files:
        if not root_candidates:
            print("錯誤：使用 --summary-files 時必須指定 --experiments-root 或 --experiments-roots")
            return
        invalid_roots = [r for r in root_candidates if not os.path.isdir(r)]
        valid_roots = [r for r in root_candidates if os.path.isdir(r)]
        if invalid_roots:
            for r in invalid_roots:
                print(f"⚠ 找不到 experiments-root: {r}")
        if not valid_roots:
            print("錯誤：沒有可用的 experiments-root")
            return
        print(f"可用實驗根目錄: {valid_roots}")

        # 自動產生 labels
        if args.labels is not None:
            if len(args.labels) != len(args.summary_files):
                print(f"錯誤：--labels ({len(args.labels)}) 與 --summary-files ({len(args.summary_files)}) 數量不一致")
                return
            labels = args.labels
        else:
            labels = [get_short_label(os.path.basename(sf)) for sf in args.summary_files]
            print(f"自動產生標籤: {labels}")

        print(f"\n讀取 {len(args.summary_files)} 個 summary files...")
        for sf, label in zip(args.summary_files, labels):
            print(f"\n處理: {os.path.basename(sf)}  →  [{label}]")
            exp_names = read_experiment_names_from_summary(sf)
            if not exp_names:
                print(f"  ⚠ 解析不到實驗目錄")
                continue

            # 依序在多個根目錄中查找每個實驗資料夾
            exp_dirs = []
            missing = []
            for name in exp_names:
                resolved = None
                for root in valid_roots:
                    candidate = os.path.join(root, name)
                    if os.path.isdir(candidate):
                        resolved = candidate
                        break
                if resolved is not None:
                    exp_dirs.append(resolved)
                else:
                    # 保留第一個根目錄下的候選路徑供提示
                    missing.append(os.path.join(valid_roots[0], name))

            if missing:
                for d in missing:
                    print(f"  ⚠ 找不到: {d}")
                fallback_metrics = parse_pair_m1_metrics_from_summary(sf)
                if fallback_metrics:
                    all_data.append(fallback_metrics)
                    valid_labels.append(label)
                    print("  ✓ 改用 summary 檔內統計值 (Pair m1 Mean ± Std)")
                    print(f"    P:  {fallback_metrics['P_mean']:.2f}% ± {fallback_metrics['P_std']:.2f}%")
                    print(f"    R:  {fallback_metrics['R_mean']:.2f}% ± {fallback_metrics['R_std']:.2f}%")
                    print(f"    F1: {fallback_metrics['F1_mean']:.2f}% ± {fallback_metrics['F1_std']:.2f}%")
                continue

            metrics = aggregate_pair_m1_from_dirs(exp_dirs)
            if metrics:
                all_data.append(metrics)
                valid_labels.append(label)
                print(f"  ✓ 成功讀取 {metrics['n_seeds']} 個 seed")
                print(f"    P:  {metrics['P_mean']:.2f}% ± {metrics['P_std']:.2f}%")
                print(f"    R:  {metrics['R_mean']:.2f}% ± {metrics['R_std']:.2f}%")
                print(f"    F1: {metrics['F1_mean']:.2f}% ± {metrics['F1_std']:.2f}%")
            else:
                fallback_metrics = parse_pair_m1_metrics_from_summary(sf)
                if fallback_metrics:
                    all_data.append(fallback_metrics)
                    valid_labels.append(label)
                    print("  ✓ 改用 summary 檔內統計值 (Pair m1 Mean ± Std)")
                    print(f"    P:  {fallback_metrics['P_mean']:.2f}% ± {fallback_metrics['P_std']:.2f}%")
                    print(f"    R:  {fallback_metrics['R_mean']:.2f}% ± {fallback_metrics['R_std']:.2f}%")
                    print(f"    F1: {fallback_metrics['F1_mean']:.2f}% ± {fallback_metrics['F1_std']:.2f}%")
                else:
                    print(f"  ⚠ 無法讀取 Pair (m1) metrics")

    # ── 舊版：--files 參數（透過 error_rate.txt）──
    if args.files:
        if args.labels is None:
            args.labels = []
            for f in args.files:
                name = os.path.basename(f).replace('.txt', '').replace('_error_rate', '')
                args.labels.append(name)
        
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
    
    # ── 舊版：--dirs 參數（直接指定目錄）──
    if args.dirs:
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
    
    if not args.summary_files and not args.files and not args.dirs:
        print("錯誤：請至少指定 --summary-files、--files 或 --dirs 參數")
        return
    
    if all_data:
        if args.sort_by_value:
            sort_key = 'F1_mean'
            if args.metric_only == 'precision':
                sort_key = 'P_mean'
            elif args.metric_only == 'recall':
                sort_key = 'R_mean'
            paired = list(zip(valid_labels, all_data))
            paired.sort(key=lambda x: x[1][sort_key])
            valid_labels = [p[0] for p in paired]
            all_data = [p[1] for p in paired]

        # 決定輸出路徑
        output_path = args.output
        if output_path is None:
            os.makedirs('png', exist_ok=True)
            output_path = 'png/pair_m1_multi_seed_comparison.png'

        if args.metric_only is not None:
            plot_metric_only(
                all_data,
                valid_labels,
                metric=args.metric_only,
                output_path=output_path,
                highlight_label=args.highlight_label,
                highlight_color=args.highlight_color,
            )
        elif args.f1_only:
            plot_f1_only(
                all_data,
                valid_labels,
                output_path,
                highlight_label=args.highlight_label,
                highlight_color=args.highlight_color,
            )
        else:
            plot_aggregated_pair_m1(all_data, valid_labels, output_path)
    else:
        print("\n錯誤: 沒有有效的數據可繪製")


if __name__ == '__main__':
    main()
