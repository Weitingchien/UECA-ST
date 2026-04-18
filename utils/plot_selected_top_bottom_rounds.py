#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
繪製 Self-Training 過程中，被選中文檔依排序前 K 筆 (Top-K) 與後 K 筆 (Bottom-K)
在各 round 的「文檔全對率」折線比較圖。
用途：觀察排序指標 (probability 或 divergence) 是否有效區分高品質與低品質的偽標籤文檔。
"""

from __future__ import annotations  # 允許在型別提示中使用 X | Y 語法，不需 Python 3.10+

import argparse   # 命令列參數解析模組
import json        # JSON 讀寫模組
from pathlib import Path  # 物件導向的路徑操作模組

import matplotlib.pyplot as plt  # Matplotlib 繪圖主模組
import numpy as np               # NumPy 數值計算模組

# ── 全域字型與背景設定（與 pair_m1 圖一致的風格） ──
# font.family：matplotlib 嘗試字型的優先順序，第一個找到的會被使用
plt.rcParams['font.family'] = ['Calisto MT', 'DFKai-SB', 'DejaVu Serif', 'Noto Serif CJK JP']
# 避免負號被 unicode_minus 渲染成方塊
plt.rcParams['axes.unicode_minus'] = False
FONT_CHINESE = 'DFKai-SB'    # 中文標籤使用的字型（標楷體）
FONT_ENGLISH = 'Calisto MT'  # 英文標籤與數字使用的字型


def parse_args() -> argparse.Namespace:
    """解析命令列參數，回傳 Namespace 物件。"""
    # 建立 ArgumentParser，description 會在 --help 時顯示
    parser = argparse.ArgumentParser(description='繪製前/後K筆在各 round 的文檔準確率差異折線圖')
    # --experiment-dir：指定實驗資料夾（必填），腳本會在其下尋找 pseudo_results_* 子目錄
    parser.add_argument('--experiment-dir', required=True, help='實驗資料夾或 pseudo_results_* 資料夾')
    # --fold：單一 fold 模式下，指定要分析的 fold 編號
    parser.add_argument('--fold', type=int, default=1, help='fold 編號（預設 1）')
    # --fold-start / --fold-end：跨 fold 平均模式下的 fold 範圍
    parser.add_argument('--fold-start', type=int, default=1, help='平均模式的起始 fold（預設 1）')
    parser.add_argument('--fold-end', type=int, default=10, help='平均模式的結束 fold（預設 10）')
    # --average-over-folds：加此旗標則啟用跨 fold 平均模式（action='store_true' 表示出現就設為 True）
    parser.add_argument('--average-over-folds', action='store_true', help='是否以 fold-start~fold-end 做每 round 平均')
    # --round-start / --round-end：要分析的 self-training round 範圍
    parser.add_argument('--round-start', type=int, default=1, help='起始 round（預設 1）')
    parser.add_argument('--round-end', type=int, default=10, help='結束 round（預設 10）')
    # --top-k：排序後取前/後各 K 筆文檔來比較
    parser.add_argument('--top-k', type=int, default=100, help='前/後各取幾筆（預設 100）')
    # --sort-by：排序依據，choices 限定只能輸入這兩個值之一
    parser.add_argument('--sort-by', choices=['probability', 'divergence'], default='probability', help='排序依據')
    # --output-dir：圖片和報告的輸出資料夾
    parser.add_argument('--output-dir', default='png', help='輸出資料夾（預設 png）')
    # --output-name：自訂輸出檔名（不含副檔名），None 表示使用預設命名
    parser.add_argument('--output-name', default=None, help='輸出檔名（不含副檔名）')
    # 解析命令列字串並回傳 Namespace 物件
    return parser.parse_args()


def resolve_pseudo_dir(experiment_dir: Path) -> Path:
    """從實驗目錄中找到 pseudo_results_* 子目錄
    若傳入的路徑本身就是 pseudo_results_ 開頭的資料夾，直接回傳
    """
    # 如果使用者直接指定了 pseudo_results_* 資料夾，不需要再搜尋
    if experiment_dir.name.startswith('pseudo_results_') and experiment_dir.is_dir():
        return experiment_dir
    # 用 glob 搜尋所有 pseudo_results_* 子目錄，排序後取最後一個（最新的時間戳）
    candidates = sorted([p for p in experiment_dir.glob('pseudo_results_*') if p.is_dir()])
    # 找不到任何候選目錄則報錯
    if not candidates:
        raise FileNotFoundError(f'找不到 pseudo_results_* 目錄: {experiment_dir}')
    # 回傳排序後最後一個（通常是時間戳最大 = 最新的）
    return candidates[-1]


def sort_rows(rows: list[dict], sort_by: str) -> list[dict]:
    """依指定欄位排序文檔列表。
    - probability 模式：高 probability 排前面（加負號做降序），同分則按 divergence 升序、doc_id 升序
    - divergence 模式：低 divergence 排前面（升序），同分則按 probability 降序、doc_id 升序
    排序後 rows[0] 是「最好」的文檔，rows[-1] 是「最差」的。
    """
    if sort_by == 'probability':
        # key 回傳 tuple：sorted 會依序比較 tuple 中每個元素
        # -probability 表示降序（值越大越前面）
        return sorted(rows, key=lambda x: (-x['probability'], x['divergence'], int(x['doc_id'])))
    # divergence 模式：divergence 越小越好，排前面
    return sorted(rows, key=lambda x: (x['divergence'], -x['probability'], int(x['doc_id'])))


def collect_round_stats(pseudo_dir: Path, fold: int, round_idx: int, top_k: int, sort_by: str) -> dict:
    """收集單一 fold、單一 round 的 Top-K / Bottom-K 文檔全對統計。

    讀取兩個 JSON 檔：
    1. nest_divergence_scores_fold{fold}_round{round}.json — 含每筆文檔的 probability、divergence、是否被選中
    2. pseudo_labeled_samples_fold{fold}_round{round}.json — 含偽標籤預測 ID 和 GT ID
    """
    # 組出兩個必要的 JSON 檔案路徑
    divergence_path = pseudo_dir / f'nest_divergence_scores_fold{fold}_round{round_idx}.json'
    pseudo_path = pseudo_dir / f'pseudo_labeled_samples_fold{fold}_round{round_idx}.json'

    # 任一檔案不存在就報錯（兩個檔案都是必需的）
    if not divergence_path.exists() or not pseudo_path.exists():
        raise FileNotFoundError(f'缺少 round{round_idx} 檔案: {divergence_path.name} 或 {pseudo_path.name}')

    # 讀取並解析 JSON 內容
    divergence_data = json.loads(divergence_path.read_text(encoding='utf-8'))
    pseudo_data = json.loads(pseudo_path.read_text(encoding='utf-8'))

    # 將 pseudo_data 建成以 doc_id 為 key 的字典，方便快速查找
    # 只保留有 ground_truth_available=True 的樣本（有 GT 才能計算正確率）
    pseudo_by_doc_id = {
        str(item['doc_id']): item  # key 統一轉為字串
        for item in pseudo_data
        if item.get('ground_truth_available', False)  # .get 提供預設值，避免 key 不存在時報錯
    }

    rows = []  # 用來收集所有「被選中且有 GT」的文檔資料
    # 遍歷 divergence_data 中的每個樣本
    for sample in divergence_data.get('samples', []):
        # 只處理被選中的樣本（selected=True），未選中的不納入分析
        if not sample.get('selected', False):
            continue
        # 取出 doc_id 並轉為字串，統一格式
        doc_id = str(sample.get('doc_id'))
        # 從 pseudo 字典中查找對應的偽標籤資料
        pseudo_item = pseudo_by_doc_id.get(doc_id)
        # 找不到表示此文檔可能被 consistency filter 拒絕，或無 GT，跳過即可
        if pseudo_item is None:
            continue

        # 取出模型的偽標籤預測 token ID 序列
        pred_ids = pseudo_item.get('pseudo_label_ids', [])
        # 取出該文檔的 Ground Truth token ID 序列
        gt_ids = pseudo_item.get('gt_label_ids', [])
        # 取兩者較短的長度（理論上應該一樣長，min 防止意外）
        n_tokens = min(len(pred_ids), len(gt_ids))
        # 逐 token 比對，計算正確的 token 數
        # 生成器表達式 (1 for i in range(...) if ...) 逐一產生 1，sum 加總
        n_correct = sum(1 for i in range(n_tokens) if pred_ids[i] == gt_ids[i])

        # 將此文檔的資訊加入 rows 列表
        rows.append({
            'doc_id': doc_id,
            'probability': float(sample.get('probability', 0.0)),  # 模型對此文檔的信心機率
            'divergence': float(sample.get('divergence_score', 0.0)),  # NeST divergence 分數
            'full_match': bool(n_correct == n_tokens),  # 所有 token 都對 = 文檔全對
        })

    # 若沒有任何可分析的樣本，報錯
    if not rows:
        raise RuntimeError(f'round{round_idx} 沒有可分析的 selected=True 且含 GT 樣本')

    # 依指定欄位排序（排序後 rows[0] 是「最好的」，rows[-1] 是「最差的」）
    rows = sort_rows(rows, sort_by)
    # 實際取用的 K 值（若樣本不足 top_k 則以實際數量為準）
    k = min(top_k, len(rows))
    # 取排序後前 K 筆（品質最好的一群）
    top_rows = rows[:k]
    # 取排序後後 K 筆（品質最差的一群）；負索引 rows[-k:] 表示從尾端向前取 k 筆
    bottom_rows = rows[-k:]

    # 計算 Top-K 中「全對」的文檔數
    top_full = sum(1 for r in top_rows if r['full_match'])
    # 計算 Bottom-K 中「全對」的文檔數
    bottom_full = sum(1 for r in bottom_rows if r['full_match'])

    # 轉成百分比（乘 100）
    top_rate = (top_full / k) * 100.0
    bottom_rate = (bottom_full / k) * 100.0

    # 回傳此 round 的統計結果（字典格式）
    return {
        'round': round_idx,            # 第幾輪 self-training
        'fold': fold,                   # 第幾個 fold
        'k': k,                         # 實際取用的 K 值
        'selected_total': len(rows),    # 此 round 被選中且有 GT 的總文檔數
        'top_full_docs': top_full,      # Top-K 全對文檔數（分子）
        'bottom_full_docs': bottom_full,  # Bottom-K 全對文檔數（分子）
        'top_rate': top_rate,           # Top-K 全對率 (%)
        'bottom_rate': bottom_rate,     # Bottom-K 全對率 (%)
        'gap_docs': top_full - bottom_full,      # 兩組全對文檔數的差距
        'gap_rate_pp': top_rate - bottom_rate,   # 兩組全對率的差距（百分點 percentage point）
    }


def aggregate_round_stats_by_folds(stats_one_round: list[dict], round_idx: int) -> dict:
    """將同一 round 下多個 fold 的統計做平均，產出跨 fold 的彙總結果。

    Args:
        stats_one_round: 同一 round、不同 fold 的統計結果列表
        round_idx: 此批統計對應的 round 編號
    """
    # 從每個 fold 的統計中提取各指標，組成 list 以便計算 mean/std
    top_rates = [s['top_rate'] for s in stats_one_round]           # 各 fold 的 Top-K 全對率
    bottom_rates = [s['bottom_rate'] for s in stats_one_round]     # 各 fold 的 Bottom-K 全對率
    gap_rates = [s['gap_rate_pp'] for s in stats_one_round]        # 各 fold 的兩組差距
    top_full_docs = [s['top_full_docs'] for s in stats_one_round]  # 各 fold 的 Top-K 全對文檔數
    bottom_full_docs = [s['bottom_full_docs'] for s in stats_one_round]  # 各 fold 的 Bottom-K 全對文檔數
    selected_totals = [s['selected_total'] for s in stats_one_round]     # 各 fold 的總被選文檔數
    k_vals = [s['k'] for s in stats_one_round]                     # 各 fold 的 K 值

    # 回傳各指標的跨 fold 平均值（np.mean 計算算術平均）
    return {
        'round': round_idx,                                                # round 編號
        'k': int(np.mean(k_vals)),                                         # K 的平均（取整數）
        'selected_total': float(np.mean(selected_totals)),                 # 被選文檔數平均
        'top_full_docs': float(np.mean(top_full_docs)),                    # Top-K 全對數平均
        'bottom_full_docs': float(np.mean(bottom_full_docs)),              # Bottom-K 全對數平均
        'top_rate': float(np.mean(top_rates)),                             # Top-K 全對率平均
        'bottom_rate': float(np.mean(bottom_rates)),                       # Bottom-K 全對率平均
        'gap_docs': float(np.mean(np.array(top_full_docs) - np.array(bottom_full_docs))),  # 文檔數差距平均
        'gap_rate_pp': float(np.mean(gap_rates)),                          # 率差距平均
        'top_rate_std': float(np.std(top_rates)),                          # Top-K 全對率的標準差（np.std）
        'bottom_rate_std': float(np.std(bottom_rates)),                    # Bottom-K 全對率的標準差
    }


def plot_round_lines(
    stats: list[dict],       # 每個 round 的統計結果列表
    top_k: int,              # 前/後各取幾筆
    output_path: Path,       # 輸出圖片的完整路徑
    title_text: str,         # 圖片標題文字
    show_counts: bool = True,  # 是否在折線點上標註分子/分母
) -> None:
    """繪製 Top-K vs Bottom-K 文檔全對率的折線比較圖。"""
    # 從統計列表中提取 x 軸（round 編號）和兩條線的 y 值
    rounds = [s['round'] for s in stats]           # x 軸：round 1, 2, ..., 10
    top_rates = [s['top_rate'] for s in stats]     # Top-K 的全對率
    bottom_rates = [s['bottom_rate'] for s in stats]  # Bottom-K 的全對率

    # 建立圖表：figsize=(寬, 高) 單位為英吋
    fig, ax = plt.subplots(figsize=(14, 8))
    # 設定圖表外框和繪圖區的背景色為淺灰色
    fig.patch.set_facecolor('#DDDDDD')  # 整個圖片的背景
    ax.set_facecolor('#DDDDDD')          # 繪圖區（軸內部）的背景

    # 繪製 Top-K 折線（藍色），marker='o' 在每個資料點畫圓形標記
    # ax.plot 回傳一個 Line2D 物件的列表，用 line_top, = 解包取出單一物件
    line_top, = ax.plot(rounds, top_rates, marker='o', linewidth=2.2, markersize=7,
                                color='#1f77b4', label=f'Top{top_k} 文檔準確率')
    # 繪製 Bottom-K 折線（紅色）
    line_bottom, = ax.plot(rounds, bottom_rates, marker='o', linewidth=2.2, markersize=7,
                                    color='#d62728', label=f'Bottom{top_k} 文檔準確率')

    # 設定圖表標題（使用中文字型）
    ax.set_title(title_text,
                 fontsize=24, fontweight='bold', fontname=FONT_CHINESE)
    # 設定 x 軸標籤（使用英文字型）
    ax.set_xlabel('Round', fontsize=22, fontname=FONT_ENGLISH)
    # 設定 y 軸標籤（使用中文字型）
    ax.set_ylabel('文檔準確率 (%)', fontsize=24, fontname=FONT_CHINESE)

    # 明確指定 x 軸刻度位置（每個 round 一個刻度）
    ax.set_xticks(rounds)
    # 設定 x 軸刻度標籤的字型大小
    ax.tick_params(axis='x', labelsize=16)
    # 設定 y 軸刻度標籤的字型大小
    ax.tick_params(axis='y', labelsize=18)

    # 將 x 軸和 y 軸的刻度標籤統一設為英文字型
    # get_xticklabels() + get_yticklabels() 取得所有 Text 物件，逐一設定字型
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontname(FONT_ENGLISH)

    # 開啟網格線：虛線樣式，透明度 0.7
    ax.grid(True, linestyle='--', alpha=0.7)

    # 每個 round 在 Top/Bottom 線上標註分子分母（單一 fold 時較直觀）
    if show_counts:
        # zip 同步迭代 rounds 和 stats，r 是 round 編號，s 是對應的統計字典
        for r, s in zip(rounds, stats):
            # 在 Top 折線點的上方 (+1.0) 放置文字標籤，顯示 "全對數/K"
            # ha='center' 水平置中，va='bottom' 文字底端對齊指定 y 座標
            ax.text(r, s['top_rate'] + 1.0, f"{s['top_full_docs']}/{s['k']}",
                    ha='center', va='bottom', fontsize=10, fontname=FONT_ENGLISH,
                    color='#1f77b4', fontweight='bold')
            # 在 Bottom 折線點的下方 (-1.0) 放置文字標籤
            # va='top' 文字頂端對齊指定 y 座標（所以文字在點的下方）
            ax.text(r, s['bottom_rate'] - 1.0, f"{s['bottom_full_docs']}/{s['k']}",
                    ha='center', va='top', fontsize=10, fontname=FONT_ENGLISH,
                    color='#d62728', fontweight='bold')

    # 建立圖例（legend）
    handles = [line_top, line_bottom]             # 圖例對應的線條物件
    labels = [h.get_label() for h in handles]     # 從每條線取出 label 文字
    # loc='best' 讓 matplotlib 自動選擇不遮擋資料的最佳位置
    legend = ax.legend(handles, labels, fontsize=15, loc='best')
    # 圖例背景也設成與圖表一致的淺灰色
    legend.get_frame().set_facecolor('#DDDDDD')
    # 依據圖例文字是否含中文字元，動態選用中文或英文字型
    for text in legend.get_texts():
        # 檢查文字中是否有 CJK 統一表意文字範圍 (\u4e00~\u9fff)
        has_cjk = any('\u4e00' <= ch <= '\u9fff' for ch in text.get_text())
        text.set_fontname(FONT_CHINESE if has_cjk else FONT_ENGLISH)

    # 調整子圖邊距：左 9%、下 12%、右 91%、上 92%（比例值，0~1）
    fig.subplots_adjust(left=0.09, bottom=0.12, right=0.91, top=0.92)
    # 儲存圖片：dpi=150 解析度，bbox_inches='tight' 自動裁切空白邊，pad_inches 額外留白
    fig.savefig(str(output_path), dpi=150, bbox_inches='tight', pad_inches=0.3)
    # 關閉 figure 釋放記憶體（避免大量圖表時記憶體洩漏）
    plt.close(fig)
    print(f'圖片已儲存: {output_path}')


def write_report(stats: list[dict], output_path: Path) -> None:
    """將各 round 的統計結果寫成 TSV 格式的文字報告檔。"""
    lines = []  # 用 list 收集所有行，最後一次 join 寫入（比逐行寫入高效）
    lines.append('=' * 90)                          # 分隔線（90 個等號）
    lines.append('各 Round Top/Bottom 差異統計')     # 報告標題
    lines.append('=' * 90)
    # 表頭：各欄位名稱以 tab 分隔
    lines.append('round\tselected_total\tk\ttop_full\tbottom_full\ttop_rate\tbottom_rate\tgap_docs\tgap_rate_pp')

    # 逐 round 輸出一行資料
    for s in stats:
        # f-string 中 :.2f 表示保留兩位小數
        lines.append(
            f"{s['round']}\t{s['selected_total']}\t{s['k']}\t{s['top_full_docs']}\t{s['bottom_full_docs']}\t"
            f"{s['top_rate']:.2f}%\t{s['bottom_rate']:.2f}%\t{s['gap_docs']}\t{s['gap_rate_pp']:.2f}"
        )

    lines.append('=' * 90)  # 結尾分隔線
    # '\n'.join(lines) 將所有行用換行符合併成一個字串，一次寫入檔案
    output_path.write_text('\n'.join(lines), encoding='utf-8')
    print(f'報告已儲存: {output_path}')


def main() -> None:
    """主程式進入點：解析參數 → 收集統計 → 繪圖 → 輸出報告"""
    # 解析命令列參數
    args = parse_args()
    # 從實驗目錄中定位 pseudo_results_* 子目錄
    pseudo_dir = resolve_pseudo_dir(Path(args.experiment_dir))

    # 參數合理性檢查
    if args.round_start > args.round_end:
        raise ValueError('--round-start 不可大於 --round-end')

    if args.fold_start > args.fold_end:
        raise ValueError('--fold-start 不可大於 --fold-end')

    stats = []  # 儲存每個 round 的統計結果
    if args.average_over_folds:
        # ── 跨 fold 平均模式 ──
        # 建立 fold 編號列表，例如 [1, 2, 3, ..., 10]
        folds = list(range(args.fold_start, args.fold_end + 1))
        # 逐 round 收集所有 fold 的統計再做平均
        for round_idx in range(args.round_start, args.round_end + 1):
            one_round_all_folds = []  # 暫存同一 round 下各 fold 的結果
            for fold in folds:
                # 收集單一 fold、單一 round 的統計
                one = collect_round_stats(
                    pseudo_dir=pseudo_dir,
                    fold=fold,
                    round_idx=round_idx,
                    top_k=args.top_k,
                    sort_by=args.sort_by,
                )
                one_round_all_folds.append(one)
            # 將此 round 的多個 fold 結果做平均，產出一筆彙總資料
            stats.append(aggregate_round_stats_by_folds(one_round_all_folds, round_idx))
    else:
        # ── 單一 fold 模式 ──
        # 逐 round 收集指定 fold 的統計
        for round_idx in range(args.round_start, args.round_end + 1):
            one = collect_round_stats(
                pseudo_dir=pseudo_dir,
                fold=args.fold,       # 使用 --fold 指定的單一 fold
                round_idx=round_idx,
                top_k=args.top_k,
                sort_by=args.sort_by,
            )
            stats.append(one)

    # 建立輸出資料夾 (exist_ok=True 表示已存在不報錯)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # 組出預設輸出檔名 (依模式不同有不同格式)
    if args.average_over_folds:
        # 跨 fold 平均模式的檔名包含 fold 範圍和 round 範圍
        default_name = (
            f'selected_top_bottom_rounds_foldavg_f{args.fold_start}-{args.fold_end}'
            f'_r{args.round_start}-{args.round_end}'
        )
    else:
        # 單一 fold 模式的檔名包含 fold 編號和 round 範圍
        default_name = f'selected_top_bottom_rounds_fold{args.fold}_r{args.round_start}-{args.round_end}'
    # 若使用者有指定 --output-name 則用之，否則用預設名稱（or 運算子：左邊為 None 時取右邊）
    base_name = args.output_name or default_name

    # 組出圖表標題文字
    if args.average_over_folds:
        title_text = (
            f'Fold 平均({args.fold_start}-{args.fold_end}) 各 Round 文檔準確率：'
            f'Top{args.top_k} vs Bottom{args.top_k}'
        )
    else:
        title_text = f'Fold {args.fold} 各 Round 文檔準確率：Top{args.top_k} vs Bottom{args.top_k}'

    # 繪製折線圖並儲存為 PNG
    plot_round_lines(
        stats=stats,
        top_k=args.top_k,
        output_path=output_dir / f'{base_name}.png',  # Path / 字串 = 路徑拼接
        title_text=title_text,
        show_counts=(not args.average_over_folds),  # 單一 fold 時顯示分子分母，平均模式不顯示
    )
    # 輸出文字報告（TSV 格式）
    write_report(stats=stats, output_path=output_dir / f'{base_name}.txt')


# 當此檔案被直接執行（而非被 import）時，才執行 main()
if __name__ == '__main__':
    main()
