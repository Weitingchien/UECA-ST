#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
比較多個實驗群組的 Top-K vs Bottom-K 文檔準確率折線圖。

每個實驗群組由一個 summary.txt 檔指定（內含多個 seed 的實驗目錄名稱），
腳本會自動解析 summary 取得所有 seed 目錄，跨 seed × 跨 fold 平均後，
繪製各群組的 Top-K 與 Bottom-K 文檔準確率折線圖。

用法範例：
    python utils/plot_selected_top_bottom_comparison.py \
        --summary-files \
            results_ep_split10_t1te1v1_u7_disjoint/UECA_Prompt_..._consistency_CE_summary.txt \
            results_ep_split10_t1te1v1_u7_disjoint/UECA_Prompt_..._CE_summary.txt \
        --labels "NeST×7+Consistency" "NeST×5" \
        --search-paths . /mnt/f/experiments \
        --top-k 500 \
        --output-name comparison_top_bottom_500
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import orjson as _json_mod
    def _load_json(path: Path):
        return _json_mod.loads(path.read_bytes())
except ImportError:
    import json as _json_mod
    def _load_json(path: Path):
        return _json_mod.loads(path.read_text(encoding='utf-8'))

import matplotlib.pyplot as plt
import numpy as np

# ── 全域字型與背景設定（與其他繪圖腳本一致） ──
plt.rcParams['font.family'] = ['Calisto MT', 'DFKai-SB', 'DejaVu Serif', 'Noto Serif CJK JP']
plt.rcParams['axes.unicode_minus'] = False
FONT_CHINESE = 'DFKai-SB'
FONT_ENGLISH = 'Calisto MT'

# 每個實驗群組的顏色（最多 6 組）
GROUP_COLORS = [
    '#1f77b4',  # 藍
    '#2ca02c',  # 綠
    '#d62728',  # 紅
    '#ff7f0e',  # 橙
    '#9467bd',  # 紫
    '#8c564b',  # 棕
]


# ──────────────────────────────────────────────────
# 命令列參數
# ──────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='比較多個實驗群組的 Top-K/Bottom-K 文檔準確率折線圖',
    )
    parser.add_argument('--summary-files', nargs='+', required=True,
                        help='各實驗群組的 summary.txt 檔案路徑')
    parser.add_argument('--labels', nargs='+', required=True,
                        help='各實驗群組的圖例標籤')
    parser.add_argument('--search-paths', nargs='+', default=['.'],
                        help='搜尋實驗目錄的基礎路徑（預設 ["."]）')
    parser.add_argument('--top-k', type=int, default=500,
                        help='前/後各取幾筆（預設 500）')
    parser.add_argument('--sort-by', choices=['probability', 'divergence'],
                        default='probability', help='排序依據（預設 probability）')
    parser.add_argument('--fold-start', type=int, default=1)
    parser.add_argument('--fold-end', type=int, default=10)
    parser.add_argument('--round-start', type=int, default=1)
    parser.add_argument('--round-end', type=int, default=10)
    parser.add_argument('--show-counts', action='store_true',
                        help='在折線點上標註平均分子/分母')
    parser.add_argument('--align-k', action='store_true',
                        help='跨群組對齊 K：每個 round 使用所有群組中最小可用樣本數作為統一 K')
    parser.add_argument('--output-dir', default='png')
    parser.add_argument('--output-name', default='comparison_top_bottom')
    args = parser.parse_args()

    if len(args.summary_files) != len(args.labels):
        parser.error('--summary-files 和 --labels 的數量必須相同')

    return args


# ──────────────────────────────────────────────────
# Summary 解析 & 目錄搜尋
# ──────────────────────────────────────────────────
def parse_summary_for_exp_dirs(summary_path: Path) -> tuple[list[str], str]:
    """從 summary.txt 中提取實驗目錄名稱。

    Returns:
        (實驗目錄名稱列表, 推測的 base_dir 如 'ep_split10_..._disjoint')
    """
    text = summary_path.read_text(encoding='utf-8')
    exp_dirs: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith('- prompt_ECPE_'):
            exp_dirs.append(stripped.lstrip('- ').strip())

    # 從 summary 所在資料夾名稱推測 base_dir
    # 例如 results_ep_split10_t1te1v1_u7_disjoint → ep_split10_t1te1v1_u7_disjoint
    parent_name = summary_path.parent.name
    base_dir = parent_name.replace('results_', '', 1) if parent_name.startswith('results_') else ''
    return exp_dirs, base_dir


def find_experiment_dir(exp_name: str, base_dir: str, search_paths: list[str]) -> Path | None:
    """在 search_paths 底下搜尋 base_dir/exp_name 目錄。"""
    for sp in search_paths:
        sp_path = Path(sp)
        # 嘗試 search_path / base_dir / exp_name
        if base_dir:
            candidate = sp_path / base_dir / exp_name
            if candidate.is_dir():
                return candidate
        # 嘗試 search_path / exp_name
        candidate = sp_path / exp_name
        if candidate.is_dir():
            return candidate
    return None


# ──────────────────────────────────────────────────
# 資料收集（重用原始腳本邏輯，修正 consistency 相容性）
# ──────────────────────────────────────────────────
def resolve_pseudo_dir(experiment_dir: Path) -> Path:
    """在實驗目錄中找到 pseudo_results_* 子目錄。"""
    if experiment_dir.name.startswith('pseudo_results_') and experiment_dir.is_dir():
        return experiment_dir
    candidates = sorted([p for p in experiment_dir.glob('pseudo_results_*') if p.is_dir()])
    if not candidates:
        raise FileNotFoundError(f'找不到 pseudo_results_* 目錄: {experiment_dir}')
    return candidates[-1]


def sort_rows(rows: list[dict], sort_by: str) -> list[dict]:
    """依指定欄位排序文檔列表。"""
    if sort_by == 'probability':
        return sorted(rows, key=lambda x: (-x['probability'], x['divergence'], int(x['doc_id'])))
    return sorted(rows, key=lambda x: (x['divergence'], -x['probability'], int(x['doc_id'])))


def collect_round_rows(
    pseudo_dir: Path, fold: int, round_idx: int, sort_by: str,
) -> list[dict] | None:
    """收集單一 fold、單一 round 的已排序文檔列表。
    檔案不存在或無合格樣本時回傳 None。
    """
    divergence_path = pseudo_dir / f'nest_divergence_scores_fold{fold}_round{round_idx}.json'
    pseudo_path = pseudo_dir / f'pseudo_labeled_samples_fold{fold}_round{round_idx}.json'

    if not divergence_path.exists() or not pseudo_path.exists():
        return None

    divergence_data = _load_json(divergence_path)
    pseudo_data = _load_json(pseudo_path)

    pseudo_by_doc_id = {
        str(item['doc_id']): item
        for item in pseudo_data
        if item.get('ground_truth_available', False)
    }

    rows: list[dict] = []
    for sample in divergence_data.get('samples', []):
        if not sample.get('selected', False):
            continue
        doc_id = str(sample.get('doc_id'))
        pseudo_item = pseudo_by_doc_id.get(doc_id)
        if pseudo_item is None:
            continue

        pred_ids = pseudo_item.get('pseudo_label_ids', [])
        gt_ids = pseudo_item.get('gt_label_ids', [])
        n_tokens = min(len(pred_ids), len(gt_ids))
        n_correct = sum(1 for i in range(n_tokens) if pred_ids[i] == gt_ids[i])

        rows.append({
            'doc_id': doc_id,
            'probability': float(sample.get('probability', 0.0)),
            'divergence': float(sample.get('divergence_score', 0.0)),
            'full_match': bool(n_correct == n_tokens),
        })

    if not rows:
        return None
    return sort_rows(rows, sort_by)


def compute_topk_stats(rows: list[dict], k: int) -> dict:
    """從已排序的 rows 中，取前 k / 後 k 筆計算全對統計。"""
    actual_k = min(k, len(rows))
    top_rows = rows[:actual_k]
    bottom_rows = rows[-actual_k:]
    top_full = sum(1 for r in top_rows if r['full_match'])
    bottom_full = sum(1 for r in bottom_rows if r['full_match'])
    return {
        'k': actual_k,
        'top_full_docs': top_full,
        'bottom_full_docs': bottom_full,
        'top_rate': (top_full / actual_k) * 100.0,
        'bottom_rate': (bottom_full / actual_k) * 100.0,
    }


def collect_round_stats(
    pseudo_dir: Path, fold: int, round_idx: int, top_k: int, sort_by: str,
) -> dict | None:
    """收集單一 fold、單一 round 的 Top-K / Bottom-K 統計。"""
    rows = collect_round_rows(pseudo_dir, fold, round_idx, sort_by)
    if rows is None:
        return None
    stats = compute_topk_stats(rows, top_k)
    stats['round'] = round_idx
    stats['fold'] = fold
    stats['selected_total'] = len(rows)
    return stats


def collect_group_rows(
    exp_dirs: list[Path],
    fold_start: int, fold_end: int,
    round_start: int, round_end: int,
    sort_by: str,
) -> dict[int, list[list[dict]]]:
    """收集一個實驗群組所有 round 的排序後 rows。

    Returns:
        {round_idx: [sorted_rows_1, sorted_rows_2, ...]}
        每個 sorted_rows 是一個 seed×fold 組合的文檔列表。
    """
    result: dict[int, list[list[dict]]] = {}
    for round_idx in range(round_start, round_end + 1):
        round_rows: list[list[dict]] = []
        for exp_dir in exp_dirs:
            pseudo_dir = resolve_pseudo_dir(exp_dir)
            for fold in range(fold_start, fold_end + 1):
                rows = collect_round_rows(pseudo_dir, fold, round_idx, sort_by)
                if rows is not None:
                    round_rows.append(rows)
        result[round_idx] = round_rows
        print(f'  round {round_idx}/{round_end} 完成 ({len(round_rows)} 個資料點)', flush=True)
    return result


def summarize_group_stats(
    group_rows: dict[int, list[list[dict]]],
    top_k: int,
) -> list[dict]:
    """從已收集的 rows，用指定 K 計算跨 seed×fold 平均統計。"""
    result: list[dict] = []
    for round_idx in sorted(group_rows.keys()):
        rows_list = group_rows[round_idx]
        if not rows_list:
            continue
        top_rates, bottom_rates = [], []
        top_fulls, bottom_fulls, k_vals = [], [], []
        for rows in rows_list:
            s = compute_topk_stats(rows, top_k)
            top_rates.append(s['top_rate'])
            bottom_rates.append(s['bottom_rate'])
            top_fulls.append(s['top_full_docs'])
            bottom_fulls.append(s['bottom_full_docs'])
            k_vals.append(s['k'])
        result.append({
            'round': round_idx,
            'top_rate': float(np.mean(top_rates)),
            'bottom_rate': float(np.mean(bottom_rates)),
            'top_rate_std': float(np.std(top_rates)),
            'bottom_rate_std': float(np.std(bottom_rates)),
            'n_samples': len(top_rates),
            'avg_top_full': float(np.mean(top_fulls)),
            'avg_bottom_full': float(np.mean(bottom_fulls)),
            'avg_k': float(np.mean(k_vals)),
        })
    return result


# ──────────────────────────────────────────────────
# 繪圖
# ──────────────────────────────────────────────────
def plot_comparison(
    all_group_stats: list[list[dict]],
    labels: list[str],
    top_k: int,
    output_path: Path,
    title_text: str,
    show_counts: bool = False,
) -> None:
    """繪製多組實驗的 Top-K vs Bottom-K 折線比較圖。"""
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor('#DDDDDD')
    ax.set_facecolor('#DDDDDD')

    handles = []
    legend_labels = []

    for i, (stats, label) in enumerate(zip(all_group_stats, labels)):
        color = GROUP_COLORS[i % len(GROUP_COLORS)]
        rounds = [s['round'] for s in stats]
        top_rates = [s['top_rate'] for s in stats]
        bottom_rates = [s['bottom_rate'] for s in stats]

        # Top 折線：實線 + 圓形標記
        line_top, = ax.plot(
            rounds, top_rates,
            marker='o', linewidth=2.2, markersize=7,
            color=color, linestyle='-',
        )
        # Bottom 折線：虛線 + ▽ 標記
        line_bottom, = ax.plot(
            rounds, bottom_rates,
            marker='v', linewidth=2.2, markersize=7,
            color=color, linestyle='--',
        )

        # 在折線點上標註平均分子/分母
        if show_counts:
            for s in stats:
                r = s['round']
                avg_k = s['avg_k']
                # Top 標籤
                avg_top = s['avg_top_full']
                ax.text(r, s['top_rate'] + 0.8,
                        f'{avg_top:.0f}/{avg_k:.0f}',
                        ha='center', va='bottom', fontsize=7,
                        fontname=FONT_ENGLISH, color=color, fontweight='bold')
                # Bottom 標籤
                avg_bot = s['avg_bottom_full']
                ax.text(r, s['bottom_rate'] - 0.8,
                        f'{avg_bot:.0f}/{avg_k:.0f}',
                        ha='center', va='top', fontsize=7,
                        fontname=FONT_ENGLISH, color=color, fontweight='bold')

        handles.extend([line_top, line_bottom])
        legend_labels.extend([
            f'{label} Top{top_k}',
            f'{label} Bottom{top_k}',
        ])

    ax.set_title(title_text, fontsize=22, fontweight='bold', fontname=FONT_CHINESE)
    ax.set_xlabel('Round', fontsize=22, fontname=FONT_ENGLISH)
    ax.set_ylabel('文檔準確率 (%)', fontsize=24, fontname=FONT_CHINESE)

    if all_group_stats:
        ax.set_xticks([s['round'] for s in all_group_stats[0]])
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=18)
    for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
        tick_label.set_fontname(FONT_ENGLISH)

    ax.grid(True, linestyle='--', alpha=0.7)

    legend = ax.legend(handles, legend_labels, fontsize=13, loc='best')
    legend.get_frame().set_facecolor('#DDDDDD')
    for text in legend.get_texts():
        has_cjk = any('\u4e00' <= ch <= '\u9fff' for ch in text.get_text())
        text.set_fontname(FONT_CHINESE if has_cjk else FONT_ENGLISH)

    fig.subplots_adjust(left=0.09, bottom=0.12, right=0.91, top=0.92)
    fig.savefig(str(output_path), dpi=150, bbox_inches='tight', pad_inches=0.3)
    plt.close(fig)
    print(f'圖片已儲存: {output_path}')


# ──────────────────────────────────────────────────
# 文字報告
# ──────────────────────────────────────────────────
def write_report(
    all_group_stats: list[list[dict]],
    labels: list[str],
    top_k: int,
    output_path: Path,
) -> None:
    """輸出 TSV 格式的比較報告。"""
    lines: list[str] = []
    lines.append('=' * 100)
    lines.append(f'比較報告：Top{top_k} vs Bottom{top_k} 各 Round 文檔準確率')
    lines.append('=' * 100)

    for stats, label in zip(all_group_stats, labels):
        lines.append('')
        lines.append(f'--- {label} (跨 seed×fold 平均, 每 round {stats[0]["n_samples"] if stats else "?"} 個資料點) ---')
        lines.append('round\tn_samples\ttop_rate\tbottom_rate\tgap_pp\ttop_std\tbottom_std')
        for s in stats:
            gap = s['top_rate'] - s['bottom_rate']
            lines.append(
                f"{s['round']}\t{s['n_samples']}\t{s['top_rate']:.2f}%\t{s['bottom_rate']:.2f}%\t"
                f"{gap:.2f}\t{s.get('top_rate_std', 0):.2f}\t{s.get('bottom_rate_std', 0):.2f}"
            )

    lines.append('')
    lines.append('=' * 100)
    output_path.write_text('\n'.join(lines), encoding='utf-8')
    print(f'報告已儲存: {output_path}')


# ──────────────────────────────────────────────────
# 主程式
# ──────────────────────────────────────────────────
def main() -> None:
    args = parse_args()

    # Phase 1: 找到所有實驗目錄
    all_exp_dirs: list[list[Path]] = []
    for summary_file, label in zip(args.summary_files, args.labels):
        summary_path = Path(summary_file)
        if not summary_path.exists():
            print(f'[ERROR] Summary 檔案不存在: {summary_path}', file=sys.stderr)
            sys.exit(1)

        exp_names, base_dir = parse_summary_for_exp_dirs(summary_path)
        if not exp_names:
            print(f'[ERROR] 無法從 {summary_path} 解析出實驗目錄', file=sys.stderr)
            sys.exit(1)

        print(f'\n[{label}] 找到 {len(exp_names)} 個 seed 目錄:')

        exp_dirs: list[Path] = []
        for name in exp_names:
            found = find_experiment_dir(name, base_dir, args.search_paths)
            if found:
                print(f'  ✓ {found}')
                exp_dirs.append(found)
            else:
                print(f'  ✗ 找不到: {name}')

        if not exp_dirs:
            print(f'[ERROR] [{label}] 沒有找到任何實驗目錄', file=sys.stderr)
            sys.exit(1)
        all_exp_dirs.append(exp_dirs)

    # Phase 2: 收集所有群組的 rows
    all_group_rows: list[dict[int, list[list[dict]]]] = []
    for exp_dirs, label in zip(all_exp_dirs, args.labels):
        print(f'\n收集 [{label}] 的資料...')
        group_rows = collect_group_rows(
            exp_dirs,
            args.fold_start, args.fold_end,
            args.round_start, args.round_end,
            args.sort_by,
        )
        all_group_rows.append(group_rows)

    # Phase 3: 決定每個 round 的 K 值
    top_k = args.top_k
    if args.align_k:
        # 每個 round 取所有群組中「最小可用樣本數的最小值」，再和 top_k 取 min
        all_rounds = sorted(set().union(*(gr.keys() for gr in all_group_rows)))
        aligned_k_per_round: dict[int, int] = {}
        for round_idx in all_rounds:
            min_available = top_k
            for group_rows in all_group_rows:
                if round_idx in group_rows and group_rows[round_idx]:
                    group_min = min(len(rows) for rows in group_rows[round_idx])
                    min_available = min(min_available, group_min)
            aligned_k_per_round[round_idx] = min_available
        print(f'\n[align-k] 各 round 統一 K 值: '
              + ', '.join(f'R{r}={k}' for r, k in sorted(aligned_k_per_round.items())))
    else:
        aligned_k_per_round = None

    # Phase 4: 計算統計
    all_group_stats: list[list[dict]] = []
    for group_rows in all_group_rows:
        if aligned_k_per_round is not None:
            # align-k 模式：每個 round 用不同的統一 K
            stats_list: list[dict] = []
            for round_idx in sorted(group_rows.keys()):
                rows_list = group_rows[round_idx]
                if not rows_list:
                    continue
                round_k = aligned_k_per_round[round_idx]
                top_rates, bottom_rates = [], []
                top_fulls, bottom_fulls, k_vals = [], [], []
                for rows in rows_list:
                    s = compute_topk_stats(rows, round_k)
                    top_rates.append(s['top_rate'])
                    bottom_rates.append(s['bottom_rate'])
                    top_fulls.append(s['top_full_docs'])
                    bottom_fulls.append(s['bottom_full_docs'])
                    k_vals.append(s['k'])
                stats_list.append({
                    'round': round_idx,
                    'top_rate': float(np.mean(top_rates)),
                    'bottom_rate': float(np.mean(bottom_rates)),
                    'top_rate_std': float(np.std(top_rates)),
                    'bottom_rate_std': float(np.std(bottom_rates)),
                    'n_samples': len(top_rates),
                    'avg_top_full': float(np.mean(top_fulls)),
                    'avg_bottom_full': float(np.mean(bottom_fulls)),
                    'avg_k': float(np.mean(k_vals)),
                })
            all_group_stats.append(stats_list)
        else:
            all_group_stats.append(summarize_group_stats(group_rows, top_k))

    # 輸出
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    if aligned_k_per_round is not None:
        k_range = sorted(set(aligned_k_per_round.values()))
        if len(k_range) == 1:
            title_text = f'Top{k_range[0]} vs Bottom{k_range[0]} 文檔準確率比較 (aligned-k)'
        else:
            title_text = f'Top-K vs Bottom-K 文檔準確率比較 (aligned-k: {min(k_range)}~{max(k_range)})'
    else:
        title_text = f'Top{args.top_k} vs Bottom{args.top_k} 文檔準確率比較'

    plot_comparison(
        all_group_stats, args.labels, args.top_k,
        output_dir / f'{args.output_name}.png',
        title_text,
        show_counts=args.show_counts,
    )

    write_report(
        all_group_stats, args.labels, args.top_k,
        output_dir / f'{args.output_name}.txt',
    )


if __name__ == '__main__':
    main()
