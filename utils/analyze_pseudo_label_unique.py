#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
偽標籤去重錯誤率分析工具

解決 retain_pseudo 模式下同一 doc_id 被重複計算的問題。
以每個 fold 內 unique doc_id 為單位統計，並追蹤跨輪預測變化。

輸出三個指標：
  1. Unique Doc Error Rate: 去重後的文檔級錯誤率
  2. 跨輪預測變化: 同一 doc_id 在不同輪次的正確→錯誤 / 錯誤→正確 比例
  3. 每輪新增 vs 重複文檔比例

使用方式：
    # 分析單一資料夾 (單一 seed)
    python utils/analyze_pseudo_label_unique.py --pred_dir <實驗資料夾>

    # 讀取 summary.txt，自動找到同方法的多個 seed 並計算平均
    python utils/analyze_pseudo_label_unique.py --summary <summary.txt> --experiments-root <根目錄>
"""

import os
import re
import argparse
import glob
from pathlib import Path


def find_pseudo_results_dir(pred_dir):
    """尋找 pseudo_results_* 資料夾"""
    dirs = glob.glob(os.path.join(pred_dir, "pseudo_results_*"))
    return dirs[0] if dirs else None


def parse_evaluation_file(filepath):
    """解析單一 evaluation 檔案，回傳 {doc_id: is_perfect} 字典
    
    is_perfect = True 表示該文檔所有 mask 預測正確 (錯誤率 0.0000)
    """
    if not os.path.exists(filepath):
        return {}

    results = {}
    current_doc_id = None

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            doc_match = re.match(r'^doc_id: (\S+)', line)
            if doc_match:
                current_doc_id = doc_match.group(1)
                continue
            # 只看整體正確率那行（不含「排除」）
            if current_doc_id and '正確/總數:' in line and '排除' not in line:
                results[current_doc_id] = '錯誤率 0.0000' in line
                current_doc_id = None

    return results


def analyze_folder(pred_dir, num_folds=10, num_rounds=10):
    """分析單一實驗資料夾"""
    pseudo_dir = find_pseudo_results_dir(pred_dir)
    if not pseudo_dir:
        print(f"警告：找不到 pseudo_results_* 於 {pred_dir}")
        return None

    # 收集所有資料: all_data[fold][round] = {doc_id: is_perfect}
    all_data = {}
    for fold in range(1, num_folds + 1):
        all_data[fold] = {}
        for r in range(1, num_rounds + 1):
            path = os.path.join(pseudo_dir, f"pseudo_label_evaluation_fold{fold}_round{r}.txt")
            all_data[fold][r] = parse_evaluation_file(path)

    # ===== 指標 1: Unique Doc Error Rate (取最後一次出現的結果) =====
    unique_stats = {}  # {fold: {doc_id: last_is_perfect}}
    for fold in range(1, num_folds + 1):
        doc_last = {}
        for r in range(1, num_rounds + 1):
            for doc_id, is_perfect in all_data[fold][r].items():
                doc_last[doc_id] = is_perfect  # 後出現的覆蓋前面的
        unique_stats[fold] = doc_last

    # ===== 指標 2: 跨輪預測變化追蹤 =====
    # 對每個 fold，追蹤 doc_id 首次出現 vs 最後出現的狀態變化
    change_stats = {'corrected': 0, 'degraded': 0, 'always_correct': 0, 'always_wrong': 0}
    for fold in range(1, num_folds + 1):
        doc_first = {}
        doc_last = {}
        for r in range(1, num_rounds + 1):
            for doc_id, is_perfect in all_data[fold][r].items():
                if doc_id not in doc_first:
                    doc_first[doc_id] = is_perfect
                doc_last[doc_id] = is_perfect
        for doc_id in doc_first:
            first, last = doc_first[doc_id], doc_last[doc_id]
            if first and last:
                change_stats['always_correct'] += 1
            elif not first and not last:
                change_stats['always_wrong'] += 1
            elif not first and last:
                change_stats['corrected'] += 1   # 首次錯 → 最終對
            else:
                change_stats['degraded'] += 1     # 首次對 → 最終錯

    # ===== 指標 3: 每輪新增 vs 重複文檔比例 =====
    round_novelty = {}  # {round: {'new': N, 'repeat': N, 'total': N}}
    for r in range(1, num_rounds + 1):
        new_count, repeat_count = 0, 0
        for fold in range(1, num_folds + 1):
            seen_before = set()
            for prev_r in range(1, r):
                seen_before.update(all_data[fold][prev_r].keys())
            for doc_id in all_data[fold][r]:
                if doc_id in seen_before:
                    repeat_count += 1
                else:
                    new_count += 1
        round_novelty[r] = {'new': new_count, 'repeat': repeat_count, 'total': new_count + repeat_count}

    # ===== 每輪去重錯誤率 (累計到該輪為止的 unique doc) =====
    round_unique_error = {}  # {round: (error, total)}
    for r in range(1, num_rounds + 1):
        cumulative = {}  # {fold: {doc_id: latest_is_perfect up to round r}}
        for fold in range(1, num_folds + 1):
            doc_latest = {}
            for rr in range(1, r + 1):
                for doc_id, is_perfect in all_data[fold][rr].items():
                    doc_latest[doc_id] = is_perfect
            cumulative[fold] = doc_latest
        total = sum(len(v) for v in cumulative.values())
        perfect = sum(sum(1 for p in v.values() if p) for v in cumulative.values())
        round_unique_error[r] = (total - perfect, total)

    return {
        'folder_name': os.path.basename(pred_dir.rstrip('/\\')),
        'unique_stats': unique_stats,
        'change_stats': change_stats,
        'round_novelty': round_novelty,
        'round_unique_error': round_unique_error,
    }


def format_report(result):
    """產生單一 seed 的報告文字"""
    lines = []
    lines.append("=" * 90)
    lines.append("偽標籤去重錯誤率分析報告 (Unique Doc)")
    lines.append(f"資料夾: {result['folder_name']}")
    lines.append("=" * 90)

    # --- 指標 1: 整體 Unique Doc Error Rate ---
    us = result['unique_stats']
    total_unique = sum(len(docs) for docs in us.values())
    total_perfect = sum(sum(1 for p in docs.values() if p) for docs in us.values())
    total_error = total_unique - total_perfect
    error_rate = total_error / total_unique * 100 if total_unique else 0

    lines.append(f"\n【指標 1】Unique Doc Error Rate (以最後一輪結果為準)")
    lines.append(f"  Unique 文檔總數 (10 fold 合計): {total_unique}")
    lines.append(f"  全對文檔數: {total_perfect}")
    lines.append(f"  錯誤文檔數: {total_error}")
    lines.append(f"  錯誤率: {error_rate:.2f}%")

    # 每 fold 明細
    lines.append(f"\n  {'Fold':<6} {'Unique Docs':<14} {'Perfect':<10} {'Error':<10} {'Error Rate':<12}")
    lines.append(f"  {'-'*52}")
    for fold in sorted(us.keys()):
        n = len(us[fold])
        p = sum(1 for v in us[fold].values() if v)
        e = n - p
        r = e / n * 100 if n else 0
        lines.append(f"  {fold:<6} {n:<14} {p:<10} {e:<10} {r:.2f}%")

    # --- 指標 2: 跨輪預測變化 ---
    cs = result['change_stats']
    total_docs = sum(cs.values())
    lines.append(f"\n{'='*90}")
    lines.append(f"【指標 2】跨輪預測變化 (首次選中 vs 最後一次)")
    lines.append(f"  始終正確 (✓→✓): {cs['always_correct']:>5} ({cs['always_correct']/total_docs*100:.1f}%)")
    lines.append(f"  始終錯誤 (✗→✗): {cs['always_wrong']:>5} ({cs['always_wrong']/total_docs*100:.1f}%)")
    lines.append(f"  被修正   (✗→✓): {cs['corrected']:>5} ({cs['corrected']/total_docs*100:.1f}%)")
    lines.append(f"  退化     (✓→✗): {cs['degraded']:>5} ({cs['degraded']/total_docs*100:.1f}%)")

    # --- 指標 3: 每輪新增 vs 重複 ---
    lines.append(f"\n{'='*90}")
    lines.append(f"【指標 3】每輪新增 vs 重複文檔 & 累計去重錯誤率")
    lines.append(f"  {'Round':<7} {'New':<8} {'Repeat':<10} {'Total':<8} {'Repeat%':<10} {'UniqueErr':<12} {'UniqueTotal':<14} {'ErrRate':<10}")
    lines.append(f"  {'-'*79}")
    rn = result['round_novelty']
    rue = result['round_unique_error']
    for r in range(1, 11):
        n = rn[r]
        repeat_pct = n['repeat'] / n['total'] * 100 if n['total'] else 0
        ue, ut = rue[r]
        ue_rate = ue / ut * 100 if ut else 0
        lines.append(f"  R{r:<5} {n['new']:<8} {n['repeat']:<10} {n['total']:<8} {repeat_pct:<9.1f}% {ue:<12} {ut:<14} {ue_rate:.2f}%")

    lines.append("")
    return '\n'.join(lines)


def read_experiment_names_from_summary(summary_file):
    """從 *_summary.txt 中解析出實驗資料夾名稱列表"""
    text = Path(summary_file).read_text(encoding="utf-8", errors="ignore")
    names = []
    for line in text.splitlines():
        m = re.match(r"\s*-\s*(prompt_ECPE_few_shot_ST_[^\s]+)", line)
        if m:
            names.append(m.group(1).strip())
    return names


def format_avg_report(summary_name, seed_results, num_rounds=10):
    """產生多 seed 平均的報告文字"""
    n_seeds = len(seed_results)
    lines = []
    lines.append("=" * 90)
    lines.append(f"偽標籤去重錯誤率分析報告 (Unique Doc, {n_seeds} seeds 平均)")
    lines.append(f"Summary: {summary_name}")
    for r in seed_results:
        lines.append(f"  - {r['folder_name']}")
    lines.append("=" * 90)

    # --- 指標 1: 平均 Unique Doc Error Rate ---
    seed_error_rates = []
    for r in seed_results:
        us = r['unique_stats']
        total = sum(len(docs) for docs in us.values())
        perfect = sum(sum(1 for p in docs.values() if p) for docs in us.values())
        seed_error_rates.append((total - perfect) / total * 100 if total else 0)

    avg_err = sum(seed_error_rates) / n_seeds
    lines.append(f"\n【指標 1】Unique Doc Error Rate (以最後一輪結果為準)")
    for i, r in enumerate(seed_results):
        us = r['unique_stats']
        t = sum(len(docs) for docs in us.values())
        e = t - sum(sum(1 for p in docs.values() if p) for docs in us.values())
        lines.append(f"  seed{i+1}: {e}/{t} = {seed_error_rates[i]:.2f}%")
    lines.append(f"  平均: {avg_err:.2f}%")

    # --- 指標 2: 平均跨輪預測變化 ---
    keys = ['always_correct', 'always_wrong', 'corrected', 'degraded']
    labels = {'always_correct': '始終正確 (✓→✓)', 'always_wrong': '始終錯誤 (✗→✗)',
              'corrected': '被修正   (✗→✓)', 'degraded': '退化     (✓→✗)'}
    avg_pct = {k: 0.0 for k in keys}
    for r in seed_results:
        cs = r['change_stats']
        total = sum(cs.values())
        for k in keys:
            avg_pct[k] += cs[k] / total * 100 if total else 0
    for k in keys:
        avg_pct[k] /= n_seeds

    lines.append(f"\n{'='*90}")
    lines.append(f"【指標 2】跨輪預測變化 (首次選中 vs 最後一次, {n_seeds} seeds 平均)")
    for k in keys:
        counts_str = ", ".join(f"{r['change_stats'][k]}" for r in seed_results)
        totals_str = ", ".join(f"{sum(r['change_stats'].values())}" for r in seed_results)
        lines.append(f"  {labels[k]}: {avg_pct[k]:.1f}%  (各seed: [{counts_str}] / [{totals_str}])")

    # --- 指標 3: 每輪累計去重錯誤率 (平均) ---
    lines.append(f"\n{'='*90}")
    lines.append(f"【指標 3】每輪累計去重錯誤率 ({n_seeds} seeds 平均)")
    lines.append(f"  {'Round':<7} {'AvgRepeat%':>10}  {'AvgErrRate':>10}  " +
                 " ".join(f"{'seed'+str(i+1)+'(Err)':>16}" for i in range(n_seeds)))
    lines.append(f"  {'-'*100}")

    for rd in range(1, num_rounds + 1):
        seed_repeat_pcts = []
        seed_err_rates = []
        seed_fractions = []
        seed_repeat_fractions = []
        for r in seed_results:
            rn = r['round_novelty'][rd]
            seed_repeat_pcts.append(rn['repeat'] / rn['total'] * 100 if rn['total'] else 0)
            seed_repeat_fractions.append((rn['repeat'], rn['total'], rn['new']))
            ue, ut = r['round_unique_error'][rd]
            seed_err_rates.append(ue / ut * 100 if ut else 0)
            seed_fractions.append((ue, ut))
        avg_repeat = sum(seed_repeat_pcts) / n_seeds
        avg_err_r = sum(seed_err_rates) / n_seeds
        per_seed = " ".join(f"{e}/{t}={e/t*100:.1f}%" if t else "N/A" for e, t in seed_fractions)
        rep_seed = " ".join(
            f"rep{rep}/{tot}={rep/tot*100:.1f}%" if tot else "N/A"
            for rep, tot, _new in seed_repeat_fractions
        )
        lines.append(f"  R{rd:<5} {avg_repeat:>9.1f}%  {avg_err_r:>9.2f}%  {per_seed}")
        lines.append(f"{'':<11} {'':>10}  {'':>10}  {rep_seed}")

    lines.append("")
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description='偽標籤去重錯誤率分析')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--pred_dir', nargs='+', help='實驗資料夾路徑 (單一 seed 模式)')
    group.add_argument('--summary', nargs='+', help='*_summary.txt 路徑 (多 seed 平均模式)')
    parser.add_argument('--experiments-root', default='.', help='實驗資料夾的根目錄 (summary 模式用)')
    parser.add_argument('--num_folds', type=int, default=10)
    parser.add_argument('--num_rounds', type=int, default=10)
    args = parser.parse_args()

    if args.pred_dir:
        # 單一 seed 模式: 與原本相同
        for pred_dir in args.pred_dir:
            result = analyze_folder(pred_dir, args.num_folds, args.num_rounds)
            if result is None:
                continue
            report = format_report(result)
            print(report)

            output_path = os.path.join(pred_dir, "pseudo_label_unique_doc_analysis.txt")
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"已儲存至: {output_path}\n")

    else:
        # 多 seed 平均模式: 從 summary.txt 讀取資料夾名稱
        for summary_file in args.summary:
            exp_names = read_experiment_names_from_summary(summary_file)
            if not exp_names:
                print(f"警告：{summary_file} 中未找到實驗資料夾名稱")
                continue

            seed_results = []
            for name in exp_names:
                pred_dir = os.path.join(args.experiments_root, name)
                result = analyze_folder(pred_dir, args.num_folds, args.num_rounds)
                if result:
                    seed_results.append(result)
                else:
                    print(f"  警告：跳過 {name}")

            if not seed_results:
                continue

            # 每個 seed 的個別報告
            for result in seed_results:
                report = format_report(result)
                pred_dir = os.path.join(args.experiments_root, result['folder_name'])
                output_path = os.path.join(pred_dir, "pseudo_label_unique_doc_analysis.txt")
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(report)

            # 平均報告
            summary_name = os.path.basename(summary_file)
            avg_report = format_avg_report(summary_name, seed_results, args.num_rounds)
            print(avg_report)

            # 儲存到 summary.txt 同目錄
            summary_dir = os.path.dirname(os.path.abspath(summary_file))
            stem = Path(summary_file).stem  # e.g. "UECA_Prompt_..._summary"
            output_path = os.path.join(summary_dir, stem.replace("_summary", "_unique_doc_analysis") + ".txt")
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(avg_report)
            print(f"已儲存至: {output_path}\n")


if __name__ == '__main__':
    main()
