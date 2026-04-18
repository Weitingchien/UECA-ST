#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, re, glob
import numpy as np
from collections import defaultdict

base = 'ep_split10_t1te1v1_u7_disjoint'
files = glob.glob(os.path.join(base, '*/pseudo_results_*/unselected_correct_summary.txt'))
print(f'找到 {len(files)} 個 unselected_correct_summary.txt\n')

method_data = defaultdict(list)

for f in sorted(files):
    exp_dir = f.split('/pseudo_results_')[0]
    folder = os.path.basename(exp_dir)
    
    parts = folder.split('f1-10_i70_lr1e-5_bs8_wd0.01_')
    method_raw = parts[1] if len(parts) > 1 else 'unknown'
    method = re.sub(r'_seed\d+', '', method_raw)
    method = re.sub(r'_gamma[\d.]+', '', method)
    method = re.sub(r'_st\d+', '', method)
    method = re.sub(r'_ste\d+', '', method)
    method = re.sub(r'_retain_pseudo', '', method)
    method = re.sub(r'_nbeta[\d.]+', '', method)
    method = re.sub(r'_nm[\d.]+', '', method)
    method = re.sub(r'__+', '_', method).strip('_')
    
    seed_match = re.search(r'seed(\d+)', folder)
    seed = seed_match.group(1) if seed_match else '?'
    
    with open(f, 'r', encoding='utf-8') as fh:
        content = fh.read()
    
    m = re.search(r'TOTAL:\s*total_docs=(\d+),\s*all_doc_correct=(\d+)', content)
    if m:
        total_docs = int(m.group(1))
        all_correct = int(m.group(2))
        method_data[method].append((seed, total_docs, all_correct))

# Print per-seed details
header = f"{'方法':<70}  {'seed':>5}  {'全對數(分子)':>12}  {'總數(分母)':>10}  {'全對率':>7}"
print(header)
print('=' * 115)

method_avgs = []
for method in sorted(method_data.keys()):
    seeds = method_data[method]
    rates = []
    for seed, total, correct in sorted(seeds):
        rate = correct / total * 100
        rates.append(rate)
        print(f"{method:<70}  {seed:>5}  {correct:>12}  {total:>10}  {rate:>6.2f}%")
    
    avg_total = sum(t for _, t, _ in seeds) / len(seeds)
    avg_correct = sum(c for _, _, c in seeds) / len(seeds)
    avg_rate = np.mean(rates)
    std_rate = np.std(rates)
    print(f"  -> 3 seeds 平均: 全對={avg_correct:.0f}, 總數={avg_total:.0f}, 全對率={avg_rate:.1f}% ± {std_rate:.1f}%")
    method_avgs.append((method, avg_rate, avg_correct, avg_total, std_rate))
    print()

print()
print('=' * 115)
print('按全對率排序（對應圖中的長條圖）:')
print('=' * 115)
print(f"{'方法':<70}  {'全對率':>7}  {'平均全對數(分子)':>16}  {'平均總數(分母)':>14}")
print('-' * 115)
for method, avg_rate, avg_correct, avg_total, std in sorted(method_avgs, key=lambda x: -x[1]):
    print(f"{method:<70}  {avg_rate:>6.1f}%  {avg_correct:>16.0f}  {avg_total:>14.0f}")
