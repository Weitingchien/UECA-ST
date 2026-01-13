#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
測試 home.txt 中有多少行的 len(parts) < 4
"""
import re

file_path = 'data/ECPE_new_dataset/home.txt'

with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

print(f"總共讀取 {len(lines)} 行\n")
print("=" * 80)
print("找出所有 len(parts) < 4 的行:")
print("=" * 80)

problem_lines = []
line_num = 0

for line_num, line in enumerate(lines, 1):
    line = line.strip()
    
    # 跳過空行
    if not line:
        continue
    
    # 檢查是否為文檔開始行 (doc_id doc_len)
    parts_space = line.split()
    if len(parts_space) == 2 and parts_space[0].isdigit() and parts_space[1].isdigit():
        continue
    
    # 檢查是否為配對行
    if line.startswith('('):
        continue
    
    # 檢查子句行
    parts = line.split(',', 3)
    
    if len(parts) < 4:
        problem_lines.append({
            'line_num': line_num,
            'line': line,
            'parts_len': len(parts),
            'parts': parts
        })

print(f"\n找到 {len(problem_lines)} 行不符合 len(parts) >= 4 的條件\n")

if problem_lines:
    for i, item in enumerate(problem_lines[:20], 1):  # 最多顯示前 20 行
        print(f"\n#{i} 第 {item['line_num']} 行 (len={item['parts_len']}):")
        print(f"   原始行: {item['line']}")
        print(f"   分割後: {item['parts']}")
else:
    print("所有行都符合 len(parts) >= 4 的條件！✓")

if len(problem_lines) > 20:
    print(f"\n... 還有 {len(problem_lines) - 20} 行類似的問題")

print(f"\n統計:")
print(f"  符合條件的行 (len >= 4): ~{line_num - len(problem_lines)}")
print(f"  不符合條件的行 (len < 4): {len(problem_lines)}")
