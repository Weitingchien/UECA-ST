#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
偽標籤錯誤率分析工具

統計每個資料夾中第 1-10 折、第 1-10 輪的偽標籤錯誤數量
錯誤數量 = 選中文檔數 - 3個[MASK]全對的文檔數

使用方式：
    # 分析單一資料夾
    python utils/analyze_pseudo_label_accuracy.py --pred_dir prompt_ECPE_few_shot_ST_2025_12_06.../
    
    # 分析多個資料夾
    python utils/analyze_pseudo_label_accuracy.py --pred_dir folder1/ folder2/ folder3/
"""

import os
import re
import argparse
import glob


def find_pseudo_results_dir(pred_dir):
    """尋找 pseudo_results_* 資料夾"""
    pattern = os.path.join(pred_dir, "pseudo_results_*") # *:匹配任意字串
    dirs = glob.glob(pattern)
    if dirs:
        return dirs[0] # 回傳匹配的路徑字串，e.g., /mnt/d/GithubRepo/UECA_ST/prompt_ECPE_few_shot_ST_2025_12_15_07_57_02_split10_home_train1_test1_val1_unlabeled7_disjoint_f1-10_i70_lr1e-5_bs8_wd0.01_bert-base-chinese_nest_k5_emotion_clause_gamma0.5_reg1e-4_st10_ste3_seed42_retain_pseudo_CE/pseudo_results_2025_12_15_07_57_02
    return None


def count_perfect_docs(filepath):
    """
    計算檔案中 3 個 [MASK] 全對的文檔數量
    
    Returns:
        (total_docs, perfect_docs, perfect_doc_ids): 選中文檔數, 全對文檔數, 全對文檔ID列表
    """
    if not os.path.exists(filepath):
        return 0, 0, []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 計算文檔數量
    doc_matches = re.findall(r'^doc_id: \d+', content, re.MULTILINE)
    total_docs = len(doc_matches)
    
    # 計算錯誤率為 0 的文檔數量並收集其 doc_id
    lines = content.split('\n')
    perfect_count = 0
    perfect_doc_ids = []
    current_doc_id = None
    
    for line in lines:
        # 先檢查是否是 doc_id 行
        doc_match = re.match(r'^doc_id: (\d+)', line)
        if doc_match:
            current_doc_id = int(doc_match.group(1))
        # 再檢查是否為全對的行
        if '正確/總數:' in line and '錯誤率 0.0000' in line and '排除' not in line:
            perfect_count += 1
            if current_doc_id is not None:
                perfect_doc_ids.append(current_doc_id)
    
    return total_docs, perfect_count, perfect_doc_ids


def analyze_folder(pred_dir):
    """分析單一資料夾，統計所有 fold 和 round 的錯誤數量"""
    pseudo_dir = find_pseudo_results_dir(pred_dir)
    if not pseudo_dir:
        print(f"警告：找不到 pseudo_results_* 資料夾於 {pred_dir}")
        return None
    
    folder_name = os.path.basename(pred_dir.rstrip('/\\'))
    
    # 儲存每個 fold 每個 round 的結果
    fold_round_results = {}
    
    total_docs = 0
    total_perfect = 0
    
    for fold in range(1, 11):
        fold_round_results[fold] = {}
        for round_idx in range(1, 11):
            filename = f"pseudo_label_evaluation_fold{fold}_round{round_idx}.txt"
            filepath = os.path.join(pseudo_dir, filename)
            
            docs, perfect, perfect_ids = count_perfect_docs(filepath)
            fold_round_results[fold][round_idx] = {
                'total': docs,
                'perfect': perfect,
                'error': docs - perfect,
                'perfect_doc_ids': perfect_ids
            }
            total_docs += docs
            total_perfect += perfect
    
    total_error = total_docs - total_perfect
    
    return {
        'folder_name': folder_name,
        'fold_round_results': fold_round_results,
        'total_docs': total_docs,
        'perfect_docs': total_perfect,
        'error_docs': total_error,
        'error_rate': total_error / total_docs if total_docs > 0 else 0
    }


def save_result_to_file(pred_dir, result, detail=False):
    """將結果儲存到對應資料夾內"""
    output_path = os.path.join(pred_dir, "pseudo_label_accuracy_summary.txt")
    
    lines = []
    lines.append("=" * 100)
    lines.append(f"偽標籤錯誤率分析報告")
    lines.append(f"資料夾: {result['folder_name']}")
    lines.append("=" * 100)
    
    if detail:
        # 顯示詳細的每個 fold 每個 round 資訊
        header = f"{'Fold':<6}"
        for r in range(1, 11):
            header += f"{'R'+str(r):<10}"
        header += f"{'折總計':<12}"
        lines.append(header)
        lines.append("-" * 100)
        
        for fold in range(1, 11):
            line = f"{fold:<6}"
            fold_total = 0
            fold_error = 0
            for round_idx in range(1, 11):
                r = result['fold_round_results'][fold][round_idx]
                line += f"{r['error']}/{r['total']:<7}"
                fold_total += r['total']
                fold_error += r['error']
            line += f"{fold_error}/{fold_total}"
            lines.append(line)
        lines.append("-" * 100)
        
        # 每輪的總計
        round_line = f"{'輪總計':<6}"
        for round_idx in range(1, 11):
            round_total = sum(result['fold_round_results'][f][round_idx]['total'] for f in range(1, 11))
            round_error = sum(result['fold_round_results'][f][round_idx]['error'] for f in range(1, 11))
            round_line += f"{round_error}/{round_total:<7}"
        lines.append(round_line)
    
    # 總結
    lines.append("")
    lines.append("=" * 100)
    lines.append("總結")
    lines.append("=" * 100)
    lines.append(f"總選中文檔數: {result['total_docs']}")
    lines.append(f"全對文檔數: {result['perfect_docs']}")
    lines.append(f"錯誤文檔數: {result['error_docs']}")
    lines.append(f"錯誤率: {result['error_rate']*100:.2f}%")
    lines.append(f"正確率: {(1 - result['error_rate'])*100:.2f}%")
    lines.append("")
    
    # 新增：全對文檔 ID 詳細資訊
    if detail:
        lines.append("=" * 100)
        lines.append("全對文檔 ID 詳細資訊 (3 個 [MASK] 全部預測正確)")
        lines.append("=" * 100)
        
        for fold in range(1, 11):
            lines.append(f"\nFold {fold}:")
            lines.append("-" * 80)
            for round_idx in range(1, 11):
                r = result['fold_round_results'][fold][round_idx]
                perfect_ids = r.get('perfect_doc_ids', [])
                if perfect_ids:
                    ids_str = ', '.join(map(str, sorted(perfect_ids)))
                    lines.append(f"  R{round_idx}: [{len(perfect_ids)}筆] {ids_str}")
                else:
                    lines.append(f"  R{round_idx}: [無]")
        lines.append("")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description='統計偽標籤錯誤數量（選中文檔數 - 全對文檔數）')
    parser.add_argument('--pred_dir', type=str, nargs='+', required=True,
                        help='要分析的資料夾路徑（可指定多個）')
    parser.add_argument('--detail', action='store_true',
                        help='顯示每個 fold 每個 round 的詳細資訊')
    parser.add_argument('--save', action='store_true',
                        help='將結果儲存到各資料夾內')
    
    args = parser.parse_args()
    
    all_results = []
    
    for pred_dir in args.pred_dir:
        if not os.path.isdir(pred_dir):
            print(f"警告：{pred_dir} 不是有效的資料夾")
            continue
        
        result = analyze_folder(pred_dir)
        
        if result:
            all_results.append((pred_dir, result))
            
            print("=" * 120)
            print(f"資料夾: {result['folder_name']}")
            print("=" * 120)
            
            if args.detail:
                # 顯示詳細的每個 fold 每個 round 資訊
                print(f"{'Fold':<6}", end="")
                for r in range(1, 11):
                    print(f"{'R'+str(r):<10}", end="")
                print(f"{'Fold總計':<12}")
                print("-" * 120)
                
                for fold in range(1, 11):
                    print(f"{fold:<6}", end="")
                    fold_total = 0
                    fold_error = 0
                    for round_idx in range(1, 11):
                        r = result['fold_round_results'][fold][round_idx]
                        # 顯示 錯誤/總數
                        print(f"{r['error']}/{r['total']:<7}", end="")
                        fold_total += r['total']
                        fold_error += r['error']
                    print(f"{fold_error}/{fold_total}")
                print("-" * 120)
                
                # 每輪的總計
                print(f"{'輪總計':<6}", end="")
                for round_idx in range(1, 11):
                    round_total = sum(result['fold_round_results'][f][round_idx]['total'] for f in range(1, 11))
                    round_error = sum(result['fold_round_results'][f][round_idx]['error'] for f in range(1, 11))
                    print(f"{round_error}/{round_total:<7}", end="")
                print()
            
            # 總結
            print()
            print(f"總選中文檔數: {result['total_docs']}")
            print(f"全對文檔數: {result['perfect_docs']}")
            print(f"錯誤文檔數: {result['error_docs']}")
            print(f"錯誤率: {result['error_rate']*100:.2f}%")
            print()
            
            # 儲存結果到資料夾
            if args.save:
                output_path = save_result_to_file(pred_dir, result, detail=args.detail)
                print(f"結果已儲存至: {output_path}")
                print()
    
    # 多資料夾比較表
    if len(all_results) > 1:
        print("=" * 120)
        print("多資料夾比較")
        print("=" * 120)
        print(f"{'資料夾':<70} {'總文檔':<10} {'錯誤':<10} {'錯誤率%':<10}")
        print("-" * 120)
        
        for pred_dir, result in all_results:
            name = result['folder_name']
            if len(name) > 65:
                name = "..." + name[-62:]
            print(f"{name:<70} {result['total_docs']:<10} {result['error_docs']:<10} {result['error_rate']*100:<10.2f}")
        
        print("=" * 120)


if __name__ == '__main__':
    main()
