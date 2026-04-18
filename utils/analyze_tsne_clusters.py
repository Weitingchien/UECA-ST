"""
分析 t-SNE 圖中左右兩群樣本的 Ground Truth Pair 數量分布

用法:
python utils/analyze_tsne_clusters.py \\
    --experiment_dir <實驗資料夾路徑> \\
    --fold 1 \\
    --round 1 \\
    --dataset_base split10_home_train1_test1_val1_unlabeled7_disjoint
"""

import os
import sys
import json
import argparse
import numpy as np
from glob import glob

# 將專案根目錄加入 sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def find_nest_json(experiment_dir, fold, round_num):
    """尋找對應的 NeST JSON 檔案"""
    # 尋找 pseudo_results 目錄
    pseudo_dirs = glob(os.path.join(experiment_dir, 'pseudo_results_*'))
    if not pseudo_dirs:
        raise FileNotFoundError(f"找不到 pseudo_results 目錄")
    
    pseudo_dir = pseudo_dirs[0]
    nest_json_pattern = os.path.join(pseudo_dir, f'nest_divergence_scores_fold{fold}_round{round_num}.json')
    
    if not os.path.exists(nest_json_pattern):
        raise FileNotFoundError(f"找不到 {nest_json_pattern}")
    
    return nest_json_pattern


def load_ground_truth(dataset_base, fold):
    """載入 Ground Truth 資料，包含 Pair 數量、self-loop 資訊和情緒類別"""
    dataset_paths = [
        os.path.join(project_root, dataset_base, f'fold{fold}_train.json'),
        os.path.join(project_root, dataset_base, f'fold{fold}_val.json'),
        os.path.join(project_root, dataset_base, f'fold{fold}_unlabeled.json'),
    ]
    
    gt_pairs = {}  # {doc_id: num_pairs}
    gt_self_loops = {}  # {doc_id: num_self_loop_pairs}
    gt_emotion_categories = {}  # {doc_id: emotion_category} (只針對單 Pair 文檔)
    
    for path in dataset_paths:
        if not os.path.exists(path):
            continue
        
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for doc in data:
            doc_id = str(doc['doc_id'])
            pairs = doc['pairs']
            num_pairs = len(pairs)
            
            # 統計 self-loop pairs (情緒子句 == 原因子句)
            num_self_loops = sum(1 for emo, cause in pairs if emo == cause)
            
            gt_pairs[doc_id] = num_pairs
            gt_self_loops[doc_id] = num_self_loops
            
            # 如果只有 1 對 Pair，提取情緒類別
            if num_pairs == 1:
                emotion_clause_id = pairs[0][0]  # Pair 的第一個元素是情緒子句 ID
                # 在 clauses 中找到對應的子句
                for clause in doc['clauses']:
                    if int(clause['clause_id']) == emotion_clause_id:
                        emotion_cat = clause.get('emotion_category', 'null')
                        gt_emotion_categories[doc_id] = emotion_cat
                        break
    
    print(f"已載入 Ground Truth: {len(gt_pairs)} 筆文檔")
    print(f"  其中 {len(gt_emotion_categories)} 筆為單 Pair 文檔")
    return gt_pairs, gt_self_loops, gt_emotion_categories


def load_tsne_data(nest_json_path):
    """從 NeST JSON 載入 t-SNE 座標和 doc_ids"""
    with open(nest_json_path, 'r', encoding='utf-8') as f:
        nest_data = json.load(f)
    
    # 檢查是否有 t-SNE 座標
    if 'tsne_2d' not in nest_data:
        raise ValueError(f"{nest_json_path} 中沒有 t-SNE 座標，請先執行 visualize_tsne.py")
    
    tsne_coords = np.array(nest_data['tsne_2d'])
    
    # 收集所有 doc_ids
    all_doc_ids = []
    all_doc_ids.extend(nest_data.get('labeled_doc_ids', []))
    all_doc_ids.extend(nest_data.get('selected_doc_ids', []))
    all_doc_ids.extend(nest_data.get('unselected_doc_ids', []))
    
    print(f"已載入 t-SNE 座標: {len(tsne_coords)} 個點")
    print(f"已載入 doc_ids: {len(all_doc_ids)} 個")
    
    return tsne_coords, all_doc_ids


def analyze_clusters(tsne_coords, doc_ids, gt_pairs, gt_self_loops, gt_emotion_categories, split_method='median'):
    """分析左右兩群的 Pair 數量分布"""
    
    # 根據 x 軸座標分群
    x_coords = tsne_coords[:, 0]
    
    if split_method == 'median':
        threshold = np.median(x_coords)
    elif split_method == 'mean':
        threshold = np.mean(x_coords)
    else:
        raise ValueError(f"不支援的分群方法: {split_method}")
    
    # 分為左右兩群
    left_mask = x_coords < threshold
    right_mask = x_coords >= threshold
    
    left_doc_ids = [doc_ids[i] for i in range(len(doc_ids)) if left_mask[i]]
    right_doc_ids = [doc_ids[i] for i in range(len(doc_ids)) if right_mask[i]]
    
    print(f"\n{'='*60}")
    print(f"分群方式: {split_method} (閾值: {threshold:.2f})")
    print(f"左群樣本數: {len(left_doc_ids)}")
    print(f"右群樣本數: {len(right_doc_ids)}")
    print(f"{'='*60}\n")
    
    # 統計每群的 Pair 數量分布
    def count_pairs(doc_list, group_name):
        single_pair = 0  # 只有 1 對
        multi_pair = 0   # 2 對以上
        missing_gt = 0   # 找不到 Ground Truth
        total_self_loops = 0  # Self-loop pairs 總數
        docs_with_self_loops = 0  # 有 self-loop 的文檔數
        
        pair_distribution = {}  # {num_pairs: count}
        emotion_distribution = {}  # {emotion_category: count} (僅單 Pair)
        
        for doc_id in doc_list:
            doc_id_str = str(doc_id)
            if doc_id_str not in gt_pairs:
                missing_gt += 1
                continue
            
            num_pairs = gt_pairs[doc_id_str]
            num_self_loop = gt_self_loops.get(doc_id_str, 0)
            
            # 統計 self-loop
            total_self_loops += num_self_loop
            if num_self_loop > 0:
                docs_with_self_loops += 1
            
            # 統計分布
            pair_distribution[num_pairs] = pair_distribution.get(num_pairs, 0) + 1
            
            # 統計情緒類別（僅單 Pair）
            if num_pairs == 1 and doc_id_str in gt_emotion_categories:
                emotion_cat = gt_emotion_categories[doc_id_str]
                emotion_distribution[emotion_cat] = emotion_distribution.get(emotion_cat, 0) + 1
            
            # 分類
            if num_pairs == 1:
                single_pair += 1
            elif num_pairs >= 2:
                multi_pair += 1
        
        print(f"【{group_name}】")
        print(f"  總樣本數: {len(doc_list)}")
        print(f"  只有 1 對 Pair: {single_pair} ({single_pair/len(doc_list)*100:.1f}%)")
        print(f"  2 對以上 Pair: {multi_pair} ({multi_pair/len(doc_list)*100:.1f}%)")
        print(f"  同時是情緒和原因子句的 Pair (Self-loop):")
        print(f"    總計 {total_self_loops} 對")
        print(f"    出現在 {docs_with_self_loops} 個文檔中 ({docs_with_self_loops/len(doc_list)*100:.1f}%)")
        if missing_gt > 0:
            print(f"  找不到 Ground Truth: {missing_gt}")
        
        print(f"\n  詳細分布:")
        for num_pairs in sorted(pair_distribution.keys()):
            count = pair_distribution[num_pairs]
            print(f"    {num_pairs} 對 Pair: {count} ({count/len(doc_list)*100:.1f}%)")
        
        # 顯示情緒類別分布（僅單 Pair）
        if emotion_distribution:
            print(f"\n  單 Pair 文檔的情緒類別分布:")
            for emotion_cat in sorted(emotion_distribution.keys()):
                count = emotion_distribution[emotion_cat]
                print(f"    {emotion_cat}: {count} ({count/single_pair*100:.1f}% of single-pair docs)")
        print()
        
        return single_pair, multi_pair, pair_distribution, total_self_loops, docs_with_self_loops, emotion_distribution
    
    # 統計左右兩群
    left_single, left_multi, left_dist, left_self_loops, left_self_loop_docs, left_emotion = count_pairs(left_doc_ids, "左群")
    right_single, right_multi, right_dist, right_self_loops, right_self_loop_docs, right_emotion = count_pairs(right_doc_ids, "右群")
    
    # 總結
    print(f"{'='*60}")
    print(f"總結:")
    print(f"  左群: 單對={left_single}, 多對={left_multi}, Self-loop Pairs={left_self_loops} (出現在 {left_self_loop_docs} 個文檔)")
    print(f"  右群: 單對={right_single}, 多對={right_multi}, Self-loop Pairs={right_self_loops} (出現在 {right_self_loop_docs} 個文檔)")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='分析 t-SNE 圖中左右兩群樣本的 Ground Truth Pair 數量分布')
    parser.add_argument('--experiment_dir', type=str, required=True, help='實驗資料夾路徑')
    parser.add_argument('--fold', type=int, required=True, help='Fold 編號')
    parser.add_argument('--round', type=int, required=True, help='Round 編號')
    parser.add_argument('--dataset_base', type=str, required=True, help='資料集資料夾名稱')
    parser.add_argument('--split_method', type=str, default='median', choices=['median', 'mean'],
                       help='分群方法 (median: 中位數, mean: 平均值)')
    
    args = parser.parse_args()
    
    print(f"分析參數:")
    print(f"  實驗目錄: {args.experiment_dir}")
    print(f"  Fold: {args.fold}, Round: {args.round}")
    print(f"  資料集: {args.dataset_base}\n")
    
    # 1. 載入 t-SNE 座標和 doc_ids
    nest_json_path = find_nest_json(args.experiment_dir, args.fold, args.round)
    tsne_coords, doc_ids = load_tsne_data(nest_json_path)
    
    # 2. 載入 Ground Truth
    gt_pairs, gt_self_loops, gt_emotion_categories = load_ground_truth(args.dataset_base, args.fold)
    
    # 3. 分析左右兩群
    analyze_clusters(tsne_coords, doc_ids, gt_pairs, gt_self_loops, gt_emotion_categories, args.split_method)


if __name__ == '__main__':
    main()
