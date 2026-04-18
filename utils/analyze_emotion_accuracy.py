"""
分析偽標籤準確度與情緒類別的關聯

用法:
python utils/analyze_emotion_accuracy.py \
    --experiment_dir <實驗資料夾路徑> \
    --fold 1 \
    --round 1 \
    --dataset_base split10_home_train1_test1_val1_unlabeled7_disjoint
"""

import os
import sys
import json
import argparse
from glob import glob
from collections import defaultdict

# 將專案根目錄加入 sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def find_pseudo_results_dir(experiment_dir):
    """尋找 pseudo_results 目錄"""
    pseudo_dirs = glob(os.path.join(experiment_dir, 'pseudo_results_*'))
    if not pseudo_dirs:
        raise FileNotFoundError(f"找不到 pseudo_results 目錄")
    return pseudo_dirs[0]


def load_accuracy_map(experiment_dir, fold, round_num):
    """載入偽標籤準確度資訊"""
    pseudo_dir = find_pseudo_results_dir(experiment_dir)
    eval_file = os.path.join(pseudo_dir, f'pseudo_label_evaluation_fold{fold}_round{round_num}.txt')
    
    if not os.path.exists(eval_file):
        raise FileNotFoundError(f"找不到準確度評估檔案: {eval_file}")
    
    accuracy_map = {}  # {doc_id: error_rate}
    
    with open(eval_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    current_doc_id = None
    for line in lines:
        line = line.strip()
        
        # 解析 doc_id 行
        if line.startswith('doc_id:'):
            current_doc_id = line.split()[1]
        
        # 解析錯誤率行（不排除「非 非 无」的版本）
        elif '正確/總數:' in line and current_doc_id and '排除' not in line:
            # 格式: "  正確/總數: 36/39 (錯誤率 0.0769)"
            parts = line.split('錯誤率')
            if len(parts) >= 2:
                error_rate_str = parts[1].strip().rstrip(')')
                error_rate = float(error_rate_str)
                accuracy_map[current_doc_id] = error_rate
                current_doc_id = None  # 重置
    
    print(f"已載入準確度資訊: {len(accuracy_map)} 筆")
    return accuracy_map


def load_ground_truth_with_emotions(dataset_base, fold):
    """載入 Ground Truth，包含情緒類別"""
    train_path = os.path.join(project_root, dataset_base, f'fold{fold}_train.json')
    unlabeled_path = os.path.join(project_root, dataset_base, f'fold{fold}_unlabeled.json')
    
    doc_emotions = {}  # {doc_id: emotion_category}
    train_docs = set()  # 標註樣本的 doc_ids (僅 train.json)
    unlabeled_docs = set()  # 未標註樣本的 doc_ids (僅 unlabeled.json)
    
    for path, is_train in [(train_path, True), (unlabeled_path, False)]:
        if not os.path.exists(path):
            continue
        
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for doc in data:
            doc_id = str(doc['doc_id'])
            pairs = doc['pairs']
            
            if is_train:
                train_docs.add(doc_id)
            else:
                unlabeled_docs.add(doc_id)
            
            # 提取所有情緒子句的情緒類別
            emotion_categories = set()
            for emo_clause_id, _ in pairs:
                for clause in doc['clauses']:
                    if int(clause['clause_id']) == emo_clause_id:
                        emotion_cat = clause.get('emotion_category', 'null')
                        if emotion_cat != 'null':
                            emotion_categories.add(emotion_cat)
                        break
            
            # 如果有多個情緒類別，記錄為 "mixed"
            if len(emotion_categories) == 0:
                doc_emotions[doc_id] = 'null'
            elif len(emotion_categories) == 1:
                doc_emotions[doc_id] = list(emotion_categories)[0]
            else:
                doc_emotions[doc_id] = 'mixed'
    
    print(f"已載入情緒類別: {len(doc_emotions)} 筆文檔")
    print(f"  標註樣本 (train.json): {len(train_docs)} 筆")
    print(f"  未標註樣本 (unlabeled.json): {len(unlabeled_docs)} 筆")
    
    return doc_emotions, train_docs, unlabeled_docs


def load_nest_data(experiment_dir, fold, round_num):
    """載入 NeST 選中樣本資訊"""
    pseudo_dir = find_pseudo_results_dir(experiment_dir)
    nest_json = os.path.join(pseudo_dir, f'nest_divergence_scores_fold{fold}_round{round_num}.json')
    
    if not os.path.exists(nest_json):
        raise FileNotFoundError(f"找不到 NeST JSON: {nest_json}")
    
    with open(nest_json, 'r', encoding='utf-8') as f:
        nest_data = json.load(f)
    
    selected_docs = set(nest_data.get('selected_doc_ids', []))
    print(f"已載入 NeST 選中樣本: {len(selected_docs)} 筆")
    
    return selected_docs


def analyze_accuracy_by_emotion(accuracy_map, doc_emotions, train_docs, unlabeled_docs, selected_docs):
    """分析準確度與情緒類別的關聯"""
    
    # 調試資訊
    print(f"\n{'='*60}")
    print(f"【調試資訊】")
    print(f"  總選中樣本數: {len(selected_docs)}")
    print(f"  有準確度資訊的樣本數: {len(accuracy_map)}")
    
    selected_train = selected_docs & train_docs
    selected_unlabeled = selected_docs & unlabeled_docs
    
    print(f"  選中的標註樣本 (train): {len(selected_train)}")
    print(f"  選中的未標註樣本 (unlabeled): {len(selected_unlabeled)}")
    print(f"{'='*60}")
    
    # 只分析被選中的樣本
    selected_with_accuracy = {doc_id for doc_id in selected_docs if doc_id in accuracy_map}
    
    print(f"\n分析被選中且有準確度資訊的樣本: {len(selected_with_accuracy)} 筆")
    print(f"{'='*60}\n")
    
    # 1. 按準確度分類的情緒分布
    correct_emotions = defaultdict(int)  # error_rate == 0.0
    partial_emotions = defaultdict(int)  # 0.0 < error_rate < 1.0
    wrong_emotions = defaultdict(int)    # error_rate >= 1.0
    
    for doc_id in selected_with_accuracy:
        error_rate = accuracy_map[doc_id]
        emotion = doc_emotions.get(doc_id, 'unknown')
        
        # 準確度分類
        if error_rate == 0.0:
            correct_emotions[emotion] += 1
        elif error_rate < 1.0:
            partial_emotions[emotion] += 1
        else:
            wrong_emotions[emotion] += 1
    
    # 2. 標註樣本整體分布 (所有 train.json)
    train_emotions = defaultdict(int)
    for doc_id in train_docs:
        emotion = doc_emotions.get(doc_id, 'unknown')
        train_emotions[emotion] += 1
    
    # 3. 被選中的未標註樣本分布 (selected from unlabeled.json)
    selected_unlabeled_emotions = defaultdict(int)
    for doc_id in selected_unlabeled:
        emotion = doc_emotions.get(doc_id, 'unknown')
        selected_unlabeled_emotions[emotion] += 1
    
    # 顯示結果
    print("【1. 準確度對應的情緒類別分布（僅被選中且有準確度資訊的樣本）】\n")
    
    print("完全正確 (error_rate = 0.0):")
    total_correct = sum(correct_emotions.values())
    if total_correct > 0:
        for emotion in sorted(correct_emotions.keys()):
            count = correct_emotions[emotion]
            print(f"  {emotion}: {count} ({count/total_correct*100:.1f}%)")
        print(f"  總計: {total_correct}\n")
    else:
        print("  (無)\n")
    
    print("部分正確 (0.0 < error_rate < 1.0):")
    total_partial = sum(partial_emotions.values())
    if total_partial > 0:
        for emotion in sorted(partial_emotions.keys()):
            count = partial_emotions[emotion]
            print(f"  {emotion}: {count} ({count/total_partial*100:.1f}%)")
        print(f"  總計: {total_partial}\n")
    else:
        print("  (無)\n")
    
    print("完全錯誤 (error_rate >= 1.0):")
    total_wrong = sum(wrong_emotions.values())
    if total_wrong > 0:
        for emotion in sorted(wrong_emotions.keys()):
            count = wrong_emotions[emotion]
            print(f"  {emotion}: {count} ({count/total_wrong*100:.1f}%)")
        print(f"  總計: {total_wrong}\n")
    else:
        print("  (無)\n")
    
    print(f"{'='*60}\n")
    print("【2. 標註樣本 (train.json) 整體情緒類別分布】\n")
    
    total_train = sum(train_emotions.values())
    for emotion in sorted(train_emotions.keys()):
        count = train_emotions[emotion]
        print(f"  {emotion}: {count} ({count/total_train*100:.1f}%)")
    print(f"  總計: {total_train}\n")
    
    print(f"{'='*60}\n")
    print("【3. 被選中的未標註樣本 (selected from unlabeled.json) 情緒類別分布】\n")
    
    total_unlabeled = sum(selected_unlabeled_emotions.values())
    if total_unlabeled > 0:
        for emotion in sorted(selected_unlabeled_emotions.keys()):
            count = selected_unlabeled_emotions[emotion]
            print(f"  {emotion}: {count} ({count/total_unlabeled*100:.1f}%)")
        print(f"  總計: {total_unlabeled}\n")
    else:
        print("  (無被選中的未標註樣本)\n")


def main():
    parser = argparse.ArgumentParser(description='分析偽標籤準確度與情緒類別的關聯')
    parser.add_argument('--experiment_dir', type=str, required=True, help='實驗資料夾路徑')
    parser.add_argument('--fold', type=int, required=True, help='Fold 編號')
    parser.add_argument('--round', type=int, required=True, help='Round 編號')
    parser.add_argument('--dataset_base', type=str, required=True, help='資料集資料夾名稱')
    
    args = parser.parse_args()
    
    print(f"分析參數:")
    print(f"  實驗目錄: {args.experiment_dir}")
    print(f"  Fold: {args.fold}, Round: {args.round}")
    print(f"  資料集: {args.dataset_base}\n")
    
    # 1. 載入準確度資訊
    accuracy_map = load_accuracy_map(args.experiment_dir, args.fold, args.round)
    
    # 2. 載入 Ground Truth 情緒類別
    doc_emotions, train_docs, unlabeled_docs = load_ground_truth_with_emotions(args.dataset_base, args.fold)
    
    # 3. 載入 NeST 選中樣本
    selected_docs = load_nest_data(args.experiment_dir, args.fold, args.round)
    
    # 4. 分析
    analyze_accuracy_by_emotion(accuracy_map, doc_emotions, train_docs, unlabeled_docs, selected_docs)


if __name__ == '__main__':
    main()
