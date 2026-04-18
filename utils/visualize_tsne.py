#!/usr/bin/env python3
"""
================================================================================
t-SNE 視覺化腳本：分析 NeST 樣本選擇分佈
================================================================================

【功能說明】
腳本將高維的文檔 embedding (768維) 降維到 2D 平面進行視覺化，
用於分析 NeST 樣本選擇策略的效果

【t-SNE 參數說明】
- n_components=2: 降維後的目標維度數
  → 2 維可以在平面上繪製成散點圖 (x, y 座標)
  
- perplexity=30: 困惑度，可理解為「每個點考慮多少個有效鄰居」
  → 小值 (5-15)：強調局部結構，小群聚更明顯
  → 中值 (15-50)：平衡局部與全局結構（推薦）
  → 大值 (50+)：強調全局結構，群聚可能被合併
  
- n_iter=1000: 優化迭代次數
  → 太小 (<500)：演算法未收斂，結構不穩定
  → 足夠 (1000)：已收斂，群聚分離清楚
  → 太大 (>3000)：更穩定但浪費計算時間

【Embedding 模式】
- cls: 使用 [CLS] token 的 embedding (768維)
- cause_clause: 對被預測為「原因子句」的子句做 mean pooling
- cause_clause_mask: cause_clause + [MASK]_cause embedding 的平均

【使用方式】
python utils/visualize_tsne.py \\
    --experiment_dir <實驗資料夾路徑> \\
    --fold 1 \\
    --round 1 \\
    --show_accuracy \\
    --show_doc_ids

【輸出】
<實驗資料夾>/tsne_analysis/tsne_accuracy_round{round}_fold{fold}.png
================================================================================
"""


import os
import sys
import re
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from glob import glob

# 設定字體
plt.rcParams['font.family'] = ['Calisto MT', 'Microsoft JhengHei', 'SimSun', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False # 用來正常顯示負號

# 將專案根目錄加入 sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 動態載入 prompt_bert 類別，避免執行訓練腳本的 argparse
# 使用 importlib 技巧：先載入模組但不執行 opt = parser.parse_args()
import importlib.util
def _load_prompt_bert_class():
    """動態載入 prompt_bert 類別，繞過 argparse 執行"""
    spec = importlib.util.spec_from_file_location(
        "training_module", 
        os.path.join(project_root, "UECA_CE_few_shot_ST_nest.py")
    )
    module = importlib.util.module_from_spec(spec)
    
    # 暫時替換 sys.argv 以繞過 argparse
    original_argv = sys.argv
    sys.argv = [sys.argv[0]]  # 只保留腳本名稱，移除其他參數
    
    try:
        spec.loader.exec_module(module)
    finally:
        sys.argv = original_argv  # 還原 sys.argv
    
    return module.prompt_bert

# 載入 prompt_bert 類別 (torch.load 反序列化時需要)
prompt_bert = _load_prompt_bert_class()

# 設定 matplotlib 支援中文
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'DejaVu Sans']

plt.rcParams['axes.unicode_minus'] = False



def parse_args():
    parser = argparse.ArgumentParser(description='t-SNE 視覺化：NeST 樣本選擇分佈')
    parser.add_argument('--experiment_dir', type=str, required=True,
                        help='實驗資料夾路徑 (包含 fold{f}.pth 和 pseudo_results_* 的資料夾)')
    parser.add_argument('--fold', type=int, default=1,
                        help='要分析的 fold 編號 (預設: 1)')
    parser.add_argument('--round', type=int, default=1,
                        help='要分析的 self-training round 編號 (預設: 1)')
    parser.add_argument('--all_folds', action='store_true',
                        help='處理所有 folds (1~10)')
    parser.add_argument('--all_rounds', action='store_true',
                        help='處理所有 rounds (1~10)')
    parser.add_argument('--show_accuracy', action='store_true',
                        help='顯示僞標籤準確度，綠色=完全正確，紅色=有錯誤')
    parser.add_argument('--show_doc_ids', action='store_true',
                        help='在圖上標記文檔 ID')
    parser.add_argument('--knn_embedding_mode', type=str, default=None,
                        help='Embedding 模式 (cls/cause_clause/emotion_clause)。若不指定會從資料夾名稱自動偵測')
    parser.add_argument('--perplexity', type=int, default=30,
                        help='t-SNE perplexity 參數 (預設: 30)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='輸出資料夾。若不指定則為 <experiment_dir>/tsne_analysis/')
    return parser.parse_args()


def detect_embedding_mode_from_folder(folder_name):
    """從資料夾名稱自動偵測 knn_embedding_mode"""
    # 搜尋 knn{mode}_mask 模式 (例如 knncause_clause_mask)
    # 注意: cause_clause_mask 需要保留完整的 cause_clause_mask
    match = re.search(r'knn(cause_clause_mask|emotion_clause_mask|cause_clause|emotion_clause|cls)', folder_name)
    if match:
        mode = match.group(1)
        print(f"自動偵測 knn_embedding_mode: {mode}")
        return mode
    
    # 嘗試更通用的模式
    match = re.search(r'knn([a-z_]+)', folder_name)
    if match:
        mode = match.group(1)
        print(f"自動偵測 knn_embedding_mode: {mode}")
        return mode
    
    print("無法自動偵測 embedding mode，使用預設值: cls")
    return 'cls'


def find_nest_json(experiment_dir, fold, round_num):
    """尋找對應的 NeST divergence JSON 檔案"""
    pattern = os.path.join(experiment_dir, 'pseudo_results_*', f'nest_divergence_scores_fold{fold}_round{round_num}.json')
    matches = glob(pattern)
    if not matches:
        raise FileNotFoundError(f"找不到 NeST JSON 檔案: {pattern}")
    return matches[0]


def find_dataset_path(experiment_dir):
    """從實驗資料夾名稱推斷資料集路徑"""
    # 取得父資料夾名稱 (例如 ep_split10_t1_te1_v1_u7_disjoint)
    parent_name = os.path.basename(os.path.dirname(experiment_dir))
    
    # 嘗試直接解析格式: ep_split10_[home_]t1[_]te1[_]v1[_]u7_disjoint
    # 資料集在根目錄，不在 data/ 子資料夾
    if 'split10' in parent_name:
        if 'home' in parent_name:
            dataset_path = 'split10_home_train1_test1_val1_unlabeled7_disjoint'
        else:
            dataset_path = 'split10_train1_test1_val1_unlabeled7_disjoint'
        print(f"推斷資料集路徑: {dataset_path}")
        return dataset_path
    
    raise ValueError(f"無法從資料夾 {parent_name} 推斷資料集路徑")




class SimpleDataset(Dataset):
    """簡化版資料集，只載入 x_bert 用於 embedding 提取"""
    def __init__(self, input_file, tokenizer):
        print(f'載入資料: {input_file}')
        self.x_bert = []
        self.doc_id = []
        self.tokenizer = tokenizer
        
        with open(input_file, 'r', encoding='utf8') as f:
            data = json.load(f)
        
        for doc in data:
            doc_id = doc["doc_id"]
            d_len = doc["doc_len"]
            part_sentence = [clause["clause"] for clause in doc["clauses"]]
            
            mask_full_document = ""
            for i in range(1, d_len + 1):
                mask_full_document += f' {i} {part_sentence[i - 1]}[MASK] [MASK] [MASK] [SEP]'
            
            # 檢查長度
            count_len = len(tokenizer.encode_plus(mask_full_document, return_tensors="pt")['input_ids'][0])
            if count_len > 512:
                continue
            
            self.doc_id.append(doc_id)
            encoded = tokenizer.encode_plus(
                mask_full_document, 
                return_tensors="pt", 
                max_length=512, 
                truncation=True, 
                pad_to_max_length=True
            )['input_ids']
            self.x_bert.append(np.array(encoded[0]))
        
        self.x_bert = np.array(self.x_bert)
        print(f'  載入完成: {len(self.x_bert)} 筆')
    
    def __getitem__(self, index):
        return self.x_bert[index]
    
    def __len__(self):
        return len(self.x_bert)


def extract_embeddings(model, dataloader, device, knn_embedding_mode='cls'):
    """
    提取文檔 embedding 向量
    
    【處理流程】
    1. 將每個文檔輸入 BERT 模型
    2. 根據 knn_embedding_mode 決定如何提取 embedding:
       - 'cls': 直接使用 [CLS] token 的 hidden state (768維)
       - 'cause_clause': 找出被預測為「原因子句」的子句，做 mean pooling
       - 'cause_clause_mask': cause_clause + [MASK]_cause embedding 的平均
    
    【參數】
    model: 訓練好的 BERT 模型
    dataloader: 資料載入器
    device: 運算裝置 (CPU/GPU)
    knn_embedding_mode: embedding 提取模式
    
    【返回】
    numpy array of shape (N, 768), N = 文檔數量
    """
    model.eval()
    all_embeddings = []
    
    with torch.no_grad():
        for batch in dataloader:
            x_bert = batch.to(device)
            
            if knn_embedding_mode == 'cls':
                # [CLS] 模式：直接取第一個 token 的 hidden state
                embeddings = model.get_cls_embeddings(x_bert)
            else:
                # 子句模式：需要先做 forward pass 取得預測機率
                # probs 用於判斷每個子句是否為「情緒句」或「原因句」
                _, logits = model(x_bert, labels=None)
                probs = F.softmax(logits, dim=-1)
                embeddings = model.get_clause_embeddings(x_bert, probs, knn_embedding_mode)
            
            all_embeddings.append(embeddings.cpu().numpy())
    
    return np.vstack(all_embeddings) # vstack: 垂直堆疊


def load_nest_selection_info(json_path):
    """載入 NeST 選擇資訊"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    labeled_doc_ids = data['labeled_doc_ids']
    
    # 檢查是否含有 'samples' 欄位（舊版格式）或直接含有 selected/unselected 欄位（新版格式）
    if 'samples' in data:
        samples = data['samples']
        selected_doc_ids = [s['doc_id'] for s in samples if s['selected']]
        all_unlabeled_doc_ids = [s['doc_id'] for s in samples]
    elif 'selected_doc_ids' in data and 'unselected_doc_ids' in data:
        selected_doc_ids = data['selected_doc_ids']
        unselected_doc_ids = data['unselected_doc_ids']
        all_unlabeled_doc_ids = selected_doc_ids + unselected_doc_ids
    else:
        # 如果只有 selected_doc_ids
        selected_doc_ids = data.get('selected_doc_ids', [])
        # 如果沒有 unselected 資訊，嘗試從其他地方獲取或設為空
        all_unlabeled_doc_ids = selected_doc_ids + data.get('unselected_doc_ids', [])
        if not all_unlabeled_doc_ids:
             print("警告: 無法從 JSON 中獲取完整的 unlabeled doc ids")

    stats = data.get('statistics', {})
    if not stats: 
         # 嘗試重建基本統計
        stats = {
            'total_unlabeled': len(all_unlabeled_doc_ids), 
            'num_selected': len(selected_doc_ids)
        }
    
    print(f"NeST 統計: 總未標記={stats.get('total_unlabeled', 'N/A')}, 被選中={stats.get('num_selected', 'N/A')}")
    
    return {
        'labeled_doc_ids': labeled_doc_ids,
        'selected_doc_ids': selected_doc_ids,
        'all_unlabeled_doc_ids': all_unlabeled_doc_ids,
        'statistics': stats
    }


def load_pseudo_label_accuracy(experiment_dir, fold, round_num):
    """載入偽標籤準確度資訊"""
    pattern = os.path.join(experiment_dir, 'pseudo_results_*', f'pseudo_labeled_samples_fold{fold}_round{round_num}.json')
    matches = glob(pattern)
    if not matches:
        print(f"警告: 找不到偽標籤檔案 {pattern}")
        return None
    
    with open(matches[0], 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 建立 doc_id -> error_rate 對應
    accuracy_map = {}
    for sample in data:
        doc_id = sample['doc_id']
        # filtered_error_rate: 0.0 = 完全正確, 1.0 = 完全錯誤
        error_rate = sample.get('filtered_error_rate', None)
        if error_rate is not None:
            accuracy_map[doc_id] = error_rate
    
    n_correct = sum(1 for er in accuracy_map.values() if er == 0.0)
    n_partial = sum(1 for er in accuracy_map.values() if 0.0 < er < 1.0)
    n_wrong = sum(1 for er in accuracy_map.values() if er == 1.0)
    print(f"偽標籤準確度: 完全正確={n_correct}, 部分正確={n_partial}, 完全錯誤={n_wrong}")
    
    return accuracy_map


def get_initial_model_best_iter(experiment_dir, fold):
    """
    從 console output 中找出初始模型是第幾次迭代 (Epoch) 選出來的。
    原理：讀取 run_console_output_*.txt，根據 fold 切分區塊，找出 "新的最佳 F1: ... at iter X" 的最後一次紀錄。
    """
    # 尋找 run_console_output 檔案
    # 優先嘗試匹配資料夾的時間戳記
    timestamp_match = re.search(r"_(\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2})_", experiment_dir)
    if timestamp_match:
        timestamp = timestamp_match.group(1)
        log_file = os.path.join(experiment_dir, f"run_console_output_{timestamp}.txt")
    else:
        # Fallback: 找任何 run_console_output 開頭的檔案
        files = glob(os.path.join(experiment_dir, "run_console_output_*.txt"))
        # 排序取最新的
        files.sort(key=os.path.getmtime, reverse=True)
        log_file = files[0] if files else None

    if not log_file or not os.path.exists(log_file):
        print(f"找不到 Console Log 檔案: {log_file}")
        return None
        
    try:
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            
        # 切分 Fold 區塊
        # 尋找 "Fold {fold} 開始時間" 到 "Fold {fold+1} 開始時間" 之間的內容
        # 或是 "Fold {fold} 開始時間" 到 "UECA Self-Training 執行時間記錄" (如果有的話，通常在最前面)
        # 簡單起見，我們找 "Fold X 開始時間"
        
        fold_start_pattern = f"Fold {fold} 開始時間:"
        fold_next_pattern = f"Fold {fold + 1} 開始時間:"
        
        start_idx = content.find(fold_start_pattern)
        if start_idx == -1:
             print(f"在 Log 中找不到 Fold {fold} 的開始標記")
             return None
             
        end_idx = content.find(fold_next_pattern)
        if end_idx == -1:
            # 如果是最後一個 Fold，則取到最後
            fold_content = content[start_idx:]
        else:
            fold_content = content[start_idx:end_idx]
            
        # 在該 Fold 的內容中尋找 "新的最佳 F1: {f1} at iter {iter}"
        # 例如: 新的最佳 F1: 0.4230 at iter 22
        # 注意：可能有多個，我們要找最後一個 (因為那是最終的最佳模型)
        best_f1_pattern = re.compile(r"新的最佳 F1: ([\d\.]+) at iter (\d+)")
        matches = best_f1_pattern.findall(fold_content)
        
        if matches:
            # 取最後一個匹配項
            final_best_f1 = float(matches[-1][0])
            final_best_iter = int(matches[-1][1])
            print(f"初始模型分析 (Fold {fold}): 最佳 Pair F1 = {final_best_f1:.4f} @ Iter {final_best_iter}")
            return final_best_iter, final_best_f1
        else:
            print(f"Fold {fold} 中找不到 '新的最佳 F1' 紀錄")
            return None
            
    except Exception as e:
        print(f"讀取 Console Log 紀錄失敗: {e}")
        return None


def get_self_training_best_round(experiment_dir, fold):
    """
    從 console output 中找出 Self-Training 最佳模型是第幾輪 (Round) 選出來的。
    原理：讀取 run_console_output_*.txt，根據 fold 切分區塊，
    找出 "[Self-training round X] Validation: ... pair_f: Y.YYYY" 中最佳的一筆。
    
    Returns:
        tuple: (best_round, best_f1) 或 None
    """
    # 尋找 run_console_output 檔案
    timestamp_match = re.search(r"_(\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2})_", experiment_dir)
    if timestamp_match:
        timestamp = timestamp_match.group(1)
        log_file = os.path.join(experiment_dir, f"run_console_output_{timestamp}.txt")
    else:
        files = glob(os.path.join(experiment_dir, "run_console_output_*.txt"))
        files.sort(key=os.path.getmtime, reverse=True)
        log_file = files[0] if files else None

    if not log_file or not os.path.exists(log_file):
        print(f"找不到 Console Log 檔案: {log_file}")
        return None
        
    try:
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            
        # 切分 Fold 區塊
        fold_start_pattern = f"Fold {fold} 開始時間:"
        fold_next_pattern = f"Fold {fold + 1} 開始時間:"
        
        start_idx = content.find(fold_start_pattern)
        if start_idx == -1:
            print(f"在 Log 中找不到 Fold {fold} 的開始標記")
            return None
             
        end_idx = content.find(fold_next_pattern)
        if end_idx == -1:
            fold_content = content[start_idx:]
        else:
            fold_content = content[start_idx:end_idx]
            
        # 在該 Fold 的內容中尋找 Self-training 驗證結果
        # 格式: [Self-training round X] Validation: e_f: ... c_f: ... pair_f: Y.YYYY
        st_pattern = re.compile(r"\[Self-training round (\d+)\] Validation:.*?pair_f: ([\d\.]+)")
        matches = st_pattern.findall(fold_content)
        
        if matches:
            # 找出 pair_f 最高的 round
            best_round = None
            best_f1 = -1.0
            for round_str, f1_str in matches:
                f1 = float(f1_str)
                if f1 > best_f1:
                    best_f1 = f1
                    best_round = int(round_str)
            
            print(f"Self-Training 分析 (Fold {fold}): 最佳 Pair F1 = {best_f1:.4f} @ Round {best_round}")
            return best_round, best_f1
        else:
            print(f"Fold {fold} 中找不到 Self-training 驗證紀錄")
            return None
            
    except Exception as e:
        print(f"讀取 Console Log 紀錄失敗: {e}")
        return None


def plot_tsne(labeled_2d, selected_2d, unselected_2d, output_path, fold, round_num, stats, 
              experiment_dir=None):  # Add experiment_dir argument
    """繪製 t-SNE 視覺化圖"""
    plt.figure(figsize=(12, 10), facecolor='#DDDDDD')
    
    # 先畫灰色（未選中）- 最底層
    plt.scatter(unselected_2d[:, 0], unselected_2d[:, 1], 
                c='gray', alpha=0.3, s=15, label=f'未被選中 ({len(unselected_2d)})')
    
    # 再畫藍色（有標籤）
    plt.scatter(labeled_2d[:, 0], labeled_2d[:, 1], 
                c='blue', alpha=0.7, s=50, marker='s', label=f'有標籤樣本 ({len(labeled_2d)})')
    
    # 最後畫綠色（被選中）- 最上層
    plt.scatter(selected_2d[:, 0], selected_2d[:, 1], 
                c='green', alpha=0.8, s=25, label=f'NeST 選中 ({len(selected_2d)})')
    
    # 加入統計資訊
    # info_text = f"Divergence: min={stats['min']:.4f}, max={stats['max']:.4f}, mean={stats['mean']:.4f}"
    # plt.figtext(0.5, 0.02, info_text, ha='center', fontsize=9, style='italic')
    
    model_info = "Initial Model" if round_num == 1 else "Self-training Best Model"
    
    # Check for initial model iteration (Round 1)
    if round_num == 1 and experiment_dir:
        res = get_initial_model_best_iter(experiment_dir, fold)
        if res:
            best_iter, best_f1 = res
            model_info += f" (Iter {best_iter}, F1={best_f1:.4f})"
    # Check for self-training best round (Round >= 2)
    elif round_num >= 2 and experiment_dir:
        res = get_self_training_best_round(experiment_dir, fold)
        if res:
            best_round, best_f1 = res
            model_info += f" (Best Round {best_round}, F1={best_f1:.4f})"

    plt.suptitle(f't-SNE: NeST 樣本選擇分佈\n(Fold {fold}, Round {round_num}, Embedding from: {model_info})', fontsize=20, y=0.98)
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend(loc='upper right', fontsize=20, facecolor='#DDDDDD')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"圖片已儲存: {output_path}")


def plot_tsne_with_accuracy(labeled_2d, selected_2d, selected_doc_ids, unselected_2d, 
                            accuracy_map, output_path, fold, round_num, stats,
                            labeled_doc_ids=None, unselected_doc_ids=None, show_doc_ids=False,
                            experiment_dir=None): # Add experiment_dir argument
    """
    繪製 t-SNE 視覺化圖（包含偽標籤準確度顏色編碼）
    不同準確度使用不同顏色標記
    """
    from matplotlib.colors import LinearSegmentedColormap
    
    plt.figure(figsize=(14, 12) if show_doc_ids else (12, 10), facecolor='#DDDDDD')
    
    # 先畫灰色（未選中）- 最底層
    plt.scatter(unselected_2d[:, 0], unselected_2d[:, 1], 
                c='gray', alpha=0.3, s=15, label=f'未被選中 ({len(unselected_2d)})')
    
    # 再畫藍色（有標籤）
    plt.scatter(labeled_2d[:, 0], labeled_2d[:, 1], 
                c='blue', alpha=0.7, s=50, marker='s', label=f'有標籤樣本 ({len(labeled_2d)})')
    
    # 選中的樣本按準確度著色
    if len(selected_2d) > 0:
        # 取得每個選中樣本的 error_rate
        error_rates = []
        for doc_id in selected_doc_ids:
            er = accuracy_map.get(doc_id, 0.5)  # 預設 0.5 如果找不到
            error_rates.append(er)
        error_rates = np.array(error_rates)
        
        # 分類: 完全正確、部分正確、完全錯誤
        correct_mask = error_rates == 0.0
        partial_mask = (error_rates > 0.0) & (error_rates < 1.0)
        wrong_mask = error_rates >= 1.0
        
        # 計算數量
        n_correct = correct_mask.sum()
        n_partial = partial_mask.sum()
        n_wrong = wrong_mask.sum()
        
        # 繪製：綠色=完全正確
        if n_correct > 0:
            plt.scatter(selected_2d[correct_mask, 0], selected_2d[correct_mask, 1],
                       c='green', alpha=0.9, s=30, marker='o', 
                       label=f'完全正確 ({n_correct})', edgecolors='darkgreen', linewidths=0.5)
        
        # 繪製：黃/橙色=部分正確（使用漸層）
        if n_partial > 0:
            partial_colors = plt.cm.YlOrRd(error_rates[partial_mask] * 0.8)  # 0~0.8 範圍
            plt.scatter(selected_2d[partial_mask, 0], selected_2d[partial_mask, 1],
                       c=partial_colors, alpha=0.9, s=30, marker='o',
                       label=f'部分正確 ({n_partial})', edgecolors='orange', linewidths=0.5)
        
        # 繪製：紅色=完全錯誤
        if n_wrong > 0:
            plt.scatter(selected_2d[wrong_mask, 0], selected_2d[wrong_mask, 1],
                       c='red', alpha=0.9, s=35, marker='o',
                       label=f'完全錯誤 ({n_wrong})', edgecolors='darkred', linewidths=0.5)
        
        # 標記選中樣本的 doc_id
        if show_doc_ids:
            for i, doc_id in enumerate(selected_doc_ids):
                plt.annotate(doc_id, (selected_2d[i, 0], selected_2d[i, 1]),
                           fontsize=5, alpha=0.7, ha='center', va='bottom')
    
    # 標記有標籤樣本的 doc_id
    if show_doc_ids and labeled_doc_ids is not None:
        for i, doc_id in enumerate(labeled_doc_ids):
            plt.annotate(doc_id, (labeled_2d[i, 0], labeled_2d[i, 1]),
                       fontsize=5, alpha=0.5, ha='center', va='bottom', color='blue')
    
    model_info = "Initial Model" if round_num == 1 else "Self-training Best Model"
    
    # Check for initial model iteration (Round 1)
    if round_num == 1 and experiment_dir:
        res = get_initial_model_best_iter(experiment_dir, fold)
        if res:
            best_iter, best_f1 = res
            model_info += f" (Iter {best_iter}, F1={best_f1:.4f})"
    # Check for self-training best round (Round >= 2)
    elif round_num >= 2 and experiment_dir:
        res = get_self_training_best_round(experiment_dir, fold)
        if res:
            best_round, best_f1 = res
            model_info += f" (Best Round {best_round}, F1={best_f1:.4f})"
            
    plt.suptitle(f't-SNE: NeST 樣本選擇分佈\n(Fold {fold}, Round {round_num}, Embedding from: {model_info})', fontsize=20, y=0.98)
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    
    # 將圖例移到圖外右側，避免遮擋
    plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=20, borderaxespad=0, facecolor='#DDDDDD')
    
    # 加入統計資訊
    # info_text = f"Divergence: min={stats['min']:.4f}, max={stats['max']:.4f}, mean={stats['mean']:.4f}"
    # plt.figtext(0.5, 0.01, info_text, ha='center', fontsize=9, style='italic')
    
    plt.tight_layout(rect=[0, 0.03, 0.85, 0.95])  # 留空間給標題和圖例
    plt.savefig(output_path, dpi=200 if show_doc_ids else 150, bbox_inches='tight')
    plt.close()
    print(f"圖片已儲存 (含準確度): {output_path}")




def process_single(args, fold, round_num, device, tokenizer, dataset_base, output_dir):
    """處理單一 fold + round 組合"""
    print(f"\n{'='*60}")
    print(f"處理 Fold {fold}, Round {round_num}")
    print(f"{'='*60}")
    
    # 尋找 NeST JSON 檔案
    try:
        nest_json_path = find_nest_json(args.experiment_dir, fold, round_num)
    except FileNotFoundError as e:
        print(f"跳過: {e}")
        return False
    
    # 載入 NeST 選擇資訊
    nest_info = load_nest_selection_info(nest_json_path)
    
    # 載入模型
    if round_num == 1:
        model_path = os.path.join(args.experiment_dir, f'fold{fold}.pth')
    else:
        model_path = os.path.join(args.experiment_dir, 'self_training_models', f'fold{fold}_self_training_best.pth')
        if not os.path.exists(model_path):
            model_path = os.path.join(args.experiment_dir, f'fold{fold}.pth')
            print(f"警告: 找不到自訓練模型，改用初始模型")
    
    if not os.path.exists(model_path):
        print(f"跳過: 找不到模型 {model_path}")
        return False
    
    # 載入模型 (先載入到 CPU 再移到 GPU，參考 UECA_CE_few_shot_ST_nest.py)
    print(f"載入模型: {model_path}")
    model = torch.load(model_path, map_location=torch.device('cpu'))
    if device.type == 'cuda':
        model = model.cuda()
    model.eval()

    
    # 載入資料集
    train_file = os.path.join(dataset_base, f'fold{fold}_train.json')
    unlabeled_file = os.path.join(dataset_base, f'fold{fold}_unlabeled.json')
    
    if not os.path.exists(train_file) or not os.path.exists(unlabeled_file):
        print(f"跳過: 找不到資料集 fold{fold}")
        return False
    
    train_dataset = SimpleDataset(train_file, tokenizer)
    unlabeled_dataset = SimpleDataset(unlabeled_file, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=False)
    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=8, shuffle=False)
    
    # 提取 embeddings
    print("提取 labeled embeddings...")
    labeled_embeddings = extract_embeddings(model, train_loader, device, args.knn_embedding_mode)
    print(f"  形狀: {labeled_embeddings.shape}")
    
    print("提取 unlabeled embeddings...")
    unlabeled_embeddings = extract_embeddings(model, unlabeled_loader, device, args.knn_embedding_mode)
    print(f"  形狀: {unlabeled_embeddings.shape}")
    
    # ========================================================================
    # 執行 t-SNE 降維
    # ========================================================================
    # 將所有 embedding 合併後一起做 t-SNE，確保相同的降維空間
    all_embeddings = np.vstack([labeled_embeddings, unlabeled_embeddings])
    print(f"執行 t-SNE (perplexity={args.perplexity})...")
    
    # t-SNE 參數說明:
    # - n_components=2: 降到 2 維，可在平面上繪製
    # - perplexity: 每個點考慮的有效鄰居數，影響群聚粗細
    # - n_iter=1000: 迭代次數，足夠讓演算法收斂
    # - random_state=42: 固定隨機種子，確保結果可重現
    tsne = TSNE(n_components=2, perplexity=args.perplexity, random_state=42, n_iter=1000)
    all_2d = tsne.fit_transform(all_embeddings)  # 輸出: (N, 2)
    
    # ========================================================================
    # 分割 t-SNE 結果
    # ========================================================================
    n_labeled = len(labeled_embeddings)
    labeled_2d = all_2d[:n_labeled]
    unlabeled_2d = all_2d[n_labeled:]
    
    # 找出 selected / unselected 的索引
    selected_set = set(nest_info['selected_doc_ids'])
    selected_indices = [i for i, doc_id in enumerate(unlabeled_dataset.doc_id) if doc_id in selected_set]
    unselected_indices = [i for i, doc_id in enumerate(unlabeled_dataset.doc_id) if doc_id not in selected_set]
    
    # 取得選中樣本的 doc_id（按索引順序）
    selected_doc_ids_ordered = [unlabeled_dataset.doc_id[i] for i in selected_indices]
    
    # 取得有標籤樣本的 doc_id
    labeled_doc_ids = train_dataset.doc_id
    
    # 取得未選中樣本的 doc_id
    unselected_doc_ids = [unlabeled_dataset.doc_id[i] for i in unselected_indices]
    
    selected_2d = unlabeled_2d[selected_indices]
    unselected_2d = unlabeled_2d[unselected_indices]
    
    print(f"樣本數: labeled={len(labeled_2d)}, selected={len(selected_2d)}, unselected={len(unselected_2d)}")
    
    # ========================================================================
    # 將 t-SNE 座標和 doc_ids 儲存至 NeST JSON（用於後續分析）
    # ========================================================================
    nest_info['tsne_2d'] = all_2d.tolist()  # 所有樣本的 t-SNE 座標
    nest_info['labeled_doc_ids'] = labeled_doc_ids
    nest_info['unselected_doc_ids'] = unselected_doc_ids
    
    # 儲存更新後的 NeST JSON
    with open(nest_json_path, 'w', encoding='utf-8') as f:
        json.dump(nest_info, f, ensure_ascii=False, indent=2)
    print(f"已儲存 t-SNE 座標至: {nest_json_path}")
    
    # 決定輸出檔名與繪圖方式
    cls_suffix = '_cls' if args.knn_embedding_mode == 'cls' else ''
    
    if args.show_accuracy:
        # 載入偽標籤準確度
        accuracy_map = load_pseudo_label_accuracy(args.experiment_dir, fold, round_num)
        if accuracy_map:
            suffix = '_docids' if args.show_doc_ids else ''
            output_path = os.path.join(output_dir, f'tsne_accuracy_round{round_num}_fold{fold}{cls_suffix}{suffix}.png')
            plot_tsne_with_accuracy(
                labeled_2d, selected_2d, selected_doc_ids_ordered, unselected_2d,
                accuracy_map, output_path, fold, round_num, nest_info['statistics'],
                labeled_doc_ids=labeled_doc_ids, show_doc_ids=args.show_doc_ids,
                experiment_dir=args.experiment_dir # Pass experiment_dir
            )
        else:
            print("警告: 無法載入準確度資訊，使用一般繪圖")
            output_path = os.path.join(output_dir, f'tsne_round{round_num}_fold{fold}{cls_suffix}.png')
            plot_tsne(labeled_2d, selected_2d, unselected_2d, output_path, fold, round_num, nest_info['statistics'],
                 experiment_dir=args.experiment_dir) # Pass experiment_dir
    else:
        output_path = os.path.join(output_dir, f'tsne_round{round_num}_fold{fold}{cls_suffix}.png')
        plot_tsne(labeled_2d, selected_2d, unselected_2d, output_path, fold, round_num, nest_info['statistics'],
                 experiment_dir=args.experiment_dir) # Pass experiment_dir
    
    # 釋放 GPU 記憶體
    del model
    torch.cuda.empty_cache()
    
    return True


def main():
    args = parse_args()
    
    # 設定裝置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用裝置: {device}")
    
    # 偵測 embedding mode
    if args.knn_embedding_mode is None:
        folder_name = os.path.basename(args.experiment_dir)
        args.knn_embedding_mode = detect_embedding_mode_from_folder(folder_name)
    print(f"Embedding 模式: {args.knn_embedding_mode}")
    
    # 推斷資料集路徑
    dataset_base = find_dataset_path(args.experiment_dir)
    
    # 載入 tokenizer (使用本地 BERT 模型路徑，與 UECA_CE_few_shot_ST_nest.py 一致)
    from transformers import BertTokenizer
    bert_path = './bert-base-chinese'
    tokenizer = BertTokenizer.from_pretrained(bert_path)
    
    # 設定輸出路徑
    if args.output_dir is None:
        output_dir = os.path.join(args.experiment_dir, 'tsne_analysis')
    else:
        output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # 決定要處理的 folds 和 rounds
    folds = list(range(1, 11)) if args.all_folds else [args.fold]
    rounds = list(range(1, 11)) if args.all_rounds else [args.round]
    
    print(f"將處理: {len(folds)} 個 folds × {len(rounds)} 個 rounds = {len(folds) * len(rounds)} 張圖")
    
    success_count = 0
    for fold in folds:
        for round_num in rounds:
            if process_single(args, fold, round_num, device, tokenizer, dataset_base, output_dir):
                success_count += 1
    
    print(f"\n{'='*60}")
    print(f"完成！成功產生 {success_count} 張圖片，儲存於: {output_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
