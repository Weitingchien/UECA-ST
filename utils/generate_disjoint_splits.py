#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
將 ECPE 格式的 txt 資料檔轉換為 10 折「不重疊輪替切分」資料集

此腳本會將全部資料一次性切分為 n_splits 個不重疊的區塊 (Blocks)，
然後透過 10 次輪替 (rotation) 來指派 Train / Val / Test / Unlabeled 角色

這可以確保在不同折 (fold)之間的 Train, Val, Test 集合是完全不重疊的

使用方式:
    python utils/generate_disjoint_splits.py --input data/ECPE_new_dataset/home.txt --output split10_home_train1_test1_val1_unlabeled7_disjoint/ --train_ratio 0.1 --val_ratio 0.1 --test_ratio 0.1 --unlabeled_ratio 0.7 --n_splits 10 --seed 42
"""
import os
import re
import json
import argparse
import numpy as np

def convert_to_unlabeled(doc):
    """建立未標籤資料輸出，目前保留完整標籤資訊供後續處理使用"""
    return json.loads(json.dumps(doc, ensure_ascii=False))


def flatten_blocks(blocks_list):
    """將多個 block 展平成單一文件列表"""
    if len(blocks_list) == 0:
        return []
    return np.concatenate(blocks_list).tolist()

# verbose=True 開啟詳細輸出
def build_disjoint_blocks(documents, n_splits=10, random_state=42, verbose=True):
    """將完整 documents 打亂後切成 n_splits 個不重疊 blocks"""
    documents_array = np.array(documents, dtype=object)
    # 步驟 1:隨機打亂索引
    rng = np.random.default_rng(random_state)
    indices = np.arange(len(documents_array))
    rng.shuffle(indices)
    # 步驟 2: 將打亂後的索引切分為n_splits個區塊，均分成10個小陣列
    # np.array_split 會處理總數無法被 n_splits整除的情況， 多出來的那一筆放在最前面的幾個區塊中
    block_indices_list = np.array_split(indices, n_splits)
    if verbose:
        print(f"block_indices_list: {[len(block_indices) for block_indices in block_indices_list]}")

    all_blocks = np.array(
        [documents_array[block_indices] for block_indices in block_indices_list],
        dtype=object,
    )
    if verbose:
        print(f"all_blocks: {[len(block) for block in all_blocks]}")

    return all_blocks


def get_rotated_blocks(all_blocks, fold_index, verbose=True):
    """取得指定 fold 的輪替後 blocks。"""
    rotated_blocks = np.roll(all_blocks, -fold_index, axis=0)
    if verbose:
        print(f"rotated_blocks (Fold {fold_index + 1}): {[len(block) for block in rotated_blocks]}")
    return rotated_blocks


def parse_ecpe_txt(file_path):
    """解析 ECPE 格式的 txt 檔案"""
    documents = [] # 用來儲存所有文檔
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    current_doc = [] # 暫存目前正在處理的文檔內容
    for line in lines:
        line = line.strip()
        
        # 檢查是否為新文檔的開始 (格式：doc_id doc_len)
        parts = line.split()
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit(): # isdigit() 檢查是否為數字
            # 處理前一個文檔
            if current_doc:
                doc = process_document(current_doc)
                if doc:
                    documents.append(doc)
            
            # 開始新文檔
            current_doc = [line]
        else:
            # 添加到當前文檔
            current_doc.append(line)
    
    # 處理最後一個文檔
    if current_doc:
        doc = process_document(current_doc)
        if doc:
            documents.append(doc)
    
    print(f"成功解析 {len(documents)} 個文檔")
    return documents


def process_document(doc_lines):
    """處理單一文檔,轉換為 JSON 格式"""
    if len(doc_lines) < 2:
        return None
    
    # 情緒 ID 到英文名稱的對應表
    emotion_id_to_name = {
        0: 'happiness',   # 快樂
        1: 'sadness',     # 悲傷  
        2: 'disgust',     # 厭惡
        3: 'surprise',    # 驚訝
        4: 'fear',        # 恐懼
        5: 'anger',       # 憤怒
        6: 'null'         # 無情緒
    }
    
    # 第一行:doc_id 和 doc_len
    first_line = doc_lines[0].strip()
    parts = first_line.split()
    if len(parts) < 2:
        return None
    
    doc_id = parts[0]
    doc_len = int(parts[1])
    
    # 第二行可能是情緒-原因配對 (emotion, cause)
    pairs = []
    sentence_start_idx = 1
    
    if len(doc_lines) > 1 and doc_lines[1].strip().startswith('('):
        pair_line = doc_lines[1].strip()
        # 匹配所有的 (數字,數字) 格式
        pair_matches = re.findall(r'\((\d+),(\d+)\)', pair_line) # re.findall: 找出所有符合模式的項目，\在正則有特殊意義需要用\來找真正的括號符號，(\d+)表示匹配一個或多個數字並分組
        pairs = [[int(emo), int(cause)] for emo, cause in pair_matches]
        sentence_start_idx = 2
    
    # 解析每個子句
    clauses = []
    for line_idx in range(sentence_start_idx, len(doc_lines)): # sentence_start_idx表示從哪行開始遍歷，到最後一行len(doc_lines)
        line = doc_lines[line_idx].strip()
        if not line or line.startswith('('): # 非空字串或是以(開頭則跳過
            continue
        
        # 格式: clause_id, emotion_category_id, emotion_token, clause_text
        parts = line.split(',', 3)
        if len(parts) >= 4:
            try:
                clause_id = int(parts[0])
                emotion_category_id = int(parts[1])  # 數字 ID (0-6)
                emotion_token = parts[2].strip()  # 中文情緒詞 (例如: "怨恨", "激動") 或數字 "6"
                clause_text = parts[3].replace(' ', '')  # 移除所有空白
                
                # 將數字 ID 轉換為英文情緒名稱
                emotion_category = emotion_id_to_name.get(emotion_category_id, 'null')
                
                # 如果 emotion_token 是純數字 (如 "1", "2", "5", "6"),轉換為 "null"
                if emotion_token.isdigit():
                    emotion_token = 'null'
                
                clauses.append({
                    "clause_id": str(clause_id),  # 字串格式
                    "emotion_category": emotion_category,  # 英文名稱 ("happiness", "sadness", 等)
                    "emotion_token": emotion_token,  # 中文情緒詞或 "null"
                    "clause": clause_text
                })
            except ValueError:
                continue
    
    if len(clauses) != doc_len:
        print(f"警告: 文檔 {doc_id} 的子句數量 ({len(clauses)}) 與聲明的長度 ({doc_len}) 不符")
    
    return {
        "doc_id": doc_id,
        "doc_len": len(clauses),
        "pairs": pairs,
        "clauses": clauses
    }


def create_disjoint_rotated_splits(
        documents,
        n_splits=10,
        output_dir='split10/',
        random_state=42,
        train_blocks=1,
        val_blocks=1,
        test_blocks=1,
        unlabeled_blocks=7):
    """
    建立 K-fold 輪替切分 (Disjoint Rotated Splits)
    
    1. 將所有資料隨機打亂，並切分為 n_splits 個不重疊的區塊 (Blocks)。
    2. 執行 n_splits 次迴圈 (folds)。
    3. 在每一折中，旋轉 (rotate) 這些區塊，並根據指定的區塊數量
       分配給 Train, Val, Test, Unlabeled。
    """
    
    # 確保輸出資料夾存在
    os.makedirs(output_dir, exist_ok=True)
    
    all_blocks = build_disjoint_blocks(
        documents,
        n_splits=n_splits,
        random_state=random_state,
        verbose=True,
    )

    # 步驟 3: 執行 n_splits 次迴圈 (Folds)
    for fold in range(n_splits):
        fold_num = fold + 1
        
        # 旋轉區塊列表 (np.roll)
        # 第 0 次 (Fold 1) 不旋轉, 第 1 次 (Fold 2) 向左旋轉 1 位, ...
        rotated_blocks = get_rotated_blocks(all_blocks, fold, verbose=True)
        # np.roll(all_blocks, -0, axis=0)  # fold=0 → [B0, B1, B2, ..., B9]  (不動)
        # np.roll(all_blocks, -1, axis=0)  # fold=1 → [B1, B2, B3, ..., B0]  (左移1位)
        # np.roll(all_blocks, -2, axis=0)  # fold=2 → [B2, B3, B4, ..., B1]  (左移2位)
        
        # 根據區塊數量分配角色
        current_idx = 0
        train_blocks_list = rotated_blocks[current_idx : current_idx + train_blocks]
        # rotated_blocks[0:1] -> 取 [rotated_blocks[0]]，下一輪就會從B1開始，因為rotated_blocks左移1位
        print(f"train_blocks_list: {[len(block) for block in train_blocks_list]}")
        current_idx += train_blocks
        
        val_blocks_list = rotated_blocks[current_idx : current_idx + val_blocks]
        current_idx += val_blocks
        
        test_blocks_list = rotated_blocks[current_idx : current_idx + test_blocks]
        current_idx += test_blocks
        
        unlabeled_blocks_list = rotated_blocks[current_idx : current_idx + unlabeled_blocks]
        
        # 展平 (flatten) 區塊列表
        train_subset = flatten_blocks(train_blocks_list)
        val_subset = flatten_blocks(val_blocks_list)
        test_subset = flatten_blocks(test_blocks_list)
        unlabeled_subset = flatten_blocks(unlabeled_blocks_list)
        
        # 處理未標籤資料 (目前只是複製)
        unlabeled_processed = [convert_to_unlabeled(doc) for doc in unlabeled_subset] if unlabeled_subset else []

        # 定義檔案路徑
        train_file = os.path.join(output_dir, f'fold{fold_num}_train.json')
        val_file = os.path.join(output_dir, f'fold{fold_num}_val.json')
        test_file = os.path.join(output_dir, f'fold{fold_num}_test.json')
        unlabeled_file = os.path.join(output_dir, f'fold{fold_num}_unlabeled.json')

        # 寫出檔案 (若集合為空則刪除既有檔案)
        
        # 寫出訓練集
        if train_subset:
            with open(train_file, 'w', encoding='utf-8') as f:
                json.dump(train_subset, f, ensure_ascii=False, indent=2)
        elif os.path.exists(train_file):
            os.remove(train_file)
            
        # 寫出驗證集
        if val_subset:
            with open(val_file, 'w', encoding='utf-8') as f:
                json.dump(val_subset, f, ensure_ascii=False, indent=2)
        elif os.path.exists(val_file):
            os.remove(val_file)

        # 寫出測試集
        if test_subset:
            with open(test_file, 'w', encoding='utf-8') as f:
                json.dump(test_subset, f, ensure_ascii=False, indent=2)
        elif os.path.exists(test_file):
            os.remove(test_file)

        # 寫出未標籤資料
        if unlabeled_processed:
            with open(unlabeled_file, 'w', encoding='utf-8') as f:
                json.dump(unlabeled_processed, f, ensure_ascii=False, indent=2)
        elif os.path.exists(unlabeled_file):
            os.remove(unlabeled_file)

        # 印出當前折的統計資訊
        total_docs_count = len(documents)
        print(f"Fold {fold_num} 統計：")
        print(f"  訓練集 (Train): {len(train_subset)} 筆 (~{len(train_subset) / max(1, total_docs_count) * 100:.1f}%)")
        print(f"  驗證集 (Val):   {len(val_subset)} 筆 (~{len(val_subset) / max(1, total_docs_count) * 100:.1f}%)")
        print(f"  測試集 (Test):  {len(test_subset)} 筆 (~{len(test_subset) / max(1, total_docs_count) * 100:.1f}%)")
        print(f"  未標籤 (Unlabeled): {len(unlabeled_subset)} 筆 (~{len(unlabeled_subset) / max(1, total_docs_count) * 100:.1f}%)")

    print(f"\n所有 {n_splits} 折「不重疊輪替切分」已儲存至: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='將 ECPE txt 檔案轉換為 10 折「不重疊輪替切分」的 JSON 格式')
    parser.add_argument('--input', type=str, default='data/ECPE_new_dataset/home.txt',
                        help='輸入的 txt 檔案路徑')
    parser.add_argument('--output', type=str, default='split10_home_disjoint/',
                        help='輸出資料夾路徑')
    parser.add_argument('--n_splits', type=int, default=10,
                        help='交叉驗證的折數 (預設: 10)')
    parser.add_argument('--seed', type=int, default=42,
                        help='隨機種子 (預設: 42)')
    parser.add_argument('--train_ratio', type=float, default=0.1,
                        help='標註訓練集比例 (預設: 0.1)')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='驗證集比例 (預設: 0.1)')
    parser.add_argument('--test_ratio', type=float, default=0.1,
                        help='測試集比例 (預設: 0.1)')
    parser.add_argument('--unlabeled_ratio', type=float, default=0.7,
                        help='未標籤資料比例 (預設: 0.7)')
    
    args = parser.parse_args()
    
    # 檢查1: 確保所有比例總和為 1.0
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio + args.unlabeled_ratio
    if not np.isclose(total_ratio, 1.0, atol=1e-6):
        parser.error('train_ratio + val_ratio + test_ratio + unlabeled_ratio 必須等於 1.0')

    # 檢查2: 確保 ratio 乘以 n_splits 必須是整數 (即 10% 對應 1 塊, 70% 對應 7 塊)
    train_blocks = args.n_splits * args.train_ratio
    val_blocks = args.n_splits * args.val_ratio
    test_blocks = args.n_splits * args.test_ratio
    unlabeled_blocks = args.n_splits * args.unlabeled_ratio
    
    block_counts = [train_blocks, val_blocks, test_blocks, unlabeled_blocks]
    
    # 使用 np.isclose 避免浮點數精度問題
    # round(x): 把每個數字四捨五入到最近的整數
    # np.isclose(x, round(x)): 檢查 x 是否接近其四捨五入的整數值(例如1.0000000001 會視為接近 1)
    # if not all: 若不是全部通過，進入報錯區塊
    if not all(np.isclose(x, round(x)) for x in block_counts):
        parser.error(f"錯誤：所有 ratio 乘以 n_splits ({args.n_splits}) 都必須是整數。\n"
                     f"  Train: {train_blocks}, Val: {val_blocks}, Test: {test_blocks}, Unlabeled: {unlabeled_blocks}\n"
                     f"  請檢查您的 ratio (例如，n_splits=10 時，ratio 應為 0.1, 0.2, 0.3... 等)")
    
    # 將區塊計數轉為整數
    train_blocks_int = int(round(train_blocks))
    val_blocks_int = int(round(val_blocks))
    test_blocks_int = int(round(test_blocks))
    unlabeled_blocks_int = int(round(unlabeled_blocks))

    # 檢查總區塊數是否等於 n_splits
    total_blocks = train_blocks_int + val_blocks_int + test_blocks_int + unlabeled_blocks_int
    if total_blocks != args.n_splits:
         parser.error(f"錯誤：計算出的總區塊數 ({total_blocks}) 與 n_splits ({args.n_splits}) 不符。")


    print(f"讀取檔案: {args.input}")
    documents = parse_ecpe_txt(args.input)
    
    print(f"\n開始建立 {args.n_splits} 折「不重疊輪替切分」...")
    print(f"分配: Train={train_blocks_int} 塊, Val={val_blocks_int} 塊, Test={test_blocks_int} 塊, Unlabeled={unlabeled_blocks_int} 塊")
    
    create_disjoint_rotated_splits(
        documents, 
        n_splits=args.n_splits, 
        output_dir=args.output, 
        random_state=args.seed,
        train_blocks=train_blocks_int,
        val_blocks=val_blocks_int,
        test_blocks=test_blocks_int,
        unlabeled_blocks=unlabeled_blocks_int
    )
    
    print("\n完成!")


if __name__ == '__main__':
    main()