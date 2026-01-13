#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
將 ECPE 格式的 txt 資料檔轉換為 10 折交叉驗證分割，並同時產出指定比例的
訓練 / 驗證 / 測試 / 未標籤資料集。

使用方式:
    python prepare_dataset_splits.py \
        --input data/ECPE_new_dataset/home.txt \
        --output split10_home/ \
        --train_ratio 0.1 --val_ratio 0.1 --test_ratio 0.1 --unlabeled_ratio 0.7
"""
import os
import re
import json
import argparse
import numpy as np
from sklearn.model_selection import KFold


def convert_to_unlabeled(doc):
    """建立未標籤資料輸出，目前保留完整標籤資訊供後續處理使用。"""
    return json.loads(json.dumps(doc, ensure_ascii=False))



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


def create_kfold_splits(
        documents,
        n_splits=10,
        output_dir='split10/',
        random_state=42,
        train_ratio=0.1,
        val_ratio=0.1,
        test_ratio=0.1,
        unlabeled_ratio=0.7):
    """建立 K-fold 交叉驗證分割並儲存為 JSON 檔案"""
    # 確保輸出資料夾存在，沒有就建立
    os.makedirs(output_dir, exist_ok=True)
    
    # 使用 KFold 進行分割
    # 建立 KFold 物件，設定折數、是否打亂以及隨機種子
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    # 產生連續的整數索引，對應每篇文檔
    doc_indices = np.arange(len(documents))
    
    # 初始化折數計數器，從第 1 折開始
    fold = 1
    # 逐折取得訓練與測試索引
    for train_idx, test_idx in kf.split(doc_indices):
        # 準備訓練集和測試集
        # 根據訓練索引擷取文檔
        train_docs = [documents[i] for i in train_idx]
        # 根據測試索引擷取文檔
        test_docs = [documents[i] for i in test_idx]
        
        # 儲存原始的訓練 / 測試集 JSON 檔案
        # 組出當前折訓練集檔案路徑
        train_file = os.path.join(output_dir, f'fold{fold}_train.json')
        # 組出當前折測試集檔案路徑
        test_file = os.path.join(output_dir, f'fold{fold}_test.json')
        
        # 寫入完整訓練集資料
        with open(train_file, 'w', encoding='utf-8') as f:
            json.dump(train_docs, f, ensure_ascii=False, indent=2)
        
        # 寫入完整測試集資料
        with open(test_file, 'w', encoding='utf-8') as f:
            json.dump(test_docs, f, ensure_ascii=False, indent=2)
        
        # 依據指定比例切分 train_docs
        # 使用 numpy 的隨機數產生器確保可重現的打亂順序
        rng = np.random.default_rng(random_state + fold)
        # 取得打亂後的訓練索引序列
        shuffled_indices = rng.permutation(len(train_docs))

        # 計算全集文檔數量（train+test）
        total_docs = len(documents)
        # 目標訓練樣本數（四捨五入後取整）
        target_train = max(0, int(round(total_docs * train_ratio)))
        # 目標驗證樣本數
        target_val = max(0, int(round(total_docs * val_ratio)))
        # 目標未標籤樣本數
        target_unlabeled = max(0, int(round(total_docs * unlabeled_ratio)))


        # 訓練資料當前可用的總數量
        available_for_split = len(train_docs)
        # 實際可分配的訓練樣本數
        train_size = min(target_train, available_for_split)
        remaining = available_for_split - train_size
        # 實際可分配的驗證樣本數
        val_size = min(target_val, remaining)
        remaining -= val_size
        # 預期的未標籤樣本數（可能因剩餘不足而縮減）
        unlabeled_size = min(target_unlabeled, remaining)
        remaining -= unlabeled_size
        # 若仍有餘數，就回補到訓練集，避免產生意料外的未標籤或驗證樣本
        if remaining > 0:
            train_size += remaining
        """
        if unlabeled_size < 0:
            # 若剩餘為負，代表資料不足，先從驗證集回補
            val_size = max(0, val_size + unlabeled_size)
            unlabeled_size = 0
            if val_size < 0:
                # 若驗證仍不足，再回補給訓練集
                train_size = max(0, train_size + val_size)
                val_size = 0
        """

        # 依照打亂後的順序擷取訓練樣本
        train_subset = [train_docs[i] for i in shuffled_indices[:train_size]]
        # 擷取驗證樣本區間
        val_subset = [train_docs[i] for i in shuffled_indices[train_size:train_size + val_size]] if val_size else []
        # 剩餘部分視為未標籤樣本
        unlabeled_start = train_size + val_size
        unlabeled_subset = [train_docs[i] for i in shuffled_indices[unlabeled_start:unlabeled_start + unlabeled_size]] if unlabeled_size else []

        # 未標籤資料目前保留標籤資訊，後續可自行處理
        # 建立未標籤資料清單，目前保留完整資訊
        unlabeled_processed = [convert_to_unlabeled(doc) for doc in unlabeled_subset] if unlabeled_subset else []

        # 建立少量訓練資料檔案路徑
        train_small_file = os.path.join(output_dir, f'fold{fold}_train.json')
        # 建立驗證資料檔案路徑
        val_file = os.path.join(output_dir, f'fold{fold}_val.json')
        # 建立未標籤資料檔案路徑
        unlabeled_file = os.path.join(output_dir, f'fold{fold}_unlabeled.json')

        # 寫出少量訓練資料
        with open(train_small_file, 'w', encoding='utf-8') as f:
            json.dump(train_subset, f, ensure_ascii=False, indent=2)

        # 寫出驗證資料 (若集合為空則刪除既有檔案)
        if val_subset:
            with open(val_file, 'w', encoding='utf-8') as f:
                json.dump(val_subset, f, ensure_ascii=False, indent=2)
        elif os.path.exists(val_file):
            os.remove(val_file)

        # 寫出未標籤資料 (若集合為空則刪除既有檔案)
        if unlabeled_processed:
            with open(unlabeled_file, 'w', encoding='utf-8') as f:
                json.dump(unlabeled_processed, f, ensure_ascii=False, indent=2)
        elif os.path.exists(unlabeled_file):
            os.remove(unlabeled_file)

        # 印出當前折的統計資訊，方便檢查比例
        print(f"Fold {fold} 統計：")
        print(f"  訓練集(原始): {len(train_docs)} 筆")
        print(f"  測試集: {len(test_docs)} 筆")
        print(f"  少量標訓練: {len(train_subset)} 筆 (~{len(train_subset) / max(1, total_docs) * 100:.1f}%)")
        print(f"  驗證集: {len(val_subset)} 筆 (~{len(val_subset) / max(1, total_docs) * 100:.1f}%)")
        print(f"  未標籤集: {len(unlabeled_subset)} 筆 (~{len(unlabeled_subset) / max(1, total_docs) * 100:.1f}%)")
        # 折數加一，準備處理下一折
        fold += 1
    
    # 全部折數處理完畢後輸出完成訊息
    print(f"\n所有分割已儲存至: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='將 ECPE txt 檔案轉換為 10 折交叉驗證的 JSON 格式')
    parser.add_argument('--input', type=str, default='data/ECPE_new_dataset/home.txt',
                        help='輸入的 txt 檔案路徑')
    parser.add_argument('--output', type=str, default='split10_home/',
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
    
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio + args.unlabeled_ratio
    if not np.isclose(total_ratio, 1.0, atol=1e-6):
        parser.error('train_ratio + val_ratio + test_ratio + unlabeled_ratio 必須等於 1.0')

    expected_test_ratio = 1.0 / args.n_splits
    if not np.isclose(args.test_ratio, expected_test_ratio, atol=1e-3):
        print(f"警告: test_ratio={args.test_ratio} 與 1/n_splits={expected_test_ratio:.3f} 不一致，"
              "實際測試比例將以 KFold 結果為準。")

    print(f"讀取檔案: {args.input}")
    documents = parse_ecpe_txt(args.input)
    
    print(f"\n開始建立 {args.n_splits} 折交叉驗證分割...")
    create_kfold_splits(documents, n_splits=args.n_splits, 
                       output_dir=args.output, random_state=args.seed,
                       train_ratio=args.train_ratio, val_ratio=args.val_ratio,
                       test_ratio=args.test_ratio, unlabeled_ratio=args.unlabeled_ratio)
    
    print("\n完成!")


if __name__ == '__main__':
    main()
