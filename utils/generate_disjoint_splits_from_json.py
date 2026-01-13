#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
將現有 JSON 格式資料集（每折獨立）轉換為 few-shot + unlabeled 格式

此腳本針對每一折的 train.json 獨立進行分割：
- 每一折的 train (約 1750 筆) → 分割為 train (10%) + val (10%) + unlabeled (80%)
- test.json 直接從原始資料夾複製

使用方式:
    python utils/generate_disjoint_splits_from_json.py \
        --input-dir split10 \
        --output-dir split10_fewshot/ \
        --train_ratio 0.1 \
        --val_ratio 0.1 \
        --unlabeled_ratio 0.8 \
        --n_splits 10 \
        --seed 42

注意：此腳本不處理 test_ratio，測試集直接從原始 fold{i}_test.json 複製
"""
import os
import json
import argparse
import numpy as np


def load_json(file_path):
    """載入 JSON 檔案"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data, file_path):
    """儲存 JSON 檔案"""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def convert_to_unlabeled(doc):
    """建立未標籤資料輸出，保留完整標籤資訊供後續處理使用。"""
    return json.loads(json.dumps(doc, ensure_ascii=False))


def split_each_fold(
        input_dir,
        output_dir,
        n_splits=10,
        train_ratio=0.1,
        val_ratio=0.1,
        unlabeled_ratio=0.8,
        random_state=42):
    """
    對每一折的 train.json 獨立進行分割
    
    Args:
        input_dir: 輸入資料夾 (包含 fold{i}_train.json 和 fold{i}_test.json)
        output_dir: 輸出資料夾
        n_splits: 折數
        train_ratio: 有標籤訓練集比例
        val_ratio: 驗證集比例
        unlabeled_ratio: 未標籤資料比例
        random_state: 隨機種子
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for fold in range(1, n_splits + 1):
        # 讀取該折的 train.json
        train_file = os.path.join(input_dir, f'fold{fold}_train.json')
        test_file = os.path.join(input_dir, f'fold{fold}_test.json')
        
        if not os.path.exists(train_file):
            print(f"警告: {train_file} 不存在，跳過 fold {fold}")
            continue
        
        documents = load_json(train_file)
        n_total = len(documents)
        
        # 計算各部分的數量
        n_train = max(1, int(round(n_total * train_ratio)))
        n_val = max(1, int(round(n_total * val_ratio)))
        n_unlabeled = n_total - n_train - n_val
        
        # 轉為 numpy array
        documents_array = np.array(documents, dtype=object)
        
        # 使用固定種子 + fold 編號，確保每折的隨機打亂是確定性的
        rng = np.random.default_rng(random_state + fold)
        indices = np.arange(n_total)
        rng.shuffle(indices)
        
        # 分割
        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train + n_val]
        unlabeled_indices = indices[n_train + n_val:]
        
        train_subset = documents_array[train_indices].tolist()
        val_subset = documents_array[val_indices].tolist()
        unlabeled_subset = documents_array[unlabeled_indices].tolist()
        
        # 處理未標籤資料
        unlabeled_processed = [convert_to_unlabeled(doc) for doc in unlabeled_subset]
        
        # 儲存檔案
        save_json(train_subset, os.path.join(output_dir, f'fold{fold}_train.json'))
        save_json(val_subset, os.path.join(output_dir, f'fold{fold}_val.json'))
        save_json(unlabeled_processed, os.path.join(output_dir, f'fold{fold}_unlabeled.json'))
        
        # 複製 test.json
        if os.path.exists(test_file):
            test_data = load_json(test_file)
            save_json(test_data, os.path.join(output_dir, f'fold{fold}_test.json'))
            test_count = len(test_data)
        else:
            test_count = 0
        
        # 印出統計
        print(f"Fold {fold} 統計 (原始 train={n_total}):")
        print(f"  訓練集 (Train): {len(train_subset)} 筆 (~{len(train_subset) / n_total * 100:.1f}%)")
        print(f"  驗證集 (Val):   {len(val_subset)} 筆 (~{len(val_subset) / n_total * 100:.1f}%)")
        print(f"  未標籤 (Unlabeled): {len(unlabeled_subset)} 筆 (~{len(unlabeled_subset) / n_total * 100:.1f}%)")
        print(f"  測試集 (Test):  {test_count} 筆 (直接複製)")
    
    print(f"\n所有 {n_splits} 折已儲存至: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='將每一折的 train.json 獨立分割為 few-shot + unlabeled 格式'
    )
    parser.add_argument('--input-dir', type=str, required=True,
                        help='輸入資料夾路徑 (包含 fold{i}_train.json 和 fold{i}_test.json)')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='輸出資料夾路徑')
    parser.add_argument('--n_splits', type=int, default=10,
                        help='折數 (預設: 10)')
    parser.add_argument('--seed', type=int, default=42,
                        help='隨機種子 (預設: 42)')
    parser.add_argument('--train_ratio', type=float, default=0.1,
                        help='有標籤訓練集比例 (預設: 0.1)')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='驗證集比例 (預設: 0.1)')
    parser.add_argument('--unlabeled_ratio', type=float, default=0.8,
                        help='未標籤資料比例 (預設: 0.8)')
    
    args = parser.parse_args()
    
    # 檢查比例總和
    total_ratio = args.train_ratio + args.val_ratio + args.unlabeled_ratio
    if not np.isclose(total_ratio, 1.0, atol=1e-6):
        parser.error(f'train_ratio + val_ratio + unlabeled_ratio 必須等於 1.0，目前為 {total_ratio}')
    
    print(f"讀取資料夾: {args.input_dir}")
    print(f"分割比例: train={args.train_ratio}, val={args.val_ratio}, unlabeled={args.unlabeled_ratio}")
    print()
    
    split_each_fold(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        n_splits=args.n_splits,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        unlabeled_ratio=args.unlabeled_ratio,
        random_state=args.seed
    )
    
    print("\n完成!")


if __name__ == '__main__':
    main()