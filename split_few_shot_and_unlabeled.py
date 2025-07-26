import os
import json
import random
import shutil

SRC_DIR = 'split10'
DST_DIR = 'split10_few_shot_st_with_val'  # 新的目錄名稱，避免覆蓋原有檔案
TRAIN_PREFIX = 'fold{}_train.json'
TEST_PREFIX = 'fold{}_test.json'
TRAIN_LABELED_PREFIX = 'fold{}_train_few_shot.json'
VAL_PREFIX = 'fold{}_val.json'
UNLABELED_PREFIX = 'fold{}_unlabeled.json'

os.makedirs(DST_DIR, exist_ok=True)

for fold in range(1, 11):
    train_path = os.path.join(SRC_DIR, TRAIN_PREFIX.format(fold))
    test_path = os.path.join(SRC_DIR, TEST_PREFIX.format(fold))
    train_labeled_path = os.path.join(DST_DIR, TRAIN_LABELED_PREFIX.format(fold))
    val_path = os.path.join(DST_DIR, VAL_PREFIX.format(fold))
    unlabeled_path = os.path.join(DST_DIR, UNLABELED_PREFIX.format(fold))
    test_dst_path = os.path.join(DST_DIR, TEST_PREFIX.format(fold))

    # 複製 test 檔案
    shutil.copyfile(test_path, test_dst_path)

    # 讀取 train 檔案
    with open(train_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 設定隨機種子以確保可重現性
    random.seed(42 + fold)  # 每個fold使用不同的種子
    random.shuffle(data)
    
    # 計算分割點
    total_size = len(data)
    train_size = max(1, int(total_size * 0.1))  # 10% 訓練集
    val_size = max(1, int(total_size * 0.1))    # 10% 驗證集
    # 剩餘的都是未標籤集 (約80%)
    
    print(f"Fold {fold}: 總計 {total_size} 筆資料")
    print(f"  - 訓練集: {train_size} 筆 ({train_size/total_size*100:.1f}%)")
    print(f"  - 驗證集: {val_size} 筆 ({val_size/total_size*100:.1f}%)")
    print(f"  - 未標籤集: {total_size - train_size - val_size} 筆 ({(total_size - train_size - val_size)/total_size*100:.1f}%)")
    
    # 分割數據
    train_labeled = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    unlabeled = data[train_size + val_size:]

    # 處理 unlabeled data 格式（移除標籤信息）
    unlabeled_processed = []
    for doc in unlabeled:
        new_doc = {
            'doc_id': doc['doc_id'],
            'doc_len': doc['doc_len'],
            'clauses': []
        }
        for clause in doc['clauses']:
            new_clause = {
                'clause_id': clause['clause_id'],
                'clause': clause['clause']
            }
            # 其餘欄位清空
            new_clause['emotion_category'] = []
            new_clause['emotion_token'] = []
            new_doc['clauses'].append(new_clause)
        # pairs 清空
        new_doc['pairs'] = []
        unlabeled_processed.append(new_doc)

    # 寫入訓練集
    with open(train_labeled_path, 'w', encoding='utf-8') as f:
        json.dump(train_labeled, f, ensure_ascii=False, indent=2)
    
    # 寫入驗證集
    with open(val_path, 'w', encoding='utf-8') as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)
    
    # 寫入未標籤集
    with open(unlabeled_path, 'w', encoding='utf-8') as f:
        json.dump(unlabeled_processed, f, ensure_ascii=False, indent=2)

print('切分完成，所有檔案已輸出到', DST_DIR)
print('每個fold包含:')
print('  - fold{}_train_few_shot.json: 訓練集 (10%)')
print('  - fold{}_val.json: 驗證集 (10%)')  
print('  - fold{}_unlabeled.json: 未標籤集 (80%)')
print('  - fold{}_test.json: 測試集 (從原始檔案複製)') 