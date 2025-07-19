import os
import json
import random
import shutil

SRC_DIR = 'split10'
DST_DIR = 'split10_few_shot_st'
TRAIN_PREFIX = 'fold{}_train.json'
TEST_PREFIX = 'fold{}_test.json'
LABELED_PREFIX = 'fold{}_train_few_shot.json'
UNLABELED_PREFIX = 'fold{}_unlabeled.json'

os.makedirs(DST_DIR, exist_ok=True)

for fold in range(1, 11):
    train_path = os.path.join(SRC_DIR, TRAIN_PREFIX.format(fold))
    test_path = os.path.join(SRC_DIR, TEST_PREFIX.format(fold))
    labeled_path = os.path.join(DST_DIR, LABELED_PREFIX.format(fold))
    unlabeled_path = os.path.join(DST_DIR, UNLABELED_PREFIX.format(fold))
    test_dst_path = os.path.join(DST_DIR, TEST_PREFIX.format(fold))

    # 複製 test 檔案
    shutil.copyfile(test_path, test_dst_path)

    # 讀取 train 檔案
    with open(train_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 隨機切分 10% labeled, 90% unlabeled
    random.shuffle(data)
    n_labeled = max(1, int(len(data) * 0.1))
    labeled = data[:n_labeled]
    unlabeled = data[n_labeled:]

    # 處理 unlabeled data 格式
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

    # 寫入 labeled
    with open(labeled_path, 'w', encoding='utf-8') as f:
        json.dump(labeled, f, ensure_ascii=False, indent=2)
    # 寫入 unlabeled
    with open(unlabeled_path, 'w', encoding='utf-8') as f:
        json.dump(unlabeled_processed, f, ensure_ascii=False, indent=2)

print('切分完成，所有檔案已輸出到', DST_DIR) 