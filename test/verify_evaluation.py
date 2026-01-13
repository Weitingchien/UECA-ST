"""
驗證 fold1_test_evaluation.txt 中的數值是否正確
比對 fold1_text_result.txt (預測) 與 fold1_test.json (真實標註)
"""
import json
import os

# 設定路徑
pred_dir = "prompt_ECPE_few_shot_ST_2026_01_03_09_37_05_split10_home_train1_test1_val1_unlabeled7_disjoint_f1-10_i70_lr1e-5_bs8_wd0.01_bert-base-chinese_th0.9_maskcause_gamma0.5_reg1e-4_st10_ste3_seed42_retain_pseudo_CE"
gt_dir = "split10_home_train1_test1_val1_unlabeled7_disjoint"

pred_file = os.path.join(pred_dir, "fold1_text_result.txt")
gt_file = os.path.join(gt_dir, "fold1_test.json")

# 讀取預測結果
predictions = {}  # doc_id -> list of (emo, cause, pair)
with open(pred_file, 'r', encoding='utf8') as f:
    current_doc_id = None
    for line in f:
        line = line.strip()
        if line.startswith("doc_id"):
            current_doc_id = line.split(":")[1].strip()
            predictions[current_doc_id] = []
        elif line != "":
            parts = line.split()
            if len(parts) == 3:
                predictions[current_doc_id].append((parts[0], parts[1], parts[2]))

# 讀取 Ground Truth
with open(gt_file, 'r', encoding='utf8') as f:
    gt_data = json.load(f)

# 建立 GT 映射
gt_map = {}
for doc in gt_data:
    doc_id = str(doc["doc_id"])
    d_len = doc["doc_len"]
    pairs = doc["pairs"] if doc["pairs"] else []
    pos, cause = zip(*pairs) if pairs else ([], [])
    pos, cause = list(pos), list(cause)
    
    gt_labels = []
    for i in range(1, d_len + 1):
        emo_tag = "是" if i in pos else "非"
        cause_tag = "是" if i in cause else "非"
        if i in cause:
            pair_id = str(pos[cause.index(i)])
        else:
            pair_id = "无"
        gt_labels.append((emo_tag, cause_tag, pair_id))
    gt_map[doc_id] = gt_labels

# 開始驗證
emo_acc = 0  # 預測正確的情緒數量
emo_pre = 0  # 預測出的情緒數量
emo_gt = 0   # 實際的情緒數量
cause_acc = 0
cause_pre = 0
cause_gt = 0
pair_acc = 0
pair_pre = 0
pair_gt = 0

# 先計算正確的 GT 總數 (直接從 JSON 計算，使用 set 去重)
for doc in gt_data:
    doc_id = str(doc["doc_id"])
    if doc_id in predictions:  # 只計算有預測結果的文檔
        pairs = doc["pairs"] if doc["pairs"] else []
        if pairs:
            # Pair GT: 使用 set 去除完全重複的配對
            pair_gt += len(set(tuple(p) for p in pairs))
            # Emotion GT: 使用 set 去除重複的情緒子句 ID
            emo_gt += len(set(p[0] for p in pairs))
            # Cause GT: 使用 set 去除重複的原因子句 ID
            cause_gt += len(set(p[1] for p in pairs))

# 詳細記錄
correct_emotion_records = []

for doc_id, pred_labels in predictions.items():
    if doc_id not in gt_map:
        print(f"[警告] doc_id {doc_id} 在 GT 中找不到")
        continue
    
    gt_labels = gt_map[doc_id]
    min_len = min(len(pred_labels), len(gt_labels))
    
    for i in range(min_len):
        pred_emo, pred_cause, pred_pair = pred_labels[i]
        gt_emo, gt_cause, gt_pair = gt_labels[i]
        
        # Emotion 統計 (注意: emo_gt 已在上方從 JSON 正確計算)
        if pred_emo == "是":
            emo_pre += 1
            if gt_emo == "是":
                emo_acc += 1
                correct_emotion_records.append(f"doc={doc_id}, clause={i+1}")
        
        # Cause 統計 (注意: cause_gt 已在上方從 JSON 正確計算)
        if pred_cause == "是":
            cause_pre += 1
            if gt_cause == "是":
                cause_acc += 1
        
        # Pair 統計 (注意: pair_gt 已在上方從 JSON 正確計算)
        if pred_pair != "无":
            pair_pre += 1
        if gt_pair != "无" and pred_pair == gt_pair:
            pair_acc += 1

print("=" * 60)
print("驗證結果 (Fold 1):")
print("=" * 60)
print(f"\nEmotion:")
print(f"  預測正確的數量 (TP): {emo_acc}")
print(f"  預測出的數量 (Pred): {emo_pre}")
print(f"  實際正確的數量 (GT): {emo_gt}")

print(f"\nCause:")
print(f"  預測正確的數量 (TP): {cause_acc}")
print(f"  預測出的數量 (Pred): {cause_pre}")
print(f"  實際正確的數量 (GT): {cause_gt}")

print(f"\nPair:")
print(f"  預測正確的數量 (TP): {pair_acc}")
print(f"  預測出的數量 (Pred): {pair_pre}")
print(f"  實際正確的數量 (GT): {pair_gt}")

print("\n" + "=" * 60)
print("預測正確的情緒子句列表 (前100筆):")
print("=" * 60)
for record in correct_emotion_records[:100]:
    print(f"  {record}")
if len(correct_emotion_records) > 100:
    print(f"  ... 還有 {len(correct_emotion_records) - 100} 筆")
