import torch
import json
from transformers import BertTokenizer

# 使用下載的 BERT-Base Chinese 模型
tokenizer = BertTokenizer.from_pretrained('./bert-base-chinese')

# 讀取測試資料的第一筆
with open('split10_few_shot/fold1_train_10_percent.json', 'r', encoding='utf8') as f:
    data = json.load(f)

# 取第一筆資料
doc = data[0]
print(f"測試第一筆資料 doc_id: {doc['doc_id']}")
print(f"doc_len: {doc['doc_len']}")
print(f"pairs: {doc['pairs']}")

doc_id = doc["doc_id"]
d_len = doc["doc_len"]
pairs = doc["pairs"]
pos, cause = zip(*pairs) if pairs else ([], [])
pairs = [tuple(pair) for pair in pairs]

full_document = ""
mask_full_document = ""
mask_label_full_document = ""
part_sentence = []
emotions = []

# 處理每個子句
for clause in doc["clauses"]:
    emotion = clause["emotion_category"].strip()
    emotions.append(emotion)
    part_sentence.append(clause["clause"])

# 重現 MyDataset 中的邏輯
for i in range(1, d_len + 1):
    full_document = full_document + ' ' + str(i) + ' ' + part_sentence[i - 1]
    mask_full_document = mask_full_document + ' ' + str(i) + ' ' + part_sentence[i - 1]
    mask_label_full_document = mask_label_full_document + ' ' + str(i) + ' ' + part_sentence[i - 1]
    
    if i in pos:
        full_document = full_document + '是 '
        if i in cause:
            full_document = full_document + '是 '
            full_document = full_document + ' ' + str(pos[cause.index(i)]) + ' '
        else:
            full_document = full_document + '非 '
            full_document = full_document + ' 无 '
    else:
        full_document = full_document + '非 '
        if i in cause:
            full_document = full_document + '是 '
            full_document = full_document + ' ' + str(pos[cause.index(i)]) + ' '
        else:
            full_document = full_document + '非 '
            full_document = full_document + ' 无 '

    full_document = full_document + '[SEP]'
    mask_full_document = mask_full_document + "[MASK] [MASK] [MASK] [SEP]"
    mask_label_full_document = mask_label_full_document + "[MASK] [MASK] [MASK] [SEP]"

print("\n=== 構建的字串內容 ===")
print(f"full_document 長度: {len(full_document)}")
print(f"full_document (前500字符): {full_document[:500]}")
print(f"\nmask_full_document 長度: {len(mask_full_document)}")
print(f"mask_full_document (前500字符): {mask_full_document[:500]}")
print(f"\nmask_label_full_document 長度: {len(mask_label_full_document)}")
print(f"mask_label_full_document (前500字符): {mask_label_full_document[:500]}")

# Tokenize
mask_full_document_tokenized = tokenizer.encode_plus(
    mask_full_document, 
    return_tensors="pt", 
    max_length=512, 
    truncation=True,
    pad_to_max_length=True
)['input_ids']

full_document_tokenized = tokenizer.encode_plus(
    full_document, 
    return_tensors="pt", 
    max_length=512, 
    truncation=True,
    pad_to_max_length=True
)['input_ids']

mask_label_full_document_tokenized = tokenizer.encode_plus(
    mask_label_full_document, 
    return_tensors="pt", 
    max_length=512,
    truncation=True,
    pad_to_max_length=True
)['input_ids']

print("\n=== Tokenized 張量 ===")
print(f"mask_full_document_tokenized shape: {mask_full_document_tokenized.shape}")
print(f"full_document_tokenized shape: {full_document_tokenized.shape}")
print(f"mask_label_full_document_tokenized shape: {mask_label_full_document_tokenized.shape}")

print(f"\nmask_full_document_tokenized (前20個token): {mask_full_document_tokenized[0][:20]}")
print(f"full_document_tokenized (前20個token): {full_document_tokenized[0][:20]}")
print(f"mask_label_full_document_tokenized (前20個token): {mask_label_full_document_tokenized[0][:20]}")

# 找出 [MASK] token 的 ID
mask_token_id = tokenizer.mask_token_id
print(f"\n[MASK] token ID: {mask_token_id}")

# 查看哪些位置是 [MASK] token (ID 103)
mask_positions = (mask_full_document_tokenized == 103).nonzero()
label_mask_positions = (mask_label_full_document_tokenized == 103).nonzero()
print(f"\nmask_full_document 中 [MASK] 位置: {mask_positions.squeeze()[:20]}")  # 只顯示前20個
print(f"mask_label_full_document 中 [MASK] 位置: {label_mask_positions.squeeze()[:20]}")  # 只顯示前20個

print("\n" + "="*80)
print("目標程式碼執行結果:")
print("="*80)

# 執行目標的兩行程式碼
labels = full_document_tokenized.masked_fill(mask_full_document_tokenized != 103, -100)
mask_labels = full_document_tokenized.masked_fill(mask_label_full_document_tokenized != 103, -100)

print(f"\nlabels shape: {labels.shape}")
print(f"mask_labels shape: {mask_labels.shape}")

print(f"\nlabels (前30個值): {labels[0][:30]}")
print(f"mask_labels (前30個值): {mask_labels[0][:30]}")

# 統計非 -100 的值
non_neg100_labels = (labels != -100).sum()
non_neg100_mask_labels = (mask_labels != -100).sum()
print(f"\nlabels 中非 -100 的值數量: {non_neg100_labels}")
print(f"mask_labels 中非 -100 的值數量: {non_neg100_mask_labels}")

# 查看非 -100 的具體值和位置
labels_non_neg100_positions = (labels != -100).nonzero()
mask_labels_non_neg100_positions = (mask_labels != -100).nonzero()
print(f"\nlabels 中非 -100 的位置 (前20個): {labels_non_neg100_positions.squeeze()[:20]}")
print(f"mask_labels 中非 -100 的位置 (前20個): {mask_labels_non_neg100_positions.squeeze()[:20]}")

if len(labels_non_neg100_positions) > 0:
    labels_non_neg100_values = labels[labels != -100]
    print(f"labels 中非 -100 的值 (前20個): {labels_non_neg100_values[:20]}")

if len(mask_labels_non_neg100_positions) > 0:
    mask_labels_non_neg100_values = mask_labels[mask_labels != -100]
    print(f"mask_labels 中非 -100 的值 (前20個): {mask_labels_non_neg100_values[:20]}")

print("\n程式執行完成！")