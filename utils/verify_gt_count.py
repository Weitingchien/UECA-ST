"""
驗證 UECA_CE_v3.py 的 MyDataset 移除超過 512 token 文檔後，
測試集 Pair (m1) 的 Total GT 是否為 2133。

使用方式: 在 UECA_ST 目錄下執行
    python verify_gt_count.py
"""
import json
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('./bert-base-chinese')
dataset_dir = 'split10'

total_pair_gt = 0
total_removed_docs = 0
total_removed_pairs = 0

for fold in range(1, 11):
    test_file = f'{dataset_dir}/fold{fold}_test.json'
    with open(test_file, 'r', encoding='utf8') as f:
        data = json.load(f)
    
    fold_pair_gt = 0
    fold_removed = 0
    fold_removed_pairs = 0
    
    for doc in data:
        doc_id = doc["doc_id"]
        d_len = doc["doc_len"]
        pairs = doc["pairs"]
        pairs_tuples = [tuple(pair) for pair in pairs]
        cnt_pair_gt = len(set(pairs_tuples))
        
        # 模擬 MyDataset 的文檔處理邏輯
        part_sentence = [clause["clause"] for clause in doc["clauses"]]
        mask_full_document = ""
        for i in range(1, d_len + 1):
            mask_full_document = mask_full_document + ' ' + str(i) + ' ' + part_sentence[i - 1]
            mask_full_document = mask_full_document + "[MASK] [MASK] [MASK] [SEP]"
        
        count_len = len(tokenizer.encode_plus(mask_full_document, return_tensors="pt")['input_ids'][0])
        
        if count_len > 512:
            # UECA_CE_v3.py 會移除此文檔
            fold_removed += 1
            fold_removed_pairs += cnt_pair_gt
            print(f"  Fold {fold}: 移除 doc_id={doc_id}, doc_len={d_len}, token_len={count_len}, pairs={cnt_pair_gt}")
        else:
            fold_pair_gt += cnt_pair_gt
    
    total_pair_gt += fold_pair_gt
    total_removed_docs += fold_removed
    total_removed_pairs += fold_removed_pairs
    print(f"Fold {fold}: Pair GT = {fold_pair_gt} (移除 {fold_removed} 篇, 失去 {fold_removed_pairs} pairs)")

print(f"\n{'='*50}")
print(f"Total Pair (m1) GT = {total_pair_gt}")
print(f"Total 移除文檔數 = {total_removed_docs}")
print(f"Total 移除的 pairs = {total_removed_pairs}")
print(f"\n預期結果: 2133 → {'✓ 驗證通過!' if total_pair_gt == 2133 else '✗ 不符!'}")
