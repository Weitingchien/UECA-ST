
import os
import sys
import torch
import argparse
import torch.nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from transformers import BertTokenizer, BertForMaskedLM
import time
import numpy as np
import json
import datetime
import random  # 添加 random 模組用於隨機種子設定
from NeST.sample_selection_prompt import select_samples_and_generate_pseudo_labels
from NeST.model_weight_averaging import build_averaged_model_for_fold, save_averaged_model
# from transformers.generation.configuration_utils import CompileConfig


def set_random_seed(seed=42):
    """
    設置所有隨機種子以確保實驗可重現性
    
    Args:
        seed (int): 隨機種子值，預設為 42
    """
    # 設置 Python 內建隨機數生成器的種子
    # 影響: random.random(), random.choice(), random.shuffle() 等
    random.seed(seed)
    
    # 設置 NumPy 隨機數生成器的種子  
    # 影響: np.random.rand(), np.random.choice(), np.random.shuffle() 等
    np.random.seed(seed)
    
    # 設置 PyTorch CPU 隨機數生成器的種子
    # 影響: 模型參數初始化、CPU 上的 Dropout 等
    torch.manual_seed(seed)
    
    # 設置 PyTorch 當前 GPU 隨機數生成器的種子
    # 影響: GPU 上的 Dropout、隨機操作等
    torch.cuda.manual_seed(seed)
    
    
    print(f"隨機種子設定為: {seed}")


def get_label_index():
    """回傳可用於配對預測的子句索引 token id 清單 (數字 1~75 對應的字詞 token)"""
    return [
        122, 123, 124, 125, 126, 127, 128, 129, 130, 8108, 8111, 8110, 8124, 8122,
        8115, 8121, 8126, 8123, 8131, 8113, 8128, 8130, 8133, 8125, 8132, 8153,
        8149, 8143, 8162, 8114, 8176, 8211, 8226, 8229, 8198, 8216, 8234, 8218,
        8240, 8164, 8245, 8239, 8250, 8252, 8208, 8248, 8264, 8214, 8249, 8145,
        8246, 8247, 8251, 8267, 8222, 8259, 8272, 8255, 8257, 8183, 8398, 8356,
        8381, 8308, 8284, 8347, 8369, 8360, 8419, 8203, 8459, 8325, 8454, 8473,
        8273
    ]


def get_binary_token_ids(tokenizer):
    """動態查詢「是」「非」兩個關鍵標籤的 token id，避免硬編數值"""
    return (
        tokenizer.convert_tokens_to_ids('是'),
        tokenizer.convert_tokens_to_ids('非')
    )


def _build_pair_candidate_ids(clause_index, window_size, pair_label_index, no_pair_token_id):
    """根據窗格大小建立情緒子句對應的候選原因子句 token id"""
    start = max(0, -window_size + clause_index - 1)
    end = min(75, window_size + clause_index)
    candidates = [pair_label_index[k] for k in range(start, end)]
    candidates.append(no_pair_token_id)
    return candidates

# 論文裡的 Eq.2 / Eq.3 是一般分類 pseudo-label 更新
# 在Prompt-based ECPE任務，我們需要根據模型對 [MASK] 的預測來產生偽標籤 token 序列，這些 token 代表情緒、原因、配對的預測結果
def generate_pseudo_label_tokens(logits, input_ids, mask_token_id, yes_token_id, window_size, tokenizer, threshold=None):
    """根據模型 logits 與原始輸入，為每個 [MASK] 位置產生偽標籤序列

    這個函式假設模板結構為「每句固定 3 個 [MASK]」，順序如下: 
    1) 情緒MASK 
    2) 原因MASK 
    3) 情緒原因組合MASK 

    核心規則:
    - 情緒/原因MASK :直接取該位置機率最大的 token 作為預測
    - 情緒原因組合MASK:
      - 若該句被預測為「原因句」，只在視窗內候選句編號 + 「无」中做 argmax
      - 若不是原因句，情緒原因MASK一律填「无」
    - 若提供 threshold，且情緒/原因任一MASK最大機率低於 threshold，立即回傳 None
       (代表這筆樣本信心不足，不納入偽標籤)

    參數：
    - logits: shape = (seq_len, vocab_size) 的機率或分數陣列 (通常已 softmax)
    - input_ids: shape = (seq_len,) 的輸入 token ids
    - mask_token_id: [MASK] 的 token id (通常 103)
    - yes_token_id: 「是」的 token id (來判斷原因句)
    - window_size: 組合候選句的視窗大小
    - tokenizer: 用於查詢「无」token id
    - threshold: 可選; 若設定，會對情緒/原因MASK做最低信心檢查

    回傳:
    - np.array(dtype=np.int64): 依 [MASK] 出現順序排列的偽標籤 token 序列
    - None: 當啟用 threshold 且信心不足時
    """
    # 找出序列中所有 [MASK] 的絕對位置 (例如 [10, 15, 20, ...])
    mask_positions = np.where(input_ids == mask_token_id)[0]

    # 用來存最後輸出的偽標籤 (順序必須對齊 mask_positions)
    pseudo_tokens = []

    # 記錄每個子句是否被判定為「原因句」: {clause_idx: True/False}
    # clause_idx 採 1-based (第 1 句、第 2 句...)
    cause_flags = {}

    # 「无」代表無配對 (no pair) 的 token id
    no_pair_token_id = tokenizer.convert_tokens_to_ids('无')

    # 句子編號 1~75 對應的 token id 清單 (用於 pair 槽位)
    pair_label_index = get_label_index()

    # 逐一處理每個 [MASK]
    for idx, pos in enumerate(mask_positions):
        # 依據每句 3 個 [MASK] 的模板：0=情緒, 1=原因, 2=情緒原因組合
        mask_type = idx % 3

        # 取出該 [MASK] 位置對整個詞彙表的機率分佈
        token_probs = logits[pos]

        # ===== 情緒MASK / 原因MASK =====
        if mask_type in (0, 1):
            # 直接取最大機率的 token id
            pred_index = int(np.argmax(token_probs))
            max_prob = float(token_probs[pred_index])

            # 若啟用信心閾值，且最大機率低於門檻，整筆樣本捨棄
            if threshold is not None and max_prob < threshold:
                return None

            # 寫入該MASK偽標籤
            pseudo_tokens.append(pred_index)

            # 若是「原因MASK」，順便記錄該句是否為原因句
            # idx//3 + 1 可把 mask 索引映射到第幾句
            if mask_type == 1:
                clause_idx = idx // 3 + 1
                cause_flags[clause_idx] = (pred_index == yes_token_id)

        # ===== 情緒原因組合MASK =====
        else:
            clause_idx = idx // 3 + 1

            # 只有被判為原因句，才需要找對應的情緒子句是第幾句
            if cause_flags.get(clause_idx, False):
                # 建立候選集合: 視窗內句編號 token + 「无」
                candidate_ids = _build_pair_candidate_ids(clause_idx, window_size, pair_label_index, no_pair_token_id)

                # 只取候選集合上的機率，避免預測到不合法句編號
                candidate_probs = token_probs[candidate_ids]

                # 在候選集合中選最大機率對應的 token id
                selected = int(candidate_ids[int(np.argmax(candidate_probs))])
                pseudo_tokens.append(selected)
            else:
                # 非原因句一律無配對
                pseudo_tokens.append(no_pair_token_id)

    # 最後回傳固定 int64 格式，方便後續資料集與比較邏輯使用
    return np.array(pseudo_tokens, dtype=np.int64)


def _compute_label_distribution_from_logits(logits, input_ids, mask_index, mask_token_id, yes_token_id, no_token_id, debug=False, doc_id=None, tokenizer=None):
    """
    從模型 softmax 結果獲得 [MASK] (是/非) 平均分佈
    
    目的：計算整篇文檔中，指定類型 (情緒/原因) 的「是/非」機率分佈的平均值
    
    參數說明：
        logits: shape (512, vocab_size) - 模型對每個位置的預測機率 (已做 softmax)
        input_ids: shape (512,) - 輸入的 token IDs
        mask_index: 0=情緒, 1=原因, 2=配對 (每個句子有 3 個 [MASK])
        mask_token_id: [MASK] 的 token ID (103)
        yes_token_id: 「是」的 token ID (3221)
        no_token_id: 「非」的 token ID (7478)
        debug: 是否印出每個句子的詳細機率 (預設 False)
        doc_id: 文檔 ID (用於 debug 輸出)
        tokenizer: BERT tokenizer (用於 debug 時 decode 句子文本)
    
    回傳：
        shape (2,) 的 numpy array，代表 [P(是), P(非)] 的平均分佈
    
    範例：假設文檔有 5 個句子，mask_index=0 (情緒)
          會取 [MASK] 位置 0, 3, 6, 9, 12 (每隔3個)
          計算各句子的 P(是)/P(非)，最後取平均
    """
    
    # 找出輸入序列中所有 [MASK] token 的位置索引
    # 找出 input_ids 中所有等於 [MASK]（token id=103）的位置
    mask_positions = np.where(input_ids == mask_token_id)[0]
    
    # 儲存每個符合條件的 [MASK] 的機率分佈
    mask_distributions = []
    
    # DEBUG: 印出標題
    mask_name = {0: '情緒', 1: '原因', 2: '配對'}.get(mask_index, f'mask{mask_index}')
    if debug:
        print(f"\n[DEBUG] 文檔 {doc_id} - {mask_name} 預測詳情:")
        print(f"  {'句子':<8} {'位置':<8} {'P(是)':<10} {'P(非)':<10} {'句內資訊':<25}")
        print(f"  {'-'*70}")
    
    # 句子計數器
    sentence_idx = 0
    
    # 遍歷所有 [MASK] 位置
    for idx, pos in enumerate(mask_positions):
        # pos: 在整個512序列中的絕對位置
        # 每個句子有 3 個 [MASK]：idx % 3 == 0 是情緒, == 1 是原因, == 2 是配對
        # 只處理符合 mask_index 的位置，其他跳過
        # 例如：mask_index=0 時，只取 idx=0,3,6,9... (情緒位置)
        if idx % 3 != mask_index:
            continue
        
        sentence_idx += 1
        
        # 從 logits 中取出該位置對「是」token 的預測機率
        yes_prob = float(logits[pos, yes_token_id])
        
        # 從 logits 中取出該位置對「非」token 的預測機率
        no_prob = float(logits[pos, no_token_id])
        
        # 計算「是」和「非」的機率總和 (用於正規化)
        prob_sum = yes_prob + no_prob
        
        # 正規化後的機率
        yes_norm = yes_prob / prob_sum
        no_norm = no_prob / prob_sum
        
        # DEBUG: 印出每個句子的詳細機率
        if debug:
            # 嘗試提取該句子的文本內容 (從前一個 [SEP] 到當前 [MASK] 之間)
            sentence_text = ""
            token_list_str = ""
            context_info = ""
            if tokenizer is not None:
                # 找出當前 [MASK] 前面最近的 [SEP] 位置
                sep_token_id = tokenizer.sep_token_id  # [SEP] = 102
                # 找出 pos 前面所有 [SEP] 的位置
                sep_positions = np.where(input_ids[:pos] == sep_token_id)[0]
                # 句子起始位置：如果有 [SEP]，從最後一個 [SEP]+1 開始；否則從位置 1 開始（跳過 [CLS]）
                start_pos = sep_positions[-1] + 1 if len(sep_positions) > 0 else 1
                # 句子結束位置：找當前 pos 之後最近的 [SEP]
                sep_after = np.where(input_ids[pos:] == sep_token_id)[0]
                end_pos = pos + sep_after[0] if len(sep_after) > 0 else pos + 20
                # Decode 句子 (移除 PAD tokens，但保留 [MASK] 等特殊 token)
                sentence_tokens = input_ids[start_pos:end_pos]
                # 去除 pad token id，避免顯示 [PAD]
                pad_id = tokenizer.pad_token_id
                filtered_sentence_tokens = [int(t) for t in sentence_tokens if int(t) != pad_id]
                # decode 完整句子 (不用截斷)，保留 [MASK] 標記
                sentence_text = tokenizer.decode(filtered_sentence_tokens, skip_special_tokens=False).strip()
                
                # 顯示 [MASK] 在句子中的相對位置
                # pos - start_pos 就是 [MASK] 在句子中的位置（從 0 開始）
                mask_offset_in_sentence = pos - start_pos
                context_info = f"[句內位置={mask_offset_in_sentence}, 句範圍={start_pos}~{end_pos}]"
                
                # 建立 token list 字串，用 (位置, token) 格式顯示，並標記 [MASK] 位置
                token_list = []
                for i, tid in enumerate(sentence_tokens):
                    # 跳過 PAD token 的顯示
                    if int(tid) == pad_id:
                        continue
                    abs_pos = start_pos + i  # 絕對位置
                    token_str = tokenizer.decode([int(tid)]).replace(' ', '')
                    if abs_pos == pos:
                        # 這是當前的 [MASK] 位置，用 ★ 標記
                        token_list.append(f"({abs_pos},★{token_str}★)")
                    else:
                        token_list.append(f"({abs_pos},{token_str})")
                token_list_str = " ".join(token_list)
            
            print(f"  句子{sentence_idx:<4} pos={pos:<4} P(是)={yes_norm:.4f} P(非)={no_norm:.4f} {context_info}")
            print(f"         Token列表: {token_list_str}")
        
        # 正規化：將「是」和「非」的機率調整為總和為 1
        # 回傳 [P(是), P(非)]，例如 [0.9, 0.1] 表示 90% 機率是「是」
        mask_distributions.append(np.array([yes_norm, no_norm], dtype=np.float32))
    
    # 將所有句子的分佈堆疊成 (n_sentences, 2) 矩陣，然後沿 axis=0 取平均
    # 例如：5 個句子的情緒分佈取平均，得到整篇文檔的情緒傾向
    averaged = np.mean(np.stack(mask_distributions, axis=0), axis=0)
    
    # DEBUG: 印出平均結果
    if debug:
        print(f"  {'-'*66}")
        print(f"  平均值: {averaged[0]:<12.6f} {averaged[1]:<12.6f}")
        print(f"  共 {sentence_idx} 個句子")
    
    # 回傳 [P(是), P(非)]
    return averaged.astype(np.float32)


def _compute_emotion_clause_distribution(logits, input_ids, mask_token_id, yes_token_id, no_token_id,
                                         debug=False, doc_id=None, tokenizer=None):
    """
    計算「被預測為情緒句」的子句的情緒分佈平均值
    
    用途：針對 NeST 的 divergence_mode='emotion_clause' 模式，
         只取模型預測為情緒句 (P(是) > P(非)) 的子句來計算情緒分佈
    
    參數說明：
        logits: shape (512, vocab_size) - 模型對每個位置的預測機率 (已做 softmax)
        input_ids: shape (512,) - 輸入的 token IDs
        mask_token_id: [MASK] 的 token ID (103)
        yes_token_id: 「是」的 token ID (3221)
        no_token_id: 「非」的 token ID (7478)
    
    回傳：
        shape (2,) 的 numpy array，代表 [P(是), P(非)] 的平均分佈
        若無預測為情緒句的子句，則回傳 None (由呼叫者決定 fallback 策略)
    """
    mask_positions = np.where(input_ids == mask_token_id)[0]
    
    # 收集被預測為情緒句的子句的情緒分佈
    emotion_clause_dists = []

    if debug and tokenizer is None:
        raise ValueError("tokenizer must be provided when debug=True")

    if debug:
        print(f"\n{'='*70}")
        print(f"[DEBUG] 文檔 {doc_id} emotion_clause 分佈計算")
        print(f"  規則: 以每個子句 emotion [MASK] 判斷是否情緒句 (P(是) > P(非))")
    
    # 每個子句有 3 個 [MASK]：idx % 3 == 0 是情緒, == 1 是原因, == 2 是配對
    sentence_idx = 0
    for idx in range(0, len(mask_positions), 3):
        if idx >= len(mask_positions):
            break
        
        # 取得情緒位置 (第 0 個 MASK)
        emotion_pos = int(mask_positions[idx])
        emotion_yes_prob = float(logits[emotion_pos, yes_token_id])
        emotion_no_prob = float(logits[emotion_pos, no_token_id])

        # 二元正規化：只在「是/非」兩類上重新歸一化
        prob_sum = emotion_yes_prob + emotion_no_prob
        assert prob_sum > 0, f"Invalid prob_sum={prob_sum} for doc_id={doc_id}, pos={emotion_pos}"
        emotion_yes_norm = emotion_yes_prob / prob_sum
        emotion_no_norm = emotion_no_prob / prob_sum
        
        # DEBUG: 印出該句的句子範圍與 token 列表 (仿照 _compute_label_distribution_from_logits)
        sentence_idx += 1
        if debug:
            sep_token_id = tokenizer.sep_token_id  # [SEP] = 102
            sep_positions = np.where(input_ids[:emotion_pos] == sep_token_id)[0]
            start_pos = int(sep_positions[-1] + 1) if len(sep_positions) > 0 else 1
            sep_after = np.where(input_ids[emotion_pos:] == sep_token_id)[0]
            end_pos = int(emotion_pos + sep_after[0]) if len(sep_after) > 0 else int(emotion_pos + 20)

            sentence_tokens = input_ids[start_pos:end_pos]
            pad_id = tokenizer.pad_token_id
            mask_offset_in_sentence = int(emotion_pos - start_pos)
            context_info = f"[句內位置={mask_offset_in_sentence}, 句範圍={start_pos}~{end_pos}]"

            token_list = []
            for i, tid in enumerate(sentence_tokens):
                if int(tid) == pad_id:
                    continue
                abs_pos = start_pos + i
                token_str = tokenizer.decode([int(tid)]).replace(' ', '')
                if abs_pos == emotion_pos:
                    token_list.append(f"({abs_pos},★{token_str}★)")
                else:
                    token_list.append(f"({abs_pos},{token_str})")
            token_list_str = " ".join(token_list)

            predicted_emotion_clause = emotion_yes_norm > emotion_no_norm
            flag = "✓ 情緒句" if predicted_emotion_clause else ""
            print(f"  句子{sentence_idx:<4} pos={emotion_pos:<4} P(是)={emotion_yes_norm:.4f} P(非)={emotion_no_norm:.4f} {flag} {context_info}")
            print(f"         Token列表: {token_list_str}")

        # 判斷是否為情緒句: P(是) > P(非)
        if emotion_yes_norm > emotion_no_norm:
            emotion_clause_dists.append(np.array([emotion_yes_norm, emotion_no_norm], dtype=np.float32))
    
    # 若沒有任何子句被預測為情緒句，回傳 None
    if len(emotion_clause_dists) == 0:
        if debug:
            print(f"  {'-'*66}")
            print("  (此文檔沒有任何子句被預測為情緒句，因此回傳 None)")
        return None
    
    # 計算被預測為情緒句的子句的情緒分佈平均
    averaged = np.mean(np.stack(emotion_clause_dists, axis=0), axis=0)
    if debug:
        print(f"  {'-'*66}")
        print(f"  emotion_clause 平均值: {averaged[0]:<12.6f} {averaged[1]:<12.6f}")
        print(f"  共 {len(emotion_clause_dists)} 個情緒句子句")
    return averaged.astype(np.float32)


def _compute_cause_clause_distribution(logits, input_ids, mask_token_id, yes_token_id, no_token_id,
                                        debug=False, doc_id=None, tokenizer=None):
    """
    計算「被預測為原因句」的子句的原因分佈平均值
    
    用途：針對 NeST 的 divergence_mode='cause_clause' 模式，
         只取模型預測為原因句 (P(是) > P(非)) 的子句來計算原因分佈。
    
    回傳：
        shape (2,) 的 numpy array，代表 [P(是), P(非)] 的平均分佈
        若無預測為原因句的子句，則回傳 None
    """
    mask_positions = np.where(input_ids == mask_token_id)[0]
    cause_clause_dists = []

    if debug and tokenizer is None:
        raise ValueError("tokenizer must be provided when debug=True")

    if debug:
        print(f"\n{'='*70}")
        print(f"[DEBUG] 文檔 {doc_id} cause_clause 分佈計算")
        print(f"  規則: 以每個子句 cause [MASK] 判斷是否原因句 (P(是) > P(非))")

    sentence_idx = 0
    for idx in range(0, len(mask_positions), 3):
        if idx + 1 >= len(mask_positions):
            break
        
        # 取得原因位置 (第 1 個 MASK，idx % 3 == 1)
        cause_pos = int(mask_positions[idx + 1])
        cause_yes_prob = float(logits[cause_pos, yes_token_id])
        cause_no_prob = float(logits[cause_pos, no_token_id])

        prob_sum = cause_yes_prob + cause_no_prob
        assert prob_sum > 0, f"Invalid prob_sum={prob_sum} for doc_id={doc_id}, pos={cause_pos}"
        cause_yes_norm = cause_yes_prob / prob_sum
        cause_no_norm = cause_no_prob / prob_sum

        sentence_idx += 1
        if debug:
            sep_token_id = tokenizer.sep_token_id  # [SEP] = 102
            sep_positions = np.where(input_ids[:cause_pos] == sep_token_id)[0]
            start_pos = int(sep_positions[-1] + 1) if len(sep_positions) > 0 else 1
            sep_after = np.where(input_ids[cause_pos:] == sep_token_id)[0]
            end_pos = int(cause_pos + sep_after[0]) if len(sep_after) > 0 else int(cause_pos + 20)

            sentence_tokens = input_ids[start_pos:end_pos]
            pad_id = tokenizer.pad_token_id
            mask_offset_in_sentence = int(cause_pos - start_pos)
            context_info = f"[句內位置={mask_offset_in_sentence}, 句範圍={start_pos}~{end_pos}]"

            token_list = []
            for i, tid in enumerate(sentence_tokens):
                if int(tid) == pad_id:
                    continue
                abs_pos = start_pos + i
                token_str = tokenizer.decode([int(tid)]).replace(' ', '')
                if abs_pos == cause_pos:
                    token_list.append(f"({abs_pos},★{token_str}★)")
                else:
                    token_list.append(f"({abs_pos},{token_str})")
            token_list_str = " ".join(token_list)

            predicted_cause_clause = cause_yes_norm > cause_no_norm
            flag = "✓ 原因句" if predicted_cause_clause else ""
            print(f"  句子{sentence_idx:<4} pos={cause_pos:<4} P(是)={cause_yes_norm:.4f} P(非)={cause_no_norm:.4f} {flag} {context_info}")
            print(f"         Token列表: {token_list_str}")

        if cause_yes_norm > cause_no_norm:
            cause_clause_dists.append(np.array([cause_yes_norm, cause_no_norm], dtype=np.float32))

    if len(cause_clause_dists) == 0:
        if debug:
            print(f"  {'-'*66}")
            print("  (此文檔沒有任何子句被預測為原因句，因此回傳 None)")
        return None

    averaged = np.mean(np.stack(cause_clause_dists, axis=0), axis=0)
    if debug:
        print(f"  {'-'*66}")
        print(f"  cause_clause 平均值: {averaged[0]:<12.6f} {averaged[1]:<12.6f}")
        print(f"  共 {len(cause_clause_dists)} 個原因句子句")
    return averaged.astype(np.float32)


def collect_labeled_statistics(model, dataloader, tokenizer, device, debug=False, debug_num_docs=3, doc_ids=None, pairs_list=None, knn_embedding_mode='cls', return_embedding_info=False):
    """
    蒐集有標註資料的嵌入向量與情緒/原因的 (是/非) 機率分佈
    
    用途：為 NeST 準備有標註樣本的特徵與標籤分佈，作為 KNN 搜尋的索引
    
    參數：
        model: BERT 模型
        dataloader: 有標籤資料的 DataLoader
        tokenizer: BERT tokenizer
        device: 計算裝置 (cuda/cpu)
        debug: 是否印出每個句子的詳細機率 (預設 False)
        debug_num_docs: debug 模式下要印出幾篇文檔的詳細資訊 (預設 3)
        doc_ids: 文檔 ID 列表 (用於 debug 輸出)
        pairs_list: 每篇文檔的情緒-原因配對列表 (用於 debug 輸出)
        knn_embedding_mode: 'cls'=使用[CLS], 'emotion_clause'/'cause_clause'=使用子句embedding
        return_embedding_info: 是否回傳每個樣本的 embedding 詳細資訊 (預設 False)
    
    回傳：
        labeled_features: shape (n_samples, 768) - embedding 向量 (CLS 或子句聚合)
        labeled_emotion: shape (n_samples, 2) - 情緒的 [P(是), P(非)] 分佈
        labeled_cause: shape (n_samples, 2) - 原因的 [P(是), P(非)] 分佈
        labeled_emotion_clause: shape (n_samples, 2) - 情緒句的情緒分佈 (用於 emotion_clause 模式)
        labeled_cause_clause: shape (n_samples, 2) - 原因句的原因分佈 (用於 cause_clause 模式)
        embedding_info: (僅當 return_embedding_info=True) list of dict
    """
    # 取得「是」和「非」對應的 token ID (用於計算二元分類機率)
    yes_token_id, no_token_id = get_binary_token_ids(tokenizer)
    # 取得 [MASK] token 的 ID (用於找出需要預測的位置)
    mask_token_id = tokenizer.mask_token_id
    
    # 初始化
    feature_chunks = []  # 儲存每個 batch 的 CLS embedding
    emotion_dists, cause_dists, emotion_clause_dists, cause_clause_dists = [], [], [], []  # 儲存每筆樣本的機率分佈
    all_embedding_info = [] if return_embedding_info else None  # 儲存每個樣本的 embedding 詳細資訊
    
    # 設定模型為評估模式
    model.eval()
    
    # 文檔計數器 (用於 debug)
    doc_counter = 0
    
    with torch.no_grad():
        for batch in dataloader:
            # 從 batch 中解包資料 (只需要 x_bert，其餘用 _ 忽略)
            # x_bert 形狀: (batch_size, 512) - 輸入的 token IDs
            x_bert, _, _, _, _, _, _, _ = batch
            
            # 將輸入移到指定裝置 (GPU/CPU)
            x_device = x_bert.to(device)
            
            # 執行前向傳播取得 logits
            _, logits = model(x_device, labels=None)
            probs = F.softmax(logits, dim=-1) 
            probs_np = probs.cpu().numpy()
            
            # 根據 knn_embedding_mode 選擇 embedding
            if knn_embedding_mode == 'cls':
                # 直接使用原本的方法取得 [CLS] embedding
                batch_embeddings = model.get_cls_embeddings(x_device)
                if return_embedding_info:
                    # CLS 模式下，每個樣本的 embedding_info 是固定的
                    batch_size = x_device.shape[0]
                    for _ in range(batch_size):
                        all_embedding_info.append({
                            'used_clauses': 0,
                            'is_fallback': False,  # CLS 模式不算 fallback
                            'embedding_type': 'cls',
                            'selected_clause_indices': [],
                            'selected_clause_texts': [],
                        })
            else:
                # 子句級別 embedding (emotion_clause 或 cause_clause)
                if return_embedding_info:
                    batch_embeddings, batch_info = model.get_clause_embeddings(x_device, probs, knn_embedding_mode, return_info=True)
                    for info in batch_info:
                        info['embedding_type'] = 'clause' if not info['is_fallback'] else 'cls_fallback'
                        all_embedding_info.append(info)
                else:
                    batch_embeddings = model.get_clause_embeddings(x_device, probs, knn_embedding_mode)
            feature_chunks.append(batch_embeddings.cpu())
            
            # 將輸入移到 CPU (用於找出 [MASK] 位置)
            inputs_cpu = x_bert.cpu().numpy()
            
            # 遍歷 batch 中的每筆文件
            for doc_probs, doc_input in zip(probs_np, inputs_cpu):
                # 判斷是否要印出 debug 資訊
                should_debug = debug and doc_counter < debug_num_docs
                current_doc_id = doc_ids[doc_counter] if doc_ids is not None else doc_counter
                
                # DEBUG: 印出該文檔的 pairs (情緒-原因配對)
                if should_debug and pairs_list is not None and doc_counter < len(pairs_list):
                    current_pairs = pairs_list[doc_counter]
                    print(f"\n{'='*70}")
                    print(f"[DEBUG] 文檔 {current_doc_id} 的情緒-原因配對 (Ground Truth):")
                    print(f"  pairs = {current_pairs}")
                    if current_pairs:
                        print(f"  解讀: ", end="")
                        pair_strs = [f"句子{e}是情緒句，句子{c}是其原因" for e, c in current_pairs]
                        print("; ".join(pair_strs))
                    else:
                        print(f"  (此文檔沒有情緒-原因配對)")
                
                # 計算情緒分佈: 從第 0 個 [MASK] (每個子句的情緒位置)
                # 回傳 shape: (2,) 即 [P(是), P(非)]
                emotion_dist = _compute_label_distribution_from_logits(
                    doc_probs, doc_input, 0, mask_token_id, yes_token_id, no_token_id,
                    debug=should_debug, doc_id=current_doc_id, tokenizer=tokenizer
                )
                emotion_dists.append(emotion_dist)
                # 計算原因分佈: 從第 1 個 [MASK] (每個子句的原因位置)
                # 回傳 shape: (2,) 即 [P(是), P(非)]
                cause_dist = _compute_label_distribution_from_logits(
                    doc_probs, doc_input, 1, mask_token_id, yes_token_id, no_token_id,
                    debug=should_debug, doc_id=current_doc_id, tokenizer=tokenizer
                )
                cause_dists.append(cause_dist)
                
                # 計算情緒句分佈: 只取預測為情緒句的子句，計算其情緒分佈平均
                # 若無預測為情緒句的子句，則 fallback 到全文檔情緒分佈平均 (emotion_dist)
                emotion_clause_dist = _compute_emotion_clause_distribution(
                    doc_probs, doc_input, mask_token_id, yes_token_id, no_token_id,
                    debug=should_debug, doc_id=current_doc_id, tokenizer=tokenizer
                )
                emotion_clause_dists.append(emotion_clause_dist if emotion_clause_dist is not None else emotion_dist)
                
                # 計算原因句分佈: 只取預測為原因句的子句，計算其原因分佈平均
                # 若無預測為原因句的子句，則 fallback 到全文檔原因分佈平均 (cause_dist)
                cause_clause_dist = _compute_cause_clause_distribution(
                    doc_probs, doc_input, mask_token_id, yes_token_id, no_token_id,
                    debug=should_debug, doc_id=current_doc_id, tokenizer=tokenizer
                )
                cause_clause_dists.append(cause_clause_dist if cause_clause_dist is not None else cause_dist)
                
                doc_counter += 1
    
    # 將所有 batch 的 CLS 向量合併成一個 tensor
    # labeled_features 形狀: (n_samples, 768)
    labeled_features = torch.cat(feature_chunks, dim=0) if feature_chunks else torch.empty(0, model.bert.config.hidden_size)
    
    # 將情緒分佈堆疊成 numpy 陣列
    # labeled_emotion 形狀: (n_samples, 2)
    labeled_emotion = np.stack(emotion_dists, axis=0) if emotion_dists else np.empty((0, 2), dtype=np.float32)
    
    # 將原因分佈堆疊成 numpy 陣列
    # labeled_cause 形狀: (n_samples, 2)
    labeled_cause = np.stack(cause_dists, axis=0) if cause_dists else np.empty((0, 2), dtype=np.float32)
    
    # 將情緒句分佈堆疊成 numpy 陣列 (用於 emotion_clause 模式)
    # labeled_emotion_clause 形狀: (n_samples, 2)
    assert len(emotion_clause_dists) > 0, "emotion_clause_dists is empty"
    labeled_emotion_clause = np.stack(emotion_clause_dists, axis=0)
    
    # 將原因句分佈堆疊成 numpy 陣列 (用於 cause_clause 模式)
    # labeled_cause_clause 形狀: (n_samples, 2)
    assert len(cause_clause_dists) > 0, "cause_clause_dists is empty"
    labeled_cause_clause = np.stack(cause_clause_dists, axis=0)
    
    if return_embedding_info:
        return labeled_features, labeled_emotion, labeled_cause, labeled_emotion_clause, labeled_cause_clause, all_embedding_info
    return labeled_features, labeled_emotion, labeled_cause, labeled_emotion_clause, labeled_cause_clause


def collect_unlabeled_statistics(model, dataloader, tokenizer, device, window_size, debug=False, debug_num_docs=3, doc_ids=None, knn_embedding_mode='cls', return_embedding_info=False):
    """
    蒐集未標註資料的 embedding 向量、模型預測的情緒/原因機率分佈，以及偽標籤 token 序列
    
    用途：為 NeST 準備未標籤樣本的特徵與預測分佈，同時產生偽標籤避免重複推論
    
    參數：
        knn_embedding_mode: 'cls'=使用[CLS], 'emotion_clause'/'cause_clause'=使用子句embedding
        return_embedding_info: 是否回傳每個樣本的 embedding 詳細資訊 (預設 False)
    
    回傳：
        unlabeled_features: shape (n_unlabeled, 768) - embedding 向量 (CLS 或子句聚合)
        unlabeled_emotion: shape (n_unlabeled, 2) - 情緒 [P(是), P(非)] 分佈
        unlabeled_cause: shape (n_unlabeled, 2) - 原因 [P(是), P(非)] 分佈
        unlabeled_emotion_clause: shape (n_unlabeled, 2) - 情緒句的情緒分佈 (用於 emotion_clause 模式)
        unlabeled_cause_clause: shape (n_unlabeled, 2) - 原因句的原因分佈 (用於 cause_clause 模式)
        pseudo_tokens_list: list of np.array - 每個樣本的偽標籤 token 序列
        embedding_info: (僅當 return_embedding_info=True) list of dict
    """
    yes_token_id, no_token_id = get_binary_token_ids(tokenizer)
    mask_token_id = tokenizer.mask_token_id
    
    feature_chunks = []
    emotion_dists, cause_dists, emotion_clause_dists, cause_clause_dists = [], [], [], []
    pseudo_tokens_list = []  # 儲存每個樣本的偽標籤 token 序列
    all_embedding_info = [] if return_embedding_info else None  # 儲存每個樣本的 embedding 詳細資訊
    
    model.eval()
    doc_counter = 0
    
    with torch.no_grad():
        for batch in dataloader:
            x_device = batch.to(device)
            
            # 執行前向傳播取得 logits
            _, logits = model(x_device, labels=None)
            probs = F.softmax(logits, dim=-1)  # 保持為 tensor 以便重用
            probs_np = probs.cpu().numpy()
            
            # 根據 knn_embedding_mode 選擇 embedding
            if knn_embedding_mode == 'cls':
                # 直接使用原本的方法取得 [CLS] embedding
                batch_embeddings = model.get_cls_embeddings(x_device)
                if return_embedding_info:
                    batch_size = x_device.shape[0]
                    for _ in range(batch_size):
                        all_embedding_info.append({
                            'used_clauses': 0,
                            'is_fallback': False,
                            'embedding_type': 'cls',
                            'selected_clause_indices': [],
                            'selected_clause_texts': [],
                        })
            else:
                # 子句級別 embedding (emotion_clause 或 cause_clause)
                if return_embedding_info:
                    batch_embeddings, batch_info = model.get_clause_embeddings(x_device, probs, knn_embedding_mode, return_info=True)
                    for info in batch_info:
                        info['embedding_type'] = 'clause' if not info['is_fallback'] else 'cls_fallback'
                        all_embedding_info.append(info)
                else:
                    batch_embeddings = model.get_clause_embeddings(x_device, probs, knn_embedding_mode)
            feature_chunks.append(batch_embeddings.cpu())
            
            inputs_cpu = batch.cpu().numpy()
            
            for doc_probs, doc_input in zip(probs_np, inputs_cpu):
                should_debug = debug and doc_counter < debug_num_docs
                current_doc_id = doc_ids[doc_counter] if doc_ids is not None and doc_counter < len(doc_ids) else f"unlabeled_{doc_counter}"
                
                # 計算情緒預測分佈
                emotion_dist = _compute_label_distribution_from_logits(
                    doc_probs, doc_input, 0, mask_token_id, yes_token_id, no_token_id,
                    debug=should_debug, doc_id=current_doc_id, tokenizer=tokenizer
                )
                emotion_dists.append(emotion_dist)
                # 計算原因預測分佈
                cause_dist = _compute_label_distribution_from_logits(
                    doc_probs, doc_input, 1, mask_token_id, yes_token_id, no_token_id,
                    debug=should_debug, doc_id=current_doc_id, tokenizer=tokenizer
                )
                cause_dists.append(cause_dist)
                
                # 計算情緒句分佈: 只取預測為情緒句的子句，計算其情緒分佈平均
                # 若無預測為情緒句的子句，則 fallback 到全文檔情緒分佈平均 (emotion_dist)
                emotion_clause_dist = _compute_emotion_clause_distribution(
                    doc_probs, doc_input, mask_token_id, yes_token_id, no_token_id,
                    debug=should_debug, doc_id=current_doc_id, tokenizer=tokenizer
                )
                emotion_clause_dists.append(emotion_clause_dist if emotion_clause_dist is not None else emotion_dist)
                
                # 計算原因句分佈: 只取預測為原因句的子句，計算其原因分佈平均
                cause_clause_dist = _compute_cause_clause_distribution(
                    doc_probs, doc_input, mask_token_id, yes_token_id, no_token_id,
                    debug=should_debug, doc_id=current_doc_id, tokenizer=tokenizer
                )
                cause_clause_dists.append(cause_clause_dist if cause_clause_dist is not None else cause_dist)
                
                # 同時產生偽標籤 token 序列 (避免後續重複推論)
                pseudo_tokens = generate_pseudo_label_tokens(
                    doc_probs, doc_input, mask_token_id, yes_token_id, window_size, tokenizer
                )
                pseudo_tokens_list.append(pseudo_tokens)
                
                doc_counter += 1
    
    unlabeled_features = torch.cat(feature_chunks, dim=0) if feature_chunks else torch.empty(0, model.bert.config.hidden_size)
    unlabeled_emotion = np.stack(emotion_dists, axis=0) if emotion_dists else np.empty((0, 2), dtype=np.float32)
    unlabeled_cause = np.stack(cause_dists, axis=0) if cause_dists else np.empty((0, 2), dtype=np.float32)
    assert len(emotion_clause_dists) > 0, "emotion_clause_dists is empty"
    unlabeled_emotion_clause = np.stack(emotion_clause_dists, axis=0)
    assert len(cause_clause_dists) > 0, "cause_clause_dists is empty"
    unlabeled_cause_clause = np.stack(cause_clause_dists, axis=0)
    
    if return_embedding_info:
        return unlabeled_features, unlabeled_emotion, unlabeled_cause, unlabeled_emotion_clause, unlabeled_cause_clause, pseudo_tokens_list, all_embedding_info
    return unlabeled_features, unlabeled_emotion, unlabeled_cause, unlabeled_emotion_clause, unlabeled_cause_clause, pseudo_tokens_list





def get_clause_boundaries(input_ids, mask_token_id=103, sep_token_id=102):
    """
    識別每個子句的內容範圍 (不含子句編號和 [MASK] tokens)
    
    Args:
        input_ids: 1D numpy array 或 tensor，單個文檔的 token ids
        mask_token_id: [MASK] 的 token id (預設 103)
        sep_token_id: [SEP] 的 token id (預設 102)
    
    Returns:
        list of (content_start, content_end, mask_positions):
            content_start: 子句內容起始位置 (不含子句編號)
            content_end: 子句內容結束位置 (不含 [MASK])
            mask_positions: 該子句的 3 個 [MASK] 位置 tuple (emotion, cause, pair)
    """
    if isinstance(input_ids, torch.Tensor):
        input_ids = input_ids.cpu().numpy()
    

    sep_positions = np.where(input_ids == sep_token_id)[0] # 找出所有 [SEP] 位置
    mask_positions = np.where(input_ids == mask_token_id)[0] # 找出所有 [MASK] 位置
    
    boundaries = []
    prev_end = 1  # 跳過 [CLS]
    mask_idx = 0  # 用於追蹤 [MASK] 位置
    
    for sep_pos in sep_positions:
        # 找出屬於此子句的 3 個 [MASK] (在 prev_end 和 sep_pos 之間)
        clause_masks = []
        while mask_idx < len(mask_positions) and mask_positions[mask_idx] < sep_pos:
            clause_masks.append(mask_positions[mask_idx])
            mask_idx += 1
        
        if len(clause_masks) != 3:
            # 文檔可能被截斷 (512 token 限制)，導致最後幾個子句不完整
            # 這種情況直接跳過，不視為錯誤
            continue
        
        # 子句內容範圍：從 prev_end+1 (跳過子句編號) 到第一個 [MASK] 之前
        content_start = prev_end + 1  # 跳過子句編號 (如 "1", "2", ...)
        content_end = clause_masks[0]  # 第一個 [MASK] 之前
        
        if content_start < content_end:
            boundaries.append((content_start, content_end, tuple(clause_masks)))
        
        prev_end = sep_pos + 1
    
    return boundaries


def is_emotion_or_cause_clause(probs, mask_positions, mode, yes_token_id, no_token_id):
    """
    判斷子句是否為情緒句或原因句 (根據模型預測 P(是) > P(非))
    
    Args:
        probs: 單個文檔的 softmax 機率，shape (512, vocab_size)
        mask_positions: 該子句的 3 個 [MASK] 位置 (emotion, cause, pair)
        mode: 'emotion_clause', 'cause_clause', 'emotion_clause_mask', 'cause_clause_mask'
        yes_token_id: 「是」的 token id
        no_token_id: 「非」的 token id
    
    Returns:
        bool: 是否為情緒句或原因句
    """
    emotion_pos, cause_pos, _ = mask_positions
    
    if mode in ('emotion_clause', 'emotion_clause_mask'):
        yes_prob = probs[emotion_pos, yes_token_id]
        no_prob = probs[emotion_pos, no_token_id]
    else:  # cause_clause, cause_clause_mask
        yes_prob = probs[cause_pos, yes_token_id]
        no_prob = probs[cause_pos, no_token_id]
    
    return yes_prob > no_prob


def compute_nest_threshold_loss(model, x_bert, mask_label, threshold, use_gpu, return_stats=False):
    """
    NeST 論文的 threshold 過濾 loss 計算 (向量化版本)
    
    實作論文公式(5)中的 ℓ_st = 𝟙{[f(x_j; θ_s)]_{ỹ_j} > γ} × ℓ_sup
    只有模型對偽標籤的預測機率 > threshold (γ) 的 [MASK] 位置才計入 loss
    
    Args:
        model: prompt_bert 模型
        x_bert: 輸入序列，shape (batch, 512)，包含 [MASK] token (id=103)
        mask_label: 偽標籤，shape (batch, 512)，[MASK] 位置有偽標籤的 token id，其他位置是 -100
        threshold: 信心閾值 γ (論文預設 0.9)
        use_gpu: 是否使用 GPU
    
    Returns:
        loss: 過濾後的平均 Cross-Entropy loss (只計算信心 > γ 的位置)
        若 return_stats=True，額外回傳 (valid_token_count, confident_token_count)
    """
    device = torch.device('cuda' if use_gpu else 'cpu')
    x_bert = x_bert.to(device)
    mask_label = mask_label.to(device)
    
    # ========== Step 1: 模型前向傳播 ==========
    outputs = model.bert(x_bert, labels=None)
    logits = outputs.logits  # (batch, seq_len=512, vocab_size=21128)
    
    # ========== Step 2: 找出所有 [MASK] 位置 ==========
    # mask_positions: 布林遮罩，標記 [MASK] 位置 (token_id = 103)
    mask_positions = (x_bert == 103)  # (batch, seq_len)
    
    # 同時檢查 mask_label 不是 -100 (既是[MASK]位置且有有效的偽標籤位置才會是True)
    valid_positions = mask_positions & (mask_label != -100)  # (batch, seq_len)
    
    valid_token_count = int(valid_positions.sum().item())

    if not valid_positions.any(): #any(): 任一個為True就回傳True
        # 沒有有效的 [MASK] 位置，回傳 0 loss
        zero_loss = torch.tensor(0.0, requires_grad=True, device=device)
        if return_stats:
            return zero_loss, 0, 0
        return zero_loss
    
    # ========== Step 3: 提取有效位置的 logits 和 targets ==========
    # valid_logits: 所有有效 [MASK] 位置的 logits
    valid_logits = logits[valid_positions]  # (num_valid, vocab_size)
    valid_targets = mask_label[valid_positions]  # (num_valid,)，每個[MASK]的偽標籤token id
    
    # ========== Step 4: 計算 softmax 機率 ==========
    probs = F.softmax(valid_logits, dim=-1)  # (num_valid, vocab_size)
    
    # 取得每個位置對應偽標籤的預測機率
    # gather: 根據 valid_targets 的索引，從 probs 中取對應的機率值
    # unsqueeze(1): 增加一個維度，使 valid_targets 的 shape 為 (num_valid, 1)
    # probs.gather: 取出對應的機率，輸入為probs: (num_valid, vocab_size)
    # squeeze(1): 移除多餘的維度
    target_probs = probs.gather(dim=1, index=valid_targets.unsqueeze(1)).squeeze(1)  # (num_valid,)
    
    # ========== Step 5: 應用 threshold 過濾 ==========
    # 只有預測機率 > threshold 的位置才計入 loss
    confident_mask = target_probs > threshold  # (num_valid,)
    confident_token_count = int(confident_mask.sum().item())
    
    if not confident_mask.any():
        # 沒有任何位置超過閾值，回傳 0 loss
        zero_loss = torch.tensor(0.0, requires_grad=True, device=device)
        if return_stats:
            return zero_loss, valid_token_count, 0
        return zero_loss
    
    # ========== Step 6: 計算 Cross-Entropy loss ==========
    # 只對符合條件的位置計算 CE = -log(p)
    confident_probs = target_probs[confident_mask]  # (num_confident,)
    ce_losses = -torch.log(confident_probs + 1e-10)  # (num_confident,)
    
    # 回傳平均 loss
    loss = ce_losses.mean()
    if return_stats:
        return loss, valid_token_count, confident_token_count
    return loss


def compute_clause_embeddings(hidden_states, input_ids, probs, mode, tokenizer, return_info=False):
    """
    計算子句級別的嵌入向量 (核心函數，可被 prompt_bert 和 collect_*_statistics 重用)
    
    Args:
        hidden_states: BERT 最後一層輸出，shape (batch, 512, 768)
        input_ids: 輸入 token ids，shape (batch, 512)
        probs: softmax 機率，shape (batch, 512, vocab_size)
        mode: embedding 模式
            - 'emotion_clause': 使用情緒子句 embedding
            - 'cause_clause': 使用原因子句 embedding
            - 'emotion_clause_mask': 使用情緒子句 embedding + [MASK]_e embedding 的平均
            - 'cause_clause_mask': 使用原因子句 embedding + [MASK]_c embedding 的平均
        tokenizer: BertTokenizer 用於取得 token ids
        return_info: 是否回傳詳細的 embedding 資訊 (預設 False)
    
    Returns:
        embeddings: shape (batch, 768)
        embedding_info: (僅當 return_info=True) list of dict，每個 dict 包含:
            - used_clauses: 使用的子句數量
            - is_fallback: 是否 fallback 到 [CLS]
            - selected_clause_indices: 被選中的子句索引 (1-based)
            - selected_clause_texts: 被選中的子句文本內容
    """
    yes_token_id, no_token_id = get_binary_token_ids(tokenizer)
    mask_token_id = tokenizer.mask_token_id
    sep_token_id = tokenizer.sep_token_id
    
    batch_size = hidden_states.shape[0]
    embeddings = []
    embedding_info = [] if return_info else None
    
    for batch_idx in range(batch_size):
        doc_input = input_ids[batch_idx]
        doc_probs = probs[batch_idx].cpu().numpy()
        
        # 取得子句邊界
        boundaries = get_clause_boundaries(doc_input, mask_token_id, sep_token_id)
        
        # 收集情緒句或原因句的 embedding
        selected_embeddings = []
        selected_clause_indices = []  # 記錄被選中的子句索引 (1-based)
        selected_clause_texts = []    # 記錄被選中的子句文本
        
        for clause_idx, (content_start, content_end, mask_pos) in enumerate(boundaries):
            if is_emotion_or_cause_clause(doc_probs, mask_pos, mode, yes_token_id, no_token_id):
                # 對子句內容做 mean pooling
                clause_emb = hidden_states[batch_idx, content_start:content_end, :].mean(dim=0) # 沿著token方向取平均
                
                # 如果是 _mask 模式，結合 [MASK] embedding
                emotion_mask_pos, cause_mask_pos, _ = mask_pos
                if mode == 'emotion_clause_mask':
                    # 驗證 mask_pos 指向的是 [MASK] token (token_id = 103)
                    token_at_mask = doc_input[emotion_mask_pos].item() if isinstance(doc_input[emotion_mask_pos], torch.Tensor) else doc_input[emotion_mask_pos]
                    assert token_at_mask == mask_token_id, f"Expected [MASK] (103), got {token_at_mask} at position {emotion_mask_pos}"
                    mask_emb = hidden_states[batch_idx, emotion_mask_pos, :]
                    clause_emb = (clause_emb + mask_emb) / 2
                elif mode == 'cause_clause_mask':
                    # 驗證 mask_pos 指向的是 [MASK] token (token_id = 103)
                    token_at_mask = doc_input[cause_mask_pos].item() if isinstance(doc_input[cause_mask_pos], torch.Tensor) else doc_input[cause_mask_pos]
                    assert token_at_mask == mask_token_id, f"Expected [MASK] (103), got {token_at_mask} at position {cause_mask_pos}"
                    mask_emb = hidden_states[batch_idx, cause_mask_pos, :]
                    clause_emb = (clause_emb + mask_emb) / 2
                
                selected_embeddings.append(clause_emb)
                
                if return_info:
                    selected_clause_indices.append(clause_idx + 1)  # 1-based 索引
                    # 解碼子句文本
                    clause_token_ids = doc_input[content_start:content_end]
                    if isinstance(clause_token_ids, torch.Tensor):
                        clause_token_ids = clause_token_ids.cpu().numpy()
                    clause_text = tokenizer.decode(clause_token_ids, skip_special_tokens=True)
                    selected_clause_texts.append(clause_text)
        
        # 聚合或 fallback
        is_fallback = False
        if selected_embeddings:
            doc_emb = torch.stack(selected_embeddings).mean(dim=0)
        else:
            # Fallback: 使用 [CLS]
            doc_emb = hidden_states[batch_idx, 0, :]
            is_fallback = True
        
        embeddings.append(doc_emb)
        
        if return_info:
            embedding_info.append({
                'used_clauses': len(selected_embeddings),
                'is_fallback': is_fallback,
                'selected_clause_indices': selected_clause_indices,
                'selected_clause_texts': selected_clause_texts,
            })
    
    if return_info:
        return torch.stack(embeddings), embedding_info
    return torch.stack(embeddings)


def predict_pseudo_with_single_model(single_model, x_np, use_gpu, tokenizer, window_size):
    """使用單一模型對單一未標註樣本產生 pseudo tokens

    流程摘要: 
    1) 把單筆 numpy 輸入轉成 batch=1 的 tensor
    2) 用模型做前向推論取得 logits
    3) 將 logits 轉為機率分佈 (softmax)
    4) 呼叫既有的偽標籤生成函式，輸出 [MASK] 位置對應的 pseudo token 序列

    參數: 
    - single_model: 用來做推論的單一模型 (emo/cause/pair 任一)
    - x_np: 單筆文件的 input_ids (numpy array)
    - use_gpu: 是否使用 GPU
    - tokenizer: 目前模型對應 tokenizer (提供 mask token id 與 token id 工具)
    - window_size: pair 預測時可配對的句子視窗大小

    回傳: 
    - pseudo_tokens: numpy array，表示此模型對該文件的偽標籤 token 序列
    """
    # 將單筆 numpy 輸入轉成 PyTorch LongTensor，並加上 batch 維度
    # 原本 shape 例： (512,) -> (1, 512)
    x_tensor = torch.tensor(x_np, dtype=torch.long).unsqueeze(0)

    # 若啟用 GPU，將輸入搬到 CUDA，避免與模型裝置不一致
    if use_gpu:
        x_tensor = x_tensor.cuda()

    # 前向推論: labels=None 表示純推論，不計算監督損失
    # 回傳通常是 (loss, logits)，此處只需要 logits
    _, single_logits = single_model(x_tensor, labels=None)

    # 將 logits 轉為機率分佈，方便後續依機率挑選 pseudo label
    # shape 維持 (1, seq_len, vocab_size)
    single_logits = F.softmax(single_logits, dim=-1)

    # 取出 batch 中唯一一筆（索引 0），切斷梯度並轉成 CPU numpy
    # 後續的 generate_pseudo_label_tokens 以 numpy 邏輯處理
    logits_np = single_logits[0].detach().cpu().numpy()

    # 取得二元標記中「是」對應的 token id
    # 目前函式只用 yes_token_id; 第二個回傳值 (no_token_id) 此處不使用
    yes_token_id, _ = get_binary_token_ids(tokenizer)

    # 依據模型機率與規則產生偽標籤序列: 
    # - input_ids: 原始輸入 (含 [MASK])
    # - mask_token_id: 指定哪些位置要被填入 pseudo token
    # - window_size: 限制 pair 可選範圍
    # - threshold=None: 此處不做信心閾值截斷 (由外層策略決定要不要過濾)
    pseudo_tokens = generate_pseudo_label_tokens(
        logits=logits_np,
        input_ids=np.array(x_np, dtype=np.int64),
        mask_token_id=tokenizer.mask_token_id,
        yes_token_id=yes_token_id,
        window_size=window_size,
        tokenizer=tokenizer,
        threshold=None,
    )

    # 回傳該模型對此樣本的最終 pseudo token 序列
    return pseudo_tokens


def apply_task_consistency_filter(
    x_np,
    default_pseudo_tokens,
    consistency_enabled,
    consistency_models,
    consistency_stats,
    use_gpu,
    tokenizer,
    window_size,
    consistency_rule='all_equal',
    doc_id=None,
    consistency_debug_records=None,
):
    """一致性過濾: 支援 all-equal 與保守拼接兩種規則

    參數說明: 
    - x_np: 單筆未標註樣本的 input_ids (numpy array)
    - default_pseudo_tokens: 主模型 (目前訓練中的模型) 已經產生好的偽標籤
    - consistency_enabled: 本輪是否啟用一致性過濾
    - consistency_models: 一致性用三模型 (emo_model, cause_model, pair_model)
    - consistency_stats: 統計字典，包含 checked/accept/reject/skip_unavailable
    - use_gpu/tokenizer/window_size: 推論與偽標籤解碼所需設定

    回傳: 
    - (True, pseudo_tokens): 通過一致性，且回傳可用偽標籤
    - (False, None): 未通過一致性，呼叫端應丟棄該樣本
    """
    # 情況 1: 若本輪未啟用一致性，直接沿用主模型輸出的偽標籤
    # 這代表「不做額外過濾」，讓資料照原本邏輯進入訓練
    if not consistency_enabled:
        return True, default_pseudo_tokens

    # 到這裡代表: 一致性啟用且三模型已可用，正式進行一次一致性檢查
    consistency_stats["checked"] += 1

    # 取出三個任務最佳模型: 情緒、原因、配對
    emo_model, cause_model, pair_model = consistency_models

    # 讓三個模型分別對同一筆 x_np 產生完整 pseudo token 序列
    # 注意：這裡比較的是「整個序列」是否一致，而不是只看單一位置
    emo_pseudo = predict_pseudo_with_single_model(emo_model, x_np, use_gpu, tokenizer, window_size)
    cause_pseudo = predict_pseudo_with_single_model(cause_model, x_np, use_gpu, tokenizer, window_size)
    pair_pseudo = predict_pseudo_with_single_model(pair_model, x_np, use_gpu, tokenizer, window_size)

    # 規則一: 三份 pseudo token 必須「完全相同」才視為一致
    # np.array_equal 會同時檢查 shape 與每個位置的值
    all_equal = np.array_equal(emo_pseudo, cause_pseudo) and np.array_equal(cause_pseudo, pair_pseudo)

    # 規則二(SOMC 拼接):
    # - [MASK]_e(情緒) 用 emo 模型
    # - [MASK]_c(原因) 用 cause 模型
    # - [MASK]_p(配對) 用 pair 模型
    # 進行子任務分工拼接
    mask_ensemble_valid = False  # SOMC 是否成功產生可用的拼接結果，預設先視為失敗
    mask_ensemble_pseudo = None  # 保存 SOMC 拼接後的 pseudo；尚未拼接前為 None
    if consistency_rule == 'somc':  # 僅在規則選擇 SOMC 時執行下面的子任務拼接流程
        if not (len(emo_pseudo) == len(cause_pseudo) == len(pair_pseudo)):  # 三模型輸出長度必須一致，否則視為資料結構異常並立即中斷
            raise ValueError(
                f"[Task Consistency][SOMC] length_mismatch: "
                f"len(emo)={len(emo_pseudo)}, len(cause)={len(cause_pseudo)}, len(pair)={len(pair_pseudo)}, doc_id={doc_id}"
            )
        elif len(emo_pseudo) % 3 != 0:  # 每個 clause 預期有 3 個 MASK 槽位（e/c/p），不符合即視為模板結構異常並立即中斷
            raise ValueError(
                f"[Task Consistency][SOMC] mask_triplet_mismatch: "
                f"len(pseudo)={len(emo_pseudo)} is not divisible by 3, doc_id={doc_id}"
            )
        else:
            merged = np.array(emo_pseudo, dtype=np.int64).copy()  # 先複製 emo 結果作為底稿，保留 [MASK]_e（索引 0,3,6,...）
            merged[1::3] = np.array(cause_pseudo, dtype=np.int64)[1::3]  # 把 [MASK]_c（索引 1,4,7,...）改用 cause 模型預測
            merged[2::3] = np.array(pair_pseudo, dtype=np.int64)[2::3]  # 把 [MASK]_p（索引 2,5,8,...）改用 pair 模型預測
            mask_ensemble_valid = True  # 到這裡代表 SOMC 拼接成功，結果可用於後續流程
            mask_ensemble_pseudo = merged  # 儲存拼接完成的 pseudo token 序列

    # 產生可讀版解碼結果 (便於離線檢查)
    # - *_tokens: token
    # - *_decoded: 將 token 序列轉回字串
    emo_tokens = tokenizer.convert_ids_to_tokens(emo_pseudo.tolist())
    cause_tokens = tokenizer.convert_ids_to_tokens(cause_pseudo.tolist())
    pair_tokens = tokenizer.convert_ids_to_tokens(pair_pseudo.tolist())

    emo_decoded = tokenizer.convert_tokens_to_string(emo_tokens)
    cause_decoded = tokenizer.convert_tokens_to_string(cause_tokens)
    pair_decoded = tokenizer.convert_tokens_to_string(pair_tokens)

    # 若使用者啟用詳細記錄，保存三模型各自預測與一致性結果 (供離線分析)
    if consistency_debug_records is not None:
        consistency_debug_records.append({
            "doc_id": doc_id,
            "consistency_rule": consistency_rule,
            "all_equal": bool(all_equal),
            "mask_ensemble_valid": bool(mask_ensemble_valid),
            "emo_pseudo_ids": emo_pseudo.tolist(),
            "emo_pseudo_tokens": emo_tokens,
            "emo_pseudo_decoded": emo_decoded,
            "cause_pseudo_ids": cause_pseudo.tolist(),
            "cause_pseudo_tokens": cause_tokens,
            "cause_pseudo_decoded": cause_decoded,
            "pair_pseudo_ids": pair_pseudo.tolist(),
            "pair_pseudo_tokens": pair_tokens,
            "pair_pseudo_decoded": pair_decoded,
            "mask_ensemble_pseudo_ids": mask_ensemble_pseudo.tolist() if mask_ensemble_pseudo is not None else None,
        })

    if consistency_rule == 'somc':
        if mask_ensemble_valid and mask_ensemble_pseudo is not None:
            consistency_stats["accept"] += 1
            return True, mask_ensemble_pseudo
    else:
        if all_equal:
            # 通過一致性: accept 計數 +1，並回傳任一份 pseudo (此處回 emo_pseudo)
            # 因為三者已確認相同，所以回傳哪一份都等價
            consistency_stats["accept"] += 1
            return True, emo_pseudo

    # 未通過一致性: reject 計數 +1，回傳 (False, None) 讓外層丟棄該樣本
    consistency_stats["reject"] += 1
    return False, None


"""setting agrparse"""
parser = argparse.ArgumentParser(description='Training')

"""model struct"""
parser.add_argument('--n_hidden', type=int, default=100, help='number of hidden unit')
parser.add_argument('--n_class', type=int, default=2, help='number of distinct class')
parser.add_argument('--window_size', type=int, default=2, help='size of the emotion cause pair window')
parser.add_argument('--feature_layer', type=int, default=3, help='number of layer iterations')
parser.add_argument('--log_file_name', type=str, default='log', help='name of log file')
parser.add_argument('--model_type', type=str, default='ISML', help='type of model')
"""training"""
parser.add_argument('--training_iter', type=int, default=20, help='number of train iterator')
parser.add_argument('--scope', type=str, default='Ind_BiLSTM', help='scope')
parser.add_argument('--batch_size', type=int, default=8, help='number of example per batch')
parser.add_argument('--learning_rate', type=float, default=0.00001, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.01, help='weight decay for bert')
parser.add_argument('--usegpu', type=bool, default=True, help='gpu')
"""other"""
parser.add_argument('--test_only', type=bool, default=False, help='no training')
parser.add_argument('--checkpoint', type=bool, default=False, help='load checkpoint')
parser.add_argument('--checkpointpath', type=str, default='checkpoint/ECPE/', help='path to load checkpoint')
parser.add_argument('--skip_initial_training', type=bool, default=False, help='skip initial supervised training and go directly to self-training')
parser.add_argument('--savecheckpoint', type=bool, default=True, help='save checkpoint')
parser.add_argument('--save_path', type=str, default='prompt_ECPE_few_shot_ST', help='path to save checkpoint')
parser.add_argument('--experiment_output_dir', type=str, default=None, help='directory to save test results with experiment parameters in name')
parser.add_argument('--device', type=str, default='0', help='device id')
parser.add_argument('--dataset', type=str, default='split10_home_train1_test1_val1_unlabeled7_disjoint/', help='path for dataset')
parser.add_argument('--start_fold', type=int, default=1)
parser.add_argument('--end_fold',   type=int, default=10)
parser.add_argument('--seed', type=int, default=42, help='random seed for reproducibility (default: 42)')
parser.add_argument('--retain_pseudo_in_unlabeled', action='store_true',
                    help='keep pseudo-labeled samples in the unlabeled pool instead of removing them after selection')
"""self-training parameters"""
parser.add_argument('--threshold', type=float, default=0.9, help='confidence threshold for pseudo-labeling')
parser.add_argument(
    '--mask_threshold_mode',
    type=str,
    default='both',
    choices=['emotion', 'cause', 'both', 'or', 'all'],
    help='which [MASK] positions must satisfy the confidence threshold: emotion-only, cause-only, both (emotion+cause), or (emotion or cause), all (emotion+cause+pair)'
)
parser.add_argument('--gamma', type=float, default=0.5, help='偽標籤 loss 的權重 (對應論文的 λ，預設 0.5)')
parser.add_argument('--self_training_rounds', type=int, default=10,help='number of self-training rounds')
parser.add_argument('--st_training_epochs', type=int, default=3, help='number of epochs to train in each self-training round')
parser.add_argument('--initial_supervised_metric', type=str, default='pair', choices=['pair', 'emotion', 'cause'],
                    help='初始監督訓練結束後，進入 self-training 前要載入的最佳模型指標: pair=最佳 pair F1, emotion=最佳 emotion F1, cause=最佳 cause F1')
parser.add_argument('--self_training_main_metric', type=str, default='pair', choices=['pair', 'emotion', 'cause'],
                    help='self-training 每輪主模型切換依據的驗證指標: pair=最佳 pair F1, emotion=最佳 emotion F1, cause=最佳 cause F1')
parser.add_argument('--test_metric', type=str, default='pair', choices=['pair', 'emotion', 'cause'],
                    help='最終測試時要載入的最佳模型指標: 對 initial/self_training 生效；若 test_model_type=self_training_avg3 則忽略')
parser.add_argument('--test_model_type', type=str, default='initial', choices=['initial', 'self_training', 'self_training_avg3'], 
                    help='which model to test: initial (監督訓練最佳模型), self_training (self-training 最佳模型), or self_training_avg3 (simple-mean average of self_training_best_emo/cause/pair)')
parser.add_argument('--final_eval_split', type=str, default='test', choices=['test', 'val'],
                    help='test_only 或最終評估時使用的資料切分，預設為 test；若要統計最終 Avg3 驗證 Pair F1 可設為 val')
parser.add_argument('--log_pseudo_quality', action='store_true',
                    help='compare pseudo labels against ground truth when available and persist error statistics')
parser.add_argument('--pseudo_selector', type=str, default='threshold', choices=['threshold', 'nest', 'random', 'hybrid', 'consistency_only'],
                    help='偽標籤選擇策略: 使用信心度閥值、NeST 選擇器、隨機選擇、混合模式(Hybrid)或僅用一致性(Consistency-only)')
parser.add_argument('--consistency_pseudo', action='store_true',
                    help='啟用 task consistency 偽標籤篩選: 多模型預測一致才納入訓練')
parser.add_argument('--consistency_require_all_equal', action='store_true',
                    help='一致性規則: 要求所有模型預測完全一致 (預設啟用)')
parser.add_argument('--consistency_all_equal_round0_only', action='store_true',
                    help='僅在 self-training 第 1 輪套用 all_equal 一致性篩選，後續輪次自動關閉')
parser.add_argument('--consistency_rule', type=str, default='all_equal',
                    choices=['all_equal', 'somc'],
                    help='一致性規則: all_equal=三模型整串相同才通過; somc(Subtask Optimal Model Collaboration)=三模型分工拼接')
parser.add_argument('--consistency_model_source', type=str, default='self_training_best',
                    choices=['supervised_best', 'self_training_best'],
                    help='一致性篩選時使用的模型來源')
parser.add_argument('--save_consistency_predictions', action='store_true',
                    help='儲存 task consistency 三模型各自預測結果 (每輪一個 JSON)')
parser.add_argument('--save_val_predictions', action='store_true',
                    help='是否將驗證集預測結果另存為 text_result 格式檔案')
parser.add_argument('--no_save_val_predictions', action='store_false', dest='save_val_predictions',
                    help='停用驗證集預測結果輸出')
parser.add_argument('--save_task_specific_checkpoints', action='store_true', dest='save_task_specific_checkpoints',
                    help='強制儲存 emotion/cause 專用最佳 checkpoint (不受 consistency_pseudo 影響)')
parser.add_argument('--no_save_task_specific_checkpoints', action='store_false', dest='save_task_specific_checkpoints',
                    help='停用 emotion/cause 專用最佳 checkpoint 儲存 (即使 consistency_pseudo 啟用)')
parser.add_argument('--hybrid_switch_round', type=int, default=5,
                    help='Hybrid 模式下，切換選擇器的輪次分界點')
parser.add_argument('--hybrid_reverse', action='store_true',
                    help='反轉 Hybrid 策略順序: 前期使用 NeST，後期切換為 Threshold (預設為 False: 前期 Threshold 後期 NeST)')
parser.add_argument('--hybrid_alternating', action='store_true',
                    help='交替 Hybrid 策略：奇數輪使用 Threshold，偶數輪使用 NeST (預設為 False)')
parser.add_argument('--nest_k', type=int, default=5,
                    help='NeST 選擇器使用的鄰域數量 (KNN k 值)')
parser.add_argument('--nest_beta', type=float, default=0.1,
                    help='NeST 散度評分中標記樣本差異度的權重')
parser.add_argument('--nest_m', type=float, default=0.6,
                    help='多輪次選擇中聚合散度分數的動量因子')
parser.add_argument('--nest_multiplier', type=float, default=3.0,
                    help='決定 NeST 每輪要選擇的樣本數 (標記集大小的倍數)')
parser.add_argument('--nest_divergence_mode', type=str, default='cause', choices=['cause', 'emotion', 'both', 'emotion_clause', 'cause_clause'],
                    help='NeST 散度計算模式: cause=只用原因標籤, emotion=只用情緒標籤, both=兩者平均, emotion_clause=只用預測為情緒句的子句, cause_clause=只用預測為原因句的子句')
parser.add_argument('--knn_embedding_mode', type=str, default='cls',
                    choices=['cls', 'emotion_clause', 'cause_clause', 'emotion_clause_mask', 'cause_clause_mask'],
                    help='KNN 使用的 embedding 類型: cls=傳統[CLS], emotion_clause=情緒子句聚合, cause_clause=原因子句聚合, emotion_clause_mask=情緒子句+[MASK]_e平均, cause_clause_mask=原因子句+[MASK]_c平均')
parser.add_argument('--self_training_eps', type=float, default=0.6,
                    help='NeST 選擇器中的距離穩定項 (保留舊參數以相容既有腳本)')
parser.add_argument('--nest_loss_mode', type=str, default='nest',
                    choices=['standard', 'nest', 'nest_dynamic_gamma_round0'],
                    help='NeST loss 計算模式: standard=所有偽標籤都計入, nest=只有信心>γ才計入, nest_dynamic_gamma_round0=搭配 consistency_all_equal_round0_only 時第1輪使用 gamma=0.1')
parser.add_argument('--nest_loss_threshold', type=float, default=0.9,
                    help='nest 模式下的信心閾值 (對應論文的 γ，預設 0.9)')

parser.set_defaults(save_val_predictions=True, save_task_specific_checkpoints=None) # 代表沒特別傳參數時，預設會儲存驗證集預測結果

opt = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = opt.device

METRIC_TO_CHECKPOINT_SUFFIX = {
    'pair': 'pair',
    'emotion': 'emo',
    'cause': 'cause',
}

METRIC_TO_LABEL = {
    'pair': 'Pair F1',
    'emotion': 'Emotion F1',
    'cause': 'Cause F1',
}


def get_supervised_best_checkpoint_path(base_dir, fold, metric):
    suffix = METRIC_TO_CHECKPOINT_SUFFIX[metric]
    return os.path.join(base_dir, f'fold{fold}_best_{suffix}.pth')


def get_self_training_best_checkpoint_path(base_dir, fold, metric):
    suffix = METRIC_TO_CHECKPOINT_SUFFIX[metric]
    return os.path.join(base_dir, 'self_training_models', f'fold{fold}_self_training_best_{suffix}.pth')


def get_test_checkpoint_path(base_dir, fold, test_model_type, metric):
    if test_model_type == 'initial':
        return get_supervised_best_checkpoint_path(base_dir, fold, metric)
    if test_model_type == 'self_training':
        return get_self_training_best_checkpoint_path(base_dir, fold, metric)
    raise ValueError(f"不支援以 metric 載入的 test_model_type: {test_model_type}")


def get_metric_folder_tags(opt):
    metric_suffix = {
        'emotion': 'emo',
        'cause': 'cau',
    }
    tags = []

    if opt.initial_supervised_metric != 'pair':
        tags.append(f"init{metric_suffix[opt.initial_supervised_metric]}")

    if opt.self_training_main_metric != 'pair':
        tags.append(f"st{metric_suffix[opt.self_training_main_metric]}")

    if opt.test_model_type in ('initial', 'self_training') and opt.test_metric != 'pair':
        tags.append(f"test{metric_suffix[opt.test_metric]}")

    return tags

# consistency-only 模式下，強制啟用 task consistency 篩選
if opt.pseudo_selector == 'consistency_only':
    opt.consistency_pseudo = True

# 若啟用 consistency_pseudo 且使用 all_equal 規則，未明確提供 all-equal 時預設啟用嚴格一致
if opt.consistency_pseudo and opt.consistency_rule == 'all_equal' and not opt.consistency_require_all_equal:
    opt.consistency_require_all_equal = True

# round0-only 模式僅對 all_equal 規則生效
if opt.consistency_all_equal_round0_only and opt.consistency_rule != 'all_equal':
    print("[Task Consistency] --consistency_all_equal_round0_only 僅適用於 --consistency_rule all_equal，已忽略此設定")

# 設置隨機種子以確保實驗可重現性
# 必須在任何模型初始化和數據載入之前設定
set_random_seed(opt.seed)

# 如果使用預設的 save_path，則自動添加參數後綴
if opt.save_path == 'prompt_ECPE_few_shot_ST' and not opt.test_only:
    # 生成時間戳記
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    
    # 格式化學習率
    lr_str = f"{opt.learning_rate:.0e}".replace("e-0", "e-").replace("e+0", "e+")
    
    # 生成參數化的 save_path
    retain_pseudo_str = "retain_pseudo" if opt.retain_pseudo_in_unlabeled else "remove_pseudo"
    if opt.consistency_pseudo:
        if opt.consistency_rule == 'all_equal':
            consistency_str = "consistency_r0only" if opt.consistency_all_equal_round0_only else "consistency"
        else:
            consistency_str = f"consistency_{opt.consistency_rule}"
    else:
        consistency_str = ""
    
    # 根據偽標籤選擇策略決定顯示的模式
    if opt.pseudo_selector == 'hybrid':
        if opt.hybrid_alternating:
            selector_str = "hybrid_alt"
        else:
            rev_str = "rev_" if opt.hybrid_reverse else ""
            selector_str = f"hybrid_{rev_str}switch{opt.hybrid_switch_round}"
        # Hybrid 模式下，NeST 參數也需要記錄
        knn_emb_str = f"_knn{opt.knn_embedding_mode}" if opt.knn_embedding_mode != 'cls' else ""
        nest_str = f"_nest_k{opt.nest_k}_{opt.nest_divergence_mode}{knn_emb_str}"
        selector_str += nest_str
        
        # 混合模式初期使用 threshold，後期使用 nest multiplier
        threshold_str = f"th{opt.threshold}_"
        nest_multiplier_str = f"_nestmul{opt.nest_multiplier:g}".replace(".", "p")
        nest_params_str = f"_nbeta{opt.nest_beta}_nm{opt.nest_m}"
    elif opt.pseudo_selector == 'nest':
        # 加入 knn_embedding_mode 以區分不同 embedding 模式 (cls, emotion_clause, cause_clause)
        knn_emb_str = f"_knn{opt.knn_embedding_mode}" if opt.knn_embedding_mode != 'cls' else ""
        selector_str = f"nest_k{opt.nest_k}_{opt.nest_divergence_mode}{knn_emb_str}"
        threshold_str = ""
        # 額外記錄 NeST 選樣倍率 (避免小數點造成路徑/排序問題，例如 0.5 -> nestmul0p5)
        nest_multiplier_str = f"_nestmul{opt.nest_multiplier:g}".replace(".", "p")
        # NeST 相關參數 (beta 和 m)
        nest_params_str = f"_nbeta{opt.nest_beta}_nm{opt.nest_m}"
    elif opt.pseudo_selector == 'random':
        # 隨機選擇模式
        selector_str = "random"
        threshold_str = ""
        # 隨機選擇也使用 nest_multiplier 控制數量
        nest_multiplier_str = f"_nestmul{opt.nest_multiplier:g}".replace(".", "p")
        nest_params_str = ""  # random 模式不顯示 nest 參數
    elif opt.pseudo_selector == 'consistency_only':
        # 只使用 task consistency 做偽標籤篩選
        selector_str = f"consistency_only_{opt.consistency_rule}"
        threshold_str = ""
        nest_multiplier_str = ""
        nest_params_str = ""
    else:  # threshold
        selector_str = f"mask{opt.mask_threshold_mode}"
        threshold_str = f"th{opt.threshold}_"
        nest_multiplier_str = ""  # threshold 模式下不顯示 nest_multiplier
        nest_params_str = ""  # threshold 模式不顯示 nest 參數

    # 生成 nest_loss_mode 字串 (非 standard 模式才顯示)
    nlm_str = f"_nlm{opt.nest_loss_mode}" if opt.nest_loss_mode != 'standard' else ""
    # 若測試模型採用 Avg3，將模式標記加到 nlm 後面，方便從資料夾名辨識
    if nlm_str and opt.test_model_type == 'self_training_avg3':
        nlm_str += "_avgs_simple"
    # 當 nest_loss_threshold 不是預設 0.9 時，額外把門檻值寫入資料夾名稱
    if opt.nest_loss_mode in ('nest', 'nest_dynamic_gamma_round0') and abs(opt.nest_loss_threshold - 0.9) > 1e-12:
        nlt_str = f"{opt.nest_loss_threshold:g}".replace(".", "p")
        nlm_str += f"_nlt_{nlt_str}"
    
    # 從 dataset 路徑提取簡短名稱 (移除尾部斜線後取最後一個路徑部分)
    dataset_name = os.path.basename(opt.dataset.rstrip('/'))
    # 生成 dataset 簡短名稱作為父目錄 (將 train/test/val/unlabeled 縮寫為 t/te/v/u)
    dataset_short = dataset_name.replace('train', 't').replace('_test', 'te').replace('_val', 'v').replace('unlabeled', 'u')
    parent_dir = f"ep_{dataset_short}"
    
    # 定義自訓練相關參數字串
    gamma_str = f"_gamma{opt.gamma}"
    ste_str = f"_ste{opt.st_training_epochs}"
    
    # 邏輯: 當 opt.self_training_rounds == 0 (純 Few-shot 監督式訓練) 時，隱藏所有自訓練參數
    if opt.self_training_rounds == 0:
        selector_str = ""
        threshold_str = ""
        nest_multiplier_str = ""
        nest_params_str = ""
        gamma_str = ""
        nlm_str = ""
        retain_pseudo_str = ""  # 純監督訓練沒有偽標籤需要保留或移除
        ste_str = ""

    # 使用 list 組合資料夾名稱，避免多餘底線
    folder_components = [
        f"prompt_ECPE_few_shot_ST_{timestamp}_f{opt.start_fold}-{opt.end_fold}",
        f"i{opt.training_iter}_lr{lr_str}_bs{opt.batch_size}",
        f"wd{opt.weight_decay}"
    ]

    # 依序加入非空組件
    if threshold_str: folder_components.append(threshold_str.rstrip('_'))
    if selector_str: folder_components.append(selector_str)
    if nest_multiplier_str: folder_components.append(nest_multiplier_str.lstrip('_'))
    if nest_params_str: folder_components.append(nest_params_str.lstrip('_'))
    if gamma_str: folder_components.append(gamma_str.lstrip('_'))
    if nlm_str: folder_components.append(nlm_str.lstrip('_'))

    folder_components.append(f"st{opt.self_training_rounds}")

    if ste_str: folder_components.append(ste_str.lstrip('_'))

    folder_components.extend(get_metric_folder_tags(opt))
    folder_components.append(f"seed{opt.seed}")

    if retain_pseudo_str: folder_components.append(retain_pseudo_str) # retain_pseudo_str 本身不含底線
    if consistency_str: folder_components.append(consistency_str)

    folder_components.append("v2")
    folder_components.append("CE")

    folder_name = "_".join(folder_components)
    
    # 完整路徑: ep_{dataset_short}/{folder_name}
    opt.save_path = os.path.join(parent_dir, folder_name)

# 動態生成實驗資料夾名稱
def generate_experiment_folder_name(opt, bert_path):
    """根據實驗參數生成資料夾名稱"""
    # 格式化學習率（避免科學記號造成的問題）
    lr_str = f"{opt.learning_rate:.0e}".replace("e-0", "e-").replace("e+0", "e+")
    
    
    # 生成時間戳記 (YYYY_MM_DD_HH_MM_SS)
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    
    # 從 bert_path 擷取模型名稱
    model_name = os.path.basename(bert_path.rstrip('/'))  # 移除末尾的斜線並取得最後一個路徑部分
    
    # 根據偽標籤選擇策略決定顯示的模式
    if opt.pseudo_selector == 'hybrid':
        if opt.hybrid_alternating:
            selector_str = "hybrid_alt"
        else:
            rev_str = "rev_" if opt.hybrid_reverse else ""
            selector_str = f"hybrid_{rev_str}switch{opt.hybrid_switch_round}"
        # Hybrid 模式下，NeST 參數也需要記錄
        knn_emb_str = f"_knn{opt.knn_embedding_mode}" if opt.knn_embedding_mode != 'cls' else ""
        nest_str = f"_nest_k{opt.nest_k}_{opt.nest_divergence_mode}{knn_emb_str}"
        selector_str += nest_str
        
        # 混合模式初期使用 threshold，後期使用 nest multiplier
        threshold_str = f"th{opt.threshold}_"
        nest_multiplier_str = f"_nestmul{opt.nest_multiplier:g}".replace(".", "p")
        nest_params_str = f"_nbeta{opt.nest_beta}_nm{opt.nest_m}"
    elif opt.pseudo_selector == 'nest':
        # 加入 knn_embedding_mode 以區分不同 embedding 模式 (cls, emotion_clause, cause_clause)
        knn_emb_str = f"_knn{opt.knn_embedding_mode}" if opt.knn_embedding_mode != 'cls' else ""
        selector_str = f"nest_k{opt.nest_k}_{opt.nest_divergence_mode}{knn_emb_str}"
        threshold_str = ""
        # 額外記錄 NeST 選樣倍率 (避免小數點造成路徑/排序問題，例如 0.5 -> nestmul0p5)
        nest_multiplier_str = f"_nestmul{opt.nest_multiplier:g}".replace(".", "p")
        # NeST 相關參數 (beta 和 m)
        nest_params_str = f"_nbeta{opt.nest_beta}_nm{opt.nest_m}"
    elif opt.pseudo_selector == 'random':
        # 隨機選擇模式
        selector_str = "random"
        threshold_str = ""
        # 隨機選擇也使用 nest_multiplier 控制數量
        nest_multiplier_str = f"_nestmul{opt.nest_multiplier:g}".replace(".", "p")
        nest_params_str = ""  # random 模式不顯示 nest 參數
    elif opt.pseudo_selector == 'consistency_only':
        # 只使用 task consistency 做偽標籤篩選
        selector_str = f"consistency_only_{opt.consistency_rule}"
        threshold_str = ""
        nest_multiplier_str = ""
        nest_params_str = ""
    else:  # threshold
        selector_str = f"mask{opt.mask_threshold_mode}"
        threshold_str = f"th{opt.threshold}_"
        nest_multiplier_str = ""  # threshold 模式下不顯示 nest_multiplier
        nest_params_str = ""  # threshold 模式不顯示 nest 參數
    
    # 從 dataset 路徑提取簡短名稱（移除尾部斜線後取最後一個路徑部分）
    dataset_name = os.path.basename(opt.dataset.rstrip('/'))
    # 生成 dataset 簡短名稱作為父目錄 (將 train/test/val/unlabeled 縮寫為 t/te/v/u)
    dataset_short = dataset_name.replace('train', 't').replace('test', 'te').replace('val', 'v').replace('unlabeled', 'u')
    parent_dir = f"ep_{dataset_short}"
    
    # 生成 nest_loss_mode 字串 (非 standard 模式才顯示)
    nlm_str = f"_nlm{opt.nest_loss_mode}" if opt.nest_loss_mode != 'standard' else ""
    # 若測試模型採用 Avg3，將模式標記加到 nlm 後面，方便從資料夾名辨識
    if nlm_str and opt.test_model_type == 'self_training_avg3':
        nlm_str += "_avgs_simple"
    # 當 nest_loss_threshold 不是預設 0.9 時，額外把門檻值寫入資料夾名稱
    if opt.nest_loss_mode in ('nest', 'nest_dynamic_gamma_round0') and abs(opt.nest_loss_threshold - 0.9) > 1e-12:
        nlt_str = f"{opt.nest_loss_threshold:g}".replace(".", "p")
        nlm_str += f"_nlt_{nlt_str}"
    
    # 定義資料夾名稱組成元素
    if opt.consistency_pseudo:
        if opt.consistency_rule == 'all_equal':
            consistency_str = "consistency_r0only" if opt.consistency_all_equal_round0_only else "consistency"
        else:
            consistency_str = f"consistency_{opt.consistency_rule}"
    else:
        consistency_str = ""
    folder_components = [
        f"UECA-CE_ST_{timestamp}",
        f"f{opt.start_fold}-{opt.end_fold}",
        f"i{opt.training_iter}",
        f"lr{lr_str}",
        f"bs{opt.batch_size}",
        f"wd{opt.weight_decay}",
        f"{threshold_str}",
        f"{selector_str}",
        f"{nest_multiplier_str}",
        f"{nest_params_str}",
        f"gamma{opt.gamma}",
        f"{nlm_str}",
        f"st{opt.self_training_rounds}",
        f"ste{opt.st_training_epochs}",
        f"{consistency_str}",
        f"test_model_type_{opt.test_model_type}",
        "v2",
        "CE"
    ]

    folder_components.extend(get_metric_folder_tags(opt))
    folder_components.append(f"seed{opt.seed}")

    # 如果有指定模型名稱，加入到資料夾名稱中 (放在前面較顯眼的位置)
    if model_name:
        folder_components.insert(1, f"{model_name}")

    # 組合資料夾名稱
    folder_name = "_".join(filter(None, folder_components)) # filter removes empty strings if any
    
    return os.path.join(parent_dir, folder_name)

use_gpu = False
if opt.usegpu and torch.cuda.is_available():
    use_gpu = True
device = torch.device('cuda' if use_gpu else 'cpu')


def print_time():
    print('\n----------{}----------'.format(time.strftime("%Y-%m-%d %X", time.localtime())))


class MyDataset(Dataset):
    def __init__(self, input_file, test=False, tokenizer=None):
        print('load data_file: {}'.format(input_file))
        self.x_bert, self.y_bert, self.label, self.mask_label = [], [], [], []
        self.gt_emotion, self.gt_cause, self.gt_pair = [], [], []
        self.doc_id = []
        self.raw_text = []  # 新增: 儲存原始文本 (用於 debug 驗證)
        self.pairs = []  # 儲存每篇文檔的情緒-原因配對 [(emotion_idx, cause_idx), ...]
        self.test = test
        self.n_cut = 0
        self.tokenizer = tokenizer
        cnt_over_limit = 0
        with open(input_file, 'r', encoding='utf8') as f:
            data = json.load(f)

        for doc in data:
            doc_id = doc["doc_id"]
            self.doc_id.append(doc_id)
            d_len = doc["doc_len"]
            pairs = doc["pairs"]
            pos, cause = zip(*pairs) if pairs else ([], [])
            pairs = [tuple(pair) for pair in pairs]  # 將每個子列表轉換為元組
            self.pairs.append(pairs)  # 儲存該文檔的所有情緒-原因配對

            full_document = ""
            mask_full_document = ""
            mask_label_full_document = ""
            part_sentence = []
            emotions = []
            cnt_emotion_gt = 0
            cnt_cause_gt = 0
            cnt_pair_gt = 0

            # 處理每個子句
            for clause in doc["clauses"]:
                emotion = clause["emotion_category"].strip()
                emotions.append(emotion)
                part_sentence.append(clause["clause"])

            cnt_emotion_gt = len(set(pos)) # 紀錄有幾個不同的情緒子句
            cnt_cause_gt = len(set(cause)) # 紀錄有幾個不同的原因子句
            cnt_pair_gt = len(set(pairs))   # 紀錄有幾個不同的情緒-原因配對
            self.gt_emotion.append(cnt_emotion_gt)
            self.gt_pair.append(cnt_pair_gt)
            self.gt_cause.append(cnt_cause_gt)
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

                full_document = full_document + '[SEP]' # 作為標準答案
                mask_full_document = mask_full_document + "[MASK] [MASK] [MASK] [SEP]" # 給模型的輸入(題目)
                mask_label_full_document = mask_label_full_document + "[MASK] [MASK] [MASK] [SEP]"
            if (self.tokenizer.encode_plus(mask_full_document, return_tensors="pt")['input_ids'][0].shape !=
                    self.tokenizer.encode_plus(mask_full_document, return_tensors="pt")['input_ids'][0].shape):
                print('length wrong')

            count_len = len(self.tokenizer.encode_plus(mask_full_document, return_tensors="pt")['input_ids'][0])
            if count_len > 512:
                print("Over limit length{} document{} - 移除此文檔".format(count_len, doc_id))
                cnt_over_limit += 1
                self.n_cut += 1
                # 移除已添加的所有相關數據，保持數據一致性
                self.doc_id.pop()
                self.gt_emotion.pop()
                self.gt_cause.pop()
                self.gt_pair.pop()
                continue
            mask_full_document = \
                self.tokenizer.encode_plus(mask_full_document, return_tensors="pt", max_length=512, truncation=True,
                                           pad_to_max_length=True)['input_ids']
            full_document = \
                self.tokenizer.encode_plus(full_document, return_tensors="pt", max_length=512, truncation=True,
                                           pad_to_max_length=True)['input_ids']
            mask_label_full_document = \
                self.tokenizer.encode_plus(mask_label_full_document, return_tensors="pt", max_length=512,
                                           truncation=True,
                                           pad_to_max_length=True)['input_ids']
            labels = full_document.masked_fill(mask_full_document != 103, -100)
            mask_labels = full_document.masked_fill(mask_label_full_document != 103, -100)

            # 儲存原始文本 (用於 debug 驗證)
            raw_text_doc = " ".join([f"[{i+1}] {part_sentence[i]}" for i in range(d_len)])
            self.raw_text.append(raw_text_doc)
            
            self.x_bert.append(np.array(mask_full_document[0]))
            self.y_bert.append(np.array(full_document[0]))
            self.label.append(np.array(labels[0]))
            self.mask_label.append(np.array(mask_labels[0]))
        self.x_bert, self.y_bert, self.label, self.mask_label = map(np.array, [self.x_bert, self.y_bert, self.label,
                                                                               self.mask_label])
        self.gt_emotion, self.gt_cause, self.gt_pair = map(np.array, [self.gt_emotion, self.gt_cause, self.gt_pair])
        for var in ['self.x_bert', 'self.y_bert', 'self.label', 'self.mask_label', 'self.gt_emotion', 'self.gt_cause',
                    'self.gt_pair']:
            print('{}.shape {}'.format(var, eval(var).shape))
        print('MyDataset: n_cut {}, over_limit_count {}, final_dataset_size {}'.format(
            self.n_cut, cnt_over_limit, len(self.x_bert)))
        print('load data done!\n')
        
        # Debug: 輸出第一筆樣本的 decoded 格式 (保留特殊 token)
        if len(self.x_bert) > 0 and self.tokenizer is not None:
            print("=== 第一筆樣本的 decoded 輸入 (x_bert，含特殊 token) ===")
            print(self.tokenizer.decode(self.x_bert[0], skip_special_tokens=False))
            print("=== 第一筆樣本的 decoded 標籤 (y_bert，含特殊 token) ===")
            print(self.tokenizer.decode(self.y_bert[0], skip_special_tokens=False))

        self.index = [i for i in range(len(self.x_bert))]

    def __getitem__(self, index):
        index = self.index[index]
        feed_list = [self.x_bert[index], self.y_bert[index], self.label[index], self.mask_label[index],
                     self.gt_emotion[index], self.gt_cause[index], self.gt_pair[index], True]
        return feed_list

    def __len__(self):
        return len(self.x_bert)




class UnlabeledDataset(Dataset):
    def __init__(self, input_file, tokenizer=None):
        print('load unlabeled data_file: {}'.format(input_file))
        self.x_bert = []
        self.doc_id = []
        self.raw_text = []  # 新增: 儲存原始文本 (用於 debug 驗證)
        self.tokenizer = tokenizer
        self.n_cut = 0
        cnt_over_limit = 0
        
        with open(input_file, 'r', encoding='utf8') as f:
            data = json.load(f)
        for doc in data:
            doc_id = doc["doc_id"]
            d_len = doc["doc_len"]
            part_sentence = [clause["clause"] for clause in doc["clauses"]]
            mask_full_document = ""
            raw_text_doc = ""  # 原始文本 (不含 MASK)
            for i in range(1, d_len + 1):
                mask_full_document = mask_full_document + ' ' + str(i) + ' ' + part_sentence[i - 1]
                mask_full_document = mask_full_document + "[MASK] [MASK] [MASK] [SEP]"
                raw_text_doc = raw_text_doc + f"[{i}] {part_sentence[i - 1]} "
            
            # 檢查長度，與 MyDataset 保持一致
            count_len = len(self.tokenizer.encode_plus(mask_full_document, return_tensors="pt")['input_ids'][0])
            if count_len > 512:
                print("UnlabeledDataset: Over limit length{} document{} - 移除此文檔".format(count_len, doc_id))
                cnt_over_limit += 1
                self.n_cut += 1
                continue
                
            self.doc_id.append(doc_id)
            self.raw_text.append(raw_text_doc.strip())  # 儲存原始文本
            mask_full_document = self.tokenizer.encode_plus(mask_full_document, return_tensors="pt", max_length=512, truncation=True, pad_to_max_length=True)['input_ids']
            self.x_bert.append(np.array(mask_full_document[0]))
            
        print(f'UnlabeledDataset: n_cut {self.n_cut}, total_documents {len(self.x_bert)}')
        print('load unlabeled data done!\n')
        
    def __getitem__(self, index):
        return self.x_bert[index]
    
    def __len__(self):
        return len(self.x_bert)
    
    def remove_by_doc_ids(self, doc_ids):
        id_set = set(doc_ids)
        keep_indices = [idx for idx, doc_id in enumerate(self.doc_id) if doc_id not in id_set]
        if len(keep_indices) == len(self.doc_id):
            return
        before = len(self.doc_id)
        self.x_bert = [self.x_bert[idx] for idx in keep_indices]
        self.doc_id = [self.doc_id[idx] for idx in keep_indices]
        self.raw_text = [self.raw_text[idx] for idx in keep_indices]  # 同步移除 raw_text
        after = len(self.doc_id)
        print(f"  自未標註資料集中移除 {before - after} 筆，剩餘 {after}")
    




class PseudoLabeledDataset(Dataset):
    def __init__(self, pseudo_labeled_samples, doc_ids, seq_len=512, mask_token_id=103):
        self.x_bert = [x for x, y in pseudo_labeled_samples]  # input_ids (masked)
        self.pseudo_labels = [y for x, y in pseudo_labeled_samples]  # 只包含 [MASK] 位置的 token id
        self.doc_ids = doc_ids
        self.seq_len = seq_len
        self.mask_token_id = mask_token_id

    def __getitem__(self, index):
        x = self.x_bert[index]  # 保持為 numpy array
        # 建立 y_bert, label, mask_label
        y = np.zeros(self.seq_len, dtype=np.int64)
        label = np.full(self.seq_len, -100, dtype=np.int64)
        mask_label = np.full(self.seq_len, -100, dtype=np.int64)

        # 找出 [MASK] 位置，把 pseudo_label 填進去
        mask_positions = np.where(x == self.mask_token_id)[0]
        pseudo_label = self.pseudo_labels[index]
        y[mask_positions] = pseudo_label
        label[mask_positions] = pseudo_label
        mask_label[mask_positions] = pseudo_label

        gt_emotion = np.int64(0)
        gt_cause = np.int64(0)
        gt_pair = np.int64(0)
        return [x, y, label, mask_label, gt_emotion, gt_cause, gt_pair, False]

    def __len__(self):
        return len(self.x_bert)



def extract_ground_truth_mask_tokens(doc, tokenizer):
    """建立文件中 [MASK] 位置的真實標籤 token ids"""
    # 若文件沒有 pairs 欄位代表沒有情緒-原因配對資訊
    if "pairs" not in doc:
        # 沒有 ground truth 可以建立時直接返回 None
        return None

    pairs = doc.get("pairs")
    # 允許 doc 中 pairs 為 None 的情況，這時無法建立映射
    if pairs is None:
        return None

    # zip(*pairs) 會把 [(a,b),(c,d)] 拆成 (a,c) 與 (b,d); 若 pairs 為空則提供預設空列表
    pos, cause = zip(*pairs) if pairs else ([], [])
    # print(f"pos: {pos}, cause: {cause}") # pos: (3,), cause: (1,)
    # 轉成 list 方便後續使用 index 查找位置
    pos = list(pos)
    # print(f"拆解後的 pos: {pos}") # 拆解後的 pos: [3]
    cause = list(cause)
    # print(f"   拆解後的 cause: {cause}") # 拆解後的 cause: [1]
    # 取出所有子句文字，若缺少 clauses 則回傳空列表
    part_sentence = [clause["clause"] for clause in doc.get("clauses", [])]

    full_document = ""
    mask_full_document = ""
    # 逐句構建完整文字與對應的 [MASK] 模板文字
    for i in range(1, len(part_sentence) + 1):
        # 將子句序號與內容接到完整輸入字串
        full_document += ' ' + str(i) + ' ' + part_sentence[i - 1]
        mask_full_document += ' ' + str(i) + ' ' + part_sentence[i - 1]
        # 檢查「當前的子句 i」是不是一個情緒子句
        if i in pos:
            # 如果是，就在 full_document 這個字串的最後面加上 "是" 這個字
            full_document += '是 '
            # 情緒句同時也是原因句時，需要接第二個判斷與指向的情緒句編號
            if i in cause:
                full_document += '是 '
                full_document += ' ' + str(pos[cause.index(i)]) + ' '
            else:
                full_document += '非 '
                full_document += ' 无 '
        else:
            full_document += '非 '
            # 非情緒句若是原因句也要標示對應的情緒編號
            if i in cause:
                full_document += '是 '
                full_document += ' ' + str(pos[cause.index(i)]) + ' '
            else:
                full_document += '非 '
                full_document += ' 无 '

        # 每個子句後面補上一個 [SEP] 分隔符
        full_document += '[SEP]'
        # 模板版本則以三個 [MASK] 取代表達佔位
        mask_full_document += "[MASK] [MASK] [MASK] [SEP]"

    # encode_plus 會回傳字典，"input_ids" 對應張量；索引 [0] 取得序列本身
    mask_ids = tokenizer.encode_plus(
        mask_full_document,
        return_tensors="pt",
        max_length=512,
        truncation=True,
        pad_to_max_length=True,
    )["input_ids"][0]
    # full_document 同樣轉成 token ids，方便與 mask 位置做對應
    full_ids = tokenizer.encode_plus(
        full_document,
        return_tensors="pt",
        max_length=512,
        truncation=True,
        pad_to_max_length=True,
    )["input_ids"][0]

    # (mask_ids == 103) 建立布林張量，nonzero 傳回非零索引; view(-1) 攤平成一維
    # as_tuple=False 以張量形式返回索引
    mask_positions = (mask_ids == 103).nonzero(as_tuple=False).view(-1)
    # 檢查 mask_positions 是否為空張量(沒有任何元素)
    # .numel(): 元素總數
    if mask_positions.numel() == 0:
        # 若沒有任何 [MASK] 則無法比對
        return None

    # 取出 full_ids 中對應 [MASK] 位置的正確 token ids
    gt_tokens = full_ids[mask_positions]
    # 轉成 numpy int64 供後續比較使用
    return gt_tokens.cpu().numpy().astype(np.int64)


def prepare_pseudo_ground_truth_map(unlabeled_dataset, unlabeled_json_path, tokenizer):
    """建立 doc_id 到真實標籤 (ground-truth mask token ids) 的對照表，用於評估偽標籤品質"""
    # 透過 getattr 安全取得資料集中的 doc_id 清單，若沒有屬性則給空列表
    doc_ids = getattr(unlabeled_dataset, 'doc_id', [])
    # summary 用來記錄可比較文件數量與缺漏原因
    summary = {
        "candidate_docs": len(doc_ids),
        "with_ground_truth": 0,
        "missing_pairs": 0,
        "not_found": 0,
        "build_error": 0,
    }

    try:
        # 讀取未標註資料集對應的 JSON，使用 utf-8 以支援中文
        with open(unlabeled_json_path, 'r', encoding='utf-8') as f:
            docs = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        # 發生檔案不存在或 JSON 解析錯誤時回報並返回空結果
        print(f"偽標籤品質記錄: 無法載入 {unlabeled_json_path}: {exc}")
        return {}, summary

    # 建立 doc_id 對應到原始文件內容的查詢表，加速後續查找
    doc_lookup = {doc.get('doc_id'): doc for doc in docs if 'doc_id' in doc}
    gt_map = {}
    # 逐一遍歷候選 doc_id，建立 ground truth token 序列
    for doc_id in doc_ids:
        doc = doc_lookup.get(doc_id)
        if doc is None:
            # JSON 找不到對應 doc_id 時記錄
            summary["not_found"] += 1
            continue
        if doc.get("pairs") is None:
            # 缺少 pairs 資料無法建立 ground truth
            summary["missing_pairs"] += 1
            continue

        gt_tokens = extract_ground_truth_mask_tokens(doc, tokenizer)
        if gt_tokens is None:
            # 子函式返回 None 代表無法取得正確 token 序列
            summary["build_error"] += 1
            continue

        # 成功時把 doc_id 與對應 tokens 存入映射
        gt_map[doc_id] = gt_tokens

    # with_ground_truth 計算成功建立的文件數
    summary["with_ground_truth"] = len(gt_map)
    return gt_map, summary


def write_pseudo_label_evaluation_report(output_dir, fold, round_idx, records, total_tokens, correct_tokens,
                                         filtered_total_tokens, filtered_correct_tokens, tokenizer, summary=None,
                                         report_filename=None, extra_summary_lines=None):
    """Persist pseudo-label vs ground-truth comparison details and return the report path."""
    os.makedirs(output_dir, exist_ok=True)
    if report_filename is None:
        report_filename = f"pseudo_label_evaluation_fold{fold}_round{round_idx + 1}.txt"
    report_path = os.path.join(output_dir, report_filename)

    with open(report_path, "w", encoding="utf-8") as f:
        if summary:
            f.write("資料概況:\n")
            f.write(f"  候選文件數: {summary.get('candidate_docs', 0)}\n")
            f.write(f"  可比較文件數: {summary.get('with_ground_truth', 0)}\n")
            if summary.get('missing_pairs', 0):
                f.write(f"  缺少 pairs 的文件數: {summary['missing_pairs']}\n")
            if summary.get('not_found', 0):
                f.write(f"  JSON 中找不到 doc_id 的數量: {summary['not_found']}\n")
            if summary.get('build_error', 0):
                f.write(f"  Ground truth 建立失敗的文件數: {summary['build_error']}\n")
            f.write("\n")

        if extra_summary_lines:
            f.write("附加統計:\n")
            for line in extra_summary_lines:
                f.write(f"  {line}\n")
            f.write("\n")

        if total_tokens > 0:
            accuracy = correct_tokens / total_tokens
            error_rate = 1.0 - accuracy
            f.write(f"總 mask 數量: {total_tokens}\n")
            f.write(f"預測正確的 mask 數量: {correct_tokens}\n")
            f.write(f"偽標籤錯誤率: {error_rate:.4f}\n\n")
        else:
            f.write("沒有可比較的偽標籤，無法計算錯誤率。\n\n")

        if filtered_total_tokens > 0:
            filtered_accuracy = filtered_correct_tokens / filtered_total_tokens
            filtered_error_rate = 1.0 - filtered_accuracy
            f.write(f"排除 '非 非 无' 後的 mask 數量: {filtered_total_tokens}\n")
            f.write(f"排除後預測正確的 mask 數量: {filtered_correct_tokens}\n")
            f.write(f"偽標籤錯誤率(排除 '非 非 无'): {filtered_error_rate:.4f}\n\n")
        elif total_tokens > 0:
            f.write("排除 '非 非 无' 後無剩餘可比較的 mask。\n\n")

        if not records:
            f.write("本輪沒有可比較的偽標籤樣本或未收集偽標籤。\n")
            return report_path

        for rec in records:
            f.write(f"doc_id: {rec['doc_id']}\n")
            skip_reason = rec.get('skip_reason')
            if skip_reason:
                f.write(f"  Skip: {skip_reason}\n\n")
                continue

            f.write(f"  正確/總數: {rec['correct']}/{rec['total']} (錯誤率 {rec['error_rate']:.4f})\n")
            filtered_total = rec.get('filtered_total')
            filtered_err = rec.get('filtered_error_rate')
            if filtered_total is not None:
                if filtered_total > 0 and filtered_err is not None:
                    filtered_err_str = f"{filtered_err:.4f}"
                elif filtered_total > 0:
                    filtered_err_str = "N/A"
                else:
                    filtered_err_str = "N/A"
                f.write(
                    f"  (排除 '非 非 无') 正確/總數: {rec.get('filtered_correct', 0)}/"
                    f"{filtered_total} (錯誤率 {filtered_err_str})\n"
                )
            mismatch_positions = rec.get('mismatch_positions', [])
            if mismatch_positions:
                mismatch_str = ', '.join(map(str, mismatch_positions))
            else:
                mismatch_str = '無'
            f.write(f"  錯誤位置 index: {mismatch_str}\n")

            pred_tokens = rec.get('pred_tokens')
            if pred_tokens is not None:
                pred_tokens_text = tokenizer.convert_ids_to_tokens(pred_tokens)
                f.write(f"  偽標籤 tokens: {' '.join(pred_tokens_text)}\n")

            gt_tokens = rec.get('gt_tokens')
            if gt_tokens is not None:
                gt_tokens_text = tokenizer.convert_ids_to_tokens(gt_tokens)
                f.write(f"  真實 tokens: {' '.join(gt_tokens_text)}\n")

            f.write("\n")

    return report_path


























class prompt_bert(torch.nn.Module):
    def __init__(self, bert_path='./bert-base-chinese'):
        super(prompt_bert, self).__init__()
        self.bert = BertForMaskedLM.from_pretrained(bert_path)
        self.tokenizer = BertTokenizer.from_pretrained(bert_path)
        self.bert.resize_token_embeddings(len(self.tokenizer))

    def forward(self, x_bert, labels):
        output = self.bert(x_bert, labels=labels)
        loss, logits = output.loss, output.logits
        return loss, logits

    def get_cls_embeddings(self, x_bert):
        outputs = self.bert.bert(x_bert, output_hidden_states=True, return_dict=True)
        return outputs.last_hidden_state[:, 0, :]

    def get_clause_embeddings(self, x_bert, probs, mode='emotion_clause', return_info=False):
        """
        取得子句級別的 embedding (被預測為情緒/原因子句)
        
        Args:
            x_bert: input_ids, shape (batch, 512)
            probs: softmax 機率, shape (batch, 512, vocab_size)
            mode: 'emotion_clause' 或 'cause_clause'
            return_info: 是否回傳詳細的 embedding 資訊 (預設 False)
        
        Returns:
            embeddings: shape (batch, 768)
            embedding_info: (僅當 return_info=True) list of dict
        """
        outputs = self.bert.bert(x_bert, output_hidden_states=True, return_dict=True)
        hidden_states = outputs.last_hidden_state  # (batch, 512, 768)
        return compute_clause_embeddings(hidden_states, x_bert, probs, mode, self.tokenizer, return_info=return_info)


def print_training_info():
    print('\n\n>>>>>>>>>>>>>>>>>>>>TRAINING INFO:\n')
    print('batch-{}, lr-{}'.format(
        opt.batch_size, opt.learning_rate))
    print('training_iter-{}\n'.format(opt.training_iter))


class Tee:
    """Duplicate stdout to both console and a log file."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def crf_prompt(logits, labels, x_bert, gt_emotion, gt_cause, gt_pair, save_path="results.txt"):
    label_index = [122, 123, 124, 125, 126, 127, 128, 129, 130, 8108, 8111, 8110, 8124, 8122, 8115, 8121, 8126, 8123,
                   8131, 8113, 8128, 8130, 8133, 8125, 8132, 8153, 8149, 8143, 8162, 8114, 8176, 8211, 8226, 8229, 8198,
                   8216, 8234, 8218, 8240, 8164, 8245, 8239, 8250, 8252, 8208, 8248, 8264, 8214, 8249, 8145, 8246, 8247,
                   8251, 8267, 8222, 8259, 8272, 8255, 8257, 8183, 8398, 8356, 8381, 8308, 8284, 8347, 8369, 8360, 8419,
                   8203, 8459, 8325, 8454, 8473, 8273] # 表示數字1-75的token id
    emo_gt = torch.sum(gt_emotion)
    emo_pre = 0
    emo_acc = 0
    cause_gt = torch.sum(gt_cause)
    cause_pre = 0
    cause_acc = 0
    pair_gt = torch.sum(gt_pair)
    pair_pre = 0
    pair_acc = 0
    for i in range(labels.shape[0]):
        count_mask = -1
        j = 0
        count_sentence = 0
        while j < 512:
            if x_bert[i][j] == 103:     #當token為'[MASK]'
                count_mask += 1
                count_mask = count_mask % 3 # 每三個[MASK]就重新從0開始
                if count_mask == 0:     #第一個'[MASK]'(預測是否為情緒子句)
                    if labels[i][j] == 3221:        #真實label為'是'
                        if torch.argmax(logits[i][j]) == 3221:      #預測為'是'
                            emo_acc += 1                            #預測正確的情緒數量+1
                    if torch.argmax(logits[i][j]) == 3221:         #預測為'是'
                        emo_pre += 1                               #預測為情緒的數量+1

                if count_mask == 1:     #第二個'[MASK]'(預測是否為原因子句)
                    if torch.argmax(logits[i][j]) == 3221:  #預測為'是'
                        cause_pre += 1                      #預測為原因的數量+1
                    if labels[i][j] == 3221:                #真實label為'是'
                        if torch.argmax(logits[i][j]) == 3221:  #預測為'是'
                            cause_acc += 1                      #預測正確的原因數量+1

                if count_mask == 2:   #第三個'[MASK]'(預測相關子句)
                    count_sentence += 1 #計算當前句子數量
                    mask = torch.zeros([21128]) #初始化一個大小為21128的全零張量作為遮罩 (對照表的大小)
                    case = [label_index[k] for k in range(max(0, -opt.window_size + count_sentence - 1),
                                                          min(75, opt.window_size + count_sentence))]
                    #建立一個case列表，包含當前句子範圍內的標籤索引，滑動視窗來選取有效的標籤範圍(避免超出句子邊界)
                    case.append(3187)  #加入'无'的token_id
                    for index in case:      #遍歷case，將對應mask設為1(lable_idex為數字1~75)
                        mask[index] = 1
                    logits_ = torch.argmax(logits[i][j] * mask)  #過濾logits，選取最高機率的索引作為預測結果(找到對應1-75數字的機率，然後經過argmax找到最大機率的索引)

                    if logits_ in label_index :  #當預測結果在label_index中(數字1~75)
                        pair_pre += 1           #預測的組合數量+1
                    if labels[i][j] in label_index:  #當真實label在label_index中(數字1~75)
                        if logits_ == labels[i][j] :  #且當預測結果與真實label相同
                            pair_acc += 1               #預測正確的組合數量+1

                j = j + 1
            else:
                j = j + 1
    p_emotion = emo_acc / (emo_pre + 1e-8)
    p_cause = cause_acc / (cause_pre + 1e-8)
    p_pair = pair_acc / (pair_pre + 1e-8) # 預測出的組合數
    r_emotion = emo_acc / (emo_gt + 1e-8)
    r_cause = cause_acc / (cause_gt + 1e-8)
    r_pair = pair_acc / (pair_gt + 1e-8) # ground truth組合數
    f_emotion = 2 * p_emotion * r_emotion / (p_emotion + r_emotion + 1e-8)
    f_cause = 2 * p_cause * r_cause / (p_cause + r_cause + 1e-8)
    f_pair = 2 * p_pair * r_pair / (p_pair + r_pair + 1e-8)
    with open(save_path, "a", encoding="utf-8") as file:
        file.write(f"Emotion: 預測正確的數量:{emo_acc}    預測出情緒的數量:{emo_pre}    實際正確情緒的數量:{emo_gt}\n")
        file.write(f"Cause:   預測正確的數量:{cause_acc}  預測出原因的數量:{cause_pre}  實際正確原因的數量:{cause_gt}\n")
        file.write(f"Pair:    預測正確的數量:{pair_acc}   預測出組合的數量:{pair_pre}   實際正確組合的數量:{pair_gt}\n")
        file.write(f"Precision (Emotion, Cause, Pair): {p_emotion:.4f}, {p_cause:.4f}, {p_pair:.4f}\n")
        file.write(f"Recall (Emotion, Cause, Pair): {r_emotion:.4f}, {r_cause:.4f}, {r_pair:.4f}\n")
        file.write(f"F1 Score (Emotion, Cause, Pair): {f_emotion:.4f}, {f_cause:.4f}, {f_pair:.4f}\n\n")
    print('emo_gt {}  cause_gt {}  pair_gt {}'.format(emo_gt, cause_gt, pair_gt))
    print(f"Emotion: 預測正確的數量:{emo_acc}, 預測出情緒的數量:{emo_pre}, 實際正確情緒的數量:{emo_gt}")
    print(f"Cause: 預測正確的數量:{cause_acc}, 預測出原因的數量:{cause_pre}, 實際正確原因的數量:{cause_gt}")
    print(f"Pair: 預測正確的數量:{pair_acc}, 預測出組合的數量:{pair_pre}, 實際正確組合的數量:{pair_gt}")
    return p_emotion, r_emotion, f_emotion, p_cause, r_cause, f_cause, p_pair, r_pair, f_pair



def save_mask_predictions(logits, x_bert, tokenizer, doc_ids, fold, output_dir=".", base_filename="text_result", pseudo_labels=None):
    label_index = [122, 123, 124, 125, 126, 127, 128, 129, 130, 8108, 8111, 8110, 8124, 8122, 8115, 8121, 8126, 8123,
                   8131, 8113, 8128, 8130, 8133, 8125, 8132, 8153, 8149, 8143, 8162, 8114, 8176, 8211, 8226, 8229, 8198,
                   8216, 8234, 8218, 8240, 8164, 8245, 8239, 8250, 8252, 8208, 8248, 8264, 8214, 8249, 8145, 8246, 8247,
                   8251, 8267, 8222, 8259, 8272, 8255, 8257, 8183, 8398, 8356, 8381, 8308, 8284, 8347, 8369, 8360, 8419,
                   8203, 8459, 8325, 8454, 8473, 8273]
    output_lines = []
    for i, doc_id in enumerate(doc_ids):
        output_lines.append(f"doc_id :{doc_id}")
        line = []
        count_sentence = 0
        count_mask = -1
        j = 0
        pseudo_idx = 0  # 追蹤 pseudo_label 的位置
        while j < 512:
            if x_bert[i][j] == 103:     #當token為'[MASK]'
                count_mask += 1
                count_mask = count_mask % 3
                if pseudo_labels is not None:
                     pred_token_id = pseudo_labels[i][pseudo_idx]
                     pseudo_idx += 1
                     pred_text = tokenizer.decode([pred_token_id]).strip()
                     line.append(pred_text)
                else:
                    # count_mask == 0 → emotion
                    if count_mask == 0:
                        pred_token_id = torch.argmax(logits[i][j]).item()
                        pred_text = tokenizer.decode([pred_token_id]).strip()
                        line.append(pred_text)
                    
                    # count_mask == 1 → cause
                    elif count_mask == 1:
                        pred_token_id = torch.argmax(logits[i][j]).item()
                        pred_text = tokenizer.decode([pred_token_id]).strip()
                        line.append(pred_text)

                    # count_mask == 2 → pair
                    elif count_mask == 2:
                        count_sentence += 1
                        case = [label_index[k] for k in range(
                            max(0, -opt.window_size + count_sentence - 1),
                            min(75, opt.window_size + count_sentence)
                        )]
                        case.append(3187)
                        candidate_logits = logits[i][j][case]
                        selected_idx = torch.argmax(candidate_logits).item()
                        pred_token_id = case[selected_idx]  
                        pred_text = tokenizer.decode([pred_token_id]).strip()
                        line.append(pred_text)
                if count_mask == 2:
                    # 一組三個 [MASK] 預測完成，寫入一行
                    output_lines.append(' '.join(line))
                    line = []
                    
                j += 1
            else:
                j += 1

    output_file = os.path.join(output_dir, f"fold{fold}_{base_filename}.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        for line in output_lines:
            f.write(line + "\n")
    print(f"儲存完成，路徑: {output_file}")


def evaluate_split(model, dataloader, fold, split_name, save_dir, tokenizer=None,
                   doc_ids=None, save_predictions=False, prediction_basename=None):
    """Evaluate model on a dataloader and persist metrics/predictions when requested."""
    model.eval()
    logits_list = []
    labels_list = []
    x_bert_list = []
    emotion_list = []
    cause_list = []
    pair_list = []
    total_loss = 0.0
    batch_count = 0

    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 8:
                x_bert, y_bert, label, mask_label, gt_emotion, gt_cause, gt_pair, _ = batch
            else:
                x_bert, y_bert, label, mask_label, gt_emotion, gt_cause, gt_pair = batch

            if use_gpu:
                x_bert = x_bert.cuda()
                label = label.cuda()

            loss, logits = model(x_bert, label)
            total_loss += loss.item()
            batch_count += 1
            logits = F.softmax(logits, dim=-1)

            labels_list.append(label.cpu())
            logits_list.append(logits.cpu())
            x_bert_list.append(x_bert.cpu())
            emotion_list.append(gt_emotion.cpu() if torch.is_tensor(gt_emotion) else torch.tensor(gt_emotion).view(-1))
            cause_list.append(gt_cause.cpu() if torch.is_tensor(gt_cause) else torch.tensor(gt_cause).view(-1))
            pair_list.append(gt_pair.cpu() if torch.is_tensor(gt_pair) else torch.tensor(gt_pair).view(-1))

    if labels_list:
        all_labels = torch.cat(labels_list, dim=0)
        all_logits = torch.cat(logits_list, dim=0)
        all_x_bert = torch.cat(x_bert_list, dim=0)
        all_emotion_gt = torch.cat(emotion_list, dim=0)
        all_cause_gt = torch.cat(cause_list, dim=0)
        all_pair_gt = torch.cat(pair_list, dim=0)
    else:
        all_labels = torch.empty((0, 0), dtype=torch.long)
        all_logits = torch.empty((0, 0, 0), dtype=torch.float32)
        all_x_bert = torch.empty((0, 0), dtype=torch.long)
        all_emotion_gt = torch.empty((0,), dtype=torch.long)
        all_cause_gt = torch.empty((0,), dtype=torch.long)
        all_pair_gt = torch.empty((0,), dtype=torch.long)

    os.makedirs(save_dir, exist_ok=True)
    evaluation_file = os.path.join(save_dir, f'fold{fold}_{split_name}_evaluation.txt')
    metrics = crf_prompt(all_logits, all_labels, all_x_bert, all_emotion_gt, all_cause_gt, all_pair_gt,
                         save_path=evaluation_file)

    if save_predictions and tokenizer is not None and doc_ids is not None and logits_list:
        base_filename = prediction_basename if prediction_basename else (
            'text_result' if split_name == 'test' else f'{split_name}_result'
        )
        save_mask_predictions(all_logits, all_x_bert, tokenizer, doc_ids, fold=fold,
                              output_dir=save_dir, base_filename=base_filename)

    avg_loss = total_loss / max(batch_count, 1)
    return metrics, avg_loss


def write_best_val_checkpoint(info_path, stage, iteration, val_loss, metrics, checkpoint_path,
                              train_loss_summary=None, pseudo_confidence_stats=None):
    """Record metadata for the current best validation checkpoint."""
    os.makedirs(os.path.dirname(info_path), exist_ok=True)
    with open(info_path, "w", encoding="utf-8") as f:
        f.write(f"Best validation checkpoint (stage: {stage})\n")
        if iteration is not None:
            f.write(f"Iteration: {iteration}\n")
        if val_loss is not None:
            f.write(f"Validation loss: {val_loss:.6f}\n")
        else:
            f.write("Validation loss: N/A\n")
        if checkpoint_path:
            f.write(f"Checkpoint path: {checkpoint_path}\n")
        if train_loss_summary is not None:
            f.write("Training loss summary (current round/epoch)\n")
            f.write(
                f"Total loss mean/std: {train_loss_summary['total_mean']:.6f} / {train_loss_summary['total_std']:.6f}\n"
            )
            f.write(
                f"Supervised loss mean/std: {train_loss_summary['sup_mean']:.6f} / {train_loss_summary['sup_std']:.6f}\n"
            )
            f.write(
                f"Pseudo loss mean/std: {train_loss_summary['pseudo_mean']:.6f} / {train_loss_summary['pseudo_std']:.6f}\n"
            )
        if pseudo_confidence_stats is not None:
            f.write("Pseudo confidence stats\n")
            f.write(
                f"Valid pseudo tokens: {pseudo_confidence_stats['valid_tokens']}\n"
            )
            f.write(
                f"Confident pseudo tokens: {pseudo_confidence_stats['confident_tokens']}\n"
            )
            f.write(
                f"Confident ratio: {pseudo_confidence_stats['confident_ratio']:.6f}\n"
            )
        f.write("Metrics (Precision, Recall, F1)\n")
        f.write(
            f"Emotion: {metrics['p_emotion']:.4f}, {metrics['r_emotion']:.4f}, {metrics['f_emotion']:.4f}\n"
        )
        f.write(
            f"Cause: {metrics['p_cause']:.4f}, {metrics['r_cause']:.4f}, {metrics['f_cause']:.4f}\n"
        )
        f.write(
            f"Pair: {metrics['p_pair']:.4f}, {metrics['r_pair']:.4f}, {metrics['f_pair']:.4f}\n"
        )


def append_round_metrics_csv(csv_path, row_values):
    """每輪附加一列 CSV，檔案不存在時自動寫入表頭"""
    header = [
        'fold', 'self_training_round', 'epoch', 'nest_loss_mode', 'gamma', 'nest_loss_threshold',
        'train_total_loss_mean', 'train_total_loss_std',
        'train_sup_loss_mean', 'train_sup_loss_std',
        'train_pseudo_loss_mean', 'train_pseudo_loss_std',
        'num_valid_pseudo_tokens', 'num_confident_tokens', 'confident_ratio',
        'val_loss', 'val_f1_emotion', 'val_f1_cause', 'val_f1_pair'
    ]
    need_header = not os.path.exists(csv_path)
    with open(csv_path, 'a', encoding='utf-8') as f:
        if need_header:
            f.write(','.join(header) + '\n')
        row_str = [str(v) for v in row_values]
        f.write(','.join(row_str) + '\n')


def append_pseudo_sample_entry(state, doc_id, x_np, pseudo_np, tokenizer, mask_confidences=None):
    """將一筆偽標籤樣本加入訓練集，並記錄評估與日誌資訊。"""
    # 複製輸入資料，避免後續修改影響原始資料
    x_copy = np.array(x_np, dtype=np.int64)
    pseudo_array = np.array(pseudo_np, dtype=np.int64)
    # ===== 1. 加入訓練用資料集 =====
    # (x_copy, pseudo_array) 將在訓練時作為 (輸入, 偽標籤) 使用
    state['round_pseudo_labeled_samples'].append((x_copy, pseudo_array))
    state['round_pseudo_doc_ids'].append(doc_id)
    # ===== 1.1 記錄「本輪實際納入訓練」的樣本歷史 =====
    # 這段放在 append_pseudo_sample，可同時覆蓋 threshold / nest / random 三種模式
    # 並且以「實際通過篩選後加入訓練」為準 (含 consistency 過濾後)
    current_round = state['self_round'] + 1
    if doc_id not in state['doc_selection_history']:
        state['doc_selection_history'][doc_id] = []
    # 避免同一輪重複記錄（例如其他分支已先記一次）
    if current_round not in state['doc_selection_history'][doc_id]:
        state['doc_selection_history'][doc_id].append(current_round)
    # ===== 2. 加入預測結果記錄 (用於後續保存) =====
    state['all_pseudo_predictions'].append({
        'doc_id': doc_id,
        'x_bert': x_copy,
        'pseudo_labels': pseudo_array
    })
    # ===== 3. 建立日誌項目 (用於 JSON 輸出) =====
    # 將偽標籤的 token ids 轉換為可讀的 token 字串
    pred_tokens_list = tokenizer.convert_ids_to_tokens(pseudo_array.tolist())
    pseudo_entry = {
        "doc_id": doc_id,
        "pseudo_label_ids": pseudo_array.tolist(), # 偽標籤的 token id 列表
        "pseudo_label_tokens": pred_tokens_list,  # 偽標籤的 token 字串列表
        "ground_truth_available": False,  # 預設無真實答案
    }
    if mask_confidences is not None:
        pseudo_entry["mask_confidences"] = mask_confidences
    # ===== 4. 如果需要評估偽標籤品質（與真實答案比較) =====
    if state['evaluate_pseudo']:
        # 從對照表取得該文檔的真實答案
        gt_tokens = state['pseudo_ground_truth_map'].get(doc_id)

        if gt_tokens is None:
            # 情況 A：該文檔沒有真實答案可比較
            state['pseudo_eval_records'].append({
                'doc_id': doc_id,
                'skip_reason': 'no_ground_truth',
            })
            pseudo_entry["skip_reason"] = "no_ground_truth"
        else:
            gt_array = np.array(gt_tokens, dtype=np.int64)

            if gt_array.shape[0] != pseudo_array.shape[0]:
                # 情況 B：長度不匹配 (可能因文檔被截斷)
                state['pseudo_eval_records'].append({
                    'doc_id': doc_id,
                    'skip_reason': f'length_mismatch(gt={gt_array.shape[0]}, pseudo={pseudo_array.shape[0]})'
                })
                pseudo_entry["skip_reason"] = "length_mismatch"
                pseudo_entry["ground_truth_available"] = False
            else:
                # 情況 C：可以正常比較
                # 計算每個位置是否匹配
                match_mask = (pseudo_array == gt_array)
                correct = int(match_mask.sum())
                total = int(match_mask.size)
                # 找出不匹配的位置索引
                mismatch_positions = np.where(~match_mask)[0].tolist()
                # 累加到全域統計
                state['pseudo_eval_correct'] += correct
                state['pseudo_eval_total'] += total
                # ===== 5. 計算過濾後的統計 (排除「非非无」的情況) =====
                # 目的：排除「非情緒、非原因、無配對」的子句，這些預測正確較容易
                gt_tokens_list = tokenizer.convert_ids_to_tokens(gt_array.tolist())
                # triple_mask: True 表示該位置要計入統計，False 表示要排除
                triple_mask = np.ones_like(gt_array, dtype=bool)
                # 每 3 個 token 為一組 (情緒、原因、配對)
                for idx_mask in range(0, len(gt_tokens_list), 3):
                    gt_triple = gt_tokens_list[idx_mask:idx_mask + 3]
                    pred_triple = pred_tokens_list[idx_mask:idx_mask + 3]
                    # 如果真實答案和預測都是「非非无」，則排除此三元組
                    if (len(gt_triple) == 3 and len(pred_triple) == 3 and
                            gt_triple == ['非', '非', '无'] and
                            pred_triple == ['非', '非', '无']):
                        triple_mask[idx_mask:idx_mask + 3] = False
                # 計算過濾後的統計
                filtered_match = match_mask[triple_mask]
                if filtered_match.size > 0:
                    filtered_correct = int(filtered_match.sum())
                    filtered_total = int(filtered_match.size)
                    filtered_error_rate = 1.0 - (filtered_correct / filtered_total)
                    state['pseudo_eval_filtered_correct'] += filtered_correct
                    state['pseudo_eval_filtered_total'] += filtered_total
                else:
                    # 所有預測都是「非非无」，無法計算過濾後統計
                    filtered_correct = 0
                    filtered_total = 0
                    filtered_error_rate = None
                # ===== 6. 記錄詳細評估結果 =====
                state['pseudo_eval_records'].append({
                    'doc_id': doc_id,
                    'correct': correct,
                    'total': total,
                    'error_rate': 1.0 - (correct / total) if total else 0.0,
                    'pred_tokens': pseudo_array.tolist(),
                    'gt_tokens': gt_array.tolist(),
                    'mismatch_positions': mismatch_positions,
                    'filtered_correct': filtered_correct,
                    'filtered_total': filtered_total,
                    'filtered_error_rate': filtered_error_rate,
                })
                # 更新 pseudo_entry 的真實答案資訊
                pseudo_entry["ground_truth_available"] = True
                pseudo_entry["gt_label_ids"] = gt_array.tolist()
                pseudo_entry["gt_label_tokens"] = gt_tokens_list
                pseudo_entry["match_ratio"] = (correct / total) if total else None
                pseudo_entry["mismatch_positions"] = mismatch_positions
                if filtered_error_rate is not None:
                    pseudo_entry["filtered_match_ratio"] = filtered_correct / filtered_total
                    pseudo_entry["filtered_error_rate"] = filtered_error_rate
                # 記錄被忽略的三元組數量
                pseudo_entry["ignored_triples"] = int((~triple_mask).sum() // 3)
    # 將此樣本的日誌項目加入列表
    state['pseudo_logging_entries'].append(pseudo_entry)


def evaluate_all_unlabeled_pseudo_entry(state, doc_id, pseudo_array, pseudo_entry, tokenizer):
    """對全部未標註 pseudo 預測做 GT 比對，並累積統計結果"""
    gt_tokens = state['pseudo_ground_truth_map'].get(doc_id)

    if gt_tokens is None:
        raise RuntimeError(
            f"全部未標註 pseudo 評估失敗: doc_id={doc_id} 找不到對應 ground truth"
        )

    gt_array = np.array(gt_tokens, dtype=np.int64)
    if gt_array.shape[0] != pseudo_array.shape[0]:
        raise RuntimeError(
            "全部未標註 pseudo 評估失敗: doc_id="
            f"{doc_id} 的長度不一致 (gt={gt_array.shape[0]}, pseudo={pseudo_array.shape[0]})"
        )

    pred_tokens_list = tokenizer.convert_ids_to_tokens(pseudo_array.tolist())
    match_mask = (pseudo_array == gt_array)
    correct = int(match_mask.sum())
    total = int(match_mask.size)
    mismatch_positions = np.where(~match_mask)[0].tolist()
    state['all_unlabeled_pseudo_eval_correct'] += correct
    state['all_unlabeled_pseudo_eval_total'] += total

    gt_tokens_list = tokenizer.convert_ids_to_tokens(gt_array.tolist())
    triple_mask = np.ones_like(gt_array, dtype=bool)
    for idx_mask in range(0, len(gt_tokens_list), 3):
        gt_triple = gt_tokens_list[idx_mask:idx_mask + 3]
        pred_triple = pred_tokens_list[idx_mask:idx_mask + 3]
        if (len(gt_triple) == 3 and len(pred_triple) == 3 and
                gt_triple == ['非', '非', '无'] and
                pred_triple == ['非', '非', '无']):
            triple_mask[idx_mask:idx_mask + 3] = False

    filtered_match = match_mask[triple_mask]
    if filtered_match.size > 0:
        filtered_correct = int(filtered_match.sum())
        filtered_total = int(filtered_match.size)
        filtered_error_rate = 1.0 - (filtered_correct / filtered_total)
        state['all_unlabeled_pseudo_eval_filtered_correct'] += filtered_correct
        state['all_unlabeled_pseudo_eval_filtered_total'] += filtered_total
    else:
        filtered_correct = 0
        filtered_total = 0
        filtered_error_rate = None

    pair_match = match_mask[2::3]
    pair_correct = int(pair_match.sum())
    pair_total = int(pair_match.size)
    pair_error_rate = 1.0 - (pair_correct / pair_total) if pair_total > 0 else None
    pair_fully_correct = (pair_total > 0 and pair_correct == pair_total)

    fully_correct = (correct == total)
    state['all_unlabeled_pseudo_eval_records'].append({
        'doc_id': doc_id,
        'correct': correct,
        'total': total,
        'error_rate': 1.0 - (correct / total) if total else 0.0,
        'pred_tokens': pseudo_array.tolist(),
        'gt_tokens': gt_array.tolist(),
        'mismatch_positions': mismatch_positions,
        'filtered_correct': filtered_correct,
        'filtered_total': filtered_total,
        'filtered_error_rate': filtered_error_rate,
        'pair_correct': pair_correct,
        'pair_total': pair_total,
        'pair_error_rate': pair_error_rate,
        'pair_fully_correct': pair_fully_correct,
        'fully_correct': fully_correct,
    })

    pseudo_entry['ground_truth_available'] = True
    pseudo_entry['gt_label_ids'] = gt_array.tolist()
    pseudo_entry['gt_label_tokens'] = gt_tokens_list
    pseudo_entry['match_ratio'] = (correct / total) if total else None
    pseudo_entry['mismatch_positions'] = mismatch_positions
    pseudo_entry['pair_match_ratio'] = (pair_correct / pair_total) if pair_total > 0 else None
    pseudo_entry['pair_error_rate'] = pair_error_rate
    pseudo_entry['pair_fully_correct'] = pair_fully_correct
    pseudo_entry['fully_correct'] = fully_correct
    pseudo_entry['error_count'] = total - correct
    if filtered_error_rate is not None:
        pseudo_entry['filtered_match_ratio'] = filtered_correct / filtered_total
        pseudo_entry['filtered_error_rate'] = filtered_error_rate


def record_all_unlabeled_pseudo_predictions(state, doc_ids, x_bert_list, pseudo_tokens_list, tokenizer):
    """記錄本輪所有未標註文檔的原始 pseudo 預測結果"""
    if not doc_ids and not x_bert_list and not pseudo_tokens_list:
        return

    if not doc_ids or not x_bert_list or not pseudo_tokens_list:
        raise RuntimeError(
            "未標註 pseudo 記錄失敗: 收到非預期的空列表 "
            f"(doc_ids={len(doc_ids)}, x_bert_list={len(x_bert_list)}, pseudo_tokens_list={len(pseudo_tokens_list)})"
        )

    if not (len(doc_ids) == len(x_bert_list) == len(pseudo_tokens_list)):
        raise RuntimeError(
            "未標註 pseudo 記錄失敗: 輸入長度不一致 "
            f"(doc_ids={len(doc_ids)}, x_bert_list={len(x_bert_list)}, pseudo_tokens_list={len(pseudo_tokens_list)})"
        )

    for doc_id, x_np, pseudo_np in zip(doc_ids, x_bert_list, pseudo_tokens_list):
        if pseudo_np is None:
            raise RuntimeError(
                f"未標註 pseudo 記錄失敗: doc_id={doc_id} 的 pseudo_tokens 為 None"
            )

        x_copy = np.array(x_np, dtype=np.int64)
        pseudo_array = np.array(pseudo_np, dtype=np.int64)
        state['all_unlabeled_pseudo_predictions'].append({
            'doc_id': doc_id,
            'x_bert': x_copy,
            'pseudo_labels': pseudo_array,
        })
        pseudo_entry = {
            'doc_id': doc_id,
            'pseudo_label_ids': pseudo_array.tolist(),
            'pseudo_label_tokens': tokenizer.convert_ids_to_tokens(pseudo_array.tolist()),
        }
        if state['evaluate_pseudo']:
            evaluate_all_unlabeled_pseudo_entry(state, doc_id, pseudo_array, pseudo_entry, tokenizer)
        state['all_unlabeled_pseudo_logging_entries'].append(pseudo_entry)


def record_consistency_rejection_entry(state, doc_id, pseudo_np, tokenizer, mask_confidences=None):
    """記錄被 task consistency 拒絕的樣本，並與 GT 比較 (若可用)
    
    用途：分析「閾值通過但 consistency 拒絕」的樣本是否為高信心錯誤預測
    """
    pseudo_array = np.array(pseudo_np, dtype=np.int64)
    pred_tokens_list = tokenizer.convert_ids_to_tokens(pseudo_array.tolist())
    record = {
        'doc_id': doc_id,
        'selector': state['current_selector'],
        'pseudo_label_tokens': pred_tokens_list,
    }
    if mask_confidences is not None:
        record['mask_confidences'] = mask_confidences

    if state['evaluate_pseudo']:
        gt_tokens = state['pseudo_ground_truth_map'].get(doc_id)
        if gt_tokens is None:
            raise RuntimeError(
                f"[Pseudo GT Error] doc_id={doc_id} 找不到 ground truth，"
                "目前流程要求每筆資料都必須可取得正確答案"
            )
        gt_array = np.array(gt_tokens, dtype=np.int64)
        if gt_array.shape[0] != pseudo_array.shape[0]:
            gt_tokens_preview = tokenizer.convert_ids_to_tokens(gt_array[:12].tolist())
            pseudo_tokens_preview = pred_tokens_list[:12]
            raise RuntimeError(
                "[Pseudo GT Error] 發生長度不一致，"
                f"doc_id={doc_id}, gt_len={gt_array.shape[0]}, pseudo_len={pseudo_array.shape[0]}, "
                f"gt_preview={gt_tokens_preview}, pseudo_preview={pseudo_tokens_preview}"
            )

        match_mask = (pseudo_array == gt_array)
        correct = int(match_mask.sum())
        total = int(match_mask.size)
        gt_tokens_list = tokenizer.convert_ids_to_tokens(gt_array.tolist())
        record['gt_eval'] = {
            'correct': correct,
            'total': total,
            'match_ratio': correct / total if total else None,
            'gt_tokens': gt_tokens_list,
            'mismatch_positions': np.where(~match_mask)[0].tolist(),
        }

    state['consistency_rejected_records'].append(record)


def _run_main(save_path):
    log_dir = os.path.join(save_path, "init_supervised_metrics")
    os.makedirs(log_dir, exist_ok=True)
    
    # 創建時間記錄目錄
    time_log_dir = os.path.join(save_path, "time_logs")
    os.makedirs(time_log_dir, exist_ok=True)
    time_log_file = os.path.join(time_log_dir, "execution_time_log.txt")

    print_time()
    overall_start_time = time.time()
    bert_path = './bert-base-chinese'
    tokenizer = BertTokenizer.from_pretrained(bert_path)

    # train
    print_training_info()  # 輸出訓練的超參數資訊

    max_result_emo_f, max_result_emo_p, max_result_emo_r = [], [], []
    max_result_pair_f, max_result_pair_p, max_result_pair_r = [], [], []
    max_result_cause_f, max_result_cause_p, max_result_cause_r = [], [], []
    
    all_folds_start_time = time.time()  # 記錄所有 fold 開始時間
    
    # 初始化時間記錄文件
    with open(time_log_file, "a", encoding="utf-8") as f:
        f.write("\n" + "=" * 60 + "\n")
        f.write("=== UECA Self-Training 執行時間記錄 ===\n")
        f.write(f"實驗開始時間: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(overall_start_time))}\n")
        f.write(f"執行模式: {'測試模式' if opt.test_only else '訓練模式'}\n")
        f.write(f"Fold 範圍: {opt.start_fold} - {opt.end_fold}\n")
        if not opt.test_only:
            f.write(f"Self-training 輪數: {opt.self_training_rounds}\n")
        f.write(f"批次大小: {opt.batch_size}\n")
        f.write(f"學習率: {opt.learning_rate}\n")
        if opt.test_only:
            f.write(f"檢查點路徑: {opt.checkpointpath}\n")
        f.write("=" * 60 + "\n\n")
    
    fold_times = []  # 記錄每個fold的時間資訊
    
    # 生成時間戳記用於 self_training_results 資料夾命名
    st_timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    eval_checkpoint_root = opt.checkpointpath if opt.test_only else save_path
    
    for fold in range(opt.start_fold, opt.end_fold + 1):
        fold_start_time = time.time()  # 記錄當前 fold 開始時間
        print(f"\n  Fold {fold} 開始時間: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
        best_val_pair_info_path = os.path.join(save_path, f'fold{fold}_best_val_checkpoint_pair.txt') # 用來記錄 pair 在驗證集上最佳的結果
        best_val_emo_info_path = os.path.join(save_path, f'fold{fold}_best_val_checkpoint_emo.txt')
        best_val_cause_info_path = os.path.join(save_path, f'fold{fold}_best_val_checkpoint_cause.txt')
        if os.path.exists(best_val_pair_info_path):
            os.remove(best_val_pair_info_path)
        if os.path.exists(best_val_emo_info_path):
            os.remove(best_val_emo_info_path)
        if os.path.exists(best_val_cause_info_path):
            os.remove(best_val_cause_info_path)
        best_val_stage = None
        
    # 載入模型
        print('build model..')
        model = prompt_bert(bert_path)
        print('build model end...')

        # 直接使用已訓練的模型進行測試不進行訓練時，這個判斷是才會為True
        if opt.checkpoint:
            if opt.test_model_type == 'self_training':
                # 載入 Self-training 的最佳模型
                st_model_path = get_self_training_best_checkpoint_path(
                    opt.checkpointpath,
                    fold,
                    opt.test_metric,
                )
                if os.path.exists(st_model_path):
                    model = torch.load(st_model_path, map_location=torch.device('cpu'))
                    print(
                        f'載入 Self-training 最佳模型: {st_model_path} '
                        f'(依據 {METRIC_TO_LABEL[opt.test_metric]})'
                    )
                else:
                    print(f'Self-training 模型不存在: {st_model_path}')
                    print('回退到初始模型...')
                    fallback_path = get_supervised_best_checkpoint_path(
                        opt.checkpointpath,
                        fold,
                        opt.test_metric,
                    )
                    model = torch.load(fallback_path, map_location=torch.device('cpu'))
                    print(
                        f'載入初始監督學習模型: {fallback_path} '
                        f'(依據 {METRIC_TO_LABEL[opt.test_metric]})'
                    )
            elif opt.test_model_type == 'self_training_avg3':
                # 與既有行為一致：checkpoint/test-only 模式下仍先載入 pair 模型，Avg3 的正式建立流程在後段測試路徑處理
                st_model_path = os.path.join(opt.checkpointpath, 'self_training_models', f'fold{fold}_self_training_best_pair.pth')
                if os.path.exists(st_model_path):
                    model = torch.load(st_model_path, map_location=torch.device('cpu'))
                    print(f'載入 Self-training 最佳模型: {st_model_path}')
                else:
                    print(f'Self-training 模型不存在: {st_model_path}')
                    print('回退到初始模型...')
                    model = torch.load(opt.checkpointpath + '/fold{}_best_pair.pth'.format(fold),
                                       map_location=torch.device('cpu'))
            else:
                # 載入初始監督學習模型
                supervised_model_path = get_supervised_best_checkpoint_path(
                    opt.checkpointpath,
                    fold,
                    opt.test_metric,
                )
                model = torch.load(supervised_model_path, map_location=torch.device('cpu'))
                print(
                    f'載入初始監督學習模型: {supervised_model_path} '
                    f'(依據 {METRIC_TO_LABEL[opt.test_metric]})'
                )
        if use_gpu:
            model = model.cuda()


        train_file_name = 'fold{}_train.json'.format(fold)
        val_file_name = 'fold{}_val.json'.format(fold)
        test_file_name = 'fold{}_test.json'.format(fold)
        unlabeled_file_name = 'fold{}_unlabeled.json'.format(fold)
        unlabeled = opt.dataset + unlabeled_file_name

        print('############# fold {} begin ###############'.format(fold))
        train = opt.dataset + train_file_name
        val = opt.dataset + val_file_name
        test = opt.dataset + test_file_name
        edict = {"train": train, "val": val, "test": test}
        unlabeled_dataset = UnlabeledDataset(unlabeled, tokenizer=tokenizer)
        pseudo_ground_truth_map = {}
        pseudo_gt_summary = None
        if opt.log_pseudo_quality:
            pseudo_ground_truth_map, pseudo_gt_summary = prepare_pseudo_ground_truth_map(unlabeled_dataset, unlabeled, tokenizer)
            available = pseudo_gt_summary.get('with_ground_truth', 0)
            candidates = pseudo_gt_summary.get('candidate_docs', 0)
            print(f"偽標籤品質記錄：可比較真實答案的樣本 {available}/{candidates}")
            # 從字典(dict)中獲取鍵(key)對應的值(value)，如果key不存在返回預設值0 (if判斷即為False不印出數量)
            if pseudo_gt_summary.get('missing_pairs', 0):
                print(f"  缺少 pairs 的樣本數: {pseudo_gt_summary['missing_pairs']}")
            if pseudo_gt_summary.get('not_found', 0):
                print(f"  JSON 中找不到對應 doc_id 的樣本數: {pseudo_gt_summary['not_found']}")
            if pseudo_gt_summary.get('build_error', 0):
                print(f"  建立真實標籤失敗的樣本數: {pseudo_gt_summary['build_error']}")
        NLP_Dataset = {x: MyDataset(edict[x], test=(x == 'test'), tokenizer=tokenizer) for x in ['train', 'val', 'test']} # Dicitionary Comprehension，一次建立三個資料集
        trainloader = DataLoader(NLP_Dataset['train'], batch_size=opt.batch_size, shuffle=True, drop_last=True)
        train_eval_loader = DataLoader(NLP_Dataset['train'], batch_size=opt.batch_size, shuffle=False)
        valloader = DataLoader(NLP_Dataset['val'], batch_size=opt.batch_size, shuffle=False)
        unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=opt.batch_size, shuffle=False)
        pseudo_dataset_list = []
        
        num_train_data = len(NLP_Dataset['train'])
        print(f"--- 訓練資料集 (NLP_Dataset['train']) 總共有: {num_train_data} 筆資料") # 1750筆資料
        
        testloader = DataLoader(NLP_Dataset['test'], batch_size=opt.batch_size, shuffle=False)
        
        num_test_data = len(NLP_Dataset['test'])
        print(f"--- 測試集 (NLP_Dataset['test']) 總共有: {num_test_data} 筆資料") # 195筆資料
        
    
        max_p_emotion, max_r_emotion, max_f1_emotion, max_p_cause, max_r_cause, max_f1_cause, max_p_pair,\
        max_r_pair, max_f1_pair = [-1.] * 9
        # emotion/cause 專用 checkpoint 儲存策略：
        # 1) 未指定新參數時沿用舊行為 (僅 consistency_pseudo=True 時儲存)
        # 2) --save_task_specific_checkpoints 可強制啟用
        # 3) --no_save_task_specific_checkpoints 可強制關閉
        # 先決定「預設要不要存 emotion/cause 專用 checkpoint」，並記錄這個決定是從哪裡來的
        if opt.save_task_specific_checkpoints is None:
            # 使用者沒有明確指定時，沿用舊行為：只有 consistency_pseudo 開啟才預設儲存
            save_task_specific_checkpoints = bool(opt.consistency_pseudo)
            save_task_specific_ckpt_source = "auto(by consistency_pseudo)"
        else:
            # 使用者有明確指定 --save_task_specific_checkpoints / --no_save_task_specific_checkpoints
            save_task_specific_checkpoints = bool(opt.save_task_specific_checkpoints)
            save_task_specific_ckpt_source = "explicit CLI override"

        # 只要訓練流程或最終測試流程有任何一處要用 emotion/cause 最佳模型，
        # 對應的 task-specific checkpoint 就會變成必需品
        task_specific_metric_needed = (
            opt.initial_supervised_metric in ('emotion', 'cause')
            or opt.self_training_main_metric in ('emotion', 'cause')
            or (
                opt.test_model_type in ('initial', 'self_training')
                and opt.test_metric in ('emotion', 'cause')
            )
        )
        if task_specific_metric_needed:
            # 如果使用者明明需要 emotion/cause checkpoint，卻又明確要求不要儲存，
            # 後面流程一定會找不到模型，所以這裡直接報錯
            if opt.save_task_specific_checkpoints is False:
                raise ValueError(
                    "當 --initial_supervised_metric、--self_training_main_metric 或 --test_metric "
                    "(搭配 initial/self_training) 使用 emotion/cause 時，"
                    "不能同時使用 --no_save_task_specific_checkpoints，"
                    "因為訓練/測試流程需要對應的最佳 checkpoint"
                )

            # 如果目前尚未啟用儲存，但後面流程又確實需要，就自動補開
            if not save_task_specific_checkpoints:
                save_task_specific_checkpoints = True
                save_task_specific_ckpt_source = (
                    "auto(因 initial_supervised_metric / self_training_main_metric / test_metric 使用 emotion/cause，故自動啟用)"
                )
        print(
            f"Task-specific checkpoint 儲存設定: {save_task_specific_checkpoints} "
            f"(來源: {save_task_specific_ckpt_source})"
        )
        print(
            f"初始監督訓練後銜接模型依據: {opt.initial_supervised_metric} "
            f"({METRIC_TO_LABEL[opt.initial_supervised_metric]})"
        )
        print(
            f"Self-training 主模型切換依據: {opt.self_training_main_metric} "
            f"({METRIC_TO_LABEL[opt.self_training_main_metric]})"
        )
        if opt.test_model_type == 'self_training_avg3':
            print("最終測試模型依據: self_training_avg3 (固定平均 emotion/cause/pair，忽略 test_metric)")
        else:
            print(
                f"最終測試模型依據: {opt.test_model_type} + {opt.test_metric} "
                f"({METRIC_TO_LABEL[opt.test_metric]})"
            )
        current_self_training_model_source = "目前記憶體中的 model (未指定來源路徑)"
        optimizer = torch.optim.AdamW(model.parameters(), lr=opt.learning_rate, weight_decay=opt.weight_decay)
        # 每折初始化訓練步驟記錄檔，與 UECA_CE_val_version.py 的行為一致
        training_metrics_file = os.path.join(save_path, f'fold{fold}_training_step_metrics.txt')
        if os.path.exists(training_metrics_file):
            os.remove(training_metrics_file)
        # 每折初始化 self-training 每輪統計 CSV
        st_round_metrics_csv = os.path.join(save_path, f'fold{fold}_self_training_round_metrics.csv')
        if os.path.exists(st_round_metrics_csv):
            os.remove(st_round_metrics_csv)
        
        # 檢查是否要跳過初始訓練
        if opt.skip_initial_training:
            print("=== 跳過初始監督訓練，直接載入預訓練模型 ===")
            pretrained_model_path = get_supervised_best_checkpoint_path(
                opt.save_path,
                fold,
                opt.initial_supervised_metric,
            )
            if os.path.exists(pretrained_model_path):
                model = torch.load(pretrained_model_path, map_location=torch.device('cuda' if use_gpu else 'cpu'))
                if use_gpu:
                    model = model.cuda()
                print(
                    f"已載入預訓練模型: {pretrained_model_path} "
                    f"(依據 {METRIC_TO_LABEL[opt.initial_supervised_metric]})"
                )
                current_self_training_model_source = pretrained_model_path
                
                # 直接跳到 self-training 部分（資料夾會在後面統一建立）
                print("=== 跳過初始訓練，準備進行 Self-Training ===")
                print("  偽標籤結果資料夾將在 Self-Training 開始時建立")
            else:
                print(f"找不到預訓練模型: {pretrained_model_path}")
                print("請確認模型檔案存在，或設定 skip_initial_training=False")
                continue
                
        elif opt.test_only:
            # 在 test_only 模式下生成實驗資料夾名稱並建立資料夾
            if opt.experiment_output_dir is None:
                opt.experiment_output_dir = generate_experiment_folder_name(opt, bert_path)
                print(f"測試結果將儲存至: {opt.experiment_output_dir}")
            
            # 確保實驗輸出資料夾存在
            os.makedirs(opt.experiment_output_dir, exist_ok=True)
            
            all_test_logits = torch.tensor([])
            all_test_label = torch.tensor([])
            all_test_mask_label = torch.tensor([])
            all_test_y_bert = torch.tensor([])
            all_test_x_bert = torch.tensor([])
            all_test_emotion_gt = torch.tensor([])
            all_test_cause_gt = torch.tensor([])
            all_test_pair_gt = torch.tensor([])
            final_eval_split = opt.final_eval_split
            final_eval_loader = testloader if final_eval_split == 'test' else valloader
            final_eval_doc_ids = NLP_Dataset['test'].doc_id if final_eval_split == 'test' else NLP_Dataset['val'].doc_id
            final_eval_output_name = 'test_results.txt' if final_eval_split == 'test' else 'val_results.txt'
            final_eval_eval_filename = f"{final_eval_split}_evaluation_fold{fold}.txt"

            if opt.test_model_type == 'self_training_avg3':
                avg_result = build_averaged_model_for_fold(
                    save_path=eval_checkpoint_root,
                    fold=fold,
                    weights=None,
                    map_location='cpu',
                    repo_root=os.path.dirname(os.path.dirname(eval_checkpoint_root)),
                    averaging_method='simple_mean',
                )
                model = avg_result['model']
                if use_gpu:
                    model = model.cuda()
                print(f"[Test-only] 使用 Avg3 最終模型進行 {final_eval_split} 評估")
                print(f"[Test-only] Avg3 來源模型: {avg_result['checkpoint_paths']}")

            model.eval()
            
            # 將結果儲存到實驗資料夾而不是 save_path
            output_file_path = os.path.join(opt.experiment_output_dir, final_eval_output_name)
            with torch.no_grad(), open(output_file_path, 'w', encoding='utf-8') as output_file:
                for idx, data in enumerate(final_eval_loader):
                    x_bert, y_bert, label, mask_label, gt_emotion, gt_cause, gt_pair, _ = data
                    if use_gpu:
                        x_bert = x_bert.cuda()
                        y_bert = y_bert.cuda()
                        label = label.cuda()
                        mask_label = mask_label.cuda()
                    loss, logits = model(x_bert, label)
                    logits = F.softmax(logits, dim=-1)
                    all_test_label = torch.cat((all_test_label, label.cpu()), 0)
                    all_test_mask_label = torch.cat((all_test_mask_label, mask_label.cpu()), 0)
                    all_test_logits = torch.cat((all_test_logits, logits.cpu()), 0)
                    all_test_y_bert = torch.cat((all_test_y_bert, y_bert.cpu()), 0)
                    all_test_x_bert = torch.cat((all_test_x_bert, x_bert.cpu()), 0)
                    all_test_emotion_gt = torch.cat((all_test_emotion_gt, gt_emotion), 0)
                    all_test_cause_gt = torch.cat((all_test_cause_gt, gt_cause), 0)
                    all_test_pair_gt = torch.cat((all_test_pair_gt, gt_pair), 0)


                p_emotion, r_emotion, f_emotion, p_cause, r_cause, f_cause, p_pair, r_pair, f_pair = crf_prompt(
                    all_test_logits, all_test_label, all_test_x_bert, all_test_emotion_gt, all_test_cause_gt,
                    all_test_pair_gt, save_path=os.path.join(opt.experiment_output_dir, final_eval_eval_filename))
                save_mask_predictions(all_test_logits, all_test_x_bert, tokenizer, final_eval_doc_ids,
                                    fold=fold, output_dir=opt.experiment_output_dir,
                                    base_filename='text_result' if final_eval_split == 'test' else 'val_text_result')
                print(
                    "e_p: {:.4f} e_r: {:.4f} e_f: {:.4f} c_p: {:.4f} c_r: {:.4f} c_f: {:.4f}"
                    " pair_p: {:.4f} pair_r: {:.4f} pair_f: {:.4f}".format(
                        p_emotion,
                        r_emotion,
                        f_emotion,
                        p_cause,
                        r_cause,
                        f_cause,
                        p_pair,
                        r_pair,
                        f_pair))
                if f_emotion > max_f1_emotion:
                    max_f1_emotion, max_p_emotion, max_r_emotion = f_emotion, p_emotion, r_emotion
                if f_cause > max_f1_cause:
                    max_f1_cause, max_p_cause, max_r_cause = f_cause, p_cause, r_cause
                if f_pair > max_f1_pair:
                    max_f1_pair, max_p_pair, max_r_pair = f_pair, p_pair, r_pair

                print(
                    "max result---- e_p: {:.4f} e_r: {:.4f} e_f: {:.4f} c_p: {:.4f} c_r: {:.4f} c_f: {:.4f}"
                    " pair_p: {:.4f} pair_r: {:.4f} pair_f: {:.4f}".format(
                        max_p_emotion, max_r_emotion, max_f1_emotion, max_p_cause, max_r_cause, max_f1_cause,
                        max_p_pair, max_r_pair, max_f1_pair))
                
                # test_only 模式下，記錄時間資訊
                fold_total_time = time.time() - fold_start_time
                print(f"\nFold {fold} (Test-only) 完成！")
                print(f"Fold {fold} 總執行時間: {fold_total_time/60:.1f} 分鐘")
                print(f"Fold {fold} 結束時間: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
                
                # 記錄 test_only 模式的時間資訊
                fold_time_info = {
                    'fold': fold,
                    'total_time': fold_total_time,
                    'supervised_time': 0,  # test_only 模式沒有訓練時間
                    'self_training_time': 0,  # test_only 模式沒有 self-training 時間
                    'is_test_only': True  # 標記為測試模式
                }
                fold_times.append(fold_time_info)
                
                # 即時寫入測試模式下當前fold的時間記錄
                with open(time_log_file, "a", encoding="utf-8") as f:
                    f.write(f"Fold {fold} 測試模式時間記錄:\n")
                    f.write(f"  開始時間: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(fold_start_time))}\n")
                    f.write(f"  結束時間: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\n")
                    f.write(f"  測試執行時間: {fold_total_time/60:.1f} 分鐘\n")
                    f.write("-" * 30 + "\n\n")
                
                # test_only 模式下，完成測試後跳過後續的 self-training
                continue

        else:
            supervised_training_start_time = time.time()  # 記錄監督訓練開始時間
            print(f"初始監督訓練開始時間: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
            
            for i in range(opt.training_iter):
                model.train()
                start_time, step = time.time(), 1
                # 此迴圈根據訓練資料集大小決定要跑幾次: 假設175筆資料、batch_size=8，\
                # 則此迴圈會跑21次(175/8=21.875，drop_last=True會捨去最後不足8筆資料) \ 
                # 21 * 8 = 168筆資料，175-168=7筆資料，所以會有7筆資料沒有被訓練到
                for index, data in enumerate(trainloader):
                    with torch.autograd.set_detect_anomaly(True):
                        x_bert, y_bert, label, mask_label, gt_emotion, gt_cause, gt_pair, is_labeled = data
                        # x_bert(輸入序列含[MASK])，shape=(batch_size, 512)
                        # y_bert(不含[MASK]的序列)，shape=(batch_size, 512)
                        # label與mask_label一樣
                        # mask_label([MASK]位置的正確答案，對應的 token ID)，shape=(batch_size, 512)，-100表示忽略
                        # gt_emotion(情緒句索引)
                        # gt_cause(原因句索引)
                        # gt_pair(情緒句和原因句索引)
                        if use_gpu:
                            x_bert = x_bert.cuda()
                            y_bert = y_bert.cuda()
                            label = label.cuda()
                            mask_label = mask_label.cuda()



                        loss, logits = model(x_bert, mask_label) # loss: 所有有效[MASK]位置的平均Cross-Entropy(有效指的是，只有 [MASK] 位置的 mask_label 才不是 -100)、shape=logits=(batch_size,512,21128)
                        logits = F.softmax(logits, dim=-1)  # shape=(batch_size, 512, 21128)


                        optimizer.zero_grad() # 清空上一輪的梯度
                        if use_gpu:
                            loss = loss.cuda()
                        loss.backward() # 反向傳播，計算梯度
                        optimizer.step() # 更新模型參數

                        print("loss: {:.4f}".format(loss))
                        # 每20個批次，執行crf_prompt，評估模型在訓練集上的表現（情緒/原因/配對的 P/R/F1）
                        if index % 20 == 0:
                            # 寫入批次到同一個文件
                            p_emotion, r_emotion, f_emotion, p_cause, r_cause, f_cause, p_pair, r_pair, f_pair = \
                                crf_prompt(logits.cpu(), label.cpu(), x_bert.cpu(), gt_emotion, gt_cause, gt_pair,
                                           save_path=training_metrics_file)
                            print(
                                "iter: {} e_p: {:.4f} e_r: {:.4f} e_f: {:.4f} c_p: {:.4f} c_r: {:.4f} c_f: {:.4f}"
                                " pair_p: {:.4f} pair_r: {:.4f} pair_f: {:.4f}".format(
                                    index,
                                    p_emotion,
                                    r_emotion,
                                    f_emotion,
                                    p_cause,
                                    r_cause,
                                    f_cause,
                                    p_pair,
                                    r_pair,
                                    f_pair))
                all_val_logits = torch.tensor([])
                all_val_label = torch.tensor([])
                all_val_mask_label = torch.tensor([])
                all_val_y_bert = torch.tensor([])
                all_val_x_bert = torch.tensor([])
                all_val_emotion_gt = torch.tensor([])
                all_val_cause_gt = torch.tensor([])
                all_val_pair_gt = torch.tensor([])
                val_loss_sum = 0.0
                val_batch_count = 0

                model.eval()
                with torch.no_grad():
                    for _, data in enumerate(valloader):  # 改為驗證集
                        x_bert, y_bert, label, mask_label, gt_emotion, gt_cause, gt_pair, _ = data
                        if use_gpu:
                            x_bert = x_bert.cuda()
                            y_bert = y_bert.cuda()
                            label = label.cuda()
                            mask_label = mask_label.cuda()
                        loss, logits = model(x_bert, label)
                        logits = F.softmax(logits, dim=-1)
                        val_loss_sum += loss.item()
                        val_batch_count += 1
                        all_val_label = torch.cat((all_val_label, label.cpu()), 0)
                        all_val_mask_label = torch.cat((all_val_mask_label, mask_label.cpu()), 0)
                        all_val_logits = torch.cat((all_val_logits, logits.cpu()), 0)
                        all_val_y_bert = torch.cat((all_val_y_bert, y_bert.cpu()), 0)
                        all_val_x_bert = torch.cat((all_val_x_bert, x_bert.cpu()), 0)
                        all_val_emotion_gt = torch.cat((all_val_emotion_gt, gt_emotion), 0)
                        all_val_cause_gt = torch.cat((all_val_cause_gt, gt_cause), 0)
                        all_val_pair_gt = torch.cat((all_val_pair_gt, gt_pair), 0)

                    avg_val_loss = val_loss_sum / max(val_batch_count, 1)
                    val_results_path = os.path.join(save_path, f"validation_results_{st_timestamp}.txt")
                    p_emotion, r_emotion, f_emotion, p_cause, r_cause, f_cause, p_pair, r_pair, f_pair = crf_prompt(
                        all_val_logits, all_val_label, all_val_x_bert, all_val_emotion_gt, all_val_cause_gt,
                        all_val_pair_gt, save_path=val_results_path)
                    print("iter{} validation result:".format(i))  # 改為驗證結果
                    print(
                        "e_p: {:.4f} e_r: {:.4f} e_f: {:.4f} c_p: {:.4f} c_r: {:.4f} c_f: {:.4f} pair_p: {:.4f}"
                        " pair_r: {:.4f} pair_f: {:.4f}".format(
                            p_emotion,
                            r_emotion,
                            f_emotion,
                            p_cause,
                            r_cause,
                            f_cause,
                            p_pair,
                            r_pair,
                            f_pair))
                    metrics_dict = {
                        "p_emotion": p_emotion,
                        "r_emotion": r_emotion,
                        "f_emotion": f_emotion,
                        "p_cause": p_cause,
                        "r_cause": r_cause,
                        "f_cause": f_cause,
                        "p_pair": p_pair,
                        "r_pair": r_pair,
                        "f_pair": f_pair,
                    }
                    if f_emotion > max_f1_emotion:
                        max_f1_emotion, max_p_emotion, max_r_emotion = f_emotion, p_emotion, r_emotion
                        emo_checkpoint_path = os.path.join(save_path, f'fold{fold}_best_emo.pth')
                        if opt.savecheckpoint and save_task_specific_checkpoints:
                            torch.save(model, emo_checkpoint_path)
                            print(f"Model for fold {fold} (best emotion F1) saved to {emo_checkpoint_path}")
                        write_best_val_checkpoint(
                            best_val_emo_info_path,
                            stage="supervised_training",
                            iteration=i + 1,
                            val_loss=avg_val_loss,
                            metrics=metrics_dict,
                            checkpoint_path=emo_checkpoint_path if (opt.savecheckpoint and save_task_specific_checkpoints) else ""
                        )
                    if f_cause > max_f1_cause:
                        max_f1_cause, max_p_cause, max_r_cause = f_cause, p_cause, r_cause
                        cause_checkpoint_path = os.path.join(save_path, f'fold{fold}_best_cause.pth')
                        if opt.savecheckpoint and save_task_specific_checkpoints:
                            torch.save(model, cause_checkpoint_path)
                            print(f"Model for fold {fold} (best cause F1) saved to {cause_checkpoint_path}")
                        write_best_val_checkpoint(
                            best_val_cause_info_path,
                            stage="supervised_training",
                            iteration=i + 1,
                            val_loss=avg_val_loss,
                            metrics=metrics_dict,
                            checkpoint_path=cause_checkpoint_path if (opt.savecheckpoint and save_task_specific_checkpoints) else ""
                        )
                    if f_pair > max_f1_pair:
                        max_f1_pair, max_p_pair, max_r_pair = f_pair, p_pair, r_pair
                        print(f"  新的最佳 F1: {f_pair:.4f} at iter {i}")
                        if opt.save_val_predictions:
                            # 只在 best pair 更新時儲存驗證集預測，確保可與最佳模型一一對應
                            save_mask_predictions(
                                all_val_logits,
                                all_val_x_bert,
                                tokenizer,
                                NLP_Dataset['val'].doc_id,
                                fold=fold,
                                output_dir=save_path,
                                base_filename="val_text_result_best_pair",
                            )
                        pair_checkpoint_path = os.path.join(save_path, f'fold{fold}_best_pair.pth')
                        if opt.savecheckpoint:
                            torch.save(model, pair_checkpoint_path)
                            print(f"Model for fold {fold} (best pair F1) saved to {pair_checkpoint_path}")
                        write_best_val_checkpoint(
                            best_val_pair_info_path,
                            stage="supervised_training",
                            iteration=i + 1,
                            val_loss=avg_val_loss,
                            metrics=metrics_dict,
                            checkpoint_path=pair_checkpoint_path if opt.savecheckpoint else ""
                        )
                        best_val_stage = "supervised_training"
                    print("iter{} test result:".format(i))
                    print(
                        "max result---- e_p: {:.4f} e_r: {:.4f} e_f: {:.4f} c_p: {:.4f} c_r: {:.4f} c_f: {:.4f}"
                        " pair_p: {:.4f} pair_r: {:.4f} pair_f: {:.4f}".format(
                            max_p_emotion, max_r_emotion, max_f1_emotion, max_p_cause, max_r_cause, max_f1_cause,
                            max_p_pair, max_r_pair, max_f1_pair))
            max_result_emo_f.append(max_f1_emotion)
            max_result_cause_f.append(max_f1_cause)
            max_result_pair_f.append(max_f1_pair)
            max_result_emo_p.append(max_p_emotion)
            max_result_cause_p.append(max_p_cause)
            max_result_pair_p.append(max_p_pair)
            max_result_emo_r.append(max_r_emotion)
            max_result_cause_r.append(max_r_cause)
            max_result_pair_r.append(max_r_pair)

            log_file = os.path.join(log_dir, f"fold{fold}_init_supervised_metrics.txt")
            with open(log_file, "w", encoding="utf-8") as f:
                f.write("Emotion F1: " + str(max_result_emo_f) + "\n")
                f.write("Emotion P: " + str(max_result_emo_p) + "\n")
                f.write("Emotion R: " + str(max_result_emo_r) + "\n")
                f.write("Cause F1: " + str(max_result_cause_f) + "\n")
                f.write("Cause P: " + str(max_result_cause_p) + "\n")
                f.write("Cause R: " + str(max_result_cause_r) + "\n")
                f.write("Pair F1: " + str(max_result_pair_f) + "\n")
                f.write("Pair P: " + str(max_result_pair_p) + "\n")
                f.write("Pair R: " + str(max_result_pair_r) + "\n")
                f.write("Early Stopping: Disabled\n")
                f.write(f"Best F1 achieved: {max_f1_pair:.4f}\n")  # 應該用 max_f1_pair 而不是 best_val_f1

            # ====== 當前第 i 折最佳模型的路徑 ======
            model_path = get_supervised_best_checkpoint_path(save_path, fold, opt.initial_supervised_metric)
            if not os.path.exists(model_path):
                raise FileNotFoundError(
                    f"找不到初始監督訓練最佳模型: {model_path} "
                    f"(initial_supervised_metric={opt.initial_supervised_metric})"
                )
            model = torch.load(model_path, map_location=torch.device('cuda' if use_gpu else 'cpu'))
            if use_gpu:
                model = model.cuda()
            print(
                f"已載入初始化監督訓練的最佳模型: {model_path} "
                f"(依據 {METRIC_TO_LABEL[opt.initial_supervised_metric]})"
            )
            current_self_training_model_source = model_path
            
            # 輸出當前 fold 的早停資訊
            supervised_training_total_time = time.time() - supervised_training_start_time
            print(f"  初始監督訓練執行時間: {supervised_training_total_time/60:.1f} 分鐘")

            print(f"  Fold {fold} - Completed all {opt.training_iter} iters, Best F1: {max_f1_pair:.4f}")

        # 如果跳過監督訓練，設定時間為0
        if opt.skip_initial_training:
            supervised_training_total_time = 0

        # ====== Self-Training 開始 ======
        # 無論是否跳過初始訓練，都會執行 self-training
        print("=== 開始 Self-Training 流程 ===")
        
        # 建立 self-training 相關資料夾，都加上時間戳記
        pseudo_results_dir = os.path.join(save_path, f"pseudo_results_{st_timestamp}")
        st_results_dir = os.path.join(save_path, f"self_training_results_{st_timestamp}")
        
        os.makedirs(pseudo_results_dir, exist_ok=True)
        os.makedirs(st_results_dir, exist_ok=True)
        
        print(f"  偽標籤結果將儲存至: {pseudo_results_dir}")
        print(f"  Self-training 結果將儲存至: {st_results_dir}")
        
        self_training_start_time = time.time()  # 記錄 self-training 開始時間
        print(f"  Self-training 開始時間: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
        nest_prev_scores = {}  # 使用 dict {doc_id: ema_score} 追蹤每個樣本的 EMA 分數
        doc_selection_history = {}  # 記錄每個 doc_id 在哪幾輪被選中 {doc_id: [round1, round2, ...]}

        # 自訓練階段的跨輪最佳指標 (用於保存全域最佳模型)
        st_max_f1_emotion = -1.0
        st_max_f1_cause = -1.0
        st_max_f1_pair = -1.0
        st_max_p_pair = -1.0
        st_max_r_pair = -1.0
        # Self-training 的每輪迴圈，根據 opt.self_training_rounds 決定要跑幾輪
        for self_round in range(opt.self_training_rounds):
            round_start_time = time.time()  # 記錄每輪開始時間
            is_dynamic_round0_mode = (
                opt.nest_loss_mode == 'nest_dynamic_gamma_round0' and
                opt.consistency_all_equal_round0_only
            )
            current_round_gamma = 0.1 if (is_dynamic_round0_mode and self_round == 0) else opt.gamma
            if is_dynamic_round0_mode:
                if self_round == 0:
                    print(f"  [NeST Dynamic Gamma] Round {self_round+1} 使用 gamma={current_round_gamma} (round0 覆寫)")
                else:
                    print(f"  [NeST Dynamic Gamma] Round {self_round+1} 使用 gamma={current_round_gamma} (回復命令列 --gamma)")
            
            # Hybrid 策略: 決定本輪使用的選擇器
            current_selector = opt.pseudo_selector
            if opt.pseudo_selector == 'hybrid':
                if opt.hybrid_alternating:
                    # 交替模式: 奇數輪 Threshold，偶數輪 NeST (輪次從 1 開始計算)
                    if (self_round + 1) % 2 == 1:  # 奇數輪 (R1, R3, R5, ...)
                        current_selector = 'threshold'
                        print(f"Hybrid Strategy (Alternating): Round {self_round+1} 使用 Threshold 模式 (奇數輪)")
                    else:  # 偶數輪 (R2, R4, R6, ...)
                        current_selector = 'nest'
                        print(f"Hybrid Strategy (Alternating): Round {self_round+1} 使用 NeST 模式 (偶數輪)")
                elif opt.hybrid_reverse:
                    # 反轉模式: 前期 NeST，後期 Threshold
                    if self_round < opt.hybrid_switch_round:
                        current_selector = 'nest'
                        print(f"Hybrid Strategy (Reverse): Round {self_round+1} 使用 NeST 模式 (尚未達到切換輪次 {opt.hybrid_switch_round})")
                    else:
                        current_selector = 'threshold'
                        print(f"Hybrid Strategy (Reverse): Round {self_round+1} 切換為 Threshold 模式")
                else:
                    # 正常模式: 前期 Threshold，後期 NeST
                    if self_round < opt.hybrid_switch_round:
                        current_selector = 'threshold'
                        print(f"Hybrid Strategy: Round {self_round+1} 使用 Threshold 模式 (尚未達到切換輪次 {opt.hybrid_switch_round})")
                    else:
                        current_selector = 'nest'
                        print(f"Hybrid Strategy: Round {self_round+1} 切換為 NeST 模式")
            print(f"=== Self-training round {self_round+1} / {opt.self_training_rounds} ===")
            print(f"  Round {self_round+1} 開始時間: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
            if unlabeled_loader is None or len(unlabeled_dataset) == 0:
                print("  未標註資料集已無可用樣本，自訓練結束")
                break
            
            # 如果不是第一輪 self-training，載入前一輪的最佳模型
            if self_round > 0:
                prev_best_model_path = get_self_training_best_checkpoint_path(
                    save_path,
                    fold,
                    opt.self_training_main_metric,
                )
                if os.path.exists(prev_best_model_path):
                    model = torch.load(prev_best_model_path, map_location=torch.device('cuda' if use_gpu else 'cpu'))
                    if use_gpu:
                        model = model.cuda()
                    print(
                        f"  已載入前一輪最佳模型: {prev_best_model_path} "
                        f"(依據 {METRIC_TO_LABEL[opt.self_training_main_metric]})"
                    )
                    current_self_training_model_source = prev_best_model_path
                    # 重新初始化優化器，使用新模型的參數
                    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.learning_rate, weight_decay=opt.weight_decay)
                else:
                    print(
                        f"  找不到前一輪最佳模型: {prev_best_model_path}，"
                        "繼續使用當前模型"
                    )

            print(f"  本輪未標註樣本主模型來源: {current_self_training_model_source}")
            
            with torch.no_grad():
                model.eval()
                round_unlabeled_size = len(unlabeled_dataset)
                round_pseudo_labeled_samples = []
                round_pseudo_doc_ids = []
                all_pseudo_predictions = []  # 收集pseudo預測結果用於最後保存
                all_unlabeled_pseudo_predictions = []  # 收集本輪所有未標註文檔的原始 pseudo 預測
                batch_start_idx = 0
                pseudo_logging_entries = []
                all_unlabeled_pseudo_logging_entries = []
                pseudo_eval_records = []
                all_unlabeled_pseudo_eval_records = []
                evaluate_pseudo = opt.log_pseudo_quality and bool(pseudo_ground_truth_map) # 決定是否要進行偽標籤 vs 真實答案的比較評估，當「使用者要求紀錄品質」且「成功建立了真實答案對照表」時
                consistency_debug_records = [] if opt.save_consistency_predictions else None
                consistency_rejected_records = []  # 記錄被 consistency 拒絕的樣本的 GT 評估

                # ===== Task Consistency 設定 =====
                is_consistency_only_mode = (current_selector == 'consistency_only')
                is_non_all_equal_rule = (opt.consistency_rule != 'all_equal')
                is_all_equal_explicitly_required = opt.consistency_require_all_equal
                is_all_equal_round0_only_active = bool(
                    opt.consistency_pseudo
                    and
                    opt.consistency_all_equal_round0_only
                    and opt.consistency_rule == 'all_equal'
                    and not is_consistency_only_mode
                )
                allow_consistency_this_round = (not is_all_equal_round0_only_active) or (self_round == 0)
                should_apply_consistency_with_rule = (
                    is_non_all_equal_rule or is_all_equal_explicitly_required
                ) and allow_consistency_this_round
                is_consistency_pseudo_enabled = (
                    opt.consistency_pseudo and should_apply_consistency_with_rule
                )
                consistency_enabled = bool(
                    is_consistency_only_mode or is_consistency_pseudo_enabled
                )  # 是否啟用一致性過濾
                if is_all_equal_round0_only_active and self_round > 0:
                    print("[Task Consistency] round0-only 模式啟用：本輪跳過 all_equal 一致性篩選")
                consistency_stats = {  # 一致性統計計數器
                    "checked": 0,  # 記錄本輪進行一致性檢查的樣本數
                    "accept": 0,  # 記錄本輪通過一致性檢查的樣本數
                    "reject": 0,  # 記錄本輪未通過一致性檢查的樣本數
                    "skip_unavailable": 0,  # 記錄因模型不可用而跳過一致性檢查的樣本數
                }
                consistency_models = None  # 暫存一致性檢查使用的三個模型 (emo, cause, pair)

                if consistency_enabled:  # 只有啟用一致性時，才需要準備三模型
                    if opt.consistency_model_source == 'self_training_best' and self_round > 0:  # 指定使用 self-training 模型且目前為第 2 輪以上
                        # 第 N 輪 (N>=2) 使用「跨輪歷史最佳」三模型，而非上一輪快照
                        emo_model_path = os.path.join(save_path, 'self_training_models', f'fold{fold}_self_training_best_emo.pth')
                        cause_model_path = os.path.join(save_path, 'self_training_models', f'fold{fold}_self_training_best_cause.pth')
                        pair_model_path = os.path.join(save_path, 'self_training_models', f'fold{fold}_self_training_best_pair.pth')
                        print("[Task Consistency] 使用跨輪歷史最佳三模型 (self_training_best_emo/cause/pair)")
                    else:
                        # Round 1 或指定 supervised_best：使用監督式三模型
                        emo_model_path = os.path.join(save_path, f'fold{fold}_best_emo.pth')  # 監督式最佳 emotion 模型路徑
                        cause_model_path = os.path.join(save_path, f'fold{fold}_best_cause.pth')  # 監督式最佳 cause 模型路徑
                        pair_model_path = os.path.join(save_path, f'fold{fold}_best_pair.pth')  # 監督式最佳 pair 模型路徑
                        if opt.consistency_model_source == 'self_training_best' and self_round == 0:  # 若使用者指定 self-training 來源但目前是第 1 輪
                            print("[Task Consistency] Round 1 尚無上一輪自訓練模型，改用監督式三模型")  # 說明自動回退行為

                    if not (os.path.exists(emo_model_path) and os.path.exists(cause_model_path) and os.path.exists(pair_model_path)):  # 任一模型檔缺失都不能做一致性檢查
                        error_msg = (
                            "[Task Consistency] 缺少一致性檢查所需三模型檔案，停止執行\n"
                            f"  emo: {emo_model_path}\n"
                            f"  cause: {cause_model_path}\n"
                            f"  pair: {pair_model_path}\n"
                            "請確認模型已正確產生，或關閉 --consistency_pseudo 後再執行"
                        )
                        raise FileNotFoundError(error_msg)
                    else:
                        emo_model = torch.load(emo_model_path, map_location=torch.device('cuda' if use_gpu else 'cpu'))  # 載入 emotion 模型
                        cause_model = torch.load(cause_model_path, map_location=torch.device('cuda' if use_gpu else 'cpu'))  # 載入 cause 模型
                        pair_model = torch.load(pair_model_path, map_location=torch.device('cuda' if use_gpu else 'cpu'))  # 載入 pair 模型
                        if use_gpu:  # 若目前使用 GPU，將三模型搬到 CUDA
                            emo_model = emo_model.cuda()  # emotion 模型移至 GPU
                            cause_model = cause_model.cuda()  # cause 模型移至 GPU
                            pair_model = pair_model.cuda()  # pair 模型移至 GPU
                        emo_model.eval()  # 設為推論模式 (關閉 dropout 等訓練行為)
                        cause_model.eval()  # 設為推論模式 (關閉 dropout 等訓練行為)
                        pair_model.eval()  # 設為推論模式 (關閉 dropout 等訓練行為)
                        consistency_models = (emo_model, cause_model, pair_model)  # 打包三模型，供後續一致性函式使用
                        print("[Task Consistency] 已載入三模型: best_emo / best_cause / best_pair")  # 載入成功提示

                pseudo_eval_state = {
                    'self_round': self_round,
                    'current_selector': current_selector,
                    'evaluate_pseudo': evaluate_pseudo,
                    'pseudo_ground_truth_map': pseudo_ground_truth_map,
                    'round_pseudo_labeled_samples': round_pseudo_labeled_samples,
                    'round_pseudo_doc_ids': round_pseudo_doc_ids,
                    'all_pseudo_predictions': all_pseudo_predictions,
                    'all_unlabeled_pseudo_predictions': all_unlabeled_pseudo_predictions,
                    'pseudo_logging_entries': pseudo_logging_entries,
                    'all_unlabeled_pseudo_logging_entries': all_unlabeled_pseudo_logging_entries,
                    'pseudo_eval_records': pseudo_eval_records,
                    'all_unlabeled_pseudo_eval_records': all_unlabeled_pseudo_eval_records,
                    'consistency_rejected_records': consistency_rejected_records,
                    'doc_selection_history': doc_selection_history,
                    'pseudo_eval_correct': 0,
                    'pseudo_eval_total': 0,
                    'pseudo_eval_filtered_correct': 0,
                    'pseudo_eval_filtered_total': 0,
                    'all_unlabeled_pseudo_eval_correct': 0,
                    'all_unlabeled_pseudo_eval_total': 0,
                    'all_unlabeled_pseudo_eval_filtered_correct': 0,
                    'all_unlabeled_pseudo_eval_filtered_total': 0,
                }

                if current_selector == 'threshold':
                    mask_token_id = tokenizer.mask_token_id
                    threshold = opt.threshold
                    apply_emotion_threshold = opt.mask_threshold_mode in ('emotion', 'both', 'all')
                    apply_cause_threshold = opt.mask_threshold_mode in ('cause', 'both', 'all')
                    apply_pair_threshold = opt.mask_threshold_mode == 'all'
                    use_or_mode = opt.mask_threshold_mode == 'or'
                    label_index = get_label_index()

                    for batch_inputs in unlabeled_loader:
                        batch_cpu = batch_inputs.cpu().numpy()
                        batch_device = batch_inputs.cuda() if use_gpu else batch_inputs
                        _, logits = model(batch_device, labels=None)
                        logits = F.softmax(logits, dim=-1)

                        batch_selected = 0
                        batch_pair_filtered = 0  # 統計被 pair 閾值過濾的樣本數 (僅 all 模式)
                        for i in range(batch_cpu.shape[0]):
                            mask_positions = np.where(batch_cpu[i] == mask_token_id)[0]
                            pseudo_label = []
                            mask_confidences = []  # 記錄每個 mask 位置的信心度
                            all_pass = True
                            cause_predictions = {}
                            clause_emotion_confident = False
                            clause_cause_confident = False

                            for j, pos in enumerate(mask_positions):
                                pos_idx = int(pos)
                                count_sentence = j // 3 + 1
                                mask_type = j % 3

                                if mask_type == 0:
                                    if use_or_mode:
                                        clause_emotion_confident = False
                                        clause_cause_confident = False
                                    prob, pred_token_id = torch.max(logits[i, pos_idx], dim=-1)
                                    prob_val = prob.item()
                                    if use_or_mode:
                                        clause_emotion_confident = prob_val >= threshold
                                    elif apply_emotion_threshold and prob_val < threshold:
                                        all_pass = False
                                        break
                                    pseudo_label.append(pred_token_id if isinstance(pred_token_id, int) else pred_token_id.item())
                                    mask_confidences.append(prob_val)
                                elif mask_type == 1:
                                    prob, pred_token_id = torch.max(logits[i, pos_idx], dim=-1)
                                    prob_val = prob.item()
                                    if use_or_mode:
                                        clause_cause_confident = prob_val >= threshold
                                        if not (clause_emotion_confident or clause_cause_confident):
                                            all_pass = False
                                            break
                                    elif apply_cause_threshold and prob_val < threshold:
                                        all_pass = False
                                        break
                                    pred_token_id_val = pred_token_id if isinstance(pred_token_id, int) else pred_token_id.item()
                                    pseudo_label.append(pred_token_id_val)
                                    mask_confidences.append(prob_val)
                                    cause_predictions[count_sentence] = (pred_token_id_val == 3221)
                                else:  # mask_type == 2 (pair position) - 第3個[MASK]，預測「配對句編號」
                                    if cause_predictions.get(count_sentence, False): # 如果該子句被預測為「原因句」
                                        # 建立候選配對編號列表 (在 window_size 範圍內的句子編號)
                                        case = [label_index[k] for k in range(
                                            max(0, -opt.window_size + count_sentence - 1),
                                            min(75, opt.window_size + count_sentence)
                                        )]
                                        case.append(3187) # 加入 "无" (無配對) 的 token id
                                        candidate_logits = logits[i, pos_idx, case] # 取得模型對這些候選的 logits
                                        # 先對 logits 做 softmax 轉成機率，然後取最大機率值和索引
                                        max_prob, selected_idx = torch.max(F.softmax(candidate_logits, dim=-1), dim=-1)
                                        pred_token_id = case[selected_idx.item()] # 取得預測的 token id
                                        # all 模式: pair 位置也需檢查閾值
                                        # 新增】如果是 'all' 模式，檢查 pair 位置的信心度
                                        if apply_pair_threshold and max_prob.item() < threshold:
                                            batch_pair_filtered += 1  # 統計被 pair 閾值過濾的樣本
                                            all_pass = False # 信心度不足，這個樣本不通過
                                            break # 跳出迴圈，不繼續處理這個樣本
                                        pseudo_label.append(pred_token_id if isinstance(pred_token_id, int) else pred_token_id.item()) # 通過檢查，加入偽標籤
                                        mask_confidences.append(max_prob.item())
                                    else:
                                        pseudo_label.append(3187) # 不是原因句 → 直接填 "无"
                                        mask_confidences.append(None)  # 非原因句，無閾值檢查

                            if all_pass and mask_positions.size > 0:
                                doc_global_idx = batch_start_idx + i
                                doc_id = unlabeled_dataset.doc_id[doc_global_idx]
                                passed, final_pseudo = apply_task_consistency_filter(
                                    x_np=batch_cpu[i],
                                    default_pseudo_tokens=np.array(pseudo_label, dtype=np.int64),
                                    consistency_enabled=consistency_enabled,
                                    consistency_models=consistency_models,
                                    consistency_stats=consistency_stats,
                                    use_gpu=use_gpu,
                                    tokenizer=tokenizer,
                                    window_size=opt.window_size,
                                    consistency_rule=opt.consistency_rule,
                                    doc_id=doc_id,
                                    consistency_debug_records=consistency_debug_records,
                                )
                                # 把本輪選中的樣本實際收集起來
                                if passed and final_pseudo is not None:
                                    append_pseudo_sample_entry(
                                        pseudo_eval_state,
                                        doc_id,
                                        batch_cpu[i],
                                        final_pseudo,
                                        tokenizer,
                                        mask_confidences=mask_confidences,
                                    )
                                    batch_selected += 1
                                elif consistency_enabled and not passed:
                                    record_consistency_rejection_entry(
                                        pseudo_eval_state,
                                        doc_id,
                                        np.array(pseudo_label, dtype=np.int64),
                                        tokenizer,
                                        mask_confidences=mask_confidences,
                                    )

                        # 輸出本 batch 的統計資訊
                        if apply_pair_threshold:
                            print(f"本 batch 收集到 {batch_selected} 筆 pseudo-labeled 樣本 (閥值: {threshold}, 模式: {opt.mask_threshold_mode}, pair閾值過濾: {batch_pair_filtered} 筆)")
                        else:
                            print(f"本 batch 收集到 {batch_selected} 筆 pseudo-labeled 樣本 (閥值: {threshold}, 模式: {opt.mask_threshold_mode})")
                        batch_start_idx += batch_cpu.shape[0]

                        if use_gpu:
                            torch.cuda.empty_cache()
                        del logits

                elif current_selector == 'nest':  # [AAAI 2023] NeST
                    # 1. 收集有標註資料集的統計資訊 (特徵向量、情緒標籤、原因標籤、情緒句分佈)
                    # 取得有標註資料集的所有 doc_id 列表，用於 debug 
                    labeled_doc_ids = [NLP_Dataset['train'].doc_id[i] for i in range(len(NLP_Dataset['train']))]
                    # 取得有標註資料集的所有 pairs 列表，用於 debug 
                    labeled_pairs_list = [NLP_Dataset['train'].pairs[i] for i in range(len(NLP_Dataset['train']))]
                    labeled_features, labeled_emotion, labeled_cause, labeled_emotion_clause, labeled_cause_clause, labeled_embedding_info = collect_labeled_statistics(
                        model, train_eval_loader, tokenizer, device,
                        debug=True,          # 啟用 debug 模式，印出每個句子的機率
                        debug_num_docs=10,    # 只對前 10 篇文檔啟用詳細 debug
                        doc_ids=labeled_doc_ids,  # 傳入文檔 ID 列表
                        pairs_list=labeled_pairs_list,  # 傳入 pairs 列表
                        knn_embedding_mode=opt.knn_embedding_mode,  # 使用參數指定的 embedding 模式
                        return_embedding_info=True  # 取得每個樣本的 embedding 詳細資訊
                    )
                    # labeled_features: 標註樣本的嵌入向量([CLS]或是情緒/原因子句)
                    # labeled_emotion: 標註樣本的情緒機率分佈[p(是)、p(非)](所有子句平均)
                    # labeled_cause: 標註樣本的原因機率分佈[p(是)、p(非)](所有子句平均)
                    # labeled_emotion_claus: 標註樣本的情緒分佈(只取預測為情緒句的子句)
                    # labeled_cause_clause: 標註樣本的原因分佈(只取預測為原因句的子句)
                    # labeled_embedding_info: 標註樣本的嵌入資訊(用於debug)

                    # ===== DEBUG: 檢查 collect_labeled_statistics 回傳值的形狀 =====
                    print("=" * 60)
                    print("[DEBUG] collect_labeled_statistics 回傳值形狀:")
                    print(f"  labeled_features: {labeled_features.shape} (type: {type(labeled_features).__name__})")
                    print(f"  labeled_emotion:  {labeled_emotion.shape} (type: {type(labeled_emotion).__name__})")
                    print(f"  labeled_cause:    {labeled_cause.shape} (type: {type(labeled_cause).__name__})")
                    print(f"  labeled_emotion_clause: {labeled_emotion_clause.shape} (type: {type(labeled_emotion_clause).__name__})")
                    print(f"  labeled_cause_clause: {labeled_cause_clause.shape} (type: {type(labeled_cause_clause).__name__})")
                    print(f"  labeled_emotion 範例 (前3筆): {labeled_emotion[:3]}")
                    print(f"  labeled_cause 範例 (前3筆):   {labeled_cause[:3]}")
                    # 顯示有標籤資料集的前3篇文檔內容
                    print("\n[DEBUG] 有標籤資料集前3篇文檔內容 (x_bert decoded):")
                    for doc_idx in range(min(3, len(NLP_Dataset['train']))):
                        x_bert_sample = NLP_Dataset['train'].x_bert[doc_idx]
                        doc_id = NLP_Dataset['train'].doc_id[doc_idx]
                        # 將 token ids 解碼成文字，移除 [PAD] 並顯示前100個字符
                        decoded_text = tokenizer.decode(x_bert_sample, skip_special_tokens=False)
                        decoded_text_clean = decoded_text.replace('[PAD]', '').strip()
                        print(f"  Doc {doc_idx} (id={doc_id}): {decoded_text_clean}")
                    print("=" * 60)
                    
                    # 2. 收集未標籤資料集的統計資訊 (特徵向量、預測分佈、情緒句分佈、偽標籤)
                    unlabeled_doc_ids = [unlabeled_dataset.doc_id[i] for i in range(len(unlabeled_dataset))]
                    unlabeled_features, unlabeled_emotion, unlabeled_cause, unlabeled_emotion_clause, unlabeled_cause_clause, all_pseudo_tokens, unlabeled_embedding_info = collect_unlabeled_statistics(
                        model, unlabeled_loader, tokenizer, device, opt.window_size,
                        debug=True, debug_num_docs=10, doc_ids=unlabeled_doc_ids,
                        knn_embedding_mode=opt.knn_embedding_mode,  # 使用命令列參數指定的 embedding 模式
                        return_embedding_info=True  # 取得每個樣本的 embedding 詳細資訊
                    )
                    record_all_unlabeled_pseudo_predictions(
                        pseudo_eval_state,
                        unlabeled_doc_ids,
                        unlabeled_dataset.x_bert,
                        all_pseudo_tokens,
                        tokenizer,
                    )
                    
                    # ===== DEBUG: 檢查 collect_unlabeled_statistics 回傳值的形狀 =====
                    print("=" * 60)
                    print("[DEBUG] collect_unlabeled_statistics 回傳值形狀:")
                    print(f"  unlabeled_features: {unlabeled_features.shape} (type: {type(unlabeled_features).__name__})")
                    print(f"  unlabeled_emotion:  {unlabeled_emotion.shape} (type: {type(unlabeled_emotion).__name__})")
                    print(f"  unlabeled_cause:    {unlabeled_cause.shape} (type: {type(unlabeled_cause).__name__})")
                    print(f"  unlabeled_emotion_clause: {unlabeled_emotion_clause.shape} (type: {type(unlabeled_emotion_clause).__name__})")
                    print(f"  unlabeled_cause_clause: {unlabeled_cause_clause.shape} (type: {type(unlabeled_cause_clause).__name__})")
                    print(f"  unlabeled_emotion 範例 (前3筆): {unlabeled_emotion[:3]}")
                    print(f"  unlabeled_cause 範例 (前3筆):   {unlabeled_cause[:3]}")
                    # 顯示未標籤資料集的前3篇文檔內容
                    print("\n[DEBUG] 未標籤資料集前3篇文檔內容 (x_bert decoded):")
                    for doc_idx in range(min(3, len(unlabeled_dataset))):
                        x_bert_sample = unlabeled_dataset.x_bert[doc_idx]
                        doc_id = unlabeled_dataset.doc_id[doc_idx]
                        # 將 token ids 解碼成文字，移除 [PAD] 並顯示前200個字符
                        decoded_text = tokenizer.decode(x_bert_sample, skip_special_tokens=False)
                        decoded_text_clean = decoded_text.replace('[PAD]', '').strip()
                        print(f"  Doc {doc_idx} (id={doc_id}): {decoded_text_clean}")
                    print("=" * 60)
                    
                    # ===== DEBUG: 中斷點 - 檢查完形狀後停在這裡 =====
                    # breakpoint()  # 在這裡暫停，在terminal按q離開
                    # 輸入 c 繼續執行，q 退出，或直接輸入變數名查看值

                    # 3. 計算本輪預計挑選的樣本數量
                    num_labeled = len(NLP_Dataset['train'])
                    num_unlabeled = len(unlabeled_dataset)
                    # 挑選數量為有標籤資料量的倍數 (opt.nest_multiplier)，至少挑 1 筆
                    num_to_select = max(1, int(opt.nest_multiplier * num_labeled)) if num_unlabeled > 0 else 0

                    if 0 < num_unlabeled < num_to_select:
                        print(
                            f"  剩餘未標註樣本數 {num_unlabeled} 少於本輪目標選樣數 {num_to_select}，"
                            "自訓練提前停止"
                        )
                        break

                    if num_to_select > 0:
                        # 4. 執行 NeST 演算法進行樣本挑選
                        # 根據特徵相似度與模型預測的一致性來選擇樣本
                        # Select \hat{X}_u^t by Eq.4 
                        nest_result = select_samples_and_generate_pseudo_labels(
                            labeled_features=labeled_features,
                            labeled_true_labels={'emotion': labeled_emotion, 'cause': labeled_cause, 'emotion_clause': labeled_emotion_clause, 'cause_clause': labeled_cause_clause},
                            unlabeled_predictions={'emotion': unlabeled_emotion, 'cause': unlabeled_cause, 'emotion_clause': unlabeled_emotion_clause, 'cause_clause': unlabeled_cause_clause},
                            unlabeled_features=unlabeled_features,
                            k=opt.nest_k,               # KNN 的 k 值
                            num_samples=num_to_select,  # 目標挑選數量
                            beta=opt.nest_beta,         # 平滑參數
                            m=opt.nest_m,               # 倍率參數
                            prev_val=nest_prev_scores,  # \mu^{(t-1)} 分數 dict {doc_id: score}
                            divergence_mode=opt.nest_divergence_mode,  # 散度計算模式
                            return_details=True,        # 返回詳細計算資訊
                            labeled_doc_ids=labeled_doc_ids,   # 有標籤樣本的 doc_id
                            unlabeled_doc_ids=unlabeled_doc_ids  # 未標籤樣本的 doc_id (用於 EMA 追蹤)
                        )
                        # 解包返回值 (新增 detailed_info)
                        selected_indices, _, current_val_array, detailed_info = nest_result
                        selected_indices = np.array(selected_indices, dtype=np.int64)
                        
                        # 更新 nest_prev_scores dict，以便下一輪使用
                        nest_prev_scores = detailed_info['current_val_dict']
                        
                        # 樣本歷史改由 append_pseudo_sample 統一記錄（僅記錄實際納入訓練者）
                        
                        print(f"NeST 選出 {len(selected_indices)} 筆候選樣本 (目標 {num_to_select})")
                        try:
                            current_val_shape = tuple(current_val_array.shape)
                        except Exception:
                            current_val_shape = 'N/A'
                        print(f"  [NeST] current_val_array.shape = {current_val_shape}")
                        try:
                            # 與 NeST/sample_selection_prompt.py 中的抽樣機率定義一致（僅用於印 log，不影響選樣結果）
                            # current_val_array: 每個未標記樣本的 (含 EMA) 散度分數，長度 = 未標記樣本數
                            # NeST 的抽樣權重公式：weights = W - μ(x)，其中 W = max(current_val)
                            weights_dbg = np.max(current_val_array) - current_val_array  # 權重越大，代表該樣本越可能被抽中

                            sum_weights_dbg = float(np.sum(weights_dbg))  # 權重總和，用於把權重正規化成機率分佈
                            if sum_weights_dbg != 0:
                                probs_dbg = weights_dbg / sum_weights_dbg  # 正規化後的抽樣機率 p(x)
                                # 計算「機率 > 0」的樣本數（replace=False 時，機率為 0 的樣本永遠抽不到）
                                # 若要求抽樣數量 > num_nonzero_p，np.random.choice 會拋出: Fewer non-zero entries in p than size
                                num_nonzero_dbg = int(np.sum(probs_dbg > 0))
                            else:
                                # 代表所有 weights 都是 0（常見原因：current_val 全部相同）→ 無法形成有效機率分佈
                                num_nonzero_dbg = 0
                            print(f"  [NeST] unlabeled={len(current_val_array)}, num_nonzero_p={num_nonzero_dbg}, sum(weights)={sum_weights_dbg:.6f}")
                        except Exception as e:
                            print(f"  [NeST] 無法計算 num_nonzero_p / sum(weights): {e}")
                        print(f"  [EMA Dict] 目前追蹤 {len(nest_prev_scores)} 個 doc_id 的 EMA 分數")
                        print(f"  [選中歷史] 累計 {len(doc_selection_history)} 個 doc_id 曾被選中")
                        
                        # ===== 記錄每個未標記樣本的散度分數 =====
                        divergence_log_path = os.path.join(
                            pseudo_results_dir,
                            f"nest_divergence_scores_fold{fold}_round{self_round + 1}.txt"
                        )
                        
                        # 從 detailed_info 取出詳細計算資訊
                        neighbor_indices = detailed_info['neighbor_indices']  # [n_unlabeled, k]
                        neighbor_distances = detailed_info.get('neighbor_distances')  # [n_unlabeled, k] - L2 平方距離
                        neighbor_doc_ids = detailed_info.get('neighbor_doc_ids')  # [n_unlabeled, k] 或 None
                        divergence_details = detailed_info.get('divergence_details')
                        raw_divergence = detailed_info.get('divergence')  # 本輪原始散度 (Round 1 時與 nest_prev_scores 相同，Round 2 以後不同)
                        
                        with open(divergence_log_path, 'w', encoding='utf-8') as f_div:
                            f_div.write(f"NeST 散度分數紀錄 - Fold {fold}, Round {self_round + 1}\n")
                            f_div.write(f"散度計算模式: {opt.nest_divergence_mode}\n")
                            f_div.write(f"KNN Embedding 模式: {opt.knn_embedding_mode}\n")
                            f_div.write(f"參數: k={opt.nest_k}, beta={opt.nest_beta}, m={opt.nest_m}\n")
                            f_div.write(f"公式: divergence = D_u + beta * D_l = D_u + {opt.nest_beta} * D_l\n")
                            f_div.write(f"KNN 距離度量: L2 平方距離 (FAISS IndexFlatL2)\n")
                            f_div.write(f"未標記樣本總數: {len(current_val_array)}\n")
                            f_div.write(f"被選中樣本數: {len(selected_indices)}\n")
                            f_div.write("=" * 80 + "\n\n")
                            
                            # ===== Embedding 類型統計 =====
                            if opt.knn_embedding_mode != 'cls':
                                f_div.write("【Embedding 類型統計 - 未標記樣本】\n")
                                n_clause = sum(1 for info in unlabeled_embedding_info if info['embedding_type'] == 'clause')
                                n_fallback = sum(1 for info in unlabeled_embedding_info if info['is_fallback'])
                                n_total = len(unlabeled_embedding_info)
                                f_div.write(f"  使用子句 embedding: {n_clause} 筆 ({n_clause/n_total*100:.1f}%)\n")
                                f_div.write(f"  Fallback 到 [CLS]: {n_fallback} 筆 ({n_fallback/n_total*100:.1f}%)\n")
                                # 子句數量統計
                                clause_counts = [info['used_clauses'] for info in unlabeled_embedding_info if not info['is_fallback']]
                                if clause_counts:
                                    f_div.write(f"  平均使用子句數: {np.mean(clause_counts):.2f}\n")
                                    f_div.write(f"  最多使用子句數: {max(clause_counts)}\n\n")
                                else:
                                    f_div.write("\n")
                                
                                f_div.write("【Embedding 類型統計 - 標記樣本 (KNN 索引)】\n")
                                n_clause_labeled = sum(1 for info in labeled_embedding_info if info['embedding_type'] == 'clause')
                                n_fallback_labeled = sum(1 for info in labeled_embedding_info if info['is_fallback'])
                                n_total_labeled = len(labeled_embedding_info)
                                f_div.write(f"  使用子句 embedding: {n_clause_labeled} 筆 ({n_clause_labeled/n_total_labeled*100:.1f}%)\n")
                                f_div.write(f"  Fallback 到 [CLS]: {n_fallback_labeled} 筆 ({n_fallback_labeled/n_total_labeled*100:.1f}%)\n\n")
                            
                            # 統計資訊 (使用 current_val_array 而非 nest_prev_scores dict)
                            f_div.write("【散度分數統計 (含 EMA 平滑)】\n")
                            f_div.write(f"  最小值: {np.min(current_val_array):.6f}\n")
                            f_div.write(f"  最大值: {np.max(current_val_array):.6f}\n")
                            f_div.write(f"  平均值: {np.mean(current_val_array):.6f}\n")
                            f_div.write(f"  標準差: {np.std(current_val_array):.6f}\n")
                            f_div.write(f"  中位數: {np.median(current_val_array):.6f}\n\n")
                            
                            # 被選中樣本的統計
                            selected_scores = current_val_array[selected_indices]
                            f_div.write("【被選中樣本的散度分數統計】\n")
                            f_div.write(f"  最小值: {np.min(selected_scores):.6f}\n")
                            f_div.write(f"  最大值: {np.max(selected_scores):.6f}\n")
                            f_div.write(f"  平均值: {np.mean(selected_scores):.6f}\n")
                            f_div.write(f"  標準差: {np.std(selected_scores):.6f}\n\n")
                            
                            # 抽樣機率統計 (對應論文 Eq. 11)
                            sample_probs_for_stats = detailed_info.get('probabilities')
                            max_divergence_for_stats = detailed_info.get('max_divergence')
                            if sample_probs_for_stats is not None:
                                max_prob_idx = np.argmax(sample_probs_for_stats)
                                min_prob_idx = np.argmin(sample_probs_for_stats)
                                max_prob_doc_id = unlabeled_dataset.doc_id[max_prob_idx]
                                min_prob_doc_id = unlabeled_dataset.doc_id[min_prob_idx]
                                
                                f_div.write("【抽樣機率統計 (論文 Eq. 11)】\n")
                                f_div.write(f"  W (最大散度): {max_divergence_for_stats:.6f}\n")
                                f_div.write(f"  最大機率: {sample_probs_for_stats[max_prob_idx]:.8f} ({sample_probs_for_stats[max_prob_idx]*100:.4f}%) - 樣本 {max_prob_idx}, Doc ID = {max_prob_doc_id}\n")
                                f_div.write(f"  最小機率: {sample_probs_for_stats[min_prob_idx]:.8f} ({sample_probs_for_stats[min_prob_idx]*100:.4f}%) - 樣本 {min_prob_idx}, Doc ID = {min_prob_doc_id}\n")
                                f_div.write(f"  平均機率: {np.mean(sample_probs_for_stats):.8f} ({np.mean(sample_probs_for_stats)*100:.4f}%)\n")
                                f_div.write(f"  機率總和: {np.sum(sample_probs_for_stats):.6f} (應為 1.0)\n\n")
                            
                            f_div.write("=" * 80 + "\n")
                            f_div.write("【所有未標記樣本的散度詳細計算】\n")
                            f_div.write("公式說明:\n")
                            f_div.write("  D_u = Σ KL(鄰居分佈 || 未標記預測分佈)，衡量與鄰居的差異\n")
                            f_div.write("  D_l = Σ KL(鄰居平均分佈 || 各鄰居分佈)，衡量鄰居間的一致性\n")
                            f_div.write("  divergence = D_u + beta * D_l\n\n")
                            
                            selected_set = set(selected_indices.tolist())
                            
                            # 取得抽樣機率相關資訊
                            sample_weights = detailed_info.get('weights')  # W - μ^(t)(x_j)
                            sample_probs = detailed_info.get('probabilities')  # 正規化後機率
                            max_divergence = detailed_info.get('max_divergence')  # W
                            
                            for idx in range(len(current_val_array)):
                                doc_id = unlabeled_dataset.doc_id[idx]
                                score = current_val_array[idx]
                                is_selected = "✓ 選中" if idx in selected_set else ""
                                
                                f_div.write("-" * 80 + "\n")
                                f_div.write(f"樣本 {idx}: Doc ID = {doc_id} {is_selected}\n")
                                f_div.write(f"  最終散度分數 (含 EMA): {score:.6f}\n")
                                
                                # 顯示本輪原始散度
                                if raw_divergence is not None:
                                    f_div.write(f"  本輪原始散度 (不含 EMA): {raw_divergence[idx]:.6f}\n")
                                
                                # 顯示抽樣機率 (對應論文 Eq. 11)
                                if sample_weights is not None and sample_probs is not None:
                                    f_div.write(f"  抽樣權重 (W - μ): {sample_weights[idx]:.6f}  (W = {max_divergence:.6f})\n")
                                    f_div.write(f"  抽樣機率 p(x): {sample_probs[idx]:.8f}  ({sample_probs[idx]*100:.4f}%)\n")
                                
                                # 顯示 D_u 和 D_l 詳細資訊
                                if divergence_details is not None:
                                    D_u = divergence_details.get('D_u')
                                    D_l = divergence_details.get('D_l')
                                    score_u_per_neighbor = divergence_details.get('score_u_per_neighbor')
                                    score_l_per_neighbor = divergence_details.get('score_l_per_neighbor')
                                    # 新增: 分子分母
                                    D_u_numerator = divergence_details.get('D_u_numerator')  # [n_unlabeled, k, n_classes]
                                    D_u_denominator = divergence_details.get('D_u_denominator')  # [n_unlabeled, n_classes]
                                    D_l_numerator = divergence_details.get('D_l_numerator')  # [n_unlabeled, n_classes] (y_bar)
                                    D_l_denominator = divergence_details.get('D_l_denominator')  # [n_unlabeled, k, n_classes]
                                    
                                    if D_u is not None and D_l is not None:
                                        f_div.write(f"  D_u (與鄰居差異): {D_u[idx]:.6f}\n")
                                        f_div.write(f"  D_l (鄰居一致性): {D_l[idx]:.6f}\n")
                                        f_div.write(f"  驗算: D_u + {opt.nest_beta} * D_l = {D_u[idx]:.6f} + {opt.nest_beta} * {D_l[idx]:.6f} = {D_u[idx] + opt.nest_beta * D_l[idx]:.6f}\n")
                                    
                                    # 顯示每個鄰居的貢獻
                                    if score_u_per_neighbor is not None:
                                        f_div.write(f"  各鄰居對 D_u 的貢獻: {score_u_per_neighbor[idx].tolist()}\n")
                                    if score_l_per_neighbor is not None:
                                        f_div.write(f"  各鄰居對 D_l 的貢獻: {score_l_per_neighbor[idx].tolist()}\n")
                                    
                                    # 顯示 D_u 的分子分母 (用於驗算)
                                    f_div.write(f"\n  【D_u 詳細計算】\n")
                                    f_div.write(f"  公式: D_u = Σ_j Σ_c [ 分子 * log(分子/分母) ]\n")
                                    f_div.write(f"       分子 = 鄰居 soft label, 分母 = 未標記樣本預測機率\n")
                                    if D_u_denominator is not None:
                                        f_div.write(f"  未標記樣本預測 (分母): {D_u_denominator[idx].tolist()}\n")
                                    if D_u_numerator is not None:
                                        f_div.write(f"  各鄰居 soft label (分子):\n")
                                        for j, neighbor_idx in enumerate(neighbor_indices[idx]):
                                            neighbor_doc = neighbor_doc_ids[idx][j] if neighbor_doc_ids is not None else f"idx={neighbor_idx}"
                                            f_div.write(f"    鄰居{j} ({neighbor_doc}): {D_u_numerator[idx][j].tolist()}\n")
                                    
                                    # 顯示 D_l 的分子分母 (用於驗算)
                                    f_div.write(f"\n  【D_l 詳細計算】\n")
                                    f_div.write(f"  公式: D_l = Σ_j Σ_c [ 分子 * log(分子/分母) ]\n")
                                    f_div.write(f"       分子 = 鄰居平均分佈 ȳ, 分母 = 各鄰居 soft label\n")
                                    if D_l_numerator is not None:
                                        f_div.write(f"  鄰居平均分佈 ȳ (分子): {D_l_numerator[idx].tolist()}\n")
                                    if D_l_denominator is not None:
                                        f_div.write(f"  各鄰居 soft label (分母): 同上 D_u 的分子\n")
                                
                                # 顯示鄰居資訊 (包含 L2 距離用於驗證 KNN)
                                neighbor_idx_list = neighbor_indices[idx].tolist()
                                f_div.write(f"\n  【KNN 鄰居資訊】\n")
                                f_div.write(f"  鄰居索引 (在 labeled 中): {neighbor_idx_list}\n")
                                if neighbor_doc_ids is not None:
                                    neighbor_doc_id_list = neighbor_doc_ids[idx].tolist()
                                    f_div.write(f"  鄰居文檔 ID: {neighbor_doc_id_list}\n")
                                if neighbor_distances is not None:
                                    neighbor_dist_list = neighbor_distances[idx].tolist()
                                    f_div.write(f"  L2 平方距離: {neighbor_dist_list}\n")
                                    f_div.write(f"  (距離越小表示特徵向量越相似，是越近的鄰居)\n")
                                
                                # ===== 新增: 顯示原始文本用於驗證向量與文檔的對應關係 =====
                                f_div.write(f"\n  【原始文本 (用於驗證向量對應)】\n")
                                # 未標記樣本的原始文本
                                if hasattr(unlabeled_dataset, 'raw_text') and idx < len(unlabeled_dataset.raw_text):
                                    unlabeled_text = unlabeled_dataset.raw_text[idx]
                                    # 截斷過長的文本，避免輸出過於冗長
                                    if len(unlabeled_text) > 200:
                                        unlabeled_text = unlabeled_text[:200] + "..."
                                    f_div.write(f"  未標記樣本文本: {unlabeled_text}\n")
                                
                                # 鄰居 (有標籤樣本) 的原始文本
                                train_dataset = NLP_Dataset['train']
                                if hasattr(train_dataset, 'raw_text'):
                                    f_div.write(f"  鄰居文本:\n")
                                    # 遍歷當前未標記樣本的所有 KNN 鄰居
                                    for j, neighbor_idx in enumerate(neighbor_indices[idx]):
                                        # 取得鄰居的文檔 ID，若無則顯示索引
                                        neighbor_doc = neighbor_doc_ids[idx][j] if neighbor_doc_ids is not None else f"idx={neighbor_idx}"
                                        if neighbor_idx < len(train_dataset.raw_text):
                                            neighbor_text = train_dataset.raw_text[neighbor_idx]
                                            # 截斷過長的文本，避免輸出過於冗長
                                            if len(neighbor_text) > 150:
                                                neighbor_text = neighbor_text[:150] + "..."
                                            # 取得該鄰居的 L2 平方距離
                                            l2_dist = neighbor_distances[idx][j] if neighbor_distances is not None else "N/A"
                                            f_div.write(f"    鄰居{j} (Doc {neighbor_doc}, L2²={l2_dist:.4f}): {neighbor_text}\n")
                                
                                # ===== Embedding 類型資訊 =====
                                if opt.knn_embedding_mode != 'cls':
                                    f_div.write(f"\n  【Embedding 資訊】\n")
                                    # 未標記樣本的 embedding 資訊
                                    unlabeled_info = unlabeled_embedding_info[idx]
                                    f_div.write(f"  類型: {unlabeled_info['embedding_type']}\n")
                                    f_div.write(f"  使用的子句數量: {unlabeled_info['used_clauses']}\n")
                                    f_div.write(f"  是否 Fallback: {'是' if unlabeled_info['is_fallback'] else '否'}\n")
                                    if unlabeled_info['selected_clause_indices']:
                                        f_div.write(f"  選中的子句索引: {unlabeled_info['selected_clause_indices']}\n")
                                        f_div.write(f"  選中的子句內容:\n")
                                        for ci, (clause_idx, clause_text) in enumerate(zip(unlabeled_info['selected_clause_indices'], unlabeled_info['selected_clause_texts'])):
                                            # 截斷過長的子句文本
                                            display_text = clause_text[:100] + "..." if len(clause_text) > 100 else clause_text
                                            f_div.write(f"    子句{clause_idx}: {display_text}\n")
                                    
                                    # 鄰居的 embedding 資訊
                                    f_div.write(f"\n  【鄰居 Embedding 資訊】\n")
                                    for j, neighbor_idx in enumerate(neighbor_indices[idx]):
                                        neighbor_doc = neighbor_doc_ids[idx][j] if neighbor_doc_ids is not None else f"idx={neighbor_idx}"
                                        if neighbor_idx < len(labeled_embedding_info):
                                            neighbor_info = labeled_embedding_info[neighbor_idx]
                                            emb_desc = "子句" if neighbor_info['embedding_type'] == 'clause' else ("Fallback [CLS]" if neighbor_info['is_fallback'] else "[CLS]")
                                            n_clauses = neighbor_info['used_clauses']
                                            f_div.write(f"    鄰居{j} (Doc {neighbor_doc}): {emb_desc}, 使用 {n_clauses} 個子句\n")
                                
                                # ===== 顯示 CLS 向量用於手動驗算 L2 距離 =====
                                # 只對第一筆樣本顯示完整 768 維，其他樣本只顯示前 10 維以節省空間
                                # 注意：unlabeled_features/labeled_features 可能是 PyTorch Tensor，需轉成 numpy 才能進行數值運算
                                unlabeled_vec = unlabeled_features[idx].cpu().numpy() if isinstance(unlabeled_features, torch.Tensor) else unlabeled_features[idx]
                                
                                if idx == 0:
                                    # 第一筆樣本：顯示完整 768 維向量，方便完整驗算
                                    f_div.write(f"\n  【CLS 向量 (完整 768 維，僅第一筆樣本顯示)】\n")
                                    f_div.write(f"  未標記樣本 CLS 向量 (768維):\n")
                                    f_div.write(f"    {unlabeled_vec.tolist()}\n")
                                    
                                    # 檢查是否有鄰居可供比較
                                    if len(neighbor_indices[idx]) > 0:
                                        # 取得第一個鄰居 (最近鄰) 的索引
                                        first_neighbor_idx = neighbor_indices[idx][0]
                                        # 將鄰居的特徵向量轉為 numpy 陣列
                                        labeled_vec = labeled_features[first_neighbor_idx].cpu().numpy() if isinstance(labeled_features, torch.Tensor) else labeled_features[first_neighbor_idx]
                                        f_div.write(f"  鄰居0 CLS 向量 (768維):\n")
                                        f_div.write(f"    {labeled_vec.tolist()}\n")
                                        
                                        # 手動計算 L2 平方距離 (使用 float32 精度與 FAISS 內部計算一致)
                                        unlabeled_vec_f32 = unlabeled_vec.astype(np.float32)
                                        labeled_vec_f32 = labeled_vec.astype(np.float32)
                                        manual_l2_squared = np.sum((unlabeled_vec_f32 - labeled_vec_f32) ** 2)
                                        # 取得 FAISS 回傳的 L2 平方距離
                                        faiss_l2_squared = neighbor_distances[idx][0] if neighbor_distances is not None else "N/A"
                                        f_div.write(f"\n  【L2 平方距離驗算-第idx個未標記樣本的鄰居0】\n")
                                        f_div.write(f"  公式: L2² = Σᵢ (uᵢ - vᵢ)², i = 0, 1, ..., 767 (float32 精度)\n")
                                        f_div.write(f"  手動計算 L2²: {manual_l2_squared:.6f}\n")
                                        f_div.write(f"  FAISS 回傳 L2²: {faiss_l2_squared:.6f}\n")
                                        # 使用 np.isclose 比較兩個浮點數是否在容差範圍內相等
                                        # 注意：即使都用 float32，np.sum() 累加與 FAISS C++ 累加仍有微小差異
                                        # 768 維累加可能產生約 0.0003 的誤差，所以用 rtol=1e-4 (0.01% 容差)
                                        is_match = np.isclose(manual_l2_squared, faiss_l2_squared, rtol=1e-4) if neighbor_distances is not None else False
                                        f_div.write(f"  驗算結果: {'✓ 一致' if is_match else '✗ 不一致'}\n")
                                else:
                                    # 其他樣本：只顯示前 10 維和驗算結果，以節省輸出空間
                                    f_div.write(f"\n  【CLS 向量 (前10維)】\n")
                                    f_div.write(f"  未標記樣本 CLS[0:10]: {unlabeled_vec[:10].tolist()}\n")
                                    
                                    # 檢查是否有鄰居可供比較
                                    if len(neighbor_indices[idx]) > 0:
                                        # 取得第一個鄰居 (最近鄰) 的索引
                                        first_neighbor_idx = neighbor_indices[idx][0]
                                        # 將鄰居的特徵向量轉為 numpy 陣列
                                        labeled_vec = labeled_features[first_neighbor_idx].cpu().numpy() if isinstance(labeled_features, torch.Tensor) else labeled_features[first_neighbor_idx]
                                        f_div.write(f"  鄰居0 CLS[0:10]: {labeled_vec[:10].tolist()}\n")
                                        
                                        # 手動計算 L2 平方距離 (使用 float32 精度與 FAISS 內部計算一致，仍用完整 768 維計算)
                                        unlabeled_vec_f32 = unlabeled_vec.astype(np.float32)
                                        labeled_vec_f32 = labeled_vec.astype(np.float32)
                                        manual_l2_squared = np.sum((unlabeled_vec_f32 - labeled_vec_f32) ** 2)
                                        # 取得 FAISS 回傳的 L2 平方距離
                                        faiss_l2_squared = neighbor_distances[idx][0] if neighbor_distances is not None else "N/A"
                                        f_div.write(f"  手動計算 L2² (768維, float32): {manual_l2_squared:.6f}\n")
                                        f_div.write(f"  FAISS 回傳 L2²: {faiss_l2_squared:.6f}\n")
                                        # 使用 np.isclose 比較兩個浮點數是否在容差範圍內相等
                                        # 注意：即使都用 float32，np.sum() 累加與 FAISS C++ 累加仍有微小差異
                                        is_match = np.isclose(manual_l2_squared, faiss_l2_squared, rtol=1e-4) if neighbor_distances is not None else False
                                        f_div.write(f"  驗算結果: {'✓ 一致' if is_match else '✗ 不一致'}\n")
                        
                        print(f"  散度分數已記錄至: {divergence_log_path}")
                        
                        # 直接使用 NeST 函數已計算好的機率 (避免重複計算)
                        probabilities = detailed_info.get('probabilities')
                        
                        # 同時輸出 JSON 格式方便後續分析
                        divergence_json_path = os.path.join(
                            pseudo_results_dir,
                            f"nest_divergence_scores_fold{fold}_round{self_round + 1}.json"
                        )
                        
                        # 每個樣本的詳細資訊
                        samples_data = []
                        for idx in range(len(current_val_array)):
                            sample_info = {
                                'index': int(idx),
                                'doc_id': unlabeled_dataset.doc_id[idx],
                                'divergence_score': float(current_val_array[idx]),
                                'probability': float(probabilities[idx]) if probabilities is not None else 0.0,
                                'selected': idx in selected_set,
                                'neighbor_indices': neighbor_indices[idx].tolist(),
                            }
                            
                            # 加入鄰居 L2 平方距離 (用於驗證 KNN)
                            if neighbor_distances is not None:
                                sample_info['neighbor_l2_distances'] = neighbor_distances[idx].tolist()
                            
                            # 加入鄰居文檔 ID
                            if neighbor_doc_ids is not None:
                                sample_info['neighbor_doc_ids'] = neighbor_doc_ids[idx].tolist()
                            
                            # 加入本輪原始散度
                            if raw_divergence is not None:
                                sample_info['raw_divergence'] = float(raw_divergence[idx])
                            
                            # 加入 D_u, D_l 詳細資訊
                            if divergence_details is not None:
                                D_u = divergence_details.get('D_u')
                                D_l = divergence_details.get('D_l')
                                score_u_per_neighbor = divergence_details.get('score_u_per_neighbor')
                                score_l_per_neighbor = divergence_details.get('score_l_per_neighbor')
                                # 分子分母
                                D_u_numerator = divergence_details.get('D_u_numerator')
                                D_u_denominator = divergence_details.get('D_u_denominator')
                                D_l_numerator = divergence_details.get('D_l_numerator')
                                
                                if D_u is not None:
                                    sample_info['D_u'] = float(D_u[idx])
                                if D_l is not None:
                                    sample_info['D_l'] = float(D_l[idx])
                                if score_u_per_neighbor is not None:
                                    sample_info['score_u_per_neighbor'] = score_u_per_neighbor[idx].tolist()
                                if score_l_per_neighbor is not None:
                                    sample_info['score_l_per_neighbor'] = score_l_per_neighbor[idx].tolist()
                                
                                # 加入分子分母詳細資訊 (用於驗算)
                                if D_u_denominator is not None:
                                    sample_info['D_u_denominator'] = D_u_denominator[idx].tolist()  # 未標記樣本預測
                                if D_u_numerator is not None:
                                    sample_info['D_u_numerator'] = D_u_numerator[idx].tolist()  # 各鄰居 soft label [k, n_classes]
                                if D_l_numerator is not None:
                                    sample_info['D_l_numerator'] = D_l_numerator[idx].tolist()  # 鄰居平均分佈 y_bar
                            
                            # 加入原始文本 (用於驗證向量對應)
                            if hasattr(unlabeled_dataset, 'raw_text') and idx < len(unlabeled_dataset.raw_text):
                                sample_info['raw_text'] = unlabeled_dataset.raw_text[idx]
                            
                            # 加入鄰居的原始文本
                            train_dataset_json = NLP_Dataset['train']
                            if hasattr(train_dataset_json, 'raw_text'):
                                neighbor_texts = []
                                for j, neighbor_idx in enumerate(neighbor_indices[idx]):
                                    if neighbor_idx < len(train_dataset_json.raw_text):
                                        neighbor_texts.append(train_dataset_json.raw_text[neighbor_idx])
                                    else:
                                        neighbor_texts.append(None)
                                sample_info['neighbor_raw_texts'] = neighbor_texts
                            
                            samples_data.append(sample_info)
                        
                        divergence_data = {
                            'fold': fold,
                            'round': self_round + 1,
                            'divergence_mode': opt.nest_divergence_mode,
                            'params': {
                                'k': opt.nest_k,
                                'beta': opt.nest_beta,
                                'm': opt.nest_m,
                                'multiplier': opt.nest_multiplier
                            },
                            'formula': 'divergence = D_u + beta * D_l',
                            'statistics': {
                                'total_unlabeled': len(current_val_array),
                                'num_selected': len(selected_indices),
                                'min': float(np.min(current_val_array)),
                                'max': float(np.max(current_val_array)),
                                'mean': float(np.mean(current_val_array)),
                                'std': float(np.std(current_val_array)),
                                'median': float(np.median(current_val_array))
                            },
                            'labeled_doc_ids': labeled_doc_ids,  # 有標籤樣本的 doc_id 列表
                            'samples': samples_data
                        }
                        with open(divergence_json_path, 'w', encoding='utf-8') as f_json:
                            json.dump(divergence_data, f_json, ensure_ascii=False, indent=2)

                        # 5. 直接使用已產生的偽標籤 (避免重複推論)
                        for idx in selected_indices.tolist():
                            pseudo_tokens = all_pseudo_tokens[idx]
                            if pseudo_tokens is None:
                                continue
                            doc_id = unlabeled_dataset.doc_id[idx]
                            passed, final_pseudo = apply_task_consistency_filter(
                                x_np=unlabeled_dataset.x_bert[idx],
                                default_pseudo_tokens=pseudo_tokens,
                                consistency_enabled=consistency_enabled,
                                consistency_models=consistency_models,
                                consistency_stats=consistency_stats,
                                use_gpu=use_gpu,
                                tokenizer=tokenizer,
                                window_size=opt.window_size,
                                consistency_rule=opt.consistency_rule,
                                doc_id=doc_id,
                                consistency_debug_records=consistency_debug_records,
                            )
                            if passed and final_pseudo is not None:
                                append_pseudo_sample_entry(
                                    pseudo_eval_state,
                                    doc_id,
                                    unlabeled_dataset.x_bert[idx],
                                    final_pseudo,
                                    tokenizer,
                                )
                            elif consistency_enabled and not passed:
                                record_consistency_rejection_entry(
                                    pseudo_eval_state,
                                    doc_id,
                                    pseudo_tokens,
                                    tokenizer,
                                )
                    else:
                        print("NeST 無可選樣本，跳過偽標籤產生")

                elif current_selector == 'random':  # 隨機選擇策略
                    # 1. 收集未標籤資料的偽標籤 (仍需模型推論來產生偽標籤)
                    unlabeled_doc_ids = [unlabeled_dataset.doc_id[i] for i in range(len(unlabeled_dataset))]
                    _, _, _, _, _, all_pseudo_tokens = collect_unlabeled_statistics(
                        model, unlabeled_loader, tokenizer, device, opt.window_size,
                        debug=False, doc_ids=unlabeled_doc_ids,
                        knn_embedding_mode='cls'  # 隨機選擇不需要特殊 embedding
                    )
                    record_all_unlabeled_pseudo_predictions(
                        pseudo_eval_state,
                        unlabeled_doc_ids,
                        unlabeled_dataset.x_bert,
                        all_pseudo_tokens,
                        tokenizer,
                    )
                    
                    # 2. 計算本輪預計挑選的樣本數量 (與 NeST 使用相同參數)
                    num_labeled = len(NLP_Dataset['train'])
                    num_unlabeled = len(unlabeled_dataset)
                    num_to_select = max(1, int(opt.nest_multiplier * num_labeled)) if num_unlabeled > 0 else 0
                    num_to_select = min(num_to_select, num_unlabeled)  # 確保不超過可用樣本數
                    
                    if num_to_select > 0:
                        # 3. 均勻隨機抽樣 (無放回)
                        selected_indices = np.random.choice(
                            np.arange(num_unlabeled),
                            size=num_to_select,
                            replace=False,
                            p=None  # 均勻分佈
                        )
                        
                        print(f"Random 選出 {len(selected_indices)} 筆候選樣本 (目標 {num_to_select})")
                        
                        # 樣本歷史改由 append_pseudo_sample 統一記錄（僅記錄實際納入訓練者）
                        
                        # 4. 將選中的樣本加入訓練集
                        for idx in selected_indices.tolist():
                            pseudo_tokens = all_pseudo_tokens[idx]
                            if pseudo_tokens is None:
                                continue
                            doc_id = unlabeled_dataset.doc_id[idx]
                            passed, final_pseudo = apply_task_consistency_filter(
                                x_np=unlabeled_dataset.x_bert[idx],
                                default_pseudo_tokens=pseudo_tokens,
                                consistency_enabled=consistency_enabled,
                                consistency_models=consistency_models,
                                consistency_stats=consistency_stats,
                                use_gpu=use_gpu,
                                tokenizer=tokenizer,
                                window_size=opt.window_size,
                                consistency_rule=opt.consistency_rule,
                                doc_id=doc_id,
                                consistency_debug_records=consistency_debug_records,
                            )
                            if passed and final_pseudo is not None:
                                append_pseudo_sample_entry(
                                    pseudo_eval_state,
                                    doc_id,
                                    unlabeled_dataset.x_bert[idx],
                                    final_pseudo,
                                    tokenizer,
                                )
                            # 紀錄被拒絕的樣本以供後續分析
                            elif consistency_enabled and not passed:
                                record_consistency_rejection_entry(
                                    pseudo_eval_state,
                                    doc_id,
                                    pseudo_tokens,
                                    tokenizer,
                                )
                    else:
                        print("Random 無可選樣本，跳過偽標籤產生")

                elif current_selector == 'consistency_only':  # 僅使用 task consistency 篩選
                    print(
                        "Consistency-only 模式: 跳過 threshold/NeST 選樣，直接用一致性規則"
                        f" ({opt.consistency_rule}) 篩選未標註樣本"
                    )
                    if consistency_models is None:
                        raise RuntimeError(
                            "Consistency-only 模式需要可用的一致性三模型，"
                            "需確認 best_emo/best_cause/best_pair 檔案存在"
                        )

                    mask_token_id = tokenizer.mask_token_id
                    yes_token_id, _ = get_binary_token_ids(tokenizer)
                    for batch_inputs in unlabeled_loader:
                        batch_cpu = batch_inputs.cpu().numpy()
                        batch_device = batch_inputs.cuda() if use_gpu else batch_inputs
                        _, logits = model(batch_device, labels=None)
                        logits = F.softmax(logits, dim=-1)

                        batch_selected = 0
                        for i in range(batch_cpu.shape[0]):
                            doc_global_idx = batch_start_idx + i
                            doc_id = unlabeled_dataset.doc_id[doc_global_idx]
                            pseudo_tokens = generate_pseudo_label_tokens(
                                logits=logits[i].detach().cpu().numpy(),
                                input_ids=batch_cpu[i],
                                mask_token_id=mask_token_id,
                                yes_token_id=yes_token_id,
                                window_size=opt.window_size,
                                tokenizer=tokenizer,
                                threshold=None,
                            )

                            if pseudo_tokens is None or len(pseudo_tokens) == 0:
                                raise RuntimeError(
                                    "[Consistency-only] 產生到空偽標籤，終止執行\n"
                                    f"  fold={fold}, round={self_round + 1}, batch_sample_idx={i}, doc_id={doc_id}\n"
                                    "  請檢查該樣本是否符合 3-mask 模板，且含有有效 [MASK] 位置"
                                )

                            passed, final_pseudo = apply_task_consistency_filter(
                                x_np=batch_cpu[i],
                                default_pseudo_tokens=np.array(pseudo_tokens, dtype=np.int64),
                                consistency_enabled=consistency_enabled,
                                consistency_models=consistency_models,
                                consistency_stats=consistency_stats,
                                use_gpu=use_gpu,
                                tokenizer=tokenizer,
                                window_size=opt.window_size,
                                consistency_rule=opt.consistency_rule,
                                doc_id=doc_id,
                                consistency_debug_records=consistency_debug_records,
                            )
                            if passed and final_pseudo is not None:
                                append_pseudo_sample_entry(
                                    pseudo_eval_state,
                                    doc_id,
                                    batch_cpu[i],
                                    final_pseudo,
                                    tokenizer,
                                )
                                batch_selected += 1
                            elif consistency_enabled and not passed:
                                record_consistency_rejection_entry(
                                    pseudo_eval_state,
                                    doc_id,
                                    np.array(pseudo_tokens, dtype=np.int64),
                                    tokenizer,
                                )

                        print(f"本 batch 收集到 {batch_selected} 筆 pseudo-labeled 樣本 (模式: consistency_only/{opt.consistency_rule})")
                        batch_start_idx += batch_cpu.shape[0]


                print(f"=== Self-training round {self_round+1} 結果 ===")
                print(f"本輪收集到 {len(round_pseudo_labeled_samples)} 筆 pseudo-labeled 樣本")
                print(f"Unlabeled 資料集大小(開始時): {round_unlabeled_size}")
                pseudo_ratio = len(round_pseudo_labeled_samples) / max(round_unlabeled_size, 1) * 100 if round_unlabeled_size > 0 else 0
                print(f"Pseudo-labeling 成功率: {pseudo_ratio:.2f}%")
                if consistency_enabled:
                    print(f"Task consistency 檢查數: {consistency_stats['checked']}")
                    print(f"Task consistency 通過: {consistency_stats['accept']}")
                    print(f"Task consistency 拒絕: {consistency_stats['reject']}")
                    if consistency_stats['checked'] > 0:
                        consistency_accept_ratio = consistency_stats['accept'] / consistency_stats['checked'] * 100
                        print(f"Task consistency 通過率: {consistency_accept_ratio:.2f}%")
                elif opt.consistency_pseudo:
                    print(f"Task consistency 已啟用但本輪未套用 (跳過: {consistency_stats['skip_unavailable']})")

                if consistency_debug_records:
                    consistency_debug_path = os.path.join(
                        pseudo_results_dir,
                        f"consistency_predictions_fold{fold}_round{self_round + 1}.json",
                    )
                    with open(consistency_debug_path, "w", encoding="utf-8") as fjson:
                        json.dump(consistency_debug_records, fjson, ensure_ascii=False, indent=2)
                    print(f"  consistency 三模型預測已寫入: {consistency_debug_path}")

                if consistency_rejected_records:
                    rejected_path = os.path.join(
                        pseudo_results_dir,
                        f"consistency_rejected_eval_fold{fold}_round{self_round + 1}.json",
                    )
                    with open(rejected_path, "w", encoding="utf-8") as fjson:
                        json.dump(consistency_rejected_records, fjson, ensure_ascii=False, indent=2)
                    print(f"  consistency 拒絕樣本 GT 評估已寫入: {rejected_path}")
                
                # 一次性保存所有pseudo預測結果
                if all_pseudo_predictions:
                    pseudo_x_bert_all = [item['x_bert'] for item in all_pseudo_predictions]
                    pseudo_labels_all = [item['pseudo_labels'] for item in all_pseudo_predictions]
                    pseudo_doc_ids_all = [item['doc_id'] for item in all_pseudo_predictions]
                    
                    save_mask_predictions(
                        logits=None,
                        x_bert=pseudo_x_bert_all,
                        tokenizer=tokenizer,
                        doc_ids=pseudo_doc_ids_all,
                        fold=fold,
                        output_dir=pseudo_results_dir,
                        base_filename="pseudo_text_result",
                        pseudo_labels=pseudo_labels_all
                    )

                    if pseudo_logging_entries:
                        pseudo_json_path = os.path.join(
                            pseudo_results_dir,
                            f"pseudo_labeled_samples_fold{fold}_round{self_round + 1}.json",
                        )
                        with open(pseudo_json_path, "w", encoding="utf-8") as fjson:
                            json.dump(pseudo_logging_entries, fjson, ensure_ascii=False, indent=2)
                        print(f"  偽標籤樣本紀錄已寫入: {pseudo_json_path}")

                if all_unlabeled_pseudo_predictions:
                    selected_doc_id_set = set(round_pseudo_doc_ids)
                    for entry in all_unlabeled_pseudo_logging_entries:
                        entry['selected_for_training'] = entry['doc_id'] in selected_doc_id_set

                    all_unlabeled_x_bert = [item['x_bert'] for item in all_unlabeled_pseudo_predictions]
                    all_unlabeled_pseudo_labels = [item['pseudo_labels'] for item in all_unlabeled_pseudo_predictions]
                    all_unlabeled_doc_ids = [item['doc_id'] for item in all_unlabeled_pseudo_predictions]

                    save_mask_predictions(
                        logits=None,
                        x_bert=all_unlabeled_x_bert,
                        tokenizer=tokenizer,
                        doc_ids=all_unlabeled_doc_ids,
                        fold=fold,
                        output_dir=pseudo_results_dir,
                        base_filename=f"all_unlabeled_pseudo_text_result_round{self_round + 1}",
                        pseudo_labels=all_unlabeled_pseudo_labels
                    )

                    all_unlabeled_json_path = os.path.join(
                        pseudo_results_dir,
                        f"all_unlabeled_pseudo_predictions_fold{fold}_round{self_round + 1}.json",
                    )
                    with open(all_unlabeled_json_path, "w", encoding="utf-8") as fjson:
                        json.dump(all_unlabeled_pseudo_logging_entries, fjson, ensure_ascii=False, indent=2)
                    print(f"  全部未標註偽標籤紀錄已寫入: {all_unlabeled_json_path}")

                if opt.log_pseudo_quality:
                    if not pseudo_eval_records and pseudo_logging_entries and not evaluate_pseudo:
                        pseudo_eval_records.append({
                            'doc_id': 'N/A',
                            'skip_reason': 'ground_truth_unavailable',
                        })
                    report_path = write_pseudo_label_evaluation_report(
                        output_dir=pseudo_results_dir,
                        fold=fold,
                        round_idx=self_round,
                        records=pseudo_eval_records,
                        total_tokens=pseudo_eval_state['pseudo_eval_total'],
                        correct_tokens=pseudo_eval_state['pseudo_eval_correct'],
                        filtered_total_tokens=pseudo_eval_state['pseudo_eval_filtered_total'],
                        filtered_correct_tokens=pseudo_eval_state['pseudo_eval_filtered_correct'],
                        tokenizer=tokenizer,
                        summary=pseudo_gt_summary,
                    )
                    if pseudo_eval_state['pseudo_eval_total'] > 0:
                        pseudo_error_rate = 1.0 - (
                            pseudo_eval_state['pseudo_eval_correct'] / pseudo_eval_state['pseudo_eval_total']
                        )
                        print(f"  偽標籤錯誤率 (round {self_round + 1}): {pseudo_error_rate:.4f}")
                    else:
                        print("  偽標籤錯誤率: 無可比較之真實答案")
                    if pseudo_eval_state['pseudo_eval_filtered_total'] > 0:
                        filtered_error_rate = 1.0 - (
                            pseudo_eval_state['pseudo_eval_filtered_correct'] /
                            pseudo_eval_state['pseudo_eval_filtered_total']
                        )
                        print(f"  偽標籤錯誤率(排除非非无) (round {self_round + 1}): {filtered_error_rate:.4f}")
                    elif evaluate_pseudo:
                        print("  偽標籤錯誤率(排除非非无): 無剩餘可比較資料")
                    print(f"  偽標籤詳細紀錄輸出至: {report_path}")

                    if not all_unlabeled_pseudo_eval_records and all_unlabeled_pseudo_logging_entries and not evaluate_pseudo:
                        raise RuntimeError(
                            "全部未標註 pseudo 評估失敗: 已收集全部偽標籤，但本輪未啟用 ground truth 比對"
                        )

                    selected_doc_id_set = set(round_pseudo_doc_ids)
                    comparable_all_unlabeled_records = [
                        rec for rec in all_unlabeled_pseudo_eval_records if not rec.get('skip_reason')
                    ]
                    all_unlabeled_full_correct_docs = sum(
                        1 for rec in comparable_all_unlabeled_records if rec.get('fully_correct')
                    )
                    unselected_full_correct_docs = sum(
                        1 for rec in comparable_all_unlabeled_records
                        if rec.get('fully_correct') and rec['doc_id'] not in selected_doc_id_set
                    )
                    selected_full_correct_docs = sum(
                        1 for rec in comparable_all_unlabeled_records
                        if rec.get('fully_correct') and rec['doc_id'] in selected_doc_id_set
                    )
                    all_unlabeled_pair_correct_docs = sum(
                        1 for rec in comparable_all_unlabeled_records if rec.get('pair_fully_correct')
                    )
                    unselected_pair_correct_docs = sum(
                        1 for rec in comparable_all_unlabeled_records
                        if rec.get('pair_fully_correct') and rec['doc_id'] not in selected_doc_id_set
                    )
                    selected_pair_correct_docs = sum(
                        1 for rec in comparable_all_unlabeled_records
                        if rec.get('pair_fully_correct') and rec['doc_id'] in selected_doc_id_set
                    )
                    all_unlabeled_total_errors = (
                        pseudo_eval_state['all_unlabeled_pseudo_eval_total'] -
                        pseudo_eval_state['all_unlabeled_pseudo_eval_correct']
                    )

                    all_unlabeled_report_path = write_pseudo_label_evaluation_report(
                        output_dir=pseudo_results_dir,
                        fold=fold,
                        round_idx=self_round,
                        records=all_unlabeled_pseudo_eval_records,
                        total_tokens=pseudo_eval_state['all_unlabeled_pseudo_eval_total'],
                        correct_tokens=pseudo_eval_state['all_unlabeled_pseudo_eval_correct'],
                        filtered_total_tokens=pseudo_eval_state['all_unlabeled_pseudo_eval_filtered_total'],
                        filtered_correct_tokens=pseudo_eval_state['all_unlabeled_pseudo_eval_filtered_correct'],
                        tokenizer=tokenizer,
                        summary=pseudo_gt_summary,
                        report_filename=f"all_unlabeled_pseudo_evaluation_fold{fold}_round{self_round + 1}.txt",
                        extra_summary_lines=[
                            f"全部未標註 pseudo 錯誤 mask 數量: {all_unlabeled_total_errors}",
                            f"全部未標註中，整篇 pseudo 全對的文檔數: {all_unlabeled_full_correct_docs}",
                            f"已被選中且整篇 pseudo 全對的文檔數: {selected_full_correct_docs}",
                            f"未被選中但整篇 pseudo 全對的文檔數: {unselected_full_correct_docs}",
                            f"全部未標註中，pair 偽標籤全對的文檔數: {all_unlabeled_pair_correct_docs}",
                            f"已被選中且 pair 偽標籤全對的文檔數: {selected_pair_correct_docs}",
                            f"未被選中但 pair 偽標籤全對的文檔數: {unselected_pair_correct_docs}",
                            f"未被選中的文檔數: {len(all_unlabeled_pseudo_predictions) - len(selected_doc_id_set)}",
                        ],
                    )
                    if pseudo_eval_state['all_unlabeled_pseudo_eval_total'] > 0:
                        all_unlabeled_error_rate = 1.0 - (
                            pseudo_eval_state['all_unlabeled_pseudo_eval_correct'] /
                            pseudo_eval_state['all_unlabeled_pseudo_eval_total']
                        )
                        print(f"  全部未標註偽標籤錯誤率 (round {self_round + 1}): {all_unlabeled_error_rate:.4f}")
                        print(f"  全部未標註偽標籤錯誤 mask 數量: {all_unlabeled_total_errors}")
                        print(f"  未被選中但整篇 pseudo 全對的文檔數: {unselected_full_correct_docs}")
                        print(f"  未被選中但 pair 偽標籤全對的文檔數: {unselected_pair_correct_docs}")
                    else:
                        print("  全部未標註偽標籤錯誤率: 無可比較之真實答案")
                    print(f"  全部未標註偽標籤詳細紀錄輸出至: {all_unlabeled_report_path}")

            # 建立 pseudo-labeled dataset
            if len(round_pseudo_labeled_samples) == 0:
                print("  本輪未選出偽標籤樣本，自訓練提前停止")
                break
            
            pseudo_dataset = PseudoLabeledDataset(round_pseudo_labeled_samples, round_pseudo_doc_ids)
            # 本輪的 \hat{X}_u^t
            pseudo_dataset_list.append(pseudo_dataset)
            total_pseudo_samples = sum(len(ds) for ds in pseudo_dataset_list)
            print(f"  新增偽標籤資料集，共 {len(pseudo_dataset)} 筆樣本，累計偽標籤樣本 {total_pseudo_samples}")

            if opt.retain_pseudo_in_unlabeled:
                print("  保留偽標籤樣本於未標註資料集中，後續輪次會重新評估這些樣本")
            else:
                unlabeled_dataset.remove_by_doc_ids(round_pseudo_doc_ids)
                if len(unlabeled_dataset) == 0:
                    print("  未標註資料集已耗盡，後續自訓練將結束")
            if not opt.retain_pseudo_in_unlabeled:
                unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=opt.batch_size, shuffle=False) if len(unlabeled_dataset) > 0 else None

            # 合併 labeled 與所有偽標籤資料集
            datasets_to_concat = [NLP_Dataset['train']] + pseudo_dataset_list
            combined_dataset = ConcatDataset(datasets_to_concat)
            print(f"  建立合併資料集: labeled={len(NLP_Dataset['train'])}, pseudo累計={total_pseudo_samples}")


            # 建立新的 DataLoader (包含初始10%有標籤樣本 + 當前第self_round的偽標籤樣本)
            combined_loader = DataLoader(combined_dataset, batch_size=opt.batch_size, shuffle=True, drop_last=True)
            
            # 在每個 self-training round 中訓練多個 epoch
            print(f"  開始訓練 {opt.st_training_epochs} 個 epoch...")
            latest_epoch_stats = None  # 記錄最後一個 epoch 的 loss 與 confident ratio
            for epoch in range(opt.st_training_epochs):
                model.train()
                epoch_total_loss = 0.0
                epoch_labeled_loss = 0.0
                epoch_pseudo_loss = 0.0
                batch_count = 0
                # 這三個列表/計數器用於每個 epoch 的統計摘要
                total_loss_values = []
                labeled_loss_values = []
                pseudo_loss_values = []
                epoch_valid_pseudo_tokens = 0
                epoch_confident_tokens = 0
                
                for batch_idx, batch in enumerate(combined_loader):
                    x_bert, y_bert, label, mask_label, gt_emotion, gt_cause, gt_pair, is_labeled = batch
                    labeled_mask = (is_labeled == True)
                    pseudo_mask = (is_labeled == False)

                    # 取出各自的資料
                    if labeled_mask.sum() > 0:
                        x_l = x_bert[labeled_mask]
                        mask_label_l = mask_label[labeled_mask]
                        loss_l, _ = model(x_l.cuda() if use_gpu else x_l, mask_label_l.cuda() if use_gpu else mask_label_l)
                    else:
                        loss_l = 0

                    if pseudo_mask.sum() > 0:
                        x_p = x_bert[pseudo_mask]
                        mask_label_p = mask_label[pseudo_mask]
                        
                        if opt.nest_loss_mode in ('nest', 'nest_dynamic_gamma_round0'):
                            # NeST 論文的 threshold 過濾方式: 只有信心 > γ 的位置才計入 loss
                            loss_p, valid_tokens_batch, confident_tokens_batch = compute_nest_threshold_loss(
                                model, x_p, mask_label_p, opt.nest_loss_threshold, use_gpu, return_stats=True
                            )
                            epoch_valid_pseudo_tokens += valid_tokens_batch
                            epoch_confident_tokens += confident_tokens_batch
                            logits_p = None  # nest 模式不需要 logits_p (debug 用)
                        else:
                            # 標準方式: 所有偽標籤都計入 loss
                            loss_p, logits_p = model(x_p.cuda() if use_gpu else x_p, mask_label_p.cuda() if use_gpu else mask_label_p)
                            # standard 模式不做門檻過濾：分子與分母相同
                            valid_tokens_batch = int(((x_p == 103) & (mask_label_p != -100)).sum().item()) # &表示須滿足x_p == 103且mask_label_p != -100才計入有效偽標籤token數量
                            epoch_valid_pseudo_tokens += valid_tokens_batch
                            epoch_confident_tokens += valid_tokens_batch
                        
                        # Debug: 在特定 round/epoch/batch 輸出 loss_p 驗證資訊
                        # 條件1: 第1輪第1個epoch | 條件2: 第5輪第3個epoch (注意: 索引從0開始)
                        should_print_debug = (self_round == 0 and epoch == 0 and batch_idx == 0) or \
                                             (self_round == 4 and epoch == 2 and batch_idx == 0)
                        if should_print_debug:
                            num_pseudo_samples = x_p.shape[0]  # 批次中的偽標籤樣本數量
                            
                            print("\n" + "="*80)
                            print(f"=== loss_p 計算驗證 (Round {self_round+1}, Epoch {epoch+1}, Batch {batch_idx+1}) ===")
                            print(f"    nest_loss_mode: {opt.nest_loss_mode}")
                            if opt.nest_loss_mode in ('nest', 'nest_dynamic_gamma_round0'):
                                print(f"    nest_loss_threshold: {opt.nest_loss_threshold}")
                            print(f"    批次中的偽標籤樣本數量: {num_pseudo_samples}")
                            print("="*80)
                            
                            # nest 模式下 logits_p 為 None，跳過詳細驗證
                            if logits_p is None:
                                print("    [nest 模式] 詳細驗證已跳過 (使用 compute_nest_threshold_loss 計算)")
                                print("="*80 + "\n")
                            else:
                                # 取第一個偽標籤樣本
                                # 從 GPU 移到 CPU 並轉成 numpy，取得原始輸入序列 (含 [MASK])
                                sample_x = x_p[0].cpu().numpy()
                                # 取得該樣本的偽標籤 (只有 [MASK] 位置有值，其他是 -100)
                                sample_mask_label = mask_label_p[0].cpu()
                                # 取得模型對該樣本的預測 logits (21128 維對應 vocab size)，detach() 斷開梯度追蹤
                                sample_logits = logits_p[0].cpu().detach()
                            
                                # 找出 [MASK] 位置
                                mask_positions = np.where(sample_x == 103)[0]
                                
                                # 顯示 decoded 的輸入文檔
                                print(f"\n[0] 文檔 decoded (x_p，含特殊 token):")
                                print(tokenizer.decode(sample_x, skip_special_tokens=False))
                                
                                print(f"\n[1] 該樣本有 {len(mask_positions)} 個 [MASK] 位置")
                                print(f"    所有 [MASK] 位置索引: {list(mask_positions)}")
                                
                                # 顯示所有 [MASK] 的詳細計算
                                print(f"\n[2] 所有 {len(mask_positions)} 個 [MASK] 位置的 Cross-Entropy 計算:")
                                print("-"*80)
                                
                                total_ce = 0.0
                                num_valid = 0
                                for i, pos in enumerate(mask_positions):
                                    # 取得該 [MASK] 位置的「正確答案」(偽標籤的 token id)，.item() 將 tensor 轉成 Python int
                                    target = sample_mask_label[pos].item()
                                    # 若該位置的標籤是 -100，代表不需要計算 loss (跳過)
                                    if target == -100:
                                        continue
                                    num_valid += 1
                                    
                                    # 取得該位置的 logits (21128 維，對應 vocab size)
                                    logits_at_pos = sample_logits[pos]
                                    
                                    # 計算 softmax 得到機率分佈(對所有 21128 個詞彙的分數做 softmax，得到每個詞的機率)
                                    probs = F.softmax(logits_at_pos, dim=-1)
                                    
                                    # 取得偽標籤和模型預測
                                    target_prob = probs[target].item()
                                    pred_token = logits_at_pos.argmax().item()
                                    pred_prob = probs[pred_token].item()
                                    
                                    # 計算該位置的 CE loss = -log(p(target))
                                    ce_loss = -np.log(target_prob + 1e-10)
                                    total_ce += ce_loss
                                    
                                    print(f"\n    [MASK] #{i+1} (位置 {pos}):")
                                    print(f"      偽標籤 token id: {target} → '{tokenizer.decode([target])}'")
                                    print(f"      偽標籤的機率 p(target): {target_prob:.6f}")
                                    print(f"      模型預測 token id: {pred_token} → '{tokenizer.decode([pred_token])}'")
                                    print(f"      模型預測的機率 p(pred): {pred_prob:.6f}")
                                    print(f"      CE loss = -log({target_prob:.6f}) = {ce_loss:.6f}")
                                
                                print("-"*80)
                                manual_avg_ce = total_ce / max(num_valid, 1)
                                hf_loss = loss_p.item()
                                
                                print(f"\n[3] Cross-Entropy 公式: CE = -Σ log(p(target_i))")
                                print(f"    所有 {num_valid} 個 [MASK] 的 CE 總和 = {total_ce:.6f}")
                                print(f"    手動計算的平均 CE (僅第一個樣本) = {manual_avg_ce:.6f}")
                                
                                # 計算整個 batch 所有偽標籤樣本的平均 CE (用來驗證 HuggingFace 的 loss_p)
                                batch_total_ce = 0.0       # 累計整個 batch 的 CE 總和
                                batch_num_valid = 0        # 累計整個 batch 有效的 [MASK] 數量
                                
                                # 遍歷批次中的每個偽標籤樣本
                                for sample_idx in range(x_p.shape[0]):
                                    s_x = x_p[sample_idx].cpu().numpy()         # 第 sample_idx 個樣本的輸入
                                    s_label = mask_label_p[sample_idx].cpu()    # 第 sample_idx 個樣本的偽標籤
                                    s_logits = logits_p[sample_idx].cpu().detach()  # 第 sample_idx 個樣本的預測 logits
                                    
                                    # 遍歷該樣本中的每個 [MASK] 位置 (token id = 103)
                                    for pos in np.where(s_x == 103)[0]:
                                        t = s_label[pos].item()  # 該位置的偽標籤 token id
                                        if t == -100:            # -100 表示忽略該位置
                                            continue
                                        # 計算該 [MASK] 位置的 CE loss
                                        prob = F.softmax(s_logits[pos], dim=-1)[t].item()  # 模型預測偽標籤的機率
                                        batch_total_ce += -np.log(prob + 1e-10)            # CE = -log(p)
                                        batch_num_valid += 1
                                
                                # 計算平均 CE (需與 HuggingFace 的計算方式一致)
                                # BertForMaskedLM 的 output.loss 已是「所有有效 [MASK] 位置」的平均 CE
                                batch_avg_ce = batch_total_ce / max(batch_num_valid, 1)
                                
                                print(f"    手動計算的平均 CE (整個 batch) = {batch_avg_ce:.6f}")
                                
                                print(f"\n[4] HuggingFace 回傳的 loss_p (有效 [MASK] 平均 CE): {hf_loss:.6f}")
                                
                                # 比較整個 batch 的差異
                                diff = abs(batch_avg_ce - hf_loss)
                                diff_percent = (diff / max(hf_loss, 1e-10)) * 100
                                print(f"\n[5] 驗證結果 (整個 batch):")
                                print(f"    手動計算 vs HuggingFace 差異: {diff:.6f} ({diff_percent:.2f}%)")
                                if diff_percent < 1:
                                    print("    ✓ 兩者一致！(差異 < 1%)")
                                else:
                                    print(f"    ⚠ 差異較大")
                                print("="*80 + "\n")
                    else:
                        loss_p = 0

                    if opt.nest_loss_mode in ('nest', 'nest_dynamic_gamma_round0'):
                        # NeST 論文公式(5): total_loss = λ × L_sup + (1-λ) × L_st
                        total_loss = current_round_gamma * loss_l + (1 - current_round_gamma) * loss_p
                    else:
                        # 標準自訓練公式: total_loss = L_sup + γ × L_st
                        total_loss = loss_l + current_round_gamma * loss_p
                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()
                    
                    # 記錄損失
                    epoch_total_loss += total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss
                    epoch_labeled_loss += loss_l.item() if isinstance(loss_l, torch.Tensor) else loss_l
                    epoch_pseudo_loss += loss_p.item() if isinstance(loss_p, torch.Tensor) else loss_p
                    total_loss_values.append(total_loss.item() if isinstance(total_loss, torch.Tensor) else float(total_loss))
                    labeled_loss_values.append(loss_l.item() if isinstance(loss_l, torch.Tensor) else float(loss_l))
                    pseudo_loss_values.append(loss_p.item() if isinstance(loss_p, torch.Tensor) else float(loss_p))
                    batch_count += 1
                
                # 每個 epoch 結束後輸出平均損失
                if batch_count > 0:
                    avg_total_loss = epoch_total_loss / batch_count
                    avg_labeled_loss = epoch_labeled_loss / batch_count
                    avg_pseudo_loss = epoch_pseudo_loss / batch_count
                    total_loss_std = float(np.std(total_loss_values)) if total_loss_values else 0.0
                    labeled_loss_std = float(np.std(labeled_loss_values)) if labeled_loss_values else 0.0
                    pseudo_loss_std = float(np.std(pseudo_loss_values)) if pseudo_loss_values else 0.0
                    confident_ratio = (
                        epoch_confident_tokens / max(epoch_valid_pseudo_tokens, 1)
                        if epoch_valid_pseudo_tokens > 0 else 0.0
                    )
                    latest_epoch_stats = {
                        'epoch': epoch + 1,
                        'total_mean': avg_total_loss,
                        'total_std': total_loss_std,
                        'sup_mean': avg_labeled_loss,
                        'sup_std': labeled_loss_std,
                        'pseudo_mean': avg_pseudo_loss,
                        'pseudo_std': pseudo_loss_std,
                        'valid_tokens': epoch_valid_pseudo_tokens,
                        'confident_tokens': epoch_confident_tokens,
                        'confident_ratio': confident_ratio,
                    }
                    print(f"    Epoch {epoch+1}/{opt.st_training_epochs}: "
                          f"Total Loss={avg_total_loss:.4f}, "
                          f"Labeled Loss={avg_labeled_loss:.4f}, "
                          f"Pseudo Loss={avg_pseudo_loss:.4f}, "
                          f"Confident Ratio={confident_ratio:.4f} "
                          f"({epoch_confident_tokens}/{epoch_valid_pseudo_tokens})")
                
            print(f"  完成 {opt.st_training_epochs} 個 epoch 的訓練")

            # 在驗證集上測試當前模型性能 (而不是測試集)
            model.eval()
            all_val_logits = torch.tensor([])
            all_val_label = torch.tensor([])
            all_val_mask_label = torch.tensor([])
            all_val_y_bert = torch.tensor([])
            all_val_x_bert = torch.tensor([])
            all_val_emotion_gt = torch.tensor([])
            all_val_cause_gt = torch.tensor([])
            all_val_pair_gt = torch.tensor([])
            st_val_loss_sum = 0.0
            st_val_batch_count = 0

            with torch.no_grad():
                for _, data in enumerate(valloader):  # 改為驗證集
                    x_bert, y_bert, label, mask_label, gt_emotion, gt_cause, gt_pair, _ = data
                    if use_gpu:
                        x_bert = x_bert.cuda()
                        y_bert = y_bert.cuda()
                        label = label.cuda()
                        mask_label = mask_label.cuda()
                    loss, logits = model(x_bert, label)
                    logits = F.softmax(logits, dim=-1)
                    st_val_loss_sum += loss.item()
                    st_val_batch_count += 1
                    all_val_label = torch.cat((all_val_label, label.cpu()), 0)
                    all_val_mask_label = torch.cat((all_val_mask_label, mask_label.cpu()), 0)
                    all_val_logits = torch.cat((all_val_logits, logits.cpu()), 0)
                    all_val_y_bert = torch.cat((all_val_y_bert, y_bert.cpu()), 0)
                    all_val_x_bert = torch.cat((all_val_x_bert, x_bert.cpu()), 0)
                    all_val_emotion_gt = torch.cat((all_val_emotion_gt, gt_emotion), 0)
                    all_val_cause_gt = torch.cat((all_val_cause_gt, gt_cause), 0)
                    all_val_pair_gt = torch.cat((all_val_pair_gt, gt_pair), 0)

            st_avg_val_loss = st_val_loss_sum / max(st_val_batch_count, 1)
            p_emotion, r_emotion, f_emotion, p_cause, r_cause, f_cause, p_pair, r_pair, f_pair = crf_prompt(
                all_val_logits, all_val_label, all_val_x_bert, all_val_emotion_gt, all_val_cause_gt,
                all_val_pair_gt, save_path=os.path.join(st_results_dir, f"self_training_val_results_fold{fold}_round{self_round+1}.txt"))
            if opt.save_val_predictions:
                # 每一輪都儲存驗證集預測，確保可與 self_training_val_results_fold{fold}_round{round}.txt 一一對應
                save_mask_predictions(
                    all_val_logits,
                    all_val_x_bert,
                    tokenizer,
                    NLP_Dataset['val'].doc_id,
                    fold=fold,
                    output_dir=st_results_dir,
                    base_filename=f"self_training_val_text_result_round{self_round+1}",
                )
            print(
                f"[Self-training round {self_round+1}] Validation: "
                f"val_loss: {st_avg_val_loss:.4f} e_f: {f_emotion:.4f} c_f: {f_cause:.4f} pair_f: {f_pair:.4f}"
            )
            # 追加每輪統計到 CSV，便於跨設定比較 loss 與 confident_ratio
            if latest_epoch_stats is not None:
                append_round_metrics_csv(
                    st_round_metrics_csv,
                    [
                        fold,
                        self_round + 1,
                        latest_epoch_stats['epoch'],
                        opt.nest_loss_mode,
                        current_round_gamma,
                        opt.nest_loss_threshold,
                        f"{latest_epoch_stats['total_mean']:.6f}",
                        f"{latest_epoch_stats['total_std']:.6f}",
                        f"{latest_epoch_stats['sup_mean']:.6f}",
                        f"{latest_epoch_stats['sup_std']:.6f}",
                        f"{latest_epoch_stats['pseudo_mean']:.6f}",
                        f"{latest_epoch_stats['pseudo_std']:.6f}",
                        latest_epoch_stats['valid_tokens'],
                        latest_epoch_stats['confident_tokens'],
                        f"{latest_epoch_stats['confident_ratio']:.6f}",
                        f"{st_avg_val_loss:.6f}",
                        f"{f_emotion:.6f}",
                        f"{f_cause:.6f}",
                        f"{f_pair:.6f}",
                    ],
                )

            # 建立 self-training 專用的子資料夾
            st_save_dir = os.path.join(save_path, 'self_training_models')
            if opt.savecheckpoint:
                os.makedirs(st_save_dir, exist_ok=True)
                # 每輪都保存三模型快照 (供下一輪 consistency 使用)
                round_idx = self_round + 1
                round_emo_path = os.path.join(st_save_dir, f'fold{fold}_round{round_idx}_best_emo.pth')
                round_cause_path = os.path.join(st_save_dir, f'fold{fold}_round{round_idx}_best_cause.pth')
                round_pair_path = os.path.join(st_save_dir, f'fold{fold}_round{round_idx}_best_pair.pth')
                if save_task_specific_checkpoints:
                    torch.save(model, round_emo_path)
                    torch.save(model, round_cause_path)
                torch.save(model, round_pair_path)

            # 更新並保存「跨輪全域最佳」三模型
            if f_emotion > st_max_f1_emotion:
                st_max_f1_emotion = f_emotion
                st_best_emo_path = os.path.join(st_save_dir, f'fold{fold}_self_training_best_emo.pth')
                if opt.savecheckpoint and save_task_specific_checkpoints:
                    torch.save(model, st_best_emo_path)
                    print(f"   Self-training 最佳 Emotion 模型已儲存: {st_best_emo_path}")
                metrics_dict = {
                    "p_emotion": p_emotion,
                    "r_emotion": r_emotion,
                    "f_emotion": f_emotion,
                    "p_cause": p_cause,
                    "r_cause": r_cause,
                    "f_cause": f_cause,
                    "p_pair": p_pair,
                    "r_pair": r_pair,
                    "f_pair": f_pair,
                }
                write_best_val_checkpoint(
                    best_val_emo_info_path,
                    stage=f"self_training_round_{self_round + 1}",
                    iteration=self_round + 1,
                    val_loss=st_avg_val_loss,
                    metrics=metrics_dict,
                    checkpoint_path=st_best_emo_path if (opt.savecheckpoint and save_task_specific_checkpoints) else "",
                    train_loss_summary=latest_epoch_stats,
                    pseudo_confidence_stats=latest_epoch_stats,
                )

            if f_cause > st_max_f1_cause:
                st_max_f1_cause = f_cause
                st_best_cause_path = os.path.join(st_save_dir, f'fold{fold}_self_training_best_cause.pth')
                if opt.savecheckpoint and save_task_specific_checkpoints:
                    torch.save(model, st_best_cause_path)
                    print(f"   Self-training 最佳 Cause 模型已儲存: {st_best_cause_path}")
                metrics_dict = {
                    "p_emotion": p_emotion,
                    "r_emotion": r_emotion,
                    "f_emotion": f_emotion,
                    "p_cause": p_cause,
                    "r_cause": r_cause,
                    "f_cause": f_cause,
                    "p_pair": p_pair,
                    "r_pair": r_pair,
                    "f_pair": f_pair,
                }
                write_best_val_checkpoint(
                    best_val_cause_info_path,
                    stage=f"self_training_round_{self_round + 1}",
                    iteration=self_round + 1,
                    val_loss=st_avg_val_loss,
                    metrics=metrics_dict,
                    checkpoint_path=st_best_cause_path if (opt.savecheckpoint and save_task_specific_checkpoints) else "",
                    train_loss_summary=latest_epoch_stats,
                    pseudo_confidence_stats=latest_epoch_stats,
                )

            if f_pair > st_max_f1_pair:
                st_max_f1_pair, st_max_p_pair, st_max_r_pair = f_pair, p_pair, r_pair
                st_model_path = os.path.join(st_save_dir, f'fold{fold}_self_training_best_pair.pth')
                if opt.savecheckpoint:
                    torch.save(model, st_model_path)
                    print(f"   Self-training 最佳 Pair 模型已儲存: {st_model_path}")
                    print(f"   當前最佳 pair F1: {st_max_f1_pair:.4f}")
                metrics_dict = {
                    "p_emotion": p_emotion,
                    "r_emotion": r_emotion,
                    "f_emotion": f_emotion,
                    "p_cause": p_cause,
                    "r_cause": r_cause,
                    "f_cause": f_cause,
                    "p_pair": p_pair,
                    "r_pair": r_pair,
                    "f_pair": f_pair,
                }
                write_best_val_checkpoint(
                    best_val_pair_info_path,
                    stage=f"self_training_round_{self_round + 1}",
                    iteration=self_round + 1,
                    val_loss=st_avg_val_loss,
                    metrics=metrics_dict,
                    checkpoint_path=st_model_path if opt.savecheckpoint else "",
                    train_loss_summary=latest_epoch_stats,
                    pseudo_confidence_stats=latest_epoch_stats,
                )
                best_val_stage = f"self_training_round_{self_round + 1}"
            
            
        
        # 在測試集上進行評估，輸出與 UECA_CE_val_version.py 相同的紀錄
        if opt.test_model_type == 'self_training_avg3':
            avg_output_dir = os.path.join(save_path, 'self_training_models')
            avg_model_path = os.path.join(avg_output_dir, f'fold{fold}_self_training_best_avg3.pth')
            avg_state_dict_path = os.path.join(avg_output_dir, f'fold{fold}_self_training_best_avg3_state_dict.pth')
            try:
                avg_result = build_averaged_model_for_fold(
                    save_path=eval_checkpoint_root,
                    fold=fold,
                    weights=None,
                    map_location='cpu',
                    repo_root=os.path.dirname(os.path.dirname(eval_checkpoint_root)),
                    averaging_method='simple_mean',
                )
                save_averaged_model(
                    model=avg_result['model'],
                    state_dict=avg_result['state_dict'],
                    output_model_path=avg_model_path,
                    output_state_dict_path=avg_state_dict_path,
                )
                test_model = avg_result['model']
                print(f"已建立並儲存 Avg3 模型: {avg_model_path}")
                print(f"Avg3 參數平均方法: {avg_result['averaging_method']}")
                print(f"Avg3 來源模型: {avg_result['checkpoint_paths']}")
                if use_gpu:
                    test_model = test_model.cuda()
            except Exception as exc:
                raise RuntimeError(
                    f"建立 Avg3 模型失敗 (fold={fold})，已中止執行: {exc}"
                ) from exc
        elif opt.test_model_type == 'self_training':
            candidate_path = get_test_checkpoint_path(
                eval_checkpoint_root,
                fold,
                opt.test_model_type,
                opt.test_metric,
            )
            if os.path.exists(candidate_path):
                test_checkpoint_path = candidate_path
            else:
                raise FileNotFoundError(
                    f"找不到 self-training 測試模型: {candidate_path}，"
                    f"已中止執行 (test_metric={opt.test_metric})"
                )
        else:
            test_checkpoint_path = get_test_checkpoint_path(
                eval_checkpoint_root,
                fold,
                opt.test_model_type,
                opt.test_metric,
            )

        if opt.test_model_type != 'self_training_avg3':
            if os.path.exists(test_checkpoint_path):
                test_model = torch.load(test_checkpoint_path, map_location=torch.device('cuda' if use_gpu else 'cpu'))
                print(
                    f"載入測試用模型: {test_checkpoint_path} "
                    f"(依據 {METRIC_TO_LABEL[opt.test_metric]})"
                )
                if use_gpu:
                    test_model = test_model.cuda()
            else:
                raise FileNotFoundError(
                    f"找不到測試模型 {test_checkpoint_path}，已中止執行"
                )

        final_eval_split = opt.final_eval_split
        final_eval_loader = testloader if final_eval_split == 'test' else valloader
        final_eval_doc_ids = NLP_Dataset['test'].doc_id if final_eval_split == 'test' else NLP_Dataset['val'].doc_id

        test_metrics, test_loss = evaluate_split(
            test_model,
            final_eval_loader,
            fold,
            final_eval_split,
            save_dir=save_path,
            tokenizer=tokenizer,
            doc_ids=final_eval_doc_ids,
            save_predictions=True,
            prediction_basename="text_result",
        )

        (test_p_emotion, test_r_emotion, test_f_emotion,
         test_p_cause, test_r_cause, test_f_cause,
         test_p_pair, test_r_pair, test_f_pair) = test_metrics

        if test_loss is not None:
            print(f"Fold {fold} {final_eval_split} result (loss {test_loss:.4f}):")
        else:
            print(f"Fold {fold} {final_eval_split} result:")
        print(
            "e_p: {:.4f} e_r: {:.4f} e_f: {:.4f} c_p: {:.4f} c_r: {:.4f} c_f: {:.4f}"
            " pair_p: {:.4f} pair_r: {:.4f} pair_f: {:.4f}".format(
                test_p_emotion,
                test_r_emotion,
                test_f_emotion,
                test_p_cause,
                test_r_cause,
                test_f_cause,
                test_p_pair,
                test_r_pair,
                test_f_pair,
            ))

        
        # Self-training 結束，計算總時間
        self_training_total_time = time.time() - self_training_start_time
        print(f"\nSelf-training 完成！")
        print(f"Self-training 總執行時間: {self_training_total_time/60:.1f} 分鐘")
        
        # ===== 輸出樣本選中歷史記錄 =====
        selection_history_path = os.path.join(
            pseudo_results_dir,
            f"doc_selection_history_fold{fold}.txt"
        )
        with open(selection_history_path, 'w', encoding='utf-8') as f_hist:
            f_hist.write(f"樣本選中歷史記錄 - Fold {fold}\n")
            f_hist.write(f"Self-training 總輪數: {opt.self_training_rounds}\n")
            f_hist.write(f"總共 {len(doc_selection_history)} 個 doc_id 曾被選中\n")
            f_hist.write("=" * 60 + "\n\n")
            
            # 按被選中次數降序排序
            sorted_docs = sorted(doc_selection_history.items(), key=lambda x: len(x[1]), reverse=True)
            
            # 統計被選中 1 次、2 次、3 次... 的樣本數量
            count_distribution = {}
            for doc_id, rounds in doc_selection_history.items():
                count = len(rounds)
                count_distribution[count] = count_distribution.get(count, 0) + 1
            
            f_hist.write("選中次數分佈:\n")
            for count in sorted(count_distribution.keys(), reverse=True):
                f_hist.write(f"  被選中 {count} 次: {count_distribution[count]} 個樣本\n")
            f_hist.write("\n" + "=" * 60 + "\n\n")
            
            f_hist.write("詳細記錄 (按選中次數降序):\n")
            f_hist.write("-" * 60 + "\n")
            for doc_id, rounds in sorted_docs:
                rounds_str = ", ".join(map(str, rounds))
                f_hist.write(f"Doc {doc_id}: 被選中 {len(rounds)} 次, 輪次=[{rounds_str}]\n")
        
        print(f"  樣本選中歷史已儲存至: {selection_history_path}")
        
        # 當前 fold 結束，計算總時間
        fold_total_time = time.time() - fold_start_time
        print(f"\nFold {fold} 完成！")
        print(f"Fold {fold} 總執行時間: {fold_total_time/60:.1f} 分鐘")
        print(f"Fold {fold} 結束時間: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
        
        # 記錄當前fold的時間資訊
        fold_time_info = {
            'fold': fold,
            'total_time': fold_total_time,
            'supervised_time': supervised_training_total_time,
            'self_training_time': self_training_total_time,
            'is_test_only': False  # 標記為訓練模式
        }
        fold_times.append(fold_time_info)
        
        # 即時寫入當前fold的時間記錄
        with open(time_log_file, "a", encoding="utf-8") as f:
            f.write(f"Fold {fold} 時間記錄:\n")
            f.write(f"  開始時間: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(fold_start_time))}\n")
            f.write(f"  結束時間: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\n")
            f.write(f"  總執行時間: {fold_total_time/60:.1f} 分鐘\n")
            f.write(f"  初始監督訓練時間: {supervised_training_total_time/60:.1f} 分鐘\n")
            f.write(f"  Self-training 時間: {self_training_total_time/60:.1f} 分鐘\n")
            f.write("-" * 30 + "\n\n")
    
    # 所有 fold 結束，計算總時間
    all_folds_total_time = time.time() - all_folds_start_time
    overall_total_time = time.time() - overall_start_time
    print(f"\n所有實驗完成!")
    print(f"所有 fold 總執行時間: {all_folds_total_time/60:.1f} 分鐘")
    print(f"程式總執行時間: {overall_total_time/60:.1f} 分鐘")
    print(f"實驗結束時間: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
    
    # 寫入最終總結時間記錄
    with open(time_log_file, "a", encoding="utf-8") as f:
        f.write("=" * 50 + "\n")
        f.write("=== 總結時間統計 ===\n")
        f.write(f"實驗結束時間: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\n")
        f.write(f"所有 fold 總執行時間: {all_folds_total_time/60:.1f} 分鐘\n")
        f.write(f"程式總執行時間: {overall_total_time/60:.1f} 分鐘\n\n")
        
        # 檢查是否為test_only模式
        is_test_only_mode = any(fold_info.get('is_test_only', False) for fold_info in fold_times)
        
        if is_test_only_mode:
            # 測試模式
            f.write("=== 測試模式時間統計 ===\n")
            f.write("注意：此次執行為純測試模式 (test_only=True)，未進行任何訓練\n\n")
        else:
            # 訓練模式
            f.write("=== 訓練模式時間統計 ===\n")
        
        # 詳細統計每個fold
        f.write("=== 各 Fold 時間詳細統計 ===\n")
        total_supervised_time = 0
        total_self_training_time = 0
        
        for fold_info in fold_times:
            total_supervised_time += fold_info['supervised_time']
            total_self_training_time += fold_info['self_training_time']
            
            # 根據模式顯示不同的時間資訊
            if fold_info.get('is_test_only', False):
                f.write(f"Fold {fold_info['fold']}: {fold_info['total_time']/60:.1f}分 (純測試時間)\n")
            else:
                f.write(f"Fold {fold_info['fold']}: {fold_info['total_time']/60:.1f}分 "
                       f"(監督: {fold_info['supervised_time']/60:.1f}分, "
                       f"Self-training: {fold_info['self_training_time']/60:.1f}分)\n")
        
        f.write(f"\n平均每折時間: {all_folds_total_time/len(fold_times)/60:.1f} 分鐘\n" if len(fold_times) > 0 else "\n平均每折時間: 無資料\n")
        
        # 根據模式顯示不同的總計資訊
        if is_test_only_mode:
            f.write("總測試時間: {:.1f} 分鐘\n".format(sum(fold_info['total_time'] for fold_info in fold_times)/60))
        else:
            f.write(f"總監督訓練時間: {total_supervised_time/60:.1f} 分鐘\n")
            f.write(f"總 Self-training 時間: {total_self_training_time/60:.1f} 分鐘\n")
        
        f.write("=" * 50 + "\n")
    
    print(f"時間記錄已保存至: {time_log_file}")




def run():
    """
    程式執行入口點
    
    功能：
    1. 建立輸出目錄
    2. 設定 console 輸出同步寫入日誌檔
    3. 執行主程式 _run_main()
    """
    # 取得輸出路徑 (從命令列參數 --save_path)
    save_path = opt.save_path
    
    # 建立輸出資料夾 (若不存在則自動建立)
    os.makedirs(save_path, exist_ok=True)

    # 產生時間戳記
    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    
    # 設定 console 輸出日誌的檔名與完整路徑
    console_log_name = f'run_console_output_{timestamp}.txt'
    console_log_path = os.path.join(save_path, console_log_name)
    
    # 開啟日誌檔
    log_file = open(console_log_path, 'w', encoding='utf-8')
    
    # 保存原始的標準輸出(指向 console)
    original_stdout = sys.stdout
    
    # 把 stdout 改成 Tee (同時輸出到 console + 檔案)，之後所有 print() 都會同時寫入 console 和 log_file
    sys.stdout = Tee(sys.stdout, log_file)

    try:
        # 輸出執行資訊
        print(f"Console output is being copied to: {console_log_path}")
        print(f"Executing Script: {os.path.basename(__file__)}")
        print(f"Start Time: {timestamp}")
        
        # 執行主程式 (包含訓練、自訓練、測試等所有流程)
        _run_main(save_path)
    finally:
        # 恢復原始標準輸出，現在 print() 只會輸出到 console
        # sys.stdout 是 Python 的標準輸出，預設是console
        sys.stdout = original_stdout
        
        # 關閉日誌檔
        log_file.close()


if __name__ == '__main__':
    # 程式進入點：當此腳本被直接執行時（而非被 import），執行 run()
    run()