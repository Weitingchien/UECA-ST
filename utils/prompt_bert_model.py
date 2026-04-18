#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
獨立的 prompt_bert 模型定義模組。

目的: torch.load 以 pickle 還原整個模型物件時
     需要在執行環境中找到 prompt_bert 這個 class
     原訓練腳本含有 argparse 模組層級執行，直接 import 會衝突
     因此把 prompt_bert 及其依賴的輔助函式抽出放在這裡
     讓比較腳本可以安全 import 而不會觸發 argparse 的執行
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from transformers import BertForMaskedLM, BertTokenizer


# ──────────────────────────────────────────
# 輔助函式 (與原訓練腳本邏輯完全一致)
# ──────────────────────────────────────────

def get_binary_token_ids(tokenizer):
    """動態查詢「是」「非」兩個關鍵標籤的 token id，避免硬編數值"""
    return (
        tokenizer.convert_tokens_to_ids('是'),
        tokenizer.convert_tokens_to_ids('非')
    )



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



# ──────────────────────────────────────────
# 主模型類別 (與原訓練腳本完全一致)
# ──────────────────────────────────────────

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
