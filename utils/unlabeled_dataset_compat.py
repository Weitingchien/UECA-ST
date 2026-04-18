#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""與 ST_nest_hybrid 系列一致的未標註資料集載入模組"""

from __future__ import annotations  # 啟用新版型別註記語法

import json  # 用來讀取 JSON 資料檔
from pathlib import Path  # 用 Path 物件管理路徑
from typing import List  # 提供型別註記用的 List

import numpy as np  # 用來儲存 token id 陣列
from torch.utils.data import Dataset  # 與訓練程式一致，繼承 PyTorch Dataset


def build_unlabeled_file_path(dataset_dir: str | Path, fold: int) -> Path:
    """建立與原訓練腳本一致的未標註檔案路徑。"""
    dataset_dir = Path(dataset_dir)  # 把字串路徑轉為 Path，方便後續拼接
    return dataset_dir / f"fold{fold}_unlabeled.json"  # 回傳 fold 對應的 unlabeled JSON 路徑


class UnlabeledDatasetCompat(Dataset):
    """與 UECA_CE_few_shot_ST_nest_hybrid*.py 相同邏輯的 UnlabeledDataset。"""

    def __init__(self, input_file: str | Path, tokenizer) -> None:
        print('load unlabeled data_file: {}'.format(input_file))
        self.input_file = Path(input_file)  # 儲存來源檔案路徑
        self.tokenizer = tokenizer  # 儲存 tokenizer 供 encode_plus 使用
        self.x_bert: List[np.ndarray] = []  # 儲存每筆未標註樣本的 token id
        self.doc_id: List[str] = []  # 儲存每筆樣本對應的 doc_id
        self.raw_text: List[str] = []  # 儲存原始句子文本，便於除錯與追蹤
        self.n_cut = 0  # 記錄超過 512 長度而被移除的樣本數

        self._load()  # 立即載入資料，保持與原始腳本使用方式一致

    def _load(self) -> None:
        with self.input_file.open("r", encoding="utf8") as f:  # 開啟未標註 JSON 檔
            data = json.load(f)  # 讀取整個 JSON 內容

        for doc in data:  # 逐篇文檔處理
            doc_id = doc["doc_id"]  # 取得文檔編號
            d_len = doc["doc_len"]  # 取得文檔子句數量
            part_sentence = [clause["clause"] for clause in doc["clauses"]]  # 取出每個子句文字
            mask_full_document = ""  # 建立與訓練程式同格式的 masked prompt 字串
            raw_text_doc = ""  # 建立人類可讀的原始文本字串

            for i in range(1, d_len + 1):  # 子句索引從 1 開始，與原腳本一致
                mask_full_document = mask_full_document + ' ' + str(i) + ' ' + part_sentence[i - 1]  # 拼接「子句編號 + 子句內容」
                mask_full_document = mask_full_document + "[MASK] [MASK] [MASK] [SEP]"  # 每個子句後加 3 個 [MASK] 與 [SEP]
                raw_text_doc = raw_text_doc + f"[{i}] {part_sentence[i - 1]} "  # 同步組出原始文本，供除錯使用

            count_len = len(self.tokenizer.encode_plus(mask_full_document, return_tensors="pt")['input_ids'][0])  # 先用原邏輯檢查長度
            if count_len > 512:  # 若超過 BERT 限制，與原腳本一致直接跳過
                print("UnlabeledDataset: Over limit length{} document{} - 移除此文檔".format(count_len, doc_id))
                self.n_cut += 1  # 累加被移除樣本數
                continue  # 不納入資料集

            self.doc_id.append(doc_id)  # 記錄通過長度檢查的 doc_id
            self.raw_text.append(raw_text_doc.strip())  # 儲存原始文本 (去尾端空白)
            mask_full_document = self.tokenizer.encode_plus(mask_full_document, return_tensors="pt", max_length=512, truncation=True, pad_to_max_length=True)['input_ids']
            self.x_bert.append(np.array(mask_full_document[0]))
            
        print(f'UnlabeledDataset: n_cut {self.n_cut}, total_documents {len(self.x_bert)}')
        print('load unlabeled data done!\n')
        
    def __getitem__(self, index: int) -> np.ndarray:
        return self.x_bert[index]  # 回傳單筆 input_ids (與原碼一致僅回傳 x_bert)

    def __len__(self) -> int:
        return len(self.x_bert)  # 回傳資料筆數

    def remove_by_doc_ids(self, doc_ids: List[str]) -> None:
        id_set = set(doc_ids)
        keep_indices = [idx for idx, doc_id in enumerate(self.doc_id) if doc_id not in id_set]  # 保留不在刪除清單的索引
        if len(keep_indices) == len(self.doc_id):
            return
        before = len(self.doc_id)
        self.x_bert = [self.x_bert[idx] for idx in keep_indices]
        self.doc_id = [self.doc_id[idx] for idx in keep_indices]
        self.raw_text = [self.raw_text[idx] for idx in keep_indices]  # 同步移除 raw_text
        after = len(self.doc_id)
        print(f"  自未標註資料集中移除 {before - after} 筆，剩餘 {after}")
