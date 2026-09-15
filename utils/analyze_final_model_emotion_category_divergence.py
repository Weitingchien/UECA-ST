#!/usr/bin/env python3
# 上面的 shebang 讓 Linux／WSL 可在直接執行本檔時，使用環境中的 python3
# -*- coding: utf-8 -*-
# 上面的編碼宣告表示原始碼採 UTF-8，因此繁體中文註解可以被正確讀取
"""以每折 self-training 最佳模型分析情緒類別與 NeST 散度的關係。"""
# 三引號包住的是「模組說明字串」，用來簡述整支程式的用途

# argparse 用來定義與解析命令列參數，例如 --k 3、--start_fold 1
import argparse
# csv 用來輸出可由 Excel 或試算表軟體開啟的 CSV 檔案
import csv
# json 用來讀取資料集 JSON，也用來輸出整體分析摘要
import json
# os 用來判斷目前是在 Windows 還是 Linux／WSL
import os
# sys 用來調整 Python 尋找模組時使用的搜尋路徑
import sys
# Path 是 pathlib 提供的跨平台路徑物件，可用 / 運算子拼接路徑
from pathlib import Path

# faiss 負責建立精確的 L2 KNN 索引並搜尋最近鄰
import faiss
# NumPy 負責矩陣運算、KL 散度、平均值與分位數等統計
import numpy as np
# PyTorch 負責載入 checkpoint、執行 BERT 推論及 Tensor 運算
import torch
# Dataset 定義資料集介面；DataLoader 會把多筆樣本組成 batch
from torch.utils.data import DataLoader, Dataset
# BertTokenizer 把 prompt 文字轉成 bert-base-chinese 的 token id
from transformers import BertTokenizer


# __file__ 是目前程式檔位置；resolve() 轉為絕對路徑
# 第一次 parent 是 utils，第二次 parent 回到專案根目錄 UECA_ST
REPO_ROOT = Path(__file__).resolve().parent.parent
# 把專案根目錄插入模組搜尋路徑最前面，確保直接執行 utils/*.py 時仍能 import utils
sys.path.insert(0, str(REPO_ROOT))

# checkpoint 是由訓練主程式以 __main__.prompt_bert 儲存的完整模型物件
# get_clause_boundaries 用來找每個子句內容範圍及三個 [MASK] 的位置
# prompt_bert 雖未被一般程式碼直接呼叫，仍必須存在於 __main__，
# 否則 torch.load 無法還原舊 checkpoint 中記錄的 __main__.prompt_bert 類別
# noqa 只關閉程式碼檢查器的 import 位置／未使用警告，不影響實際執行
from utils.prompt_bert_model import get_clause_boundaries, prompt_bert  # noqa: E402,F401


# 保存指定實驗的完整資料夾名稱；括號內相鄰字串會由 Python 自動接在一起
EXPERIMENT_NAME = (
    "prompt_ECPE_few_shot_ST_2026_05_29_16_49_46_f1-10_i70_lr1e-5_bs8_wd0.01_nest_k5_emotion_clause_knnemotion_clause_nestmul1_nbeta0.1_nm0.6_gamma0.5_nlmnest_avgs_simple_st5_ste20_seed42_remove_pseudo_v2_CE"
)
# 依作業系統決定預設實驗路徑：Windows 使用 H:\，WSL 使用 /mnt/h/
# 這裡的「條件運算式」語法是：條件成立時的值 if 條件 else 條件不成立時的值
DEFAULT_EXPERIMENT_DIR = (
    # Path(r"...") 前面的 r 代表 raw string，Windows 反斜線不會被當成跳脫字元
    # Path / EXPERIMENT_NAME 中的 / 是 pathlib 的路徑拼接，不是數值除法
    Path(r"H:\ep_split10_t1v1te1_u7_aligned_disjoint_2019") / EXPERIMENT_NAME
    # os.name == "nt" 表示目前是 Windows
    if os.name == "nt"
    # 非 Windows（例如 WSL）則使用自動掛載的 /mnt/h 路徑
    else Path("/mnt/h/ep_split10_t1v1te1_u7_aligned_disjoint_2019") / EXPERIMENT_NAME
)
# 每組分別移除 raw divergence 最高的 5%，用於平均數敏感度分析
UPPER_TAIL_FRACTION = 0.05


# 繼承 PyTorch Dataset，讓 DataLoader 可以用 len(dataset) 與 dataset[index] 取資料
class PromptInferenceDataset(Dataset):
    """建立與原訓練程式相同的 masked prompt，並保留原始文檔資料。"""

    # __init__ 是建立資料集物件時自動執行的初始化函式
    # self 代表目前這個資料集物件；json_path 是資料檔，tokenizer 是 BERT tokenizer
    def __init__(self, json_path, tokenizer):
        # with 區塊結束後會自動關閉檔案；encoding 指定以 UTF-8 讀取中文
        with Path(json_path).open("r", encoding="utf-8") as file:
            # json.load 會把最外層 JSON array 轉成 Python list
            raw_documents = json.load(file)

        # documents 保存通過長度篩選的原始文檔字典，供之後查 GT pairs/category
        self.documents = []
        # input_ids 保存與 documents 完全同順序的固定 512-token Tensor
        self.input_ids = []
        # len(...) 取得長度；total_raw 是尚未排除過長文檔前的總篇數
        self.total_raw = len(raw_documents)

        # 逐篇處理 JSON 中的原始文檔
        for document in raw_documents:
            # "".join(...) 將每個子句的 prompt 片段串成一篇完整 masked prompt
            prompt = "".join(
                # f-string 會把 index 與 clause['clause'] 的值插入字串
                # 每個子句依序放入 emotion、cause、pair 三個 [MASK]，再以 [SEP] 分隔
                f" {index} {clause['clause']}[MASK] [MASK] [MASK] [SEP]"
                # enumerate(..., start=1) 同時取得 1-based 子句編號與子句字典
                # 這是一個 generator expression，會逐項交給 join，而不先建立額外 list
                for index, clause in enumerate(document["clauses"], start=1)
            )
            # encode_plus 把整篇 prompt 轉成 token id
            # return_tensors="pt" 回傳 PyTorch Tensor；["input_ids"][0] 移除大小為 1 的 batch 維
            unpadded = tokenizer.encode_plus(prompt, return_tensors="pt")["input_ids"][0]
            # numel() 是 Tensor 的元素數；BERT 最長只能接受 512 個 token
            if unpadded.numel() > 512:
                # continue 會跳過本篇文檔，直接進入下一次 for 迴圈
                continue

            # 對長度合格的文檔再編碼一次，產生固定長度的模型輸入
            encoded = tokenizer.encode_plus(
                # 第一個位置參數是要編碼的完整 prompt 字串
                prompt,
                # 要求輸出 PyTorch Tensor，而不是 Python list
                return_tensors="pt",
                # 每篇輸入最多保留 512 token
                max_length=512,
                # 前面已排除超長文檔；這裡保留 truncation 以符合原程式設定
                truncation=True,
                # 不足 512 token 時補 [PAD]，讓 DataLoader 能疊成同尺寸 batch
                padding="max_length",
            # encode_plus 回傳 dict；取 input_ids 的第 0 列後，形狀為 (512,)
            )["input_ids"][0]
            # 只有成功建立 input_ids 後才同步加入 document，避免兩個 list 索引錯位
            self.documents.append(document)
            # 保存與上面 document 相同索引的模型輸入 Tensor
            self.input_ids.append(encoded)

        # 原始文檔數減去保留數，就是 tokenized length > 512 而移除的篇數
        self.over_limit = self.total_raw - len(self.documents)

    # __len__ 讓 Python 的 len(dataset) 回傳實際可推論樣本數
    def __len__(self):
        return len(self.input_ids)

    # __getitem__ 讓 dataset[index] 回傳指定文檔的 input_ids
    def __getitem__(self, index):
        # DataLoader 會自動把多個 (512,) Tensor 疊成 (batch_size, 512)
        return self.input_ids[index]


# 將所有可由命令列調整的分析參數集中定義
def parse_args():
    # ArgumentParser 會產生 --help 說明並負責解析參數
    parser = argparse.ArgumentParser(
        description="使用 final self-training best model 比較 A1/A2 的 emotion-clause NeST 散度"
    )
    # type=Path 表示讀到的路徑字串會自動轉成 Path；未提供時使用指定 H 槽實驗
    parser.add_argument("--experiment_dir", type=Path, default=DEFAULT_EXPERIMENT_DIR)
    # 設定 10-fold train/unlabeled JSON 所在的資料夾
    parser.add_argument(
        "--dataset_dir",
        type=Path,
        default=REPO_ROOT / "split10_train1_val1_test1_unlabeled7_aligned_disjoint_2019",
    )
    # 設定與訓練時一致的 bert-base-chinese tokenizer 路徑
    parser.add_argument("--bert_path", type=Path, default=REPO_ROOT / "bert-base-chinese")
    # 設定四份分析結果要寫入的資料夾
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=REPO_ROOT / "analysis" / "final_model_emotion_category_divergence",
    )
    # 起始與結束 fold；main 中會讓 end_fold 也包含在執行範圍內
    parser.add_argument("--start_fold", type=int, default=1)
    parser.add_argument("--end_fold", type=int, default=10)
    # k 是每篇未標註文檔要尋找的 labeled KNN 數量
    parser.add_argument("--k", type=int, default=3)
    # beta 是 NeST raw divergence 中 D_l 的權重
    parser.add_argument("--beta", type=float, default=0.1)
    # batch_size 決定一次送進 BERT 推論的文檔數
    parser.add_argument("--batch_size", type=int, default=8)
    # device 指定使用 cuda 或 cpu；預設使用 GPU
    parser.add_argument("--device", default="cuda")
    # parse_args() 真正讀取使用者執行指令中的命令列參數並回傳 Namespace
    return parser.parse_args()


# 對一個 Dataset 進行推論，回傳三項彼此索引對齊的結果：
# 1. 每篇文檔的特徵向量；2. 每篇文檔的二元預測分布；3. 預測情緒子句編號
# model 是已載入的最佳模型；dataset 是 PromptInferenceDataset；
# tokenizer 用來取得「是／非」與特殊 token 的 id；batch_size 是推論批次大小；
# device 是 torch.device，例如 cuda 或 cpu
def infer_documents(model, dataset, tokenizer, batch_size, device):
    """取得每篇文檔的 emotion-clause embedding、分布及預測子句索引。"""
    # DataLoader 依 batch_size 將 Dataset 樣本疊成 batch
    # shuffle=False 不打亂順序，才能讓輸出索引繼續對應 dataset.documents
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    # 將中文字 token「是」轉成 BERT 詞彙表中的整數 id
    # 這裡的 yes_id／no_id 是詞彙表位置，不是一般分類器的類別 0／1
    yes_id = tokenizer.convert_tokens_to_ids("是")
    # 取得「非」在 BERT 詞彙表中的整數 id
    no_id = tokenizer.convert_tokens_to_ids("非")
    # 同時建立三個空 list；之後會依相同文檔順序分別加入特徵、分布與預測
    features, distributions, predicted_clauses = [], [], []

    # 切換到評估模式，關閉 dropout 等只應在訓練時啟用的行為
    model.eval()
    # inference_mode() 關閉梯度追蹤，降低純推論時的記憶體與運算成本
    with torch.inference_mode():
        # DataLoader 每次產生一批仍位於 CPU 的 input_ids
        for cpu_input_ids in loader:
            # 將這批 token id 移到指定的 CPU 或 GPU，供模型執行 forward
            input_ids = cpu_input_ids.to(device)
            # model.bert 是 BertForMaskedLM，第二個 .bert 是其中的 BertModel 主體
            # return_dict=True 讓輸出可用欄位名稱存取；last_hidden_state 的形狀通常是
            # (目前 batch 大小, 512, hidden_size)，每個 token 都有一個 hidden vector
            # 這裡刻意維持原訓練程式的呼叫方式，沒有額外傳入 attention_mask
            hidden_states = model.bert.bert(input_ids, return_dict=True).last_hidden_state

            # input_ids.size(0) 是這個 batch 實際包含的文檔數；最後一批可能小於 batch_size
            for row in range(input_ids.size(0)):
                # 解析目前文檔每個子句的 token 範圍，以及 emotion／cause／pair 三個
                # [MASK] 的位置；邊界解析留在 CPU Tensor 上，與原工具函式介面一致
                boundaries = get_clause_boundaries(
                    # [row] 取出 batch 中第 row 篇文檔，形狀由 (B, 512) 變成 (512,)
                    cpu_input_ids[row],
                    # tokenizer.mask_token_id 是 [MASK] 的整數 token id
                    tokenizer.mask_token_id,
                    # tokenizer.sep_token_id 是 [SEP] 的整數 token id
                    tokenizer.sep_token_id,
                )
                # boundaries 每項是 (內容起點, 內容終點, 三個 MASK 位置)
                # `for _, _, masks` 用底線忽略此處不需要的前兩項，masks[0] 是 emotion MASK
                emotion_positions = [masks[0] for _, _, masks in boundaries]
                # 用進階索引一次取出所有 emotion MASK 的 hidden vector；
                # 形狀是 (子句數, hidden_size)
                emotion_hidden = hidden_states[row, emotion_positions]
                # model.bert.cls 是 masked-language-model prediction head，不是 [CLS] 向量
                # 先得到每個 emotion MASK 對完整詞彙表的 logits，再只保留「是／非」兩欄
                binary_logits = model.bert.cls(emotion_hidden)[:, [yes_id, no_id]]
                # dim=-1 表示在最後一維的「是／非」兩個 logits 上做 softmax，
                # 因此每個子句會得到 [P(是), P(非)]，而且兩者總和為 1
                binary_probs = torch.softmax(binary_logits, dim=-1)
                # 逐子句比較 P(是) 是否嚴格大於 P(非)，結果是布林 Tensor
                selected = binary_probs[:, 0] > binary_probs[:, 1]
                # 將被判為情緒子句的位置轉成 JSON 使用的 1-based 子句編號
                selected_indices = (
                    # nonzero 找出 True 的 0-based 位置；view(-1) 攤平成一維；
                    # add(1) 改成 1-based；cpu().tolist() 轉成一般 Python list
                    torch.nonzero(selected, as_tuple=False).view(-1).add(1).cpu().tolist()
                )

                # Python 中非空 list 會被視為 True，表示模型至少預測到一個情緒子句
                if selected_indices:
                    # 為每個「預測為情緒」的子句建立一個 mean-pooled embedding
                    clause_embeddings = [
                        # [start:end] 取該子句內容 token，mean(dim=0) 沿 token 維取平均
                        hidden_states[row, start:end].mean(dim=0)
                        # enumerate 產生 0-based clause_index；每項 boundary 再拆成三部分
                        for clause_index, (start, end, _) in enumerate(boundaries)
                        # selected_indices 是 1-based，所以 clause_index 必須加 1 再比較
                        if clause_index + 1 in selected_indices
                    ]
                    # stack 把多個子句向量疊起來，再平均成一個文檔特徵向量
                    document_feature = torch.stack(clause_embeddings).mean(dim=0)
                    # 只平均被預測為情緒子句的 [P(是), P(非)]，作為文檔預測分布
                    document_distribution = binary_probs[selected].mean(dim=0)
                # 若沒有任何子句被預測為情緒子句，就使用既有的 fallback 規則
                else:
                    # hidden_states[row, 0] 是該文檔第一個 token（通常為 [CLS]）的向量
                    document_feature = hidden_states[row, 0]
                    # 此時沒有 selected 子句可平均，因此改為平均全部子句的二元分布
                    document_distribution = binary_probs.mean(dim=0)

                # detach 在 inference_mode 下不必另外呼叫；先搬回 CPU，再轉成
                # FAISS 所需的 NumPy float32，並依文檔順序加入 features
                features.append(document_feature.cpu().numpy().astype(np.float32))
                # 保存模型推論分布；這不是 ground-truth one-hot 標籤
                distributions.append(document_distribution.cpu().numpy().astype(np.float32))
                # tuple(...) 將 list 轉成不可變 tuple，之後可與 (GT_emotion_clause,) 精確比較
                predicted_clauses.append(tuple(selected_indices))

    # 一次回傳特徵矩陣、分布矩陣與預測 tuple list
    return (
        # stack 將 list 疊成 (文檔數, hidden_size)；ascontiguousarray 確保
        # 記憶體連續且為 float32，以符合 FAISS 的輸入要求
        np.ascontiguousarray(np.stack(features), dtype=np.float32),
        # 分布矩陣的形狀為 (文檔數, 2)，也轉成連續的 float32 陣列
        np.ascontiguousarray(np.stack(distributions), dtype=np.float32),
        # 每個元素是一篇文檔所預測到的 1-based 情緒子句編號 tuple
        predicted_clauses,
    )


# 以 labeled train 文檔作為 KNN anchor，對每篇 unlabeled query 尋找 k 個最近鄰，
# 再用模型對 anchor/query 產生的二元分布計算 NeST 的 D_u、D_l 與 raw divergence
# train_features 形狀為 (labeled 數, hidden_size)，query_features 為
# (unlabeled 數, hidden_size)；兩個 distributions 的最後一維都是 2
def compute_knn_and_divergence(train_features, train_distributions, query_features, query_distributions, k, beta):
    """依照 NeST 原公式計算 FAISS KNN、D_u、D_l 與 raw divergence"""
    # train_features.shape[1] 是 embedding 維度
    # IndexFlatL2 會做精確暴力搜尋，回傳「平方 L2 距離」，不是開根號後的 L2
    # 此處沒有對 labeled/query features 額外做中心化或正規化
    index = faiss.IndexFlatL2(train_features.shape[1])
    # 把所有 labeled train 特徵加入索引，作為可被搜尋的 KNN anchor
    index.add(train_features)
    # 對每篇 query 搜尋 k 個最近鄰；兩個回傳矩陣形狀皆為 (query 數, k)
    # distances 保存平方 L2；neighbor_indices 保存對應的 train_dataset 索引
    distances, neighbor_indices = index.search(query_features, k)

    # 避免後續 log(0) 或除以 0；這是數值穩定用的極小常數
    epsilon = 1e-10
    # NumPy 進階索引：以 (query 數, k) 的鄰居索引，取出每位鄰居的二元分布
    # 結果形狀為 (query 數, k, 2)
    neighbor_distributions = train_distributions[neighbor_indices]
    # 計算每個 query 與每個鄰居的 KL 項，方向是
    # KL(neighbor_distribution || query_distribution)
    score_u = (
        # np.log 對陣列逐元素取自然對數
        np.log(
            # 分子形狀為 (query 數, k, 2)
            (epsilon + neighbor_distributions)
            # np.newaxis 在 query 與類別維之間加入一維，
            # 將 (query 數, 2) 變成 (query 數, 1, 2)，再廣播到所有 k 個鄰居
            / (epsilon + query_distributions[:, np.newaxis])
        )
        # KL(p || q) 的逐類別形式是 p * log(p / q)
        * (epsilon + neighbor_distributions)
    )
    # axis=(-1, -2) 同時加總最後的「類別維」與倒數第二個「鄰居維」，
    # 得到每篇 query 一個 D_u 值
    d_u = score_u.sum(axis=(-1, -2))

    # 沿 k 個鄰居的維度取平均，得到每篇 query 的鄰居平均分布 y_bar；
    # 結果形狀為 (query 數, 2)。
    neighbor_mean = neighbor_distributions.mean(axis=1)
    # 計算每個鄰居相對於鄰居平均分布的 KL 項，方向是
    # KL(neighbor_mean || neighbor_distribution)
    score_l = (
        np.log(
            # [:, np.newaxis] 將平均分布由 (query 數, 2) 變成
            # (query 數, 1, 2)，讓 NumPy 對 k 個鄰居做 broadcasting
            (epsilon + neighbor_mean[:, np.newaxis])
            / (epsilon + neighbor_distributions)
        )
        # 依照既有 NeST 公式，以 neighbor_mean 作為 KL 的 p 分布
        * neighbor_mean[:, np.newaxis]
    )
    # 同樣加總 k 個鄰居與兩個類別，得到每篇 query 一個 D_l 值
    d_l = score_l.sum(axis=(-1, -2))

    # 用 dict 同時回傳後續篩選、輸出與統計需要的所有陣列
    return {
        # 每篇 query 的 k 個 labeled train 索引，形狀為 (query 數, k)
        "neighbor_indices": neighbor_indices,
        # 與上述索引對齊的平方 L2 距離，形狀為 (query 數, k)
        "neighbor_distances": distances,
        # 只沿最後的二元類別維加總，因此保留每一位鄰居對 D_u 的 KL 貢獻
        "neighbor_kl": score_u.sum(axis=-1),
        # 每篇 query 的 D_u
        "d_u": d_u,
        # 每篇 query 的 D_l
        "d_l": d_l,
        # 本分析使用尚未套用跨輪 EMA 的原始散度：D_u + beta × D_l
        "raw_divergence": d_u + beta * d_l,
    }


# 將複合情緒類別轉成順序一致的字串，例如 surprise&joy 轉成 joy&surprise
def normalize_category(category):
    return "&".join(
        sorted(part.strip() for part in category.strip().lower().split("&"))
    )


# 將一篇文檔所有 GT 情緒子句與類別整理成「子句編號:類別」格式
def format_gt_emotion_categories(document):
    emotion_clause_ids = {int(pair[0]) for pair in document.get("pairs", [])}
    return "|".join(
        f"{int(clause['clause_id'])}:{normalize_category(clause['emotion_category'])}"
        for clause in document["clauses"]
        if int(clause["clause_id"]) in emotion_clause_ids
    )


# 從一篇原始 JSON 文檔中取出老師分析所需的 single-pair GT 資訊
def single_pair_info(document):
    """回傳唯一 GT 情緒子句與其 emotion_category；非 single-pair 回傳 None"""
    # dict.get("pairs", []) 表示：有 pairs 就取其值，沒有則使用空 list
    pairs = document.get("pairs", [])
    # 只有恰好一組 emotion-cause pair 的文檔符合本分析條件
    if len(pairs) != 1:
        # None 表示這篇文檔不屬於 single-pair 分析範圍
        return None

    # 每組 pair 的第 0 個值是 GT 情緒子句編號；int 統一轉為整數
    emotion_clause = int(pairs[0][0])
    # next(...) 從 generator 產生的候選中，取第一個 clause_id 符合 GT 情緒子句者
    category = next(
        # 這是資料集提供的 GT metadata，不是模型預測出的情緒類別
        clause["emotion_category"]
        # 逐一檢查 document["clauses"] 中的子句
        for clause in document["clauses"]
        # 只保留 clause_id 與 emotion_clause 相同的那一項
        if int(clause["clause_id"]) == emotion_clause
    )
    # 回傳二元素 tuple：(GT 情緒子句編號, 標準化後的情緒類別字串)
    return emotion_clause, normalize_category(category)


def build_unlabeled_divergence_rows(fold, train_dataset, query_dataset, divergence):
    """建立每篇可推論 unlabeled query、k 位鄰居類別及散度的明細。"""
    rows = []
    for query_index, query_document in enumerate(query_dataset.documents):
        row = {
            "fold": fold,
            "query_doc_id": str(query_document["doc_id"]),
            "query_gt_emotion_clause_categories": format_gt_emotion_categories(
                query_document
            ),
        }
        for rank, neighbor_index in enumerate(
            divergence["neighbor_indices"][query_index], start=1
        ):
            neighbor_document = train_dataset.documents[int(neighbor_index)]
            row[f"neighbor_{rank}_doc_id"] = str(neighbor_document["doc_id"])
            row[f"neighbor_{rank}_gt_emotion_clause_categories"] = (
                format_gt_emotion_categories(neighbor_document)
            )
        row["D_u"] = float(divergence["d_u"][query_index])
        row["D_l"] = float(divergence["d_l"][query_index])
        row["raw_divergence"] = float(divergence["raw_divergence"][query_index])
        rows.append(row)
    return rows


# 依序套用 single-pair、預測完全正確與有效 category 等條件，
# 最後把符合條件的 query 分成 A1_same 或 A2_different
def build_eligible_rows(
    # 目前處理的 fold 編號
    fold,
    # labeled train Dataset，KNN 鄰居索引會指向其中的 documents
    train_dataset,
    # original unlabeled Dataset，是本分析的 query 集合
    query_dataset,
    # 每篇 labeled train 文檔預測到的情緒子句 tuple
    train_predictions,
    # 每篇 unlabeled query 文檔預測到的情緒子句 tuple
    query_predictions,
    # compute_knn_and_divergence 回傳的索引、距離與散度 dict
    divergence,
):
    """套用 exact-correct 與 single-pair gate，建立 A1/A2 明細"""
    # rows 保存通過所有 gate 的逐篇分析明細
    rows = []
    # counts 記錄漏斗式篩選每一階段剩下多少文檔
    # 後面的數值是累積通過該階段的數量，不是互斥的失敗原因數量
    counts = {
        # 目前 fold 編號
        "fold": fold,
        # 原始 unlabeled JSON 在任何長度篩選前的總篇數
        "total_raw_unlabeled": query_dataset.total_raw,
        # prompt token 數超過 512，因此未送進模型的文檔數
        "over_512": query_dataset.over_limit,
        # 通過 512-token 條件、實際可推論的 query 數
        "usable_candidates": len(query_dataset),
        # query 本身恰好只有一組 GT pair 的數量
        "query_single_pair": 0,
        # query 的預測情緒子句集合與 single-pair GT 完全相等的數量
        "query_exact_correct": 0,
        # query 的所有 k 個鄰居也全部是 single-pair 文檔的數量
        "neighbors_all_single_pair": 0,
        # query 及所有 k 個鄰居皆 exact-correct 的數量
        "all_documents_exact_correct": 0,
        # query 與所有鄰居 category 都不是 "null" 的數量
        "valid_category": 0,
        # 有效樣本中，query 與所有鄰居 category 都相同的數量
        "A1_same": 0,
        # 有效樣本中，至少一位鄰居 category 與 query 不同的數量
        "A2_different": 0,
    }

    # enumerate 同時產生 0-based query_index 與該索引的原始文檔
    # query_index 也用來取 query_predictions 和 divergence 中的同一篇結果
    for query_index, query_document in enumerate(query_dataset.documents):
        # Gate 1：讀取 query 的 single-pair GT 資訊
        query_info = single_pair_info(query_document)
        # 非 single-pair 時，continue 直接跳到下一篇 query
        if query_info is None:
            continue
        # `+= 1` 表示把目前計數加一
        counts["query_single_pair"] += 1

        # tuple unpacking：把 (情緒子句編號, category) 分別指定給兩個變數
        query_emotion, query_category = query_info
        # Gate 2：要求整個預測 tuple 恰好等於唯一 GT `(query_emotion,)`
        # 因此漏掉 GT，或雖命中 GT 但多預測其他子句，都不算 exact-correct
        if query_predictions[query_index] != (query_emotion,):
            continue
        counts["query_exact_correct"] += 1

        # 取出目前 query 在 labeled train 中找到的 k 個最近鄰索引
        neighbor_indices = divergence["neighbor_indices"][query_index]
        # list comprehension 依 KNN 順序，把索引轉回原始 labeled train 文檔
        neighbor_documents = [train_dataset.documents[index] for index in neighbor_indices]
        # 對每位鄰居取得 single-pair GT；非 single-pair 者會得到 None
        neighbor_info = [single_pair_info(document) for document in neighbor_documents]
        # Gate 3：any(...) 只要發現一個 None 就成立，因此整篇 query 被排除
        if any(info is None for info in neighbor_info):
            continue
        counts["neighbors_all_single_pair"] += 1

        # Gate 4：all(...) 要求 k 個鄰居的模型預測都與各自 single-pair GT 完全相等
        neighbors_exact = all(
            # info[0] 是該鄰居的 GT 情緒子句；單元素 tuple 寫成 `(值,)`
            train_predictions[index] == (info[0],)
            # zip 將每個 train index 與同順序的 neighbor_info 配成一組
            for index, info in zip(neighbor_indices, neighbor_info)
        )
        # 只要一位鄰居不 exact-correct，就排除目前 query
        if not neighbors_exact:
            continue
        counts["all_documents_exact_correct"] += 1

        # 取出 k 個鄰居的標準化 GT emotion_category
        categories = [info[1] for info in neighbor_info]
        # Gate 5：query 或任一鄰居的類別是 "null" 時，不進行 A1/A2 比較
        if query_category == "null" or any(category == "null" for category in categories):
            continue
        counts["valid_category"] += 1

        # 布林值在 sum 中可當成 1／0，因此此值等於 category 不同的鄰居數，範圍 0～k
        mismatch_count = sum(category != query_category for category in categories)
        # 條件運算式：0 位不同歸 A1；至少 1 位不同歸 A2
        # A1 要求「query 與所有 k 位鄰居」都同類，不只是鄰居彼此相同
        group = "A1_same" if mismatch_count == 0 else "A2_different"
        # group 的字串恰好也是 counts 的 key，因此可直接累加對應組別
        counts[group] += 1

        # 取出目前 query 的 k 個平方 L2 距離
        distances = divergence["neighbor_distances"][query_index]
        # 取出每位鄰居對 D_u 的個別 KL 貢獻
        neighbor_kl = divergence["neighbor_kl"][query_index]
        # 建立一個 dict，保存這篇 query 的完整可稽核明細，再加入 rows
        rows.append(
            {
                # fold 與 query doc_id 用來追溯原始資料
                "fold": fold,
                # str(...) 統一輸出成文字，避免不同 JSON 的 doc_id 型別不一致
                "query_doc_id": str(query_document["doc_id"]),
                # A1_same 或 A2_different
                "group": group,
                # k 位鄰居中，有幾位 category 與 query 不同
                "mismatch_count": mismatch_count,
                # query 的標準化 GT emotion_category
                "query_category": query_category,
                # query 的唯一 GT 情緒子句編號
                "query_emotion_clause": query_emotion,
                # "|" 用來分隔多個預測子句；通過 gate 後理論上只會有一個
                "query_predicted_emotion_clauses": "|".join(
                    # join 只能連接字串，所以先逐一以 str 轉換整數
                    str(value) for value in query_predictions[query_index]
                ),
                # 依 KNN 排名，以 "|" 串起所有鄰居 doc_id
                "neighbor_doc_ids": "|".join(str(document["doc_id"]) for document in neighbor_documents),
                # 依 KNN 排名，以 "|" 串起所有鄰居的標準化 GT category
                "neighbor_categories": "|".join(categories),
                # 依 KNN 排名，以 "|" 串起所有鄰居的 GT 情緒子句
                "neighbor_emotion_clauses": "|".join(str(info[0]) for info in neighbor_info),
                # 每位鄰居內若有多個預測子句以 "," 分隔，鄰居彼此再用 "|" 分隔
                "neighbor_predicted_emotion_clauses": "|".join(
                    ",".join(str(value) for value in train_predictions[index])
                    for index in neighbor_indices
                ),
                # f"{value:.8f}" 將每個平方 L2 距離格式化為小數點後 8 位
                "neighbor_l2_squared": "|".join(f"{value:.8f}" for value in distances),
                # 同樣輸出每位鄰居對 D_u 的 KL 值，方便逐鄰居檢查
                "neighbor_kl": "|".join(f"{value:.8f}" for value in neighbor_kl),
                # distances.mean() 計算這篇 query 對 k 位鄰居的平均平方 L2；
                # float(...) 將 NumPy scalar 轉成可直接寫入 JSON/CSV 的 Python float
                "mean_neighbor_l2_squared": float(distances.mean()),
                # 這篇 query 的 D_u
                "D_u": float(divergence["d_u"][query_index]),
                # 這篇 query 的 D_l
                "D_l": float(divergence["d_l"][query_index]),
                # 這篇 query 尚未套用 EMA 的 D_u + beta × D_l
                "raw_divergence": float(divergence["raw_divergence"][query_index]),
            }
        )

    # 回傳逐篇明細，以及本 fold 的篩選漏斗計數
    return rows, counts


# 將一組逐篇 rows 彙整成固定欄位的描述統計
def summarize(rows):
    # 當某組沒有樣本時仍回傳相同 schema，讓 JSON 與 CSV 欄位保持一致
    empty = {
        # n 是樣本數；空集合固定為 0
        "n": 0,
        # None 表示該統計量沒有可計算的樣本，不把它誤寫成 0
        "raw_mean": None,
        "raw_std": None,
        "raw_median": None,
        "raw_q1": None,
        "raw_q3": None,
        "raw_max": None,
        "upper_5pct_removed_n": 0,
        "upper_5pct_remaining_n": 0,
        "raw_mean_without_upper_5pct": None,
        "D_u_mean": None,
        "D_l_mean": None,
        "l2_mean": None,
    }
    # 空 list 在布林判斷中是 False；not rows 因此表示沒有任何樣本
    if not rows:
        # 提早 return，避免 NumPy 對空陣列計算平均值而產生警告與 NaN
        return empty

    # list comprehension 取出每列的 raw_divergence；
    # float64 只用於最後的統計彙整，提高加總與分位數計算的數值精度
    raw = np.array([row["raw_divergence"] for row in rows], dtype=np.float64)
    # 移除筆數採 ceil(n × 5%)；若該組只有一筆則保留它
    removed_n = min(int(np.ceil(raw.size * UPPER_TAIL_FRACTION)), raw.size - 1)
    remaining_raw = np.sort(raw)[: raw.size - removed_n]
    # 回傳此組樣本的描述統計 dict
    return {
        # len(rows) 是組內 fold-document observation 數
        "n": len(rows),
        # raw divergence 算術平均數，是 A1/A2 比較的主要數值
        "raw_mean": float(raw.mean()),
        # std() 預設 ddof=0，計算母體標準差
        "raw_std": float(raw.std()),
        # 中位數較不容易被極端散度值拉動
        "raw_median": float(np.median(raw)),
        # 第 25 百分位數（第一四分位數，Q1）# 25% 樣本不大於此值
        "raw_q1": float(np.quantile(raw, 0.25)),
        # 第 75 百分位數（第三四分位數，Q3） # 75% 樣本不大於此值
        "raw_q3": float(np.quantile(raw, 0.75)),
        # 最大 raw divergence，可協助檢查離群值 # 檢查是否存在極端高散度樣本
        "raw_max": float(raw.max()),
        # 上尾 5% 敏感度分析的移除數、剩餘數與剩餘樣本平均
        "upper_5pct_removed_n": removed_n,
        "upper_5pct_remaining_n": int(remaining_raw.size),
        "raw_mean_without_upper_5pct": float(remaining_raw.mean()),
        # 平均 D_u；Python list 會先交給 NumPy 轉成可運算陣列 # query 與 KNN 鄰居是否一致
        "D_u_mean": float(np.mean([row["D_u"] for row in rows])),
        # 平均 D_l # k 個鄰居彼此是否一致
        "D_l_mean": float(np.mean([row["D_l"] for row in rows])),
        # 平均的「每篇 query 對 k 位鄰居之平方 L2 平均」 # query 在特徵空間是否接近鄰居
        "l2_mean": float(np.mean([row["mean_neighbor_l2_squared"] for row in rows])),
    }


# 將由同一欄位 schema 組成的 list[dict] 寫成 CSV
def write_csv(path, rows):
    # 沒有資料列時不建立空白 CSV
    if not rows:
        return
    # Path(path) 確保輸入可當作 pathlib 路徑；
    # "w" 代表覆寫寫入，newline="" 避免 Windows CSV 出現多餘空白列；
    # utf-8-sig 會加入 BOM，讓 Excel 較容易正確辨識繁體中文
    with Path(path).open("w", newline="", encoding="utf-8-sig") as file:
        # 使用第一列 dict 的 key 順序作為全部 CSV 欄位順序
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        # 先寫入欄位名稱
        writer.writeheader()
        # 再依序寫入所有 dict 資料列
        writer.writerows(rows)


# 程式主流程：解析參數、逐 fold 推論與分組、跨 fold 彙整，最後寫出分析檔案
def main():
    # 讀取 parse_args() 定義的命令列參數
    args = parse_args()
    # 建立 PyTorch device：
    # 若參數不是 cuda 就直接使用指定裝置；若指定 cuda 且 CUDA 可用也使用 cuda；
    # 若指定 cuda 但環境沒有 CUDA，則自動退回 cpu
    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    # 從本機 bert-base-chinese 路徑載入與訓練時一致的 tokenizer
    # str(...) 將 Path 轉成 transformers 可接受的路徑字串
    tokenizer = BertTokenizer.from_pretrained(str(args.bert_path))
    # 建立輸出資料夾；parents=True 會連同缺少的上層資料夾一起建立，
    # exist_ok=True 表示資料夾已存在時不報錯。
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 收集所有 fold 中通過 gate 的逐篇資料，用於 pooled fold-document 統計
    all_rows = []
    # 收集所有可推論 unlabeled query 及其 KNN 類別與散度，不套用 A1/A2 gate
    unlabeled_divergence_rows = []
    # 每個元素是一個 fold 的漏斗式篩選計數
    filter_counts = []
    # 以字串 fold 編號為 key，保存各 fold 的 A1/A2 描述統計
    fold_summaries = {}
    # 以字串 fold 編號為 key，記錄實際使用的 checkpoint 完整路徑
    checkpoints = {}
    # 來源實驗資料夾名稱只要包含 avgs_simple，就一律使用 avg3 checkpoint；
    # 其他實驗則保留原本的 best_pair checkpoint，避免改變既有分析行為
    checkpoint_variant = (
        "avg3" if "avgs_simple" in args.experiment_dir.name.lower() else "pair"
    )
    # 保留 {fold} 佔位符，供逐 fold 路徑及輸出 JSON 共用同一個檔名規則
    checkpoint_pattern = (
        f"self_training_models/fold{{fold}}_self_training_best_{checkpoint_variant}.pth"
    )

    # f-string 將執行裝置、KNN k 與 beta 插入啟動訊息
    print(f"device={device}, k={args.k}, beta={args.beta}")
    # 顯示本次載入最佳模型的來源實驗資料夾
    print(f"experiment={args.experiment_dir}")
    # 顯示本次實際採用的 checkpoint 樣板，便於執行前核對 avg3／pair
    print(f"checkpoint_pattern={checkpoint_pattern}")

    # range 的停止值不包含在內，因此 end_fold + 1 才會真的執行到 end_fold
    for fold in range(args.start_fold, args.end_fold + 1):
        # 將目前 fold 編號代入統一樣板，再接到實驗根目錄
        checkpoint = args.experiment_dir / checkpoint_pattern.format(fold=fold)
        # 載入目前 fold 的少量 labeled train JSON，作為 KNN anchor
        train_dataset = PromptInferenceDataset(
            args.dataset_dir / f"fold{fold}_train.json", tokenizer
        )
        # 載入目前 fold 的原始完整 unlabeled JSON，作為待分析 query
        # Dataset 內仍會套用與原程式一致的 >512 token 篩選
        query_dataset = PromptInferenceDataset(
            args.dataset_dir / f"fold{fold}_unlabeled.json", tokenizer
        )

        # \n 會先換一行，讓不同 fold 的 console 訊息容易區分
        print(
            f"\nFold {fold}: checkpoint={checkpoint.name}, "
            f"train={len(train_dataset)}, unlabeled={len(query_dataset)}"
        )
        # 載入訓練程式保存的完整 PyTorch 模型物件
        # map_location="cpu" 先安全地載入 CPU，之後再明確移至目標 device；
        # weights_only=False 是因為檔案保存的是完整 prompt_bert 物件，不只是 state_dict
        # 請只對可信任的本機 checkpoint 使用這種完整物件反序列化方式
        model = torch.load(checkpoint, map_location="cpu", weights_only=False)
        # 將模型參數與 buffers 移到本次使用的 GPU 或 CPU
        model = model.to(device)

        # 顯示目前開始推論 labeled train。
        print("  inference: labeled train")
        # 同時取得 labeled anchors 的 embedding、模型分布與預測情緒子句
        train_features, train_distributions, train_predictions = infer_documents(
            model, train_dataset, tokenizer, args.batch_size, device
        )
        # 顯示目前開始推論 original full unlabeled
        print("  inference: original full unlabeled")
        # 使用完全相同的模型與推論規則處理 unlabeled queries
        query_features, query_distributions, query_predictions = infer_documents(
            model, query_dataset, tokenizer, args.batch_size, device
        )
        # 目前 fold 的兩邊推論都已完成，刪除大型模型的 Python 參照
        del model
        # 只有使用 CUDA 時才清除 PyTorch 已不再使用的 GPU cache
        if device.type == "cuda":
            # empty_cache 不會刪除仍被 Tensor 使用的記憶體，只釋放可重用的快取
            torch.cuda.empty_cache()

        # 以 labeled train 作 anchor，替每篇 unlabeled 找 KNN 並算 NeST raw divergence
        divergence = compute_knn_and_divergence(
            # labeled train emotion-clause embeddings
            train_features,
            # labeled train 的模型預測分布，不是真實 one-hot 標籤
            train_distributions,
            # original unlabeled emotion-clause embeddings
            query_features,
            # original unlabeled 的模型預測分布
            query_distributions,
            # 最近鄰數量 k
            args.k,
            # D_l 權重 beta
            args.beta,
        )
        # 保存目前 fold 每篇可推論 query、k 位鄰居 GT 類別及完整散度明細
        unlabeled_divergence_rows.extend(
            build_unlabeled_divergence_rows(
                fold, train_dataset, query_dataset, divergence
            )
        )
        # 套用 single-pair 與 exact-correct gate，再依 category 分成 A1/A2
        rows, counts = build_eligible_rows(
            # 當前 fold
            fold,
            # 供 KNN 索引回查 labeled 原始文檔
            train_dataset,
            # 供 query 索引回查 unlabeled 原始文檔
            query_dataset,
            # labeled train 預測子句 tuple list
            train_predictions,
            # unlabeled query 預測子句 tuple list
            query_predictions,
            # KNN 與散度結果
            divergence,
        )

        # list comprehension 只取 A1 rows，再計算此 fold 的 A1 描述統計
        same = summarize([row for row in rows if row["group"] == "A1_same"])
        # 同理，只取 A2 rows 計算此 fold 的 A2 描述統計
        different = summarize([row for row in rows if row["group"] == "A2_different"])
        # JSON object 的 key 最終是字串，因此先用 str(fold) 當 key
        fold_summaries[str(fold)] = {"A1_same": same, "A2_different": different}
        # 記錄目前 fold 實際載入的 checkpoint 路徑
        checkpoints[str(fold)] = str(checkpoint)
        # extend 會把 rows 中每一個 dict 逐項加入 all_rows；不是把整個 list 當成一項
        all_rows.extend(rows)
        # append 則把目前 fold 的 counts dict 當成一個元素加入
        filter_counts.append(counts)

        # dict.get(key, default) 只有在 key 不存在時才回傳 default
        # summarize 的固定 schema 一定有 raw_mean；若該組為空，其值會是 None，
        # 而不是這裡的 NaN，因此現有 console 格式化預期每個 fold 的 A1/A2 都非空
        same_mean = same.get("raw_mean", float("nan"))
        different_mean = different.get("raw_mean", float("nan"))
        # `:.8f` 以固定小數點後 8 位顯示平均散度
        print(
            f"  eligible={len(rows)}, A1={same['n']} mean={same_mean:.8f}, "
            f"A2={different['n']} mean={different_mean:.8f}"
        )

    # 將巢狀的 fold_summaries 轉成一列一組的平面資料，方便寫成 CSV
    fold_summary_rows = []
    # dict.items() 每次回傳一組 (fold key, 該 fold 的 groups dict)
    for fold, groups in fold_summaries.items():
        # 再逐一取出 A1_same／A2_different 與其統計 dict
        for group, statistics in groups.items():
            # `**statistics` 是 dictionary unpacking：
            # 將 n、raw_mean 等 key-value 展開並合併到目前這個新 dict
            fold_summary_rows.append({"fold": fold, "group": group, **statistics})

    # pooled fold-document 統計：把所有 fold 的逐篇 observation 合併後，
    # 分別對 A1 與 A2 統計；樣本多的 fold 對 pooled 結果權重自然較高
    pooled = {
        # 對指定 group 篩出 all_rows，再呼叫 summarize
        group: summarize([row for row in all_rows if row["group"] == group])
        # tuple 中列出要建立的兩個 group key
        for group in ("A1_same", "A2_different")
    }
    # 依「category 不同的鄰居數」0、1、...、k 另外做細分統計
    mismatch_summary = {
        # JSON key 使用字串；value 是目前 mismatch_count 的描述統計
        str(count): summarize(
            [row for row in all_rows if row["mismatch_count"] == count]
        )
        # range(args.k + 1) 產生 0 到 k，因為停止值 k+1 本身不包含在內
        for count in range(args.k + 1)
    }
    # 建立可做「同 fold 配對比較」的資料列；只有 A1、A2 都有樣本的 fold 才納入
    paired_fold_results = [
        {
            # fold 原本是 dict 的字串 key，輸出時轉回整數
            "fold": int(fold),
            # 該 fold A1 的 raw divergence 平均
            "A1_mean": groups["A1_same"]["raw_mean"],
            # 該 fold A2 的 raw divergence 平均
            "A2_mean": groups["A2_different"]["raw_mean"],
            # 正值代表該 fold 的 A2 平均散度高於 A1
            "difference_A2_minus_A1": (
                groups["A2_different"]["raw_mean"] - groups["A1_same"]["raw_mean"]
            ),
        }
        # 從每個 fold 的 A1/A2 統計建立 list comprehension
        for fold, groups in fold_summaries.items()
        # 只留下 A1 與 A2 樣本數都大於 0 的 fold，確保兩邊平均數都存在
        if groups["A1_same"]["n"] > 0 and groups["A2_different"]["n"] > 0
    ]
    # paired fold macro：先在每個 fold 內求組別平均，再讓每個有效 fold 等權平均
    # 它與 pooled fold-document 不同，不會讓樣本數多的 fold 取得較大權重
    paired_macro = {
        # 清楚列出實際參與 macro 平均的 fold
        "contributing_folds": [row["fold"] for row in paired_fold_results],
        # 括號內是 Python 條件運算式：有配對 fold 才計算，否則輸出 None
        "A1_mean": (
            # 先取每個有效 fold 的 A1_mean，再用 np.mean 做 fold 等權平均
            float(np.mean([row["A1_mean"] for row in paired_fold_results]))
            if paired_fold_results
            else None
        ),
        "A2_mean": (
            # 同樣對每個有效 fold 的 A2_mean 做 fold 等權平均
            float(np.mean([row["A2_mean"] for row in paired_fold_results]))
            if paired_fold_results
            else None
        ),
        "difference_A2_minus_A1": (
            # 平均各 fold 內的 A2−A1；不是先 pooled 全部樣本再相減
            float(np.mean([row["difference_A2_minus_A1"] for row in paired_fold_results]))
            if paired_fold_results
            else None
        ),
    }
    # 組成最終 JSON 的整體資料結構
    overall = {
        # config 保存本次分析的可重現設定與資料來源
        "config": {
            # 最佳模型來源實驗
            "experiment_dir": str(args.experiment_dir),
            # fold JSON 資料來源
            "dataset_dir": str(args.dataset_dir),
            # 各 fold 使用的 checkpoint 檔名樣板
            "checkpoint": checkpoint_pattern,
            # 使用者要求的起訖 fold
            "fold_range": [args.start_fold, args.end_fold],
            # 實際展開後執行的所有 fold 編號
            "folds": list(range(args.start_fold, args.end_fold + 1)),
            # KNN 鄰居數
            "k": args.k,
            # D_l 權重
            "beta": args.beta,
            # KNN 使用的 embedding 類型
            "embedding": "emotion_clause",
            # raw divergence 的公式文字
            "divergence": "D_u + beta * D_l",
            # 每個統計集合分別移除 ceil(n × 5%) 筆最高散度樣本
            "upper_5pct_rule": "remove ceil(n * 0.05) highest raw divergence values per group",
            # False 明確表示此分析沒有套用 self-training 跨輪 EMA
            "ema": False,
            # 說明 query 來自各 fold 原始完整 unlabeled，僅排除原程式的 >512 樣本
            "unlabeled_source": "original full fold unlabeled after the original >512 filter",
            # 模型判定情緒子句的二元機率規則
            "prediction_rule": "emotion clause iff binary P(是) > P(非)",
            # 進入 A1/A2 前要求的 exact-correct 規則
            "correctness_rule": "query and every neighbor predicted clause set exactly equals single-pair GT",
            # A1 判定時比較 category 的範圍
            "category_match_scope": "query and all neighbors have the same canonical category",
            # composite category 的既有 canonicalization 說明；
            # 實作會 split、strip、sort、join，再做 canonical string 完全相等比較
            "composite_category_rule": "split '&', sort, and require exact set equality",
            # 各 fold 真正使用的 checkpoint 絕對或相對路徑
            "checkpoints": checkpoints,
        },
        # observation_counts 說明 pooled 樣本的計數單位與去重後文檔數
        "observation_counts": {
            # 所有 folds 通過 512-token 條件、實際具有 KNN 與散度的 query 數
            "usable_unlabeled_fold_documents": len(unlabeled_divergence_rows),
            # 一篇 query 若出現在多個 fold，會在每個 fold 各算一筆 observation
            "eligible_fold_documents": len(all_rows),
            # set comprehension 以 doc_id 去重，計算至少一次符合條件的獨立 query 文檔數
            "unique_query_documents": len({row["query_doc_id"] for row in all_rows}),
        },
        # 所有 fold-document observations 合併後的 A1/A2 統計
        "pooled_fold_doc": pooled,
        # 依 0～k 位鄰居 category 不同數量分層的統計
        "by_category_mismatch_count": mismatch_summary,
        # 每個同時具有 A1/A2 的 fold 及其組內平均差
        "paired_fold_results": paired_fold_results,
        # 上述有效 folds 的等權 macro 統計
        "paired_fold_macro": paired_macro,
    }

    # 所有輸出檔名加入本次 k 值，避免 k=3 與 k=5 的結果互相覆蓋
    prefix = f"final_model_k{args.k}"
    # 輸出每篇可推論 unlabeled query、各 KNN 鄰居 GT 類別及 D_u／D_l／總散度
    write_csv(
        args.output_dir / f"{prefix}_all_unlabeled_divergence.csv",
        unlabeled_divergence_rows,
    )
    # 輸出每一篇符合 A1/A2 條件的完整明細
    write_csv(args.output_dir / f"{prefix}_eligible_samples.csv", all_rows)
    # 輸出每 fold 各篩選 gate 的累積通過數量
    write_csv(args.output_dir / f"{prefix}_filter_counts.csv", filter_counts)
    # 輸出每 fold、每 group 的描述統計
    write_csv(args.output_dir / f"{prefix}_fold_summary.csv", fold_summary_rows)
    # 開啟整體 JSON 摘要檔；with 區塊結束時檔案會自動關閉
    with (args.output_dir / f"{prefix}_overall_summary.json").open(
        # "w" 表示覆寫；UTF-8 保留繁體中文
        "w", encoding="utf-8"
    ) as file:
        # ensure_ascii=False 讓中文直接寫出而不是 \uXXXX；
        # indent=2 以兩格縮排，讓 JSON 容易閱讀
        json.dump(overall, file, ensure_ascii=False, indent=2)

    # 在 console 顯示輸出位置
    print(f"\n分析完成，輸出位置: {args.output_dir}")
    print(f"all usable unlabeled fold-doc details: n={len(unlabeled_divergence_rows)}")
    # 顯示 pooled fold-document A1 樣本數與平均 raw divergence；
    # 這不是 paired fold macro 結果
    print(
        f"pooled fold-doc A1: n={pooled['A1_same']['n']}, "
        f"mean={pooled['A1_same'].get('raw_mean', float('nan')):.8f}"
    )
    # 顯示 pooled fold-document A2 樣本數與平均 raw divergence
    print(
        f"pooled fold-doc A2: n={pooled['A2_different']['n']}, "
        f"mean={pooled['A2_different'].get('raw_mean', float('nan')):.8f}"
    )


# Python 直接執行本檔時，特殊變數 __name__ 會等於 "__main__"
# 若本檔只是被其他模組 import，這個條件不成立，因此不會自動開始完整分析
if __name__ == "__main__":
    # 呼叫上面定義的主流程
    main()
