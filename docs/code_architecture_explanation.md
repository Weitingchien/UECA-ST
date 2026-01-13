# UECA_CE_few_shot_ST_nest.py 程式碼架構解說

## 📌 程式概述

### 傳統 Self-Training vs NeST

| 方法 | 樣本篩選策略 |
|:---|:---|
| **傳統 Self-Training** | 使用人工定義的閥值 (如 confidence > 0.9) 篩選偽標籤 |
| **NeST (本專案)** | 基於鄰居一致性的散度分數 (Divergence Score) 篩選偽標籤 |

**UECA-Prompt + NeST** 演算法：
- **UECA-Prompt**: 使用 Prompt-based 方式進行情緒-原因配對抽取 (ECPE)
- **NeST (Neighborhood Sample Selection)**: 根據鄰居一致性選擇高品質偽標籤


---

## 🏗️ 程式架構總覽

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        UECA_CE_few_shot_ST_nest.py                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  1. 輔助函數 (Utility Functions)                                         │
│     ├── set_random_seed()         # 設定隨機種子                         │
│     ├── get_label_index()         # 取得標籤 token IDs                   │
│     └── get_binary_token_ids()    # 取得「是/非」token IDs               │
│                                                                         │
│  2. 分佈計算函數 (Distribution Computing)                                │
│     ├── _compute_label_distribution_from_logits()  # 計算情緒/原因分佈   │
│     ├── _compute_emotion_clause_distribution()     # 情緒句分佈          │
│     └── _compute_cause_clause_distribution()       # 原因句分佈          │
│                                                                         │
│  3. 統計收集函數 (Statistics Collection)                                 │
│     ├── collect_labeled_statistics()    # 收集已標記樣本的 embedding     │
│     └── collect_unlabeled_statistics()  # 收集未標記樣本的 embedding     │
│                                                                         │
│  4. 資料集類別 (Dataset Classes)                                         │
│     ├── MyDataset              # 有標籤資料集                            │
│     ├── UnlabeledDataset       # 未標籤資料集                            │
│     └── PseudoLabeledDataset   # 偽標籤資料集                            │
│                                                                         │
│  5. 模型類別 (Model Class)                                               │
│     └── prompt_bert            # BERT-based Prompt 模型                  │
│         ├── forward()          # 前向傳播                                │
│         ├── get_cls_embeddings()     # 取得 [CLS] embedding             │
│         └── get_clause_embeddings()  # 取得子句 embedding               │
│                                                                         │
│  6. 主程式 (_run_main)                                                   │
│     └── 包含完整的訓練與自訓練流程                                        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 🔄 NeST 演算法流程

> 以下行數對應 `UECA_CE_few_shot_ST_nest.py`

```
Phase 1: 初始訓練 (Initial Training) [Line 1949-2260]
================================================
┌─────────────────────────────────────────────────────────────────────────┐
│  輸入: 少量標記資料 (10% labeled data)                                   │
│        ↓                                                                │
│  ┌─────────────┐                                                        │
│  │ BERT Model  │ ← 使用 Masked Language Modeling 方式訓練               │
│  │ (Prompt)    │                                                        │
│  └─────────────┘                                                        │
│        ↓                                                                │
│  模型儲存: Pair F1 > 最佳值時儲存 [Line 2172-2177]                       │
│        ↓                                                                │
│  輸出: 訓練好的初始模型 M₀                                               │
└─────────────────────────────────────────────────────────────────────────┘

Phase 2: 自訓練迴圈 (Self-Training Loop) × N 輪 [Line 2265-3320]
================================================================
    for self_round in range(opt.self_training_rounds):  # Line 2265

    ┌─────────────────────────────────────────────────────────────────────┐
    │  Round t = 1, 2, 3, ..., N                                          │
    │                                                                     │
    │  Step 1: 收集 Labeled 統計資訊 [Line 2522-2530]                      │
    │  ┌───────────────┐    ┌─────────────────────────────────────────┐   │
    │  │ Labeled Data  │ →  │ collect_labeled_statistics()            │   │
    │  │ (X_l)         │    │ ├── embedding (768維向量)                │   │
    │  └───────────────┘    │ └── 機率分佈 [P(是), P(非)]              │   │
    │                       └─────────────────────────────────────────┘   │
    │                                           ↓                         │
    │  Step 2: 收集 Unlabeled 統計資訊 [Line 2555-2560]                    │
    │  ┌───────────────┐    ┌─────────────────────────────────────────┐   │
    │  │ Unlabeled Data│ →  │ collect_unlabeled_statistics()          │   │
    │  │ (X_u)         │    │ ├── embedding (768維向量)                │   │
    │  └───────────────┘    │ ├── 機率分佈 [P(是), P(非)]              │   │
    │                       │ └── 偽標籤 (pseudo labels)                │   │
    │                       └─────────────────────────────────────────┘   │
    │                                           ↓                         │
    │  Step 3: NeST 樣本選擇 [Line 2596-2610]                              │
    │  ┌─────────────────────────────────────────────────────────────┐    │
    │  │  select_samples_and_generate_pseudo_labels()                │    │
    │  │                                                             │    │
    │  │  3.1 KNN 搜尋: 對每個 unlabeled 樣本找 k 個最近的 labeled    │    │
    │  │      FAISS L2 距離: d = √Σ(v_i - v_j)²                      │    │
    │  │                                                             │    │
    │  │  3.2 計算 Divergence Score: D = D_u + β × D_l               │    │
    │  │  3.3 EMA 平滑: μ^(t) = (1-m)×μ^(t-1) + m×D^(t)             │    │
    │  │  3.4 依機率抽樣: 選擇低散度的樣本                            │    │
    │  └─────────────────────────────────────────────────────────────┘    │
    │                                           ↓                         │
    │  Step 4: 建立偽標籤資料集 [Line 3159-3162]                           │
    │  ┌─────────────────────────────────────────────────────────────┐    │
    │  │ PseudoLabeledDataset(round_pseudo_labeled_samples, ...)     │    │
    │  └─────────────────────────────────────────────────────────────┘    │
    │                                           ↓                         │
    │  Step 5: 合併訓練集 [Line 3174-3176]                                 │
    │  ┌─────────────────────────────────────────────────────────────┐    │
    │  │ combined_dataset = ConcatDataset([labeled] + [pseudo_list]) │    │
    │  └─────────────────────────────────────────────────────────────┘    │
    │                                           ↓                         │
    │  Step 6: 重新訓練模型 [Line 3180-3260]                               │
    │  ┌─────────────────────────────────────────────────────────────┐    │
    │  │ combined_loader = DataLoader(combined_dataset, ...)         │    │
    │  │ 使用合併後的資料集繼續訓練模型                               │    │
    │  └─────────────────────────────────────────────────────────────┘    │
    │                                           ↓                         │
    │  Step 7: 模型儲存 [Line 3283-3290]                                   │
    │  ┌─────────────────────────────────────────────────────────────┐    │
    │  │ 儲存條件: Validation Pair F1 > 目前最佳值                    │    │
    │  │ torch.save(model, st_model_path)                            │    │
    │  └─────────────────────────────────────────────────────────────┘    │
    └─────────────────────────────────────────────────────────────────────┘
    
    重複 N 輪後，得到最終模型 M_N
```

**模型儲存策略：**
- 儲存條件：當 **Validation 的 Pair F1** 超過目前最佳值時自動儲存
- Phase 1 儲存路徑：`fold{N}.pth` [Line 2175-2177]
- Phase 2 儲存路徑：`self_training_models/fold{N}_self_training_best.pth` [Line 3285-3290]

> 儲存的是**歷史最佳模型**，不是最後一輪的模型

```python
# Line 2172-2177 儲存邏輯
if f_pair > max_f1_pair:           # 只有達到新最佳時才儲存
    max_f1_pair = f_pair           # 更新最佳值
    torch.save(model, checkpoint)  # 覆蓋舊的 checkpoint
```

範例流程：
```
iter 10: F1=0.35 → 新最佳 → 儲存 fold1.pth (版本1)
iter 25: F1=0.41 → 新最佳 → 儲存 fold1.pth (版本2，覆蓋)
iter 40: F1=0.38 → 未超過 → 不儲存
iter 69: F1=0.40 → 未超過 → 不儲存
最終 fold1.pth = iter 25 的模型 (F1=0.41)
```


---

## 📉 Loss 計算

### BERT Masked Language Modeling Loss

使用 Hugging Face 的 `BertForMaskedLM`，loss 自動計算 [MASK] 位置的 Cross-Entropy Loss

```python
# Line 1536-1539: prompt_bert.forward()
def forward(self, x_bert, labels):
    output = self.bert(x_bert, labels=labels)
    loss, logits = output.loss, output.logits
    return loss, logits
```

### Phase 1: 初始訓練 Loss [Line 2087-2094]
```
Loss = CrossEntropy(預測的 [MASK], 真實標籤)
```

### Phase 2: 自訓練 Loss [Line 3197-3213]

```python
# Line 3213
total_loss = loss_l + γ × loss_p
```

#### `loss_l` vs `loss_p` 差異

| 項目 | `loss_l` | `loss_p` |
|:---|:---|:---|
| 資料來源 | `MyDataset` (Line 1180) | `PseudoLabeledDataset` (Line 1258) |
| 標籤來源 | **人工標註** (JSON pairs) | **模型預測** (偽標籤) |
| `is_labeled` | `True` | `False` |
| 計算方式 | Cross-Entropy | Cross-Entropy (**相同**) |

#### `mask_label` 內容差異

**有標籤資料 (`MyDataset`):**
```
mask_label_l = [-100, ..., 3221, 7478, 8135, ...]
                           ↑     ↑     ↑
                          "是"  "非"   "无"  ← 來自 JSON 人工標註
```

**偽標籤資料 (`PseudoLabeledDataset`, Line 1266-1270):**
```python
mask_label = np.full(512, -100)              # 先填 -100 (忽略)
mask_positions = np.where(x == 103)[0]       # 找 [MASK] 位置
mask_label[mask_positions] = pseudo_label    # 填入偽標籤
```
```
mask_label_p = [-100, ..., 3221, 7478, 8135, ...]
                           ↑     ↑     ↑
                          "是"  "非"   "无"  ← 來自模型預測
```

#### Cross-Entropy 計算 (單一 [MASK] 位置)

```
模型輸出 logits[pos] = [0.1, 0.05, ..., 0.78, ..., 0.02]  (21128 維)
                                         ↑
                                      token_id 3221 ("是")

標籤 mask_label[pos] = 3221

CE Loss = -log(softmax(logits[pos])[3221])
        = -log(0.78) ≈ 0.248
```

**BERT 只計算 `labels != -100` 的位置，即只計算 [MASK] 位置的 loss。**

#### 為什麼要乘以 γ？

偽標籤可能有錯誤，所以用 `γ = 0.5` 降低偽標籤對訓練的影響：
```
total_loss = loss_l + 0.5 × loss_p
```

### 範例：fold1_train.json (doc_id: 1071)

```json
{
  "doc_id": "1071",
  "pairs": [[12, 10]],  // 情緒句=12, 原因句=10
  "clauses": [
    {"clause_id": "1", "clause": "2012年6月11日"},
    {"clause_id": "10", "clause": "..."},  // 原因句
    {"clause_id": "12", "clause": "..."}   // 情緒句
  ]
}
```

**Prompt 格式:**
```
[CLS] 子句1内容 [MASK]_e [MASK]_c [MASK]_p ... 子句12内容 [MASK]_e [MASK]_c [MASK]_p [SEP]
                 ↑        ↑        ↑            ↑        ↑        ↑
                非        非       无            是        非       10
```


---

## 🔑 函式對照表


| 流程步驟 | 對應函數 | 程式碼位置 |
|:---|:---|:---:|
| 載入有標籤資料 | `MyDataset` | Line 1063-1187 |
| 載入未標籤資料 | `UnlabeledDataset` | Line 1192-1244 |
| 收集 labeled 統計 | `collect_labeled_statistics()` | Line 423-584 |
| 收集 unlabeled 統計 | `collect_unlabeled_statistics()` | Line 587-703 |
| 取得 CLS embedding | `prompt_bert.get_cls_embeddings()` | Line 1541-1543 |
| 取得子句 embedding | `prompt_bert.get_clause_embeddings()` | Line 1545-1561 |
| NeST 樣本選擇 | `select_samples_and_generate_pseudo_labels()` | (外部檔案) |
| 主程式流程 | `_run_main()` | Line 1824-3490 |

---

## 📊 Embedding 獲取方式

| Mode | 方法 | 優點 | 缺點 |
|:---|:---|:---|:---|
| `cls` | 取 [CLS] 位置的 768 維向量 | 簡單、穩定 | 可能包含太多無關資訊 |
| `emotion_clause` | 對情緒句做 Mean Pooling | 聚焦於任務相關子句 | 依賴模型預測品質 |
| `cause_clause` | 對原因句做 Mean Pooling | 聚焦於任務相關子句 | 依賴模型預測品質 |

---

## 📝 Prompt 模板格式

```
輸入序列: [CLS] 子句1 [MASK]_e [MASK]_c [MASK]_p 子句2 [MASK]_e [MASK]_c [MASK]_p ... [SEP]

[MASK]_e: 預測該子句是否為情緒句 → 「是」或「非」
[MASK]_c: 預測該子句是否為原因句 → 「是」或「非」
[MASK]_p: 若為情緒句，預測對應的原因子句編號 → 「1」~「75」
```

---


