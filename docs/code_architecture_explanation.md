# UECA_CE_few_shot_ST_nest.py 程式碼架構解說

## 程式概述

### 傳統 Self-Training vs NeST

| 方法 | 樣本篩選策略 |
|:---|:---|
| **傳統 Self-Training** | 使用人工定義的閥值 (如 confidence >= 0.9) 篩選偽標籤 |
| **NeST** | 基於鄰居一致性的散度分數 (Divergence Score) 篩選偽標籤 |

**UECA-Prompt + NeST** 演算法：
- **UECA-Prompt**: 使用 Prompt-based 方式進行情緒-原因配對抽取 (ECPE)
- **NeST (Neighborhood Sample Selection)**: 根據鄰居一致性選擇高品質偽標籤


---

##  程式架構總覽

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
│     ├── MyDataset              # 有標註資料集                            │
│     ├── UnlabeledDataset       # 未標註資料集                            │
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

##  整體訓練流程

> 以下行數對應 `UECA_CE_few_shot_ST_nest.py`

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           整體訓練流程                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Phase 1: 初始監督訓練 [Line 2077-2260]                                  │
│  ─────────────────────────────────────────────────────────────────────  │
│    輸入: 少量標記資料 (例如 10% labeled data)                             │
│    輸出: 初始訓練完成的模型 M₀                                            │
│                                                                         │
│                              ↓                                          │
│                                                                         │
│  Phase 2: 自訓練 [Line 2255-3430]                                        │
│  ─────────────────────────────────────────────────────────────────────  │
│    for each self_round in self_training_rounds:                         │
│      1. 用當前模型對未標註資料預測 → 產生偽標籤                           │
│      2. 根據 --pseudo_selector 選擇高品質樣本 (見下表)                    │
│         - threshold [Line 2449]                                         │
│         - nest [Line 2530]                                              │
│         - random [Line 3063]                                            │
│      3. 合併有標籤 + 偽標籤資料 [Line 3188]                               │
│      4. 訓練 st_training_epochs 個 epoch [Line 3197]                     │
│      5. 計算 total_loss = loss_l + γ * loss_p [Line 3340]                │
│                                                                         │
│                              ↓                                          │
│                                                                         │
│  Phase 3: 測試 [Line 3444-3495]                                          │
│  ─────────────────────────────────────────────────────────────────────  │
│    使用最佳模型在測試集上評估 Pair F1                                     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### --pseudo_selector 偽標籤選擇策略

| 模式 | 說明 | 選擇邏輯 | 程式碼位置 |
|:---|:---|:---|:---|
| `threshold` | **閾值篩選** (預設) | 選擇 softmax 機率 > threshold 的樣本 | Line 2449 |
| `nest` | **NeST 鄰居一致性** | 根據散度分數 + EMA 平滑選擇樣本 | Line 2530 |
| `random` | **隨機選擇** | 隨機抽樣固定比例的未標註樣本 | Line 3063 |

---

## 🔄 NeST 演算法流程

> 以下行數對應 `UECA_CE_few_shot_ST_nest.py`

```
Phase 1: 初始訓練 (Initial Training) [Line 1949-2260]
================================================
┌─────────────────────────────────────────────────────────────────────────┐
│  輸入: 少量標註資料 (10% labeled data)                                   │
│        ↓                                                                │
│  ┌─────────────┐                                                        │
│  │ BERT Model  │ ← 使用 Masked Language Modeling 方式訓練               │
│  │ (Prompt)    │                                                        │
│  └─────────────┘                                                        │
│        ↓                                                                │
│  模型儲存: Pair F1 > 最佳值時儲存 [Line 2186-2191]                       │
│        ↓                                                                │
│  輸出: 訓練好的初始模型 M₀                                               │
└─────────────────────────────────────────────────────────────────────────┘

Phase 2: 自訓練迴圈 (Self-Training Loop) × N 輪 [Line 2279-3440]
================================================================
    for self_round in range(opt.self_training_rounds):  # Line 2279

    ┌─────────────────────────────────────────────────────────────────────┐
    │  Round t = 1, 2, 3, ..., N                                          │
    │                                                                     │
    │  Step 1: 收集 Labeled 統計資訊 [Line 2536-2550]                      │
    │  ┌───────────────┐    ┌─────────────────────────────────────────┐   │
    │  │ Labeled Data  │ →  │ collect_labeled_statistics()            │   │
    │  │ (X_l)         │    │ ├── embedding (768維向量)                │   │
    │  └───────────────┘    │ └── 機率分佈 [P(是), P(非)]              │   │
    │                       └─────────────────────────────────────────┘   │
    │                                           ↓                         │
    │  Step 2: 收集 Unlabeled 統計資訊 [Line 2569-2590]                    │
    │  ┌───────────────┐    ┌─────────────────────────────────────────┐   │
    │  │ Unlabeled Data│ →  │ collect_unlabeled_statistics()          │   │
    │  │ (X_u)         │    │ ├── embedding (768維向量)                │   │
    │  └───────────────┘    │ ├── 機率分佈 [P(是), P(非)]              │   │
    │                       │ └── 偽標籤 (pseudo labels)                │   │
    │                       └─────────────────────────────────────────┘   │
    │                                           ↓                         │
    │  Step 3: NeST 樣本選擇 [Line 2610-2650]                              │
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
    │  Step 4: 建立偽標籤資料集 [Line 3173-3180]                           │
    │  ┌─────────────────────────────────────────────────────────────┐    │
    │  │ PseudoLabeledDataset(round_pseudo_labeled_samples, ...)     │    │
    │  └─────────────────────────────────────────────────────────────┘    │
    │                                           ↓                         │
    │  Step 5: 合併訓練集 [Line 3189-3194]                                 │
    │  ┌─────────────────────────────────────────────────────────────┐    │
    │  │ combined_dataset = ConcatDataset([labeled] + [pseudo_list]) │    │
    │  └─────────────────────────────────────────────────────────────┘    │
    │                                           ↓                         │
    │  Step 6: 重新訓練模型 [Line 3198-3360]                               │
    │  ┌─────────────────────────────────────────────────────────────┐    │
    │  │ combined_loader = DataLoader(combined_dataset, ...)         │    │
    │  │ 使用合併後的資料集繼續訓練模型                               │    │
    │  └─────────────────────────────────────────────────────────────┘    │
    │                                           ↓                         │
    │  Step 7: 模型儲存 [Line 3412-3420]                                   │
    │  ┌─────────────────────────────────────────────────────────────┐    │
    │  │ 儲存條件: Validation Pair F1 > 目前最佳值                    │    │
    │  │ torch.save(model, st_model_path)                            │    │
    │  └─────────────────────────────────────────────────────────────┘    │
    └─────────────────────────────────────────────────────────────────────┘
    
    重複 N 輪後，得到最終模型 M_N
```

**模型儲存策略：**
- 儲存條件：當 **Validation 的 Pair F1** 超過目前最佳值時自動儲存
- Phase 1 儲存路徑：`fold{N}.pth` [Line 2189]
- Phase 2 儲存路徑：`self_training_models/fold{N}_self_training_best.pth` [Line 3412-3417]

> 儲存的是**歷史最佳模型**，不是最後一輪的模型

```python
# Line 2186-2191 儲存邏輯
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

## Loss 計算

### BERT Masked Language Modeling Loss

Hugging Face 的 `BertForMaskedLM`，計算 [MASK] 位置的 Cross-Entropy Loss

#### 數學公式

**Step 1: Softmax（將 logits 轉為機率）**

$$\hat{y}_{t} = \text{softmax}(z_t) = \frac{e^{z_{t,j}}}{\sum_{j=1}^{|V|} e^{z_{t,j}}}$$

**Step 2: 單一位置的 Cross-Entropy**

$$\text{CE}_t = -\log \hat{y}_{t, y_t}$$

**Step 3: 整體 MLM Loss（僅計算有效位置）**

$$\mathcal{L}_{MLM} = -\frac{1}{T} \sum_{t=1}^{n} \mathbf{1}_{[y_t \neq -100]} \cdot (\log \hat{y}_{t, y_t})$$

#### 符號定義

| 符號 | 意義 |
|:---|:---|
| $t$ | 序列位置的索引（1, 2, ..., n） |
| $n$ | 輸入序列長度（512） |
| $T$ | 有效 token 數量（$y_t \neq -100$ 的位置數，即 [MASK] 數量） |
| $z_t$ | 第 $t$ 個位置的模型輸出 logits（$\|V\|$ 維向量） |
| $z_{t,j}$ | 第 $t$ 個位置、詞彙表第 $j$ 個詞的 logit 值 |
| $\hat{y}_t$ | 第 $t$ 個位置的機率分佈（Softmax 後） |
| $y_t$ | 第 $t$ 個位置的標籤（正確答案 token id，或 -100 表示忽略） |
| $\hat{y}_{t, y_t}$ | 第 $t$ 個位置、正確答案 $y_t$ 的預測機率 |
| $\|V\|$ | 詞彙表大小（BERT-base-chinese = 21128） |
| $\mathbf{1}_{[y_t \neq -100]}$ | 指示函數：當 $y_t \neq -100$ 時為 1，否則為 0 |
| $\text{CE}_t$ | 第 $t$ 個位置的 Cross-Entropy Loss |

#### 程式碼實作

```python
# Line 1543-1546: prompt_bert.forward()
def forward(self, x_bert, labels):
    output = self.bert(x_bert, labels=labels)  # labels=-100 的位置會被忽略
    loss, logits = output.loss, output.logits
    return loss, logits
```

### Phase 1: 初始訓練 Loss [Line 2097-2104]
```
Loss = CrossEntropy(預測的 [MASK], 真實標籤)
```

### Phase 2: 自訓練 Loss [Line 3200-3340]

```python
# Line 3335
total_loss = loss_l + γ × loss_p
```

### Pair F1 計算方式 [Line 1641-1658]

**Pair** 是針對**第三個 [MASK]**（預測相關子句編號）計算的：

```
[MASK] [MASK] [MASK]
  ↑      ↑      ↑
 是/非  是/非  "1"~"75" 或 "无"
 情緒   原因   ← Pair 編號(對應第幾個子句)
```

#### 計算邏輯

| 條件 | 累加項目 |
|:---|:---|
| 預測結果在 `1~75` 中 | `pair_pre += 1`（模型預測出的配對數） |
| 預測 == Ground Truth（且都在 `1~75`） | `pair_acc += 1`（預測正確的配對數） |

#### 公式

$$\text{Precision} = \frac{\text{pair\_acc}}{\text{pair\_pre}}$$

$$\text{Recall} = \frac{\text{pair\_acc}}{\text{pair\_gt}}$$

$$F1 = \frac{2 \times P \times R}{P + R}$$

#### 教師模型 vs 學生模型

在自訓練 (Self-Training) 中，**同一個模型**在不同時間點扮演不同角色：

| 角色 | 時間點 | 功能 |
|:---|:---|:---|
| **教師模型** | 每輪自訓練**開始時** | 對未標註資料預測，產生偽標籤 (`mask_label_p`) |
| **學生模型** | **訓練過程中** | 學習預測正確答案，與偽標籤計算 `loss_p` |

```
Self-Training Round N:
├── T₀: 教師模型 (M_start) 預測 → 產生 mask_label_p (偽標籤)
├── T₁: 學生模型 (M_start) 對 x_p 預測 → 計算 loss_p → 模型更新
├── T₂: 學生模型 (M_updated) 對 x_p 預測 → 計算 loss_p → 模型更新
└── ...
```

**關鍵**：偽標籤在每輪開始時生成後**固定不變**，而學生模型在訓練過程中**持續更新**。

#### `loss_l` vs `loss_p` 差異

| 項目 | `loss_l` | `loss_p` |
|:---|:---|:---|
| 資料來源 | `MyDataset` (Line 1063) | `PseudoLabeledDataset` (Line 1257) |
| 標籤來源 | **人工標註** (JSON pairs) | **教師模型預測** (偽標籤) |
| `is_labeled` | `True` | `False` |
| 計算方式 | Cross-Entropy | Cross-Entropy (**相同**) |

#### `mask_label` 內容差異

**有標註資料 (`MyDataset`):**
```
mask_label_l = [-100, ..., 3221, 7478, 8135, ...]
                           ↑     ↑     ↑
                          "是"  "非"   "无"  ← 來自 JSON 人工標註
```

**偽標籤資料 (`PseudoLabeledDataset`, Line 1273-1277):**
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

## 函式對照表


| 流程步驟 | 對應函數 | 程式碼位置 |
|:---|:---|:---:|
| 載入有標註資料 | `MyDataset` | Line 1063-1194 |
| 載入未標註資料 | `UnlabeledDataset` | Line 1199-1252 |
| 收集 labeled 統計 | `collect_labeled_statistics()` | Line 423-584 |
| 收集 unlabeled 統計 | `collect_unlabeled_statistics()` | Line 587-703 |
| 取得 CLS embedding | `prompt_bert.get_cls_embeddings()` | Line 1548-1550 |
| 取得子句 embedding | `prompt_bert.get_clause_embeddings()` | Line 1552-1568 |
| NeST 樣本選擇 | `select_samples_and_generate_pseudo_labels()` | (外部檔案) |
| 主程式流程 | `_run_main()` | Line 1831-3500 |

---

## Embedding 獲取方式

| Mode | 方法 |
|:---|:---|
| `cls` | 取 [CLS] 位置的 768 維向量 |
| `emotion_clause` | 對情緒句做 Mean Pooling |
| `cause_clause` | 對原因句做 Mean Pooling |

---

## Prompt 模板格式

**輸入序列 (x_bert):**
```
[CLS] 1 子句1內容 [MASK] [MASK] [MASK] [SEP] 2 子句2內容 [MASK] [MASK] [MASK] [SEP] ... [SEP] [PAD] [PAD] ...
```

**標籤序列 (y_bert):**
```
[CLS] 1 子句1內容 非 非 无 [SEP] 2 子句2內容 是 是 3 [SEP] ... [SEP] [PAD] [PAD] ...
```

**三個 [MASK] 的意義：**
| 位置 | 預測內容 | 可能的值 |
|:---|:---|:---|
| [MASK]₁ | 是否為情緒句 | 「是」或「非」 |
| [MASK]₂ | 是否為原因句 | 「是」或「非」 |
| [MASK]₃ | 若為原因句，對應的情緒句編號 | 「1」~「75」或「无」 |

**範例 (doc_id: 1071):**
```
輸入: [CLS] 1 2012 年 6 月 11 日 [MASK] [MASK] [MASK] [SEP] ... 12 王 什 彩 十 分 激 动 [MASK] [MASK] [MASK] [SEP] ...
標籤: [CLS] 1 2012 年 6 月 11 日 非 非 无 [SEP] ... 12 王 什 彩 十 分 激 动 是 非 无 [SEP] ...
                                                    ↑情緒句 ↑非原因句 ↑無配對
```

---


