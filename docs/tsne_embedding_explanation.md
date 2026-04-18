# t-SNE 視覺化分析說明

## 問題 1: 為什麼 Round 10 右邊會出現藍點？

### 藍點的含義

根據 `utils/visualize_tsne.py` 的程式碼（第 529-530 行）：

```python
# 再畫藍色（有標籤）
plt.scatter(labeled_2d[:, 0], labeled_2d[:, 1], 
            c='blue', alpha=0.7, s=50, marker='s', label=f'有標籤樣本 ({len(labeled_2d)})')
```

**藍點 = 有標籤樣本（Labeled Samples）**，也就是 `fold1_train.json` 中的標註樣本。

### Round 1 vs Round 10 的差異

| 輪次 | 模型狀態 | Embedding 空間特性 |
|------|----------|-------------------|
| **Round 1** | 初始模型（僅用少量標註資料訓練） | 標註樣本（藍點）聚集在**左側**，未標註樣本分散 |
| **Round 10** | Self-training 後的模型（經過 9 輪偽標籤訓練） | Embedding 空間**重新分布**，部分藍點移到**右側** |

### 為什麼會發生這種現象？

這是 **Self-training 過程中模型學習的自然結果**：

1. **模型持續更新**：
   - 每一輪 Self-training 都會選擇高品質偽標籤樣本加入訓練
   - 模型的參數不斷調整，學習到新的特徵表示

2. **Embedding 空間演變**：
   - Round 1：模型只見過 `fold1_train.json` 的標註樣本，Embedding 空間主要反映這些樣本之間的關係
   - Round 10：模型額外見過約 `3 × 標註樣本數` × 9 輪 = **大量偽標籤樣本**
   - 這些偽標籤樣本可能具有與原始標註樣本不同的特徵分布

3. **t-SNE 的非線性映射**：
   - t-SNE 將 768 維的 Embedding 壓縮到 2 維，保留**局部鄰域結構**
   - 隨著 Embedding 空間改變，2D 投影的相對位置也會變化
   - **右側出現藍點不代表這些樣本變差，而是它們在新的 Embedding 空間中與其他樣本的相對關係改變了**

4. **可能的原因**：
   - **特徵擴展**：模型學會了更豐富的特徵表示，原本聚集的標註樣本在新空間中分散開
   - **偽標籤影響**：某些偽標籤樣本與部分標註樣本相似，拉動了 Embedding 空間的分布
   - **過擬合風險**：如果藍點過度分散或遠離其他同類樣本，可能暗示模型過擬合

### 如何解讀？

✅ **正常現象**：
- 如果大部分藍點仍然聚集，只有少數分散到右側
- 如果右側藍點附近有綠色點（完全正確的偽標籤），說明模型正確學習到了這些特徵

⚠️ **需要警惕**：
- 如果藍點過度分散且孤立（周圍沒有其他點）
- 如果右側藍點附近有大量紅色點（錯誤的偽標籤），可能表示模型學習到了錯誤的模式

**建議**：結合 `pseudo_label_evaluation_fold1_round10.txt` 的準確率數據，以及測試集 F1 分數的變化趨勢來綜合判斷。

---

## 問題 2: Embedding 提取的詳細實作說明

您的實驗使用 `--knn_embedding_mode cause_clause_mask`，以下詳細說明提取流程。

### 整體流程圖

```
文檔輸入 (x_bert: 512 tokens)
    ↓
模型前向傳播 (BERT)
    ↓
取得 hidden_states (512, 768) 和 probs (512, vocab_size)
    ↓
compute_clause_embeddings(mode='cause_clause_mask')
    ↓
【步驟 1】識別所有子句邊界
    ↓
【步驟 2】篩選「原因子句」(P(是) > P(非))
    ↓
【步驟 3】計算每個原因子句的 Embedding
    ↓
【步驟 4】加入 [MASK]_cause Embedding
    ↓
【步驟 5】聚合或 Fallback
    ↓
最終文檔 Embedding (768 維)
```

### 詳細步驟說明

#### 步驟 1: 識別子句邊界

程式碼位置：`UECA_CE_few_shot_ST_nest.py` 第 914-915 行

```python
boundaries = get_clause_boundaries(doc_input, mask_token_id, sep_token_id)
```

**功能**：找出文檔中每個子句的範圍，返回格式為：
```python
[(content_start, content_end, mask_pos), ...]
```

其中：
- `content_start`：子句內容開始位置（token 索引）
- `content_end`：子句內容結束位置（不含 [SEP]）
- `mask_pos`：該子句對應的 3 個 [MASK] 位置 `(emotion_mask, cause_mask, pair_mask)`

**範例**：
```
文檔結構: [CLS] 子句1內容 [SEP] [MASK]_e [MASK]_c [MASK]_p 子句2內容 [SEP] [MASK]_e [MASK]_c [MASK]_p ... [SEP]
         ↑       ↑           ↑     ↑      ↑      ↑      ↑        ↑           ↑     ↑      ↑      ↑      ↑
        0        1-5         6     7      8      9      10      11-15       16    17     18     19     20
boundaries[0] = (1, 6, (7, 8, 9))    # 子句1: 內容在 token 1~5, [MASK] 在 7,8,9
boundaries[1] = (10, 16, (17, 18, 19)) # 子句2: 內容在 token 10~15, [MASK] 在 17,18,19
```

#### 步驟 2: 判斷是否為原因子句

程式碼位置：第 923 行

```python
if is_emotion_or_cause_clause(doc_probs, mask_pos, mode, yes_token_id, no_token_id):
```

**判斷邏輯**（第 784-807 行）：

```python
def is_emotion_or_cause_clause(probs, mask_positions, mode, yes_token_id, no_token_id):
    emotion_pos, cause_pos, _ = mask_positions
    
    if mode in ('cause_clause', 'cause_clause_mask'):
        yes_prob = probs[cause_pos, yes_token_id]  # P(是)
        no_prob = probs[cause_pos, no_token_id]    # P(非)
    
    return yes_prob > no_prob  # 當 P(是) > P(非) 時，該子句被視為原因子句
```

**關鍵**：
- 只看第 2 個 [MASK]（`cause_mask`）位置的預測
- `yes_token_id = 3221`（「是」的 token）
- `no_token_id = 7478`（「非」的 token）
- 只要 **P(是) > P(非)** 就算是原因子句

#### 步驟 3: 計算子句內容的 Embedding

程式碼位置：第 924-925 行

```python
# 對子句內容做 mean pooling
clause_emb = hidden_states[batch_idx, content_start:content_end, :].mean(dim=0)
```

**說明**：
- `hidden_states[batch_idx, content_start:content_end, :]`：取出該子句所有 token 的 hidden states
  - Shape: `(子句長度, 768)`
- `.mean(dim=0)`：沿著 token 維度取平均
  - Shape: `(768,)`

**範例**：
```python
假設子句1內容為 "他很生氣" (對應 token 索引 1~5，共 5 個 token)
hidden_states[0, 1:6, :] → shape (5, 768)
clause_emb = mean(五個向量) → shape (768,)
```

#### 步驟 4: 加入 [MASK]_cause Embedding（`cause_clause_mask` 模式特有）

程式碼位置：第 935-940 行

```python
elif mode == 'cause_clause_mask':
    # 驗證位置確實是 [MASK]
    token_at_mask = doc_input[cause_mask_pos].item()
    assert token_at_mask == mask_token_id  # 確保是 103
    
    # 提取 [MASK]_cause 的 hidden state
    mask_emb = hidden_states[batch_idx, cause_mask_pos, :]  # (768,)
    
    # 計算平均: (子句內容 embedding + [MASK] embedding) / 2
    clause_emb = (clause_emb + mask_emb) / 2
```

**數學公式**：

$$
\text{clause\_emb}_{\text{final}} = \frac{\text{clause\_emb}_{\text{content}} + \text{mask\_emb}_{\text{cause}}}{2}
$$

其中：
- $\text{clause\_emb}_{\text{content}} = \frac{1}{L} \sum_{i=\text{start}}^{\text{end}} \mathbf{h}_i$（子句內容的平均）
- $\text{mask\_emb}_{\text{cause}} = \mathbf{h}_{\text{cause\_mask\_pos}}$（[MASK]_cause 位置的 hidden state）

**為什麼要加入 [MASK] embedding？**
- [MASK] 位置在訓練時學習到了「是否為原因」的全局資訊
- 結合 [MASK] embedding 可以增強判別能力，使 Embedding 更好地反映「原因」這個語義

#### 步驟 5: 聚合多個原因子句或 Fallback

程式碼位置：第 953-960 行

```python
if selected_embeddings:
    # 如果找到至少一個原因子句，取平均
    doc_emb = torch.stack(selected_embeddings).mean(dim=0)  # (768,)
else:
    # Fallback: 沒有任何原因子句 → 使用 [CLS]
    doc_emb = hidden_states[batch_idx, 0, :]  # (768,)
    is_fallback = True
```

**數學公式**：

$$
\mathbf{e}_{\text{doc}} = 
\begin{cases}
\frac{1}{N_{\text{cause}}} \sum_{i=1}^{N_{\text{cause}}} \text{clause\_emb}_i, & \text{if } N_{\text{cause}} > 0 \\
\mathbf{h}_{\text{[CLS]}}, & \text{otherwise (fallback)}
\end{cases}
$$

---

## 視覺化流程

當你執行 `visualize_tsne.py` 時，以下是完整流程：

```python
# 1. 載入模型
model = prompt_bert.from_pretrained('模型路徑')

# 2. 準備資料（所有未標註樣本 + 標註樣本）
all_unlabeled_docs + labeled_docs → MyDataset

# 3. 提取 Embedding
embeddings = extract_embeddings(model, dataloader, device, knn_embedding_mode='cause_clause_mask')
# 內部呼叫 model.get_clause_embeddings() → compute_clause_embeddings()

# 4. t-SNE 降維
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
embeddings_2d = tsne.fit_transform(embeddings)  # (N, 768) → (N, 2)

# 5. 繪圖
plt.scatter(labeled_2d, c='blue', marker='s')       # 藍色方塊 = 標註樣本
plt.scatter(unselected_2d, c='gray', alpha=0.3)     # 灰色小點 = 未被選中
plt.scatter(selected_2d, c=根據準確率著色)           # 綠/黃/紅 = 被選中的偽標籤
```

---

## 總結

### Embedding 提取方式（`cause_clause_mask`）

1. **識別**：找出所有被預測為「原因子句」的子句（P(是) > P(非)）
2. **計算**：對每個原因子句的內容做 mean pooling
3. **增強**：與該子句的 [MASK]_cause embedding 取平均
4. **聚合**：所有原因子句 embedding 取平均；若無原因子句則 fallback 到 [CLS]

### 藍點出現的原因

- Round 1：初始模型，Embedding 空間反映少量標註樣本的特徵
- Round 10：Self-training 後，模型學習了大量偽標籤，Embedding 空間重新分布
- **右側藍點**：部分標註樣本在新空間中與其他樣本的相對位置改變，屬於正常現象

### 進一步分析建議

1. **檢查右側藍點的 doc_ids**：看看它們是否具有共同特徵（如情緒類型、文檔長度等）
2. **對比 Round 1 和 Round 10 的測試集 F1**：如果 F1 持續上升，說明模型確實在進步
3. **觀察偽標籤準確率**：如果 Round 10 的偽標籤平均準確率高於 Round 1，說明 Self-training 有效
