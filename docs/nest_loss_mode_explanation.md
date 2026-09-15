# NeST Loss Mode 計算方式說明

本文檔詳細說明 `nest_loss_mode` 參數在目前 [UECA_CE_few_shot_ST_nest_consistency_somc_v2.py](UECA_CE_few_shot_ST_nest_consistency_somc_v2.py) 中的實作方式，特別是：

1. 監督式損失 $L_{sup}$ 的正確定義
2. 偽標籤損失 $L_{st}$ 在 `standard` 與 `nest` 模式下的差異
3. self-training 階段總損失如何由 $L_{sup}$ 與 $L_{st}$ 組成

---

## 參數定義

在 [UECA_CE_few_shot_ST_nest_consistency_somc_v2.py](UECA_CE_few_shot_ST_nest_consistency_somc_v2.py) 中：

```python
parser.add_argument('--nest_loss_mode', type=str, default='nest',
                                        choices=['standard', 'nest', 'nest_dynamic_gamma_round0'],
                                        help='NeST loss 計算模式: standard=所有偽標籤都計入, nest=只有信心>γ才計入, nest_dynamic_gamma_round0=搭配 consistency_all_equal_round0_only 時第1輪使用 gamma=0.1')
parser.add_argument('--nest_loss_threshold', type=float, default=0.9,
                                        help='nest 模式下的信心閾值 (對應論文的 γ，預設 0.9)')
```

---

## 目前程式中的 $L_{sup}$ 是什麼

如果要和目前的程式實作精確對應，$L_{sup}$ 應該理解成：

- 學生模型對 labeled 樣本的有效 [MASK] 位置
- 與真實 token 標籤之間的平均 Cross-Entropy loss

### 1. 最貼近程式碼的 batch-level 定義

對於目前一個 self-training mini-batch 中的 labeled 子集合 $\mathcal{B}_l$，監督式損失可寫為：

$$
L_{sup}^{(\mathcal{B})}(\theta_s)
=
-\frac{1}{|M_l^{(\mathcal{B})}|}
\sum_{(x_i, y_i) \in \mathcal{B}_l}
\sum_{m \in M(x_i)}
\log p_{\theta_s}(y_{i,m} \mid x_i, m)
$$

其中：

- $\mathcal{B}_l$
    - 目前 mini-batch 中的 labeled 樣本集合
- $M(x_i)$
    - 樣本 $x_i$ 中所有有效 [MASK] 位置的集合
- $|M_l^{(\mathcal{B})}|$
    - 當前 labeled mini-batch 中所有有效 [MASK] 位置的總數
- $y_{i,m}$
    - 樣本 $x_i$ 在位置 $m$ 的真實 token 標籤
- $p_{\theta_s}(y_{i,m} \mid x_i, m)$
    - 學生模型在位置 $m$ 對真實答案 token 的預測機率

這個寫法比資料集層級版本更貼近目前程式，因為程式實際上是以 mini-batch 方式訓練與更新參數

### 2. 若寫成資料集層級，也可以這樣表示

$$
L_{sup}(\theta_s)
=
-\frac{1}{|M_l|}
\sum_{(x_i, y_i) \in D_L}
\sum_{m \in M(x_i)}
\log p_{\theta_s}(y_{i,m} \mid x_i, m)
$$

其中：

- $D_L$ 為 labeled dataset
- $|M_l|$ 為所有 labeled 文件中有效 [MASK] 位置的總數

要特別注意：

- $|M_l|$ 不是單一文件的 [MASK] 數量
- 而是整個 labeled 集合或當前 labeled batch 中的有效 [MASK] 總數

### 3. HuggingFace BertForMaskedLM

目前 `prompt_bert.forward(...)` 的實作是：

```python
def forward(self, x_bert, labels):
        output = self.bert(x_bert, labels=labels)
        loss, logits = output.loss, output.logits
        return loss, logits
```

這裡直接呼叫 HuggingFace `BertForMaskedLM` 的內建 masked language modeling loss。它的計算方式是：

1. 對 `labels != -100` 的位置計算 Cross-Entropy
2. 對 `labels == -100` 的位置完全忽略
3. 最後回傳所有有效位置的平均 loss

因此，程式中的 `loss_l` 正是上面公式所表示的「有效 [MASK] 位置平均 Cross-Entropy」

### 4. labeled 樣本的真實 target 是怎麼建立的

在 `MyDataset` 中，程式先建立完整文件的 token 序列 `full_document`，再建立只在 [MASK] 位置保留真實答案、其他位置設為 `-100` 的 `mask_labels`：

```python
labels = full_document.masked_fill(mask_full_document != 103, -100)
mask_labels = full_document.masked_fill(mask_label_full_document != 103, -100)
```

其中和監督式損失直接對應的是：

- `mask_labels`

因為只有它保留了有效 [MASK] 位置上的真實答案 token id

也就是說：

- 非 [MASK] 位置不參與 loss
- 每個有效 [MASK] 位置都會對應一個真實 token label
- 這些位置包含情緒、原因與配對三種類型的 [MASK]

所以目前程式中的 $L_{sup}$ 並不是只算 emotion 或 cause，而是：

- 同時對情緒 [MASK]
- 原因 [MASK]
- 配對 [MASK]

共同計算平均 Cross-Entropy

### 5. self-training 階段的 `loss_l` 在哪裡對應到 $L_{sup}$

在 self-training 內層訓練迴圈中：

```python
if labeled_mask.sum() > 0:
        x_l = x_bert[labeled_mask]
        mask_label_l = mask_label[labeled_mask]
        loss_l, _ = model(x_l.cuda() if use_gpu else x_l, mask_label_l.cuda() if use_gpu else mask_label_l)
else:
        loss_l = 0
```

這裡的：

- `x_l`
    - 就是目前 mini-batch 中的 labeled 輸入
- `mask_label_l`
    - 就是目前 mini-batch 中 labeled 樣本在有效 [MASK] 位置上的真實 token 標籤
- `loss_l`
    - 就是目前程式中實際計算出來的 $L_{sup}$

因此，若要在論文中直接對應程式碼，最精確的一句話是：

> 在目前的 `UECA_CE_few_shot_ST_nest_consistency_somc_v2.py` 中，$L_{sup}$ 就是對 labeled mini-batch 的 `mask_label_l` 所對應之所有有效 [MASK] 位置，計算平均 Cross-Entropy loss 的結果，也就是程式中的 `loss_l`

---


## 模式 1: `nest` 

### 概念說明

**只有模型對偽標籤預測信心度超過閾值 γ 的 [MASK] 位置才計入損失**。沿用了既有 semi-supervised / self-training 文獻中的做法，其中 Sohn et al. (2020) 是 FixMatch

因此 NeST 在學生模型訓練階段仍採用帶有 confidence threshold 的 pseudo-label loss

### 數學公式

若要**精確對應目前 `UECA_CE_few_shot_ST_nest_consistency_somc_v2.py` 的實作**，`nest` 模式下的 pseudo loss 更適合寫成下式，而不要直接寫成以 $|\mathcal{P}|$ 為分母的 sample-level 簡寫：

$$
\mathcal{L}_{\text{pseudo}}^{(\mathcal{B})}
=
\begin{cases}
\dfrac{1}{|M_{p,\gamma}^{(\mathcal{B})}|}
\sum\limits_{(x_j, \hat{y}_j) \in \mathcal{B}_p}
\sum\limits_{i \in M(x_j)}
\mathbb{1}\left\{ p_\theta(\hat{y}_j^{(i)} \mid x_j) > \gamma \right\}
\left( -\log p_\theta(\hat{y}_j^{(i)} \mid x_j) \right), & |M_{p,\gamma}^{(\mathcal{B})}| > 0 \\
0, & |M_{p,\gamma}^{(\mathcal{B})}| = 0
\end{cases}
$$

其中：

- $\mathcal{B}_p$
    - 目前 mini-batch 中的 pseudo-labeled 樣本集合
- $M(x_j)$
    - 樣本 $x_j$ 中所有有效 pseudo [MASK] 位置的集合，也就是同時滿足 `x_bert == 103` 且 `mask_label != -100` 的位置
- $M_{p,\gamma}^{(\mathcal{B})}$
    - 當前 pseudo mini-batch 中，所有同時滿足「有效 pseudo [MASK] 位置」且「模型對對應偽標籤的預測機率大於 $\gamma$」的位置集合
- $|M_{p,\gamma}^{(\mathcal{B})}|$
    - 也就是目前程式在 `nest` 模式下實際作為平均分母的數量，也就是通過 threshold 的有效 pseudo token 數

若只是想對照論文 Section 3.2 的概念，可以把它理解為「對 unlabeled loss 加上一個 thresholding function」；但若要和目前程式逐行對齊，分母必須寫成通過 threshold 的 token 數，而不是 $|\mathcal{P}|$。

**分步說明**：

1. **計算預測機率**（針對每個 [MASK] 位置）：
   $$
    p_\theta(\hat{y}_j^{(i)} | x_j) = \text{softmax}(f(x_j; \theta))_{\hat{y}_j^{(i)}}
   $$

2. **應用信心度過濾**（指示函數）：
   $$
    \mathbb{1}\left\{ p_\theta(\hat{y}_j^{(i)} | x_j) > \gamma \right\} = 
   \begin{cases}
    1, & \text{if } p_\theta(\hat{y}_j^{(i)} | x_j) > \gamma \\
   0, & \text{otherwise}
   \end{cases}
   $$

3. **只對信心度高的位置計算 Cross-Entropy**：
   $$
    \mathcal{L}_{\text{CE}}^{\text{filtered}}(x_j, \hat{y}_j) = \frac{1}{N_j^{\text{confident}}} \sum_{i \in \mathcal{M}_j^{\text{confident}}} \left( -\log p_\theta(\hat{y}_j^{(i)} | x_j) \right)
   $$
   
   其中：
    - $\mathcal{M}_j^{\text{confident}} = \{ i \in \mathcal{M}_j : p_\theta(\hat{y}_j^{(i)} | x_j) > \gamma \}$：通過信心度篩選的 [MASK] 位置
   - $N_j^{\text{confident}} = |\mathcal{M}_j^{\text{confident}}|$：通過篩選的位置數量

4. **在目前 v2 程式中，最後不是對 pseudo 樣本數取平均，而是直接對所有通過 threshold 的有效 token 取平均**：
    $$
    \mathcal{L}_{\text{pseudo}}^{(\mathcal{B})}
    =
    \frac{1}{|M_{p,\gamma}^{(\mathcal{B})}|}
     \sum_{(x_j, \hat{y}_j) \in \mathcal{B}_p}
    \sum_{i \in M(x_j)}
     \mathbb{1}\left\{ p_\theta(\hat{y}_j^{(i)} \mid x_j) > \gamma \right\}
     \left( -\log p_\theta(\hat{y}_j^{(i)} \mid x_j) \right)
    $$
    也就是說，對應目前程式的實際分母是 `confident_token_count`，不是 pseudo sample 數 $|\mathcal{P}|$。

### 程式碼實作

```python
if opt.nest_loss_mode in ('nest', 'nest_dynamic_gamma_round0'):
    loss_p, valid_tokens_batch, confident_tokens_batch = compute_nest_threshold_loss(
        model, x_p, mask_label_p, opt.nest_loss_threshold, use_gpu, return_stats=True
    )
```

**`compute_nest_threshold_loss` 函數實作細節**：

```python
def compute_nest_threshold_loss(model, x_bert, mask_label, threshold, use_gpu):
    """
    實作論文公式(5)中的 ℓ_st = 𝟙{[f(x_j; θ_s)]_{ỹ_j} > γ} × ℓ_sup
    """
    device = torch.device('cuda' if use_gpu else 'cpu')
    
    # Step 1: 模型前向傳播
    outputs = model.bert(x_bert, labels=None)
    logits = outputs.logits  # (batch, 512, vocab_size)
    
    # Step 2: 找出所有有效 [MASK] 位置
    mask_positions = (x_bert == 103)  # [MASK] token id = 103
    valid_positions = mask_positions & (mask_label != -100)
    
    # Step 3: 提取有效位置的 logits 和 targets
    valid_logits = logits[valid_positions]    # (num_valid, vocab_size)
    valid_targets = mask_label[valid_positions]  # (num_valid,)
    
    # Step 4: 計算 softmax 機率
    probs = F.softmax(valid_logits, dim=-1)  # (num_valid, vocab_size)
    
    # 取得每個位置對應偽標籤的預測機率: p_θ(ỹ_j^(i) | x_j)
    target_probs = probs.gather(dim=1, index=valid_targets.unsqueeze(1)).squeeze(1)
    
    # Step 5: 應用 threshold 過濾 (對應公式中的指示函數 𝟙{...})
    confident_mask = target_probs > threshold  # (num_valid,)
    
    if not confident_mask.any():
        return torch.tensor(0.0, requires_grad=True, device=device)
    
    # Step 6: 只對信心度高的位置計算 Cross-Entropy
    confident_probs = target_probs[confident_mask]  # (num_confident,)
    ce_losses = -torch.log(confident_probs + 1e-10)  # -log p_θ(...)
    
    # 回傳平均 loss (實際分母是通過 threshold 的有效 pseudo token 數)
    return ce_losses.mean()
```

---

## self-training 階段總損失如何由 $L_{sup}$ 與 $L_{st}$ 組成

在目前 v2 程式的 self-training 迴圈中，總損失組合如下：

```python
if opt.nest_loss_mode in ('nest', 'nest_dynamic_gamma_round0'):
        total_loss = current_round_gamma * loss_l + (1 - current_round_gamma) * loss_p
else:
        total_loss = loss_l + current_round_gamma * loss_p
```

這表示：

### 1. `nest` / `nest_dynamic_gamma_round0` 模式

$$
L_{total}
=
\lambda L_{sup} + (1-\lambda)L_{st}
$$

其中：

- $L_{sup}$ 對應 `loss_l`
- $L_{st}$ 對應 `loss_p`
- $\lambda$ 對應 `current_round_gamma`

### 2. `standard` 模式

$$
L_{total}
=
L_{sup} + \gamma L_{st}
$$

所以如果論文文字要完全對應目前程式，必須分清楚：

- 你若寫 $L_{total}=\lambda L_{sup}+(1-\lambda)L_{st}$
    - 這對應的是 `nest` 類模式
- 你若寫 `standard` 模式
    - 目前程式實作其實是 $L_{sup} + \gamma L_{st}$

---

## 總結對比

| 特性 | `standard` 模式 | `nest` 模式 |
|------|----------------|------------|
| **處理方式** | 所有 pseudo [MASK] 位置都計入 | 只計算信心度 > γ 的 pseudo [MASK] 位置 |
| **公式** | $\mathcal{L} = \frac{1}{\|\mathcal{P}\|} \sum_{x_j} \mathcal{L}_{\text{CE}}(f(x_j), \hat{y}_j)$ | $\mathcal{L} = \frac{1}{\|\mathcal{P}\|} \sum_{x_j} \sum_i \mathbb{1}\\{p > \gamma\\} \cdot (-\log p)$ |
| **優點** | 簡單直接，利用所有偽標籤 | 過濾低質量預測，提升偽標籤質量 |
| **缺點** | 可能被錯誤偽標籤誤導 | 可能過度保守，丟棄部分有用訊號 |

---

## 一句話總結目前程式中的 $L_{sup}$

若只保留一句最精確、最能對應程式碼的說明，可以寫成：

> 在 `UECA_CE_few_shot_ST_nest_consistency_somc_v2.py` 中，$L_{sup}$ 是學生模型對 labeled mini-batch 中所有有效 [MASK] 位置之真實 token 標籤所計算的平均 Cross-Entropy loss，程式中對應的變數是 `loss_l`，其 targets 來自 `MyDataset` 建立的 `mask_label`

---

## 參數建議

- **`nest_loss_mode`**:
  - 使用 `'nest'` 時搭配 NeST 選擇器 (`--pseudo_selector nest`)
  - 使用 `'standard'` 可搭配任何選擇器

- **`nest_loss_threshold`** (僅 `nest` 模式):
  - 預設 `0.9`（NeST 論文建議值）
  - 數值越高越嚴格（保留更少位置，但質量更高）
  - 範圍建議：0.7 ~ 0.95

---

## 實驗範例

```bash
# Standard 模式（傳統 Self-training）
python UECA_CE_few_shot_ST_nest.py \
    --pseudo_selector threshold \
    --nest_loss_mode standard \
    --threshold 0.9

# NeST 模式（論文推薦）
python UECA_CE_few_shot_ST_nest.py \
    --pseudo_selector nest \
    --nest_loss_mode nest \
    --nest_loss_threshold 0.9 \
    --nest_k 5 --nest_beta 0.1 --nest_m 0.6
```
