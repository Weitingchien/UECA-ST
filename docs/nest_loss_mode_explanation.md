# NeST Loss Mode 計算方式說明

本文檔詳細說明 `nest_loss_mode` 參數的兩種模式（`standard` 和 `nest`）在計算偽標籤損失時的不同實作方式。

---

## 參數定義

在 `UECA_CE_few_shot_ST_nest.py` 中：

```python
parser.add_argument('--nest_loss_mode', type=str, default='standard',
                    choices=['standard', 'nest'],
                    help='NeST loss 計算模式: standard=所有偽標籤都計入, nest=只有信心>γ才計入')
parser.add_argument('--nest_loss_threshold', type=float, default=0.9,
                    help='nest 模式下的信心閾值 (對應論文的 γ，預設 0.9)的nest_loss_threshold: 信心閾值 γ (僅在 nest 模式下使用，預設 0.9)
```

---

## 模式 1: `standard` (標準模式)

### 概念說明

**所有偽標籤樣本的所有 [MASK] 位置都計入損失**，不使用閥值過濾。

### 數學公式

對於一個批次中的偽標籤樣本集合 $\mathcal{P}$，損失計算為：

$$
\mathcal{L}_{\text{pseudo}} = \frac{1}{|\mathcal{P}|} \sum_{x_j \in \mathcal{P}} \mathcal{L}_{\text{CE}}(f(x_j; \theta), \tilde{y}_j)
$$

其中：
- $\mathcal{P}$：批次中的偽標籤樣本集合
- $|\mathcal{P}|$：偽標籤樣本數量 (`pseudo_mask.sum()`)
- $x_j$：第 $j$ 個偽標籤樣本的輸入序列
- $\tilde{y}_j$：第 $j$ 個樣本的偽標籤序列（僅 [MASK] 位置有值，其餘為 -100）
- $f(x_j; \theta)$：模型對輸入 $x_j$ 的預測輸出
- $\mathcal{L}_{\text{CE}}$：Cross-Entropy 損失函數

**展開 Cross-Entropy 計算**（針對單個樣本的所有 [MASK] 位置）：

$$
\mathcal{L}_{\text{CE}}(f(x_j; \theta), \tilde{y}_j) = -\frac{1}{M_j} \sum_{i \in \mathcal{M}_j} \log p_\theta(\tilde{y}_j^{(i)} | x_j)
$$

其中：
- $\mathcal{M}_j$：樣本 $x_j$ 中所有 [MASK] 位置的集合
- $M_j = |\mathcal{M}_j|$：[MASK] 位置的數量
- $\tilde{y}_j^{(i)}$：第 $i$ 個 [MASK] 位置的偽標籤 token id
- $p_\theta(\tilde{y}_j^{(i)} | x_j)$：模型對該位置預測為 $\tilde{y}_j^{(i)}$ 的機率

### 程式碼實作

```python
# 位置: UECA_CE_few_shot_ST_nest.py, 行 3399-3402
if opt.nest_loss_mode == 'standard':
    # 標準方式: 所有偽標籤都計入 loss
    loss_p, logits_p = model(x_p.cuda() if use_gpu else x_p, 
                             mask_label_p.cuda() if use_gpu else mask_label_p)
    loss_p = loss_p / pseudo_mask.sum()
```

**關鍵說明**：
1. `model(x_p, mask_label_p)` 會自動計算 Cross-Entropy loss，忽略 `mask_label_p` 中值為 `-100` 的位置
2. 回傳的 `loss_p` 是所有樣本、所有有效 [MASK] 位置的 **總損失之和**
3. `loss_p / pseudo_mask.sum()` 除以批次中的偽標籤**樣本數量**，得到每個樣本的平均損失

---

## 模式 2: `nest` 

### 概念說明

**只有模型對偽標籤預測信心度超過閾值 γ 的 [MASK] 位置才計入損失**。這是 NeST 論文的核心改進，通過過濾低信心預測來提升偽標籤質量。

### 數學公式

對應 NeST 論文公式 (5)：

$$
\mathcal{L}_{\text{pseudo}} = \frac{1}{|\mathcal{P}|} \sum_{x_j \in \mathcal{P}} \sum_{i \in \mathcal{M}_j} \mathbb{1}\left\{ p_\theta(\tilde{y}_j^{(i)} | x_j) > \gamma \right\} \cdot \left( -\log p_\theta(\tilde{y}_j^{(i)} | x_j) \right)
$$

**分步說明**：

1. **計算預測機率**（針對每個 [MASK] 位置）：
   $$
   p_\theta(\tilde{y}_j^{(i)} | x_j) = \text{softmax}(f(x_j; \theta))_{\tilde{y}_j^{(i)}}
   $$

2. **應用信心度過濾**（指示函數）：
   $$
   \mathbb{1}\left\{ p_\theta(\tilde{y}_j^{(i)} | x_j) > \gamma \right\} = 
   \begin{cases}
   1, & \text{if } p_\theta(\tilde{y}_j^{(i)} | x_j) > \gamma \\
   0, & \text{otherwise}
   \end{cases}
   $$

3. **只對信心度高的位置計算 Cross-Entropy**：
   $$
   \mathcal{L}_{\text{CE}}^{\text{filtered}}(x_j, \tilde{y}_j) = \frac{1}{N_j^{\text{confident}}} \sum_{i \in \mathcal{M}_j^{\text{confident}}} \left( -\log p_\theta(\tilde{y}_j^{(i)} | x_j) \right)
   $$
   
   其中：
   - $\mathcal{M}_j^{\text{confident}} = \{ i \in \mathcal{M}_j : p_\theta(\tilde{y}_j^{(i)} | x_j) > \gamma \}$：通過信心度篩選的 [MASK] 位置
   - $N_j^{\text{confident}} = |\mathcal{M}_j^{\text{confident}}|$：通過篩選的位置數量

4. **對所有樣本取平均**：
   $$
   \mathcal{L}_{\text{pseudo}} = \frac{1}{|\mathcal{P}|} \sum_{x_j \in \mathcal{P}} \mathcal{L}_{\text{CE}}^{\text{filtered}}(x_j, \tilde{y}_j)
   $$

### 程式碼實作

```python
# 位置: UECA_CE_few_shot_ST_nest.py, 行 3395-3398
if opt.nest_loss_mode == 'nest':
    # NeST 論文的 threshold 過濾方式: 只有信心 > γ 的位置才計入 loss
    loss_p = compute_nest_threshold_loss(model, x_p, mask_label_p, 
                                         opt.nest_loss_threshold, use_gpu)
```

**`compute_nest_threshold_loss` 函數實作細節**（行 810-875）：

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
    
    # 回傳平均 loss (對應公式中的 1/|P| Σ)
    return ce_losses.mean()
```

---

## 總結對比

| 特性 | `standard` 模式 | `nest` 模式 |
|------|----------------|------------|
| **處理方式** | 所有 [MASK] 位置都計入 | 只計算信心度 > γ 的位置 |
| **公式** | $\mathcal{L} = \frac{1}{\|\mathcal{P}\|} \sum_{x_j} \mathcal{L}_{\text{CE}}(f(x_j), \tilde{y}_j)$ | $\mathcal{L} = \frac{1}{\|\mathcal{P}\|} \sum_{x_j} \sum_i \mathbb{1}\\{p > \gamma\\} \cdot (-\log p)$ |
| **優點** | 簡單直接，利用所有偽標籤 | 過濾低質量預測，提升偽標籤質量 |
| **缺點** | 可能被錯誤偽標籤誤導 | 可能過度保守，丟棄部分有用訊號 |
| **適用場景** | 偽標籤質量較高時 | 偽標籤質量參差不齊時（NeST 推薦） |
| **計算成本** | 較低（直接調用 model 內建 loss） | 較高（需手動計算 softmax 和過濾） |

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
