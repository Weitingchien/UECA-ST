# mask_positions 語法解釋

## 反白處原始程式碼（第 436 行）

```python
# (mask_ids == 103) 建立布林張量，nonzero 傳回非零索引；view(-1) 攤平成一維
mask_positions = (mask_ids == 103).nonzero(as_tuple=False).view(-1)
```

---

## 語法逐步拆解

### 1️⃣ 第一步：比較操作 `(mask_ids == 103)`

```python
mask_ids = torch.tensor([101, 122, 127, 3299, 127, 3189, 103, 103, 103, 102, ...])

# 第一步：逐元素比較
mask_ids == 103
# 結果: tensor([False, False, False, False, False, False, True, True, True, False, ...])

# 布林張量形狀
result = (mask_ids == 103)
# shape: (512,)
# dtype: torch.bool
```

**說明：**
- `==` 運算符逐元素進行比較
- 103 是 BERT 的 `[MASK]` token ID（可用 `tokenizer.mask_token_id` 取得）
- 結果是**布林值張量**（True/False 對應每一位置）

### 2️⃣ 第二步：找出非零位置 `.nonzero(as_tuple=False)`

```python
# 步驟 1 的結果（布林張量）
bool_tensor = tensor([False, False, False, False, False, False, True, True, True, False, ...])

# 第二步：找非零位置
bool_tensor.nonzero(as_tuple=False)
# 結果: tensor([[6], [7], [8], [22], [23], [24], [36], [37], [38], [56], ...])
#
# 說明：
# - 找出所有值為 True 的位置（位置編號）
# - 返回二維張量：每個非零位置用 [索引] 表示
# - shape: (num_masks, 1)，例如 (30, 1) 表示有 30 個 [MASK]

# 參數 as_tuple=False 表示：
# - 返回結果形狀為 (n, 1) 的二維張量
# - 而非元組形式的一維張量
```

**對比 `as_tuple=True`：**

```python
# as_tuple=True 時的結果
bool_tensor.nonzero(as_tuple=True)
# 結果: (tensor([6, 7, 8, 22, 23, 24, 36, 37, 38, 56, ...]),)
#
# 這是一個元組，包含一個 1D 張量
# 通常用於多維張量的索引化
```

### 3️⃣ 第三步：攤平成一維 `.view(-1)`

```python
# 步驟 2 的結果（二維張量）
two_d = tensor([[6], [7], [8], [22], [23], [24], [36], [37], [38], [56], ...])
# shape: (30, 1)

# 第三步：攤平
two_d.view(-1)
# 結果: tensor([6, 7, 8, 22, 23, 24, 36, 37, 38, 56, ...])
# shape: (30,)

# 說明：
# - .view(-1) 將張量攤平成一維
# - -1 表示「自動計算該維度的大小」
# - 30 * 1 = 30，所以結果是 1D 張量，包含 30 個元素
```

**為什麼需要 `.view(-1)`？**
- `nonzero()` 的結果是二維張量 `(num_masks, 1)`
- 我們需要的是一維張量 `(num_masks,)`
- `.view(-1)` 將 `[[6], [7], [8], ...]` 變成 `[6, 7, 8, ...]`

---

## 完整流程示意

```
輸入 mask_ids：[101, 122, 127, 3299, 127, 3189, 103, 103, 103, 102, ...]
                                                  ↓    ↓    ↓
                                                 [MASK位置]
                
                          ↓ (mask_ids == 103)
                
布林張量：       [False, False, False, ..., False, True, True, True, False, ...]
                                                  ↓     ↓     ↓
                                                 [位置]
                
                  ↓ .nonzero(as_tuple=False)
                
索引張量：       [[6], [7], [8], [22], [23], [24], [36], [37], [38], [56], ...]
                形狀 (30, 1)
                
                  ↓ .view(-1)
                
最終結果：       [6, 7, 8, 22, 23, 24, 36, 37, 38, 56, ...]
                 形狀 (30,)
                 
← mask_positions
```

---

## 實際執行範例

### 完整程式碼

```python
import torch
from transformers import BertTokenizer

# 初始化
tokenizer = BertTokenizer.from_pretrained('./bert-base-chinese')

# 建立一個包含 [MASK] 的例子
mask_ids = torch.tensor([101, 122, 127, 3299, 127, 3189, 103, 103, 103, 102, 
                         123, 7942, 3378, 1139, 2345, 1168, 3736, 7305, 2356, 3173, 
                         833, 1277, 103, 103, 103, 102, 124, 7444, 1288, 702])

# 第一步：比較
bool_result = (mask_ids == 103)
print("Step 1 - Boolean comparison:")
print(f"  bool_result: {bool_result}")
print(f"  shape: {bool_result.shape}")
# 輸出: tensor([False, False, ..., True, True, True, ..., True, True, True, ...])
# 輸出: torch.Size([30])

# 第二步：找非零位置
nonzero_result = bool_result.nonzero(as_tuple=False)
print("\nStep 2 - Nonzero positions:")
print(f"  nonzero_result: {nonzero_result}")
print(f"  shape: {nonzero_result.shape}")
# 輸出: tensor([[6], [7], [8], [22], [23], [24]])
# 輸出: torch.Size([6, 1])

# 第三步：攤平
mask_positions = nonzero_result.view(-1)
print("\nStep 3 - Flatten to 1D:")
print(f"  mask_positions: {mask_positions}")
print(f"  shape: {mask_positions.shape}")
# 輸出: tensor([6, 7, 8, 22, 23, 24])
# 輸出: torch.Size([6])

# 用 mask_positions 索引提取 token IDs
print("\nExtract tokens at mask positions:")
full_ids = torch.tensor([101, 122, 127, 3299, 127, 3189, 7478, 3221, 124, 102, ...])
gt_tokens = full_ids[mask_positions]
print(f"  gt_tokens: {gt_tokens}")
# 輸出: tensor([7478, 3221, 124, 7478, 7478, 3187])
```

### Print 輸出示例

```
Step 1 - Boolean comparison:
  bool_result: tensor([False, False, False, False, False, False,  True,  True,  True, False,
         False, False, False, False, False, False, False, False, False, False,
         False, False,  True,  True,  True, False, False, False, False, False])
  shape: torch.Size([30])

Step 2 - Nonzero positions:
  nonzero_result: tensor([[ 6],
                          [ 7],
                          [ 8],
                          [22],
                          [23],
                          [24]])
  shape: torch.Size([6, 1])

Step 3 - Flatten to 1D:
  mask_positions: tensor([ 6,  7,  8, 22, 23, 24])
  shape: torch.Size([6])

Extract tokens at mask positions:
  gt_tokens: tensor([7478, 3221,  124, 7478, 7478, 3187])
```

---

## 在 inspect_mydataset.py 中的應用

`inspect_mydataset.py` 的 `preview_mask_positions()` 函式實現：

```python
def preview_mask_positions(x_bert: np.ndarray, label: np.ndarray, tokenizer: BertTokenizer, limit: int = 10) -> None:
    """Show [MASK] token positions and corresponding labels."""
    mask_id = tokenizer.mask_token_id
    
    # 核心語法：找出所有 [MASK] 的位置
    # 在 numpy 中等價於：np.where(x_bert == mask_id)[0]
    mask_positions = np.where(x_bert == mask_id)[0]
    
    print(f"mask positions count={mask_positions.size}") 
    # 輸出: mask positions count=30
    
    if mask_positions.size == 0:
        return
    
    sample_positions = mask_positions[:limit]
    print(f"  first positions: {sample_positions.tolist()}") 
    # 輸出: first positions: [6, 7, 8, 22, 23, 24, 36, 37, 38, 56]
    
    label_values = label[sample_positions]
    tokens = tokenizer.convert_ids_to_tokens(label_values.tolist()) 
    print(f"  label ids at positions: {label_values.tolist()}")
    # 輸出: label ids at positions: [7478, 7478, 3187, 7478, 7478, 3187, 7478, 7478, 3187, 7478]
    
    print(f"  tokens at positions: {tokens}") 
    # 輸出: tokens at positions: ['非', '非', '无', '非', '非', '无', '非', '非', '无', '非']
```

---

## PyTorch vs NumPy 對比

### PyTorch 方式（UECA_CE_few_shot_ST.py 第 436 行）

```python
import torch

mask_ids = torch.tensor([...], dtype=torch.int64)
mask_positions = (mask_ids == 103).nonzero(as_tuple=False).view(-1)
```

### NumPy 等價方式（inspect_mydataset.py 中使用）

```python
import numpy as np

x_bert = np.array([...], dtype=np.int64)
mask_positions = np.where(x_bert == 103)[0]
```

**比較：**

| 特徵 | PyTorch | NumPy |
|------|---------|-------|
| 比較 | `(tensor == 103)` | `(array == 103)` |
| 找位置 | `.nonzero().view(-1)` | `np.where(...)[0]` |
| 結果形狀 | 1D tensor | 1D array |
| 索引方式 | `tensor[mask_positions]` | `array[mask_positions]` |

---

## 常見用途

### ✅ 使用情景 1：提取 [MASK] 處的標籤

```python
# 找出所有 [MASK] token 的位置
mask_positions = (mask_ids == 103).nonzero(as_tuple=False).view(-1)

# 從 y_bert 中提取這些位置的正確 token（ground truth）
gt_tokens = y_bert[mask_positions]
# → [7478, 3221, 124, 7478, 7478, 3187, 7478, 7478, 3187]
```

### ✅ 使用情景 2：驗證模型預測

```python
# 從模型預測中取出 [MASK] 位置的預測
pred_tokens = model_output[mask_positions]

# 與 ground truth 比較
accuracy = (pred_tokens == gt_tokens).float().mean()
```

### ✅ 使用情景 3：建立訓練标籤

```python
# 初始化所有位置為 -100（被忽略）
label = torch.full_like(mask_ids, -100)

# 只在 [MASK] 位置設定真實標籤
label[mask_positions] = y_bert[mask_positions]
# → [-100, -100, ..., 7478, 3221, 124, ..., -100, -100, ...]
```

---

## 重點總結

| 語法 | 意義 |
|------|------|
| `(mask_ids == 103)` | 建立布林張量，True 表示 [MASK] 位置 |
| `.nonzero(as_tuple=False)` | 找出所有 True 的位置索引（返回 (n,1) 的二維張量） |
| `.view(-1)` | 將 (n,1) 攤平成 (n,)（一維張量） |
| 合併結果 | `mask_positions` = [6, 7, 8, 22, 23, 24, ...] |
| **會 print 什麼？** | **一維張量，包含所有 [MASK] token 的位置索引** |

