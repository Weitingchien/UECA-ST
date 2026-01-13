# inspect_mydataset.py 更新說明

## 更新內容

`inspect_mydataset.py` 已更新，現在支持演示 `extract_ground_truth_mask_tokens` 的完整流程，並使用完全一致的 PyTorch 張量操作。

---

## 核心改進

### 1. `preview_mask_positions()` 函式

#### 更新前
- 使用 NumPy 方式：`np.where(x_bert == 103)[0]`
- 不能完全重現 UECA 中的邏輯

#### 更新後
- 使用完全一致的 PyTorch 張量操作
- **完整重現 UECA_CE_few_shot_ST.py 第 436 行的寫法**：
  ```python
  mask_positions = (mask_ids == 103).nonzero(as_tuple=False).view(-1)
  ```

#### 逐步演示
```python
# 第一步：轉換為 PyTorch 張量
mask_ids = torch.tensor(x_bert, dtype=torch.int64)

# 第二步：執行完全一致的語句
mask_positions = (mask_ids == 103).nonzero(as_tuple=False).view(-1)

# 第三步：分析結果
print(f"mask_positions 形狀: {mask_positions.shape}")  # (n,)
print(f"mask_positions 內容: {mask_positions.tolist()}")
print(f"元素個數: {mask_positions.numel()}")
```

### 2. 新增 `demo_extract_mask_tokens()` 函式

完整演示 `extract_ground_truth_mask_tokens` 的所有步驟：

#### 演示內容
1. **載入 JSON 檔案**
   - 選擇包含 pairs 的文件

2. **拆解 pairs**
   - 顯示 pos（情緒子句編號）和 cause（原因子句編號）

3. **構建文件**
   - `mask_full_document`：用 `[MASK]` 取代答案
   - `full_document`：包含正確答案

4. **編碼為 token IDs**
   - 使用完全一致的 `tokenizer.encode_plus()`
   - `mask_ids` 和 `full_ids` 的編碼方式

5. **執行核心語句**
   - 展示 `(mask_ids == 103)` 產生布林張量
   - 展示 `.nonzero(as_tuple=False)` 返回二維張量 (n, 1)
   - 展示 `.view(-1)` 攤平成一維 (n,)

6. **提取 ground truth tokens**
   - 使用索引 `full_ids[mask_positions]` 提取正確答案

7. **驗證結果**
   - 與 `extract_ground_truth_mask_tokens()` 函式的返回值比對
   - 確認結果完全一致

---

## 使用方法

### 方式 1：查看 MyDataset 摘要（原有功能）

```bash
python debug/inspect_mydataset.py split10/fold1_train.json
```

輸出：
- 資料集大小
- 各陣列的形狀和前幾個值

### 方式 2：檢查單個樣本（原有功能）

```bash
python debug/inspect_mydataset.py split10/fold1_train.json --index 6
```

輸出：
- 樣本的 x_bert、y_bert、label、mask_label
- 解碼後的文字
- [MASK] 位置分析

### 方式 3：演示 mask_positions 語法（新功能）🆕

```bash
python debug/inspect_mydataset.py split10/fold1_train.json --demo-mask-positions
```

**這個模式會：**
1. 自動選擇第一個包含 pairs 的文件
2. 完整重現 `extract_ground_truth_mask_tokens` 的邏輯
3. 逐步展示 PyTorch 張量操作
4. 驗證結果的正確性

#### 預期輸出範例

```
================================================================================
演示 extract_ground_truth_mask_tokens 完整流程
================================================================================

✅ 選中文件: doc_id=212, 子句數=3, pairs=[[1, 1], [3, 1]]

拆解 pairs:
  pos (情緒子句編號): [1, 3]
  cause (原因子句編號): [1, 1]

構建文件:
  full_document (前 100 字): 1吸毒后不仅对妻子拳打脚踢甚至持刀闯入娘家威胁丈母娘是是 1 [SEP] 2妻子忍无可忍非是 1 [SEP] 3愤而报警是非 无 [SEP]...
  mask_full_document (前 100 字): 1吸毒后不仅对妻子拳打脚踢甚至持刀闯入娘家威胁丈母娘[MASK] [MASK] [MASK] [SEP] 2妻子忍无可忍[MASK] [MASK] [MASK] [SEP] 3愤而报警[MASK] [MASK] [MASK] [SEP]...

編碼為 token IDs:
  mask_ids 形狀: torch.Size([512])
  mask_ids 前 50 個: [101, 122, 1429, 3681, 1400, 679, 788, 2190, 1988, 2094, ..., 103, 103, 103, ...]
  full_ids 形狀: torch.Size([512])
  full_ids 前 50 個: [101, 122, 1429, 3681, 1400, 679, 788, 2190, 1988, 2094, ..., 7478, 3221, 124, ...]

執行核心語句:
  mask_positions = (mask_ids == 103).nonzero(as_tuple=False).view(-1)

  第一步：(mask_ids == 103)
    形狀: torch.Size([512])
    前 50 個: [False, False, False, ..., False, True, True, True, ...]

  第二步：.nonzero(as_tuple=False)
    形狀: torch.Size([9, 1])  ← (非零個數, 1)
    前 10 個: [[6], [7], [8], [22], [23], [24], [36], [37], [38]]

  第三步：.view(-1)  ← 攤平成一維
    形狀: torch.Size([9])  ← 一維
    內容: [6, 7, 8, 22, 23, 24, 36, 37, 38]

提取 ground truth tokens:
  gt_tokens = full_ids[mask_positions]
  形狀: torch.Size([9])
  token IDs: [7478, 3221, 124, 7478, 7478, 3187, 7478, 3221, 124]
  解碼文字: ['非', '是', '3', '非', '非', '无', '非', '是', '3']

與 extract_ground_truth_mask_tokens 函式對比:
  函式返回: [7478, 3221, 124, 7478, 7478, 3187, 7478, 3221, 124]
  我們計算: [7478, 3221, 124, 7478, 7478, 3187, 7478, 3221, 124]
  ✅ 完全一致！
```

---

## 對應 UECA_CE_few_shot_ST.py 的邏輯

### extract_ground_truth_mask_tokens 函式（第 362-449 行）

```python
def extract_ground_truth_mask_tokens(doc, tokenizer):
    # 檢查 pairs
    if "pairs" not in doc or doc.get("pairs") is None:
        return None
    
    # 拆解 pairs
    pairs = doc.get("pairs")
    pos, cause = zip(*pairs) if pairs else ([], [])
    
    # 構建 full_document 和 mask_full_document
    # ...（構建邏輯）
    
    # 編碼為 token IDs
    mask_ids = tokenizer.encode_plus(mask_full_document)[]["input_ids"][0]  # PyTorch 張量
    full_ids = tokenizer.encode_plus(full_document)[]["input_ids"][0]       # PyTorch 張量
    
    # 核心語句：找出所有 [MASK] 的位置
    mask_positions = (mask_ids == 103).nonzero(as_tuple=False).view(-1)
    
    # 檢查是否找到 [MASK]
    if mask_positions.numel() == 0:
        return None
    
    # 提取 ground truth tokens
    gt_tokens = full_ids[mask_positions]
    
    # 轉回 NumPy
    return gt_tokens.cpu().numpy().astype(np.int64)
```

### inspect_mydataset.py 的 demo_extract_mask_tokens 函式

完全重現上述邏輯，逐步展示每一步的中間結果。

---

## 驗證步驟

### 自動驗證
`demo_extract_mask_tokens()` 會自動比較：
- `extract_ground_truth_mask_tokens()` 函式的返回值
- `demo` 函式中手動計算的結果

結果顯示 `✅ 完全一致！` 表示邏輯正確。

### 手動驗證
如果想進一步驗證，可以：
1. 選擇特定的 doc_id
2. 修改 `demo_extract_mask_tokens()` 中的文件選擇邏輯
3. 手動對比各步驟的輸出

---

## 技術細節

### PyTorch 張量操作流程

```
輸入 mask_ids（一維張量）
  ↓ shape: (512,)
  
進行比較 == 103
  ↓ 布林張量 shape: (512,)
  
.nonzero(as_tuple=False)
  ↓ 二維張量 shape: (9, 1)
    [[6], [7], [8], [22], [23], [24], [36], [37], [38]]
  
.view(-1)
  ↓ 一維張量 shape: (9,)
    [6, 7, 8, 22, 23, 24, 36, 37, 38]
    
輸出 mask_positions ✅
```

### 為什麼是 (n, 1) 而不是 (n,)？

- `.nonzero(as_tuple=False)` 返回二維結構
- 因為原始張量是一維的，每個坐標只需 1 個數字
- 所以結果是 (非零個數, 1) = (9, 1)
- 需要 `.view(-1)` 才能攤平成一維

---

## 常見問題

### Q: 為什麼一定要用 `as_tuple=False`？

A: 如果用 `as_tuple=True`，會返回元組格式，無法直接用於索引操作。

```python
# as_tuple=True 的結果（❌ 無法索引）
result = bool_tensor.nonzero(as_tuple=True)
# (tensor([6, 7, 8]),)  # 元組，不能用 full_ids[result]

# as_tuple=False 的結果（✅ 可以索引）
result = bool_tensor.nonzero(as_tuple=False).view(-1)
# tensor([6, 7, 8])  # 張量，可以用 full_ids[result]
```

### Q: 為什麼一定要 `.view(-1)`？

A: `.nonzero(as_tuple=False)` 返回二維張量 (n, 1)，用來索引時需要一維 (n,)。

```python
# 不用 .view(-1) 的結果（❌ 維度不匹配）
result = bool_tensor.nonzero(as_tuple=False)  # shape: (9, 1)
gt_tokens = full_ids[result]  # ❌ TypeError!

# 用 .view(-1) 的結果（✅ 維度匹配）
result = bool_tensor.nonzero(as_tuple=False).view(-1)  # shape: (9,)
gt_tokens = full_ids[result]  # ✅ 成功
```

---

## 相關文件

- `UECA_CE_few_shot_ST.py` - 主要實現
- `debug_explanation_mask_positions_syntax.md` - mask_positions 完整解釋
- `debug_explanation_as_tuple_view.md` - as_tuple 和 view(-1) 詳解
- `debug_explanation_why_3_1_shape.md` - 為什麼是 (3, 1) 形狀

