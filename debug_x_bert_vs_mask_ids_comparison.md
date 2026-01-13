# x_bert vs mask_ids 的對比分析

## 問題
inspect_mydataset.py 中第 74 行的：
```python
mask_ids = torch.tensor(x_bert, dtype=torch.int64)
```

與 UECA_CE_few_shot_ST.py 中 extract_ground_truth_mask_tokens 的：
```python
mask_ids = tokenizer.encode_plus(
    mask_full_document,
    return_tensors="pt",
    max_length=512,
    truncation=True,
    pad_to_max_length=True,
)["input_ids"][0]
```

**這兩個是否相同？**

---

## ✅ 答案：是的，它們在數值上完全相同！

### 原因分析

#### 1️⃣ x_bert 的來源

**在 MyDataset 中（UECA_CE_few_shot_ST.py 第 249 行）：**

```python
# 第 1. 構建 mask_full_document 字符串
mask_full_document = mask_full_document + "[MASK] [MASK] [MASK] [SEP]"

# 第 2. 使用 tokenizer.encode_plus 編碼
mask_full_document = \
    self.tokenizer.encode_plus(
        mask_full_document,
        return_tensors="pt",
        max_length=512,
        truncation=True,
        pad_to_max_length=True
    )['input_ids']  # shape: (1, 512)

# 第 3. 提取第一個序列
self.x_bert.append(np.array(mask_full_document[0]))
```

**轉換過程：**
```
mask_full_document (PyTorch 張量，形狀 (1, 512))
    ↓ [0]
mask_full_document[0] (PyTorch 張量，形狀 (512,))
    ↓ np.array()
np.array (numpy 陣列，形狀 (512,))
    ↓ 存入 self.x_bert
最終
    ↓ 取出使用
x_bert (numpy 陣列，形狀 (512,))
```

#### 2️⃣ extract_ground_truth_mask_tokens 中的 mask_ids

**相同的編碼過程（UECA_CE_few_shot_ST.py 第 420-426 行）：**

```python
mask_ids = tokenizer.encode_plus(
    mask_full_document,  # ← 相同的 mask_full_document 字符串
    return_tensors="pt",
    max_length=512,
    truncation=True,
    pad_to_max_length=True,
)["input_ids"][0]  # ← 直接取出 [0]，得到 shape (512,)
```

---

## 📊 對比表

| 項目 | MyDataset.x_bert | extract_ground_truth_mask_tokens.mask_ids |
|------|-----------------|-----------------------------------|
| **建立位置** | MyDataset.__init__() 第 249 行 | extract_ground_truth_mask_tokens() 第 420-426 行 |
| **使用的文本** | mask_full_document（子句 + [MASK] [MASK] [MASK] [SEP]） | 相同的 mask_full_document |
| **編碼方式** | tokenizer.encode_plus(...)[0] | 相同 |
| **tokenizer** | 同一個 BertTokenizer 實例 | 同一個 BertTokenizer 實例 |
| **max_length** | 512 | 512 |
| **truncation** | True | True |
| **pad_to_max_length** | True | True |
| **最終形狀** | numpy (512,) | PyTorch (512,) |
| **數值內容** | ✅ **完全相同** | ✅ **完全相同** |

---

## 🔄 完整流程圖

### MyDataset 中 x_bert 的構建

```
JSON 文件
    ↓
讀取文件和 doc["clauses"]
    ↓
構建 mask_full_document 字符串
    ├─ 每個子句後加 "[MASK] [MASK] [MASK] [SEP]"
    └─ 例子: "1 吸毒后... [MASK] [MASK] [MASK] [SEP] 2 妻子... [MASK] [MASK] [MASK] [SEP] ..."
    ↓
tokenizer.encode_plus(mask_full_document, ...)
    ↓
["input_ids"]
    ↓ (返回 PyTorch 張量)
tensor, shape (1, 512)
    ↓
[0]
    ↓ (取第一個序列)
tensor, shape (512,)
    ↓
np.array()
    ↓ (轉換為 numpy)
numpy array, shape (512,)
    ↓
自.x_bert.append()
    ↓
MyDataset.x_bert (numpy 陣列)
```

### extract_ground_truth_mask_tokens 中 mask_ids 的構建

```
同一個文件中的同一個 doc
    ↓
構建相同的 mask_full_document 字符串
    ├─ 每個子句後加 "[MASK] [MASK] [MASK] [SEP]"
    └─ 內容完全相同
    ↓
tokenizer.encode_plus(mask_full_document, ...)
    ↓
["input_ids"]
    ↓ (返回 PyTorch 張量)
tensor, shape (1, 512)
    ↓
[0]
    ↓ (取第一個序列)
tensor, shape (512,)
    ↓
mask_ids (PyTorch 張量)
```

---

## 💡 為什麼 inspect_mydataset.py 要轉換回 PyTorch？

```python
# 第 74 行（inspect_mydataset.py）
mask_ids = torch.tensor(x_bert, dtype=torch.int64)
```

**原因：**

1. **x_bert 是 numpy 陣列**
   - MyDataset 將其存儲為 numpy array
   - 便於數據加載和批處理

2. **但核心語句需要 PyTorch 張量**
   ```python
   mask_positions = (mask_ids == 103).nonzero(as_tuple=False).view(-1)
   ```
   - `.nonzero()` 是 PyTorch 方法
   - NumPy 的 `.nonzero()` 返回不同格式
   - `.view(-1)` 是 PyTorch 張量方法

3. **轉換是可逆的且無損**
   ```python
   x_bert (numpy)
       ↓ torch.tensor(..., dtype=torch.int64)
   mask_ids (PyTorch)
       ↓ 與 UECA 中的 mask_ids 數值完全相同
   ```

---

## 🎯 驗證方式

### 方式 1：數值驗證

```python
# 如果有同一個文件的同一個樣本
doc_from_json = ...
mask_ids_from_extract = extract_ground_truth_mask_tokens(doc_from_json, tokenizer)
# 提取值，例如 [7478, 3221, 124, ...]

# 在 MyDataset 中找到相同 doc_id 的樣本
x_bert_from_dataset = dataset.x_bert[sample_index]
# 檢查 [MASK] 位置 (103)
mask_positions = np.where(x_bert_from_dataset == 103)[0]

# x_bert_from_dataset[mask_positions] 應該與
# mask_ids_from_extract 相同的位置上有相同的值
```

### 方式 2：使用 --demo-mask-positions 選項

```bash
python debug/inspect_mydataset.py split10/fold1_train.json --demo-mask-positions
```

輸出會顯示：
```
與 extract_ground_truth_mask_tokens 函式對比:
  函式返回: [7478, 3221, 124, ...]
  我們計算: [7478, 3221, 124, ...]
  ✅ 完全一致！
```

---

## 📝 關鍵區別

| 特徵 | MyDataset.x_bert | extract_ground_truth_mask_tokens.mask_ids |
|------|-----------------|-----------------------------------|
| **儲存形式** | numpy array | PyTorch tensor（臨時） |
| **目的** | 訓練數據 | 提取 ground truth 標籤 |
| **使用場景** | DataLoader 批處理 | 單個文件的 token 提取 |
| **數值內容** | ✅ 相同 | ✅ 相同 |

---

## ✨ 總結

### 答案：**是的，它們在數值上完全相同**

1. ✅ 使用相同的編碼方式（tokenizer.encode_plus）
2. ✅ 使用相同的參數（max_length=512, truncation=True 等）
3. ✅ 編碼相同的文本（mask_full_document）
4. ✅ 結果都是 512 個 token IDs

### 唯一的區別

| inspect_mydataset.py | UECA_CE_few_shot_ST.py |
|-----|-----|
| numpy 陣列 | PyTorch 張量 |
| 後來轉換為 PyTorch | 直接使用 PyTorch |

### 為什麼要轉換？

```python
# inspect_mydataset.py 中需要執行 PyTorch 操作
mask_ids = torch.tensor(x_bert, dtype=torch.int64)  # ← 轉換
mask_positions = (mask_ids == 103).nonzero(as_tuple=False).view(-1)  # ← PyTorch 特定方法
```

這樣才能讓 inspect_mydataset.py 完全重現 UECA_CE_few_shot_ST.py 的邏輯！

