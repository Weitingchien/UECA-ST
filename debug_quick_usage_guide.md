# 快速使用指南：inspect_mydataset.py 新功能

## 📌 三種使用方式

### 1️⃣ 查看資料集摘要（原有功能）

```bash
cd /mnt/d/GithubRepo/UECA
python debug/inspect_mydataset.py split10/fold1_train.json
```

**輸出內容：**
- 資料集大小
- x_bert、y_bert、label、mask_label、gt_emotion、gt_cause、gt_pair 的形狀和前幾個值

---

### 2️⃣ 檢查單個樣本（原有功能）

```bash
python debug/inspect_mydataset.py split10/fold1_train.json --index 6
```

**輸出內容：**
- 樣本編號 6 的完整詳情
- x_bert、y_bert 的解碼文字
- label 和 mask_label 的非 -100 部分
- [MASK] 位置分析

---

### 3️⃣ 演示 mask_positions 語法（新功能）🆕

```bash
python debug/inspect_mydataset.py split10/fold1_train.json --demo-mask-positions
```

**這個模式完整展示：**
1. 文件載入和 pairs 拆解
2. mask_full_document 和 full_document 的構建
3. token 編碼（mask_ids 和 full_ids）
4. **核心語句執行**：
   ```python
   mask_positions = (mask_ids == 103).nonzero(as_tuple=False).view(-1)
   ```
5. 逐步分析 (mask_ids == 103)、.nonzero()、.view(-1) 各步驟
6. ground truth tokens 提取
7. 與 extract_ground_truth_mask_tokens() 函式的驗證

---

## 📊 輸出示例（--demo-mask-positions）

```
================================================================================
演示 extract_ground_truth_mask_tokens 完整流程
================================================================================

✅ 選中文件: doc_id=212, 子句數=3, pairs=[[1, 1], [3, 1]]

拆解 pairs:
  pos (情緒子句編號): [1, 3]
  cause (原因子句編號): [1, 1]

構建文件:
  full_document (前 100 字): 1吸毒后不仅对妻子拳打脚踢甚至持刀闯入娘家威胁丈母娘是是 1 [SEP]...
  mask_full_document (前 100 字): 1吸毒后不仅对妻子拳打脚踢甚至持刀闯入娘家威胁丈母娘[MASK] [MASK] [MASK] [SEP]...

編碼為 token IDs:
  mask_ids 形狀: torch.Size([512])
  full_ids 形狀: torch.Size([512])

執行核心語句:
  mask_positions = (mask_ids == 103).nonzero(as_tuple=False).view(-1)

  第一步：(mask_ids == 103)
    形狀: torch.Size([512])
    前 50 個: [False, False, ..., True, True, True, ...]

  第二步：.nonzero(as_tuple=False)
    形狀: torch.Size([9, 1])  ← (非零個數, 1) 的二維張量
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

## 🔍 key 變化點

### 變更 1：PyTorch 張量操作（preview_mask_positions）

**原來的 NumPy 方式：**
```python
mask_positions = np.where(x_bert == 103)[0]
```

**現在的 PyTorch 方式（完全一致 UECA）：**
```python
mask_ids = torch.tensor(x_bert, dtype=torch.int64)
mask_positions = (mask_ids == 103).nonzero(as_tuple=False).view(-1)
```

### 變更 2：新增 demo 函式（demo_extract_mask_tokens）

完整演示 `extract_ground_truth_mask_tokens` 的所有內部步驟。

### 變更 3：更新 parse_args()

新增 `--demo-mask-positions` 選項

### 變更 4：更新 main()

檢查 `args.demo_mask_positions` 並執行相應的演示函式

---

## 📁 相關文檔

- `debug_explanation_mask_positions_syntax.md` - 完整語法解釋
- `debug_explanation_as_tuple_view.md` - as_tuple 和 view(-1) 詳解
- `debug_explanation_why_3_1_shape.md` - 為什麼是 (3, 1) 形狀
- `debug_explanation_numel_method.md` - .numel() 方法說明
- `debug_inspect_mydataset_update_guide.md` - 詳細更新說明

---

## ✅ 驗證清單

使用 `--demo-mask-positions` 時會自動驗證：

- [x] mask_ids 正確編碼
- [x] (mask_ids == 103) 產生布林張量
- [x] .nonzero(as_tuple=False) 返回二維張量 (n, 1)
- [x] .view(-1) 攤平成一維 (n,)
- [x] full_ids[mask_positions] 正確索引提取
- [x] 最終結果與 extract_ground_truth_mask_tokens() 函式完全一致 ✅

