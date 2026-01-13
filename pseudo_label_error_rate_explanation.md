# 偽標籤錯誤率計算說明

## 圖表：各輪偽標籤錯誤率比較 (`pseudo_label_error_rate_comparison.png`)

### Y 軸：錯誤率 (Error Rate) 計算方式

錯誤率定義為：**在自訓練過程中被選中的文檔中，有多少比例的文檔其偽標籤預測不完全正確**。

#### 數學公式

$$
\text{Error Rate} = \frac{\text{錯誤文檔數}}{\text{總選中文檔數}} \times 100\%
$$

其中：

$$
\text{錯誤文檔數} = \text{總選中文檔數} - \text{完全正確文檔數}
$$

#### 完全正確的定義

一個文檔被視為「完全正確」，當且僅當該文檔中 **所有 3 個 [MASK] 位置的預測都正確**：

$$
\text{完全正確} \iff \forall i \in \{1, 2, 3\}: \hat{y}_i = y_i
$$

其中：
- $\hat{y}_i$：第 $i$ 個 [MASK] 位置的模型預測值
- $y_i$：第 $i$ 個 [MASK] 位置的真實標籤

#### 3 個 [MASK] 位置說明

在 UECA 模型中，每個文檔有 3 個需要預測的 [MASK] 位置：

| [MASK] 位置 | 說明 |
|-------------|------|
| [MASK]₁ | 情緒子句位置 (Emotion Clause Position) |
| [MASK]₂ | 原因子句位置 (Cause Clause Position) |
| [MASK]₃ | 情緒類別 (Emotion Category) |

#### 計算範例

假設在第 $r$ 輪自訓練中：
- 總共選中了 $N = 368$ 個文檔加入訓練
- 其中有 $C = 27$ 個文檔的 3 個 [MASK] 預測全部正確

則：

$$
\text{錯誤文檔數} = N - C = 368 - 27 = 341
$$

$$
\text{Error Rate} = \frac{341}{368} \times 100\% \approx 92.7\%
$$

### X 軸：自訓練輪次 (Round)

X 軸表示自訓練的第 $r$ 輪，$r \in \{1, 2, ..., 10\}$。

每一輪會：
1. 使用當前模型對未標記數據進行預測
2. 根據選擇策略（閾值法或 NeST）選取高信心樣本
3. 將選中的樣本及其偽標籤加入訓練集
4. 重新訓練模型

### 圖例說明

| 標籤 | 選擇策略 | 說明 |
|------|---------|------|
| `th0.9_mask_emotion` | 閾值法 (threshold=0.9) | 以情緒預測信心度 ≥ 0.9 作為選擇條件 |
| `th0.9_mask_cause` | 閾值法 (threshold=0.9) | 以原因預測信心度 ≥ 0.9 作為選擇條件 |
| `nest_k5_emotion` | NeST (k=5) | 使用 NeST 演算法，基於情緒預測的 KL 散度選擇最近鄰 |
| `nest_k5_cause` | NeST (k=5) | 使用 NeST 演算法，基於原因預測的 KL 散度選擇最近鄰 |

### 資料來源

錯誤率數據來自各實驗資料夾中的 `pseudo_label_accuracy_summary.txt` 檔案，該檔案由 `utils/analyze_pseudo_label_accuracy.py` 生成。
