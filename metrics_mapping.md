# UECA 情緒分類評估公式與程式碼對照（6 類 micro，排除 null）

本文件對應 `compute_weighted_f1_from_dump.py` 目前的評估邏輯（6 類 micro，排除 null=6）到數學公式，方便在 VS Code（已裝 Markdown+Math）或 Notebook 中預覽與核對。

> 預覽提示：此檔使用 LaTeX（KaTeX/MathJax）語法，直接在 VS Code Markdown 預覽即可渲染。

---

## 符號定義

- 樣本數：$N$
- 第 $i$ 筆資料的真實標籤與預測標籤：$y_i,\ \hat{y}_i$，其中 $i=1,\dots,N$
- 評估類別集合（排除 null 類）：$S=\{0,1,2,3,4,5\}$

對照程式碼：
- `y_true_np`, `y_pred_np` → $y_i,\ \hat{y}_i$；`len(y_true_np)` → $N$
- `labels6 = [0,1,2,3,4,5]` → $S=\{0,1,2,3,4,5\}$

---

## 逐類別計數（TP/FP/FN）

對每個類別 $c\in S$，定義：
$$
\begin{aligned}
TP_c &= \sum_{i=1}^{N} \mathbf{1}\{y_i=c \land \hat{y}_i=c\},\\
FP_c &= \sum_{i=1}^{N} \mathbf{1}\{y_i\neq c \land \hat{y}_i=c\},\\
FN_c &= \sum_{i=1}^{N} \mathbf{1}\{y_i=c \land \hat{y}_i\neq c\}.
\end{aligned}
$$

對照程式碼（在 `for lab in labels6:` 迴圈中）：
- `tp = sum((y_true_np == lab) & (y_pred_np == lab))` → $TP_c$
- `fp = sum((y_true_np != lab) & (y_pred_np == lab))` → $FP_c$
- `fn = sum((y_true_np == lab) & (y_pred_np != lab))` → $FN_c$

---

## Micro（微平均）彙總

將 6 類的 TP/FP/FN 加總：
$$
TP=\sum_{c\in S} TP_c,\quad
FP=\sum_{c\in S} FP_c,\quad
FN=\sum_{c\in S} FN_c.
$$

對照程式碼：
- `tp_sum += tp`, `fp_sum += fp`, `fn_sum += fn`

---

## Precision / Recall / F1（micro）

$$
P=\frac{TP}{TP+FP},\quad
R=\frac{TP}{TP+FN},\quad
F1=\frac{2PR}{P+R}.
$$

對照程式碼：
- `p = tp_sum / (tp_sum + fp_sum + 1e-8)` → $P$（以 $10^{-8}$ 作為數值平滑）
- `r = tp_sum / (tp_sum + fn_sum + 1e-8)` → $R$
- `f1 = 2 * p * r / (p + r + 1e-8)` → $F1$

---

## Accuracy（僅在真值屬於 6 類子集上）

$$
\mathrm{Acc}
=\frac{\sum_{i=1}^{N}\mathbf{1}\{y_i\in S\}\cdot \mathbf{1}\{\hat{y}_i=y_i\}}
{\sum_{i=1}^{N}\mathbf{1}\{y_i\in S\}}.
$$

對照程式碼：
- `mask_true6 = np.isin(y_true_np, labels6)` 過濾真值屬於 $S$ 的樣本
- `acc6 = mean(y_pred_np[mask_true6] == y_true_np[mask_true6])` → $\mathrm{Acc}$

---