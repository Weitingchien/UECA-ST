# UECA Consistency SOMC: standard vs nest Loss Mode 說明

本文檔專門說明你目前兩支程式的差異：

- 舊版: `UECA_CE_few_shot_ST_nest_consistency_somc.py` 常用 `--nest_loss_mode standard`
- v2 版: `UECA_CE_few_shot_ST_nest_consistency_somc_v2.py` 常用 `--nest_loss_mode nest`

本檔已依 AAAI 2023 論文 *Neighborhood-Regularized Self-Training for Learning with Few Labels*（arXiv:2301.03726）的 TeX 原始碼核對 self-training loss 相關公式。

---

## 1. 論文原始公式（已核對）

以下是論文在 `03-prelim.tex` 中對 self-training 的核心定義（Eq. 4 與其文字定義）：

$$
\min_{\theta_s}\;\lambda\,\mathcal{L}_{sup}(\theta_s)
+(1-\lambda)\,\mathbb{E}_{x_j\in\hat{\mathcal{X}}_u}
\ell_{st}\!
\left(f(x_j;\theta_s),\tilde{y}_j\right)
$$

並且論文文字直接定義：

$$
\ell_{st}
=
\mathbf{1}\!\left\{\left[f(x_j;\theta_s)\right]_{\tilde{y}_j}>\gamma\right\}
\cdot
\ell_{sup}.
$$

也就是說，論文版本的 unlabeled loss 本質上是「帶信心門檻 $\gamma$ 的過濾式訓練」。

---

## 2. 你專案中的兩種模式

在你的程式中，兩種模式可理解為：

- `standard`: 對有效 [MASK] 全部計算偽標籤 CE loss（不做 $\gamma$ 過濾）
- `nest`: 對有效 [MASK] 先做信心門檻，僅保留 $p_\theta(\tilde{y}|x)>\gamma$ 的位置計算 loss

令第 $j$ 筆偽標籤樣本有效 [MASK] 集合為 $\mathcal{M}_j$。

### 2.1 standard 模式

$$
\mathcal{L}_{pseudo}^{standard}
=
\frac{1}{|\mathcal{P}|}
\sum_{x_j\in\mathcal{P}}
\left(
-\frac{1}{|\mathcal{M}_j|}
\sum_{i\in\mathcal{M}_j}
\log p_\theta\left(\tilde{y}_j^{(i)}\mid x_j\right)
\right).
$$

### 2.2 nest 模式

先定義高信心位置集合：

$$
\mathcal{M}_j^{conf}
=
\left\{i\in\mathcal{M}_j\;\middle|\;p_\theta\left(\tilde{y}_j^{(i)}\mid x_j\right)>\gamma\right\}.
$$

再計算：

$$
\mathcal{L}_{pseudo}^{nest}
=
\frac{1}{|\mathcal{P}|}
\sum_{x_j\in\mathcal{P}}
\left(
-\frac{1}{\max\left(1,|\mathcal{M}_j^{conf}|\right)}
\sum_{i\in\mathcal{M}_j^{conf}}
\log p_\theta\left(\tilde{y}_j^{(i)}\mid x_j\right)
\right).
$$

也可寫成指示函數形式：

$$
\mathcal{L}_{pseudo}^{nest}
=
\frac{1}{|\mathcal{P}|}
\sum_{x_j\in\mathcal{P}}
\sum_{i\in\mathcal{M}_j}
\mathbf{1}\!\left[p_\theta\left(\tilde{y}_j^{(i)}\mid x_j\right)>\gamma\right]
\cdot
\left(-\log p_\theta\left(\tilde{y}_j^{(i)}\mid x_j\right)\right).
$$

---

## 3. 與 AAAI 2023 原式的對照結論

1. 論文原式的 $\ell_{st}$ 明確包含門檻指示函數，因此概念上更接近你的 `nest` 模式。
2. 你的 `standard` 是實務上常見的「不加門檻」版本，訊號更多但對 noisy pseudo label 也更敏感。
3. 在你目前 consistency 流程下：
	- consistency 是樣本級過濾
	- nest loss 是 token 級過濾
	兩層過濾可同時降低噪音。

---

## 4. 兩支程式的「實作級」loss 差異（重點）

除了 `standard` / `nest` 這個模式差異外，
`UECA_CE_few_shot_ST_nest_consistency_somc.py` 與 `UECA_CE_few_shot_ST_nest_consistency_somc_v2.py`
在 self-training 損失的正規化方式有關鍵不同：

### 4.1 舊版 (`consistency_somc.py`)：有額外除法

在 self-training 迴圈中：

1. labeled loss 會再除以 `labeled_mask.sum()`
2. standard pseudo loss 會再除以 `pseudo_mask.sum()`

可抽象成（以 batch 記）：

$$
L_l^{old} = \frac{\bar{L}_l}{N_l},
\qquad
L_p^{old,standard} = \frac{\bar{L}_p}{N_p},
$$

其中：

- $\bar{L}_l,\bar{L}_p$ 是模型回傳的平均 CE（對有效 token 平均）
- $N_l$ / $N_p$ 分別是該 batch 的 labeled / pseudo 樣本數

因此總損失為：

$$
L_{total}^{old}=
\begin{cases}
\gamma L_l^{old} + (1-\gamma)L_p^{old} & (\text{nest mode})\\
L_l^{old} + \gamma L_p^{old} & (\text{standard mode})
\end{cases}
$$

### 4.2 v2 (`consistency_somc_v2.py`)：移除額外除法

v2 不再做上述兩個額外的 `/ N_l`、`/ N_p`。

即：

$$
L_l^{v2}=\bar{L}_l,
\qquad
L_p^{v2,standard}=\bar{L}_p.
$$

總損失仍維持同一個加權結構：

$$
L_{total}^{v2}=
\begin{cases}
\gamma L_l^{v2} + (1-\gamma)L_p^{v2} & (\text{nest mode})\\
L_l^{v2} + \gamma L_p^{v2} & (\text{standard mode})
\end{cases}
$$

### 4.3 這個差異的影響

1. 舊版在 batch size 較大時，loss 量級會被額外縮小（尤其 pseudo 分支）。
2. v2 的量級更貼近 HuggingFace `BertForMaskedLM` 回傳 loss 的原始定義（有效 token 平均 CE）。
3. 相同 `gamma` 下，v2 的梯度強度通常會比舊版更大一些。
4. 這也是為什麼同樣參數下，舊版與 v2 的收斂速度/最佳點可能不同。

### 4.4 補充

- `compute_nest_threshold_loss(...)` 在兩支程式中實作一致（先取 target prob，再用門檻過濾，最後平均）。
- 主要差異不在 threshold 公式本身，而在 self-training 主迴圈對 `loss_l`/`loss_p` 是否再除一次樣本數。

---

## 5. 實務建議

1. 若 early rounds 偽標籤品質波動大，優先用 `nest`（例如 $\gamma=0.9$）。
2. 若 pseudo label 已穩定且想增加學習訊號，可做 `standard` 對照。
3. 建議固定其他參數，只替換 `nest_loss_mode` 與 $\gamma$，做公平 ablation。

---

## 6. 一句話總結

- 舊版常用 `standard`：偏「吃滿偽標籤訊號」。
- v2 常用 `nest`：偏「信心過濾後再學習」，與論文中的 $\ell_{st}$ 定義更一致。
