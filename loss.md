$$L_{\text{labeled}} = 
\begin{cases}
\dfrac{1}{N_{\text{labeled}}} \sum_{i=1}^{N_{\text{labeled}}} \ell(x_i^{(l)}, y_i^{(l)}) & N_{\text{labeled}} > 0 \\
0 & \text{otherwise}
\end{cases}$$



$$L_{\text{pseudo}} = 
\begin{cases}
\dfrac{1}{N_{\text{pseudo}}} \sum_{j=1}^{N_{\text{pseudo}}} \ell(x_j^{(p)}, y_j^{(p)}) & N_{\text{pseudo}} > 0 \\
0 & \text{otherwise}
\end{cases}$$


$$\text{total\_loss} = L_{\text{labeled}} + \gamma \cdot L_{\text{pseudo}}$$


$$\text{loss}_p = -\frac{1}{|\mathcal{B}p|}\sum{i\in\mathcal{B}p} \frac{1}{|\mathcal{M}i|} \sum{t\in\mathcal{M}i} \log p\theta!\big(y{i,t} \mid x_i\big)$$

## 實例驗算：Pseudo-Label Loss ($L_{\text{pseudo}}$)

以下根據 Log 中的一筆實際數據 (Loss = 0.123070) 進行數學驗算與解析。

### 1. 數學公式定義

該 Loss 為 **Masked Language Modeling (MLM)** 的 Cross-Entropy Loss。

$$ \mathcal{L} = - \frac{1}{N} \sum_{i=1}^{N} \log(P(y_i | x)) $$

#### 符號定義：
*   $\mathcal{L}$ : 計算出的 Loss 值 (本例為 0.123070)。
*   $N$ : 該筆資料中被 Mask 的 Token 總數 (有效 Mask Label 數量)。
*   $i$ : 遍歷所有被 Mask 的位置索引，從 1 到 $N$。
*   $x$ : 輸入的模型序列 (包含 `[MASK]` token)。
*   $y_i$ : 第 $i$ 個 Mask 位置的**目標 Token** (即 Pseudo-label 的真實 ID)。
*   $P(y_i | x)$ : 模型預測在第 $i$ 個位置出現 Token $y_i$ 的**機率** (Probability)，範圍為 $[0, 1]$。
*   $\log$ : 自然對數 (Natural Logarithm, $\ln$)。

### 2. 數據來源解析

根據 Log 紀錄，此筆資料的統計資訊如下：

*   **總 Mask 數量 ($N$)**: 27 (即 `mask_label_p != -100` 的位置總數)。
*   **目標 Token 序列 ($y_i$)**: 包含 27 個位置的 Pseudo-label。

### 3. 計算過程展開 (完整無省略)

#### 以下偽標籤樣本以及對應的loss來自於prompt_ECPE_few_shot_ST_2025_11_22_14_25_27_f1-1_i50_lr1e-5_bs8_wd0.01_bert-base-chinese_th0.9_maskemotion_gamma0.5_reg1e-4_st1_ste3_seed42_retain_pseudo_CE資料夾內的run_console_output_2025_11_22_14_25_27.txt的內容(ctrl+c搜尋loss_p即可查找到)


將 Log 中實際記錄的機率值代入公式，展開所有 27 個項：

$$
\begin{aligned}
\mathcal{L} &= \frac{1}{27} \sum_{i=1}^{27} -\log(P(y_i | x)) \\
&= \frac{1}{27} \bigg[ \\
&\quad \underbrace{-\log(0.993398)}_{0.006624} \quad (\text{Idx 6: 非}) \\
&\quad + \underbrace{-\log(0.999934)}_{0.000066} \quad (\text{Idx 7: 非}) \\
&\quad + \underbrace{-\log(0.999941)}_{0.000059} \quad (\text{Idx 8: 无}) \\
&\quad + \underbrace{-\log(0.999998)}_{0.000002} \quad (\text{Idx 24: 非}) \\
&\quad + \underbrace{-\log(0.999988)}_{0.000012} \quad (\text{Idx 25: 非}) \\
&\quad + \underbrace{-\log(0.999992)}_{0.000008} \quad (\text{Idx 26: 无}) \\
&\quad + \underbrace{-\log(0.999663)}_{0.000337} \quad (\text{Idx 39: 非}) \\
&\quad + \underbrace{-\log(0.999993)}_{0.000007} \quad (\text{Idx 40: 非}) \\
&\quad + \underbrace{-\log(0.999965)}_{0.000035} \quad (\text{Idx 41: 无}) \\
&\quad + \underbrace{-\log(0.999999)}_{0.000001} \quad (\text{Idx 55: 非}) \\
&\quad + \underbrace{-\log(0.998731)}_{0.001270} \quad (\text{Idx 56: 非}) \\
&\quad + \underbrace{-\log(0.999979)}_{0.000021} \quad (\text{Idx 57: 无}) \\
&\quad + \underbrace{-\log(0.999886)}_{0.000114} \quad (\text{Idx 68: 非}) \\
&\quad + \underbrace{-\log(0.999706)}_{0.000294} \quad (\text{Idx 69: 非}) \\
&\quad + \underbrace{-\log(0.999869)}_{0.000131} \quad (\text{Idx 70: 无}) \\
&\quad + \underbrace{-\log(0.999969)}_{0.000031} \quad (\text{Idx 79: 非}) \\
&\quad + \underbrace{-\log(0.255918)}_{1.362900} \quad (\text{Idx 80: 是}) \leftarrow \text{信心較低} \\
&\quad + \underbrace{-\log(0.145826)}_{1.925339} \quad (\text{Idx 81: 7}) \leftarrow \text{信心較低} \\
&\quad + \underbrace{-\log(0.987635)}_{0.012442} \quad (\text{Idx 91: 是}) \\
&\quad + \underbrace{-\log(0.999132)}_{0.000869} \quad (\text{Idx 92: 非}) \\
&\quad + \underbrace{-\log(0.999501)}_{0.000499} \quad (\text{Idx 93: 无}) \\
&\quad + \underbrace{-\log(1.000000)}_{0.000000} \quad (\text{Idx 106: 非}) \\
&\quad + \underbrace{-\log(0.999979)}_{0.000021} \quad (\text{Idx 107: 非}) \\
&\quad + \underbrace{-\log(0.999982)}_{0.000018} \quad (\text{Idx 108: 无}) \\
&\quad + \underbrace{-\log(0.988621)}_{0.011444} \quad (\text{Idx 124: 非}) \\
&\quad + \underbrace{-\log(0.999957)}_{0.000043} \quad (\text{Idx 125: 非}) \\
&\quad + \underbrace{-\log(0.999707)}_{0.000293} \quad (\text{Idx 126: 无}) \\
\bigg] \\
&= \frac{3.322880}{27} \\
&= 0.123070
\end{aligned}
$$

### 4. 數值意義反推 (Reverse Engineering)

從上述詳細數據可以看出：
1.  **絕大多數位置** (如 Idx 6, 7, 8 等) 的預測機率都非常高 (>99%)，對應的 Loss 極小 (<0.001)。
2.  **少數位置** (如 Idx 80, 81) 的預測機率較低 (約 25% 和 14%)，貢獻了絕大部分的 Loss (1.36 和 1.92)。
    *   這兩個位置 (Idx 80, 81) 對應的是 "是" 和 "7"，可能因為上下文較難判斷，導致模型信心不足。
3.  最終的平均 Loss **0.123070** 是由這些高信心與低信心的預測平均而來的結果。

這證明了 Log 中的數值計算是完全正確的，且反映了模型對不同位置的掌握程度差異。