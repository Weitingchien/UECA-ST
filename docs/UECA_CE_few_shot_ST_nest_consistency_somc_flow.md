# UECA_CE_few_shot_ST_nest_consistency_somc.py 流程說明 (論文版)

針對以下指令的執行流程作說明，作為碩士論文方法章節的說明

指令：

    python UECA_CE_few_shot_ST_nest_consistency_somc.py --dataset split10_train1_test1_val1_unlabeled7_disjoint/ --pseudo_selector nest --nest_k 5 --nest_beta 0.1 --nest_m 0.6 --nest_multiplier 5.0 --nest_divergence_mode cause_clause --knn_embedding_mode cause_clause_mask --nest_loss_mode standard --consistency_pseudo --consistency_rule somc --consistency_model_source self_training_best --start_fold 1 --end_fold 10 --training_iter 70 --self_training_rounds 10 --st_training_epochs 3 --savecheckpoint True --test_model_type self_training --seed 60 --batch_size 8 --learning_rate 1e-5 --gamma 0.5 --weight_decay 0.01 --log_pseudo_quality --save_consistency_predictions --retain_pseudo_in_unlabeled

---

## 1. 方法總覽

本實驗是「少樣本作監督式學習 + 多輪自訓練 (Self-Training) + NeST 偽標籤選樣 + SOMC」的十折流程

核心流程可分為四層：

1. 初始監督訓練 (每折 70 iterations)
2. 自訓練偽標籤選樣 (NeST)
3. 一致性過濾 (SOMC)
4. 合併有標記與偽標記資料再訓練，並在驗證集更新最佳模型

最終以 test_model_type=self_training 指定的最佳自訓練模型，在每折測試集上評估並輸出結果

---

## 2. 參數到方法機制的對應

### 2.1 資料與折數控制

- dataset=split10_train1_test1_val1_unlabeled7_disjoint/
  - 每折載入 fold{i}_train.json、fold{i}_val.json、fold{i}_test.json、fold{i}_unlabeled.json
- start_fold=1, end_fold=10
  - 執行 10 折
- seed=60
  - 固定 Python / NumPy / Torch 隨機性，影響 NeST 採樣與訓練可重現性

### 2.2 初始監督訓練

- training_iter=70
  - 每折初始監督訓練 70 個 iteration
- batch_size=8, learning_rate=1e-5, weight_decay=0.01
  - AdamW 優化器配置
- savecheckpoint=True
  - 儲存每折最佳 emotion/cause/pair 模型

### 2.3 自訓練總體設定

- self_training_rounds=10
  - 每折做 10 輪偽標籤→再訓練循環
- st_training_epochs=3
  - 每輪自訓練將合併資料訓練 3 個 epoch
- retain_pseudo_in_unlabeled
  - 偽標記樣本保留在 unlabeled pool，不移除; 後續輪次可再次被評估

### 2.4 偽標籤選樣 (NeST)

- pseudo_selector=nest
  - 使用 NeST 分數採樣 (不是 threshold)
- nest_k=5
  - KNN 鄰居數 k=5
- nest_beta=0.1
  - divergence 中 D_l 權重係數
- nest_m=0.6
  - 跨輪 EMA 平滑係數 (當輪 + 歷史分數)
- nest_multiplier=5.0
  - 每輪目標選樣數 = int(nest_multiplier × 當折 labeled 數量)
- nest_divergence_mode=cause_clause
  - divergence 僅使用「原因句子句分佈」計算
- knn_embedding_mode=cause_clause_mask
  - KNN 特徵使用「原因句子句 + $[MASK]_c$  聚合」embedding

其中 cause_clause_mask 的具體定義如下:

1. 先對每個子句的原因 $[MASK]_c$ 比較 P(是) 與 P(非)
2. 僅保留 P(是) > P(非) 的子句，視為「原因句候選」
3. 對每個候選子句做子句 token 平均池化得到 clause embedding
4. 取該子句 $[MASK]_c$ 位置的 hidden state，與 clause embedding 取平均
   $$
   h_{clause\_mask} = \frac{h_{clause} + h_{[MASK]_c}}{2}
   $$
5. 若一篇文檔有多個候選子句，對其 $h_{clause\_mask}$ 再做平均，得到文檔向量供 KNN 使用
6. 若無任何候選子句，回退使用 [CLS] 向量 (fallback)

而 cause_clause 分佈則是另一條資訊流: 對同一批「原因句候選」的 [P(是), P(非)] 做平均，作為散度計算的輸入分佈；也就是「KNN 用向量特徵、散度用機率分佈」，兩者分工不同。

可直接放在論文中的形式化敘述如下:

令文檔 $x$ 的第 $j$ 個子句為 $c_j$，其$[MASK]_e$為 $m_j^c$。若
$$
P_j^c(\text{是}) > P_j^c(\text{非})
$$
則子句 $c_j$ 被納入原因句候選集合 $\mathcal{C}(x)$。對每個 $c_j \in \mathcal{C}(x)$，先計算子句語意向量
$$
h_j^{clause} = \frac{1}{|c_j|}\sum_{t \in c_j} h_t
$$
再與原因$[MASK]_e$向量做平均得到
$$
h_j^{ccm} = \frac{h_j^{clause} + h_{m_j^c}}{2}
$$
最後將所有候選子句聚合為文檔向量
$$
h_x^{ccm} = \frac{1}{|\mathcal{C}(x)|}\sum_{c_j \in \mathcal{C}(x)} h_j^{ccm}
$$
若 $|\mathcal{C}(x)|=0$，則回退使用 [CLS] 向量作為 $h_x^{ccm}$。

### 2.5 一致性過濾 (根據情緒/原因以及情緒原因組合的最佳模型，對未標註樣本進行拼接，取各自的預測)

- consistency_pseudo
  - 啟用一致性過濾
- consistency_rule=somc
  - 使用 SOMC (子任務協作拼接) 規則，不要求三模型整串全相等
- consistency_model_source=self_training_best
  - 一致性三模型來源:
    - Round 1: 因尚無上一輪，回退到 supervised best 三模型
    - Round >= 2: 改用 self_training_best_emo/cause/pair 三模型
- save_consistency_predictions
  - 每輪輸出三模型預測與 SOMC 拼接結果 JSON

### 2.6 損失函數模式

- nest_loss_mode=standard
  - 自訓練總損失採:

  $$
  L_{total} = L_{sup} + \gamma L_{pseudo}
  $$

- gamma=0.5
  - 偽標籤損失權重 $\gamma$

### 2.7 其他輸出控制

- log_pseudo_quality
  - 若可取得 unlabeled 的對應 GT，會輸出偽標籤錯誤率報告
- test_model_type=self_training
  - 最終測試時優先載入 fold{i}_self_training_best_pair.pth

---

## 3. 資料表示與任務模板

每個子句固定建立三個 $[MASK]$ ，順序為:

1. $[MASK]_e$: 是否情緒句 (是/非)
2. $[MASK]_c$: 是否原因句 (是/非)
3. $[MASK]_p$: 對應情緒句索引 (1..75 或 无)

因此每份文檔會被轉成:

- 輸入: 原子句文字 + 三個 [MASK] + [SEP] 串接
- 監督標籤: 對應三個位置的目標 token id

MyDataset 會同時統計每份文檔的 GT 計數 (emotion/cause/pair)供後續 crf_prompt 評分

---

## 4. 單一 Fold 的完整執行流程

以下以每一折為單位說明

### 4.1 初始化與資料載入

1. 建立 prompt_bert (BertForMaskedLM)
2. 載入 train/val/test/unlabeled 四個 split
3. 若啟用 log_pseudo_quality: 建立 unlabeled 文檔對應 GT mask token map

### 4.2 初始監督訓練  (70 iterations)

1. 在 train 上訓練
2. 每個 iteration 後在 val 上評估 (crf_prompt)
3. 依 val 指標分別更新並儲存:
   - fold{i}_best_emo.pth
   - fold{i}_best_cause.pth
   - fold{i}_best_pair.pth
4. best pair 更新時，同步輸出 val_text_result_best_pair

### 4.3 進入 Self-Training (10 rounds)

每輪流程如下

#### A. 決定選樣器

- 本指令固定 pseudo_selector=nest，所以每輪都走 NeST 分支

#### B. 收集 labeled 與 unlabeled 的統計向量

1. labeled: 提取特徵向量 (cause_clause_mask) 與 cause_clause 分佈
2. unlabeled: 同樣提取特徵向量與 cause_clause 分佈，並先生成 all_pseudo_tokens

#### C. 計算本輪目標選樣數

$$
N_{select}^{target} = \max\left(1, \left\lfloor nest\_multiplier \times N_{labeled} \right\rfloor\right)
$$

本指令 nest_multiplier=5.0，因此 target 與當折 labeled 數量成正比

#### D. NeST divergence 與抽樣

NeST 中，對每個 unlabeled 樣本計算:

$$
D = D_u + \beta D_l
$$

其中:

- $D_u$：未標記分佈與鄰居分佈差異 (KL 組合)
- $D_l$：鄰居間一致性差 (異KL 組合)
- $\beta = 0.1$

跨輪使用 EMA:

$$
\mu^{(t)}(x) = (1-m)\mu^{(t-1)}(x) + mD^{(t)}(x), \quad m=0.6
$$

再將分數轉成採樣權重:

$$
W = \max_x \mu(x), \quad w(x)=W-\mu(x), \quad p(x)=\frac{w(x)}{\sum_x w(x)}
$$

依 $p(x)$ 無放回抽樣得到 selected_indices

#### E. SOMC 一致性過濾

對每個被 NeST 選中的樣本:

1. 三模型分別對同一輸入做推論: emo/cause/pair
2. 在 SOMC 規則下做$[MASK]$拼接:
   - $[MASK]_e$來自 emo 模型
   - $[MASK]_c$cause 模型
   - $[MASK]_p$來自 pair 模型
3. 若拼接結構合法 (長度一致且可按 3 個$[MASK]$切分) 則通過
4. 通過樣本才加入 round_pseudo_labeled_samples

一致性模型來源 self_training_best 的實際行為:

- Round 1: 用 supervised best (因 self-training best 尚不存在)
- Round >= 2: 用 self_training_best_emo/cause/pair

#### F. 偽標籤品質評估與紀錄

若啟用 log_pseudo_quality 且 GT 可用，輸出每輪:

- 偽標籤錯誤率
- 排除「非 非 无」後錯誤率
- 每份文檔逐位比較詳表

#### G. 合併資料再訓練 (3 epochs)

訓練資料 = 原 train + 歷輪 pseudo datasets

本指令 nest_loss_mode=standard，故:

$$
L_{total} = L_{sup} + \gamma L_{pseudo}, \quad \gamma=0.5
$$

每輪結束後在 val 評估，更新 self-training 最佳模型:

- fold{i}_self_training_best_emo.pth
- fold{i}_self_training_best_cause.pth
- fold{i}_self_training_best_pair.pth

### 4.4 Fold 測試評估

每折自訓練完成後:

1. 若 test_model_type=self_training，優先載入 fold{i}_self_training_best_pair.pth
2. 在 test split 以 evaluate_split + crf_prompt 計算 e/c/p 的 P/R/F1
3. 輸出 fold{i}_text_result.txt 與評估檔

---

## 5. 重要數字的來源 (論文解釋可直接引用)

以 log 常見句子為例:

- NeST 選出 865 筆候選樣本 (目標 865)

其來源為：

$$
\text{target}=\left\lfloor 5.0 \times N_{labeled} \right\rfloor
$$

若某折 $N_{labeled}=173$，則 target=865

actual selected 可能小於 target 的唯一常見原因是抽樣機率非零樣本不足，程式會在 select_samples_and_generate_pseudo_labels 內做 min 截斷

---

## 6. 輸出檔案地圖 (你這條指令會產生)

每個實驗資料夾底下主要有:

1. fold{i}_best_emo/cause/pair.pth
2. self_training_models/fold{i}_self_training_best_emo/cause/pair.pth
3. pseudo_results_{timestamp}/
   - fold{i}_pseudo_text_result.txt
   - pseudo_labeled_samples_fold{i}_round{r}.json
   - pseudo_label_evaluation_fold{i}_round{r}.txt
   - consistency_predictions_fold{i}_round{r}.json
   - consistency_rejected_eval_fold{i}_round{r}.json
   - nest_divergence_scores_fold{i}_round{r}.txt
   - nest_divergence_scores_fold{i}_round{r}.json
   - doc_selection_history_fold{i}.txt
4. self_training_results_{timestamp}/
   - self_training_val_results_fold{i}_round{r}.txt
   - fold{i}_self_training_val_text_result_round{r}.txt
5. 最終測試輸出
   - fold{i}_text_result.txt
   - test_evaluation_fold{i}.txt (或 evaluate_split 對應結果檔)
6. 全程主控台日誌
   - run_console_output_{timestamp}.txt
7. 執行時間統計
   - time_log 檔 (含每折監督訓練與 self-training 耗時)

---

## 7. 論文方法章節的敘述

本研究在每個 fold 先以少量標記資料執行 70 iterations 的監督訓練，取得初始最佳模型；其後進行 10 輪自訓練。每輪首先以 NeST 在未標記池中進行候選抽樣: 在表示層面，先找出被預測為原因句的子句，將子句語意向量與該子句 [MASK]_c 向量平均，形成 cause_clause_mask 文檔向量，並以此做 KNN（k=5）；在分佈層面，另以原因句候選的 [P(是), P(非)] 平均得到 cause_clause 分佈，再計算散度分數 $D_u + \beta D_l$（$\beta=0.1$）。之後透過 $m=0.6$ 的 EMA 進行跨輪平滑，最終依機率無放回抽樣，候選數量設為標記集大小的 5 倍。接著對候選樣本套用 SOMC 一致性機制：分別由 emotion/cause/pair 最佳模型輸出後進行[MASK]拼接，僅保留拼接合法樣本作為偽標籤。訓練階段採 standard 損失整合 $L_{total}=L_{sup}+\gamma L_{pseudo}$（$\gamma=0.5$），每輪訓練 3 個 epoch 並於驗證集更新 self-training 最佳模型。最終以 self-training 最佳 pair 模型在測試集報告 Emotion、Cause、Pair 三任務的 Precision、Recall 與 F1

---

## 8. 其他注意事項

1. 本設定下 nest_loss_mode=standard，不是 NeST threshold loss 版本
2. consistency_rule=somc 時，all_equal 不是接受條件
3. consistency_model_source=self_training_best 在 Round 1 會自動回退 supervised best
4. retain_pseudo_in_unlabeled 使樣本可重複被評估，不是單輪即從池中移除

---

如果你要，我可以再幫你產一版「論文圖說版本」，把整個流程整理成一頁式流程圖 (含公式與檔案輸出節點)
