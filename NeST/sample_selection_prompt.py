import numpy as np
import torch
import faiss


def _to_numpy(array_like):
    if isinstance(array_like, torch.Tensor):
        return array_like.detach().cpu().numpy() #detach()移除梯度
    return np.asarray(array_like)




# 假設 labeled_probs 是 4 筆標註樣本的 soft label（每筆 3 類），形狀 [4, 3]:
    #labeled_probs = np.array([
    #    [0.8, 0.1, 0.1],  # 標註樣本 0
    #    [0.2, 0.7, 0.1],  # 標註樣本 1
    #    [0.3, 0.3, 0.4],  # 標註樣本 2
    #    [0.1, 0.2, 0.7],  # 標註樣本 3
    #])
    
    # KNN 找到兩個未標註樣本各自的 2 個鄰居:
        #neighbor_indices = np.array([
            #[0, 2],  # 無標註樣本 0 的鄰居 → 標註 0、2
            #[1, 3],  # 無標註樣本 1 的鄰居 → 標註 1、3
        #])
        # 形狀 [n_unlabeled=2, k=2]
    # 做 labeled_probs[neighbor_indices] 時，NumPy 會把上面索引套到 labeled_probs 的「列」，回傳:
        #labeled_neighbors = np.array([
        #[[0.8, 0.1, 0.1],  # 無標註 0 的鄰居 0 (原標註 0)
        # [0.3, 0.3, 0.4]], # 無標註 0 的鄰居 1 (原標註 2)

        #[[0.2, 0.7, 0.1],  # 無標註 1 的鄰居 0 (原標註 1)
        # [0.1, 0.2, 0.7]], # 無標註 1 的鄰居 1 (原標註 3)
        #])
        # 形狀 [2, 2, 3] = [n_unlabeled, k, n_classes]
        
def _compute_divergence(labeled_probs, unlabeled_probs, neighbor_indices, beta, return_details=False):
    """
    計算未標記樣本與其鄰居之間的散度分數。
    
    Args:
        labeled_probs: 標記樣本的機率分佈，形狀 [n_labeled, n_classes]
        unlabeled_probs: 未標記樣本的機率分佈，形狀 [n_unlabeled, n_classes]
        neighbor_indices: 每個未標記樣本的 k 個最近鄰索引，形狀 [n_unlabeled, k]
        beta: D_l 的權重係數
        return_details: 是否返回詳細的計算資訊
        
    Returns:
        如果 return_details=False:
            divergence: 最終散度分數 D_u + beta * D_l，形狀 [n_unlabeled]
        如果 return_details=True:
            (divergence, details_dict) 其中 details_dict 包含:
                - D_u: 每個未標記樣本的 D_u 值
                - D_l: 每個未標記樣本的 D_l 值
                - score_u_per_neighbor: 每個未標記樣本對每個鄰居的 KL 分數，形狀 [n_unlabeled, k]
                - score_l_per_neighbor: 鄰居一致性分數，形狀 [n_unlabeled, k]
                - y_bar: 鄰居平均分佈，形狀 [n_unlabeled, n_classes]
    """
    # 依照鄰居索引取得對應的鄰居標籤分佈，形狀為 [n_unlabeled, k, n_classes]
    labeled_neighbors = labeled_probs[neighbor_indices]
    # 對每個鄰居分佈與未標註樣本預測分佈計算 KL(labeled || unlabeled)
    # unlabeled_probs[:, np.newaxis]: 形狀擴展為 [n_unlabeled, 1, n_classes]，在第2軸插入一個長度為1的維度，才能和 labeled_neighbors 做廣播運算
    # 廣播運算: 運算時會被視為複製k份，變成 [n_unlabeled, k, n_classes]和labeled_neighbors做逐元素除法、log等操作
    score_u = np.log((1e-10 + labeled_neighbors) / (1e-10 + unlabeled_probs[:, np.newaxis])) * (1e-10 + labeled_neighbors)
    # 將 KL 值在類別與鄰居維度加總，得到每個無標註樣本的鄰居散度 D_u，-1: 對類別加總，-2: 對K個鄰居加總，D_u[0]: 第0個未標註樣本的D_u分數
    D_u = np.sum(score_u, axis=(-1, -2))
    # 將鄰居 soft label 平均後得到 ȳ，計算鄰居的平均分佈
    y_bar = np.mean(labeled_neighbors, axis=1)
    # 計算 KL(ȳ || labeled) 計算鄰居分佈彼此的散度
    score_l = np.log((1e-10 + y_bar[:, np.newaxis]) / (1e-10 + labeled_neighbors)) * y_bar[:, np.newaxis]
    # 加總鄰居與各類別散度取得 D_l
    D_l = np.sum(score_l, axis=(-1, -2))
    
    divergence = D_u + beta * D_l
    
    if return_details:
        # 計算每個鄰居的貢獻 (在 n_classes 軸上加總)
        score_u_per_neighbor = np.sum(score_u, axis=-1)  # [n_unlabeled, k]
        score_l_per_neighbor = np.sum(score_l, axis=-1)  # [n_unlabeled, k]
        
        details = {
            'D_u': D_u,                           # [n_unlabeled]
            'D_l': D_l,                           # [n_unlabeled]
            'score_u_per_neighbor': score_u_per_neighbor,  # [n_unlabeled, k]
            'score_l_per_neighbor': score_l_per_neighbor,  # [n_unlabeled, k]
            'y_bar': y_bar,                       # [n_unlabeled, n_classes]
            # 新增: 分子分母的原始數值
            'D_u_numerator': labeled_neighbors,   # [n_unlabeled, k, n_classes] - D_u 的分子 (鄰居 soft label)
            'D_u_denominator': unlabeled_probs,   # [n_unlabeled, n_classes] - D_u 的分母 (未標記樣本預測)
            'D_l_numerator': y_bar,               # [n_unlabeled, n_classes] - D_l 的分子 (鄰居平均分佈)
            'D_l_denominator': labeled_neighbors, # [n_unlabeled, k, n_classes] - D_l 的分母 (鄰居 soft label)
        }
        return divergence, details
    
    return divergence


def select_samples_and_generate_pseudo_labels(
    labeled_features,
    labeled_true_labels,
    unlabeled_predictions,
    unlabeled_features,
    k,
    num_samples,
    beta=0.1,
    m=0.6,
    prev_val=None,
    divergence_mode='cause',  # 'cause', 'emotion', 'both', 'emotion_clause'
    return_details=False,     # 新增: 是否返回詳細計算資訊
    labeled_doc_ids=None,     # 新增: 有標籤樣本的 doc_id 列表 (用於記錄鄰居對應的文檔ID)
    unlabeled_doc_ids=None,   # 新增: 未標籤樣本的 doc_id 列表
):
    # np.ascontiguousarray: 保證記憶體是連續排列
    labeled_features_np = np.ascontiguousarray(_to_numpy(labeled_features).astype(np.float32))
    unlabeled_features_np = np.ascontiguousarray(_to_numpy(unlabeled_features).astype(np.float32))

    if isinstance(labeled_true_labels, dict):  # 若傳入的是(emotion/cause) 的分佈
        head_names = list(labeled_true_labels.keys())  # 取出每個key
        labeled_probs_dict = {
            head: np.ascontiguousarray(_to_numpy(labeled_true_labels[head]).astype(np.float32))
            for head in head_names
        }  # 將各個 key 的標註分佈轉為連續 float32 的 numpy 陣列
        unlabeled_probs_dict = {
            head: np.ascontiguousarray(_to_numpy(unlabeled_predictions[head]).astype(np.float32))
            for head in head_names
        }  # 同樣處理未標註資料的預測分佈
    else:
        head_names = None  # 單 head 情況下不需要額外字典
        labeled_probs_array = np.ascontiguousarray(_to_numpy(labeled_true_labels).astype(np.float32))  # 標註分佈轉為 numpy 陣列
        unlabeled_probs_array = np.ascontiguousarray(_to_numpy(unlabeled_predictions).astype(np.float32))  # 未標註分佈轉為 numpy 陣列
    # 來源: https://github.com/facebookresearch/faiss/wiki/Getting-started
    # Squared Euclidean distance: 兩個向量之間的距離取平方
    # labeled_features_np 的形狀是 [num_labeled, hidden_size]
    index = faiss.IndexFlatL2(labeled_features_np.shape[1])  # 建立 L2 距離的平面索引，維度等於 CLS 向量長度(隱藏層維度: 768維)
    index.add(labeled_features_np)  # 將標註資料的 CLS 向量加入索引，作為 KNN 候選點
    faiss_distances, neighbor_indices = index.search(unlabeled_features_np, k)  # 對每個未標註 CLS 向量尋找 k 個最近鄰
    # faiss_distances: [n_unlabeled, k] - 每個未標籤樣本到其 k 個鄰居的 L2 平方距離
    # neighbor_indices: [n_unlabeled, k] - 每個未標籤樣本的 k 個鄰居在 labeled 中的索引
    
    # ===== DEBUG: 驗證 FAISS L2 距離計算 (可選，驗證完後可註解掉) =====
    # 只驗證第一個未標籤樣本的第一個鄰居
    if len(unlabeled_features_np) > 0 and len(labeled_features_np) > 0:
        u_vec = unlabeled_features_np[0]
        neighbor_0_idx = neighbor_indices[0][0]
        l_vec = labeled_features_np[neighbor_0_idx]
        manual_l2_squared = np.sum((u_vec - l_vec) ** 2)
        faiss_l2_squared = faiss_distances[0][0]
        is_match = np.isclose(manual_l2_squared, faiss_l2_squared, rtol=1e-5)
        print(f"[DEBUG] FAISS L2 驗證: 手動={manual_l2_squared:.6f}, FAISS={faiss_l2_squared:.6f}, 一致={'✓' if is_match else '✗'}")

    # 初始化詳細資訊變數
    divergence_details = None
    
    if head_names is None:
        if return_details:
            divergence, divergence_details = _compute_divergence(
                labeled_probs_array, unlabeled_probs_array, neighbor_indices, beta, return_details=True
            )
        else:
            divergence = _compute_divergence(labeled_probs_array, unlabeled_probs_array, neighbor_indices, beta)
    else:
        # 根據 divergence_mode 參數決定散度計算方式
        if divergence_mode == 'cause' and 'cause' in head_names:
            # 只計算 cause 的散度(針對所有[MASK]_c的預測分佈進行平均)
            if return_details:
                divergence, divergence_details = _compute_divergence(
                    labeled_probs_dict['cause'], unlabeled_probs_dict['cause'], neighbor_indices, beta, return_details=True
                )
                # labeled_probs_dict['cause']: 每個標註樣本的原因分佈(74, 2)
                # unlabeled_probs_dict['cause']: 每個未標註樣本的原因分佈(518, 2)
                # neighbor_indices: 找到的鄰居索引，shape=(518, k)
            else:
                divergence = _compute_divergence(
                    labeled_probs_dict['cause'], unlabeled_probs_dict['cause'], neighbor_indices, beta
                )
        elif divergence_mode == 'emotion' and 'emotion' in head_names:
            # 只計算 emotion 的散度 (針對所有[MASK]_e的預測分佈進行平均)
            if return_details:
                divergence, divergence_details = _compute_divergence(
                    labeled_probs_dict['emotion'], unlabeled_probs_dict['emotion'], neighbor_indices, beta, return_details=True
                )
            else:
                divergence = _compute_divergence(
                    labeled_probs_dict['emotion'], unlabeled_probs_dict['emotion'], neighbor_indices, beta
                )
        elif divergence_mode == 'emotion_clause' and 'emotion_clause' in head_names:
            # 只計算情緒句的情緒分佈散度 (只取預測為情緒句的子句)
            if return_details:
                divergence, divergence_details = _compute_divergence(
                    labeled_probs_dict['emotion_clause'], unlabeled_probs_dict['emotion_clause'], neighbor_indices, beta, return_details=True
                )
            else:
                divergence = _compute_divergence(
                    labeled_probs_dict['emotion_clause'], unlabeled_probs_dict['emotion_clause'], neighbor_indices, beta
                )
        elif divergence_mode == 'cause_clause' and 'cause_clause' in head_names:
            # 只計算原因句的原因分佈散度 (只取預測為原因句的子句)
            if return_details:
                divergence, divergence_details = _compute_divergence(
                    labeled_probs_dict['cause_clause'], unlabeled_probs_dict['cause_clause'], neighbor_indices, beta, return_details=True
                )
            else:
                divergence = _compute_divergence(
                    labeled_probs_dict['cause_clause'], unlabeled_probs_dict['cause_clause'], neighbor_indices, beta
                )
        else:
            # divergence_mode == 'both' 或指定的 head 不存在時，計算所有 head 的平均
            head_divergences = []
            head_details_list = []
            for head in head_names:
                if return_details:
                    divergence_head, details_head = _compute_divergence(
                        labeled_probs_dict[head], unlabeled_probs_dict[head], neighbor_indices, beta, return_details=True
                    )
                    head_details_list.append((head, details_head))
                else:
                    divergence_head = _compute_divergence(
                        labeled_probs_dict[head], unlabeled_probs_dict[head], neighbor_indices, beta
                    )
                head_divergences.append(divergence_head)
            divergence = np.mean(np.stack(head_divergences, axis=0), axis=0)  # 產生每個未標註樣本的最終散度分數
            
            # 當 mode='both' 時，彙整各 head 的詳細資訊
            if return_details and head_details_list:
                divergence_details = {
                    'per_head': {head: details for head, details in head_details_list},
                    'D_u': np.mean([d['D_u'] for _, d in head_details_list], axis=0),
                    'D_l': np.mean([d['D_l'] for _, d in head_details_list], axis=0),
                }

    # (論文 Eq. 10):
    # μ^(t)(x_j) = (1-m) × μ^(t-1)(x_j) + m × D^(t)(x_j)
    # m 控制「本輪散度」的權重，(1-m) 控制 μ^(t-1)(x_j) 的權重
    # 統一使用 dict {doc_id: ema_score} 格式，避免樣本順序變化導致 index 對不上
    if unlabeled_doc_ids is None:
        raise ValueError("unlabeled_doc_ids 不可為 None，必須提供未標籤樣本的 doc_id 列表以進行 EMA 追蹤")
    
    if prev_val is not None and isinstance(prev_val, dict):
        # 根據 doc_id 查找對應的前一輪 μ^(t-1)(x_j) 分數
        prev_val_array = np.array([
            prev_val.get(doc_id, divergence[i])  # 若無歷史紀錄，使用本輪散度
            for i, doc_id in enumerate(unlabeled_doc_ids)
        ])
        current_val = (1 - m) * prev_val_array + m * divergence
    else:
        # 首輪 (prev_val 為 None 或空 dict)，直接使用本輪散度
        current_val = divergence
    
    # 建立 doc_id -> 散度分數的對應表，供下一輪使用
    current_val_dict = {doc_id: float(current_val[i]) for i, doc_id in enumerate(unlabeled_doc_ids)}

    max_val = np.max(current_val)  # 取出當前分數的最大值，作為 W
    weights = max_val - current_val  # 越接近最大值代表 divergence 越低，權重越大
    sum_weights = float(np.sum(weights))
    if np.isfinite(sum_weights) and sum_weights > 0:
        probabilities = weights / sum_weights  # 將權重正規化成機率分佈
    else:
        probabilities = np.zeros_like(weights, dtype=np.float64)

    # 計算有效樣本數（機率 > 0 的樣本），避免要求的樣本數超過可用數量
    num_nonzero = int(np.sum(probabilities > 0))
    num_samples = min(num_samples, len(current_val), num_nonzero) # 確保抽樣數量<=總樣本數、抽樣數量<=機率非零的樣本數
    
    if num_samples == 0:
        empty_selected_indices = np.array([], dtype=np.int64)
        empty_pseudo_labels = torch.empty((0,), dtype=torch.long)

        if return_details:
            detailed_info = {
                'neighbor_indices': neighbor_indices,  # [n_unlabeled, k] - 每個未標記樣本的 k 個鄰居在 labeled 中的索引
                'neighbor_distances': faiss_distances, # [n_unlabeled, k] - 每個未標記樣本到其 k 個鄰居的 L2 平方距離
                'divergence_details': divergence_details,  # D_u, D_l 等詳細資訊
                'divergence': divergence,              # 本輪計算的原始散度 (不含 EMA 平滑)
                'beta': beta,
                'weights': weights,                    # [n_unlabeled] - 論文 Eq.11: W - μ^(t)(x_j)
                'probabilities': probabilities,        # [n_unlabeled] - 正規化後的抽樣機率；若無法形成有效分佈則全為 0
                'max_divergence': max_val,             # W = max(μ^(t)(x)) 正規化因子
                'current_val_dict': current_val_dict,  # {doc_id: ema_score} - 供下一輪使用的 EMA dict
                'selection_skipped': True,
                'selection_skip_reason': f"沒有可選的樣本: len(current_val)={len(current_val)}, num_nonzero={num_nonzero}, sum(weights)={sum_weights}",
            }
            if labeled_doc_ids is not None:
                labeled_doc_ids_array = np.array(labeled_doc_ids)
                detailed_info['neighbor_doc_ids'] = labeled_doc_ids_array[neighbor_indices]
            return empty_selected_indices, empty_pseudo_labels, current_val, detailed_info

        return empty_selected_indices, empty_pseudo_labels, current_val
    
    selected_indices = np.random.choice(
        np.arange(len(current_val)),
        size=num_samples,
        replace=False,
        p=probabilities,
    )  # 依據前面算出的機率無放回抽樣

    if head_names is None:
        pseudo_source = unlabeled_probs_array  # 單 head 情況直接使用該 head 預測分佈取 argmax
    else:
        reference_head = head_names[0]  # 多 head 時先挑一個參考 head 產生硬偽標籤
        pseudo_source = unlabeled_probs_dict[reference_head]

    pseudo_labels = torch.from_numpy(np.argmax(pseudo_source[selected_indices], axis=1))  # 取每個被選樣本的類別 argmax

    if return_details:
        # 組裝完整的詳細資訊
        detailed_info = {
            'neighbor_indices': neighbor_indices,  # [n_unlabeled, k] - 每個未標記樣本的 k 個鄰居在 labeled 中的索引
            'neighbor_distances': faiss_distances, # [n_unlabeled, k] - 每個未標記樣本到其 k 個鄰居的 L2 平方距離
            'divergence_details': divergence_details,  # D_u, D_l 等詳細資訊
            'divergence': divergence,              # 本輪計算的原始散度 (不含 EMA 平滑)
            'beta': beta,
            'weights': weights,                    # [n_unlabeled] - 論文 Eq.11: W - μ^(t)(x_j)
            'probabilities': probabilities,        # [n_unlabeled] - 論文 Eq.11: 正規化後的抽樣機率
            'max_divergence': max_val,             # W = max(μ^(t)(x)) 正規化因子
            'current_val_dict': current_val_dict,  # {doc_id: ema_score} - 供下一輪使用的 EMA dict
        }
        # 如果有提供 labeled_doc_ids，則加入鄰居對應的文檔 ID
        if labeled_doc_ids is not None:
            labeled_doc_ids_array = np.array(labeled_doc_ids)
            # neighbor_doc_ids: [n_unlabeled, k] - 每個未標記樣本的鄰居文檔 ID
            detailed_info['neighbor_doc_ids'] = labeled_doc_ids_array[neighbor_indices]
        
        return selected_indices, pseudo_labels, current_val, detailed_info

    return selected_indices, pseudo_labels, current_val
