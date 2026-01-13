import os
import json
import argparse


def _calculate_metrics(tp, pred_total, gt_total):
    """
    計算評估指標的輔助函數。
    為了避免除以零的錯誤，這裡做了安全檢查。
    
    Args:
        tp (int): True Positive，預測正確的數量
        pred_total (int): 預測出的總數量 (用於計算 Precision)
        gt_total (int): 真實標註的總數量 (用於計算 Recall)
    
    Returns:
        tuple: (precision, recall, f1)
    """
    # 計算精確率 (Precision) = TP / (TP + FP)
    # 即：預測正確數 / 預測出的總數
    if pred_total == 0:
        precision = 0.0
    else:
        precision = tp / pred_total

    # 計算召回率 (Recall) = TP / (TP + FN)
    # 即：預測正確數 / 真實的總數
    if gt_total == 0:
        recall = 0.0
    else:
        recall = tp / gt_total

    # 計算 F1-Score = 2 * (P * R) / (P + R)
    # 這是 Precision 和 Recall 的調和平均數
    if (precision + recall) == 0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
    
    return precision, recall, f1


def save_ground_truth_as_text(gt_file, pred_file, output_file):
    """
    將原始 JSON 格式的 Ground Truth 轉換為與預測結果一致的文字檔格式，以便進行逐行比對
    注意：這個中間格式有缺陷（每行只能存一個 Pair ID)，會導致一對多的 Pair 遺失
    因此我們在後續計算總數時，會改用 override 機制直接讀取 JSON，而不依賴這個函式的計數

    Args:
        gt_file (str): 原始 Ground Truth JSON 檔案路徑 (例如 fold1_test.json)
        pred_file (str): 模型輸出的預測結果文字檔 (用來確定每個文檔的子句數量)
        output_file (str): 要輸出的 Ground Truth 文字檔路徑
    
    Returns:
        list: 預測檔案中包含的所有 doc_id 列表 (用於後續統計過濾)
    """
    # Step 1: 解析 pred_file，建立 doc_id 到子句數量的映射
    # 我們需要知道每個文檔在預測結果中有幾行，才能生成對應行數的 GT
    pred_line_map = {}
    pred_doc_ids = []  # 依序儲存出現的 doc_id
    
    with open(pred_file, 'r', encoding='utf8') as f:
        current_doc_id = None
        current_count = 0
        for line in f:
            line = line.strip()
            if line.startswith("doc_id"):
                # 如果遇到新的 doc_id，先把上一個 doc_id 的子句數存起來
                if current_doc_id is not None:
                    pred_line_map[current_doc_id] = current_count
                try:
                    current_doc_id = line.split(":")[1].strip()
                except IndexError:
                    print(f"[警告] 無法解析 doc_id 行: {line}")
                    current_doc_id = None
                current_count = 0
            elif line != "":
                # 計算該文檔有多少個子句 (每一行代表一個子句)
                current_count += 1
                pred_doc_ids.append(current_doc_id)
        
        # 處理最後一個文檔
        if current_doc_id is not None:
            pred_line_map[current_doc_id] = current_count

    # Step 2: 讀取原始 GT JSON 檔案
    with open(gt_file, 'r', encoding='utf8') as f:
        data = json.load(f)

    output_lines = []

    # Step 3: 遍歷每個文檔，生成對應的 GT 文字行
    for doc in data:
        doc_id = str(doc["doc_id"])
        
        # 如果這個文檔不在預測結果中（可能被截斷或過濾了），則跳過
        if doc_id not in pred_line_map:
            continue

        output_lines.append(f"doc_id :{doc_id}")

        d_len = doc["doc_len"]
        pairs = doc["pairs"] if doc["pairs"] else []
        # 解壓縮 pairs，分別取得情緒子句索引 (pos) 和原因子句索引 (cause)
        pos, cause = zip(*pairs) if pairs else ([], [])
        pos, cause = list(pos), list(cause)

        # 遍歷每個子句 (從 1 到 d_len)
        for i in range(1, d_len + 1):
            # 如果超過了預測結果中的行數，就停止 (避免長度不一致)
            if i > pred_line_map[doc_id]:
                break

            # 判斷當前子句 i 是否為情緒子句
            if i in pos:
                emo_tag = "是"
            else:
                emo_tag = "非"

            # 判斷當前子句 i 是否為原因子句
            if i in cause:
                cause_tag = "是"
                # !!! 注意：這裡有潛在問題 !!!
                # 如果一個原因 i 對應多個情緒 (一對多)，cause.index(i) 只會回傳第一個
                # 導致第二個配對被遺漏。這就是為什麼我們需要後面的 total_gt_override 機制。
                pair_id = str(pos[cause.index(i)])
            else:
                cause_tag = "非"
                pair_id = "无"

            # 格式：[是否情緒] [是否原因] [配對的情緒ID或无]
            output_lines.append(f"{emo_tag} {cause_tag} {pair_id}")

    # Step 4: 將生成的內容寫入檔案
    with open(output_file, "w", encoding="utf-8") as f:
        for line in output_lines:
            f.write(line + "\n")

    print(f"[Ground Truth] 儲存完成 -> {output_file}")

    return pred_doc_ids  # 回傳 doc_id 序列，供主程式做一致性檢查


def load_text_result(file_path):
    """
    讀取模型輸出或 GT 文字檔，將其轉換為列表格式以便評估。
    
    Args:
        file_path (str): 檔案路徑
        
    Returns:
        tuple: (result_list, doc_ids_list)
        - result_list: 包含每行 (emo, cause, pair) 的元組列表
        - doc_ids_list: 對應每一行的 doc_id
    """
    result = []
    doc_ids = []
    current_doc_id = None
    
    with open(file_path, 'r', encoding='utf8') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if line.startswith("doc_id"):
            current_doc_id = line.split(":")[1].strip()
        elif line != "":
            parts = line.split()
            # 確保每一行都有三個部分 (emo, cause, pair)
            if len(parts) == 3:
                result.append((parts[0], parts[1], parts[2]))
                doc_ids.append(current_doc_id)
    return result, doc_ids


def evaluate(gt_list, gt_doc_ids, pred_list, pred_doc_ids, total_gt_override=None):
    """
    核心評估函數。比對 Ground Truth 與預測結果，計算 P/R/F1 指標。
    
    Args:
        gt_list: 真實標註列表 [(emo, cause, pair), ...]
        gt_doc_ids: 真實標註對應的 doc_id 列表
        pred_list: 預測結果列表 [(emo, cause, pair), ...]
        pred_doc_ids: 預測結果對應的 doc_id 列表
        total_gt_override: (iconal) 一個包含 (emo_gt, cause_gt, pair_gt) 的 tuple。
                           如果提供，將強制使用這些值作為 Recall 的分母，
                           以修正文字檔格式導致的一對多配對遺漏問題。
    """
    # 初始化計數器
    emo_acc = emo_pre = emo_gt_total = 0
    cause_acc = cause_pre = cause_gt_total = 0
    pair1_pre = pair1_acc = pair_gt_total = 0
    pair2_pre = pair2_acc = 0
    pair3_pre = pair3_acc = 0

    # 建立預測結果的快速查詢映射 (用於 Pair 方式三的評估)
    # key = (doc_id, pair_item_id) -> value = label ('是'/'非')
    pred_emo_map = {} 
    for i, (emo, _, _) in enumerate(pred_list):
        # 計算這是該 doc 的第幾個子句 (從 1 開始)
        clause_idx = str((pred_doc_ids[:i+1].count(pred_doc_ids[i])))
        pred_emo_map[(pred_doc_ids[i], clause_idx)] = emo

    # 建立 GT 的快速查詢映射
    gt_emo_map = {}
    for j, (gtemo, _, _) in enumerate(gt_list):
        clause_idx = str((gt_doc_ids[:j+1].count(gt_doc_ids[j])))
        gt_emo_map[(gt_doc_ids[j], clause_idx)] = gtemo

    # 遍歷每一個子句進行比對
    for i, ((gt_emo, gt_cause, gt_pair), (pred_emo, pred_cause, pred_pair)) in enumerate(zip(gt_list, pred_list)):
        doc_id = gt_doc_ids[i]
        
        # === 累積真實標註總數 (TP + FN) ===
        # 這些將作為 Recall 的分母 (如果沒有 override)
        if gt_emo == '是': emo_gt_total += 1
        if gt_cause == '是': cause_gt_total += 1
        if gt_pair != '无': pair_gt_total += 1

        # === 評估 Emotion (情緒抽取) ===
        if pred_emo == '是':
            emo_pre += 1  # 預測為正例 (TP + FP)
            if gt_emo == '是':
                emo_acc += 1  # 預測正確 (TP)

        # === 評估 Cause (原因抽取) ===
        if pred_cause == '是':
            cause_pre += 1
            if gt_cause == '是':
                cause_acc += 1

        # === 評估 Pair (配對抽取) - 方式一 (Strict) ===
        # 直接比對預測的 Pair ID 是否正確
        if pred_pair != '无':
            pair1_pre += 1
        if gt_pair != '无' and pred_pair == gt_pair:
            pair1_acc += 1

        # === 評估 Pair (配對抽取) - 方式二 (Relaxed Cause) ===
        # 預測了 Pair，且該子句也被預測為 Cause (Double Check)
        if pred_pair != '无' and pred_cause == '是':
            pair2_pre += 1
            if pred_pair == gt_pair and gt_cause == '是':
                pair2_acc += 1

        # === 評估 Pair (配對抽取) - 方式三 (Indirect Check) ===
        # 這是最嚴格的方式，檢查 Cause 和指向的 Emotion 是否都被標記為正確類別
        if pred_cause == '是' and pred_pair != '无':
            # 檢查預測指向的那個子句，是否也被預測為 Emotion
            if pred_emo_map.get((doc_id, pred_pair), '非') == '是':
                pair3_pre += 1
                # 檢查是否完全匹配 GT
                if gt_cause == '是' and pred_pair == gt_pair:
                    if gt_emo_map.get((doc_id, gt_pair), '非') == '是':
                        pair3_acc += 1
    
    # === Step 2.5: 使用正確的 GT 總數覆蓋 (Override) ===
    # 這是修正一對多配對遺漏的關鍵步驟
    if total_gt_override:
        emo_gt_total, cause_gt_total, pair_gt_total = total_gt_override

    # === Step 3: 計算指標 (P, R, F1) ===
    p_emo, r_emo, f1_emo = _calculate_metrics(emo_acc, emo_pre, emo_gt_total)
    p_cause, r_cause, f1_cause = _calculate_metrics(cause_acc, cause_pre, cause_gt_total)
    p_pair1, r_pair1, f1_pair1 = _calculate_metrics(pair1_acc, pair1_pre, pair_gt_total)
    p_pair2, r_pair2, f1_pair2 = _calculate_metrics(pair2_acc, pair2_pre, pair_gt_total)
    p_pair3, r_pair3, f1_pair3 = _calculate_metrics(pair3_acc, pair3_pre, pair_gt_total)

    # 建立輸出結果的文字列表
    output_lines = []
    output_lines.append(f"Emotion: 預測正確的數量:{emo_acc}    預測出情緒的數量:{emo_pre}    實際正確情緒的數量:{emo_gt_total}")
    output_lines.append(f"Cause:   預測正確的數量:{cause_acc}  預測出原因的數量:{cause_pre}  實際正確原因的數量:{cause_gt_total}")
    output_lines.append(f"Pair方式一: 預測正確: {pair1_acc} 預測出配對: {pair1_pre} 實際正確配對的數量:{pair_gt_total}")
    output_lines.append(f"Pair方式二: 預測正確: {pair2_acc} 預測出配對: {pair2_pre} 實際正確配對的數量:{pair_gt_total}")
    output_lines.append(f"Pair方式三: 預測正確: {pair3_acc} 預測出配對: {pair3_pre} 實際正確配對的數量:{pair_gt_total}")
    output_lines.append("-" * 50)
    output_lines.append(f"{'Metric':<12} | {'Precision':<12} | {'Recall':<12} | {'F1-Score':<12}")
    output_lines.append("-" * 50)
    output_lines.append(f"{'Emotion':<12} | {p_emo:<12.2%} | {r_emo:<12.2%} | {f1_emo:<12.2%}")
    output_lines.append(f"{'Cause':<12} | {p_cause:<12.2%} | {r_cause:<12.2%} | {f1_cause:<12.2%}")
    output_lines.append(f"{'Pair (m1)':<12} | {p_pair1:<12.2%} | {r_pair1:<12.2%} | {f1_pair1:<12.2%}")
    output_lines.append(f"{'Pair (m2)':<12} | {p_pair2:<12.2%} | {r_pair2:<12.2%} | {f1_pair2:<12.2%}")
    output_lines.append(f"{'Pair (m3)':<12} | {p_pair3:<12.2%} | {r_pair3:<12.2%} | {f1_pair3:<12.2%}")
    output_lines.append("-" * 50)
    
    # 輸出到終端機
    for line in output_lines:
        print(line)
    
    # 回傳所有輸出行的合併字串，供寫入結果檔案
    return "\n".join(output_lines)
    

def main():
    # 設定命令列參數
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_dir", type=str, required=True, help="Ground Truth 資料夾路徑")
    parser.add_argument("--pred_dir", type=str, required=True, help="預測結果資料夾路徑")
    parser.add_argument("--fold", type=int, default=None, help="指定要評估的 Fold (預設為 None，代表跑全部 1-10)")
    args = parser.parse_args()

    # 決定要跑哪些 Fold
    folds = [args.fold] if args.fold is not None else list(range(1, 11))
    
    # 用於收集所有 fold 的結果，最後寫入總表
    all_results = []

    for fold in folds:
        # 定義檔案路徑
        gt_file = os.path.join(args.gt_dir, f"fold{fold}_test.json")  # 原始 JSON GT
        pred_file = os.path.join(args.pred_dir, f"fold{fold}_text_result.txt")  # 模型預測結果
        gt_text_file = os.path.join(args.pred_dir, f"fold{fold}_gt_text_result.txt")  # 轉換後的中間 GT 文字檔

        fold_header = f"\n=== Fold {fold} ==="
        print(fold_header)
        all_results.append(fold_header)
        
        # Step 1: 將 JSON GT 轉換為文字格式 (主要用於對齊行數)
        pred_doc_ids = save_ground_truth_as_text(gt_file, pred_file, gt_text_file)

        # Step 2: 計算正確的 GT 總數 (直接從 JSON 資料計算)
        # 這是為了解決文字檔格式無法存儲一對多配對導致的計數錯誤
        json_emo_gt = 0
        json_cause_gt = 0
        json_pair_gt = 0
        
        with open(gt_file, 'r', encoding='utf8') as f:
            json_data = json.load(f)
            for doc in json_data:
                # 只有當 doc_id 存在於預測結果中才計算 
                # (排除因 token 長度限制而被截斷/移除的文檔，確保公平比較)
                if str(doc['doc_id']) in pred_doc_ids:
                    pairs = doc.get("pairs", [])
                    if pairs:
                        # 統計 Pair GT
                        # 使用 set(tuple(p)) 去除完全重複的標註 (如 [[2,1], [2,1]])
                        # 這樣既修正了一對多遺漏，也避免了重複計算
                        json_pair_gt += len(set(tuple(p) for p in pairs))
                        
                        # 統計 Emotion GT & Cause GT (使用 set 去重)
                        pos_set = set()
                        cause_set = set()
                        for p in pairs:
                            pos_set.add(p[0]) # pair[0] 是 emotion
                            cause_set.add(p[1]) # pair[1] 是 cause
                        json_emo_gt += len(pos_set)
                        json_cause_gt += len(cause_set)

        # Step 3: 讀取文字檔內容進行逐行比對
        gt_list, gt_doc_ids = load_text_result(gt_text_file)
        pred_list, pred_doc_ids_check = load_text_result(pred_file)

        # 進行一致性檢查
        assert pred_doc_ids == pred_doc_ids_check, "預測文件中的 doc_id 序列與解析時不一致"
        assert len(gt_list) == len(pred_list), f"資料量不一致: GT {len(gt_list)} vs 預測 {len(pred_list)}"
        
        # Step 4: 執行評估，並傳入正確的 GT 總數
        fold_result = evaluate(gt_list, gt_doc_ids, pred_list, pred_doc_ids, 
                             total_gt_override=(json_emo_gt, json_cause_gt, json_pair_gt))
        all_results.append(fold_result)
    
    # Step 5: 將所有結果寫入 test_results.txt
    output_file = os.path.join(args.pred_dir, "test_results.txt")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(all_results))
    print(f"\n✓ 評估結果已儲存至: {output_file}")

if __name__ == "__main__":
    main()
