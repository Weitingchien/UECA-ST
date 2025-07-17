import os
import json
import argparse


def _calculate_metrics(tp, pred_total, gt_total):
    """一個安全的指標計算函式，避免除以零的錯誤"""
    # 計算精確率 (Precision)
    if pred_total == 0:
        precision = 0.0
    else:
        precision = tp / pred_total

    # 計算召回率 (Recall)
    if gt_total == 0:
        recall = 0.0
    else:
        recall = tp / gt_total

    # 計算 F1-Score
    if (precision + recall) == 0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
    
    return precision, recall, f1



def save_ground_truth_as_text(gt_file, pred_file, output_file):
    # Step 1: 解析 pred_file 中每個 doc_id 的子句數與 doc_id 序列
    pred_line_map = {}
    pred_doc_ids = []  # 對應每行子句的 doc_id
    with open(pred_file, 'r', encoding='utf8') as f:
        current_doc_id = None
        current_count = 0
        for line in f:
            line = line.strip()
            if line.startswith("doc_id"):
                if current_doc_id is not None:
                    pred_line_map[current_doc_id] = current_count
                try:
                    current_doc_id = line.split(":")[1].strip()
                except IndexError:
                    print(f"[警告] 無法解析 doc_id 行: {line}")
                    current_doc_id = None
                current_count = 0
            elif line != "":
                current_count += 1
                pred_doc_ids.append(current_doc_id)
        if current_doc_id is not None:
            pred_line_map[current_doc_id] = current_count

    # Step 2: 處理 gt_file 並依 cause 對應輸出 是 <pos>  / 非 无 
    with open(gt_file, 'r', encoding='utf8') as f:
        data = json.load(f)

    output_lines = []

    for doc in data:
        doc_id = str(doc["doc_id"])
        if doc_id not in pred_line_map:
            continue

        output_lines.append(f"doc_id :{doc_id}")

        d_len = doc["doc_len"]
        pairs = doc["pairs"] if doc["pairs"] else []
        pos, cause = zip(*pairs) if pairs else ([], [])
        pos, cause = list(pos), list(cause)

        for i in range(1, d_len + 1):
            if i > pred_line_map[doc_id]:
                break

            if i in pos:
                emo_tag = "是"
            else:
                emo_tag = "非"

            if i in cause:
                cause_tag = "是"
                pair_id = str(pos[cause.index(i)])
            else:
                cause_tag = "非"
                pair_id = "无"

            output_lines.append(f"{emo_tag} {cause_tag} {pair_id}")

    # Step 3: 寫入輸出檔案
    with open(output_file, "w", encoding="utf-8") as f:
        for line in output_lines:
            f.write(line + "\n")

    print(f"[Ground Truth] 儲存完成 -> {output_file}")

    return pred_doc_ids  # 回傳 doc_id 序列以便後續使用

def load_text_result(file_path):
    """讀取每行(情緒, 原因子句, 是否pair)的格式"""
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
            if len(parts) == 3:
                result.append((parts[0], parts[1], parts[2]))
                doc_ids.append(current_doc_id)
    return result, doc_ids

def evaluate(gt_list, gt_doc_ids, pred_list, pred_doc_ids):
    """比對Ground Truth與預測"""
    emo_acc = emo_pre = emo_gt_total = 0
    cause_acc = cause_pre = cause_gt_total = 0
    pair1_pre = pair1_acc = pair_gt_total = 0
    pair2_pre = pair2_acc = 0
    pair3_pre = pair3_acc = 0

    pred_emo_map = {}  # key = (doc_id, clause_id)
    for i, (emo, _, _) in enumerate(pred_list):
        pred_emo_map[(pred_doc_ids[i], str((pred_doc_ids[:i+1].count(pred_doc_ids[i]))))] = emo

    gt_emo_map = {}  # key = (doc_id, clause_id)
    for j, (gtemo, _, _) in enumerate(gt_list):
        gt_emo_map[(gt_doc_ids[j], str((gt_doc_ids[:j+1].count(gt_doc_ids[j]))))] = gtemo

    for i, ((gt_emo, gt_cause, gt_pair), (pred_emo, pred_cause, pred_pair)) in enumerate(zip(gt_list, pred_list)):
        doc_id = gt_doc_ids[i]
        clause_id = str(gt_doc_ids[:i+1].count(doc_id))
    
    # === Step 2: 遍歷資料，累積計數 ===
    for i, ((gt_emo, gt_cause, gt_pair), (pred_emo, pred_cause, pred_pair)) in enumerate(zip(gt_list, pred_list)):
        doc_id = gt_doc_ids[i]
        
        # 累積真實總數 (TP + FN)
        if gt_emo == '是': emo_gt_total += 1
        if gt_cause == '是': cause_gt_total += 1
        if gt_pair != '无': pair_gt_total += 1

        if pred_emo == '是':
            emo_pre += 1
            if gt_emo == '是':
                emo_acc += 1

        if pred_cause == '是':
            cause_pre += 1
            if gt_cause == '是':
                cause_acc += 1

        if pred_pair != '无':
            pair1_pre += 1
        if gt_pair != '无' and pred_pair == gt_pair:
            pair1_acc += 1

        if pred_pair != '无' and pred_cause == '是':
            pair2_pre += 1
            if pred_pair == gt_pair and gt_cause == '是':
                pair2_acc += 1

        if pred_cause == '是' and pred_pair != '无':
            if pred_emo_map.get((doc_id, pred_pair), '非') == '是':
                pair3_pre += 1
                if gt_cause == '是' and pred_pair == gt_pair:
                    if gt_emo_map.get((doc_id, gt_pair), '非') == '是':
                        pair3_acc += 1
                        #clause_id = str(gt_doc_ids[:i+1].count(doc_id))  # 當前子句在該 doc 中的編號
                        #print(f"[Pair3預測成立] doc_id={doc_id}, 子句={clause_id}, 預測=(emo: {pred_emo}, cause: {pred_cause}, pair: {pred_pair})")
    
    # === Step 3: 計算並印出格式化的結果 ===
    p_emo, r_emo, f1_emo = _calculate_metrics(emo_acc, emo_pre, emo_gt_total)
    p_cause, r_cause, f1_cause = _calculate_metrics(cause_acc, cause_pre, cause_gt_total)
    p_pair1, r_pair1, f1_pair1 = _calculate_metrics(pair1_acc, pair1_pre, pair_gt_total)
    p_pair2, r_pair2, f1_pair2 = _calculate_metrics(pair2_acc, pair2_pre, pair_gt_total)
    p_pair3, r_pair3, f1_pair3 = _calculate_metrics(pair3_acc, pair3_pre, pair_gt_total)

    print(f"Emotion: 預測正確的數量:{emo_acc}    預測出情緒的數量:{emo_pre}")
    print(f"Cause:   預測正確的數量:{cause_acc}  預測出原因的數量:{cause_pre}")
    print(f"Pair方式一: 預測正確: {pair1_acc} 預測出配對: {pair1_pre}")
    print(f"Pair方式二: 預測正確: {pair2_acc} 預測出配對: {pair2_pre}")
    print(f"Pair方式三: 預測正確: {pair3_acc} 預測出配對: {pair3_pre}")
    
    print("-" * 50)
    print(f"{'Metric':<12} | {'Precision':<12} | {'Recall':<12} | {'F1-Score':<12}")
    print("-" * 50)
    print(f"{'Emotion':<12} | {p_emo:<12.2%} | {r_emo:<12.2%} | {f1_emo:<12.2%}")
    print(f"{'Cause':<12} | {p_cause:<12.2%} | {r_cause:<12.2%} | {f1_cause:<12.2%}")
    print(f"{'Pair (m1)':<12} | {p_pair1:<12.2%} | {r_pair1:<12.2%} | {f1_pair1:<12.2%}")
    print(f"{'Pair (m2)':<12} | {p_pair2:<12.2%} | {r_pair2:<12.2%} | {f1_pair2:<12.2%}")
    print(f"{'Pair (m3)':<12} | {p_pair3:<12.2%} | {r_pair3:<12.2%} | {f1_pair3:<12.2%}")
    print("-" * 50)
    
    
    
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_dir", type=str, required=True)
    parser.add_argument("--pred_dir", type=str, required=True)
    parser.add_argument("--fold", type=int, default=None)
    args = parser.parse_args()

    folds = [args.fold] if args.fold is not None else list(range(1, 4))

    for fold in folds:
        gt_file = os.path.join(args.gt_dir, f"fold{fold}_test.json")
        pred_file = os.path.join(args.pred_dir, f"fold{fold}_text_result.txt")
        gt_text_file = os.path.join(args.pred_dir, f"fold{fold}_gt_text_result.txt")

        print(f"\n=== Fold {fold} ===")
        pred_doc_ids = save_ground_truth_as_text(gt_file, pred_file, gt_text_file)

        gt_list, gt_doc_ids = load_text_result(gt_text_file)
        pred_list, pred_doc_ids_check = load_text_result(pred_file)

        assert pred_doc_ids == pred_doc_ids_check, "預測文件中的 doc_id 序列與解析時不一致"
        assert len(gt_list) == len(pred_list), f"資料量不一致: GT {len(gt_list)} vs 預測 {len(pred_list)}"

        evaluate(gt_list, gt_doc_ids, pred_list, pred_doc_ids)

if __name__ == "__main__":
    main()
