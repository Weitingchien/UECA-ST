import os
import json
import argparse


def _calculate_metrics(tp, pred_total, gt_total):
    """安全計算指標，避免除以零。"""
    precision = tp / pred_total if pred_total else 0.0
    recall = tp / gt_total if gt_total else 0.0
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def save_ground_truth_as_text(gt_file, pred_file, output_file):
    """根據預測文件的子句數量，產生對應格式的 Ground Truth 文字檔。"""
    pred_line_map = {}
    pred_doc_ids = []
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

            emo_tag = "是" if i in pos else "非"
            cause_tag = "是" if i in cause else "非"
            if i in pos:
                pair_id = str(cause[pos.index(i)])
            else:
                pair_id = "无"

            output_lines.append(f"{emo_tag} {cause_tag} {pair_id}")

    with open(output_file, "w", encoding="utf-8") as f:
        for line in output_lines:
            f.write(line + "\n")

    print(f"[Ground Truth] 儲存完成 -> {output_file}")
    return pred_doc_ids


def load_text_result(file_path):
    """讀取模型輸出或 GT 文字檔，回傳 (label 三元組, doc_id) 序列。"""
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


def evaluate(gt_list, gt_doc_ids, pred_list, pred_doc_ids, total_gt_override=None):
    """比對 Ground Truth 與預測結果並印出完整指標，回傳結果字串供寫入檔案。"""
    emo_acc = emo_pre = emo_gt_total = 0
    cause_acc = cause_pre = cause_gt_total = 0
    pair1_pre = pair1_acc = pair_gt_total = 0
    pair2_pre = pair2_acc = 0
    pair3_pre = pair3_acc = 0

    pred_cause_map = {}
    for i, (_, c, _) in enumerate(pred_list):
        key = (pred_doc_ids[i], str(pred_doc_ids[:i + 1].count(pred_doc_ids[i])))
        pred_cause_map[key] = c

    gt_cause_map = {}
    for j, (_, gtc, _) in enumerate(gt_list):
        key = (gt_doc_ids[j], str(gt_doc_ids[:j + 1].count(gt_doc_ids[j])))
        gt_cause_map[key] = gtc

    for i, ((gt_emo, gt_cause, gt_pair), (pred_emo, pred_cause, pred_pair)) in enumerate(zip(gt_list, pred_list)):
        doc_id = gt_doc_ids[i]

        if gt_emo == '是':
            emo_gt_total += 1
        if gt_cause == '是':
            cause_gt_total += 1
        if gt_pair != '无':
            pair_gt_total += 1

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

        if pred_pair != '无' and pred_emo == '是':
            pair2_pre += 1
            if pred_pair == gt_pair and gt_emo == '是':
                pair2_acc += 1

        if pred_emo == '是' and pred_pair != '无':
            if pred_cause_map.get((doc_id, pred_pair), '非') == '是':
                pair3_pre += 1
                if gt_emo == '是' and pred_pair == gt_pair:
                    if gt_cause_map.get((doc_id, gt_pair), '非') == '是':
                        pair3_acc += 1

    # === Step 2.5: Override GT totals if provided ===
    # 這一步確保我們使用從 JSON 計算的正確 GT 總數（包含一對多關係的所有 pair），而非依賴可能遺漏資訊的文字檔統計
    if total_gt_override:
        emo_gt_total, cause_gt_total, pair_gt_total = total_gt_override

    p_emo, r_emo, f1_emo = _calculate_metrics(emo_acc, emo_pre, emo_gt_total)
    p_cause, r_cause, f1_cause = _calculate_metrics(cause_acc, cause_pre, cause_gt_total)
    p_pair1, r_pair1, f1_pair1 = _calculate_metrics(pair1_acc, pair1_pre, pair_gt_total)
    p_pair2, r_pair2, f1_pair2 = _calculate_metrics(pair2_acc, pair2_pre, pair_gt_total)
    p_pair3, r_pair3, f1_pair3 = _calculate_metrics(pair3_acc, pair3_pre, pair_gt_total)

    # 建立輸出內容（同時用於 print 和寫入檔案）
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
    
    # 輸出到 terminal
    for line in output_lines:
        print(line)
    
    # 回傳結果字串供寫入檔案
    return "\n".join(output_lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_dir", type=str, required=True)
    parser.add_argument("--pred_dir", type=str, required=True)
    parser.add_argument("--fold", type=int, default=None)
    args = parser.parse_args()

    folds = [args.fold] if args.fold is not None else list(range(1, 11))
    
    # 用於收集所有 fold 的結果
    all_results = []

    for fold in folds:
        gt_file = os.path.join(args.gt_dir, f"fold{fold}_test.json")
        pred_file = os.path.join(args.pred_dir, f"fold{fold}_text_result.txt")
        gt_text_file = os.path.join(args.pred_dir, f"fold{fold}_gt_text_result.txt")

        fold_header = f"\n=== Fold {fold} ==="
        print(fold_header)
        all_results.append(fold_header)
        
        pred_doc_ids = save_ground_truth_as_text(gt_file, pred_file, gt_text_file)

        gt_list, gt_doc_ids = load_text_result(gt_text_file)
        pred_list, pred_doc_ids_check = load_text_result(pred_file)

        assert pred_doc_ids == pred_doc_ids_check, "預測文件中的 doc_id 序列與解析時不一致"
        assert len(gt_list) == len(pred_list), f"資料量不一致: GT {len(gt_list)} vs 預測 {len(pred_list)}"

        # 計算正確的 GT 總數 (直接從 JSON 資料計算)
        json_emo_gt = 0
        json_cause_gt = 0
        json_pair_gt = 0
        
        with open(gt_file, 'r', encoding='utf8') as f:
            json_data = json.load(f)
            for doc in json_data:
                # 只有當 doc_id 存在於預測結果中才計算 (避免因長度截斷而被移除的文檔影響統計)
                if str(doc['doc_id']) in pred_doc_ids:
                    pairs = doc.get("pairs", [])
                    if pairs:
                        # 統計 Pair GT (包含重複的一對多情況，修正之前文字檔的遺漏，但排除完全重複的標註)
                        json_pair_gt += len(set(tuple(p) for p in pairs))
                        
                        # 統計 Emotion GT & Cause GT (使用 set 去重)
                        pos_set = set()
                        cause_set = set()
                        for p in pairs:
                            pos_set.add(p[0])
                            cause_set.add(p[1])
                        json_emo_gt += len(pos_set)
                        json_cause_gt += len(cause_set)

        # 傳入正確的 GT 總數進行評估
        fold_result = evaluate(gt_list, gt_doc_ids, pred_list, pred_doc_ids,
                             total_gt_override=(json_emo_gt, json_cause_gt, json_pair_gt))
        all_results.append(fold_result)
    
    # 寫入 test_results.txt
    output_file = os.path.join(args.pred_dir, "test_results.txt")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(all_results))
    print(f"\n✓ 評估結果已儲存至: {output_file}")


if __name__ == "__main__":
    main()
