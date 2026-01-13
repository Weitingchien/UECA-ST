"""
分析 Pair GT 差異的來源 (驗證修復效果)
比較 正確的 JSON 計算方式 (New Eval Logic) 與 舊版 run_evaluation.py (Text File Logic) 的差異
"""
import json
import os

def analyze_fold(fold_num, gt_dir, pred_dir):
    gt_file = os.path.join(gt_dir, f"fold{fold_num}_test.json")
    gt_text_file = os.path.join(pred_dir, f"fold{fold_num}_gt_text_result.txt")
    
    # 方法一：從 JSON 計算 (正確邏輯 - New Eval)
    with open(gt_file, 'r', encoding='utf8') as f:
        data = json.load(f)
    
    correct_pair_gt = 0
    correct_details = []
    for doc in data:
        doc_id = doc["doc_id"]
        pairs = doc["pairs"] if doc["pairs"] else []
        # 正確邏輯：直接統計所有 pair (使用 set 去除完全重複的標註，例如 [[2,1], [2,1]])
        cnt_pair = len(set(tuple(p) for p in pairs))
        correct_pair_gt += cnt_pair
        correct_details.append((doc_id, cnt_pair, pairs))
    
    # 方法二：從 gt_text_result.txt 計算 (舊版有缺陷邏輯 - Old Eval)
    flawed_pair_gt = 0
    flawed_details = {}
    current_doc_id = None
    
    if os.path.exists(gt_text_file):
        with open(gt_text_file, 'r', encoding='utf8') as f:
            for line in f:
                line = line.strip()
                if line.startswith("doc_id"):
                    current_doc_id = line.split(":")[1].strip()
                    flawed_details[current_doc_id] = 0
                elif line != "":
                    parts = line.split()
                    if len(parts) == 3:
                        gt_pair = parts[2]
                        if gt_pair != '无':
                            flawed_pair_gt += 1
                            flawed_details[current_doc_id] += 1
    else:
        print(f"[警告] 找不到舊版格式檔案: {gt_text_file}，無法進行比較")
        return

    print(f"\n=== Fold {fold_num} ===")
    print(f"正確 Count (JSON): {correct_pair_gt}")
    print(f"舊版 Count (Text): {flawed_pair_gt}")
    
    diff = correct_pair_gt - flawed_pair_gt
    if diff != 0:
        print(f"-> 發現差異: {diff} (修正後追回了 {diff} 個遺漏的 Ground Truth)")
        print("   受影響文檔:")
        for doc_id, cnt, pairs in correct_details:
            flawed_cnt = flawed_details.get(str(doc_id), 0)
            if cnt != flawed_cnt:
                print(f"   - Doc {doc_id}: JSON={cnt}, TextFile={flawed_cnt}, Missing={cnt-flawed_cnt}, Pairs={pairs}")
    else:
        print("-> 無差異 (Perfect Match)")

if __name__ == "__main__":
    gt_dir = "split10_home_train1_test1_val1_unlabeled7_disjoint"
    pred_dir = "prompt_ECPE_home_train1_test1_val1_unlabeled7_disjoint_supervised_few_shot_CE__UECA_CE_few_shot_ST_nest_py"
    
    for fold in range(1, 11):
        analyze_fold(fold, gt_dir, pred_dir)
