import argparse  # 匯入 argparse 以解析命令列參數
import json  # 匯入 json 處理資料檔案
from collections import Counter  # 匯入 Counter 幫助統計次數
from pathlib import Path  # 匯入 Path 方便處理檔案路徑


def normalize_pairs(pairs):  # 將情緒-原因配對去重並維持原順序
    """Return unique emotion-cause pairs while preserving order."""
    seen = set()  # 用集合記錄已出現的配對
    deduped = []  # 儲存去重後的配對列表
    for pair in pairs:  # 逐一檢查每個配對
        tpl = tuple(pair)  # 轉為不可變 tuple 以便放入集合
        if tpl not in seen:  # 若未出現過才保留
            seen.add(tpl)  # 標記已看過
            deduped.append(pair)  # 加入輸出列表
    return deduped  # 回傳去重結果


def count_pairs(doc, deduplicate=False, split_multi_emotion=False):  # 計算單篇文件的配對數量
    pairs = doc.get("pairs", []) or []  # 取出原始配對清單
    if deduplicate:  # 若要求去重
        pairs = normalize_pairs(pairs)  # 先去除重複配對

    if not split_multi_emotion:  # 若不拆複合情緒
        return len(pairs)  # 直接回傳配對數

    clauses = {str(clause.get("clause_id")): clause for clause in doc.get("clauses", [])}  # 建立子句索引
    # clauses = {'1':{'clause_id': '1',  'emotion_category': 'null', '2':  {'clause_id': '2',  'emotion_category': 'null',}......}
    total = 0  # 累計拆分後的配對數
    for emotion_idx, _ in pairs:  # 逐一處理每個配對
        clause = clauses.get(str(emotion_idx))  # 找出對應情緒子句
        if not clause:  # 若情緒子句缺失
            total += 1  # 視為單一配對
            continue  # 繼續下一個
        category = (clause.get("emotion_category") or "").strip()  # 取得情緒標籤
        if "&" in category:  # 若標籤含多個情緒
            count = 0
            for token in category.split("&"):
                if token.strip():
                    count += 1
            if count == 0:
                count = 1
            total += count
        else:
            total += 1  # 單一情緒只算一次
    return total  # 回傳拆分後的配對數


def load_documents(paths, deduplicate=False, split_multi_emotion=False):  # 讀取多個檔案並統計配對分佈
    counter = Counter()  # 紀錄各配對數出現的次數
    total = 0  # 計算總文件數
    for raw_path in paths:  # 逐一讀取輸入檔案
        path = Path(raw_path)  # 產生 Path 物件
        with path.open(encoding="utf-8") as handle:  # 開啟檔案讀取內容
            docs = json.load(handle)  # 解析 JSON 文件列表
        for doc in docs:  # 逐篇文件處理
            pair_count = count_pairs(  # 取得當前文件的配對數
                doc,
                deduplicate=deduplicate,
                split_multi_emotion=split_multi_emotion,
            )
            counter[pair_count] += 1  # 統計對應配對數的文件數量
            total += 1  # 累計總文件數
    return total, counter  # 回傳總數與分佈


def main():  # 主程式入口
    parser = argparse.ArgumentParser(  # 建立參數解析器
        description="Analyze distribution of emotion-cause pair counts across JSON documents"
    )
    parser.add_argument(  # 必填：要分析的 JSON 檔案路徑
        "--files",
        nargs="+",
        required=True,
        help="Paths to JSON files to include in the analysis",
    )
    parser.add_argument(  # 選項：是否去除重複配對
        "--deduplicate",
        action="store_true",
        help="Treat duplicate emotion-cause pairs within a document as a single pair",
    )
    parser.add_argument(  # 選項：是否拆分複合情緒
        "--split-multi-emotion",
        action="store_true",
        help="Split emotion categories like 'joy&surprise' into multiple pairs when counting",
    )
    args = parser.parse_args()  # 解析命令列參數

    total_docs, pair_counter = load_documents(  # 讀取並統計所有文件
        args.files,
        deduplicate=args.deduplicate,
        split_multi_emotion=args.split_multi_emotion,
    )

    one_pair = pair_counter.get(1, 0)  # 拿出恰有一對的文件數
    two_pairs = pair_counter.get(2, 0)  # 拿出恰有兩對的文件數
    three_plus = sum(count for pairs, count in pair_counter.items() if pairs >= 3)  # 計算三對以上的文件數

    rows = [  # 整理輸出格式
        ("only_one_pair", one_pair),
        ("exactly_two_pairs", two_pairs),
        ("three_or_more_pairs", three_plus),
        ("total", total_docs),
    ]

    for label, count in rows:  # 逐列輸出統計
        percentage = (count / total_docs) * 100 if total_docs else 0.0  # 計算百分比
        print(f"{label}: {count} ({percentage:.2f}%)")  # 印出結果


if __name__ == "__main__":  # 確認是否由命令列執行
    main()  # 執行主程式


"""
    python utils/analyze_pair_distribution.py --files split10/fold1_train.json split10/fold1_test.json --split-multi-emotion --deduplicate
    only_one_pair: 1759 (90.44%)
    exactly_two_pairs: 164 (8.43%)
    three_or_more_pairs: 22 (1.13%)
    total: 1945 (100.00%)
    
    
    python utils/analyze_pair_distribution.py --files split10/fold{1..10}_train.json split10/fold{1..10}_test.json --split-multi-emotion --deduplicate
    only_one_pair: 17590 (90.44%)   
    exactly_two_pairs: 1640 (8.43%) 
    three_or_more_pairs: 220 (1.13%)
    total: 19450 (100.00%)
"""