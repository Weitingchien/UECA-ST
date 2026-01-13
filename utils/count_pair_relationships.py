#!/usr/bin/env python3
"""Classify documents by emotion-cause pair relationships."""
import argparse
import json
from pathlib import Path
from itertools import combinations
from typing import Iterable, List, Sequence, Set, Tuple


Pair = Tuple[str, str]


def load_documents(paths: Iterable[str]) -> Iterable[dict]:
    """Yield documents from the provided JSON file paths."""
    for raw_path in paths:
        path = Path(raw_path)
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        for doc in payload:
            yield doc


def extract_emotion_keys(clause: dict, emotion_idx: str, split_multi_emotion: bool) -> Sequence[str]:
    """提取情緒鍵標籤，支援複合情緒拆分成多個情緒。"""
    # 如果不拆分複合情緒，或子句為空，直接回傳原始情緒索引
    if not split_multi_emotion or not clause:
        return [emotion_idx]

    # 從子句中取出情緒類別，去除空白
    category = (clause.get("emotion_category") or "").strip()
    # 按 '&' 分割複合情緒，例如 "happiness&sadness" → ["happiness", "sadness"]
    tokens = [token.strip() for token in category.split("&") if token.strip()]

    # 如果沒有取到情緒類別，嘗試從情緒詞欄位取出
    if not tokens:
        # 從子句的情緒詞欄位取出備選值
        raw_token = (clause.get("emotion_token") or "").strip()
        if raw_token:
            tokens = [raw_token]

    # 如果仍然沒有取到任何情緒標籤，回傳原始情緒索引
    if not tokens:
        return [emotion_idx]

    # 回傳組合後的情緒鍵清單，格式為 "情緒索引:情緒名稱"
    return [f"{emotion_idx}:{token}" for token in tokens]

# * 之後的所有參數都必須用「參數名=值」的方式傳入
def extract_pairs(doc: dict, *, deduplicate: bool, split_multi_emotion: bool) -> List[Pair]:
    """處理並回傳文檔中的情緒-原因對，支援去重和複合情緒拆分"""
    clause_map = {
        str(clause.get("clause_id")): clause
        for clause in doc.get("clauses", [])
        if clause is not None
    }

    processed: List[Pair] = []
    seen: Set[Pair] = set()

    for pair in doc.get("pairs", []) or []:
        try:
            emotion_idx = str(int(pair[0]))
            cause_idx = str(int(pair[1]))
        except (ValueError, TypeError, IndexError):
            continue

        clause = clause_map.get(emotion_idx)
        # print(f"clause: {clause}") # clause: {'clause_id': '6', 'emotion_category': 'fear', 'emotion_token': '心焦', 'clause': '而更让老齐心焦的是'}
        emotion_keys = extract_emotion_keys(clause, emotion_idx, split_multi_emotion)
        # print(f"emotion_keys: {emotion_keys}") # emotion_keys: ['13']，表示這個pair的情緒在第20個子句

        for emotion_key in emotion_keys:
            tpl: Pair = (emotion_key, cause_idx) # 建立情緒-原因對
            if not deduplicate or tpl not in seen: # 檢查是否重複
                processed.append(tpl) # 沒重複才加入
                seen.add(tpl)

    return processed


def classify_pair_relation(first: Pair, second: Pair) -> str:
    """分類兩個情緒-原因對之間的關係類型
    
    比較兩個 pair 的情緒索引和原因索引，判斷它們屬於四種關係中的哪一種
    """
    # unpack第一個 pair: emotion_idx, cause_idx
    e1, c1 = first
    # unpack第二個 pair: emotion_idx, cause_idx
    e2, c2 = second
    
    # 情況1：情緒相同但原因不同 → "相同情緒但不同原因"
    if e1 == e2 and c1 != c2:
        return "same_emotion_diff_cause"
    
    # 情況2：情緒不同但原因相同 → "不同情緒但相同原因"
    if e1 != e2 and c1 == c2:
        return "diff_emotion_same_cause"
    
    # 情況3：情緒不同且原因也不同 → "不同情緒且不同原因"
    if e1 != e2 and c1 != c2:
        return "diff_emotion_diff_cause"
    
    # 情況4：兩個 pair 完全相同 (e1==e2 且 c1==c2) → "完全相同的對"
    # 這種情況通常不會發生，因為重複的 pair 在 extract_pairs() 時就已被移除
    return "identical_pair"



def analyze(
    files: Iterable[str],
    *,
    deduplicate: bool,
    split_multi_emotion: bool,
    allow_duplicate_docs: bool,
) -> Tuple[int, int, int, int]:
    """統計三類情緒-原因對文檔的數量: 相同情緒但不同原因、不同情緒但相同原因、不同情緒且不同原因"""
    cat1 = cat2 = cat3 = singles = 0
    seen_docs: Set[str] = set()  # 存儲已計算過的文檔 ID，用於去重
    for doc in load_documents(files): # 逐個讀取檔案中的文檔
        doc_id = str(doc.get("doc_id")) if doc.get("doc_id") is not None else None # 取出文檔的 ID，轉成字串; 如果沒有 ID 就設為 None
        if not allow_duplicate_docs and doc_id is not None: # 如果使用者沒有加 --allow-duplicate-docs (即預設去重)，且文檔有 ID
            if doc_id in seen_docs: # 檢查這個doc_id 是否已經計算過
                continue # 已計算過，跳過此文檔，不再計算 (去重效果)
            seen_docs.add(doc_id)  # 記錄已計算過的 doc_id，之後遇到重複的會被跳過

        pairs = extract_pairs(doc, deduplicate=deduplicate, split_multi_emotion=split_multi_emotion)
        # print(f"doc_id: {doc_id}, pairs: {pairs}")
        combos = list(combinations(pairs, 2)) # 從 pairs 中取出所有2個pair的組合
        # print(f"combos: {combos}") # combos: [(('8', '5'), ('8', '6')), (('8', '5'), ('8', '7')), (('8', '6'), ('8', '7'))]
        if not combos:
            singles += 1
            continue
        # 預設採用 document_multi 統計邏輯，允許單篇文檔同時屬於多個類別
        categories_present = set()  # 存放此文檔涵蓋的所有配對類型
        for first, second in combos:
            # 判斷這個 2-pair 組合屬於哪一類
            category = classify_pair_relation(first, second)
            # 只記錄三大類 (排除 "identical_pair")
            if category in {
                "same_emotion_diff_cause",
                "diff_emotion_same_cause",
                "diff_emotion_diff_cause",
            }:
                categories_present.add(category)  # 將類型加入集合
                
        print(f"doc_id: {doc_id}, categories_present: {categories_present}")

        # 如果沒有任何有效的配對類型，則算作 "單一對或重複" 的文檔
        if not categories_present:
            singles += 1  # 無法分類的文檔計入 singles
            continue  # 跳過此文檔，不計入三大類

        # 如果存在 "相同情緒但不同原因"，則計入 cat1
        if "same_emotion_diff_cause" in categories_present:
            cat1 += 1

        # 如果存在 "不同情緒但相同原因"，則計入 cat2
        if "diff_emotion_same_cause" in categories_present:
            cat2 += 1

        # 判斷是否應該計入 "不同情緒且不同原因" cat3
        if "diff_emotion_diff_cause" in categories_present:
            # 建立「情緒→原因集合」的對應表，用來檢查每個情緒有幾個原因
            emotion_to_causes = {}
            for emotion, cause in pairs:
                # 用 setdefault() 取得 (或建立) 該情緒的原因集合
                bucket = emotion_to_causes.setdefault(emotion, set())
                bucket.add(cause)  # 將原因加入集合

            # 計算有多少個情緒同時擁有多個原因
            multi_emotion_with_multi_causes = sum(
                1 for causes in emotion_to_causes.values() if len(causes) > 1
            )
            # 如果只有一個情緒對應多個原因，其他情緒還是各自對應單一原因，文檔仍可視為不同情緒、不同原因的組合
            # 限制<=1，意味結構太複雜則不計入cat 3
            # 預設應該計入 cat3 (除非有特殊情況需要排除)
            should_count_cat3 = multi_emotion_with_multi_causes <= 1
            # 如果同時存在 cat2（不同情緒同原因），則不計入 cat3 (排除重複計數)
            if "diff_emotion_same_cause" in categories_present:
                should_count_cat3 = False

            # 最後確認是否真的要計入 cat3
            if should_count_cat3:
                cat3 += 1  # 計入 "不同情緒且不同原因" 類
    return cat1, cat2, cat3, singles


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Count pair relationship categories across JSON documents"
    )
    # 必填：指定需要統計的 JSON 檔案
    parser.add_argument(
        "--files",
        nargs="+",
        required=True,
        help="JSON files to include in the analysis",
    )
    # 選填: 維持重複的情緒-原因對，不做去重
    parser.add_argument(
        "--no-deduplicate",
        action="store_true",
        help="Keep duplicate emotion-cause pairs when counting",
    )
    # 選填: 將含有 '&' 的情緒類別拆成多個標籤
    parser.add_argument(
        "--split-multi-emotion",
        action="store_true",
        help="Treat emotion categories containing '&' as separate labels",
    )
    # 選填: 允許同一 doc_id 在不同檔案都被計算
    parser.add_argument(
        "--allow-duplicate-docs",
        action="store_true",
        help="Count repeated doc_id entries (e.g., across different folds)",
    )
    args = parser.parse_args()

    cat1, cat2, cat3, singles = analyze(
        args.files,
        deduplicate=not args.no_deduplicate,
        split_multi_emotion=args.split_multi_emotion,
        allow_duplicate_docs=args.allow_duplicate_docs,
    )
    total = cat1 + cat2 + cat3
    rows = [
        ("same_emotion_diff_cause", cat1),
        ("diff_emotion_same_cause", cat2),
        ("diff_emotion_diff_cause", cat3),
        ("total", total),
    ]
    for label, count in rows:
        percentage = (count / total * 100) if total else 0.0
        print(f"{label}: {count} ({percentage:.2f}%)")
    print(f"single_pair_or_duplicates: {singles}")


if __name__ == "__main__":
    main()
