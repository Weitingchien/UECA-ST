#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""依每篇文件的 pair 數量分層統計 Pair(m1) 表現"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple


Pair = Tuple[str, str]


@dataclass
class DocPrediction:
    """單篇文件的 Pair(m1) 預測結果

    輸入：由 foldX_text_result.txt 解析而來
    輸出：pred_pairs 供分層計算 TP/Pred/GT 使用
    """

    doc_id: str
    pred_pairs: list[Pair]
    invalid_pair_labels: int = 0


@dataclass
class MetricTally:
    """累計某一組文件的 Pair(m1) 計數

    輸入：逐篇文件呼叫 add_doc() 累加
    輸出：to_row() 會轉成 CSV 報表列
    """

    doc_count: int = 0
    official_tp: int = 0
    set_based_tp: int = 0
    pred_total: int = 0
    gt_total: int = 0
    oracle_possible_tp: int = 0
    invalid_pair_labels: int = 0

    def add_doc(
        self,
        *,
        official_tp: int,
        set_based_tp: int,
        pred_total: int,
        gt_total: int,
        oracle_possible_tp: int,
        invalid_pair_labels: int,
    ) -> None:
        """累加單篇文件的 Pair(m1) 計數

        輸入：單篇文件的 TP、Pred、GT 與理論可達 TP
        輸出：直接更新目前 MetricTally 物件，沒有回傳值
        """

        self.doc_count += 1
        self.official_tp += official_tp
        self.set_based_tp += set_based_tp
        self.pred_total += pred_total
        self.gt_total += gt_total
        self.oracle_possible_tp += oracle_possible_tp
        self.invalid_pair_labels += invalid_pair_labels


def parse_args() -> argparse.Namespace:
    """解析命令列參數

    輸入：使用者在命令列輸入的參數
    輸出：argparse.Namespace，包含 gt_dir、pred_dirs、folds 等設定
    """

    parser = argparse.ArgumentParser(description="依文件 pair 數量分層分析 Pair(m1) P/R/F1")
    parser.add_argument("--gt-dir", required=True, help="包含 foldX_test.json 的資料集資料夾")
    parser.add_argument("--pred-dirs", nargs="+", required=True, help="一個或多個包含 foldX_text_result.txt 的實驗資料夾")
    parser.add_argument("--output-csv", required=True, help="輸出 CSV 路徑")
    parser.add_argument("--folds", nargs="*", default=["1-10"], help="要分析的 folds，例如 1-10 或 1 2 3")
    parser.add_argument("--split-multi-emotion", action="store_true", help="分層時把 emotion_category 內的 A&B 視為多個 pair")
    parser.add_argument("--keep-duplicate-pairs", action="store_true", help="不要去除同篇文件內完全重複的 GT pairs")
    parser.add_argument("--include-fold-rows", action="store_true", help="除了 all 累計列，也輸出每個 fold 的分層結果")
    parser.add_argument("--no-pair-labels", nargs="*", default=["无", "none"], help="預測檔中代表無 pair 的標籤")
    return parser.parse_args()


def parse_folds(raw_items: Iterable[str]) -> list[int]:
    """把 fold 參數轉成整數清單

    輸入：raw_items 可包含 '1-10' 或 '1' 這類字串
    輸出：排序且去重後的 fold 整數清單
    """

    folds: set[int] = set()
    for raw_item in raw_items:
        if "-" in raw_item:
            start_text, end_text = raw_item.split("-", 1)
            start_fold = int(start_text)
            end_fold = int(end_text)
            folds.update(range(start_fold, end_fold + 1))
        else:
            folds.add(int(raw_item))
    return sorted(folds)


def normalize_id(value: object) -> str:
    """把 doc_id、clause_id 或 pair id 正規化為字串

    輸入：value 可為 int、float 或 str
    輸出：去除空白且盡量轉成整數形式的字串
    """

    text = str(value).strip()
    try:
        return str(int(text))
    except ValueError:
        return text


def normalize_no_pair_labels(labels: Iterable[str]) -> set[str]:
    """正規化無 pair 標籤集合

    輸入：labels 例如 ['无', 'none']
    輸出：小寫後的標籤集合，用來判斷預測行是否沒有 pair
    """

    return {label.strip().lower() for label in labels}


def load_json_documents(path: Path) -> list[dict]:
    """讀取 JSON 文件列表

    輸入：path 是 foldX_test.json 路徑
    輸出：該 fold 的文件 dict 清單
    """

    with path.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def unique_pairs(pairs: Iterable[Pair], *, keep_duplicates: bool) -> list[Pair]:
    """對 pairs 去重並保留原順序

    輸入：pairs 是情緒-原因 pair 清單，keep_duplicates 控制是否保留重複
    輸出：去重或原樣保留後的 pair 清單
    """

    if keep_duplicates:
        return list(pairs)
    seen: set[Pair] = set()
    output: list[Pair] = []
    for pair in pairs:
        if pair in seen:
            continue
        seen.add(pair)
        output.append(pair)
    return output


def extract_eval_pairs(doc: dict, *, keep_duplicates: bool) -> list[Pair]:
    """取出用於 Pair(m1) 評估的 JSON pairs

    輸入：doc 是資料集中的單篇文件，keep_duplicates 控制是否保留重複 pair
    輸出：[(emotion_clause_id, cause_clause_id), ...]。
    """

    raw_pairs: list[Pair] = []
    for pair in doc.get("pairs", []) or []:
        if len(pair) < 2:
            continue
        raw_pairs.append((normalize_id(pair[0]), normalize_id(pair[1])))
    return unique_pairs(raw_pairs, keep_duplicates=keep_duplicates)


def split_emotion_keys(doc: dict, emotion_clause_id: str, *, split_multi_emotion: bool) -> list[str]:
    """取得分層用的 emotion key

    輸入：doc 是單篇文件，emotion_clause_id 是 emotion 子句 id
    輸出：不拆複合情緒時回傳 [emotion_clause_id]；拆分時可能回傳多個 emotion key
    """

    if not split_multi_emotion:
        return [emotion_clause_id]

    clause_map = {
        normalize_id(clause.get("clause_id")): clause
        for clause in doc.get("clauses", [])
        if clause is not None
    }
    clause = clause_map.get(emotion_clause_id, {})
    category = str(clause.get("emotion_category") or "").strip()
    tokens = [token.strip() for token in category.split("&") if token.strip()]
    if not tokens:
        return [emotion_clause_id]
    return [f"{emotion_clause_id}:{token}" for token in tokens]


def count_complexity_pairs(doc: dict, *, keep_duplicates: bool, split_multi_emotion: bool) -> int:
    """計算分層用的 pair 數量

    輸入：doc 是單篇文件；split_multi_emotion 對齊 analyze_pair_distribution.py 的統計方式
    輸出：此文件被視為有幾個情緒-原因組合
    """

    processed_pairs: list[Pair] = []
    for emotion_id, cause_id in extract_eval_pairs(doc, keep_duplicates=keep_duplicates):
        for emotion_key in split_emotion_keys(doc, emotion_id, split_multi_emotion=split_multi_emotion):
            processed_pairs.append((emotion_key, cause_id))
    return len(unique_pairs(processed_pairs, keep_duplicates=keep_duplicates))


def group_name(pair_count: int) -> str:
    """把 pair 數量轉成分層名稱

    輸入：pair_count 是單篇文件的 pair 數
    輸出：0_pair、1_pair、2_pairs 或 3plus_pairs
    """

    if pair_count <= 0:
        return "0_pair"
    if pair_count == 1:
        return "1_pair"
    if pair_count == 2:
        return "2_pairs"
    return "3plus_pairs"


def first_emotion_by_cause(pairs: list[Pair]) -> dict[str, str]:
    """建立 official-like 評估用的 cause -> 第一個 emotion 對應

    輸入：pairs 是 JSON 原始順序去重後的 pairs
    輸出：每個 cause 子句在文字 GT 格式中會保留的第一個 emotion id
    """

    mapping: dict[str, str] = {}
    for emotion_id, cause_id in pairs:
        mapping.setdefault(cause_id, emotion_id)
    return mapping


def parse_prediction_file(path: Path, no_pair_labels: set[str]) -> dict[str, DocPrediction]:
    """解析 foldX_text_result.txt

    輸入：path 是模型輸出的文字預測檔；no_pair_labels 是無 pair 標籤集合
    輸出：doc_id 到 DocPrediction 的對應表
    """

    predictions: dict[str, DocPrediction] = {}
    current_doc_id: str | None = None
    current_clause_index = 0
    with path.open("r", encoding="utf-8") as file_obj:
        for raw_line in file_obj:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("doc_id"):
                current_doc_id = line.split(":", 1)[1].strip()
                current_clause_index = 0
                predictions[current_doc_id] = DocPrediction(doc_id=current_doc_id, pred_pairs=[])
                continue
            if current_doc_id is None:
                continue
            current_clause_index += 1
            parts = line.split()
            if len(parts) < 3:
                predictions[current_doc_id].invalid_pair_labels += 1
                continue
            pair_label = parts[2].strip()
            if pair_label.lower() in no_pair_labels:
                continue
            normalized_pair_label = normalize_id(pair_label)
            if not normalized_pair_label:
                predictions[current_doc_id].invalid_pair_labels += 1
                continue
            predictions[current_doc_id].pred_pairs.append((normalized_pair_label, str(current_clause_index)))
    return predictions


def safe_div(numerator: int, denominator: int) -> float:
    """安全除法

    輸入：numerator 是分子，denominator 是分母
    輸出：分母為 0 時回傳 0.0，否則回傳 numerator / denominator
    """

    return numerator / denominator if denominator else 0.0


def f1_score(precision: float, recall: float) -> float:
    """由 precision 與 recall 計算 F1

    輸入：precision 與 recall 都是 0 到 1 的小數
    輸出：F1 小數值
    """

    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def score_doc(doc: dict, prediction: DocPrediction, *, keep_duplicates: bool) -> dict[str, int]:
    """計算單篇文件的 Pair(m1) 計數

    輸入：doc 是 GT JSON 文件，prediction 是同 doc_id 的預測
    輸出：包含 official_tp、set_based_tp、pred_total、gt_total、oracle_possible_tp 的 dict
    """

    eval_pairs = extract_eval_pairs(doc, keep_duplicates=keep_duplicates)
    gt_pair_set = set(eval_pairs)
    first_emotion_map = first_emotion_by_cause(eval_pairs)
    official_tp = 0
    set_based_tp = 0
    for pred_pair in prediction.pred_pairs:
        pred_emotion_id, pred_cause_id = pred_pair
        if first_emotion_map.get(pred_cause_id) == pred_emotion_id:
            official_tp += 1
        if pred_pair in gt_pair_set:
            set_based_tp += 1
    unique_gt_causes = {cause_id for _, cause_id in gt_pair_set}
    return {
        "official_tp": official_tp,
        "set_based_tp": set_based_tp,
        "pred_total": len(prediction.pred_pairs),
        "gt_total": len(gt_pair_set),
        "oracle_possible_tp": len(unique_gt_causes),
        "invalid_pair_labels": prediction.invalid_pair_labels,
    }


def empty_group_tallies() -> dict[str, MetricTally]:
    """建立固定分層的空統計表

    輸入：無
    輸出：group 名稱到 MetricTally 的對應表
    """

    return {
        "overall": MetricTally(),
        "0_pair": MetricTally(),
        "1_pair": MetricTally(),
        "2_pairs": MetricTally(),
        "3plus_pairs": MetricTally(),
        "2plus_pairs": MetricTally(),
    }


def add_to_group(tallies: dict[str, MetricTally], group: str, values: dict[str, int]) -> None:
    """把單篇文件分數加到指定 group

    輸入：tallies 是分層統計表，group 是分層名稱，values 是 score_doc() 的結果
    輸出：直接更新 tallies，沒有回傳值
    """

    tallies[group].add_doc(
        official_tp=values["official_tp"],
        set_based_tp=values["set_based_tp"],
        pred_total=values["pred_total"],
        gt_total=values["gt_total"],
        oracle_possible_tp=values["oracle_possible_tp"],
        invalid_pair_labels=values["invalid_pair_labels"],
    )


def analyze_fold(
    *,
    gt_path: Path,
    pred_path: Path,
    keep_duplicates: bool,
    split_multi_emotion: bool,
    no_pair_labels: set[str],
) -> dict[str, MetricTally]:
    """分析單一 fold 的分層 Pair(m1) 表現

    輸入：gt_path 是 foldX_test.json，pred_path 是 foldX_text_result.txt
    輸出：固定 groups 的 MetricTally 統計表
    """

    docs = load_json_documents(gt_path)
    predictions = parse_prediction_file(pred_path, no_pair_labels)
    tallies = empty_group_tallies()
    for doc in docs:
        doc_id = str(doc.get("doc_id"))
        prediction = predictions.get(doc_id)
        if prediction is None:
            continue
        values = score_doc(doc, prediction, keep_duplicates=keep_duplicates)
        complexity_count = count_complexity_pairs(
            doc,
            keep_duplicates=keep_duplicates,
            split_multi_emotion=split_multi_emotion,
        )
        primary_group = group_name(complexity_count)
        add_to_group(tallies, "overall", values)
        add_to_group(tallies, primary_group, values)
        if complexity_count >= 2:
            add_to_group(tallies, "2plus_pairs", values)
    return tallies


def merge_tallies(target: dict[str, MetricTally], source: dict[str, MetricTally]) -> None:
    """把一份分層統計合併到另一份

    輸入：target 是累計表，source 是單 fold 統計表
    輸出：直接更新 target，沒有回傳值
    """

    for group, source_tally in source.items():
        target[group].doc_count += source_tally.doc_count
        target[group].official_tp += source_tally.official_tp
        target[group].set_based_tp += source_tally.set_based_tp
        target[group].pred_total += source_tally.pred_total
        target[group].gt_total += source_tally.gt_total
        target[group].oracle_possible_tp += source_tally.oracle_possible_tp
        target[group].invalid_pair_labels += source_tally.invalid_pair_labels


def format_percent(value: float) -> str:
    """把小數格式化為百分比文字

    輸入：value 是 0 到 1 的小數
    輸出：例如 '74.16%'
    """

    return f"{value * 100:.2f}%"


def tally_to_row(*, experiment: str, fold_label: str, group: str, tally: MetricTally, total_docs: int) -> dict[str, object]:
    """把 MetricTally 轉成 CSV 資料列

    輸入：experiment/fold_label/group 是列標籤，tally 是計數，total_docs 是同 fold/all 的總文件數
    輸出：可交給 csv.DictWriter 寫出的 dict
    """

    official_precision = safe_div(tally.official_tp, tally.pred_total)
    official_recall = safe_div(tally.official_tp, tally.gt_total)
    official_f1 = f1_score(official_precision, official_recall)
    set_precision = safe_div(tally.set_based_tp, tally.pred_total)
    set_recall = safe_div(tally.set_based_tp, tally.gt_total)
    set_f1 = f1_score(set_precision, set_recall)
    oracle_recall_ceiling = safe_div(tally.oracle_possible_tp, tally.gt_total)
    return {
        # experiment：目前分析的實驗資料夾名稱，通常可從名稱看出 seed 與實驗設定
        "experiment": experiment,
        # fold：目前統計的是哪一折；all 代表已把指定 folds 全部累計在一起
        "fold": fold_label,
        # group：文件依 GT pair 數量分出的群組，例如 1_pair、2_pairs、3plus_pairs、2plus_pairs、overall
        "group": group,
        # doc_count：此 group 中實際納入評估的文件數，只計算 prediction 檔中有出現的 doc_id
        "doc_count": tally.doc_count,
        # doc_percent：此 group 文件數佔同一個 fold/all 全部納入評估文件數的比例
        "doc_percent": format_percent(safe_div(tally.doc_count, total_docs)),
        # total_tp_official_like：模擬原本 Pair(m1) 評估方式得到的 TP，若同一 cause 有多個 emotion 只採第一個對應
        "total_tp_official_like": tally.official_tp,
        # total_tp_set_based：以 JSON pair 集合比對得到的 TP，只要預測 pair 存在於 GT pair 集合就算正確
        "total_tp_set_based": tally.set_based_tp,
        # total_pred：此 group 中模型預測出的 pair 總數，也就是 Precision 的分母
        "total_pred": tally.pred_total,
        # total_gt：此 group 中 GT pair 總數，也就是 Recall 的分母
        "total_gt": tally.gt_total,
        # official_precision：用 total_tp_official_like / total_pred 算出的 Pair(m1) precision
        "official_precision": format_percent(official_precision),
        # official_recall：用 total_tp_official_like / total_gt 算出的 Pair(m1) recall
        "official_recall": format_percent(official_recall),
        # official_f1：由 official_precision 與 official_recall 算出的 Pair(m1) F1
        "official_f1": format_percent(official_f1),
        # set_based_precision：用 total_tp_set_based / total_pred 算出的較寬鬆 precision
        "set_based_precision": format_percent(set_precision),
        # set_based_recall：用 total_tp_set_based / total_gt 算出的較寬鬆 recall
        "set_based_recall": format_percent(set_recall),
        # set_based_f1：由 set_based_precision 與 set_based_recall 算出的較寬鬆 F1
        "set_based_f1": format_percent(set_f1),
        # oracle_recall_ceiling_one_pair_per_clause：假設每個 cause 子句最多只能輸出一個 pair 時，理論最高 recall
        "oracle_recall_ceiling_one_pair_per_clause": format_percent(oracle_recall_ceiling),
        # gt_pairs_not_recoverable_by_one_pair_per_clause：因同一 cause 對多個 emotion 而無法被 one-pair-per-clause 格式完整表示的 GT pair 數
        "gt_pairs_not_recoverable_by_one_pair_per_clause": tally.gt_total - tally.oracle_possible_tp,
        # avg_gt_pairs_per_doc：此 group 平均每篇文件有多少 GT pair
        "avg_gt_pairs_per_doc": f"{safe_div(tally.gt_total, tally.doc_count):.4f}",
        # avg_pred_pairs_per_doc：此 group 平均每篇文件被模型預測出多少 pair
        "avg_pred_pairs_per_doc": f"{safe_div(tally.pred_total, tally.doc_count):.4f}",
        # invalid_pair_labels：預測檔中格式異常或無法解析的 pair label 數量，正常情況通常應為 0
        "invalid_pair_labels": tally.invalid_pair_labels,
    }


def write_csv(rows: list[dict[str, object]], output_csv: Path) -> None:
    """寫出 CSV 報表

    輸入：rows 是 tally_to_row() 產生的資料列，output_csv 是輸出路徑
    輸出：在磁碟建立 CSV 檔
    """

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("沒有任何可輸出的分析結果。")
    with output_csv.open("w", encoding="utf-8-sig", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    """主流程。

    輸入：命令列參數指定 GT 資料夾與一個或多個 prediction 資料夾
    輸出：CSV 報表；成功回傳 0
    """

    args = parse_args()
    folds = parse_folds(args.folds)
    gt_dir = Path(args.gt_dir)
    no_pair_labels = normalize_no_pair_labels(args.no_pair_labels)
    keep_duplicates = args.keep_duplicate_pairs
    output_rows: list[dict[str, object]] = []

    for raw_pred_dir in args.pred_dirs:
        pred_dir = Path(raw_pred_dir)
        experiment_label = pred_dir.name
        all_tallies = empty_group_tallies()
        fold_tallies_by_label: dict[str, dict[str, MetricTally]] = {}
        for fold in folds:
            gt_path = gt_dir / f"fold{fold}_test.json"
            pred_path = pred_dir / f"fold{fold}_text_result.txt"
            if not gt_path.is_file():
                print(f"[skip] 找不到 GT: {gt_path}")
                continue
            if not pred_path.is_file():
                print(f"[skip] 找不到 prediction: {pred_path}")
                continue
            fold_tallies = analyze_fold(
                gt_path=gt_path,
                pred_path=pred_path,
                keep_duplicates=keep_duplicates,
                split_multi_emotion=args.split_multi_emotion,
                no_pair_labels=no_pair_labels,
            )
            merge_tallies(all_tallies, fold_tallies)
            fold_tallies_by_label[f"fold{fold}"] = fold_tallies

        if args.include_fold_rows:
            for fold_label, fold_tallies in fold_tallies_by_label.items():
                total_docs = fold_tallies["overall"].doc_count
                for group, tally in fold_tallies.items():
                    output_rows.append(tally_to_row(experiment=experiment_label, fold_label=fold_label, group=group, tally=tally, total_docs=total_docs))

        total_docs = all_tallies["overall"].doc_count
        for group, tally in all_tallies.items():
            output_rows.append(tally_to_row(experiment=experiment_label, fold_label="all", group=group, tally=tally, total_docs=total_docs))

    write_csv(output_rows, Path(args.output_csv))
    print(f"已輸出分層 Pair(m1) 分析: {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())