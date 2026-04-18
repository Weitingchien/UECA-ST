#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析「實際預測全對，但沒有被選中」的 doc_id

流程：
1) 讀取 compare_unfiltered_mask3_correct_CE.py 產生的 all_doc_correct 快取
2) 讀取 doc_selection_history_fold{fold}_reconstructed_from_json.txt (可自訂)
3) 做差集: all_doc_correct_doc_ids - selected_doc_ids

支援兩種輸入模式：
- 單一實驗資料夾：--experiment-dir
- 多 seed summary: --summary-file + --experiments-root

輸出：
- unfiltered_all_doc_correct_fold{fold}.txt
- correct_but_not_selected_fold{fold}.txt
- unselected_correct_summary.txt
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path as _Path

# 讓腳本可從任意工作目錄執行。
# 作法：把「專案根目錄」插到 sys.path 最前面，確保 `from utils...` 可被解析。
# __file__                   -> 目前腳本檔案路徑
# .resolve()                 -> 轉絕對路徑
# .parent                    -> utils 目錄
# .parent.parent             -> 專案根目錄
sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

from dataclasses import dataclass
from pathlib import Path
from typing import List, Set


@dataclass
class FoldResult:
    # fold 編號（例如 1~10）
    fold: int
    # 該 fold 的未標註文件總數
    total_docs: int
    # 該 fold 中「整篇文件三個 MASK 都正確」的 doc 數
    all_doc_correct_count: int
    # 該 fold 中「在 selection history 出現過」的 doc 數
    selected_count: int
    # 該 fold 中「正確但未被選中」的 doc 數（差集）
    correct_but_not_selected_count: int


def json_load(path: Path):
    # 延遲匯入 json，讓模組載入更乾淨；真正需要時才匯入。
    import json

    # 以 UTF-8 讀檔，回傳 Python 物件（list/dict）。
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_args() -> argparse.Namespace:
    # 建立命令列解析器。
    p = argparse.ArgumentParser(description="分析每 fold『實際全對但未被選中』的 doc_id")

    # 互斥參數：三者只能擇一，且必填其一。
    # - --experiment-dir：只處理單一實驗目錄
    # - --summary-file：從單一 summary 解析多個 seed 目錄
    # - --summary-files：同時處理多個 summary (每個 summary 內含多 seed)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--experiment-dir", help="單一實驗資料夾")
    group.add_argument("--summary-file", help="summary 檔案路徑，會從『彙整的實驗目錄』讀出所有 seed 目錄")
    group.add_argument("--summary-files", nargs="+",
                       help="多個 summary 檔案路徑 (批次處理多組實驗)")

    # summary 模式需要的實驗根目錄；單一模式可不填
    exp_root_group = p.add_mutually_exclusive_group(required=False)
    exp_root_group.add_argument(
        "--experiments-root",
        default=None,
        help="單一實驗根目錄 (summary 模式必填)，例如 ep_split10_t1te1v1_u7_disjoint",
    )
    exp_root_group.add_argument(
        "--experiments-roots",
        nargs="+",
        default=None,
        help="多個實驗根目錄，會依序搜尋 summary 內的實驗資料夾",
    )
    # 資料集根目錄（內含各 fold 的 unlabeled 檔）
    p.add_argument("--dataset-dir", required=True, help="資料集目錄")
    # fold 範圍起點
    p.add_argument("--fold-start", type=int, default=1, help="起始 fold")
    # fold 範圍終點
    p.add_argument("--fold-end", type=int, default=10, help="結束 fold")
    p.add_argument(
        "--selection-file-pattern",
        default="doc_selection_history_fold{fold}_reconstructed_from_json.txt",
        help="選中歷史檔名樣板（需包含 {fold}）",
    )
    # 統一輸出根目錄；若不提供則輸出到各實驗的 pseudo_results_*
    p.add_argument(
        "--output-dir",
        default=None,
        help="輸出資料夾；預設為該實驗最新 pseudo_results_*",
    )
    p.add_argument(
        "--all-correct-cache-pattern",
        default="cache_unfiltered_all_doc_correct_fold{fold}.json",
        help="all_doc_correct 快取檔名樣板（需包含 {fold}）",
    )
    p.add_argument(
        "--tokenizer-path",
        default="bert-base-chinese",
        help="tokenizer 路徑，用於過濾超過 512 token 的文件 "
             "(與 UnlabeledDataset 一致); 設為 none 則使用原始 JSON 長度",
    )
    return p.parse_args()


def read_experiment_names_from_summary(summary_file: Path) -> List[str]:
    # 讀取 summary 全文 (忽略非法字元，避免編碼雜訊中斷)
    text = summary_file.read_text(encoding="utf-8", errors="ignore")
    names: List[str] = []
    # 逐行找出像「- prompt_ECPE_few_shot_ST_...」的實驗名稱
    for line in text.splitlines():
        m = re.match(r"\s*-\s*(prompt_ECPE_few_shot_ST_[^\s]+)", line)
        if m:
            names.append(m.group(1).strip())
    # 回傳解析到的實驗目錄名稱列表 (不含 root)
    return names


def resolve_experiments_roots(args: argparse.Namespace) -> List[Path]:
    """解析並驗證實驗根目錄（支援單一路徑與多路徑）。"""
    if args.experiments_roots:
        roots = [Path(p) for p in args.experiments_roots]
    elif args.experiments_root:
        roots = [Path(args.experiments_root)]
    else:
        roots = []

    missing = [str(p) for p in roots if not p.is_dir()]
    if missing:
        raise FileNotFoundError(
            "以下 experiments root 不存在:\n" + "\n".join(missing)
        )
    return roots


def resolve_experiment_dir(exp_name: str, roots: List[Path]) -> Path:
    """在多個根目錄中解析單一實驗資料夾。"""
    hits = [root / exp_name for root in roots if (root / exp_name).is_dir()]
    if not hits:
        searched = "\n".join(str(root / exp_name) for root in roots)
        raise FileNotFoundError(
            f"找不到實驗目錄: {exp_name}\n已搜尋:\n{searched}"
        )

    if len(hits) > 1:
        print(
            f"[警告] 實驗目錄 {exp_name} 在多個 roots 都存在，"
            f"將使用第一個: {hits[0]}"
        )
    return hits[0]


def _resolve_one_summary(summary_file: Path, roots: List[Path]) -> List[Path]:
    """從單一 summary 檔解析出實驗目錄清單."""
    if not summary_file.exists():
        raise FileNotFoundError(f"找不到 summary 檔案: {summary_file}")

    names = read_experiment_names_from_summary(summary_file)
    if not names:
        raise RuntimeError(f"在 summary 中解析不到任何 prompt 實驗目錄: {summary_file}")

    targets: List[Path] = []
    for n in names:
        p = resolve_experiment_dir(n, roots)
        targets.append(p)

    return targets


def resolve_target_experiments(args: argparse.Namespace) -> List[Path]:
    # 模式 A: 單一實驗目錄
    if args.experiment_dir:
        exp_dir = Path(args.experiment_dir)
        if not exp_dir.is_dir():
            raise FileNotFoundError(f"找不到實驗資料夾: {exp_dir}")
        return [exp_dir]

    # 模式 B / C 需要 experiments_root(s)
    summary_list = []
    if args.summary_file:
        summary_list = [args.summary_file]
    elif args.summary_files:
        summary_list = args.summary_files

    if not summary_list:
        raise ValueError("請提供 --experiment-dir、--summary-file 或 --summary-files")
    roots = resolve_experiments_roots(args)
    if not roots:
        raise ValueError("使用 --summary-file(s) 時，請提供 --experiments-root 或 --experiments-roots")

    # 收集所有 summary 的實驗目錄 (去重但保持順序)
    all_targets: List[Path] = []
    seen: set = set()
    for sf in summary_list:
        for p in _resolve_one_summary(Path(sf), roots):
            if p not in seen:
                all_targets.append(p)
                seen.add(p)

    return all_targets


def find_pseudo_results_dir(exp_dir: Path) -> Path:
    # 找出所有 pseudo_results_* 子資料夾
    candidates = sorted([p for p in exp_dir.glob("pseudo_results_*") if p.is_dir()])
    if not candidates:
        raise FileNotFoundError(f"{exp_dir} 底下找不到 pseudo_results_* 資料夾")
    # 取名稱排序最後一個 (通常是最新)
    return candidates[-1]


def build_unlabeled_file_path(dataset_dir: str | Path, fold: int) -> Path:
    """建立 fold 對應的未標註資料路徑"""
    return Path(dataset_dir) / f"fold{fold}_unlabeled.json"


def count_filtered_docs(unlabeled_path: Path, tokenizer) -> int:
    """
    計算過濾超長文件後的實際文件數量
    與 UnlabeledDataset.__init__ 的過濾邏輯一致:
    將每篇文件編碼後，長度超過 512 者會被移除

    Args:
        unlabeled_path: fold{fold}_unlabeled.json 路徑
        tokenizer: 已載入的 transformers tokenizer; 為 None 時退化為原始 JSON 長度
    Returns:
        過濾後的文件數量
    """
    data = json_load(unlabeled_path)
    if tokenizer is None:
        return len(data)

    kept = 0
    for doc in data:
        d_len = doc["doc_len"]
        part_sentence = [clause["clause"] for clause in doc["clauses"]]
        mask_full_document = ""
        for i in range(1, d_len + 1):
            mask_full_document = mask_full_document + " " + str(i) + " " + part_sentence[i - 1]
            mask_full_document = mask_full_document + "[MASK] [MASK] [MASK] [SEP]"
        count_len = len(
            tokenizer.encode_plus(mask_full_document, return_tensors="pt")["input_ids"][0]
        )
        if count_len <= 512:
            kept += 1
    return kept


def parse_selected_doc_ids(selection_file: Path) -> Set[str]:
    # 選中歷史檔不存在就中止，避免統計失真
    if not selection_file.exists():
        raise FileNotFoundError(f"找不到選中歷史檔案: {selection_file}")

    selected: Set[str] = set()
    # 匹配格式：Doc <id>: 被選中 <n> 次 ...
    pat = re.compile(r"^Doc\s+([^:]+):\s+被選中\s+\d+\s+次")
    for line in selection_file.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = pat.match(line.strip())
        if m:
            # 只記錄 doc_id
            selected.add(m.group(1).strip())
    # 回傳被選中 doc_id 集合
    return selected


def sort_doc_ids(doc_ids: Set[str]) -> List[str]:
    # 排序規則：
    # 1) 可轉 int 的 doc_id 先按數值排序
    # 2) 不能轉 int 的放後面按字串排序
    def key(v: str):
        try:
            return (0, int(v), v)
        except ValueError:
            return (1, 10**18, v)

    return sorted(doc_ids, key=key)


def write_doc_id_list(path: Path, title: str, doc_ids: Set[str]) -> None:
    # 將集合轉排序後清單，便於比對與追蹤
    ordered = sort_doc_ids(doc_ids)
    # 檔案格式：標題 + 數量 + 分隔線 + doc_id 清單
    lines = [title, f"count={len(ordered)}", "=" * 60]
    lines.extend(ordered)
    # 寫入 UTF-8 純文字
    path.write_text("\n".join(lines), encoding="utf-8")


def load_all_correct_ids_from_cache(cache_file: Path, fold: int, exp_dir: Path) -> Set[str]:
    """
    從快取檔讀取 all_doc_correct doc_id。

    嚴格模式：
    - 快取不存在 -> 直接報錯
    - 結構不符   -> 直接報錯
    """
    if not cache_file.exists():
        raise FileNotFoundError(
            f"找不到 all_doc_correct 快取檔: {cache_file} "
            f"(exp={exp_dir.name}, fold={fold})"
        )

    payload = json_load(cache_file)
    if not isinstance(payload, dict):
        raise RuntimeError(f"快取格式錯誤 (預期 dict): {cache_file}")

    doc_ids = payload.get("doc_ids")
    if not isinstance(doc_ids, list):
        raise RuntimeError(f"快取格式錯誤 (缺少 doc_ids:list): {cache_file}")

    # 一律轉字串，避免與 selection doc_id 型別不一致。
    return {str(x) for x in doc_ids}


def run_one_experiment(
    args: argparse.Namespace,
    exp_dir: Path,
    dataset_dir: Path,
    tokenizer=None,
) -> None:
    # 找最新 pseudo_results_*，預設也在這裡輸出
    pseudo_dir = find_pseudo_results_dir(exp_dir)
    if args.output_dir:
        # 若使用者指定輸出根目錄，先取 base
        out_base = Path(args.output_dir)
        # summary 批次模式下，避免不同 seed 寫到同名檔案而互相覆蓋
        output_dir = out_base / exp_dir.name if args.summary_file else out_base
    else:
        # 未指定輸出目錄時，直接輸出到 pseudo_results_*
        output_dir = pseudo_dir
    output_dir.mkdir(exist_ok=True)

    # 蒐集每 fold 的統計列，最後寫 summary
    summary_rows: List[FoldResult] = []

    print(f"\n[Experiment] {exp_dir}")
    print(f"[Pseudo dir] {pseudo_dir}")
    print(f"[Output dir] {output_dir}\n")

    # 逐 fold 分析
    for fold in range(args.fold_start, args.fold_end + 1):
        # 1) all_doc_correct doc_id 集合 (只讀快取，缺檔即報錯)
        cache_file = pseudo_dir / args.all_correct_cache_pattern.format(fold=fold)
        all_correct = load_all_correct_ids_from_cache(cache_file, fold, exp_dir)

        # 2) 讀 selection history 的被選中 doc_id 集合
        selection_file = pseudo_dir / args.selection_file_pattern.format(fold=fold)
        selected = parse_selected_doc_ids(selection_file)

        # 3) 差集：正確但未被選中
        missed = all_correct - selected

        # 兩份 doc_id 清單輸出檔路徑
        f_all = output_dir / f"unfiltered_all_doc_correct_fold{fold}.txt"
        f_missed = output_dir / f"correct_but_not_selected_fold{fold}.txt"

        # 寫 all_doc_correct 清單
        write_doc_id_list(
            f_all,
            title=f"Fold {fold} - unfiltered all_doc_correct doc_ids",
            doc_ids=all_correct,
        )
        # 寫 correct_but_not_selected 清單
        write_doc_id_list(
            f_missed,
            title=f"Fold {fold} - correct but not selected doc_ids",
            doc_ids=missed,
        )

        # 累積 fold 摘要
        unlabeled_path = build_unlabeled_file_path(dataset_dir, fold)
        summary_rows.append(
            FoldResult(
                fold=fold,
                total_docs=count_filtered_docs(unlabeled_path, tokenizer),
                all_doc_correct_count=len(all_correct),
                selected_count=len(selected),
                correct_but_not_selected_count=len(missed),
            )
        )

        # 即時列印 fold 統計
        print(
            f"fold{fold}: all_doc_correct={len(all_correct)}, "
            f"selected={len(selected)}, correct_but_not_selected={len(missed)}"
        )

    # 準備寫入本實驗的 summary
    summary_file = output_dir / "unselected_correct_summary.txt"
    lines: List[str] = []
    lines.append("=== 正確但未被選中 分析摘要 ===")
    lines.append(f"experiment_dir: {exp_dir}")
    lines.append(f"dataset_dir: {dataset_dir}")
    lines.append(f"selection_file_pattern: {args.selection_file_pattern}")
    lines.append(f"fold_range: {args.fold_start}~{args.fold_end}")
    lines.append("")

    # 用於統計 TOTAL 列
    total_correct = 0
    total_selected = 0
    total_missed = 0
    total_docs = 0

    # 寫入每 fold 統計並累加
    for r in summary_rows:
        lines.append(
            f"fold{r.fold}: total_docs={r.total_docs}, all_doc_correct={r.all_doc_correct_count}, "
            f"selected={r.selected_count}, correct_but_not_selected={r.correct_but_not_selected_count}"
        )
        total_docs += r.total_docs
        total_correct += r.all_doc_correct_count
        total_selected += r.selected_count
        total_missed += r.correct_but_not_selected_count

    lines.append("")
    # 寫入跨 fold 的總和
    lines.append(
        f"TOTAL: total_docs={total_docs}, all_doc_correct={total_correct}, "
        f"selected={total_selected}, correct_but_not_selected={total_missed}"
    )

    # 寫檔完成後，列印路徑
    summary_file.write_text("\n".join(lines), encoding="utf-8")
    print(f"完成，摘要輸出: {summary_file}")


def main() -> None:
    # 解析命令列參數
    args = parse_args()
    # fold 範圍檢查
    if args.fold_start > args.fold_end:
        raise ValueError("--fold-start 不能大於 --fold-end")
    # selection pattern 必須能帶入 fold
    if "{fold}" not in args.selection_file_pattern:
        raise ValueError("--selection-file-pattern 必須包含 {fold}")
    if "{fold}" not in args.all_correct_cache_pattern:
        raise ValueError("--all-correct-cache-pattern 必須包含 {fold}")

    # 解析目標實驗清單 (單一或 summary 批次)
    targets = resolve_target_experiments(args)
    # 建立資料集路徑物件
    dataset_dir = Path(args.dataset_dir)

    # 載入 tokenizer，用於過濾超過 512 token 的文件
    tokenizer = None
    if args.tokenizer_path and args.tokenizer_path.lower() != "none":
        from transformers import BertTokenizer
        print(f"載入 tokenizer: {args.tokenizer_path}")
        tokenizer = BertTokenizer.from_pretrained(args.tokenizer_path)
    else:
        print("未指定 tokenizer，total_docs 將使用原始 JSON 長度 (可能含超長文件)")

    # 顯示目標實驗數量
    print(f"目標實驗數: {len(targets)}")
    # 逐實驗執行完整流程
    for exp_dir in targets:
        run_one_experiment(
            args=args,
            exp_dir=exp_dir,
            dataset_dir=dataset_dir,
            tokenizer=tokenizer,
        )


if __name__ == "__main__":
    main()
