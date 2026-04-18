#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
由 pseudo_labeled_samples_fold{fold}_round{round}.json 回推 doc_selection_history_fold{fold}.txt

支援兩種模式:
1) 單一實驗資料夾
2) 一次處理 experiments root 底下所有 prompt_ECPE_few_shot_ST* 實驗資料夾

輸出檔案:
    doc_selection_history_fold{fold}_reconstructed_from_json.txt

預設不覆蓋原始 doc_selection_history_fold{fold}.txt
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="回推 doc_selection_history_fold{i}.txt")

    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--experiment-dir",
        type=str,
        help="單一實驗資料夾，例如 ep_split10.../prompt_ECPE_few_shot_ST_...",
    )
    exp_root_group = group.add_mutually_exclusive_group()
    exp_root_group.add_argument(
        "--experiments-root",
        type=str,
        help="單一實驗根目錄，例如 ep_split10_t1te1v1_u7_disjoint",
    )
    exp_root_group.add_argument(
        "--experiments-roots",
        nargs="+",
        help="多個實驗根目錄，會依序搜尋 prompt_ECPE_few_shot_ST* 目錄",
    )

    p.add_argument(
        "--all-prompts",
        action="store_true",
        help="搭配 --experiments-root: 處理所有 prompt_ECPE_few_shot_ST* 目錄",
    )
    p.add_argument("--fold-start", type=int, default=1, help="起始 fold (預設 1)")
    p.add_argument("--fold-end", type=int, default=10, help="結束 fold (預設 10)")
    p.add_argument(
        "--sort-by",
        choices=["doc_id", "count"],
        default="doc_id",
        help="詳細記錄排序方式:doc_id(小到大) 或 count(選中次數降序)",
    )
    p.add_argument(
        "--output-suffix",
        type=str,
        default="_reconstructed_from_json",
        help="輸出檔名後綴 (預設: _reconstructed_from_json)",
    )
    return p.parse_args()


def find_target_experiments(args: argparse.Namespace) -> List[Path]:
    if args.experiment_dir:
        exp = Path(args.experiment_dir)
        if not exp.is_dir():
            raise FileNotFoundError(f"找不到實驗資料夾: {exp}")
        return [exp]

    roots: List[Path] = []
    if args.experiments_roots:
        roots = [Path(p) for p in args.experiments_roots]
    elif args.experiments_root:
        roots = [Path(args.experiments_root)]

    if not roots:
        raise ValueError("使用 experiments root 模式時，請提供 --experiments-root 或 --experiments-roots")

    missing_roots = [str(r) for r in roots if not r.is_dir()]
    if missing_roots:
        raise FileNotFoundError(
            "以下 experiments root 不存在:\n" + "\n".join(missing_roots)
        )

    if not args.all_prompts:
        raise ValueError("使用 --experiments-root(s) 時，請加上 --all-prompts")

    # 從多個 roots 收集所有 prompt 目錄，去重並保持排序穩定
    exps: List[Path] = []
    seen = set()
    for root in roots:
        for p in sorted(root.iterdir()):
            if p.is_dir() and p.name.startswith("prompt_ECPE_few_shot_ST"):
                if p not in seen:
                    exps.append(p)
                    seen.add(p)

    if not exps:
        root_str = "\n".join(str(r) for r in roots)
        raise RuntimeError(f"在以下 roots 下找不到 prompt_ECPE_few_shot_ST* 目錄:\n{root_str}")
    return exps


def find_pseudo_results_dir(exp_dir: Path) -> Path:
    candidates = sorted([p for p in exp_dir.glob("pseudo_results_*") if p.is_dir()])
    if not candidates:
        raise FileNotFoundError(f"{exp_dir} 底下找不到 pseudo_results_* 資料夾")
    # 若有多個，使用名稱排序最後一個（通常是最新）
    return candidates[-1]


def collect_selected_rounds(
    pseudo_dir: Path,
    fold: int,
) -> Tuple[int, Dict[str, List[int]]]:
    # 建立檔名比對規則: 只抓當前 fold 的 pseudo_labeled_samples_fold{fold}_round{n}.json
    pattern = re.compile(rf"^pseudo_labeled_samples_fold{fold}_round(\d+)\.json$")

    # round_files 用來存 (輪次, 檔案路徑)
    round_files: List[Tuple[int, Path]] = []
    # 走訪 pseudo_dir 底下所有檔案/資料夾
    for p in pseudo_dir.iterdir():
        # 嘗試用正則比對檔名
        m = pattern.match(p.name)
        # 若檔名符合，代表是目標 round 的 JSON 檔
        if m:
            # 取出 round 編號並轉成 int，與路徑一起記錄
            round_files.append((int(m.group(1)), p)) # m.group(1) 是 round 的數字部分

    # 依 round 編號由小到大排序，確保處理順序一致
    round_files.sort(key=lambda x: x[0]) #把round編號由小到大排序(round1, round2, ...)
    # 若完全沒有找到任何 round JSON，回傳 0 與空 dict
    if not round_files:
        return 0, {}

    # selected: doc_id -> 被選中的 round 集合 (用 set 避免重複)
    selected: Dict[str, set] = defaultdict(set)
    # 逐輪讀取 JSON，蒐集每個 doc_id 被選中的輪次
    for round_idx, fp in round_files:
        # 讀取單一 round 的 pseudo 樣本紀錄
        data = json.loads(fp.read_text(encoding="utf-8"))
        # 逐筆樣本處理
        for item in data:
            # 取出 doc_id，統一轉字串避免型別差異
            doc_id = str(item.get("doc_id"))
            # 將該 round 加入此 doc_id 的輪次集合
            selected[doc_id].add(round_idx)

    # 把 set 轉成排序後 list，輸出格式為 doc_id -> [round1, round2, ...]
    selected_rounds: Dict[str, List[int]] = {
        doc_id: sorted(round_set) for doc_id, round_set in selected.items()
    }
    # 回傳「總輪數」與「每個 doc_id 的被選輪次」
    return len(round_files), selected_rounds


def build_history_text(fold: int, total_rounds: int, selected_rounds: Dict[str, List[int]], sort_by: str) -> str:
    # 統計「被選中幾次 -> 有幾個 doc」的分佈
    counts = Counter(len(v) for v in selected_rounds.values())

    # lines 用來逐行組裝最終輸出的文字內容
    lines: List[str] = []
    # 寫入標題
    lines.append(f"樣本選中歷史記錄 - Fold {fold}")
    # 寫入 self-training 的總輪數
    lines.append(f"Self-training 總輪數: {total_rounds}")
    # 寫入曾被選中的 doc_id 總數
    lines.append(f"總共 {len(selected_rounds)} 個 doc_id 曾被選中")
    # 分隔線
    lines.append("=" * 60)
    # 空行
    lines.append("")

    # 區塊標題：選中次數分佈
    lines.append("選中次數分佈:")
    # 依「被選次數」由大到小列出統計
    for k in sorted(counts.keys(), reverse=True):
        # 例如：被選中 3 次: 120 個樣本
        lines.append(f"  被選中 {k} 次: {counts[k]} 個樣本")

    # 空行
    lines.append("")
    # 分隔線
    lines.append("=" * 60)
    # 空行
    lines.append("")
    # 區塊標題: 詳細記錄
    lines.append("詳細記錄 (按選中次數降序):")
    # 次分隔線
    lines.append("-" * 60)

    # 先取出所有 doc_id，稍後依規則排序
    docs = list(selected_rounds.keys())

    # 若指定按 doc_id 排序
    if sort_by == "doc_id":
        # 定義 doc_id 排序鍵
        def sort_key(doc_id: str):
            try:
                # 可轉 int 的 doc_id 先排，依數值遞增，再用原字串作次序保險
                return (0, int(doc_id), doc_id)
            except ValueError:
                # 非純數字 doc_id 排到後面，保留字串字典序
                return (1, 10**18, doc_id)
        # 套用排序鍵進行排序
        docs.sort(key=sort_key)
    else:
        # 否則按「被選次數」排序（由多到少）
        def sort_key(doc_id: str):
            # 取出該 doc 的被選輪次清單
            rounds = selected_rounds[doc_id]
            try:
                # 嘗試把 doc_id 轉數字，方便同分時穩定排序
                doc_num = int(doc_id)
            except ValueError:
                # 非數字 doc_id 用大值，讓數字型 doc_id 優先
                doc_num = 10**18
            # 排序規則: 
            # 1) -len(rounds): 被選次數多的在前
            # 2) rounds: 輪次序列較小者在前
            # 3) doc_num: 數字 id 由小到大
            # 4) doc_id: 最後字串保底
            return (-len(rounds), rounds, doc_num, doc_id)
        # 套用排序鍵進行排序
        docs.sort(key=sort_key)

    # 逐筆輸出每個 doc 的選中次數與輪次
    for doc_id in docs:
        # 取得該 doc 的輪次清單
        rounds = selected_rounds[doc_id]
        # 寫入一行詳細記錄
        lines.append(f"Doc {doc_id}: 被選中 {len(rounds)} 次, 輪次={rounds}")

    # 將所有行以換行字元串成最終文字並回傳
    return "\n".join(lines)


def reconstruct_for_experiment(
    exp_dir: Path,
    fold_start: int,
    fold_end: int,
    sort_by: str,
    output_suffix: str,
) -> None:
    pseudo_dir = find_pseudo_results_dir(exp_dir)
    print(f"\n[Experiment] {exp_dir}")
    print(f"  pseudo_results_dir: {pseudo_dir}")

    for fold in range(fold_start, fold_end + 1):
        total_rounds, selected_rounds = collect_selected_rounds(pseudo_dir, fold)

        if total_rounds == 0:
            print(f"  - fold{fold}: 找不到 pseudo_labeled_samples_fold{fold}_round*.json，跳過")
            continue

        text = build_history_text(fold, total_rounds, selected_rounds, sort_by)
        out_path = pseudo_dir / f"doc_selection_history_fold{fold}{output_suffix}.txt"
        out_path.write_text(text, encoding="utf-8")
        print(
            f"  - fold{fold}: 已輸出 {out_path.name} "
            f"(docs={len(selected_rounds)}, rounds={total_rounds})"
        )


def main() -> None:
    args = parse_args()

    if args.fold_start > args.fold_end:
        raise ValueError("--fold-start 不能大於 --fold-end")

    targets = find_target_experiments(args)
    print(f"目標實驗數: {len(targets)}")

    for exp in targets:
        reconstruct_for_experiment(
            exp_dir=exp,
            fold_start=args.fold_start,
            fold_end=args.fold_end,
            sort_by=args.sort_by,
            output_suffix=args.output_suffix,
        )

    print("\n完成")


if __name__ == "__main__":
    main()
