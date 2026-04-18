#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""公平比較兩組實驗在「不套篩選」下，未標註資料 3 個 [MASK] 全對的筆數。"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path as _Path

# 將專案根目錄加入 sys.path，確保不論從哪個位置執行都能找到 utils 底下的模組
sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from utils.unlabeled_dataset_compat import UnlabeledDatasetCompat, build_unlabeled_file_path

# torch.load 以 pickle 還原整個模型物件時，需要在執行環境中找到 prompt_bert 類別
# 使用獨立的 utils/prompt_bert_model.py，避免 import 訓練腳本時觸發其模組層級 argparse
from utils.prompt_bert_model import prompt_bert  # noqa: F401



# @dataclass自動產生__init__、__repr__等方法，方便統計結果的儲存與顯示
@dataclass
class FoldStat:
    total_docs: int = 0        # 可比較的文件總數
    all_doc_correct: int = 0   # 所有子句的三個 MASK 均正確的文件數
    # 每個 fold 的明細：fold -> (all_doc_correct, total_docs, ckpt_name)
    per_fold: Dict[int, Tuple[int, int, str]] = field(default_factory=dict)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="比較 baseline vs consistency 的 unfiltered 3-MASK 全對筆數")
    p.add_argument("--summary-baseline", nargs="+", default=None, help="一或多個 baseline summary 檔案路徑")
    p.add_argument("--summary-consistency", default=None, help="consistency summary 檔案路徑 (可選，不提供則只跑 baseline)")
    exp_root_group = p.add_mutually_exclusive_group(required=True)
    exp_root_group.add_argument("--experiments-root", help="單一實驗目錄根路徑，例如 ep_split10_t1te1v1_u7_disjoint")
    exp_root_group.add_argument(
        "--experiments-roots",
        nargs="+",
        help="多個實驗目錄根路徑，會依序搜尋 summary 內的實驗資料夾",
    )
    p.add_argument("--dataset-dir", required=True, help="資料集目錄，例如 split10_train1_test1_val1_unlabeled7_disjoint")
    p.add_argument("--tokenizer", default="bert-base-chinese", help="tokenizer 路徑或名稱")
    p.add_argument("--batch-size", type=int, default=8, help="推論 batch size")
    p.add_argument("--fold-start", type=int, default=1, help="起始 fold")
    p.add_argument("--fold-end", type=int, default=10, help="結束 fold")
    p.add_argument("--device", default="cuda", help="cuda 或 cpu")
    p.add_argument(
        "--report-dir",
        default="analysis_unfiltered_mask3_reports",
        help="輸出 txt 報表的資料夾",
    )
    p.add_argument(
        "--save-all-correct-cache",
        action="store_true",
        help="輸出每 fold 的 all_doc_correct doc_id 快取，供其他分析腳本重用",
    )
    p.add_argument(
        "--all-correct-cache-pattern",
        default="cache_unfiltered_all_doc_correct_fold{fold}.json",
        help="all_doc_correct 快取檔名樣板 (需包含 {fold})",
    )
    return p.parse_args()


def read_experiment_names_from_summary(summary_file: Path) -> List[str]:
    text = summary_file.read_text(encoding="utf-8", errors="ignore")
    names: List[str] = []
    for line in text.splitlines():
        m = re.match(r"\s*-\s*(prompt_ECPE_few_shot_ST_[^\s]+)", line)
        if m:
            names.append(m.group(1).strip())
    return names


def resolve_experiments_roots(args: argparse.Namespace) -> List[Path]:
    """解析並驗證實驗根目錄 (支援單一路徑與多路徑) """
    # 如果使用者提供了 --experiments-roots (多路徑模式)
    if args.experiments_roots:
        # 將每個字串路徑轉成 Path 物件，方便後續做檔案系統操作
        roots = [Path(p) for p in args.experiments_roots]
    # 否則走 --experiments-root (單一路徑模式)
    else:
        # 將單一路徑包成 list，讓後續流程可以統一用同一套邏輯處理
        roots = [Path(args.experiments_root)]

    # 檢查每個 root 是否真的是存在的資料夾; 不存在者收集到 missing
    missing = [str(p) for p in roots if not p.is_dir()]
    # 只要有任一路徑不存在，就直接拋出錯誤並列出所有缺失路徑
    if missing:
        # 透過換行把多個不存在路徑清楚列出，方便使用者一次修正
        raise FileNotFoundError(
            "以下 experiments root 不存在:\n" + "\n".join(missing)
        )
    # 全部驗證通過後，回傳可用的 root 路徑清單
    return roots


def resolve_experiment_dir(exp_name: str, roots: List[Path]) -> Path:
    """在多個根目錄中解析單一實驗資料夾"""
    # 逐一嘗試把 exp_name 接到每個 root 底下，並只保留「實際存在且是資料夾」的候選路徑
    hits = [root / exp_name for root in roots if (root / exp_name).is_dir()]
    # 如果完全找不到任何候選路徑
    if not hits:
        # 把所有「嘗試過的完整路徑」整理成多行字串，方便錯誤訊息一次看清楚
        searched = "\n".join(str(root / exp_name) for root in roots)
        # 拋出 FileNotFoundError，明確指出找不到哪個實驗名稱，以及已搜尋的路徑清單
        raise FileNotFoundError(
            f"找不到實驗目錄: {exp_name}\n已搜尋:\n{searched}"
        )

    # 如果同一個實驗名稱在多個 roots 都存在
    if len(hits) > 1:
        # 印出警告提醒使用者目前有重複，程式將採用第一個命中的路徑
        print(
            f"[警告] 實驗目錄 {exp_name} 在多個 roots 都存在，"
            f"將使用第一個: {hits[0]}"
        )
    # 回傳最終選定的實驗資料夾路徑 (第一個命中)
    return hits[0]


def parse_checkpoint_file(ckpt_txt: Path) -> str | None:
    if not ckpt_txt.exists():
        return None
    for line in ckpt_txt.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("Checkpoint path:"):
            raw = line.split("Checkpoint path:", 1)[1].strip()
            return Path(raw).name
    return None


def resolve_checkpoint(exp_dir: Path, fold: int, is_consistency: bool) -> Path:
    if is_consistency:
        meta = exp_dir / f"fold{fold}_best_val_checkpoint_pair.txt"
        default_name = f"fold{fold}_self_training_best_pair.pth"
    else:
        meta = exp_dir / f"fold{fold}_best_val_checkpoint.txt"
        default_name = f"fold{fold}_self_training_best.pth"

    parsed_name = parse_checkpoint_file(meta)
    name = parsed_name if parsed_name else default_name

    cand1 = exp_dir / "self_training_models" / name
    cand2 = exp_dir / name
    if cand1.exists():
        return cand1
    if cand2.exists():
        return cand2
    raise FileNotFoundError(f"找不到 checkpoint: {cand1} / {cand2}")


def find_pseudo_results_dir(exp_dir: Path) -> Path:
    candidates = sorted([p for p in exp_dir.glob("pseudo_results_*") if p.is_dir()])
    if not candidates:
        raise FileNotFoundError(f"{exp_dir} 底下找不到 pseudo_results_* 資料夾")
    return candidates[-1]


def sort_doc_ids(doc_ids: List[str]) -> List[str]:
    def key(v: str):
        try:
            return (0, int(v), v)
        except ValueError:
            return (1, 10**18, v)

    return sorted(doc_ids, key=key)


def write_all_correct_cache(
    exp_dir: Path,
    fold: int,
    ckpt: Path,
    doc_ids: List[str],
    cache_pattern: str,
) -> Path:
    import json

    pseudo_dir = find_pseudo_results_dir(exp_dir)
    cache_file = pseudo_dir / cache_pattern.format(fold=fold)
    payload = {
        "experiment_dir": str(exp_dir),
        "fold": fold,
        "checkpoint": str(ckpt),
        "count": len(doc_ids),
        "doc_ids": sort_doc_ids(doc_ids),
    }
    cache_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return cache_file


def build_gt_mask_labels(unlabeled_json: Path, tokenizer: BertTokenizer) -> Dict[str, np.ndarray]:
    # 讀取單一 fold 的未標註資料 (實際上含有可用來比對的 pairs 真值)
    data = json_load(unlabeled_json)
    # 先把會用到的特殊 token 轉成 id，避免在迴圈中重複轉換
    yes_id = tokenizer.convert_tokens_to_ids("是")
    no_id = tokenizer.convert_tokens_to_ids("非")
    none_id = tokenizer.convert_tokens_to_ids("无")

    # 輸出格式: doc_id -> 該文件所有子句展平後的 [emotion, cause, pair] id 序列
    out: Dict[str, np.ndarray] = {}
    # 逐篇文件建立對應的 GT 三元組標籤
    for doc in data:
        # 取出文件編號，作為字典 key
        doc_id = str(doc["doc_id"])
        # 將 pairs 轉成 tuple，方便後續索引與比較
        pairs = [tuple(p) for p in doc.get("pairs", [])]
        # 拆成「情緒句索引列表」與「原因句索引列表」；若無 pairs 則給空列表
        pos, cause = zip(*pairs) if pairs else ([], [])
        pos = list(pos)
        cause = list(cause)

        # 存放此文件每個子句對應的 3 個 mask 的正確答案 (tokenizer id)
        triples: List[int] = []
        # 子句索引從 1 開始，與訓練資料組裝邏輯一致
        for i in range(1, int(doc["doc_len"]) + 1):
            # 第 1 個 [MASK]: 此子句是否為情緒句 (是/非)
            emo_id = yes_id if i in pos else no_id
            # 第 2 個 [MASK]: 此子句是否為原因句 (是/非)
            cause_id = yes_id if i in cause else no_id
            # 第 3 個 [MASK]: 若是原因句，填對應情緒句編號；否則填「无」
            if i in cause:
                # 取出造成這個原因句的情緒句位置（以字串 token 編號表示)
                src_pos = str(pos[cause.index(i)])
                # 將情緒句位置 (例如 "7") 轉成 tokenizer id
                pair_id = tokenizer.convert_tokens_to_ids(src_pos)
            else:
                # 非原因句的 pair 標籤固定是「无」
                pair_id = none_id
            # 將此子句的 3 個標籤依序加入 (對齊 3 個 [MASK])
            triples.extend([emo_id, cause_id, pair_id])

        # 轉為 numpy 陣列，便於後續與模型預測向量化比較
        out[doc_id] = np.array(triples, dtype=np.int64)
    # 回傳整個 fold 的 doc_id -> GT 3-mask 序列映射
    return out


def json_load(path: Path):
    import json

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def eval_one_experiment(
    exp_dir: Path,
    dataset_dir: Path,
    tokenizer: BertTokenizer,
    fold_start: int,
    fold_end: int,
    batch_size: int,
    device: torch.device,
    is_consistency: bool,
    save_all_correct_cache: bool,
    all_correct_cache_pattern: str,
) -> FoldStat:
    # 建立累計統計物件 (跨 fold 累加)
    # - total_triples: 所有可比較三元組總數
    # - all3_correct: 三個 [MASK] 同時正確的三元組數
    stat = FoldStat()
    # 取得 tokenizer 中 [MASK] 的 token id，後續用來抓每筆輸入的 mask 位置
    mask_id = tokenizer.mask_token_id

    # 逐 fold 執行評估 (例如 fold1~fold10)
    for fold in range(fold_start, fold_end + 1):
        # fold 層級計數器：紀錄「本 fold」的 all_doc_correct / total_docs
        fold_total_docs = 0
        fold_all_doc_correct = 0
        fold_all_doc_correct_ids: List[str] = []

        # 依照與訓練一致的規則組出該 fold 的 unlabeled 檔案路徑
        unlabeled_json = build_unlabeled_file_path(dataset_dir, fold)
        # 為公平比較：若缺任一 fold 的資料，直接中止
        if not unlabeled_json.exists():
            raise FileNotFoundError(
                f"找不到未標註資料檔案: {unlabeled_json} (fold={fold})，"
                "為了公平比較已停止執行"
            )

        # 建立 doc_id -> GT 三元組 token 序列的映射（作為比對標準答案）
        gt_map = build_gt_mask_labels(unlabeled_json, tokenizer)
        # 用與原訓練程式碼相容的未標註 Dataset 載入資料
        ds = UnlabeledDatasetCompat(unlabeled_json, tokenizer)
        # 建立推論 DataLoader (不打亂，確保索引對應可追蹤)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

        # 解析該 fold 應使用的 checkpoint (baseline / consistency 規則不同)
        ckpt = resolve_checkpoint(exp_dir, fold, is_consistency)
        # 載入模型到指定裝置
        model = torch.load(str(ckpt), map_location=device)
        # 切換為 eval 模式 (關閉 dropout / BN 訓練行為)
        model.eval()
        # 若使用 GPU，將模型搬到 CUDA
        if device.type == "cuda":
            model = model.cuda()

        # ptr 用來追蹤目前 batch 在整個 dataset 的起始索引
        # 因為 DataLoader 只回傳 x_bert，需要靠 ptr 對回 ds.doc_id
        ptr = 0
        # 推論階段關閉梯度
        with torch.no_grad():
            # 逐 batch 跑未標註樣本推論
            for batch in loader:
                # 將 batch 放到對應裝置
                x = batch.cuda() if device.type == "cuda" else batch
                # 前向推論：取得 logits (不提供 labels，純推論)
                _, logits = model(x, labels=None)
                # 在 vocab 維度取 argmax 作為每個位置的預測 token id
                pred_ids = torch.argmax(logits, dim=-1).detach().cpu().numpy()
                # 保留原輸入的 numpy 版本，用來找 [MASK] 位置
                x_np = batch.detach().cpu().numpy()

                # 逐筆樣本做 3-MASK 三元組比對
                for i in range(x_np.shape[0]):
                    # 用 ptr + i 對回這筆樣本在 dataset 中的 doc_id
                    doc_id = str(ds.doc_id[ptr + i])
                    # 取得該 doc 的 GT 三元組 token 序列
                    gt = gt_map.get(doc_id)
                    # 依你的需求：若找不到 GT 視為資料異常，直接報錯中止
                    if gt is None:
                        raise RuntimeError(
                            f"找不到 GT: doc_id={doc_id}，fold={fold}，實驗目錄={exp_dir}"
                        )
                    # 找出此筆輸入中所有 [MASK] 位置
                    mask_pos = np.where(x_np[i] == mask_id)[0]
                    # 取出模型在這些 [MASK] 位置的預測 token id
                    pred_mask = pred_ids[i, mask_pos]
                    # 依你的需求改為嚴格檢查：
                    # 1) 預測與 GT 長度必須一致
                    # 2) 長度必須是 3 的倍數 (每個子句 3 個 [MASK])
                    pred_len = len(pred_mask)
                    gt_len = len(gt)
                    if pred_len != gt_len:
                        raise RuntimeError(
                            f"MASK 長度不一致: pred_len={pred_len}, gt_len={gt_len}, "
                            f"doc_id={doc_id}, fold={fold}, exp={exp_dir}"
                        )
                    if pred_len % 3 != 0:
                        raise RuntimeError(
                            f"MASK 長度不是 3 的倍數: len={pred_len}, "
                            f"doc_id={doc_id}, fold={fold}, exp={exp_dir}"
                        )

                    triple_n = pred_len // 3
                    # 若連一組完整三元組都沒有，視為資料異常直接報錯
                    if triple_n <= 0:
                        raise RuntimeError(
                            f"沒有可比較的完整三元組: pred_len={pred_len}, "
                            f"doc_id={doc_id}, fold={fold}, exp={exp_dir}"
                        )

                    # 轉成 [N, 3] 形狀，每列代表一個子句的三個 [MASK]
                    pred_3 = pred_mask.reshape(-1, 3)
                    gt_3 = gt.reshape(-1, 3)
                    # 判斷每個三元組是否「三個都正確」
                    hit = np.all(pred_3 == gt_3, axis=1)

                    # 以文件為單位計算：整篇文件所有子句的三個 MASK 均正確才算 correct
                    doc_all_correct = int(np.all(hit))
                    stat.total_docs += 1
                    stat.all_doc_correct += doc_all_correct
                    fold_total_docs += 1
                    fold_all_doc_correct += doc_all_correct
                    if doc_all_correct:
                        fold_all_doc_correct_ids.append(doc_id)

                # 更新全域索引偏移量，供下一個 batch 對回正確 doc_id
                ptr += x_np.shape[0]

            # 記錄本 fold 明細，供後續報表列出每輪（fold）結果
            stat.per_fold[fold] = (fold_all_doc_correct, fold_total_docs, str(ckpt))

            # 需要時輸出快取，供 analyze_unselected_correct_docs.py 直接讀取
            if save_all_correct_cache:
                cache_path = write_all_correct_cache(
                    exp_dir=exp_dir,
                    fold=fold,
                    ckpt=ckpt,
                    doc_ids=fold_all_doc_correct_ids,
                    cache_pattern=all_correct_cache_pattern,
                )
                print(
                    f"[cache] exp={exp_dir.name} fold{fold}: "
                    f"count={len(fold_all_doc_correct_ids)} -> {cache_path.name}"
                )

    # 回傳此實驗 (單 seed) 在所有 fold 的累計統計
    return stat


def build_group_result_text(name: str, stats: List[FoldStat]) -> str:
    # 取出每個 seed 的「可比較文件總數」
    totals = [s.total_docs for s in stats]
    # 取出每個 seed 的「所有子句均正確的文件數」
    rights = [s.all_doc_correct for s in stats]
    # 計算每個 seed 的正確率；若分母為 0 則回傳 0.0 避免除以 0
    ratios = [r / t if t > 0 else 0.0 for r, t in zip(rights, totals)]

    # pooled 統計: 先把所有 seed 的分子、分母加總，再算整體正確率
    total_all = sum(totals)
    right_all = sum(rights)
    ratio_all = right_all / total_all if total_all > 0 else 0.0

    # seed-level 統計: 計算各 seed 正確率的平均與標準差
    mean_ratio = float(np.mean(ratios)) if ratios else 0.0
    std_ratio = float(np.std(ratios)) if ratios else 0.0

    lines: List[str] = []
    # 建立群組標題 (例如 baseline / consistency)
    lines.append(f"[{name}]")
    # 逐一建立每個 seed 的原始計數與正確率
    for i, (r, t, p) in enumerate(zip(rights, totals, ratios), start=1):
        lines.append(f"  seed{i}: all_doc_correct={r}, total_docs={t}, acc={p*100:.2f}%")

        # 加入每個 seed 的每 fold 明細 (每一輪未標註樣本結果)
        seed_stat = stats[i - 1]
        if seed_stat.per_fold:
            for f in sorted(seed_stat.per_fold.keys()):
                fr, ft, ckpt_name = seed_stat.per_fold[f]
                fp = (fr / ft * 100.0) if ft > 0 else 0.0
                lines.append(
                    f"    fold{f}: all_doc_correct={fr}, total_docs={ft}, acc={fp:.2f}% [ckpt: {ckpt_name}]"
                )

    # 建立 pooled 結果 (跨 seed 合併後的整體表現)
    lines.append(
        f"  pooled: all_doc_correct={right_all}, total_docs={total_all}, acc={ratio_all*100:.2f}%"
    )
    # 建立 seed 間變異 (mean ± std)
    lines.append(f"  seed mean±std: {mean_ratio*100:.2f}% ± {std_ratio*100:.2f}%")
    return "\n".join(lines)


def print_group_result(name: str, stats: List[FoldStat]) -> str:
    text = build_group_result_text(name, stats)
    print(f"\n{text}")
    return text


def main() -> None:
    args = parse_args()

    if "{fold}" not in args.all_correct_cache_pattern:
        raise ValueError("--all-correct-cache-pattern 必須包含 {fold}")
    if not args.summary_baseline and not args.summary_consistency:
        raise ValueError("至少提供 --summary-baseline 或 --summary-consistency 其中之一")

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(args.tokenizer)

    roots = resolve_experiments_roots(args)
    ds_dir = Path(args.dataset_dir)

    # 若未提供 baseline，允許使用 consistency summary 作為單組分析輸入
    baseline_files = args.summary_baseline if args.summary_baseline else [args.summary_consistency]
    consistency_only_mode = bool((not args.summary_baseline) and args.summary_consistency)

    # ── 處理 consistency (可選) ──
    has_consistency = bool(args.summary_baseline and args.summary_consistency)
    consistency_names: List[str] = []
    consistency_stats: List[FoldStat] = []
    if has_consistency:
        consistency_names = read_experiment_names_from_summary(Path(args.summary_consistency))
        if len(consistency_names) < 1:
            raise RuntimeError("summary_consistency 中未解析到任何 seed 目錄")
        for n in consistency_names:
            exp_dir = resolve_experiment_dir(n, roots)
            consistency_stats.append(
                eval_one_experiment(
                    exp_dir=exp_dir,
                    dataset_dir=ds_dir,
                    tokenizer=tokenizer,
                    fold_start=args.fold_start,
                    fold_end=args.fold_end,
                    batch_size=args.batch_size,
                    device=device,
                    is_consistency=True,
                    save_all_correct_cache=args.save_all_correct_cache,
                    all_correct_cache_pattern=args.all_correct_cache_pattern,
                )
            )

    # ── 逐一處理每個 baseline summary ──
    for baseline_file in baseline_files:
        baseline_path = Path(baseline_file)
        print(f"\n{'='*70}")
        print(f"處理 baseline: {baseline_path.name}")
        print(f"{'='*70}")

        baseline_names = read_experiment_names_from_summary(baseline_path)
        if len(baseline_names) < 1:
            print(f"  [警告] {baseline_path} 中未解析到任何 seed 目錄，跳過")
            continue

        baseline_stats: List[FoldStat] = []
        for n in baseline_names:
            exp_dir = resolve_experiment_dir(n, roots)
            baseline_stats.append(
                eval_one_experiment(
                    exp_dir=exp_dir,
                    dataset_dir=ds_dir,
                    tokenizer=tokenizer,
                    fold_start=args.fold_start,
                    fold_end=args.fold_end,
                    batch_size=args.batch_size,
                    device=device,
                    is_consistency=consistency_only_mode,
                    save_all_correct_cache=args.save_all_correct_cache,
                    all_correct_cache_pattern=args.all_correct_cache_pattern,
                )
            )

        if args.summary_baseline:
            baseline_group_name = "baseline (unfiltered)"
        elif consistency_only_mode:
            baseline_group_name = "consistency (unfiltered)"
        baseline_text = print_group_result(baseline_group_name, baseline_stats)
        consistency_text = ""
        if has_consistency:
            consistency_text = print_group_result("consistency (unfiltered)", consistency_stats)

        # 從 summary_baseline 的上層目錄名稱擷取資料集標籤
        baseline_parent = baseline_path.parent.name
        dataset_tag = re.sub(r'^results_', '', baseline_parent)

        # 從 summary_baseline 檔名擷取 wd 之後、_summary.txt 之前的字串
        baseline_stem = baseline_path.name
        m_suffix = re.search(r'wd[\d.]+_(.+?)_summary\.txt$', baseline_stem)
        file_suffix = m_suffix.group(1) if m_suffix else "unknown"

        # 建立報表輸出資料夾
        report_dir = Path(args.report_dir) / dataset_tag
        report_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = report_dir / f"unfiltered_mask3_compare_{file_suffix}_{ts}.txt"

        # 組合報表內容
        lines: List[str] = []
        lines.append("=== Unfiltered 3-MASK 全對比較報表 ===")
        lines.append(f"generated_at: {datetime.now().isoformat(timespec='seconds')}")
        lines.append(f"summary_baseline: {baseline_file}")
        lines.append(f"summary_consistency: {args.summary_consistency if has_consistency else '(未提供)'}")
        lines.append("experiments_roots:")
        for r in roots:
            lines.append(f"  - {r}")
        lines.append(f"dataset_dir: {args.dataset_dir}")
        lines.append(f"fold_range: {args.fold_start}~{args.fold_end}")
        lines.append("")

        if has_consistency:
            lines.append("[Seed Name Pairing: baseline vs consistency]")
            for i, (b, c) in enumerate(zip(baseline_names, consistency_names), start=1):
                lines.append(f"  seed{i}: {b}  <->  {c}")
            lines.append("")

        lines.append("[baseline_names]")
        for i, b in enumerate(baseline_names, start=1):
            lines.append(f"  seed{i}: {b}")
        lines.append("")

        if has_consistency:
            lines.append("[consistency_names]")
            for i, c in enumerate(consistency_names, start=1):
                lines.append(f"  seed{i}: {c}")
            lines.append("")

        lines.append(baseline_text)
        if consistency_text:
            lines.append("")
            lines.append(consistency_text)
        lines.append("")

        report_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"\n報表已輸出: {report_path}")


if __name__ == "__main__":
    main()
