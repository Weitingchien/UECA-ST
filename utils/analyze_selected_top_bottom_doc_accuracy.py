#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 延後型別註解求值，避免舊版 Python 對 list[dict] 報錯
from __future__ import annotations

# 匯入命令列參數模組
import argparse
# 匯入 JSON 讀寫模組
import json
import re
# 匯入路徑處理工具
from pathlib import Path


SUMMARY_EXPERIMENT_PATTERN = re.compile(r"^\s*-\s*(prompt_ECPE_few_shot_ST_[^\s]+)\s*$")
SEED_PATTERN = re.compile(r"seed(\d+)")


# 定義命令列參數解析函式
def parse_args() -> argparse.Namespace:
    # 建立參數解析器並設定說明文字
    parser = argparse.ArgumentParser(description="統計 selected=True 前/後N筆文檔準確率（依機率或散度排序）")
    # 加入實驗目錄參數 (可傳 prompt 目錄或 pseudo_results 目錄)
    parser.add_argument("--experiment-dir", required=True, help="實驗資料夾、pseudo_results_* 資料夾，或 multi-seed summary.txt")
    parser.add_argument(
        "--experiments-root",
        action="append",
        default=[],
        help="summary.txt 模式下的實驗根目錄；可重複指定多個，會依指定順序搜尋",
    )
    # 加入 fold 參數（若不提供則走 fold-start~fold-end）
    parser.add_argument("--fold", type=int, default=None, help="fold 編號，例如 1")
    # 加入 round 參數（若不提供則走 round-start~round-end）
    parser.add_argument("--round", type=int, default=None, help="round 編號，例如 1")
    # 加入 fold 起訖參數，預設 1~10
    parser.add_argument("--fold-start", type=int, default=1, help="起始 fold，預設 1")
    parser.add_argument("--fold-end", type=int, default=10, help="結束 fold，預設 10")
    # 加入 round 起訖參數，預設 1~10
    parser.add_argument("--round-start", type=int, default=1, help="起始 round，預設 1")
    parser.add_argument("--round-end", type=int, default=10, help="結束 round，預設 10")
    # 加入前/後樣本數參數，預設 100
    parser.add_argument("--top-k", type=int, default=100, help="前/後各取幾筆，預設 100")
    # 加入排序鍵參數，預設以機率由高到低排序
    parser.add_argument("--sort-by", choices=["probability", "divergence"], default="probability", help="排序依據")
    # 加入輸出檔名參數，未提供時自動命名
    parser.add_argument("--output", default=None, help="輸出報告檔名（放在 pseudo_results_* 內）")
    # 回傳解析後參數
    parser.add_argument(
        "--skip-missing-rounds",
        action="store_true",
        help="Skip fold/round units whose divergence or pseudo file is missing",
    )
    return parser.parse_args()


# 定義將輸入目錄解析為 pseudo_results 目錄的函式
def resolve_pseudo_dir(experiment_dir: Path) -> Path:
    # 若傳入本身就是 pseudo_results 目錄，直接回傳
    if experiment_dir.name.startswith("pseudo_results_") and experiment_dir.is_dir():
        # 直接使用該目錄
        return experiment_dir
    # 若傳入的是 prompt 實驗目錄，搜尋底下 pseudo_results 子目錄
    candidates = sorted([p for p in experiment_dir.glob("pseudo_results_*") if p.is_dir()])
    # 若找不到任何 pseudo_results 子目錄，拋出錯誤
    if not candidates:
        # 提示使用者路徑不正確或缺資料
        raise FileNotFoundError(f"找不到 pseudo_results_* 目錄: {experiment_dir}")
    # 回傳排序後最後一個（通常為最新結果）
    return candidates[-1]


def extract_experiment_names(summary_path: Path) -> list[str]:
    """從 multi-seed summary.txt 解析出各 seed 的實驗資料夾名稱。"""

    experiment_names: list[str] = []
    for line in summary_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = SUMMARY_EXPERIMENT_PATTERN.match(line)
        if match:
            experiment_names.append(match.group(1))

    if not experiment_names:
        raise ValueError(f"summary 中找不到實驗資料夾: {summary_path}")

    return experiment_names


def extract_seed(experiment_name: str) -> str:
    """從實驗資料夾名稱抽出 seed 名稱，例如 seed20。"""

    match = SEED_PATTERN.search(experiment_name)
    if match is None:
        raise ValueError(f"無法從實驗資料夾名稱解析 seed: {experiment_name}")
    return f"seed{match.group(1)}"


def seed_sort_key(seed_name: str) -> tuple[int, str]:
    """讓 seed20、seed42、seed60 依數字排序。"""

    match = SEED_PATTERN.search(seed_name)
    if match:
        return (int(match.group(1)), seed_name)
    return (10**9, seed_name)


def resolve_summary_sources(summary_path: Path, experiments_roots: list[Path] | None = None) -> list[dict]:
    """把 multi-seed summary.txt 解析成可分析的 seed/pseudo_dir 清單。"""

    candidate_roots = []
    if experiments_roots is not None:
        candidate_roots.extend(experiments_roots)
    candidate_roots.append(summary_path.parent)
    if summary_path.parent.name.startswith("results_"):
        candidate_roots.append(summary_path.parent.parent / summary_path.parent.name[len("results_") :])
    sources = []
    seen_seeds = set()

    for experiment_name in extract_experiment_names(summary_path):
        seed_name = extract_seed(experiment_name)
        if seed_name in seen_seeds:
            raise ValueError(f"summary 中出現重複 seed: {seed_name}")
        seen_seeds.add(seed_name)

        experiment_dir = next(
            (root / experiment_name for root in candidate_roots if (root / experiment_name).is_dir()),
            None,
        )
        if experiment_dir is None:
            tried = ", ".join(str(root / experiment_name) for root in candidate_roots)
            raise FileNotFoundError(f"找不到 summary 指向的實驗資料夾，已嘗試: {tried}")

        sources.append(
            {
                "seed": seed_name,
                "experiment_dir": experiment_dir,
                "pseudo_dir": resolve_pseudo_dir(experiment_dir),
            }
        )

    return sorted(sources, key=lambda source: seed_sort_key(source["seed"]))


# 定義依參數決定排序方式的函式
def sort_rows(rows: list[dict], sort_by: str) -> list[dict]:
    # 若指定用機率排序
    if sort_by == "probability":
        # 以機率高到低、散度低到高、doc_id 小到大排序
        return sorted(rows, key=lambda x: (-x["probability"], x["divergence"], int(x["doc_id"])))
    # 否則以散度排序
    return sorted(rows, key=lambda x: (x["divergence"], -x["probability"], int(x["doc_id"])))


# 定義計算子集合統計的函式
def summarize(rows: list[dict]) -> dict:
    # 計算文檔數量
    n_docs = len(rows)
    # 計算文檔全對數量（每篇文檔所有 MASK 都正確）
    full_match_docs = sum(1 for r in rows if r["full_match"])
    # 計算文檔全對比例（百分比）
    full_match_rate = (full_match_docs / n_docs * 100.0) if n_docs else 0.0
    # 計算 token 級正確總數
    total_correct_tokens = sum(r["correct_tokens"] for r in rows)
    # 計算 token 級總數
    total_tokens = sum(r["total_tokens"] for r in rows)
    # 計算 token 級 micro accuracy（百分比）
    token_micro_acc = (total_correct_tokens / total_tokens * 100.0) if total_tokens else 0.0
    # 計算平均機率
    avg_prob = (sum(r["probability"] for r in rows) / n_docs) if n_docs else 0.0
    # 計算平均散度
    avg_div = (sum(r["divergence"] for r in rows) / n_docs) if n_docs else 0.0
    # 將結果打包回傳
    return {
        "docs": n_docs,
        "full_match_docs": full_match_docs,
        "full_match_rate": full_match_rate,
        "token_micro_acc": token_micro_acc,
        "total_correct_tokens": total_correct_tokens,
        "total_tokens": total_tokens,
        "avg_prob": avg_prob,
        "avg_div": avg_div,
    }


# 定義單一 fold/round 的分析函式
def analyze_one(pseudo_dir: Path, fold: int, round_idx: int, top_k: int, sort_by: str) -> dict:
    # 組合 divergence JSON 檔路徑
    divergence_path = pseudo_dir / f"nest_divergence_scores_fold{fold}_round{round_idx}.json"
    # 組合 pseudo labeled JSON 檔路徑
    pseudo_path = pseudo_dir / f"pseudo_labeled_samples_fold{fold}_round{round_idx}.json"

    # 檢查 divergence 檔是否存在
    if not divergence_path.exists():
        # 若不存在就拋出明確錯誤
        raise FileNotFoundError(f"找不到檔案: {divergence_path}")
    # 檢查 pseudo labeled 檔是否存在
    if not pseudo_path.exists():
        # 若不存在就拋出明確錯誤
        raise FileNotFoundError(f"找不到檔案: {pseudo_path}")

    # 讀取 divergence JSON
    divergence_data = json.loads(divergence_path.read_text(encoding="utf-8"))
    # 讀取 pseudo labeled JSON
    pseudo_data = json.loads(pseudo_path.read_text(encoding="utf-8"))

    # 建立 doc_id 到 pseudo 記錄的索引（只保留有 GT 的樣本）
    pseudo_by_doc_id = {
        str(item["doc_id"]): item
        for item in pseudo_data
        if item.get("ground_truth_available", False)
    }

    # 建立 selected=True 且可與 GT 比對的資料列容器
    rows = []
    # 逐一走訪 divergence 的樣本清單
    for sample in divergence_data.get("samples", []):
        # 若此樣本未被選中則跳過。
        if not sample.get("selected", False):
            # 跳過未選中樣本
            continue
        # 取出 doc_id 並轉成字串
        doc_id = str(sample.get("doc_id"))
        # 取得對應 pseudo+gt 記錄
        pseudo_item = pseudo_by_doc_id.get(doc_id)
        # 若找不到可比對資料則立即報錯（此情況理論上不應發生）
        if pseudo_item is None:
            # 直接拋出錯誤，避免靜默遺漏造成統計偏差
            raise RuntimeError(
                f"selected 樣本缺少 GT 對應資料: doc_id={doc_id}; "
                f"請檢查 {pseudo_path.name} 是否完整"
            )

        # 取出偽標籤 token id 序列
        pred_ids = pseudo_item.get("pseudo_label_ids", [])
        # 取出真值 token id 序列
        gt_ids = pseudo_item.get("gt_label_ids", [])
        # 以最短長度做安全比對，避免長度不一致
        n_tokens = min(len(pred_ids), len(gt_ids))
        # 計算 token 級正確數
        n_correct = sum(1 for i in range(n_tokens) if pred_ids[i] == gt_ids[i])
        # 判斷是否為文檔全對
        is_full_match = (n_correct == n_tokens)

        # 加入一筆統計列
        rows.append(
            {
                "doc_id": doc_id,
                "probability": float(sample.get("probability", 0.0)),
                "divergence": float(sample.get("divergence_score", 0.0)),
                "correct_tokens": int(n_correct),
                "total_tokens": int(n_tokens),
                "full_match": bool(is_full_match),
            }
        )

    # 若沒有任何可分析資料則直接報錯
    if not rows:
        # 告知使用者此 fold/round 沒有可用 selected+GT 資料
        raise RuntimeError("沒有可分析的 selected=True 且含 GT 樣本")

    # 依指定規則排序
    sorted_rows = sort_rows(rows, sort_by)
    # 取實際可用 K（避免 K 大於樣本數）
    k = min(top_k, len(sorted_rows))
    # 取前 K 筆
    top_rows = sorted_rows[:k]
    # 取後 K 筆
    bottom_rows = sorted_rows[-k:]

    # 計算前 K 統計
    top_stats = summarize(top_rows)
    # 計算後 K 統計
    bottom_stats = summarize(bottom_rows)

    # 回傳此 fold/round 的完整結果
    return {
        "fold": fold,
        "round": round_idx,
        "k": k,
        "selected_with_gt_total": len(sorted_rows),
        "top_stats": top_stats,
        "bottom_stats": bottom_stats,
    }


# 定義輸出單一 fold/round 報告的函式
def write_one_report(result: dict, pseudo_dir: Path, sort_by: str, output_name: str | None = None) -> Path:
    # 決定輸出檔名
    report_name = output_name or (
        f"selected_true_top_bottom_doc_accuracy_fold{result['fold']}_round{result['round']}.txt"
    )
    # 組合輸出完整路徑
    output_path = pseudo_dir / report_name

    # 讀取 top/bottom 統計
    top_stats = result["top_stats"]
    bottom_stats = result["bottom_stats"]

    # 準備輸出文字內容
    lines = []
    # 寫入標題。
    lines.append("=" * 90)
    # 寫入分析主題。
    lines.append("selected=True 前/後K筆 文檔準確率統計")
    # 寫入分隔線。
    lines.append("=" * 90)
    # 寫入資料來源目錄。
    lines.append(f"pseudo_dir: {pseudo_dir}")
    # 寫入 fold 與 round。
    lines.append(f"fold={result['fold']}, round={result['round']}")
    # 寫入排序規則。
    lines.append(f"sort_by={sort_by}")
    # 寫入總樣本數。
    lines.append(f"selected_with_gt_total={result['selected_with_gt_total']}")
    # 寫入 K 值。
    lines.append(f"k={result['k']}")
    # 空行。
    lines.append("")

    # 定義一個內部函式，用於格式化輸出區塊。
    def append_block(title: str, stats: dict) -> None:
        # 加入區塊標題。
        lines.append(f"[{title}]")
        # 加入文檔數。
        lines.append(f"docs={stats['docs']}")
        # 加入文檔全對數。
        lines.append(f"full_match_docs={stats['full_match_docs']}")
        # 加入文檔全對率。
        lines.append(f"full_match_rate={stats['full_match_rate']:.2f}%")
        # 加入 token micro accuracy。
        lines.append(
            f"token_micro_acc={stats['total_correct_tokens']}/{stats['total_tokens']}={stats['token_micro_acc']:.2f}%"
        )
        # 加入平均機率。
        lines.append(f"avg_probability={stats['avg_prob']:.8f}")
        # 加入平均散度。
        lines.append(f"avg_divergence={stats['avg_div']:.8f}")
        # 空行。
        lines.append("")

    # 寫入前 K 區塊。
    append_block("TOP_K", top_stats)
    # 寫入後 K 區塊。
    append_block("BOTTOM_K", bottom_stats)

    # 計算文檔全對數差值。
    diff = top_stats["full_match_docs"] - bottom_stats["full_match_docs"]
    # 寫入差值。
    lines.append(f"difference_full_match_docs=TOP_K-BOTTOM_K={diff}")

    # 寫入輸出檔。
    output_path.write_text("\n".join(lines), encoding="utf-8")
    # 回傳輸出路徑。
    return output_path


def output_prefix_from_summary(summary_path: Path) -> str:
    """依 summary 檔名產生輸出檔前綴。"""

    name = summary_path.name
    if name.endswith("_summary.txt"):
        return name[: -len("_summary.txt")]
    return summary_path.stem


def pooled_stats(results: list[dict], stats_key: str) -> dict:
    """合併多個 fold/seed 單位的分子分母後計算 pooled rate。"""

    docs = sum(res[stats_key]["docs"] for res in results)
    full_match_docs = sum(res[stats_key]["full_match_docs"] for res in results)
    total_correct_tokens = sum(res[stats_key]["total_correct_tokens"] for res in results)
    total_tokens = sum(res[stats_key]["total_tokens"] for res in results)
    weighted_prob_sum = sum(res[stats_key]["avg_prob"] * res[stats_key]["docs"] for res in results)
    weighted_div_sum = sum(res[stats_key]["avg_div"] * res[stats_key]["docs"] for res in results)

    return {
        "docs": docs,
        "full_match_docs": full_match_docs,
        "full_match_rate": (full_match_docs / docs * 100.0) if docs else 0.0,
        "token_micro_acc": (total_correct_tokens / total_tokens * 100.0) if total_tokens else 0.0,
        "total_correct_tokens": total_correct_tokens,
        "total_tokens": total_tokens,
        "avg_prob": (weighted_prob_sum / docs) if docs else 0.0,
        "avg_div": (weighted_div_sum / docs) if docs else 0.0,
    }


def mean_rate(results: list[dict], stats_key: str) -> float:
    """計算多個 fold/seed 單位 rate 的算術平均。"""

    if not results:
        return 0.0
    return sum(res[stats_key]["full_match_rate"] for res in results) / len(results)


def write_results_summary(
    all_results: list[dict],
    output_path: Path,
    sort_by: str,
    top_k: int,
    folds: list[int],
    rounds: list[int],
    include_seed: bool,
    summary_input: Path | None = None,
) -> Path:
    """輸出單一實驗或 multi-seed 的 top/bottom 彙整表。"""

    lines = []
    lines.append("=" * 130)
    lines.append("selected=True top/bottom document accuracy summary")
    lines.append("=" * 130)
    if summary_input is not None:
        lines.append(f"summary_file: {summary_input}")
    lines.append(f"sort_by: {sort_by}")
    lines.append(f"top_k: {top_k}")
    lines.append(f"folds: {folds[0]}-{folds[-1]}")
    lines.append(f"rounds: {rounds[0]}-{rounds[-1]}")
    lines.append("")

    if include_seed:
        lines.append(
            "seed\tfold\tround\tk\tselected_total\tTOP_full_docs\tTOP_full_rate\t"
            "BOTTOM_full_docs\tBOTTOM_full_rate\tDIFF_docs"
        )
    else:
        lines.append(
            "fold\tround\tk\tselected_total\tTOP_full_docs\tTOP_full_rate\t"
            "BOTTOM_full_docs\tBOTTOM_full_rate\tDIFF_docs"
        )

    for res in all_results:
        top = res["top_stats"]
        bottom = res["bottom_stats"]
        diff_docs = top["full_match_docs"] - bottom["full_match_docs"]
        prefix = f"{res['seed']}\t" if include_seed else ""
        lines.append(
            f"{prefix}{res['fold']}\t{res['round']}\t{res['k']}\t{res['selected_with_gt_total']}\t"
            f"{top['full_match_docs']}\t{top['full_match_rate']:.2f}%\t"
            f"{bottom['full_match_docs']}\t{bottom['full_match_rate']:.2f}%\t{diff_docs}"
        )

    lines.append("")
    lines.append("[Round pooled summary]")
    lines.append(
        "round\tunits\tTOP_full_docs\tTOP_docs\tTOP_full_rate\tBOTTOM_full_docs\t"
        "BOTTOM_docs\tBOTTOM_full_rate\tDIFF_docs"
    )
    for round_idx in rounds:
        round_results = [res for res in all_results if res["round"] == round_idx]
        top = pooled_stats(round_results, "top_stats")
        bottom = pooled_stats(round_results, "bottom_stats")
        diff_docs = top["full_match_docs"] - bottom["full_match_docs"]
        lines.append(
            f"{round_idx}\t{len(round_results)}\t"
            f"{top['full_match_docs']}\t{top['docs']}\t{top['full_match_rate']:.2f}%\t"
            f"{bottom['full_match_docs']}\t{bottom['docs']}\t{bottom['full_match_rate']:.2f}%\t{diff_docs}"
        )

    lines.append("")
    lines.append("[Round mean summary]")
    lines.append("round\tunits\tTOP_mean_full_rate\tBOTTOM_mean_full_rate\tDIFF_pp")
    for round_idx in rounds:
        round_results = [res for res in all_results if res["round"] == round_idx]
        top_mean = mean_rate(round_results, "top_stats")
        bottom_mean = mean_rate(round_results, "bottom_stats")
        lines.append(
            f"{round_idx}\t{len(round_results)}\t{top_mean:.2f}%\t{bottom_mean:.2f}%\t"
            f"{top_mean - bottom_mean:.2f}"
        )

    lines.append("=" * 130)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


# 定義主程式函式
def main() -> None:
    # 解析命令列參數
    args = parse_args()
    input_path = Path(args.experiment_dir)

    if input_path.is_file():
        if args.fold is not None:
            folds = [args.fold]
        else:
            if args.fold_start > args.fold_end:
                raise ValueError("--fold-start cannot be greater than --fold-end")
            folds = list(range(args.fold_start, args.fold_end + 1))

        if args.round is not None:
            rounds = [args.round]
        else:
            if args.round_start > args.round_end:
                raise ValueError("--round-start cannot be greater than --round-end")
            rounds = list(range(args.round_start, args.round_end + 1))

        experiments_roots = [Path(root) for root in args.experiments_root]
        sources = resolve_summary_sources(input_path, experiments_roots=experiments_roots)
        all_results: list[dict] = []

        for source in sources:
            pseudo_dir = source["pseudo_dir"]
            for fold in folds:
                for round_idx in rounds:
                    try:
                        result = analyze_one(
                            pseudo_dir=pseudo_dir,
                            fold=fold,
                            round_idx=round_idx,
                            top_k=args.top_k,
                            sort_by=args.sort_by,
                        )
                    except FileNotFoundError as exc:
                        if not args.skip_missing_rounds:
                            raise
                        print(f"skip missing: seed={source['seed']} fold={fold} round={round_idx}: {exc}")
                        continue
                    result["seed"] = source["seed"]
                    result["experiment_dir"] = str(source["experiment_dir"])
                    all_results.append(result)

                    one_path = write_one_report(
                        result=result,
                        pseudo_dir=pseudo_dir,
                        sort_by=args.sort_by,
                    )
                    print(f"report: {one_path}")

        if len(all_results) > 1:
            summary_name = args.output or (
                f"{output_prefix_from_summary(input_path)}_selected_true_top_bottom_doc_accuracy_"
                f"top{args.top_k}_{args.sort_by}_f{folds[0]}-{folds[-1]}_r{rounds[0]}-{rounds[-1]}.txt"
            )
            summary_path = Path(summary_name)
            if not summary_path.is_absolute():
                summary_path = input_path.parent / summary_path

            summary_path = write_results_summary(
                all_results=all_results,
                output_path=summary_path,
                sort_by=args.sort_by,
                top_k=args.top_k,
                folds=folds,
                rounds=rounds,
                include_seed=True,
                summary_input=input_path,
            )
            print(f"summary: {summary_path}")

        print("[done] selected=True top/bottom multi-seed statistics")
        return

    # 轉成 Path 物件
    experiment_dir = Path(args.experiment_dir)
    # 解析出真正的 pseudo_results 目錄
    pseudo_dir = resolve_pseudo_dir(experiment_dir)

    # 決定 fold 清單：若指定 --fold 則只跑單一 fold，否則跑 fold-start~fold-end
    if args.fold is not None:
        folds = [args.fold]
    else:
        if args.fold_start > args.fold_end:
            raise ValueError("--fold-start 不可大於 --fold-end")
        folds = list(range(args.fold_start, args.fold_end + 1))

    # 決定 round 清單：若指定 --round 則只跑單一 round，否則跑 round-start~round-end
    if args.round is not None:
        rounds = [args.round]
    else:
        if args.round_start > args.round_end:
            raise ValueError("--round-start 不可大於 --round-end")
        rounds = list(range(args.round_start, args.round_end + 1))

    # 儲存所有結果（供多組合摘要使用）
    all_results: list[dict] = []

    # 逐一分析每個 fold/round 組合
    for fold in folds:
        for round_idx in rounds:
            try:
                result = analyze_one(
                    pseudo_dir=pseudo_dir,
                    fold=fold,
                    round_idx=round_idx,
                    top_k=args.top_k,
                    sort_by=args.sort_by,
                )
            except FileNotFoundError as exc:
                if not args.skip_missing_rounds:
                    raise
                print(f"skip missing: fold={fold} round={round_idx}: {exc}")
                continue
            all_results.append(result)

            # 單一組合時沿用 --output 檔名；多組合時固定每組一檔
            one_output_name = args.output if (len(folds) == 1 and len(rounds) == 1) else None
            one_path = write_one_report(
                result=result,
                pseudo_dir=pseudo_dir,
                sort_by=args.sort_by,
                output_name=one_output_name,
            )
            print(f"report: {one_path}")

    # 若是多組合，額外輸出一份總表，方便一次檢視 10x10
    if len(all_results) > 1:
        summary_name = args.output or (
            f"selected_true_top_bottom_doc_accuracy_summary_f{folds[0]}-{folds[-1]}_r{rounds[0]}-{rounds[-1]}.txt"
        )
        summary_path = pseudo_dir / summary_name
        lines = []
        lines.append("=" * 110)
        lines.append("selected=True 前/後K筆 文檔準確率總表")
        lines.append("=" * 110)
        lines.append(
            "fold\tround\tk\tselected_total\tTOP_full_docs\tTOP_full_rate\tBOTTOM_full_docs\tBOTTOM_full_rate\tDIFF_docs"
        )
        for res in all_results:
            top = res["top_stats"]
            bottom = res["bottom_stats"]
            diff_docs = top["full_match_docs"] - bottom["full_match_docs"]
            lines.append(
                f"{res['fold']}\t{res['round']}\t{res['k']}\t{res['selected_with_gt_total']}\t"
                f"{top['full_match_docs']}\t{top['full_match_rate']:.2f}%\t"
                f"{bottom['full_match_docs']}\t{bottom['full_match_rate']:.2f}%\t{diff_docs}"
            )
        lines.append("=" * 110)
        summary_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"summary: {summary_path}")

    # 在終端列印摘要標題
    print("[完成] selected=True 前/後K筆統計")


# 若此檔案被直接執行。
if __name__ == "__main__":
    # 執行主程式。
    main()
