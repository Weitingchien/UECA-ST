#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import numpy as np


DEFAULT_SUMMARY_SEARCH_ROOTS = [
    r"D:\GithubRepo\UECA_ST\ep_split10_t1te1v1_u7_disjoint",
    r"I:\ep_split10_t1te1v1_u7_disjoint",
    r"F:\experiments\ep_split10_t1te1v1_u7_disjoint",
    r"H:\ep_split10_t1te1v1_u7_disjoint",
]

WINDOWS_DRIVE_PATH_RE = re.compile(r"^([A-Za-z]):[\\/](.*)$")
WSL_MOUNT_PATH_RE = re.compile(r"^/mnt/([A-Za-z])/(.*)$")
SUMMARY_BULLET_RE = re.compile(r"^\s*-\s+(.+?)\s*$")
DEFAULT_OUTPUT_PATH = Path("analysis_unfiltered_mask3_reports/divergence_selection_reliability_report.txt")
SUMMARY_OUTPUT_SUFFIX = "_divergence_selection_reliability_from_summary_report.txt"
_has_printed_pseudo_item_example = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze whether lower NeST divergence implies higher pseudo-label reliability "
            "and higher selection likelihood."
        )
    )
    parser.add_argument(
        "--experiment-dir",
        action="append",
        default=[],
        help="Experiment directory or pseudo_results_* directory. Repeatable.",
    )
    parser.add_argument(
        "--summary-file",
        default=None,
        help=(
            "Summary txt file containing the section '彙整的實驗目錄 (共 N 個):'. "
            "The script will read experiment folder names from it and search them under known roots."
        ),
    )
    parser.add_argument(
        "--search-root",
        action="append",
        default=[],
        help="Root directory containing experiment folders. Repeatable. In --summary-file mode these are checked before the built-in roots.",
    )
    parser.add_argument(
        "--name-contains",
        action="append",
        default=[],
        help="Only keep experiment folders whose names contain all of these substrings.",
    )
    parser.add_argument("--fold-start", type=int, default=1, help="Start fold (default: 1)")
    parser.add_argument("--fold-end", type=int, default=10, help="End fold (default: 10)")
    parser.add_argument("--round-start", type=int, default=1, help="Start round (default: 1)")
    parser.add_argument("--round-end", type=int, default=5, help="End round (default: 5)")
    parser.add_argument(
        "--quantiles",
        type=int,
        default=5,
        help="How many divergence buckets to split into (default: 5)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output report path. If omitted in --summary-file mode, derive the file name from the summary file name.",
    )
    return parser.parse_args()


def get_path_candidates(raw_path: str) -> list[Path]:
    raw_path = raw_path.strip()
    candidates: list[Path] = []
    seen: set[str] = set()

    def add_candidate(candidate: Path) -> None:
        candidate_key = str(candidate)
        if candidate_key in seen:
            return
        seen.add(candidate_key)
        candidates.append(candidate)

    add_candidate(Path(raw_path))

    windows_match = WINDOWS_DRIVE_PATH_RE.match(raw_path)
    if windows_match:
        drive = windows_match.group(1).lower()
        tail = windows_match.group(2).replace("\\", "/")
        add_candidate(Path(f"/mnt/{drive}/{tail}"))

    wsl_match = WSL_MOUNT_PATH_RE.match(raw_path)
    if wsl_match:
        drive = wsl_match.group(1).upper()
        tail = wsl_match.group(2).replace("/", "\\")
        add_candidate(Path(f"{drive}:\\{tail}"))

    return candidates


def resolve_existing_path(raw_path: str, label: str, notes: list[str]) -> Path | None:
    for candidate in get_path_candidates(raw_path):
        if candidate.exists():
            if str(candidate) != raw_path:
                notes.append(f"[resolve] {label}: {raw_path} -> {candidate}")
            return candidate
    notes.append(f"[skip] missing {label}: {raw_path}")
    return None


def parse_summary_experiment_names(summary_path: Path) -> list[str]:
    lines = summary_path.read_text(encoding="utf-8").splitlines()
    experiment_names: list[str] = []
    in_target_block = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("彙整的實驗目錄"):
            in_target_block = True
            continue

        if not in_target_block:
            continue

        bullet_match = SUMMARY_BULLET_RE.match(line)
        if bullet_match:
            candidate_name = bullet_match.group(1).strip()
            if candidate_name and set(candidate_name) != {"-"}:
                experiment_names.append(candidate_name)
                continue

        if experiment_names and not stripped:
            continue

        if experiment_names and stripped:
            break

    if not experiment_names:
        raise RuntimeError(
            f"Cannot find experiment names under '彙整的實驗目錄' in summary file: {summary_path}"
        )

    return experiment_names


def resolve_summary_search_roots(args: argparse.Namespace, notes: list[str]) -> list[Path]:
    roots: list[Path] = []
    seen: set[str] = set()

    for raw_root in list(args.search_root) + DEFAULT_SUMMARY_SEARCH_ROOTS:
        resolved_root = resolve_existing_path(raw_root, "search root", notes)
        if resolved_root is None or not resolved_root.is_dir():
            continue
        root_key = str(resolved_root)
        if root_key in seen:
            continue
        seen.add(root_key)
        roots.append(resolved_root)

    return roots


def resolve_pseudo_dir(experiment_dir: Path) -> Path:
    # 如果傳入的本身就是 pseudo_results_* 目錄，就直接使用，不需要再往下找
    if experiment_dir.name.startswith("pseudo_results_") and experiment_dir.is_dir():
        return experiment_dir

    # 否則把實驗目錄底下所有名稱符合 pseudo_results_* 的子目錄都找出來
    candidates = sorted(p for p in experiment_dir.glob("pseudo_results_*") if p.is_dir())

    # 一個都找不到時直接報錯，表示這個實驗目錄不是預期格式，無法繼續分析
    if not candidates:
        raise FileNotFoundError(f"Cannot find pseudo_results_* under: {experiment_dir}")

    # 若有多個候選目錄，這裡會取排序後最後一個，通常代表名稱較新的那個結果目錄
    return candidates[-1]


def get_summary_report_output_path(summary_path: Path) -> Path:
    summary_name = summary_path.name
    if summary_name.endswith("_summary.txt"):
        summary_prefix = summary_name[: -len("_summary.txt")]
    elif summary_name.endswith(".txt"):
        summary_prefix = summary_name[: -len(".txt")]
    else:
        summary_prefix = summary_name
    return DEFAULT_OUTPUT_PATH.parent / f"{summary_prefix}{SUMMARY_OUTPUT_SUFFIX}"


def resolve_output_path(args: argparse.Namespace, summary_path: Path | None) -> Path:
    if args.output:
        return Path(args.output)
    if summary_path is not None:
        return get_summary_report_output_path(summary_path)
    return DEFAULT_OUTPUT_PATH


def resolve_experiment_dirs(args: argparse.Namespace) -> tuple[list[Path], list[str], Path | None]:
    selected_by_name: dict[str, Path] = {}
    notes: list[str] = []
    summary_path: Path | None = None

    def add_candidate(path: Path, source: str) -> None:
        if not path.exists() or not path.is_dir():
            notes.append(f"[skip] missing directory from {source}: {path}")
            return
        key = path.name
        if key in selected_by_name:
            notes.append(f"[dedupe] keep {selected_by_name[key]} ; ignore duplicate from {source}: {path}")
            return
        selected_by_name[key] = path

    for raw_path in args.experiment_dir:
        resolved_path = resolve_existing_path(raw_path, "--experiment-dir", notes)
        if resolved_path is not None:
            add_candidate(resolved_path, "--experiment-dir")

    if args.summary_file:
        summary_path = resolve_existing_path(args.summary_file, "--summary-file", notes)
        if summary_path is None:
            raise FileNotFoundError(f"Cannot find summary file: {args.summary_file}")

        experiment_names = parse_summary_experiment_names(summary_path)
        search_roots = resolve_summary_search_roots(args, notes)
        if not search_roots:
            raise RuntimeError("No available search roots for summary-file mode.")

        notes.append(f"[summary] loaded {len(experiment_names)} experiment names from: {summary_path}")
        for experiment_name in experiment_names:
            matched_paths = [root / experiment_name for root in search_roots if (root / experiment_name).is_dir()]
            if not matched_paths:
                notes.append(f"[skip] summary experiment not found under configured roots: {experiment_name}")
                continue

            add_candidate(matched_paths[0], f"--summary-file {summary_path}")
            if len(matched_paths) > 1:
                extra_paths = ", ".join(str(path) for path in matched_paths[1:])
                notes.append(
                    f"[multi-match] use {matched_paths[0]} ; ignore additional matches for {experiment_name}: {extra_paths}"
                )

    for raw_root in args.search_root:
        root = resolve_existing_path(raw_root, "search root", notes)
        if root is None or not root.is_dir():
            continue
        for child in sorted(root.iterdir()):
            if not child.is_dir():
                continue
            if args.name_contains and not all(token in child.name for token in args.name_contains):
                continue
            add_candidate(child, f"--search-root {root}")

    if not selected_by_name:
        raise RuntimeError(
            "No experiment directories were resolved. Check --experiment-dir/--summary-file/--search-root arguments."
        )

    return sorted(selected_by_name.values(), key=lambda path: path.name), notes, summary_path


def load_round_rows(pseudo_dir: Path, experiment_name: str, fold: int, round_idx: int) -> tuple[list[dict], str | None]:
    # 避免 pseudo_item 範例被重複印很多次
    global _has_printed_pseudo_item_example

    # 依照 fold 與 round 組出該輪的 divergence 結果檔路徑
    divergence_path = pseudo_dir / f"nest_divergence_scores_fold{fold}_round{round_idx}.json"
    # 組出同一輪所有未標註文檔的偽標籤路徑
    all_unlabeled_path = pseudo_dir / f"all_unlabeled_pseudo_predictions_fold{fold}_round{round_idx}.json"

    # 任一必要檔案不存在時，不直接報錯整個程式，而是回傳缺檔訊息給上層統一記錄
    if not divergence_path.exists() or not all_unlabeled_path.exists():
        return [], f"missing fold={fold}, round={round_idx}: {divergence_path.name} or {all_unlabeled_path.name}"

    # 讀入該輪的 NeST divergence / selection 統計資料
    divergence_data = json.loads(divergence_path.read_text(encoding="utf-8"))
    # 讀入該輪所有未標註文檔的偽標籤與 GT 對照資訊
    all_unlabeled_data = json.loads(all_unlabeled_path.read_text(encoding="utf-8"))

    # 建立 doc_id -> 原始 pseudo item 的完整索引，方便出錯時把原始內容一起印出來
    all_unlabeled_by_doc_id = {str(item["doc_id"]): item for item in all_unlabeled_data}
    # 只保留有 ground truth 可對照的文檔，因為這支分析只統計可驗證正確性的樣本
    unlabeled_by_doc_id = {
        str(item["doc_id"]): item
        for item in all_unlabeled_by_doc_id.values()
        if item.get("ground_truth_available", False)
    }

    # rows 會收集這一個 fold / round 中，每篇文檔對應的一列分析資料
    rows: list[dict] = []
    # 逐筆走訪 divergence 檔中的 sample，並和 pseudo prediction 檔做 doc_id 對齊
    for sample in divergence_data.get("samples", []):
        # 先把 sample 的 doc_id 統一轉成字串，避免型別不同造成對不到
        doc_id = str(sample.get("doc_id"))
        # 保留原始 pseudo item，不論它有沒有 GT，都可在報錯時直接印出來
        raw_pseudo_item = all_unlabeled_by_doc_id.get(doc_id)
        # 這裡只抓有 GT 的 pseudo item，因為後面要計算 correct / wrong 相關統計
        pseudo_item = unlabeled_by_doc_id.get(doc_id)
        # 如果 divergence sample 對不到可用的 GT-backed pseudo item，代表資料對齊有問題，直接中止
        if pseudo_item is None:
            # 先把目前這筆 sample 轉成好閱讀的 JSON 文字，方便除錯
            sample_json = json.dumps(sample, ensure_ascii=False, indent=2)
            # 若原始 pseudo item 存在，也一併轉成 JSON；否則直接標示成 None
            raw_pseudo_item_json = (
                json.dumps(raw_pseudo_item, ensure_ascii=False, indent=2)
                if raw_pseudo_item is not None
                else "None"
            )
            # 直接報錯並附上 experiment / fold / round / doc_id 與原始內容，方便定位資料不一致來源
            raise RuntimeError(
                "Cannot align divergence sample with GT-backed pseudo prediction: "
                f"experiment={experiment_name}, fold={fold}, round={round_idx}, doc_id={doc_id}\n"
                f"sample={sample_json}\n"
                f"raw_pseudo_item={raw_pseudo_item_json}"
            )

        # 只在整次程式執行第一次成功對到 pseudo_item 時，印一次完整範例看結構
        if not _has_printed_pseudo_item_example:
            print(
                "[debug] first pseudo_item example\n"
                f"experiment={experiment_name}, fold={fold}, round={round_idx}, doc_id={doc_id}\n"
                f"pseudo_item={json.dumps(pseudo_item, ensure_ascii=False, indent=2)}"
            )
            # 設成 True 之後，後續其他輪次就不再重複印相同類型資訊
            _has_printed_pseudo_item_example = True

        # 優先使用 pseudo prediction 檔中較明確的 selected_for_training 欄位
        selected = pseudo_item.get("selected_for_training")
        # 如果 pseudo prediction 檔沒有這個欄位，就退回使用 divergence sample 裡的 selected
        # sample.get("selected", False) 的意思是：
        # 1. 先嘗試讀 sample["selected"]
        # 2. 如果 sample 裡根本沒有 selected 這個鍵，就用 False 當預設值
        # 也就是把這筆文檔視為「未被選中」
        if selected is None:
            selected = sample.get("selected", False)

        # 每一列代表某個 fold / round / doc 的一筆可對照 GT 的未標註文檔統計
        # 後面 [Overall] 區塊中的 rows_with_gt、selected_rows、full_correct_rows、
        # pair_correct_rows 與各種 mean_divergence，都是由這些欄位彙整而來
        rows.append(
            {
                # 實驗資料夾名稱，用來追蹤這筆資料來自哪個 seed / 實驗
                "experiment": experiment_name,
                # 這筆資料對應的 fold 編號
                "fold": fold,
                # 這筆資料對應的 self-training round 編號
                "round": round_idx,
                # 文檔編號，作為跨檔案對齊的主鍵
                "doc_id": doc_id,
                # 該文檔在 NeST 中計算出的 divergence 分數
                "divergence": float(sample.get("divergence_score", 0.0)),
                # 該文檔被抽樣選中的機率
                "probability": float(sample.get("probability", 0.0)),
                # 是否被選進自訓練，最後統一轉成 0/1 方便做平均與相關係數
                "selected": int(bool(selected)),
                # 整份文檔的偽標籤是否完全與 GT 一致
                "fully_correct": int(bool(pseudo_item.get("fully_correct", False))),
                # 只看 pair 偽標籤時，是否完全與 GT 一致
                "pair_fully_correct": int(bool(pseudo_item.get("pair_fully_correct", False))),
                # 整份文檔偽標籤與 GT 的整體匹配比例；若缺值則記成 None
                "match_ratio": float(pseudo_item["match_ratio"]) if pseudo_item.get("match_ratio") is not None else None,
                # 只看 pair 偽標籤時的匹配比例；若缺值則記成 None
                "pair_match_ratio": float(pseudo_item["pair_match_ratio"]) if pseudo_item.get("pair_match_ratio") is not None else None,
            }
        )

    # 回傳這一輪整理好的所有 rows，第二個回傳值為 None 代表這輪沒有缺檔問題
    return rows, None


def safe_mean(values: list[float]) -> float:
    # 空清單時回傳 NaN；否則回傳一般平均值。
    # 這個函式也會被拿來算 0/1 欄位的平均，因此可同時表示「比例 / rate」
    return float(sum(values) / len(values)) if values else float("nan")


def summarize_rows(rows: list[dict], quantile_idx: int) -> dict:
    # 只保留有值的 match_ratio，避免 None 影響平均計算
    match_ratios = [row["match_ratio"] for row in rows if row["match_ratio"] is not None]
    # 只保留有值的 pair_match_ratio，避免 None 影響平均計算
    pair_match_ratios = [row["pair_match_ratio"] for row in rows if row["pair_match_ratio"] is not None]
    # 取出這個 quantile 內所有文檔的 divergence，供平均值與範圍使用
    divergences = [row["divergence"] for row in rows]
    # 取出這個 quantile 內所有文檔的抽樣機率；目前雖然表格沒印出，但摘要仍保留此欄位
    probabilities = [row["probability"] for row in rows]

    return {
        # 這是第幾個 quantile，例如 Q1、Q2、Q3
        "quantile": quantile_idx,
        # docs 對應報表中的文檔數量。
        "docs": len(rows),
        # div_mean：這一組文檔 divergence 的平均值
        "mean_divergence": safe_mean(divergences),
        # div_range 左端：這一組文檔中最小的 divergence
        "min_divergence": min(divergences) if divergences else float("nan"),
        # div_range 右端：這一組文檔中最大的 divergence
        "max_divergence": max(divergences) if divergences else float("nan"),
        # 這組文檔平均被抽中的機率，目前未直接印到報表，但可供後續擴充
        "mean_probability": safe_mean(probabilities),
        # selected_rate：selected 是 0/1，取平均後就等於「被選中比例」
        "selected_rate": safe_mean([row["selected"] for row in rows]),
        # full_rate：fully_correct 是 0/1，取平均後就等於「整份文檔完全正確比例」
        "full_rate": safe_mean([row["fully_correct"] for row in rows]),
        # pair_rate：pair_fully_correct 是 0/1，取平均後就等於「pair 完全正確比例」
        "pair_rate": safe_mean([row["pair_fully_correct"] for row in rows]),
        # match_ratio：這組文檔整體匹配比例的平均值
        "mean_match_ratio": safe_mean(match_ratios),
        # pair_match_ratio：這組文檔在 pair 層級匹配比例的平均值
        "mean_pair_match_ratio": safe_mean(pair_match_ratios),
    }


def build_quantile_summaries(rows: list[dict], quantiles: int) -> list[dict]:
    # 沒有資料時直接回傳空摘要
    if not rows:
        return []
    # 先依 divergence 由小到大排序
    # 若 divergence 相同，則優先把 probability 較高者排前面；最後再用 doc_id 穩定排序
    sorted_rows = sorted(rows, key=lambda row: (row["divergence"], -row["probability"], row["doc_id"]))
    # quantiles 至少為 1，且不會超過資料筆數，避免切分數量非法
    actual_quantiles = min(max(1, quantiles), len(sorted_rows))
    # 把排序後的資料索引平均切成 actual_quantiles 份
    # 因此 Q1 會是 divergence 最低的一批，Q5 會是 divergence 最高的一批
    grouped_indices = np.array_split(np.arange(len(sorted_rows)), actual_quantiles)

    summaries = []
    # 逐一把每個 quantile 的索引轉回原始 row，並計算該組摘要
    for idx, indices in enumerate(grouped_indices, start=1):
        # 取出這個 quantile 對應的所有文檔資料
        subset = [sorted_rows[i] for i in indices.tolist()]
        # 把這一組文檔整理成報表要印的一列摘要
        summaries.append(summarize_rows(subset, idx))
    return summaries


def format_quantile_table(title: str, summaries: list[dict]) -> list[str]:
    # 第一行先放區塊標題，例如 [All docs grouped by divergence quantiles]
    lines = [title]
    # 如果沒有摘要資料，就直接印 no data
    if not summaries:
        lines.append("  (no data)")
        return lines

    # 逐個 quantile 把摘要欄位格式化成報表中的一行文字
    for summary in summaries:
        lines.append(
            "  Q{q}: docs={docs}, div_mean={div_mean:.6f}, div_range=[{div_min:.6f}, {div_max:.6f}], "
            "selected_rate={sel:.4f}, pair_full_rate={pair:.4f}, full_rate={full:.4f}, "
            "match_ratio={match:.4f}, pair_match_ratio={pair_match:.4f}".format(
                # q 會對應成 Q1、Q2、Q3...
                q=summary["quantile"],
                # docs：這個 quantile 內總共有幾篇文檔
                docs=summary["docs"],
                # div_mean：平均 divergence
                div_mean=summary["mean_divergence"],
                # div_range 左端：最小 divergence
                div_min=summary["min_divergence"],
                # div_range 右端：最大 divergence
                div_max=summary["max_divergence"],
                # selected_rate：被選中比例
                sel=summary["selected_rate"],
                # pair_full_rate：pair 完全正確比例
                pair=summary["pair_rate"],
                # full_rate：整份文檔完全正確比例
                full=summary["full_rate"],
                # match_ratio：整份文檔的平均匹配比例
                match=summary["mean_match_ratio"],
                # pair_match_ratio：pair 層級的平均匹配比例
                pair_match=summary["mean_pair_match_ratio"],
            )
        )
    return lines


def format_low_high_round_summary(round_rows: list[dict], quantiles: int) -> str:
    summaries = build_quantile_summaries(round_rows, quantiles)
    if len(summaries) < 2:
        return "  insufficient data"
    low = summaries[0]
    high = summaries[-1]
    return (
        "  Q1(low div) vs Q{high_q}(high div): "
        "selected_rate {low_sel:.4f} -> {high_sel:.4f} (gap {sel_gap:+.4f}); "
        "pair_full_rate {low_pair:.4f} -> {high_pair:.4f} (gap {pair_gap:+.4f}); "
        "full_rate {low_full:.4f} -> {high_full:.4f} (gap {full_gap:+.4f})"
    ).format(
        high_q=high["quantile"],
        low_sel=low["selected_rate"],
        high_sel=high["selected_rate"],
        sel_gap=low["selected_rate"] - high["selected_rate"],
        low_pair=low["pair_rate"],
        high_pair=high["pair_rate"],
        pair_gap=low["pair_rate"] - high["pair_rate"],
        low_full=low["full_rate"],
        high_full=high["full_rate"],
        full_gap=low["full_rate"] - high["full_rate"],
    )


def build_report(
    experiment_dirs: list[Path],
    pseudo_dirs: list[Path],
    rows: list[dict],
    missing_notes: list[str],
    resolution_notes: list[str],
    quantiles: int,
) -> str:
    # 先把所有文檔依不同條件切成子集合，後面 [Overall] 的數值都是針對這些集合做統計
    selected_rows = [row for row in rows if row["selected"] == 1]
    unselected_rows = [row for row in rows if row["selected"] == 0]
    full_correct_rows = [row for row in rows if row["fully_correct"] == 1]
    full_wrong_rows = [row for row in rows if row["fully_correct"] == 0]
    pair_correct_rows = [row for row in rows if row["pair_fully_correct"] == 1]
    pair_wrong_rows = [row for row in rows if row["pair_fully_correct"] == 0]

    lines = []
    lines.append("=" * 100)
    lines.append("Divergence vs Selection / Reliability Report")
    lines.append("=" * 100)
    lines.append(f"Experiments analyzed: {len(experiment_dirs)}")
    for experiment_dir, pseudo_dir in zip(experiment_dirs, pseudo_dirs):
        lines.append(f"  - experiment: {experiment_dir}")
        lines.append(f"    pseudo_dir: {pseudo_dir}")

    if resolution_notes:
        lines.append("")
        lines.append("Resolution notes:")
        for note in resolution_notes:
            lines.append(f"  {note}")

    lines.append("")
    lines.append("[Overall]")
    # rows_with_gt: 有 ground truth 可對照，因此能納入可靠性分析的文檔總數
    lines.append(f"  rows_with_gt={len(rows)}")
    # selected_rows / unselected_rows: 本輪自訓練有被選入 / 沒被選入的文檔數量
    lines.append(f"  selected_rows={len(selected_rows)}")
    lines.append(f"  unselected_rows={len(unselected_rows)}")
    # full_correct_rows: 整份文檔的偽標籤完全正確的文檔數
    # pair_correct_rows: 只看 pair 偽標籤時完全正確的文檔數
    lines.append(f"  full_correct_rows={len(full_correct_rows)}")
    lines.append(f"  pair_correct_rows={len(pair_correct_rows)}")
    # mean_divergence(*): 在指定子集合內，把每篇文檔的 divergence 取平均
    # 數值越小，代表該群文檔平均散度越低
    lines.append(f"  mean_divergence(selected)={safe_mean([row['divergence'] for row in selected_rows]):.6f}")
    lines.append(f"  mean_divergence(unselected)={safe_mean([row['divergence'] for row in unselected_rows]):.6f}")
    lines.append(f"  mean_divergence(full_correct)={safe_mean([row['divergence'] for row in full_correct_rows]):.6f}")
    lines.append(f"  mean_divergence(full_wrong)={safe_mean([row['divergence'] for row in full_wrong_rows]):.6f}")
    lines.append(f"  mean_divergence(pair_correct)={safe_mean([row['divergence'] for row in pair_correct_rows]):.6f}")
    lines.append(f"  mean_divergence(pair_wrong)={safe_mean([row['divergence'] for row in pair_wrong_rows]):.6f}")
    lines.append("")
    # 這裡直接把全部文檔 rows 依 divergence 由低到高切成 quantiles 份，
    # 所以報表中的 [All docs grouped by divergence quantiles] 就是從這行產生的
    lines.extend(format_quantile_table("[All docs grouped by divergence quantiles]", build_quantile_summaries(rows, quantiles)))
    lines.append("")
    lines.extend(
        format_quantile_table(
            # 這裡不是用全部 rows，而是只用 selected_rows
            # 因此 [Selected-only docs grouped by divergence quantiles] 裡的 selected_rate 會固定是 1.0000
            "[Selected-only docs grouped by divergence quantiles]",
            build_quantile_summaries(selected_rows, quantiles),
        )
    )

    rounds = sorted({row["round"] for row in rows})
    lines.append("")
    lines.append("[Per-round low-vs-high divergence comparison]")
    for round_idx in rounds:
        round_rows = [row for row in rows if row["round"] == round_idx]
        lines.append(f"Round {round_idx}: rows={len(round_rows)}")
        lines.append(format_low_high_round_summary(round_rows, quantiles))

    if missing_notes:
        lines.append("")
        lines.append("[Missing files skipped]")
        for note in missing_notes:
            lines.append(f"  {note}")

    lines.append("")
    lines.append("Interpretation guide:")
    lines.append("  1. 如果 Q1 (最低 divergence) 的 selected_rate 高於最高 divergence 分組，表示 divergence 較低的文檔更可能被選中")
    lines.append("  2. 如果 Q1 (最低 divergence) 的 pair_full_rate 或 full_rate 較高，表示 divergence 較低的文檔偽標籤更可靠")
    lines.append("  3. 如果 mean_divergence(selected) 小於 mean_divergence(unselected)，且 mean_divergence(correct) 小於 mean_divergence(wrong)，也能從另一個角度支持相同結論")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    # 先依照命令列參數解析出要分析的實驗目錄、解析過程中的附加說明，以及 summary 檔路徑
    experiment_dirs, resolution_notes, summary_path = resolve_experiment_dirs(args)

    # all_rows: 收集所有實驗、所有 fold、所有 round 的文檔級統計資料
    all_rows: list[dict] = []
    # missing_notes: 收集缺檔或缺輪次時的說明，最後會附在報告中
    missing_notes: list[str] = []
    # pseudo_dirs: 記錄每個實驗實際對應到哪個 pseudo_results_* 目錄，供報告列出
    pseudo_dirs: list[Path] = []

    # 逐一走訪本次要分析的每個實驗目錄
    for experiment_dir in experiment_dirs:
        # 先解析出這個實驗真正要讀的 pseudo_results_* 目錄
        pseudo_dir = resolve_pseudo_dir(experiment_dir)
        # 把解析出的 pseudo 目錄記下來，之後輸出報告時會一併列出
        pseudo_dirs.append(pseudo_dir)
        # 依照使用者指定的 fold 範圍逐一處理
        for fold in range(args.fold_start, args.fold_end + 1):
            # 依照使用者指定的 self-training round 範圍逐一處理
            for round_idx in range(args.round_start, args.round_end + 1):
                # 讀取這個實驗 / fold / round 的資料
                # 若該輪缺檔，note 會帶回缺失原因
                rows, note = load_round_rows(pseudo_dir, experiment_dir.name, fold, round_idx)
                # 把這一輪成功整理出的 rows 併入全域 all_rows，供後續整體統計使用
                all_rows.extend(rows)
                # 若這一輪有缺檔或其他可記錄的訊息，就加上實驗名稱後存入 missing_notes
                if note is not None:
                    missing_notes.append(f"{experiment_dir.name}: {note}")

    if not all_rows:
        raise RuntimeError("No rows were collected. Check experiment paths and fold/round range.")

    # 把前面整理好的實驗資訊、pseudo 目錄、所有統計 rows、缺檔訊息與 quantile 設定
    # 組裝成最終要輸出的完整報告文字
    report_text = build_report(
        experiment_dirs=experiment_dirs,
        pseudo_dirs=pseudo_dirs,
        rows=all_rows,
        missing_notes=missing_notes,
        resolution_notes=resolution_notes,
        quantiles=args.quantiles,
    )

    # 先決定實際輸出的檔案路徑：若有傳 --output 就用使用者指定值，
    # 否則在 summary-file 模式下自動用 summary 檔名推導輸出檔名
    output_path = resolve_output_path(args, summary_path)
    # 如果輸出目錄不存在，就先建立整個父目錄
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # 將報告文字以 utf-8 編碼寫入檔案
    output_path.write_text(report_text, encoding="utf-8")
    print(report_text)
    print(f"Report written to: {output_path}")


if __name__ == "__main__":
    main()