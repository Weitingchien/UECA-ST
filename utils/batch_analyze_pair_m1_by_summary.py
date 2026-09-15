#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""從 multi-seed summary 自動產生 Pair(m1) 分層分析 CSV"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path, PureWindowsPath
from statistics import mean, stdev
from typing import Iterable, Optional


try:
    from analyze_pair_m1_by_doc_complexity import (
        MetricTally,
        analyze_fold,
        empty_group_tallies,
        merge_tallies,
        normalize_no_pair_labels,
        parse_folds,
        tally_to_row,
    )
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parent))
    from analyze_pair_m1_by_doc_complexity import (
        MetricTally,
        analyze_fold,
        empty_group_tallies,
        merge_tallies,
        normalize_no_pair_labels,
        parse_folds,
        tally_to_row,
    )


DEFAULT_RESULT_DIRS = [
    "results_ep_enecpe_reccon_merged_t1v1te1_u7",
    "results_ep_enecpe_reccon_merged_t9te1",
    "results_ep_split10_t1v1te1_u7_aligned_disjoint_2019",
    "results_ep_split10_t9te1_aligned_disjoint_2019",
]

DEFAULT_SEARCH_DRIVES = ["D", "F", "I", "H"]

GROUP_ORDER = ["overall", "0_pair", "1_pair", "2_pairs", "3plus_pairs", "2plus_pairs"]

PERCENT_FIELDS = {
    "doc_percent",
    "official_precision",
    "official_recall",
    "official_f1",
    "set_based_precision",
    "set_based_recall",
    "set_based_f1",
    "oracle_recall_ceiling_one_pair_per_clause",
}

NUMERIC_FIELDS = {
    "doc_count",
    "total_tp_official_like",
    "total_tp_set_based",
    "total_pred",
    "total_gt",
    "gt_pairs_not_recoverable_by_one_pair_per_clause",
    "avg_gt_pairs_per_doc",
    "avg_pred_pairs_per_doc",
    "invalid_pair_labels",
}


def parse_args() -> argparse.Namespace:
    """解析命令列參數。"""

    parser = argparse.ArgumentParser(description="依 results_* summary 自動產生 Pair(m1) 分層分析 CSV。")
    parser.add_argument("--result-dirs", nargs="*", default=DEFAULT_RESULT_DIRS, help="要掃描的 results_* 資料夾。")
    parser.add_argument("--summary-glob", default="*_summary.txt", help="summary 檔案 glob。")
    parser.add_argument("--output-dir", default="analysis", help="輸出 CSV 的資料夾。")
    parser.add_argument("--folds", nargs="*", default=["1-10"], help="要分析的 folds，例如 1-10 或 1 2 3。")
    parser.add_argument("--search-drives", nargs="*", default=DEFAULT_SEARCH_DRIVES, help="搜尋實驗根目錄的磁碟代號。")
    parser.add_argument("--extra-experiment-roots", nargs="*", default=[], help="額外指定的實驗根目錄。")
    parser.add_argument("--gt-dir-map", nargs="*", default=[], help="手動覆寫 GT 目錄，格式 result_dir=gt_dir。")
    parser.add_argument("--split-multi-emotion", dest="split_multi_emotion", action="store_true", help="分層時拆分複合 emotion_category。")
    parser.add_argument("--no-split-multi-emotion", dest="split_multi_emotion", action="store_false", help="分層時不拆分複合 emotion_category。")
    parser.set_defaults(split_multi_emotion=True)
    parser.add_argument("--keep-duplicate-pairs", action="store_true", help="保留同篇文件內重複的 GT pairs。")
    parser.add_argument("--include-fold-rows", action="store_true", help="除了 all 累計列，也輸出每個 fold 的分層結果。")
    parser.add_argument("--no-pair-labels", nargs="*", default=["无", "none"], help="預測檔中代表無 pair 的標籤。")
    parser.add_argument("--long-file-names", action="store_true", help="使用完整 summary 名稱輸出檔案；預設使用 Excel 友善短檔名。")
    parser.add_argument("--strict", action="store_true", help="遇到缺少 summary、GT 或實驗目錄時直接失敗。")
    return parser.parse_args()


def read_text(path: Path) -> str:
    """讀取文字檔，忽略少數壞字元。"""

    return path.read_text(encoding="utf-8", errors="ignore")


def normalize_experiment_name(raw_name: str) -> str:
    """把 summary 中的實驗目錄字串正規化成 basename。"""

    cleaned = raw_name.strip().strip("'\"")
    windows_name = PureWindowsPath(cleaned).name
    unix_name = Path(cleaned).name
    return unix_name if len(unix_name) <= len(windows_name) else windows_name


def parse_experiment_names(summary_path: Path) -> list[str]:
    """解析 summary 檔內「彙整的實驗目錄」清單。"""

    names: list[str] = []
    in_section = False
    for line in read_text(summary_path).splitlines():
        if "彙整的實驗目錄" in line:
            in_section = True
            continue
        if not in_section:
            continue
        if not line.strip():
            break
        match = re.match(r"\s*-\s+(.+?)\s*$", line)
        if match:
            names.append(normalize_experiment_name(match.group(1)))
    return names


def parse_gt_dir_map(items: Iterable[str]) -> dict[str, Path]:
    """解析 result_dir=gt_dir 形式的 GT 目錄覆寫設定。"""

    mapping: dict[str, Path] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--gt-dir-map 格式錯誤，應為 result_dir=gt_dir: {item}")
        key, value = item.split("=", 1)
        mapping[key.strip()] = Path(value.strip())
    return mapping


def unique_paths(paths: Iterable[Path]) -> list[Path]:
    """保留順序並移除重複路徑。"""

    seen: set[str] = set()
    output: list[Path] = []
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        output.append(path)
    return output


def result_suffix(result_dir_name: str) -> str:
    """取得 results_* 對應的 ep_* 名稱。"""

    return result_dir_name[len("results_") :] if result_dir_name.startswith("results_") else result_dir_name


def summary_output_stem(summary_path: Path) -> str:
    """由 summary 檔名取得輸出檔名前半段。"""

    name = summary_path.name
    if name.endswith("_summary.txt"):
        stem = name[: -len("_summary.txt")]
    else:
        stem = summary_path.stem
    if stem.endswith("_CE"):
        stem = stem[: -len("_CE")]
    return stem


def result_short_name(result_dir_name: str) -> str:
    """取得 Excel 友善的 results 資料夾短名稱。"""

    suffix = result_suffix(result_dir_name)
    mapping = {
        "ep_enecpe_reccon_merged_t1v1te1_u7": "en_t1v1",
        "ep_enecpe_reccon_merged_t9te1": "en_t9",
        "ep_enecpe_reccon_merged_t9": "en_t9",
        "ep_split10_t1v1te1_u7_aligned_disjoint_2019": "zh_t1v1",
        "ep_split10_t9te1_aligned_disjoint_2019": "zh_t9",
    }
    return mapping.get(suffix, suffix.replace("ep_", ""))


def summary_short_stem(summary_path: Path) -> str:
    """把很長的 summary 名稱壓成可辨識短名稱。"""

    stem = summary_output_stem(summary_path)
    replacements = [
        ("UECA_Prompt_f1-10_i70_lr1e-5_bs8_wd0.01_", ""),
        ("prompt_ECPE_enecpe_reccon_merged_train9_test1_UECA_CE_supervised_full_truncated_en", "trunc_en"),
        ("prompt_2013_2015_ECPE_split10_aligned_disjoint_2019_UECA_CE_supervised_full_truncated", "trunc_zh"),
        ("emotion_clause_knnemotion_clause", "emo"),
        ("cause_clause_knncause_clause", "cause"),
        ("emotion_clause", "emo"),
        ("cause_clause", "cause"),
        ("nest_k", "k"),
        ("nestmul", "mul"),
        ("nbeta", "nb"),
        ("gamma", "g"),
        ("nlmnest_", ""),
        ("avgs_simple", "avgS"),
        ("avgs_iter", "avgI"),
        ("remove_pseudo_v2", "rmPv2"),
        ("initemo_stemo_testemo", "init_st_test_emo"),
        ("initcau_stcau_testcau", "init_st_test_cause"),
        ("initemo_stemo", "init_st_emo"),
        ("initcau_stcau", "init_st_cause"),
    ]
    for old_text, new_text in replacements:
        stem = stem.replace(old_text, new_text)
    stem = re.sub(r"_+", "_", stem).strip("_")
    return stem[:120].rstrip("_")


def infer_experiment_root_names(result_dir_name: str) -> list[str]:
    """由 results 資料夾名稱推測實驗根目錄名稱。"""

    base_name = result_suffix(result_dir_name)
    names = [base_name]
    if base_name.endswith("_t9"):
        names.append(f"{base_name}te1")
    if base_name.endswith("_t9te1"):
        names.append(base_name[: -len("te1")])
    return list(dict.fromkeys(names))


def windows_and_wsl_drive_roots(drive: str, root_name: str) -> list[Path]:
    """產生 Windows 與 WSL 兩種磁碟根目錄候選。"""

    drive_upper = drive.rstrip(":\\/").upper()
    drive_lower = drive_upper.lower()
    return [Path(f"{drive_upper}:/") / root_name, Path("/mnt") / drive_lower / root_name]


def resolve_result_dir(raw_dir: str) -> Path:
    """解析 results 資料夾路徑，並處理 t9/t9te1 命名差異。"""

    path = Path(raw_dir)
    if path.is_dir():
        return path
    cwd_path = Path.cwd() / raw_dir
    if cwd_path.is_dir():
        return cwd_path
    if raw_dir.endswith("_t9"):
        alt = Path(f"{raw_dir}te1")
        if alt.is_dir():
            return alt
        if (Path.cwd() / alt).is_dir():
            return Path.cwd() / alt
    return path


def candidate_experiment_roots(result_dir: Path, extra_roots: list[str], search_drives: list[str]) -> list[Path]:
    """建立搜尋 seed 實驗資料夾時的根目錄候選。"""

    root_names = infer_experiment_root_names(result_dir.name)
    candidates = [Path(root) for root in extra_roots]
    candidates.append(Path.cwd())
    candidates.extend(result_dir.parent / root_name for root_name in root_names)
    candidates.extend(Path.cwd() / root_name for root_name in root_names)
    for drive in search_drives:
        for root_name in root_names:
            candidates.extend(windows_and_wsl_drive_roots(drive, root_name))
    return unique_paths(candidates)


def resolve_experiment_dir(experiment_name: str, result_dir: Path, extra_roots: list[str], search_drives: list[str]) -> Optional[Path]:
    """從多個候選根目錄中找到 seed 實驗資料夾。"""

    direct_path = Path(experiment_name)
    if direct_path.is_dir():
        return direct_path
    for root in candidate_experiment_roots(result_dir, extra_roots, search_drives):
        candidate = root / experiment_name
        if candidate.is_dir():
            return candidate
    return None


def gt_dir_names_for_result(result_dir_name: str) -> list[str]:
    """由 results 資料夾名稱推測 GT split 資料夾名稱。"""

    base_name = result_suffix(result_dir_name)
    if base_name.startswith("ep_"):
        base_name = base_name[len("ep_") :]
    candidates = [base_name]
    replacements = [
        ("_t1v1te1_u7", "_train1_val1_test1_unlabeled7"),
        ("_t1te1v1_u7", "_train1_test1_val1_unlabeled7"),
        ("_t9te1", "_train9_test1"),
        ("_t9", "_train9_test1"),
    ]
    for old_text, new_text in replacements:
        if old_text in base_name:
            candidates.append(base_name.replace(old_text, new_text))
    return list(dict.fromkeys(candidates))


def candidate_gt_dirs(result_dir: Path, search_drives: list[str]) -> list[Path]:
    """建立 GT split 資料夾候選清單。"""

    names = gt_dir_names_for_result(result_dir.name)
    candidates: list[Path] = []
    candidates.extend(result_dir.parent / name for name in names)
    candidates.extend(Path.cwd() / name for name in names)
    for drive in search_drives:
        for name in names:
            candidates.extend(windows_and_wsl_drive_roots(drive, name))
    return unique_paths(candidates)


def resolve_gt_dir(result_dir: Path, gt_map: dict[str, Path], search_drives: list[str]) -> Optional[Path]:
    """解析某個 results 資料夾對應的 GT split 資料夾。"""

    for key in (result_dir.name, result_suffix(result_dir.name)):
        mapped = gt_map.get(key)
        if mapped is not None:
            return mapped if mapped.is_dir() else None
    for candidate in candidate_gt_dirs(result_dir, search_drives):
        if candidate.is_dir():
            return candidate
    return None


def analyze_experiment(
    *,
    gt_dir: Path,
    pred_dir: Path,
    folds: list[int],
    keep_duplicates: bool,
    split_multi_emotion: bool,
    include_fold_rows: bool,
    no_pair_labels: set[str],
) -> list[dict[str, object]]:
    """分析單一 seed 實驗目錄並回傳 CSV rows。"""

    output_rows: list[dict[str, object]] = []
    all_tallies = empty_group_tallies()
    fold_tallies_by_label: dict[str, dict[str, MetricTally]] = {}
    missing_files: list[str] = []

    for fold in folds:
        gt_path = gt_dir / f"fold{fold}_test.json"
        pred_path = pred_dir / f"fold{fold}_text_result.txt"
        if not gt_path.is_file():
            missing_files.append(str(gt_path))
            continue
        if not pred_path.is_file():
            missing_files.append(str(pred_path))
            continue
        fold_tallies = analyze_fold(
            gt_path=gt_path,
            pred_path=pred_path,
            keep_duplicates=keep_duplicates,
            split_multi_emotion=split_multi_emotion,
            no_pair_labels=no_pair_labels,
        )
        merge_tallies(all_tallies, fold_tallies)
        fold_tallies_by_label[f"fold{fold}"] = fold_tallies

    if all_tallies["overall"].doc_count == 0:
        missing_hint = ", ".join(missing_files[:3])
        raise FileNotFoundError(f"沒有成功分析任何 fold: {pred_dir}; missing examples: {missing_hint}")

    if include_fold_rows:
        for fold_label, fold_tallies in fold_tallies_by_label.items():
            total_docs = fold_tallies["overall"].doc_count
            for group in GROUP_ORDER:
                output_rows.append(
                    tally_to_row(
                        experiment=pred_dir.name,
                        fold_label=fold_label,
                        group=group,
                        tally=fold_tallies[group],
                        total_docs=total_docs,
                    )
                )

    total_docs = all_tallies["overall"].doc_count
    for group in GROUP_ORDER:
        output_rows.append(
            tally_to_row(
                experiment=pred_dir.name,
                fold_label="all",
                group=group,
                tally=all_tallies[group],
                total_docs=total_docs,
            )
        )
    return output_rows


def parse_percent(text: object) -> float:
    """把 '31.63%' 轉成 31.63。"""

    return float(str(text).replace("%", "").strip())


def parse_number(text: object) -> float:
    """把 CSV 欄位轉成浮點數。"""

    return float(str(text).strip())


def format_number(value: float) -> str:
    """格式化平均後的數值欄位。"""

    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.4f}" if abs(value) < 10 else f"{value:.2f}"


def sample_std(values: list[float]) -> float:
    """計算三 seed 樣本標準差；只有一筆時回傳 0。"""

    return stdev(values) if len(values) >= 2 else 0.0


def std_field_name(field: str) -> str:
    """取得平均列中對應的標準差欄位名稱。"""

    return f"{field}_std"


def make_mean_rows(seed_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """針對 fold=all 的各 group 追加三 seed 平均列。"""

    if not seed_rows:
        return []
    fieldnames = list(seed_rows[0].keys())
    mean_rows: list[dict[str, object]] = []
    all_rows = [row for row in seed_rows if row.get("fold") == "all"]
    for group in GROUP_ORDER:
        group_rows = [row for row in all_rows if row.get("group") == group]
        if not group_rows:
            continue
        mean_row: dict[str, object] = {}
        for field in fieldnames:
            if field == "experiment":
                mean_row[field] = f"MEAN_OF_{len(group_rows)}_SEEDS"
            elif field == "fold":
                mean_row[field] = "all"
            elif field == "group":
                mean_row[field] = group
            elif field in PERCENT_FIELDS:
                values = [parse_percent(row[field]) for row in group_rows]
                mean_row[field] = f"{mean(values):.2f}%"
                mean_row[std_field_name(field)] = f"{sample_std(values):.2f}%"
            elif field in NUMERIC_FIELDS:
                values = [parse_number(row[field]) for row in group_rows]
                mean_row[field] = format_number(mean(values))
                mean_row[std_field_name(field)] = format_number(sample_std(values))
            else:
                mean_row[field] = ""
        mean_rows.append(mean_row)
    return mean_rows


def csv_fieldnames(rows: list[dict[str, object]]) -> list[str]:
    """建立 CSV 欄位，將每個平均值欄位的 std 欄放在原欄位後方。"""

    base_fields = list(rows[0].keys())
    output: list[str] = []
    for field in base_fields:
        if field.endswith("_std"):
            continue
        output.append(field)
        std_field = std_field_name(field)
        if any(std_field in row for row in rows):
            output.append(std_field)
    for row in rows:
        for field in row.keys():
            if field not in output:
                output.append(field)
    return output


def write_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    """寫出 CSV。"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("沒有任何可輸出的 rows。")
    with output_path.open("w", encoding="utf-8-sig", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=csv_fieldnames(rows), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def output_csv_path(summary_path: Path, result_dir: Path, output_dir: Path, *, long_file_names: bool) -> Path:
    """產生使用者指定格式的輸出 CSV 檔名。"""

    if long_file_names:
        filename = f"{summary_output_stem(summary_path)}_{result_suffix(result_dir.name)}_pair_m1_by_complexity.csv"
    else:
        filename = f"{result_short_name(result_dir.name)}_{summary_short_stem(summary_path)}_pair_m1_by_complexity.csv"
    return output_dir / filename


def handle_summary(
    *,
    summary_path: Path,
    result_dir: Path,
    gt_dir: Path,
    folds: list[int],
    args: argparse.Namespace,
) -> Optional[Path]:
    """處理單一 summary 檔。"""

    experiment_names = parse_experiment_names(summary_path)
    if not experiment_names:
        message = f"[skip] 找不到彙整實驗目錄: {summary_path}"
        if args.strict:
            raise ValueError(message)
        print(message)
        return None

    pred_dirs: list[Path] = []
    missing_experiments: list[str] = []
    for experiment_name in experiment_names:
        pred_dir = resolve_experiment_dir(experiment_name, result_dir, args.extra_experiment_roots, args.search_drives)
        if pred_dir is None:
            missing_experiments.append(experiment_name)
            continue
        pred_dirs.append(pred_dir)

    if missing_experiments:
        message = f"[warn] {summary_path.name} 找不到 {len(missing_experiments)} 個實驗目錄: {', '.join(missing_experiments)}"
        if args.strict:
            raise FileNotFoundError(message)
        print(message)
    if not pred_dirs:
        print(f"[skip] 沒有可分析的 seed 實驗目錄: {summary_path}")
        return None

    no_pair_labels = normalize_no_pair_labels(args.no_pair_labels)
    rows: list[dict[str, object]] = []
    for pred_dir in pred_dirs:
        rows.extend(
            analyze_experiment(
                gt_dir=gt_dir,
                pred_dir=pred_dir,
                folds=folds,
                keep_duplicates=args.keep_duplicate_pairs,
                split_multi_emotion=args.split_multi_emotion,
                include_fold_rows=args.include_fold_rows,
                no_pair_labels=no_pair_labels,
            )
        )
    rows.extend(make_mean_rows(rows))

    output_path = output_csv_path(summary_path, result_dir, Path(args.output_dir), long_file_names=args.long_file_names)
    write_csv(rows, output_path)
    print(f"[ok] {output_path} ({len(pred_dirs)} seed dirs)")
    return output_path


def main() -> int:
    """主流程。"""

    args = parse_args()
    folds = parse_folds(args.folds)
    gt_map = parse_gt_dir_map(args.gt_dir_map)
    output_paths: list[Path] = []

    for raw_result_dir in args.result_dirs:
        result_dir = resolve_result_dir(raw_result_dir)
        if not result_dir.is_dir():
            message = f"[skip] 找不到 results 資料夾: {raw_result_dir}"
            if args.strict:
                raise FileNotFoundError(message)
            print(message)
            continue

        gt_dir = resolve_gt_dir(result_dir, gt_map, args.search_drives)
        if gt_dir is None:
            message = f"[skip] 找不到 GT split 資料夾: {result_dir.name}"
            if args.strict:
                raise FileNotFoundError(message)
            print(message)
            continue

        summary_paths = sorted(result_dir.glob(args.summary_glob))
        if not summary_paths:
            message = f"[skip] 找不到 summary 檔: {result_dir / args.summary_glob}"
            if args.strict:
                raise FileNotFoundError(message)
            print(message)
            continue

        print(f"[result] {result_dir.name}: GT={gt_dir}, summaries={len(summary_paths)}")
        for summary_path in summary_paths:
            output_path = handle_summary(
                summary_path=summary_path,
                result_dir=result_dir,
                gt_dir=gt_dir,
                folds=folds,
                args=args,
            )
            if output_path is not None:
                output_paths.append(output_path)

    print(f"完成，輸出 {len(output_paths)} 個 CSV。")
    return 0 if output_paths or not args.strict else 1


if __name__ == "__main__":
    raise SystemExit(main())