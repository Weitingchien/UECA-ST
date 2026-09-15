#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
檢查 self_training_avg3 使用的 emotion / cause / pair 三個來源 checkpoint 是否真的不同

這支工具會：
1. 從單一實驗目錄或 multi-seed summary 檔解析實驗資料夾
2. 讀取每個 fold 的 fold*_best_val_checkpoint_emo/cause/pair.txt
3. 列出三個 task-specific 最佳模型對應的 stage、iteration 與 checkpoint 名稱
4. 統計每個 fold 三者是否來自相同 round / 相同 checkpoint

使用範例：
    python utils/inspect_avg3_checkpoint_sources.py \
      --summary results_ep_split10_t1v1te1_u7_aligned_disjoint_2019/UECA_Prompt_..._avgs_simple_..._summary.txt \
      --experiments-root /mnt/h/ep_split10_t1v1te1_u7_aligned_disjoint_2019

    python utils/inspect_avg3_checkpoint_sources.py \
      --experiment-dir /mnt/h/ep_split10_t1v1te1_u7_aligned_disjoint_2019/prompt_ECPE_few_shot_ST_... \
      --seed-label seed42
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import torch


# 把專案根目錄加入 sys.path，讓工具可直接匯入 NeST 底下的模組
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from NeST.model_weight_averaging import extract_state_dict, load_checkpoint_object


# 這個正則用來從 summary 檔裡抓出實驗資料夾名稱
SUMMARY_EXPERIMENT_PATTERN = re.compile(r"^\s*-\s*(prompt_ECPE_few_shot_ST_[^\s]+)\s*$")

# 這個正則用來從實驗資料夾名稱抓出 seed，例如 seed42
SEED_PATTERN = re.compile(r"seed(\d+)")

# 這個正則用來從 stage 中抓 round，例如 self_training_round_4
ROUND_PATTERN = re.compile(r"round_(\d+)")


def parse_args() -> argparse.Namespace:
    """解析命令列參數"""
    parser = argparse.ArgumentParser(description="檢查 avg3 的三個來源 checkpoint 是否真的不同")

    # summary 與 experiment-dir 二擇一，避免輸入來源混亂
    source_group = parser.add_mutually_exclusive_group(required=True)

    # 提供 multi-seed summary 檔時，工具會自動解析出多個實驗資料夾
    source_group.add_argument("--summary", help="multi-seed summary 檔路徑")

    # 也支援直接指定單一實驗資料夾，適合先檢查單一 seed
    source_group.add_argument("--experiment-dir", help="單一實驗資料夾路徑")

    # 若使用 summary，就必須搭配 experiments-root 才能組成完整路徑
    parser.add_argument("--experiments-root", default=None, help="實驗父目錄，搭配 --summary 使用")

    # 若直接指定單一實驗資料夾，可以手動指定顯示在表格裡的 seed 標籤
    parser.add_argument("--seed-label", default=None, help="單一實驗模式下的 seed 標籤，例如 seed42")

    # 預設檢查 fold1 到 fold10
    parser.add_argument("--fold-start", type=int, default=1, help="起始 fold 編號")
    parser.add_argument("--fold-end", type=int, default=10, help="結束 fold 編號")

    # 若啟用這個選項，會直接載入 checkpoint 並比對 state_dict 是否完全相同
    parser.add_argument(
        "--compare-weights",
        action="store_true",
        help="直接比較 emo/cause/pair checkpoint 的 state_dict 是否完全相同",
    )

    # 可選的純文字報告輸出路徑。
    parser.add_argument("--output", default=None, help="可選的純文字報告輸出路徑")

    # 可選的 CSV 輸出路徑，方便直接匯入 Excel 或論文表格整理
    parser.add_argument("--csv-output", default=None, help="可選的 CSV 輸出路徑")

    return parser.parse_args()


def read_text(path: Path) -> str:
    """用統一設定讀文字檔"""
    return path.read_text(encoding="utf-8", errors="ignore")


def extract_seed(experiment_name: str) -> str:
    """從實驗資料夾名稱擷取 seed"""
    match = SEED_PATTERN.search(experiment_name)
    if match is None:
        raise ValueError(f"無法從實驗資料夾名稱擷取 seed: {experiment_name}")
    return f"seed{match.group(1)}"


def seed_sort_key(seed_name: str) -> tuple[int, str]:
    """讓 seed20、seed42、seed60 依數字排序"""
    match = SEED_PATTERN.search(seed_name)
    if match:
        return (int(match.group(1)), seed_name)
    return (10**9, seed_name)


def parse_summary_experiment_map(summary_path: Path, experiments_root: Path) -> dict[str, Path]:
    """從 summary 檔建立 seed -> experiment_dir 對照表。"""
    text = read_text(summary_path)
    experiment_map: dict[str, Path] = {}

    for line in text.splitlines():
        match = SUMMARY_EXPERIMENT_PATTERN.match(line)
        if not match:
            continue

        experiment_name = match.group(1)
        seed_name = extract_seed(experiment_name)
        experiment_dir = experiments_root / experiment_name

        if not experiment_dir.is_dir():
            raise FileNotFoundError(f"找不到實驗資料夾: {experiment_dir}")

        if seed_name in experiment_map:
            raise ValueError(f"summary 中出現重複 seed: {seed_name}")

        experiment_map[seed_name] = experiment_dir

    if not experiment_map:
        raise ValueError(f"無法從 summary 中解析任何實驗資料夾: {summary_path}")

    return experiment_map


def build_experiment_map(args: argparse.Namespace) -> dict[str, Path]:
    """依照命令列參數建立 seed -> experiment_dir 對照表"""
    if args.summary:
        if not args.experiments_root:
            raise ValueError("使用 --summary 時，必須同時提供 --experiments-root")

        summary_path = Path(args.summary)
        experiments_root = Path(args.experiments_root)

        if not summary_path.is_file():
            raise FileNotFoundError(f"找不到 summary 檔案: {summary_path}")
        if not experiments_root.is_dir():
            raise FileNotFoundError(f"找不到 experiments_root: {experiments_root}")

        return parse_summary_experiment_map(summary_path, experiments_root)

    experiment_dir = Path(args.experiment_dir)
    if not experiment_dir.is_dir():
        raise FileNotFoundError(f"找不到 experiment_dir: {experiment_dir}")

    if args.seed_label:
        seed_name = args.seed_label
    else:
        seed_name = extract_seed(experiment_dir.name)

    return {seed_name: experiment_dir}


def parse_metadata(metadata_path: Path) -> dict[str, str | int | None]:
    """解析單一 fold 的 task-specific metadata 檔"""
    if not metadata_path.is_file():
        raise FileNotFoundError(f"找不到 metadata 檔案: {metadata_path}")

    result: dict[str, str | int | None] = {
        "stage": None,
        "iteration": None,
        "round": None,
        "validation_loss": None,
        "checkpoint_path": None,
        "checkpoint_name": None,
    }

    for raw_line in read_text(metadata_path).splitlines():
        line = raw_line.strip()

        if line.startswith("Best validation checkpoint (stage: ") and line.endswith(")"):
            stage = line[len("Best validation checkpoint (stage: ") : -1]
            result["stage"] = stage
            round_match = ROUND_PATTERN.search(stage)
            result["round"] = int(round_match.group(1)) if round_match else None
        elif line.startswith("Iteration: "):
            iteration_text = line.split(": ", 1)[1]
            result["iteration"] = int(iteration_text) if iteration_text.isdigit() else iteration_text
        elif line.startswith("Validation loss: "):
            result["validation_loss"] = line.split(": ", 1)[1]
        elif line.startswith("Checkpoint path: "):
            checkpoint_path = line.split(": ", 1)[1]
            result["checkpoint_path"] = checkpoint_path
            result["checkpoint_name"] = Path(checkpoint_path).name

    return result


def resolve_task_checkpoint_path(experiment_dir: Path, checkpoint_name: str | None) -> Path | None:
    """把 metadata 中的 checkpoint 檔名解析成實際檔案路徑"""
    if checkpoint_name is None:
        return None

    candidate = experiment_dir / "self_training_models" / checkpoint_name
    if candidate.is_file():
        return candidate

    return None


def load_state_dict_cached(checkpoint_path: Path, cache: dict[Path, dict]) -> dict:
    """載入 checkpoint 的 state_dict，並用快取避免重複讀檔"""
    if checkpoint_path not in cache:
        checkpoint_object = load_checkpoint_object(checkpoint_path, map_location="cpu")
        cache[checkpoint_path] = extract_state_dict(checkpoint_object)
    return cache[checkpoint_path]


def are_state_dicts_equal(left_state_dict: dict, right_state_dict: dict) -> bool:
    """逐個 tensor 比對兩個 state_dict 是否完全一致"""
    if list(left_state_dict.keys()) != list(right_state_dict.keys()):
        return False

    for key in left_state_dict.keys():
        if not torch.equal(left_state_dict[key], right_state_dict[key]):
            return False

    return True


def compare_checkpoint_weights(
    emo_path: Path | None,
    cause_path: Path | None,
    pair_path: Path | None,
    cache: dict[Path, dict],
) -> dict[str, bool | None]:
    """比較 emo / cause / pair 三個 checkpoint 的權重是否完全相同"""
    if emo_path is None or cause_path is None or pair_path is None:
        return {
            "emo_cause_same_weights": None,
            "emo_pair_same_weights": None,
            "cause_pair_same_weights": None,
            "all_same_weights": None,
            "weight_compared": False,
        }

    emo_state_dict = load_state_dict_cached(emo_path, cache)
    cause_state_dict = load_state_dict_cached(cause_path, cache)
    pair_state_dict = load_state_dict_cached(pair_path, cache)

    emo_cause_same = are_state_dicts_equal(emo_state_dict, cause_state_dict)
    emo_pair_same = are_state_dicts_equal(emo_state_dict, pair_state_dict)
    cause_pair_same = are_state_dicts_equal(cause_state_dict, pair_state_dict)

    return {
        "emo_cause_same_weights": emo_cause_same,
        "emo_pair_same_weights": emo_pair_same,
        "cause_pair_same_weights": cause_pair_same,
        "all_same_weights": emo_cause_same and emo_pair_same and cause_pair_same,
        "weight_compared": True,
    }

def collect_rows(
    experiment_map: dict[str, Path],
    fold_start: int,
    fold_end: int,
    compare_weights: bool,
) -> list[dict[str, str | int | bool | None]]:
    """蒐集所有 seed / fold 的來源 checkpoint 對照資料。"""
    rows: list[dict[str, str | int | bool | None]] = []

    # 這個快取用來避免同一個 checkpoint 在多次比較時被重複載入
    state_dict_cache: dict[Path, dict] = {}

    for seed_name in sorted(experiment_map.keys(), key=seed_sort_key):
        experiment_dir = experiment_map[seed_name]

        for fold_id in range(fold_start, fold_end + 1):
            emo = parse_metadata(experiment_dir / f"fold{fold_id}_best_val_checkpoint_emo.txt")
            cause = parse_metadata(experiment_dir / f"fold{fold_id}_best_val_checkpoint_cause.txt")
            pair = parse_metadata(experiment_dir / f"fold{fold_id}_best_val_checkpoint_pair.txt")

            rounds = [emo["round"], cause["round"], pair["round"]]
            iterations = [emo["iteration"], cause["iteration"], pair["iteration"]]
            checkpoint_names = [emo["checkpoint_name"], cause["checkpoint_name"], pair["checkpoint_name"]]

            round_set = {value for value in rounds if value is not None}
            iteration_set = {value for value in iterations if value is not None}

            # 把三個 task-specific checkpoint 檔名轉成實際路徑，供後續比對權重是否相同
            emo_checkpoint_path = resolve_task_checkpoint_path(experiment_dir, emo["checkpoint_name"])
            cause_checkpoint_path = resolve_task_checkpoint_path(experiment_dir, cause["checkpoint_name"])
            pair_checkpoint_path = resolve_task_checkpoint_path(experiment_dir, pair["checkpoint_name"])

            # 只有使用者要求，且三個 task 恰好來自同一個 round 時，才做 state_dict 比對
            # 若 round 已經不同，僅靠 round 差異就足以說明它們不是同一次最佳模型選擇
            if compare_weights and len(round_set) == 1:
                weight_comparison = compare_checkpoint_weights(
                    emo_checkpoint_path,
                    cause_checkpoint_path,
                    pair_checkpoint_path,
                    state_dict_cache,
                )
            else:
                weight_comparison = {
                    "emo_cause_same_weights": None,
                    "emo_pair_same_weights": None,
                    "cause_pair_same_weights": None,
                    "all_same_weights": None,
                    "weight_compared": False,
                }

            rows.append(
                {
                    "seed": seed_name,
                    "fold": fold_id,
                    "experiment_dir": experiment_dir.name,
                    "emo_stage": emo["stage"],
                    "emo_round": emo["round"],
                    "emo_iteration": emo["iteration"],
                    "emo_checkpoint": emo["checkpoint_name"],
                    "cause_stage": cause["stage"],
                    "cause_round": cause["round"],
                    "cause_iteration": cause["iteration"],
                    "cause_checkpoint": cause["checkpoint_name"],
                    "pair_stage": pair["stage"],
                    "pair_round": pair["round"],
                    "pair_iteration": pair["iteration"],
                    "pair_checkpoint": pair["checkpoint_name"],
                    "distinct_round_count": len(round_set),
                    "all_same_round": len(round_set) == 1,
                    "all_same_iteration": len(iteration_set) == 1,
                    "emo_cause_same_weights": weight_comparison["emo_cause_same_weights"],
                    "emo_pair_same_weights": weight_comparison["emo_pair_same_weights"],
                    "cause_pair_same_weights": weight_comparison["cause_pair_same_weights"],
                    "all_same_weights": weight_comparison["all_same_weights"],
                    "weight_compared": weight_comparison["weight_compared"],
                }
            )

    return rows


def summarize_rows(rows: list[dict[str, str | int | bool | None]]) -> dict[str, int]:
    """統計三個來源 checkpoint 是否相同。"""
    summary = {
        "total_rows": len(rows),
        "all_same_round": 0,
        "same_two_rounds": 0,
        "all_different_rounds": 0,
        "same_round_weight_compared": 0,
        "same_round_all_same_weights": 0,
        "same_round_different_weights": 0,
    }

    for row in rows:
        distinct_round_count = int(row["distinct_round_count"])
        weight_compared = bool(row["weight_compared"])
        all_same_weights = row["all_same_weights"]

        if distinct_round_count == 1:
            summary["all_same_round"] += 1
        elif distinct_round_count == 2:
            summary["same_two_rounds"] += 1
        elif distinct_round_count >= 3:
            summary["all_different_rounds"] += 1

        if weight_compared and all_same_weights is not None:
            summary["same_round_weight_compared"] += 1
            if all_same_weights:
                summary["same_round_all_same_weights"] += 1
            else:
                summary["same_round_different_weights"] += 1

    return summary


def build_report(
    rows: list[dict[str, str | int | bool | None]],
    summary: dict[str, int],
    args: argparse.Namespace,
) -> str:
    """把結果組成容易閱讀的純文字報告。"""
    lines: list[str] = []

    lines.append("=" * 86)
    lines.append("Avg3 來源 Checkpoint 檢查報告")
    lines.append("=" * 86)

    if args.summary:
        lines.append(f"summary: {args.summary}")
        lines.append(f"experiments_root: {args.experiments_root}")
    else:
        lines.append(f"experiment_dir: {args.experiment_dir}")

    lines.append(f"fold 範圍: {args.fold_start} - {args.fold_end}")
    lines.append("")

    lines.append("整體統計:")
    lines.append(f"  - 檢查列數 (seed x fold): {summary['total_rows']}")
    lines.append(f"  - 三個 task 來自同一個 round 的列數: {summary['all_same_round']}")
    lines.append(f"  - 三個 task 中有兩個 round 相同的列數: {summary['same_two_rounds']}")
    lines.append(f"  - 三個 task 來自三個不同 round 的列數: {summary['all_different_rounds']}")

    # 只有真的做過 state_dict 比對時，才輸出權重是否相同的統計。
    if summary["same_round_weight_compared"] > 0:
        lines.append(f"  - 已做權重比對的同 round 列數: {summary['same_round_weight_compared']}")
        lines.append(f"  - 同 round 且三個 task 權重完全相同的列數: {summary['same_round_all_same_weights']}")
        lines.append(f"  - 同 round 但三個 task 權重至少有一個不同的列數: {summary['same_round_different_weights']}")

    lines.append("")

    lines.append("逐 fold 對照表:")
    if summary["same_round_weight_compared"] > 0:
        lines.append(
            "  seed\tfold\temo_round\tcause_round\tpair_round\tdistinct_rounds\tall_same_round\t"
            "weight_compared\tall_same_weights\temo_ckpt\tcause_ckpt\tpair_ckpt"
        )
    else:
        lines.append(
            "  seed\tfold\temo_round\tcause_round\tpair_round\tdistinct_rounds\tall_same_round\t"
            "emo_ckpt\tcause_ckpt\tpair_ckpt"
        )

    for row in rows:
        row_prefix = (
            "  "
            f"{row['seed']}\t"
            f"{row['fold']}\t"
            f"{row['emo_round']}\t"
            f"{row['cause_round']}\t"
            f"{row['pair_round']}\t"
            f"{row['distinct_round_count']}\t"
            f"{row['all_same_round']}"
        )

        if summary["same_round_weight_compared"] > 0:
            row_prefix += f"\t{row['weight_compared']}\t{row['all_same_weights']}"

        lines.append(
            row_prefix
            + f"\t{row['emo_checkpoint']}"
            + f"\t{row['cause_checkpoint']}"
            + f"\t{row['pair_checkpoint']}"
        )

    lines.append("")
    lines.append("解讀建議:")
    lines.append("  - 若很多列的 emo_round / cause_round / pair_round 不同，代表 Avg3 確實在融合互補模型。")
    lines.append("  - 只有三個 task 恰好同一個 round 時，才需要再看 all_same_weights。")
    lines.append("  - 若同一 round 但 all_same_weights=True，代表該列其實是同一組權重被存成三個 task 檔案。")
    lines.append("  - 若同一 round 且 all_same_weights=False，才可更直接說明 Avg3 融合的是不同權重模型。")
    lines.append("  - 可將 distinct_round_count > 1 的比例寫成論文中的補充證據。")

    return "\n".join(lines).rstrip() + "\n"


def write_csv(rows: list[dict[str, str | int | bool | None]], csv_path: Path) -> None:
    """把逐列結果寫成 CSV"""
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "seed",
        "fold",
        "experiment_dir",
        "emo_stage",
        "emo_round",
        "emo_iteration",
        "emo_checkpoint",
        "cause_stage",
        "cause_round",
        "cause_iteration",
        "cause_checkpoint",
        "pair_stage",
        "pair_round",
        "pair_iteration",
        "pair_checkpoint",
        "distinct_round_count",
        "all_same_round",
        "all_same_iteration",
        "weight_compared",
        "emo_cause_same_weights",
        "emo_pair_same_weights",
        "cause_pair_same_weights",
        "all_same_weights",
    ]

    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """主流程"""
    args = parse_args()

    experiment_map = build_experiment_map(args)
    rows = collect_rows(experiment_map, args.fold_start, args.fold_end, args.compare_weights)
    summary = summarize_rows(rows)
    report = build_report(rows, summary, args)

    print(report, end="")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report, encoding="utf-8")
        print(f"\n純文字報告已寫入: {output_path}")

    if args.csv_output:
        csv_path = Path(args.csv_output)
        write_csv(rows, csv_path)
        print(f"CSV 已寫入: {csv_path}")


if __name__ == "__main__":
    main()