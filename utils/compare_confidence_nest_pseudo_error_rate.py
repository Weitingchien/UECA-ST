#!/usr/bin/env python3
"""比較兩種選樣方法所選文檔的完整偽標籤錯誤率。"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from label_utils import configure_fonts


REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_EXPERIMENT_DIR = "ep_split10_t1v1te1_u7_aligned_disjoint_2019"
SUMMARY_DIR = REPO_ROOT / "results_ep_split10_t1v1te1_u7_aligned_disjoint_2019"
CONFIDENCE_SUMMARY = SUMMARY_DIR / (
    "UECA_Prompt_f1-10_i70_lr1e-5_bs8_wd0.01_confidence_emotion_clause_"
    "confmul1_gamma0.5_nlmnest_st5_ste20_remove_pseudo_v2_CE_summary.txt"
)
NEST_SUMMARY = SUMMARY_DIR / (
    "UECA_Prompt_f1-10_i70_lr1e-5_bs8_wd0.01_nest_k5_emotion_clause_"
    "knnemotion_clause_nestmul1_nbeta0.1_nm0.6_gamma0.5_nlmnest_st5_ste20_"
    "remove_pseudo_v2_CE_summary.txt"
)
ANALYSIS_DIR = REPO_ROOT / "analysis" / "confidence_vs_nest_pseudo_error_rate"
PNG_DIR = REPO_ROOT / "png" / "confidence_vs_nest_pseudo_error_rate"

EXPERIMENT_PATTERN = re.compile(r"^\s*-\s*(prompt_ECPE_few_shot_ST_\S+)\s*$")
SEED_PATTERN = re.compile(r"_seed(\d+)_")
METHOD_COLORS = ("#4C78A8", "#D68C45")
GRAY_BACKGROUND = "#DDDDDD"


@dataclass(frozen=True)
class UnitCount:
    """單一方法、seed、fold 與 round 的文件級計數。"""

    method: str
    seed: int
    fold: int
    round_id: int
    selected_documents: int
    error_documents: int

    @property
    def exact_documents(self) -> int:
        return self.selected_documents - self.error_documents

    @property
    def error_rate_percent(self) -> float:
        return self.error_documents / self.selected_documents * 100.0


def default_experiments_root(drive: str) -> Path:
    """依 Windows 或 WSL 執行環境建立實驗根目錄。"""

    drive_root = Path(f"{drive}:/") if os.name == "nt" else Path(f"/mnt/{drive.lower()}")
    return drive_root / DATASET_EXPERIMENT_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="比較兩種選樣方法被選文檔的完整 e+c+p 偽標籤錯誤率"
    )
    parser.add_argument(
        "--confidence-summary", "--method-a-summary",
        dest="method_a_summary", type=Path, default=CONFIDENCE_SUMMARY,
    )
    parser.add_argument(
        "--confidence-root", "--method-a-root",
        dest="method_a_root", type=Path, default=default_experiments_root("I"),
    )
    parser.add_argument(
        "--nest-summary", "--method-b-summary",
        dest="method_b_summary", type=Path, default=NEST_SUMMARY,
    )
    parser.add_argument(
        "--nest-root", "--method-b-root",
        dest="method_b_root", type=Path, default=default_experiments_root("H"),
    )
    parser.add_argument("--method-a-label", default="Confidence")
    parser.add_argument("--method-b-label", default="NeST")
    parser.add_argument("--analysis-dir", type=Path, default=ANALYSIS_DIR)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--background", choices=("gray", "white"), default="gray")
    return parser.parse_args()


def experiment_directories(summary_path: Path, experiments_root: Path) -> list[tuple[int, Path]]:
    """從 multi-seed summary 取得 seed 與實驗資料夾。"""

    names = [
        match.group(1)
        for line in summary_path.read_text(encoding="utf-8").splitlines()
        if (match := EXPERIMENT_PATTERN.match(line))
    ]
    return sorted(
        (int(SEED_PATTERN.search(name).group(1)), experiments_root / name)
        for name in names
    )


def find_pseudo_results_dir(experiment_dir: Path) -> Path:
    """取得實驗內唯一的 pseudo_results 資料夾。"""

    pseudo_dirs = sorted(experiment_dir.glob("pseudo_results_*"))
    if len(pseudo_dirs) != 1:
        raise ValueError(f"預期只有一個 pseudo_results_* 資料夾：{experiment_dir}")
    return pseudo_dirs[0]


def count_selected_documents(json_path: Path) -> tuple[int, int]:
    """計算被選文檔總數，以及 e／c／p 任一位置錯誤的文檔數。"""

    samples = json.loads(json_path.read_text(encoding="utf-8"))
    if not all(sample["ground_truth_available"] for sample in samples):
        raise ValueError(f"存在沒有 ground truth 的文檔：{json_path}")

    error_documents = sum(
        sample["pseudo_label_tokens"] != sample["gt_label_tokens"] for sample in samples
    )
    return len(samples), error_documents


def collect_method_counts(
    method: str,
    summary_path: Path,
    experiments_root: Path,
) -> list[UnitCount]:
    """讀取三個 seed、十折、五輪的實際偽標籤訓練清單。"""

    counts: list[UnitCount] = []
    for seed, experiment_dir in experiment_directories(summary_path, experiments_root):
        pseudo_dir = find_pseudo_results_dir(experiment_dir)
        for round_id in range(1, 6):
            for fold in range(1, 11):
                json_path = pseudo_dir / f"pseudo_labeled_samples_fold{fold}_round{round_id}.json"
                selected, errors = count_selected_documents(json_path)
                counts.append(UnitCount(method, seed, fold, round_id, selected, errors))
    return counts


def pool_by_method_and_round(
    counts: list[UnitCount], methods: tuple[str, str]
) -> list[UnitCount]:
    """跨三個 seed 與十折先合併分子分母，再計算每輪錯誤率"""

    pooled: list[UnitCount] = []
    for method in methods:
        for round_id in range(1, 6):
            rows = [row for row in counts if row.method == method and row.round_id == round_id]
            pooled.append(
                UnitCount(
                    method=method,
                    seed=-1,
                    fold=-1,
                    round_id=round_id,
                    selected_documents=sum(row.selected_documents for row in rows),
                    error_documents=sum(row.error_documents for row in rows),
                )
            )
    return pooled


def write_csv_files(counts: list[UnitCount], pooled: list[UnitCount], output_dir: Path) -> None:
    """輸出可追溯的單位明細與 round-pooled 統計。"""

    output_dir.mkdir(parents=True, exist_ok=True)
    columns = (
        "method",
        "seed",
        "fold",
        "round",
        "selected_documents",
        "exact_documents",
        "error_documents",
        "error_rate_percent",
    )

    for path, rows, include_unit_ids in (
        (output_dir / "seed_fold_round_counts.csv", counts, True),
        (output_dir / "round_pooled_error_rates.csv", pooled, False),
    ):
        with path.open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=columns)
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        "method": row.method,
                        "seed": row.seed if include_unit_ids else "all",
                        "fold": row.fold if include_unit_ids else "all",
                        "round": row.round_id,
                        "selected_documents": row.selected_documents,
                        "exact_documents": row.exact_documents,
                        "error_documents": row.error_documents,
                        "error_rate_percent": f"{row.error_rate_percent:.6f}",
                    }
                )


def annotate_bars(ax: plt.Axes, bars) -> None:
    """在每根柱狀圖上方標示兩位小數。"""

    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.12,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=12,
            color="black",
            zorder=6,
        )


def plot_error_rates(
    pooled: list[UnitCount],
    output_path: Path,
    background: str,
    methods: tuple[str, str],
) -> None:
    """繪製與既有 round-pooled 圖相同字型與版面的雙方法比較圖。"""

    configure_fonts(font_size=16)
    rounds = np.arange(5, dtype=float)
    width = 0.32
    rates = {
        method: [row.error_rate_percent for row in pooled if row.method == method]
        for method in methods
    }

    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=180)
    facecolor = GRAY_BACKGROUND if background == "gray" else "white"
    fig.patch.set_facecolor(facecolor)
    ax.set_facecolor(facecolor)

    method_a_bars = ax.bar(
        rounds - width / 2,
        rates[methods[0]],
        width,
        label=methods[0],
        color=METHOD_COLORS[0],
        edgecolor="white",
        linewidth=1.2,
        zorder=2,
    )
    method_b_bars = ax.bar(
        rounds + width / 2,
        rates[methods[1]],
        width,
        label=methods[1],
        color=METHOD_COLORS[1],
        edgecolor="white",
        linewidth=1.2,
        zorder=2,
    )
    annotate_bars(ax, method_a_bars)
    annotate_bars(ax, method_b_bars)

    ax.set_xticks(rounds)
    ax.set_xticklabels([f"R{round_id}" for round_id in range(1, 6)])
    ax.set_title("偽標籤錯誤率", pad=14)
    ax.set_xlabel("自訓練輪次")
    ax.set_ylabel("文檔的偽標籤錯誤比例(%)")
    ax.set_ylim(0, max(max(values) for values in rates.values()) + 15)
    grid_alpha = 0.95 if background == "gray" else 0.35
    grid_width = 1.1 if background == "gray" else 1.0
    ax.grid(
        True,
        axis="y",
        linestyle="--",
        linewidth=grid_width,
        alpha=grid_alpha,
        color="black",
        zorder=10,
    )
    ax.set_axisbelow(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=True)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> int:
    args = parse_args()
    methods = (args.method_a_label, args.method_b_label)
    counts = collect_method_counts(methods[0], args.method_a_summary, args.method_a_root)
    counts += collect_method_counts(methods[1], args.method_b_summary, args.method_b_root)
    pooled = pool_by_method_and_round(counts, methods)

    write_csv_files(counts, pooled, args.analysis_dir)
    output_path = args.output or PNG_DIR / (
        f"confidence_vs_nest_round_pooled_full_ecp_document_error_{args.background}.png"
    )
    plot_error_rates(pooled, output_path, args.background, methods)

    for round_id in range(1, 6):
        values = [row for row in pooled if row.round_id == round_id]
        summary = ", ".join(
            f"{row.method}={row.error_documents}/{row.selected_documents} "
            f"({row.error_rate_percent:.2f}%)"
            for row in values
        )
        print(f"R{round_id}: {summary}")
    print(f"統計輸出：{args.analysis_dir}")
    print(f"圖檔輸出：{output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
