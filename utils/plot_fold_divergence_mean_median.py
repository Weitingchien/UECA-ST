#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""繪製 Fold 1～10 的 A1/A2 NeST 散度與樣本數圖。"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from label_utils import configure_fonts


REPO_ROOT = Path(__file__).resolve().parent.parent
REFERENCE_BACKGROUND_COLOR = "#DDDDDD"
WHITE_BACKGROUND_COLOR = "#FFFFFF"
A1_COLOR = "#0072B2"
A2_COLOR = "#D55E00"
GROUPS = ("A1_same", "A2_different")
EXPECTED_FOLDS = tuple(range(1, 11))
REQUIRED_COLUMNS = {
    "fold",
    "group",
    "n",
    "raw_mean",
    "raw_median",
    "upper_5pct_removed_n",
    "upper_5pct_remaining_n",
    "raw_mean_without_upper_5pct",
}


def format_power_of_ten(value: float, _position: float) -> str:
    """使用 STIX mathtext 顯示 log 軸刻度，避免中文字型缺少數學負號。"""
    exponent = int(round(math.log10(value)))
    return rf"$10^{{{exponent}}}$"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="繪製 Fold 1～10 的 A1/A2 raw divergence 平均數，可選擇同時顯示中位數。"
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=(
            REPO_ROOT
            / "analysis"
            / "final_model_emotion_category_divergence"
            / "final_model_k3_fold_summary.csv"
        ),
        help="analyze_final_model_emotion_category_divergence.py 產生的 fold summary CSV。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="輸出圖片路徑；未指定時依是否使用 --mean-only 自動命名，副檔名也可以使用 .pdf。",
    )
    parser.add_argument(
        "--background",
        choices=("reference", "white"),
        default="reference",
        help="reference 使用附圖的 #DDDDDD；white 使用純白色。",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="自訂圖表標題；未指定時依是否使用 --mean-only 自動產生。",
    )
    parser.add_argument(
        "--mean-only",
        action="store_true",
        help="只顯示 A1/A2 平均數，不繪製中位數。",
    )
    parser.add_argument(
        "--raw-mean-bar-output",
        type=Path,
        default=None,
        help="若指定，會另外輸出保留全部樣本的平均散度長條圖。",
    )
    parser.add_argument(
        "--raw-mean-bar-title",
        default=None,
        help="自訂保留全部樣本的長條圖標題。",
    )
    parser.add_argument(
        "--upper-5pct-bar-output",
        type=Path,
        default=None,
        help="若指定，會在第一張圖後另外輸出移除最高 5% 後的平均散度長條圖。",
    )
    parser.add_argument(
        "--upper-5pct-bar-title",
        default=None,
        help="自訂第二張長條圖標題。",
    )
    parser.add_argument(
        "--upper-5pct-count-bar-output",
        type=Path,
        default=None,
        help="若指定，會另外輸出移除最高 5% 後的剩餘樣本數量長條圖。",
    )
    parser.add_argument(
        "--upper-5pct-count-bar-title",
        default=None,
        help="自訂剩餘樣本數量長條圖標題。",
    )
    return parser.parse_args()


def read_fold_summary(path: Path) -> dict[int, dict[str, dict[str, float]]]:
    """讀取每折 A1/A2 的原始與上尾 5% 移除後統計。"""
    rows: dict[int, dict[str, dict[str, float]]] = {}

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        missing_columns = REQUIRED_COLUMNS - set(reader.fieldnames or ())
        if missing_columns:
            raise ValueError(f"CSV 缺少欄位: {sorted(missing_columns)}")

        for row_number, row in enumerate(reader, start=2):
            group = row["group"]
            if group not in GROUPS:
                raise ValueError(f"第 {row_number} 列出現未知 group: {group}")

            fold = int(row["fold"])
            if group in rows.setdefault(fold, {}):
                raise ValueError(f"fold{fold} 的 {group} 重複出現")

            n = int(row["n"])
            raw_mean = float(row["raw_mean"])
            raw_median = float(row["raw_median"])
            upper_5pct_removed_n = int(row["upper_5pct_removed_n"])
            upper_5pct_remaining_n = int(row["upper_5pct_remaining_n"])
            raw_mean_without_upper_5pct = float(
                row["raw_mean_without_upper_5pct"]
            )
            if n <= 0:
                raise ValueError(f"fold{fold} 的 {group} n 必須大於 0")
            if not all(
                math.isfinite(value) and value > 0
                for value in (
                    raw_mean,
                    raw_median,
                    raw_mean_without_upper_5pct,
                )
            ):
                raise ValueError(
                    f"fold{fold} 的 {group} 散度統計必須是有限正數，才能使用 log 軸"
                )

            rows[fold][group] = {
                "n": n,
                "raw_mean": raw_mean,
                "raw_median": raw_median,
                "upper_5pct_removed_n": upper_5pct_removed_n,
                "upper_5pct_remaining_n": upper_5pct_remaining_n,
                "raw_mean_without_upper_5pct": raw_mean_without_upper_5pct,
            }

    if tuple(sorted(rows)) != EXPECTED_FOLDS:
        raise ValueError("CSV 必須完整包含 fold1～fold10")

    incomplete = [
        f"fold{fold}"
        for fold in EXPECTED_FOLDS
        if set(rows[fold]) != set(GROUPS)
    ]
    if incomplete:
        raise ValueError(f"以下 folds 未同時包含 A1_same 與 A2_different: {incomplete}")

    return rows


def apply_background_style(fig: plt.Figure, ax: plt.Axes, background: str) -> str:
    """套用附圖灰底或白底，並回傳實際背景色碼。"""
    background_color = (
        REFERENCE_BACKGROUND_COLOR
        if background == "reference"
        else WHITE_BACKGROUND_COLOR
    )
    grid_alpha = 0.95 if background == "reference" else 0.35

    fig.patch.set_facecolor(background_color)
    ax.set_facecolor(background_color)
    ax.set_axisbelow(True)
    ax.grid(
        True,
        axis="y",
        which="major",
        linestyle="--",
        linewidth=1.1,
        alpha=grid_alpha,
        color="#000000",
        zorder=1,
    )
    ax.grid(False, axis="x")
    return background_color


def plot_fold_summary(
    rows: dict[int, dict[str, dict[str, float]]],
    output_path: Path,
    background: str,
    title: str,
    show_median: bool,
) -> None:
    """建立單一座標軸的 A1/A2 平均數點圖，並視需要加入中位數。"""
    configure_fonts(font_size=16)
    plt.rcParams["mathtext.fontset"] = "stix"

    folds = list(EXPECTED_FOLDS)
    a1_x = [fold - 0.12 for fold in folds]
    a2_x = [fold + 0.12 for fold in folds]
    a1_mean = [rows[fold]["A1_same"]["raw_mean"] for fold in folds]
    a2_mean = [rows[fold]["A2_different"]["raw_mean"] for fold in folds]
    a1_median = (
        [rows[fold]["A1_same"]["raw_median"] for fold in folds]
        if show_median
        else []
    )
    a2_median = (
        [rows[fold]["A2_different"]["raw_median"] for fold in folds]
        if show_median
        else []
    )

    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=180)
    background_color = apply_background_style(fig, ax, background)

    point_style = {
        "s": 80,
        "edgecolors": "white",
        "linewidths": 1.2,
        "zorder": 3,
    }
    ax.scatter(a1_x, a1_mean, color=A1_COLOR, marker="D", label="A1 平均數", **point_style)
    if show_median:
        ax.scatter(a1_x, a1_median, color=A1_COLOR, marker="o", label="A1 中位數", **point_style)
    ax.scatter(a2_x, a2_mean, color=A2_COLOR, marker="D", label="A2 平均數", **point_style)
    if show_median:
        ax.scatter(a2_x, a2_median, color=A2_COLOR, marker="o", label="A2 中位數", **point_style)

    all_values = a1_mean + a2_mean + a1_median + a2_median
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(format_power_of_ten))
    ax.set_ylim(min(all_values) * 0.45, max(all_values) * 2.5)
    ax.set_xlim(0.5, 10.5)
    ax.set_xticks(folds, [f"{fold}" for fold in folds])
    ax.set_xlabel("Fold")
    ax.set_ylabel("散度 (對數刻度) ")
    fig.suptitle(title, y=0.98)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncol=4 if show_median else 2,
        frameon=True,
        facecolor=background_color,
        edgecolor="#333333",
        framealpha=1.0,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.90))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)


def plot_grouped_bar(
    rows: dict[int, dict[str, dict[str, float]]],
    output_path: Path,
    background: str,
    title: str,
    value_key: str,
    ylabel: str,
    log_scale: bool,
    show_integer_labels: bool = False,
) -> None:
    """依指定欄位另畫每折 A1/A2 分組長條圖。"""
    configure_fonts(font_size=16)
    plt.rcParams["mathtext.fontset"] = "stix"

    folds = list(EXPECTED_FOLDS)
    width = 0.36
    a1_values = [rows[fold]["A1_same"][value_key] for fold in folds]
    a2_values = [rows[fold]["A2_different"][value_key] for fold in folds]
    all_values = a1_values + a2_values
    lower_limit = min(all_values) * 0.45 if log_scale else 0

    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=180)
    background_color = apply_background_style(fig, ax, background)
    a1_bars = ax.bar(
        [fold - width / 2 for fold in folds],
        [value - lower_limit for value in a1_values],
        width,
        bottom=lower_limit,
        color=A1_COLOR,
        edgecolor="white",
        linewidth=1.0,
        label="A1",
        zorder=3,
    )
    a2_bars = ax.bar(
        [fold + width / 2 for fold in folds],
        [value - lower_limit for value in a2_values],
        width,
        bottom=lower_limit,
        color=A2_COLOR,
        edgecolor="white",
        linewidth=1.0,
        label="A2",
        zorder=3,
    )

    if log_scale:
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(FuncFormatter(format_power_of_ten))
        ax.set_ylim(lower_limit, max(all_values) * 2.5)
    else:
        ax.set_ylim(0, max(all_values) * 1.18)
    if show_integer_labels:
        ax.bar_label(a1_bars, labels=[str(value) for value in a1_values], padding=3)
        ax.bar_label(a2_bars, labels=[str(value) for value in a2_values], padding=3)
    ax.set_xlim(0.5, 10.5)
    ax.set_xticks(folds, [f"{fold}" for fold in folds])
    ax.set_xlabel("Fold")
    ax.set_ylabel(ylabel)
    fig.suptitle(title, y=0.98)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncol=2,
        frameon=True,
        facecolor=background_color,
        edgecolor="#333333",
        framealpha=1.0,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.90))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)


def main() -> int:
    args = parse_args()
    rows = read_fold_summary(args.summary_csv)
    show_median = not args.mean_only
    output_path = args.output or (
        REPO_ROOT
        / "png"
        / (
            "final_model_k3_fold_raw_divergence_mean_median.png"
            if show_median
            else "final_model_k3_fold_raw_divergence_mean.png"
        )
    )
    title = args.title or (
        "各折 A1 與 A2 的 NeST 散度平均數與中位數（k=3）"
        if show_median
        else "各折 A1 與 A2 的 NeST 散度平均數（k=3）"
    )
    plot_fold_summary(rows, output_path, args.background, title, show_median)
    if args.raw_mean_bar_output:
        raw_bar_title = args.raw_mean_bar_title or (
            "各折 A1 與 A2 未移除最高 5% 的 NeST 散度平均數（k=3）"
        )
        plot_grouped_bar(
            rows,
            args.raw_mean_bar_output,
            args.background,
            raw_bar_title,
            "raw_mean",
            "散度（對數刻度）",
            True,
        )
    if args.upper_5pct_bar_output:
        bar_title = args.upper_5pct_bar_title or (
            "各折 A1 與 A2 移除最高 5% 後的 NeST 散度平均數（k=3）"
        )
        plot_grouped_bar(
            rows,
            args.upper_5pct_bar_output,
            args.background,
            bar_title,
            "raw_mean_without_upper_5pct",
            "散度（對數刻度）",
            True,
        )
    if args.upper_5pct_count_bar_output:
        count_bar_title = args.upper_5pct_count_bar_title or (
            "各折 A1 與 A2 移除散度最高 5% 樣本後的未標註樣本數量（k=3）"
        )
        plot_grouped_bar(
            rows,
            args.upper_5pct_count_bar_output,
            args.background,
            count_bar_title,
            "upper_5pct_remaining_n",
            "剩餘未標註樣本數量（筆）",
            False,
            show_integer_labels=True,
        )

    actual_background = (
        REFERENCE_BACKGROUND_COLOR
        if args.background == "reference"
        else WHITE_BACKGROUND_COLOR
    )
    print(f"輸入資料: {args.summary_csv}")
    print(f"輸出圖片: {output_path}")
    if args.raw_mean_bar_output:
        print(f"輸出圖片: {args.raw_mean_bar_output}")
    if args.upper_5pct_bar_output:
        print(f"輸出圖片: {args.upper_5pct_bar_output}")
    if args.upper_5pct_count_bar_output:
        print(f"輸出圖片: {args.upper_5pct_count_bar_output}")
    print(f"顯示統計量: {'平均數＋中位數' if show_median else '平均數'}")
    print(f"背景顏色: {actual_background}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
