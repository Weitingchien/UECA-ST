#!/usr/bin/env python3
"""Plot grouped bar chart of cross-seed exact-match rates from a report file."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from label_utils import configure_fonts

MEAN_DELTA_MATCH_BG_COLOR = "#DDDDDD"
GROUPED_LOW_COLOR = "#2E8B57"
GROUPED_MID_COLOR = "#D68C45"
GROUPED_HIGH_COLOR = "#C44E52"
SELECTED_ALL_COLOR = "#4C78A8"

ROUND_MEAN_PATTERN = re.compile(
    r"^round(?P<round>\d+):\s+all=(?P<all>[0-9.]+)%\s+low=(?P<low>[0-9.]+)%\s+mid=(?P<mid>[0-9.]+)%\s+high=(?P<high>[0-9.]+)%$"
)
ROUND_POOLED_PATTERN = re.compile(
    r"^round(?P<round>\d+):\s+"
    r"all=\d+/\d+\s+rate=(?P<all>[0-9.]+)%\s+"
    r"low=\d+/\d+\s+rate=(?P<low>[0-9.]+)%\s+"
    r"mid=\d+/\d+\s+rate=(?P<mid>[0-9.]+)%\s+"
    r"high=\d+/\d+\s+rate=(?P<high>[0-9.]+)%$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot grouped bar chart from cross-seed summary lines in report."
    )
    parser.add_argument("--report", required=True, help="Path to selected_divergence_statistics report txt")
    parser.add_argument("--output", required=True, help="Output image path for low/mid/high grouped bar chart")
    parser.add_argument("--all-output", required=True, help="Output image path for selected-all bar chart")
    parser.add_argument(
        "--summary-type",
        choices=("auto", "pooled", "mean"),
        default="pooled",
        help="Summary section to plot; 'auto' prefers pooled and falls back to mean",
    )
    parser.add_argument(
        "--title",
        default="被選中的未標註文檔在不同散度區間的完全匹配率",
        help="Grouped chart title",
    )
    parser.add_argument(
        "--all-title",
        default="被選中的未標註文檔完全匹配率",
        help="Selected-all chart title",
    )
    parser.add_argument(
        "--match-mean-delta-background",
        action="store_true",
        help="Use the same gray background style as mean_delta_NeST_a_emo_pair_vs_NeST_emo_pair.png",
    )
    return parser.parse_args()


def collect_round_rates(
    lines: list[str],
    pattern: re.Pattern[str],
) -> tuple[list[int], list[float], list[float], list[float], list[float]]:
    rounds: list[int] = []
    all_rates: list[float] = []
    low_rates: list[float] = []
    mid_rates: list[float] = []
    high_rates: list[float] = []

    for line in lines:
        match = pattern.match(line.strip())
        if not match:
            continue
        rounds.append(int(match.group("round")))
        all_rates.append(float(match.group("all")))
        low_rates.append(float(match.group("low")))
        mid_rates.append(float(match.group("mid")))
        high_rates.append(float(match.group("high")))

    return rounds, all_rates, low_rates, mid_rates, high_rates


def parse_round_rates(
    report_path: Path,
    summary_type: str,
) -> tuple[str, list[int], list[float], list[float], list[float], list[float]]:
    lines = report_path.read_text(encoding="utf-8", errors="ignore").splitlines()

    mean_data = collect_round_rates(lines, ROUND_MEAN_PATTERN)
    pooled_data = collect_round_rates(lines, ROUND_POOLED_PATTERN)

    if summary_type == "pooled":
        if not pooled_data[0]:
            raise ValueError("No pooled round summary lines found in report")
        return "pooled", *pooled_data

    if summary_type == "mean":
        if not mean_data[0]:
            raise ValueError("No mean round summary lines found in report")
        return "mean", *mean_data

    if pooled_data[0]:
        return "pooled", *pooled_data

    if mean_data[0]:
        return "mean", *mean_data

    raise ValueError("No supported round summary lines found in report")


def annotate_bars(ax: plt.Axes, bars, text_color: str = "black") -> None:
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.12,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=12,
            color=text_color,
            zorder=6,
        )


def build_round_labels(rounds: list[int]) -> list[str]:
    return [f"R{round_id}" for round_id in rounds]


def apply_mean_delta_background_style(fig: plt.Figure, ax: plt.Axes) -> None:
    fig.patch.set_facecolor(MEAN_DELTA_MATCH_BG_COLOR)
    ax.set_facecolor(MEAN_DELTA_MATCH_BG_COLOR)
    ax.set_axisbelow(False)
    ax.grid(True, axis="y", linestyle="--", linewidth=1.1, alpha=0.95, color="#000000", zorder=10)


def style_axes(ax: plt.Axes, title: str, y_max: float, match_mean_delta_background: bool = False) -> None:
    if not match_mean_delta_background:
        ax.set_facecolor("white")
    ax.set_title(title, pad=14)
    ax.set_xlabel("自訓練輪次")
    ax.set_ylabel("完全匹配率 (%)")
    ax.set_ylim(0, y_max)
    if not match_mean_delta_background:
        ax.grid(True, axis="y", linestyle="--", color="black", alpha=0.35, linewidth=1.0, zorder=5)
        ax.set_axisbelow(False)


def plot_grouped_bar(
    rounds: list[int],
    low_rates: list[float],
    mid_rates: list[float],
    high_rates: list[float],
    output_path: Path,
    title: str,
    match_mean_delta_background: bool = False,
) -> None:
    configure_fonts(font_size=16)

    x = np.arange(len(rounds), dtype=float)
    width = 0.24

    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=180)
    fig.patch.set_facecolor("white")
    if match_mean_delta_background:
        apply_mean_delta_background_style(fig, ax)

    low_bars = ax.bar(
        x - width,
        low_rates,
        width=width,
        label="低散度區間",
        color=GROUPED_LOW_COLOR,
        edgecolor="white",
        linewidth=1.2,
        zorder=2,
    )
    mid_bars = ax.bar(
        x,
        mid_rates,
        width=width,
        label="中散度區間",
        color=GROUPED_MID_COLOR,
        edgecolor="white",
        linewidth=1.2,
        zorder=2,
    )
    high_bars = ax.bar(
        x + width,
        high_rates,
        width=width,
        label="高散度區間",
        color=GROUPED_HIGH_COLOR,
        edgecolor="white",
        linewidth=1.2,
        zorder=2,
    )

    annotate_bars(ax, low_bars)
    annotate_bars(ax, mid_bars)
    annotate_bars(ax, high_bars)

    ax.set_xticks(x)
    ax.set_xticklabels(build_round_labels(rounds))
    style_axes(
        ax,
        title=title,
        y_max=max(max(low_rates), max(mid_rates), max(high_rates)) + 8,
        match_mean_delta_background=match_mean_delta_background,
    )
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        frameon=True,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_selected_all_bar(
    rounds: list[int],
    all_rates: list[float],
    output_path: Path,
    title: str,
    match_mean_delta_background: bool = False,
) -> None:
    configure_fonts(font_size=16)

    labels = build_round_labels(rounds)
    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=180)
    fig.patch.set_facecolor("white")
    if match_mean_delta_background:
        apply_mean_delta_background_style(fig, ax)

    bars = ax.bar(
        labels,
        all_rates,
        color=SELECTED_ALL_COLOR,
        edgecolor="white",
        linewidth=1.2,
        zorder=2,
        label="被選中的未標註文檔(不另外分散度區間)",
    )

    annotate_bars(ax, bars)
    style_axes(
        ax,
        title=title,
        y_max=max(all_rates) + 8,
        match_mean_delta_background=match_mean_delta_background,
    )
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=1,
        frameon=True,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> int:
    args = parse_args()
    report_path = Path(args.report)
    output_path = Path(args.output)
    all_output_path = Path(args.all_output)

    summary_type, rounds, all_rates, low_rates, mid_rates, high_rates = parse_round_rates(
        report_path,
        args.summary_type,
    )
    plot_grouped_bar(
        rounds=rounds,
        low_rates=low_rates,
        mid_rates=mid_rates,
        high_rates=high_rates,
        output_path=output_path,
        title=args.title,
        match_mean_delta_background=args.match_mean_delta_background,
    )
    plot_selected_all_bar(
        rounds=rounds,
        all_rates=all_rates,
        output_path=all_output_path,
        title=args.all_title,
        match_mean_delta_background=args.match_mean_delta_background,
    )

    print(f"Summary type used: {summary_type}")
    print(f"Figure written to: {output_path}")
    print(f"Figure written to: {all_output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
