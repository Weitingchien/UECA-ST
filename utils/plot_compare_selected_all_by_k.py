#!/usr/bin/env python3
"""比較兩個 k 設定在各輪 selected-all pooled 完全匹配率的長條圖"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from label_utils import configure_fonts

MEAN_DELTA_MATCH_BG_COLOR = "#DDDDDD"
COLOR_A = "#4C78A8"
COLOR_B = "#D68C45"
K_PATTERN = re.compile(r"_nest_k(?P<k>\d+)_")
ROUND_SPAN_PATTERN = re.compile(r"_round(?P<start>\d+)_(?P<end>\d+)_")
ROUND_POOLED_PATTERN = re.compile(
    r"^round(?P<round>\d+):\s+"
    r"all=\d+/\d+\s+rate=(?P<all>[0-9.]+)%\s+"
    r"low=\d+/\d+\s+rate=(?P<low>[0-9.]+)%\s+"
    r"mid=\d+/\d+\s+rate=(?P<mid>[0-9.]+)%\s+"
    r"high=\d+/\d+\s+rate=(?P<high>[0-9.]+)%$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two selected-all pooled exact-match charts for different k values."
    )
    parser.add_argument("--source-a", required=True, help="First source path: selected-all png or report txt")
    parser.add_argument("--source-b", required=True, help="Second source path: selected-all png or report txt")
    parser.add_argument("--output", required=True, help="Output image path")
    parser.add_argument(
        "--title",
        default="不同 k 值下被選中的未標註文檔情緒原因組合提取完全匹配率",
        help="Chart title",
    )
    parser.add_argument("--label-a", default=None, help="Legend label for source A; default infers k value")
    parser.add_argument("--label-b", default=None, help="Legend label for source B; default infers k value")
    parser.add_argument(
        "--match-mean-delta-background",
        action="store_true",
        help="Use the same gray background style as the selected round pooled grouped bar chart",
    )
    return parser.parse_args()


def infer_k_label(path: Path) -> str:
    match = K_PATTERN.search(path.name)
    if not match:
        return path.stem
    return f"k={match.group('k')}"


def extract_round_span(path: Path) -> tuple[int, int]:
    match = ROUND_SPAN_PATTERN.search(path.name)
    if not match:
        return (0, 0)
    return int(match.group("start")), int(match.group("end"))


def resolve_report_path(source_path: Path) -> Path:
    if source_path.suffix.lower() == ".txt":
        if not source_path.is_file():
            raise FileNotFoundError(f"Report file not found: {source_path}")
        return source_path

    if source_path.suffix.lower() != ".png":
        raise ValueError(f"Unsupported source type: {source_path}")

    png_match = re.match(
        r"^(?P<prefix>.+)_round_pooled_selected_all_bar_(?P<q>q\d+)(?P<suffix>_.+)?\.png$",
        source_path.name,
    )
    if not png_match:
        raise ValueError(f"Cannot derive report txt from png name: {source_path}")

    prefix = png_match.group("prefix")
    q_label = png_match.group("q")
    suffix = png_match.group("suffix") or ""
    candidates = sorted(
        source_path.parent.glob(f"{prefix}_selected_divergence_statistics_round*_{q_label}{suffix}.txt")
    )
    if not candidates:
        raise FileNotFoundError(f"No matching report txt found for png: {source_path}")

    best_candidate = max(candidates, key=lambda path: extract_round_span(path)[1] - extract_round_span(path)[0])
    return best_candidate


def parse_selected_all_rates(report_path: Path) -> tuple[list[int], list[float]]:
    rounds: list[int] = []
    all_rates: list[float] = []
    lines = report_path.read_text(encoding="utf-8", errors="ignore").splitlines()

    for line in lines:
        match = ROUND_POOLED_PATTERN.match(line.strip())
        if not match:
            continue
        rounds.append(int(match.group("round")))
        all_rates.append(float(match.group("all")))

    if not rounds:
        raise ValueError(f"No pooled selected-all round summary found in report: {report_path}")
    return rounds, all_rates


def annotate_bars(ax: plt.Axes, bars) -> None:
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


def build_round_labels(rounds: list[int]) -> list[str]:
    return [f"R{round_id}" for round_id in rounds]


def apply_mean_delta_background_style(fig: plt.Figure, ax: plt.Axes) -> None:
    fig.patch.set_facecolor(MEAN_DELTA_MATCH_BG_COLOR)
    ax.set_facecolor(MEAN_DELTA_MATCH_BG_COLOR)
    ax.set_axisbelow(False)


def style_axes(ax: plt.Axes, title: str, y_max: float, match_mean_delta_background: bool = False) -> None:
    if not match_mean_delta_background:
        ax.set_facecolor("white")
    ax.set_title(title, pad=16)
    ax.set_xlabel("自訓練輪次")
    ax.set_ylabel("完全匹配率 (%)")
    ax.set_ylim(0, y_max)
    ax.set_axisbelow(False)

    # 先關閉預設 grid，改用手動畫線，確保虛線一定覆蓋在長條圖上層。
    ax.grid(False)
    # 逐一對目前的 y 軸主要刻度畫出黑色虛線。
    for tick in ax.get_yticks():
        if tick <= 0 or tick >= y_max:
            continue
        ax.axhline(
            y=tick,
            linestyle="--",
            color="black",
            alpha=0.95 if match_mean_delta_background else 0.45,
            linewidth=1.1,
            zorder=8,
        )


def plot_comparison(
    rounds: list[int],
    values_a: list[float],
    values_b: list[float],
    label_a: str,
    label_b: str,
    output_path: Path,
    title: str,
    match_mean_delta_background: bool,
) -> None:
    configure_fonts(font_size=16)

    x = np.arange(len(rounds), dtype=float) * 1.35
    width = 0.46

    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=180)
    fig.patch.set_facecolor("white")
    if match_mean_delta_background:
        apply_mean_delta_background_style(fig, ax)

    bars_a = ax.bar(
        x - width / 2,
        values_a,
        width=width,
        color=COLOR_A,
        edgecolor="white",
        linewidth=1.2,
        label=label_a,
        zorder=3,
    )
    bars_b = ax.bar(
        x + width / 2,
        values_b,
        width=width,
        color=COLOR_B,
        edgecolor="white",
        linewidth=1.2,
        label=label_b,
        zorder=3,
    )

    annotate_bars(ax, bars_a)
    annotate_bars(ax, bars_b)

    ax.set_xticks(x)
    ax.set_xticklabels(build_round_labels(rounds))
    style_axes(
        ax,
        title=title,
        y_max=max(max(values_a), max(values_b)) + 8,
        match_mean_delta_background=match_mean_delta_background,
    )
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
        frameon=True,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> int:
    args = parse_args()
    source_a = Path(args.source_a)
    source_b = Path(args.source_b)
    output_path = Path(args.output)

    report_a = resolve_report_path(source_a)
    report_b = resolve_report_path(source_b)
    rounds_a, values_a = parse_selected_all_rates(report_a)
    rounds_b, values_b = parse_selected_all_rates(report_b)

    if rounds_a != rounds_b:
        raise ValueError(f"Round labels do not match: {rounds_a} vs {rounds_b}")

    label_a = args.label_a or infer_k_label(source_a)
    label_b = args.label_b or infer_k_label(source_b)

    plot_comparison(
        rounds=rounds_a,
        values_a=values_a,
        values_b=values_b,
        label_a=label_a,
        label_b=label_b,
        output_path=output_path,
        title=args.title,
        match_mean_delta_background=args.match_mean_delta_background,
    )

    print(f"Source A report: {report_a}")
    print(f"Source B report: {report_b}")
    print(f"Figure written to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())