"""繪製各實驗 (prompt_ECPE_few_shot_ST_*) 的 Pair (m1) micro 指標（Precision/Recall/F1）。

輸入：目錄模式、報表檔名、背景顏色、是否排序等。
輸出：一張單獨的長條圖 (grouped horizontal bars)。
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager

# reuse the simplification and font config from the other script
import sys
from pathlib import Path as _Path
_base = _Path(__file__).resolve().parent.parent
if str(_base) not in sys.path:
    sys.path.insert(0, str(_base))
from utils.label_utils import _simplify_label, configure_fonts


def _compute_left_padding(fig: plt.Figure, labels: List[str], font_size: int) -> float:
    """計算 y 標籤所需的左側 padding，模仿 pseudo_error 圖的測量方式。"""
    if not labels:
        return 0.12

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    fp = font_manager.FontProperties(size=font_size)

    max_width_px = 0
    temp_texts = []
    for lbl in labels:
        t = fig.text(0, 0, lbl, fontproperties=fp)
        temp_texts.append(t)
        bbox = t.get_window_extent(renderer=renderer)
        max_width_px = max(max_width_px, bbox.width)
    for t in temp_texts:
        t.remove()

    left_pad = (max_width_px / fig.dpi) / fig.get_figwidth() + 0.02
    return min(max(left_pad, 0.12), 0.6)


def _calc_vertical_pads(label_fontsize: int) -> Tuple[float, float]:
    """根據字體大小動態決定 top/bottom padding 並盡量保持上下間距對稱。"""
    margin = min(0.32, max(0.12, 0.08 + label_fontsize / 250.0))
    top = 1.0 - margin
    bottom = margin

    axis_height = top - bottom
    min_height = 0.42
    if axis_height < min_height:
        deficit = (min_height - axis_height) / 2.0
        bottom = max(0.05, bottom - deficit)
        top = min(0.95, top + deficit)

    return top, bottom


def _calc_fig_width(num_labels: int) -> float:
    """模仿 pseudo_error 圖的寬度策略，讓多張圖保持一致感。"""
    return max(8.0, min(12.0, num_labels * 0.6))


def _calc_x_limit(max_value: float) -> float:
    """提供額外 headroom，避免最大值貼齊右緣被裁切。"""
    margin = 0.02
    upper_cap = 0.6
    base = 0.42
    limit = max(base, max_value + margin)
    return min(limit, upper_cap)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="統計 counts_summary_detailed.txt 中 Pair(m1) 並繪圖")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("."),
        help="搜尋 prompt_ECPE_few_shot_ST_* 的起始目錄 (預設：當前目錄)",
    )
    parser.add_argument(
        "--pattern",
        default="prompt_ECPE_few_shot_ST_*",
        help="實驗資料夾的 glob 模式 (預設: prompt_ECPE_few_shot_ST_*)",
    )
    parser.add_argument(
        "--report-name",
        default="counts_summary_detailed.txt",
        help="要讀取的詳細統計檔名",
    )
    parser.add_argument(
        "--bg-color",
        default="#DDDDDD",
        help="圖表背景顏色 (hex)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("pseudo_pair_m1_metrics.png"),
        help="輸出的 PNG 檔名",
    )
    parser.add_argument(
        "--sort",
        action="store_true",
        help="是否依 F1 值做排序 (由高到低)",
    )
    parser.add_argument(
        "--grouped",
        action="store_true",
        help="輸出群組 P/R/F1 長條圖（若不指定則預設僅輸出 F1）",
    )
    parser.add_argument(
        "--only-f1",
        action="store_true",
        help="(相容參數) 僅輸出 F1 長條圖（不顯示 P/R）；與 --grouped 互斥，預設為僅 F1",
    )
    return parser.parse_args()


def parse_counts_summary(report_path: Path) -> Optional[Tuple[float, float, float]]:
    """從 counts_summary_detailed.txt 讀取 Pair (m1) 的 P/R/F1 (micro 百分比)。

    回傳 (precision, recall, f1) 作為 0..1 浮點數。
    若找不到則回傳 None 波
    """
    text = report_path.read_text(encoding="utf-8")
    # 找到 '10 折累計統計結果：' 之後的 Pair (m1) 區段
    # 匹配整行: Pair (m1): ... P (micro): 41.53% | R (micro): 38.20% | F1 (micro): 39.79%
    m = re.search(r"Pair \(m1\):[\s\S]*?P \(micro\):\s*(?P<P>[0-9]+\.?[0-9]*)%\s*\|\s*R \(micro\):\s*(?P<R>[0-9]+\.?[0-9]*)%\s*\|\s*F1 \(micro\):\s*(?P<F>[0-9]+\.?[0-9]*)%", text)
    if not m:
        return None
    p = float(m.group("P")) / 100.0
    r = float(m.group("R")) / 100.0
    f1 = float(m.group("F")) / 100.0
    return p, r, f1


def collect_pair_metrics(args: argparse.Namespace) -> List[Tuple[str, float, float, float]]:
    results = []
    for exp_dir in sorted(args.base_dir.glob(args.pattern)):
        if not exp_dir.is_dir():
            continue
        report_path = exp_dir / args.report_name
        if not report_path.exists():
            print(f"[警告] 找不到報告：{report_path}")
            continue
        parsed = parse_counts_summary(report_path)
        if not parsed:
            print(f"[警告] 無法從 {report_path} 解析 Pair (m1) 指標")
            continue
        p, r, f1 = parsed
        label = _simplify_label(exp_dir.name)
        results.append((label, p, r, f1))
    return results


def plot_grouped_bars(
    data: List[Tuple[str, float, float, float]], output_path: Path, bg_color: str
) -> None:
    if not data:
        print("[資訊] 沒有任何資料可繪製")
        return

    labels = [d[0] for d in data]
    p_vals = [d[1] for d in data]
    r_vals = [d[2] for d in data]
    f_vals = [d[3] for d in data]

    n = len(labels)
    # 動態計算 y 標籤字型大小：n 越多字越小 (但不低於 8)
    label_fontsize = min(24, max(12, int(260 / max(1, n))))
    # 根據字型大小動態計算圖高
    fig_height = max(6.0, n * 0.6)
    fig_width = _calc_fig_width(n)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)

    y = np.arange(n)
    # bar height tuned to look similar to pseudo_error chart spacing
    bar_height = min(0.3, max(0.16, 0.36 * (label_fontsize / 24)))
    offsets = [-bar_height, 0, bar_height]

    colors = ["#5B8FF9", "#5AD8A6", "#FF6B6B"]

    ax.barh(y + offsets[0], p_vals, height=bar_height, color=colors[0], label="Precision (micro)")
    ax.barh(y + offsets[1], r_vals, height=bar_height, color=colors[1], label="Recall (micro)")
    ax.barh(y + offsets[2], f_vals, height=bar_height, color=colors[2], label="F1 (micro)")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, ha="right", fontsize=label_fontsize)
    x_labelsize = label_fontsize
    ax.set_xlabel("Score", fontsize=x_labelsize)
    ax.set_title("Pair (m1) metrics 比較", pad=max(10, label_fontsize * 0.5))
    max_val = max(max(p_vals, default=0), max(r_vals, default=0), max(f_vals, default=0))
    x_limit = _calc_x_limit(max_val)
    ax.set_xlim(0, x_limit)
    ax.tick_params(axis="x", labelsize=x_labelsize)
    handles, leg_labels = ax.get_legend_handles_labels()

    # label inside/outside similar to previous behavior
    safe_right = x_limit - 0.005
    for i in range(n):
        for j, vals in enumerate((p_vals, r_vals, f_vals)):
            val = vals[i]
            xpos = val
            ypos = y[i] + offsets[j]
            if val > 0.15:
                text_x = max(0.0, xpos - 0.02)
                ax.text(text_x, ypos, f"{val:.3f}", va="center", ha="right", color="black", fontsize=max(8, int(label_fontsize * 0.9)))
            else:
                text_x = min(safe_right, xpos + 0.01)
                ax.text(text_x, ypos, f"{val:.3f}", va="center", ha="left", color="black", fontsize=max(8, int(label_fontsize * 0.9)))

    left_pad = _compute_left_padding(fig, labels, label_fontsize)
    top_pad, bottom_pad = _calc_vertical_pads(label_fontsize)
    legend_reserved = 0.22
    axis_bottom = max(bottom_pad, legend_reserved)
    fig.subplots_adjust(left=left_pad, right=0.95, top=top_pad, bottom=axis_bottom)

    legend_y = max(0.04, axis_bottom - 0.12)
    if handles:
        fig.legend(
            handles,
            leg_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, legend_y),
            ncol=3,
            frameon=False,
            columnspacing=1.4,
            handletextpad=0.6,
        )

    fig.savefig(output_path, dpi=200)
    print(f"[資訊] 已生成 Pair (m1) 指標圖表：{output_path}")


def plot_f1_bars(data: List[Tuple[str, float, float, float]], output_path: Path, bg_color: str) -> None:
    labels = [d[0] for d in data]
    f_vals = [d[3] for d in data]
    n = len(labels)
    label_fontsize = min(24, max(12, int(260 / max(1, n))))
    fig_height = max(6.0, n * 0.6)
    fig_width = _calc_fig_width(n)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)

    y = np.arange(n)
    bar_height = min(0.8, max(0.4, 0.7 * (label_fontsize / 24)))
    bars = ax.barh(y, f_vals, height=bar_height, color="#5B8FF9")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, ha="right", fontsize=label_fontsize)
    x_labelsize = label_fontsize
    ax.set_xlabel("F1", fontsize=x_labelsize)
    ax.set_title("Pair (m1)", pad=max(10, label_fontsize * 0.5))
    max_val = max(f_vals, default=0)
    x_limit = _calc_x_limit(max_val)
    ax.set_xlim(0, x_limit)
    ax.tick_params(axis="x", labelsize=x_labelsize)

    limit = x_limit
    inner_threshold = limit - 0.08  # switch to inside labels when close to edge
    for i, bar in enumerate(bars):
        val = f_vals[i]
        xpos = val
        ypos = bar.get_y() + bar.get_height() / 2
        if val >= inner_threshold:
            text_x = min(limit - 0.01, max(0.02, val - 0.01))
            ax.text(text_x, ypos, f"{val:.3f}", va="center", ha="right", color="black", fontsize=max(8, int(label_fontsize * 0.9)))
        elif val >= 0.15:
            ax.text(xpos - 0.01, ypos, f"{val:.3f}", va="center", ha="right", color="black", fontsize=max(8, int(label_fontsize * 0.9)))
        else:
            text_x = min(limit - 0.005, xpos + 0.01)
            ax.text(text_x, ypos, f"{val:.3f}", va="center", ha="left", color="black", fontsize=max(8, int(label_fontsize * 0.9)))

    left_pad = _compute_left_padding(fig, labels, label_fontsize)
    top_pad, bottom_pad = _calc_vertical_pads(label_fontsize)
    fig.subplots_adjust(left=left_pad, right=0.95, top=top_pad, bottom=bottom_pad)

    fig.savefig(output_path, dpi=200)
    print(f"[資訊] 已生成 Pair (m1) F1 圖表：{output_path}")


def main() -> None:
    args = parse_args()
    configure_fonts()
    data = collect_pair_metrics(args)
    if not data:
        return
    if args.sort:
        # sort by f1 descending
        data = sorted(data, key=lambda x: x[3], reverse=True)
    # Default behavior: only F1 unless --grouped specified
    if args.grouped:
        plot_grouped_bars(data, args.output, args.bg_color)
    else:
        plot_f1_bars(data, args.output, args.bg_color)


if __name__ == "__main__":
    main()
