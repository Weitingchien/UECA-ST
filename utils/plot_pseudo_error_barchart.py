"""繪製 prompt_ECPE_few_shot_ST_* 實驗的偽標籤平均錯誤率長條圖。"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
from matplotlib import font_manager
import sys
from pathlib import Path as _Path
_base = _Path(__file__).resolve().parent.parent
if str(_base) not in sys.path:
    sys.path.insert(0, str(_base))
from utils.label_utils import _simplify_label, configure_fonts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="統計 pseudo_error_report.txt 並繪製長條圖")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("."),
        help="搜尋 prompt_ECPE_few_shot_ST_* 的起始目錄 (預設：當前目錄)",
    )
    parser.add_argument(
        "--pattern",
        default="prompt_ECPE_few_shot_ST_*",
        help="實驗資料夾的 glob 模式 (預設：prompt_ECPE_few_shot_ST_*)",
    )
    parser.add_argument(
        "--report-name",
        default="pseudo_error_report.txt",
        help="要讀取的報告檔名",
    )
    parser.add_argument(
        "--use-normalized",
        action="store_true",
        help="預設使用 actual 平均，若指定此參數則改用 normalized 平均",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("pseudo_error_bar.png"),
        help="輸出的 PNG 檔名",
    )
    parser.add_argument(
        "--bg-color",
        default="#DDDDDD",
        help="圖表背景顏色（hex），預設 #DDDDDD",
    )
    return parser.parse_args()


def collect_reports(args: argparse.Namespace) -> List[Tuple[str, float]]:
    pattern = re.compile(
        r"overall folds: avg_error\(normalized\)=(?P<norm>\d+\.\d+), "
        r"avg_error\(actual\)=(?P<actual>\d+\.\d+)"
    )
    results: List[Tuple[str, float]] = []
    for exp_dir in sorted(args.base_dir.glob(args.pattern)):
        if not exp_dir.is_dir():
            continue
        report_path = exp_dir / args.report_name
        if not report_path.exists():
            print(f"[警告] 找不到報告：{report_path}")
            continue
        match = pattern.search(report_path.read_text(encoding="utf-8"))
        if not match:
            print(f"[警告] {report_path} 缺少 overall folds 行")
            continue
        value = float(match.group("norm" if args.use_normalized else "actual"))
        results.append((_simplify_label(exp_dir.name), value))
    return results


def plot_bar(data: List[Tuple[str, float]], output_path: Path, use_normalized: bool, bg_color: str = "#DDDDDD") -> None:
    if not data:
        print("[資訊] 沒有可繪製的資料")
        return

    labels, values = zip(*data)
    # 控制圖寬：根據標籤數量決定寬度，但不要太大
    fig_height = max(6, len(labels) * 0.6)
    fig_width = max(8, min(12, len(labels) * 0.6))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    # 預設背景顏色（可由 CLI --bg-color 修改）
    # use provided bg_color argument
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)
    y_pos = range(len(values))
    bars = ax.barh(y_pos, values, color="#5B8FF9")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("平均錯誤率")
    ax.set_ylabel("實驗資料夾")
    ax.set_title("偽標籤平均錯誤率比較")
    # 讓右側留一點空間給數值標籤
    max_v = max(values)
    x_margin = max(0.03, max_v * 0.08)
    ax.set_xlim(0, max_v + x_margin)

    # 使用 renderer 測量最大標籤實際像素寬度，換算為 figure 的 left fraction
    # 以避免文字被裁切
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    fp = font_manager.FontProperties(size=plt.rcParams.get("font.size"))
    max_width_px = 0
    for lbl in labels:
        t = fig.text(0, 0, lbl, fontproperties=fp)
        bbox = t.get_window_extent(renderer=renderer)
        max_width_px = max(max_width_px, bbox.width)
        t.remove()
    # left fraction = text width (in inches) / figure width (in inches) + padding
    left_pad = (max_width_px / fig.dpi) / fig.get_figwidth() + 0.02
    # enforce reasonable range
    left_pad = min(max(left_pad, 0.12), 0.6)

    # 右對齊 yticklabels 以避免左側文字因對齊方式被遮擋
    ax.set_yticklabels(labels, ha="right")
    for bar in bars:
        width = bar.get_width()
        # 如果 bar 夠長，將數字放到 bar 內側（靠右）並改為白色文字
        if width >= 0.12:
            ax.text(
                width - x_margin * 0.4,
                bar.get_y() + bar.get_height() / 2,
                f"{width:.3f}",
                va="center",
                ha="right",
                color="black",
            )
        else:
            # 否則放到 bar 右邊（可見範圍內）
            ax.text(
                width + 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{width:.3f}",
                va="center",
                ha="left",
                color="black",
            )

    fig.subplots_adjust(left=left_pad, right=0.95, top=0.95, bottom=0.05)
    fig.savefig(output_path, dpi=200)
    print(f"[資訊] 已輸出長條圖：{output_path}")


def main() -> None:
    args = parse_args()
    configure_fonts()
    data = collect_reports(args)
    if not data:
        return
    plot_bar(data, args.output, args.use_normalized, args.bg_color)


if __name__ == "__main__":
    main()
