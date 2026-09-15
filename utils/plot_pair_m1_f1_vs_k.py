#!/usr/bin/env python3
# 這支腳本用來讀取多個 multi-seed summary 檔，並繪製 Pair (m1) F1 對 k 值的折線圖。

from __future__ import annotations

# 匯入 argparse 以解析命令列參數。
import argparse
import math
# 匯入 re 以便從檔名與摘要文字中抓取 k 值與 F1 數值。
import re
# 匯入 Path 以便處理跨平台檔案路徑。
from pathlib import Path

# 匯入 Matplotlib 用來繪圖。
import matplotlib.pyplot as plt

# 匯入既有的字型設定函式，讓新圖的字形與字級跟現有圖一致。
from label_utils import configure_fonts

MEAN_DELTA_MATCH_BG_COLOR = "#DDDDDD"
# 定義折線顏色，沿用現有 selected-all 長條圖的藍色。
LINE_COLOR = "#4C78A8"
# 定義誤差棒顏色，使用稍深的藍色以提升辨識度。
ERROR_BAR_COLOR = "#2F5F93"

# 這個正則用來從 summary 檔名中抓取 nest_k 的數值。
K_PATTERN = re.compile(r"_nest_k(?P<k>\d+)_")

# 這個正則用來從「10 折累計統計結果 (Mean ± Std)」區塊抓取 Pair (m1) 的 F1 Mean ± Std。
PAIR_M1_F1_PATTERN = re.compile(
    r"10 折累計統計結果 \(Mean ± Std\):\s*\n"
    r".*?Pair \(m1\):\s*\n"
    r"\s*P \(micro\):\s*[\d.]+%\s*±\s*[\d.]+%\s*\n"
    r"\s*R \(micro\):\s*[\d.]+%\s*±\s*[\d.]+%\s*\n"
    r"\s*F1 \(micro\):\s*(?P<f1>[\d.]+)%\s*±\s*(?P<std>[\d.]+)%",
    re.DOTALL,
)


# 定義單一 summary 檔解析後的資料結構，方便後續排序與繪圖。
class SummaryPoint:
    # 初始化一個資料點，包含 k 值、F1 平均值、F1 標準差與原始路徑。
    def __init__(self, k_value: int, f1_mean: float, f1_std: float, summary_path: Path) -> None:
        # 保存 k 值。
        self.k_value = k_value
        # 保存 F1 平均值。
        self.f1_mean = f1_mean
        # 保存 F1 標準差。
        self.f1_std = f1_std
        # 保存來源檔案路徑，方便錯誤追蹤或輸出資訊。
        self.summary_path = summary_path


# 解析命令列參數。
def parse_args() -> argparse.Namespace:
    # 建立參數解析器。
    parser = argparse.ArgumentParser(
        description="Plot Pair F1 curve against different NeST k values from summary files."
    )
    # 接收多個 summary 檔案路徑。
    parser.add_argument("--summaries", nargs="+", required=True, help="Summary txt files to plot")
    # 接收輸出圖檔路徑。
    parser.add_argument(
        "--output",
        default="png/pair_m1_f1_vs_k.png",
        help="Output image path for the Pair F1 curve",
    )
    # 接收圖表標題，提供使用者覆蓋預設值。
    parser.add_argument(
        "--title",
        default="Pair F1 與 k 值關係圖",
        help="Chart title",
    )
    # 提供可選參數，讓使用者決定是否顯示誤差棒。
    parser.add_argument(
        "--hide-error-bars",
        action="store_true",
        help="Hide standard-deviation error bars and only plot the line with markers",
    )
    parser.add_argument(
        "--match-mean-delta-background",
        action="store_true",
        help="Use the same gray background style as the selected round pooled grouped bar chart",
    )
    parser.add_argument("--y-min", type=float, default=None, help="Optional lower bound of y-axis")
    parser.add_argument("--y-max", type=float, default=None, help="Optional upper bound of y-axis")
    parser.add_argument(
        "--sparse-y-ticks",
        action="store_true",
        help="Only show the lowest, middle, and highest y-axis ticks",
    )
    # 回傳解析結果。
    return parser.parse_args()


# 從 summary 檔名中解析 k 值。
def parse_k_from_name(summary_path: Path) -> int:
    # 嘗試在檔名中尋找 _nest_kX_ 模式。
    match = K_PATTERN.search(summary_path.name)
    # 如果找不到，就拋出錯誤，提醒使用者檔名格式不符。
    if not match:
        raise ValueError(f"Cannot parse k value from file name: {summary_path}")
    # 回傳解析出的 k 值整數。
    return int(match.group("k"))


# 從 summary 文字內容中解析 Pair (m1) 的 F1 Mean ± Std。
def parse_pair_m1_f1(summary_path: Path) -> SummaryPoint:
    # 讀取整份 summary 文字內容。
    content = summary_path.read_text(encoding="utf-8", errors="ignore")
    # 在文字中尋找 Pair (m1) 的 F1 Mean ± Std。
    match = PAIR_M1_F1_PATTERN.search(content)
    # 如果找不到，就拋出錯誤，表示此 summary 格式和預期不一致。
    if not match:
        raise ValueError(f"Cannot parse Pair (m1) F1 from summary: {summary_path}")
    # 從檔名取得 k 值。
    k_value = parse_k_from_name(summary_path)
    # 從比對結果取出 F1 平均值。
    f1_mean = float(match.group("f1"))
    # 從比對結果取出 F1 標準差。
    f1_std = float(match.group("std"))
    # 將這些資訊整理成 SummaryPoint 回傳。
    return SummaryPoint(k_value=k_value, f1_mean=f1_mean, f1_std=f1_std, summary_path=summary_path)


# 將多個 summary 路徑整理成可繪圖的資料點，並依 k 值排序。
def load_points(summary_paths: list[Path]) -> list[SummaryPoint]:
    # 先逐一解析每個 summary 檔。
    points = [parse_pair_m1_f1(path) for path in summary_paths]
    # 依 k 值由小到大排序，讓 x 軸順序正確。
    points.sort(key=lambda point: point.k_value)
    # 回傳排序後的資料點。
    return points


# 將每個 k 值對應的 mean、std 與誤差棒上下界印到終端，方便和圖表對照。
def print_point_summary(points: list[SummaryPoint]) -> None:
    # 先印出標題，讓終端輸出更容易閱讀。
    print("[資訊] Pair F1 各 k 值摘要：")
    # 逐一列出每個 k 值的統計資訊。
    for point in points:
        # 上界 = mean + std。
        upper = point.f1_mean + point.f1_std
        # 下界 = mean - std。
        lower = point.f1_mean - point.f1_std
        # 印出目前 k 值的所有重點數字。
        print(
            f"  - k={point.k_value}: "
            f"mean={point.f1_mean:.2f}%, "
            f"std={point.f1_std:.2f}%, "
            f"lower={lower:.2f}%, "
            f"upper={upper:.2f}%"
        )


# 在每個資料點上方標示 F1 數值。
def annotate_points(ax: plt.Axes, x_values: list[int], y_values: list[float], offset: float) -> None:
    value_range = max(y_values) - min(y_values)
    # 逐一處理每個資料點。
    for x_value, y_value in zip(x_values, y_values):
        # 在點的上方放上兩位小數的文字標籤。
        ax.text(
            x_value,
            y_value + offset,
            f"{y_value:.2f}",
            ha="center",
            va="bottom",
            fontsize=12,
            color="black",
            zorder=6,
        )


def compute_y_axis_bounds_and_ticks(
    y_values: list[float],
    y_errors: list[float],
    hide_error_bars: bool,
) -> tuple[float, float, list[int]]:
    """依資料範圍計算較緊的 y 軸上下界與 0.2 間隔刻度。"""
    lower_values = y_values if hide_error_bars else [value - error for value, error in zip(y_values, y_errors)]
    upper_values = y_values if hide_error_bars else [value + error for value, error in zip(y_values, y_errors)]

    if hide_error_bars:
        lower_margin = 0.03
        upper_margin = 0.05
        tick_step = 0.1
    else:
        lower_margin = 0.06
        upper_margin = 0.08
        tick_step = 0.2

    lower_bound = min(lower_values) - lower_margin
    upper_bound = max(upper_values) + upper_margin

    tick_start = math.floor(lower_bound / tick_step) * tick_step
    tick_end = math.ceil(upper_bound / tick_step) * tick_step
    tick_count = int(round((tick_end - tick_start) / tick_step)) + 1
    ticks = [round(tick_start + index * tick_step, 1) for index in range(tick_count)]

    return lower_bound, upper_bound, ticks


def apply_manual_y_axis_bounds(
    default_y_min: float,
    default_y_max: float,
    requested_y_min: float | None,
    requested_y_max: float | None,
) -> tuple[float, float, list[float]]:
    """套用使用者指定的 y 軸上下界，並建立 0.1 間隔刻度。"""
    y_min = default_y_min if requested_y_min is None else requested_y_min
    y_max = default_y_max if requested_y_max is None else requested_y_max

    if y_min >= y_max:
        raise ValueError(f"Invalid y-axis bounds: y_min={y_min} must be smaller than y_max={y_max}")

    tick_step = 0.1
    tick_start = math.ceil(y_min / tick_step) * tick_step
    tick_end = math.floor(y_max / tick_step) * tick_step
    tick_count = int(round((tick_end - tick_start) / tick_step)) + 1
    ticks = [round(tick_start + index * tick_step, 1) for index in range(max(tick_count, 0))]

    if not ticks:
        ticks = [round(y_min, 2), round(y_max, 2)]

    return y_min, y_max, ticks


def compress_y_ticks(y_ticks: list[float]) -> list[float]:
    """將 y 軸刻度壓縮為最低、中間、最高三個刻度。"""
    if len(y_ticks) <= 3:
        return y_ticks

    middle_index = len(y_ticks) // 2
    compressed = [y_ticks[0], y_ticks[middle_index], y_ticks[-1]]
    deduped: list[float] = []
    for tick in compressed:
        if tick not in deduped:
            deduped.append(tick)
    return deduped


# 套用和既有圖一致的座標軸樣式。
def apply_mean_delta_background_style(fig: plt.Figure, ax: plt.Axes) -> None:
    # 套用和目標圖一致的灰色背景與較深虛線。
    fig.patch.set_facecolor(MEAN_DELTA_MATCH_BG_COLOR)
    ax.set_facecolor(MEAN_DELTA_MATCH_BG_COLOR)
    ax.set_axisbelow(False)
    ax.grid(True, axis="y", linestyle="--", linewidth=1.1, alpha=0.95, color="#000000", zorder=10)


# 套用和既有圖一致的座標軸樣式。
def style_axes(
    ax: plt.Axes,
    title: str,
    y_min: float,
    y_max: float,
    y_ticks: list[int],
    match_mean_delta_background: bool = False,
) -> None:
    # 設定繪圖區背景為白色。
    if not match_mean_delta_background:
        ax.set_facecolor("white")
    # 設定圖表標題。
    ax.set_title(title, pad=14)
    # 設定 x 軸標題。
    ax.set_xlabel("k")
    # 設定 y 軸標題。
    ax.set_ylabel("Pair F1 (%)")
    # 設定 y 軸範圍，讓圖面留有適當上下邊界。
    ax.set_ylim(y_min, y_max)
    ax.set_yticks(y_ticks)
    # 加上 y 軸虛線格線，風格和現有圖一致。
    if not match_mean_delta_background:
        ax.grid(True, axis="y", linestyle="--", color="black", alpha=0.35, linewidth=1.0, zorder=1)
    # 讓格線繪製在資料下方。
    ax.set_axisbelow(True)


# 根據解析結果繪製 Pair (m1) F1 對 k 值的折線圖。
def plot_curve(
    points: list[SummaryPoint],
    output_path: Path,
    title: str,
    hide_error_bars: bool,
    match_mean_delta_background: bool,
    requested_y_min: float | None,
    requested_y_max: float | None,
    sparse_y_ticks: bool,
) -> None:
    # 套用和現有 selected-all 圖相同的字型設定與字級。
    configure_fonts(font_size=16)
    # 萃取 x 軸的 k 值。
    x_values = [point.k_value for point in points]
    # 萃取 y 軸的 F1 平均值。
    y_values = [point.f1_mean for point in points]
    # 萃取誤差棒所需的標準差。
    y_errors = [point.f1_std for point in points]
    # 依資料範圍計算較緊的 y 軸區間，避免出現多餘空白。
    y_min, y_max, y_ticks = compute_y_axis_bounds_and_ticks(
        y_values=y_values,
        y_errors=y_errors,
        hide_error_bars=hide_error_bars,
    )
    y_min, y_max, y_ticks = apply_manual_y_axis_bounds(
        default_y_min=y_min,
        default_y_max=y_max,
        requested_y_min=requested_y_min,
        requested_y_max=requested_y_max,
    )
    if sparse_y_ticks:
        y_ticks = compress_y_ticks(y_ticks)
    # 建立圖與座標軸，尺寸與現有 bar 圖相同。
    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=180)
    # 設定整張圖背景為白色。
    fig.patch.set_facecolor("white")
    if match_mean_delta_background:
        apply_mean_delta_background_style(fig, ax)

    # 依照參數決定要畫帶誤差棒的折線圖，或只畫單純折線。
    if hide_error_bars:
        # 不顯示誤差棒時，只保留折線與圓點標記。
        ax.plot(
            x_values,
            y_values,
            "o-",
            color=LINE_COLOR,
            linewidth=2.2,
            markersize=7,
            markerfacecolor=LINE_COLOR,
            markeredgecolor="white",
            markeredgewidth=1.0,
            label="Pair F1",
            zorder=4,
        )
    else:
        # 顯示誤差棒時，使用標準差作為上下誤差範圍。
        ax.errorbar(
            x_values,
            y_values,
            yerr=y_errors,
            fmt="o-",
            color=LINE_COLOR,
            ecolor=ERROR_BAR_COLOR,
            elinewidth=1.6,
            capsize=5,
            capthick=1.6,
            linewidth=2.2,
            markersize=7,
            markerfacecolor=LINE_COLOR,
            markeredgecolor="white",
            markeredgewidth=1.0,
            label="Pair F1",
            zorder=4,
        )

    # 讓數值標籤更貼近資料點，並避免最高點文字擠出上界。
    value_range = max(y_values) - min(y_values)
    annotation_offset = max(0.015, value_range * 0.04)
    # 在每個點上方標上數值。
    annotate_points(ax, x_values, y_values, offset=annotation_offset)
    # 設定 x 軸刻度位置。
    ax.set_xticks(x_values)
    # 設定 x 軸刻度文字為對應的 k 值。
    ax.set_xticklabels([str(value) for value in x_values])
    style_axes(
        ax,
        title=title,
        y_min=y_min,
        y_max=y_max,
        y_ticks=y_ticks,
        match_mean_delta_background=match_mean_delta_background,
    )
    # 顯示圖例，位置與現有 selected-all 圖一致。
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=1, frameon=True)
    # 自動調整版面，避免標題與圖例重疊。
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    # 若輸出資料夾不存在就先建立。
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # 將圖輸出成檔案。
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    # 關閉圖形，避免批次執行時累積記憶體。
    plt.close(fig)


# 主程式入口。
def main() -> int:
    # 先解析命令列參數。
    args = parse_args()
    # 將 summary 參數轉成 Path 物件列表。
    summary_paths = [Path(path) for path in args.summaries]
    # 將輸出路徑轉成 Path 物件。
    output_path = Path(args.output)
    # 讀入並排序所有資料點。
    points = load_points(summary_paths)
    # 先把每個 k 值的 mean/std 與誤差棒上下界印到終端，方便使用者核對。
    print_point_summary(points)
    # 將是否隱藏誤差棒的設定傳入繪圖函式。
    plot_curve(
        points=points,
        output_path=output_path,
        title=args.title,
        hide_error_bars=args.hide_error_bars,
        match_mean_delta_background=args.match_mean_delta_background,
        requested_y_min=args.y_min,
        requested_y_max=args.y_max,
        sparse_y_ticks=args.sparse_y_ticks,
    )
    # 在終端輸出完成訊息與儲存路徑。
    print(f"[資訊] 已生成 Pair F1 曲線圖：{output_path}")
    # 正常結束。
    return 0


# 僅在直接執行此腳本時才呼叫主程式。
if __name__ == "__main__":
    # 將主程式回傳值轉成程序結束代碼。
    raise SystemExit(main())