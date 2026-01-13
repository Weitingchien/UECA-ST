"""Combine pseudo error averages and Pair (m1) F1 scores into a single grouped chart."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

import sys
from pathlib import Path as _Path
_base = _Path(__file__).resolve().parent.parent
if str(_base) not in sys.path:
    sys.path.insert(0, str(_base))
from utils.label_utils import _simplify_label, configure_fonts


def _compute_left_padding(fig: plt.Figure, labels: List[str], font_size: int) -> float:
    """Measure label widths (same as pseudo error plot) to determine padding."""
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
    """Determine top/bottom margins so titles and x labels are never clipped."""
    margin = min(0.32, max(0.1, 0.08 + label_fontsize / 260.0))
    top = 1.0 - margin
    bottom = margin

    min_band = 0.45
    band = top - bottom
    if band < min_band:
        deficit = (min_band - band) / 2.0
        bottom = max(0.05, bottom - deficit)
        top = min(0.95, top + deficit)

    return top, bottom


def _apply_alias(label: str, alias_map: Dict[str, str]) -> str:
    return alias_map.get(label, label)


def _parse_aliases(alias_str: str) -> Dict[str, str]:
    aliases: Dict[str, str] = {}
    if not alias_str:
        return aliases
    for pair in alias_str.split(","):
        if "=" not in pair:
            continue
        src, dst = pair.split("=", 1)
        src = src.strip()
        dst = dst.strip()
        if src and dst:
            aliases[src] = dst
    return aliases


def _parse_pseudo_error(report_path: Path) -> Optional[Tuple[float, float]]:
    pattern = re.compile(
        r"overall folds: avg_error\(normalized\)=(?P<norm>\d+\.\d+), avg_error\(actual\)=(?P<actual>\d+\.\d+)"
    )
    match = pattern.search(report_path.read_text(encoding="utf-8"))
    if not match:
        return None
    return float(match.group("norm")), float(match.group("actual"))


def _parse_f1(report_path: Path) -> Optional[float]:
    text = report_path.read_text(encoding="utf-8")
    m = re.search(
        r"Pair \(m1\):[\s\S]*?F1 \(micro\):\s*(?P<F>[0-9]+\.?[0-9]*)%",
        text,
    )
    if not m:
        return None
    return float(m.group("F")) / 100.0


def collect_combined(
    base_dir: Path,
    pattern: str,
    pseudo_report: str,
    counts_report: str,
    use_normalized: bool,
    alias_map: Dict[str, str],
) -> List[Tuple[str, Optional[float], float]]:
    combined = []
    for exp_dir in sorted(base_dir.glob(pattern)):
        if not exp_dir.is_dir():
            continue
        pseudo_path = exp_dir / pseudo_report
        counts_path = exp_dir / counts_report
        if not pseudo_path.exists() or not counts_path.exists():
            continue
        pseudo_vals = _parse_pseudo_error(pseudo_path)
        f1_val = _parse_f1(counts_path)
        if not pseudo_vals or f1_val is None:
            continue
        label = _simplify_label(exp_dir.name)
        label = _apply_alias(label, alias_map)
        error_val = pseudo_vals[0] if use_normalized else pseudo_vals[1]
        combined.append((label, error_val, f1_val))
    return combined


def select_subset(
    data: List[Tuple[str, Optional[float], float]],
    top_k: int,
    bottom_k: int,
    extra_labels: List[str],
) -> List[Tuple[str, Optional[float], float]]:
    if not data:
        return []

    seen = set()
    subset: List[Tuple[str, Optional[float], float]] = []
    label_map: Dict[str, Tuple[str, Optional[float], float]] = {item[0]: item for item in data}

    def _add(item: Tuple[str, Optional[float], float]) -> None:
        lbl = item[0]
        if lbl not in seen:
            subset.append(item)
            seen.add(lbl)

    ordered = sorted(data, key=lambda x: x[2], reverse=True)
    if top_k > 0:
        for item in ordered[:top_k]:
            _add(item)

    if bottom_k > 0:
        bottom_ordered = sorted(data, key=lambda x: x[2])
        for item in bottom_ordered[:bottom_k]:
            _add(item)

    for label in extra_labels:
        entry = label_map.get(label)
        if entry:
            _add(entry)
        else:
            print(f"[警告] subset_extra label '{label}' 不存在於資料集中，略過")

    return subset


def collect_extra_counts(
    dirs: List[Path],
    counts_report: str,
    alias_map: Dict[str, str],
) -> List[Tuple[str, Optional[float], float]]:
    extras: List[Tuple[str, Optional[float], float]] = []
    for dir_path in dirs:
        report_path = dir_path / counts_report
        if not report_path.exists():
            print(f"[警告] subset_extra 資料夾缺少報告：{report_path}")
            continue
        f1_val = _parse_f1(report_path)
        if f1_val is None:
            print(f"[警告] subset_extra 無法解析 Pair (m1)：{report_path}")
            continue
        label = _simplify_label(dir_path.name)
        label = _apply_alias(label, alias_map)
        extras.append((label, None, f1_val))
    return extras


def plot_combined(
    data: List[Tuple[str, Optional[float], float]],
    output_path: Path,
    use_normalized: bool,
    bg_color: str,
    x_max: float,
    group_gap: float,
    bar_gap: float,
    title_loc: str,
    title_x: Optional[float],
    legend_anchor_x: float,
    legend_anchor_y: float,
) -> None:
    if not data:
        print("[資訊] 沒有任何可繪製的資料")
        return

    labels = [d[0] for d in data]
    err_vals = [d[1] for d in data]
    f1_vals = [d[2] for d in data]

    n = len(labels)
    label_fontsize = min(24, max(12, int(260 / max(1, n))))
    fig_height = max(6.0, n * 0.6)
    fig_width = max(8.0, min(12.0, n * 0.6))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)

    group_gap = max(0.0, group_gap)
    bar_gap = max(0.0, min(1.0, bar_gap))
    y = np.arange(n) * (1.2 + group_gap)
    bar_height = min(0.35, max(0.18, 0.5 * (label_fontsize / 24)))
    offset = bar_height * (1.0 + bar_gap)

    error_label = "Pseudo error (normalized)" if use_normalized else "Pseudo error"

    err_numeric = []
    err_visible = []
    for val in err_vals:
        if val is None:
            err_numeric.append(0.0)
            err_visible.append(False)
        else:
            err_numeric.append(val)
            err_visible.append(True)

    err_bars = ax.barh(y - offset, err_numeric, height=bar_height, color="#FF9F7F", label=error_label)
    ax.barh(y + offset, f1_vals, height=bar_height, color="#5B8FF9", label="Pair (m1) F1")

    for idx, bar in enumerate(err_bars):
        if not err_visible[idx]:
            bar.set_visible(False)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, ha="right", fontsize=label_fontsize)
    ax.set_xlabel("Score", fontsize=label_fontsize)
    title_text = ax.set_title("Pseudo error vs Pair (m1) F1", pad=max(10, label_fontsize * 0.5), loc=title_loc)
    if title_x is not None:
        title_text.set_x(title_x)

    combined_vals = [v for v in err_vals if v is not None] + f1_vals
    max_val = max(combined_vals) if combined_vals else x_max
    limit = max(x_max, max_val + 0.02)
    ax.set_xlim(0, limit)
    ax.tick_params(axis="x", labelsize=label_fontsize)
    legend = ax.legend(
        loc="upper center",
        bbox_to_anchor=(legend_anchor_x, legend_anchor_y),
        ncol=2,
        frameon=False,
    )

    for vals, y_offset, visibility in ((err_vals, -offset, err_visible), (f1_vals, offset, [True] * len(f1_vals))):
        for idx, val in enumerate(vals):
            if val is None or (visibility[idx] if visibility else True) is False:
                continue
            y_pos = y[idx] + y_offset
            if val > limit - 0.06:
                text_x = min(limit - 0.01, val)
                ax.text(text_x, y_pos, f"{val:.3f}", va="center", ha="right", color="black", fontsize=max(10, int(label_fontsize * 0.9)))
            elif val > 0.18:
                ax.text(val - 0.01, y_pos, f"{val:.3f}", va="center", ha="right", color="black", fontsize=max(10, int(label_fontsize * 0.9)))
            else:
                ax.text(val + 0.01, y_pos, f"{val:.3f}", va="center", ha="left", color="black", fontsize=max(10, int(label_fontsize * 0.9)))

    left_pad = _compute_left_padding(fig, labels, label_fontsize)
    top_pad, bottom_pad = _calc_vertical_pads(label_fontsize)
    fig.subplots_adjust(left=left_pad, right=0.95, top=top_pad, bottom=max(bottom_pad, 0.18))
    fig.savefig(output_path, dpi=200)
    print(f"[資訊] 已輸出結合圖表：{output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine pseudo error report and Pair (m1) F1 chart")
    parser.add_argument("--base-dir", type=Path, default=Path("."))
    parser.add_argument("--pattern", default="prompt_ECPE_few_shot_ST_*")
    parser.add_argument("--pseudo-report", default="pseudo_error_report.txt")
    parser.add_argument("--counts-report", default="counts_summary_detailed.txt")
    parser.add_argument("--use-normalized", action="store_true", help="使用 normalized error (預設 actual)")
    parser.add_argument("--bg-color", default="#DDDDDD")
    parser.add_argument("--output", type=Path, default=Path("pseudo_error_vs_f1.png"))
    parser.add_argument("--x-max", type=float, default=0.42, help="X 軸上限 (預設 0.42，若數值超過會自動延展)")
    parser.add_argument("--sort", action="store_true", help="依 F1 由高到低排序")
    parser.add_argument("--group-gap", type=float, default=0.4, help="不同指標組之間的縱向間距，預設 0.4")
    parser.add_argument("--bar-gap", type=float, default=0.25, help="同組兩條長條的縱向間距，預設 0.25")
    parser.add_argument("--title-loc", type=str, default="center", choices=["left", "center", "right"], help="主圖標題對齊")
    parser.add_argument("--title-x", type=float, help="主圖標題 x 座標 (0~1，若未指定則依 loc)")
    parser.add_argument("--legend-anchor-x", type=float, default=0.35, help="主圖圖例水平定位 (0~1，預設 0.35)")
    parser.add_argument("--legend-anchor-y", type=float, default=-0.12, help="主圖圖例垂直定位 (預設 -0.12)")
    parser.add_argument("--subset-output", type=Path, help="若指定則輸出僅包含部分實驗的額外圖檔")
    parser.add_argument("--subset-top", type=int, default=0, help="在額外圖中加入 F1 最高的前 k 組")
    parser.add_argument("--subset-bottom", type=int, default=0, help="在額外圖中加入 F1 最低的後 k 組")
    parser.add_argument(
        "--subset-extra-labels",
        type=str,
        default="",
        help="以逗號分隔的額外實驗標籤 (使用簡化後名稱)",
    )
    parser.add_argument(
        "--subset-exclude-labels",
        type=str,
        default="",
        help="以逗號分隔，在 subset 中排除的實驗標籤",
    )
    parser.add_argument(
        "--subset-extra-counts",
        type=str,
        default="",
        help="以逗號分隔的資料夾路徑，僅使用 counts 報告加入 subset (無 pseudo error)",
    )
    parser.add_argument(
        "--label-aliases",
        type=str,
        default="",
        help="以逗號分隔 old=new 的映射，套用在簡化後的標籤名稱",
    )
    parser.add_argument("--subset-title-loc", type=str, choices=["left", "center", "right"], help="subset 圖標題對齊 (預設沿用主圖)")
    parser.add_argument("--subset-title-x", type=float, help="subset 圖標題 x 座標 (0~1，若未指定則沿用主圖設定)")
    parser.add_argument("--subset-legend-anchor-x", type=float, help="subset 圖例水平定位 (預設沿用主圖)")
    parser.add_argument("--subset-legend-anchor-y", type=float, help="subset 圖例垂直定位 (預設沿用主圖)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_fonts()
    alias_map = _parse_aliases(args.label_aliases)
    data = collect_combined(
        base_dir=args.base_dir,
        pattern=args.pattern,
        pseudo_report=args.pseudo_report,
        counts_report=args.counts_report,
        use_normalized=args.use_normalized,
        alias_map=alias_map,
    )
    if not data:
        print("[警告] 找不到可結合的實驗 (需要同時存在 pseudo_error_report 與 counts_summary_detailed)")
        return
    if args.sort:
        data = sorted(data, key=lambda x: x[2], reverse=True)
    plot_combined(
        data,
        args.output,
        args.use_normalized,
        args.bg_color,
        args.x_max,
        args.group_gap,
        args.bar_gap,
        args.title_loc,
        args.title_x,
        args.legend_anchor_x,
        args.legend_anchor_y,
    )

    if args.subset_output:
        extras = [lbl.strip() for lbl in args.subset_extra_labels.split(",") if lbl.strip()]
        excludes = {lbl.strip() for lbl in args.subset_exclude_labels.split(",") if lbl.strip()}
        subset = select_subset(
            data,
            max(0, args.subset_top),
            max(0, args.subset_bottom),
            extras,
        )

        extra_dirs: List[Path] = []
        if args.subset_extra_counts:
            for raw in args.subset_extra_counts.split(","):
                raw = raw.strip()
                if not raw:
                    continue
                path = Path(raw)
                if not path.is_absolute():
                    path = args.base_dir / raw
                extra_dirs.append(path)
        extra_entries = collect_extra_counts(extra_dirs, args.counts_report, alias_map) if extra_dirs else []
        if extra_entries:
            existing = {item[0] for item in subset}
            for entry in extra_entries:
                if entry[0] not in existing:
                    subset.append(entry)
                    existing.add(entry[0])
        if excludes:
            subset = [item for item in subset if item[0] not in excludes]
        if subset:
            plot_combined(
                subset,
                args.subset_output,
                args.use_normalized,
                args.bg_color,
                args.x_max,
                args.group_gap,
                args.bar_gap,
                args.subset_title_loc or args.title_loc,
                args.subset_title_x if args.subset_title_x is not None else args.title_x,
                args.subset_legend_anchor_x if args.subset_legend_anchor_x is not None else args.legend_anchor_x,
                args.subset_legend_anchor_y if args.subset_legend_anchor_y is not None else args.legend_anchor_y,
            )
        else:
            print("[警告] 無法建立 subset 圖表：沒有符合條件的資料")


if __name__ == "__main__":
    main()
