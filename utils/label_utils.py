"""Utilities for label simplification and font configuration used by plotting scripts."""

from __future__ import annotations

import re
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
from matplotlib import font_manager


def _ensure_font(preferred_name: str, candidate_paths: List[Path]) -> str:
    for path in candidate_paths:
        if path.exists():
            font_manager.fontManager.addfont(str(path))
            loaded = font_manager.FontProperties(fname=str(path)).get_name()
            return loaded
    print(f"[警告] 找不到 {preferred_name} 字型，將由 Matplotlib 嘗試自動替代")
    return preferred_name


def configure_fonts(font_size: int = 24) -> None:
    calisto = _ensure_font(
        "Calisto MT",
        [
            Path("C:/Windows/Fonts/CALIST.TTF"),
            Path("C:/Windows/Fonts/calisto.ttf"),
            Path("/mnt/c/Windows/Fonts/CALIST.TTF"),
            Path("/mnt/c/Windows/Fonts/calisto.ttf"),
        ],
    )
    dfkai = _ensure_font(
        "DFKai-SB",
        [
            Path("C:/Windows/Fonts/KAIU.TTF"),
            Path("C:/Windows/Fonts/kaiu.ttf"),
            Path("/mnt/c/Windows/Fonts/KAIU.TTF"),
            Path("/mnt/c/Windows/Fonts/kaiu.ttf"),
        ],
    )
    plt.rcParams.update({"font.family": [calisto, dfkai], "font.size": font_size, "axes.unicode_minus": False})


def _simplify_label(name: str) -> str:
    # 如果包含 _th，直接保留從 _th 開始的片段（去掉前面的全部）
    m = re.search(r"_th", name)
    if m:
        simplified = name[m.start() + 1 :]
    else:
        simplified = name

    # 支援 _seed42 與 -seed42 兩種寫法（在任何情況都移除）
    simplified = re.sub(r"[-_]seed\d+", "", simplified)

    # 將 retain_pseudo 保留後綴 CE/EC
    def _replace_retain(m):
        suffix = m.group(1)
        return f"_{suffix}" if suffix else ""

    simplified = re.sub(r"_retain_pseudo(?:_(CE|EC))?", _replace_retain, simplified)

    # remove timestamp patterns like _2025_11_08_15_39_04
    simplified = re.sub(r"_\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}", "", simplified)
    # remove fold info _f1-10 or _f3 etc.
    simplified = re.sub(r"_f\d+(?:-\d+)?", "", simplified)
    # remove iterations/iters _i70
    simplified = re.sub(r"_i\d+", "", simplified)
    # remove learning rate _lr1e-5
    simplified = re.sub(r"_lr[0-9.eE+\-]+", "", simplified)
    # remove batch size _bs8
    simplified = re.sub(r"_bs\d+", "", simplified)
    # remove weight decay _wd0.01
    simplified = re.sub(r"_wd[0-9.eE+\-]+", "", simplified)
    # remove bert spec like _bert-base-chinese
    simplified = re.sub(r"_bert[^_]*", "", simplified)
    # 將 _maskboth/_maskor/_maskemotion/_maskcause 改成 _both/_or/_emotion/_cause
    simplified = re.sub(r"_maskboth", "_both", simplified)
    simplified = re.sub(r"_maskor", "_or", simplified)
    simplified = re.sub(r"_maskemotion", "_emotion", simplified)
    simplified = re.sub(r"_maskcause", "_cause", simplified)
    # 替換 _both->_and, _emotion->_emo
    simplified = re.sub(r"_both", "_and", simplified)
    simplified = re.sub(r"_emotion", "_emo", simplified)
    # 刪除未指定的 _mask
    simplified = re.sub(r"_mask(?!or|emotion|both|cause)", "", simplified)
    simplified = re.sub(r"_gamma[0-9eE.+-]+", "", simplified)
    simplified = re.sub(r"_reg[0-9a-zA-Z.+-]+", "", simplified)
    simplified = simplified.rstrip("_")
    return simplified
