#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把多個 results_* 資料夾中的 multi-seed summary 匯出成分頁報表。"""

from __future__ import annotations

# 匯入 argparse，用來讀取命令列參數。
import argparse
# 匯入 csv，用來在沒有 openpyxl 時輸出多個 CSV 檔。
import csv
# 匯入 re，用來解析 summary 與 counts_summary_detailed 文字內容。
import re
# 匯入 sys，用來回傳命令列結束狀態。
import sys
# 匯入 dataclass，用來建立清楚的小型資料容器。
from dataclasses import dataclass
# 匯入 Path 與 PureWindowsPath，用來同時處理 Windows 與 WSL 路徑。
from pathlib import Path, PureWindowsPath
# 匯入 Iterable，用來標註可迭代輸入型別。
from typing import Iterable


# 預設要彙整的四個 results 資料夾。
DEFAULT_RESULT_DIRS = [
    "results_ep_split10_t1v1te1_u7_aligned_disjoint_2019",
    "results_ep_split10_t9te1_aligned_disjoint_2019",
    "results_ep_enecpe_reccon_merged_t1v1te1_u7",
    "results_ep_enecpe_reccon_merged_t9",
]

# 預設跨磁碟搜尋順序：先找 D，再找 F、I、H。
DEFAULT_SEARCH_DRIVES = ["D", "F", "I", "H"]

# counts_summary_detailed 內固定會出現的五種指標。
METRIC_NAMES = ["Emotion", "Cause", "Pair (m1)", "Pair (m2)", "Pair (m3)"]

# 每個實驗目錄中可能出現的詳細計數檔名。
COUNTS_FILENAMES = ["counts_summary_detailed.txt", "counts_summary_detailed"]


@dataclass(frozen=True)
class MetricCounts:
    """單一 seed、單一指標的 10 折累計計數。

    輸入：由 parse_counts_file() 從 counts_summary_detailed 解析而來。
    輸出：提供工作表列印 Total TP / Pred / GT、P/R/F1 與 Std 欄位。
    """

    # 指標名稱，例如 Emotion 或 Pair (m1)。
    metric: str
    # 10 折累計的預測正確數量。
    total_tp: int
    # 10 折累計的預測正例數量。
    total_pred: int
    # 10 折累計的實際正例數量。
    total_gt: int
    # 檔案內已算好的 micro precision，小數格式，例如 0.7416。
    p_micro: float
    # 檔案內已算好的 micro recall，小數格式，例如 0.7871。
    r_micro: float
    # 檔案內已算好的 micro F1，小數格式，例如 0.7636。
    f1_micro: float
    # 檔案內已算好的 per-fold precision 標準差，小數格式。
    std_p: float
    # 檔案內已算好的 per-fold recall 標準差，小數格式。
    std_r: float
    # 檔案內已算好的 per-fold F1 標準差，小數格式。
    std_f1: float


@dataclass(frozen=True)
class SummaryMetric:
    """summary 檔中三個 seed 平均後的 Mean ± Std。

    輸入：由 parse_summary_means() 從 *_summary.txt 解析而來。
    輸出：提供「三次平均彙整」分頁的對照值。
    """

    # 三個 seed 的 precision 平均，小數格式。
    p_mean: float
    # 三個 seed 的 precision 標準差，小數格式。
    p_std: float
    # 三個 seed 的 recall 平均，小數格式。
    r_mean: float
    # 三個 seed 的 recall 標準差，小數格式。
    r_std: float
    # 三個 seed 的 F1 平均，小數格式。
    f1_mean: float
    # 三個 seed 的 F1 標準差，小數格式。
    f1_std: float


@dataclass(frozen=True)
class SeedMetricRow:
    """輸出到各 results 分頁的一列 seed-level 指標。

    輸入：summary 檔、實驗目錄、counts_summary_detailed 共同組合而成。
    輸出：一列包含分子分母與 Excel 公式的資料列。
    """

    # results 資料夾的短名稱。
    result_group: str
    # summary 檔案名稱。
    summary_file: str
    # summary 檔解析出的實驗目錄名稱。
    experiment_name: str
    # 從實驗目錄名稱抓到的 seed，例如 seed42。
    seed: str
    # 實際找到的實驗資料夾路徑，找不到時為空字串。
    experiment_dir: str
    # 實際找到的 counts_summary_detailed 路徑，找不到時為空字串。
    counts_file: str
    # 目前列代表的指標名稱。
    metric: str
    # 解析成功時的計數；解析失敗或找不到檔案時為 None。
    counts: MetricCounts | None
    # 目前列的狀態說明，例如 ok 或 missing_counts_file。
    status: str


@dataclass(frozen=True)
class SummaryContext:
    """單一 summary 檔解析後的上下文。

    輸入：一個 *_summary.txt 檔案。
    輸出：summary 檔名、三個實驗目錄名稱、以及已彙整的 Mean ± Std。
    """

    # summary 檔所在的 results 資料夾短名稱。
    result_group: str
    # summary 檔完整路徑。
    summary_path: Path
    # summary 檔列出的實驗目錄名稱清單。
    experiment_names: list[str]
    # summary 檔中「10 折累計統計結果 (Mean ± Std)」的解析結果。
    summary_means: dict[str, SummaryMetric]


def parse_args() -> argparse.Namespace:
    """解析命令列參數。

    輸入：使用者在命令列輸入的參數。
    輸出：argparse.Namespace，供 main() 後續使用。
    """

    # 建立命令列解析器。
    parser = argparse.ArgumentParser(description="將 multi-seed summary 與 counts_summary_detailed 匯出成 Excel/CSV 分頁報表。")
    # 加入 results 資料夾參數；不給時使用 DEFAULT_RESULT_DIRS。
    parser.add_argument("--result-dirs", nargs="*", default=DEFAULT_RESULT_DIRS, help="要彙整的 results_* 資料夾。")
    # 加入輸出檔參數；xlsx 才是真正有分頁的格式。
    parser.add_argument("--output", default="results_counts_summary_workbook.xlsx", help="輸出 .xlsx 路徑。")
    # 加入 summary 檔 glob；預設抓所有 *_summary.txt。
    parser.add_argument("--summary-glob", default="*_summary.txt", help="results 資料夾內 summary 檔的 glob。")
    # 加入搜尋磁碟；會同時支援 Windows D:/ 與 WSL /mnt/d。
    parser.add_argument("--search-drives", nargs="*", default=DEFAULT_SEARCH_DRIVES, help="要搜尋實驗根目錄的磁碟代號。")
    # 加入額外根目錄；適合直接指定 H:/ep_xxx 或 /mnt/h/ep_xxx。
    parser.add_argument("--extra-experiment-roots", nargs="*", default=[], help="額外要搜尋的實驗根目錄。")
    # 加入 strict 模式；有缺檔時直接失敗。
    parser.add_argument("--strict", action="store_true", help="遇到缺少 summary/counts 檔案時直接回傳錯誤。")
    # 加入可選 CSV 輸出資料夾；CSV 沒有真正分頁，所以用一個資料夾存多個 CSV。
    parser.add_argument("--also-csv-dir", default=None, help="可選：同時輸出每個分頁對應的 CSV 到此資料夾。")
    # 回傳解析結果。
    return parser.parse_args()


def read_text(path: Path) -> str:
    """讀取 UTF-8 文字檔。

    輸入：path 是要讀取的文字檔路徑。
    輸出：檔案內容字串；遇到少數壞字元時以 errors='ignore' 略過。
    """

    # 使用 utf-8 讀取，並忽略少數可能的編碼雜訊。
    return path.read_text(encoding="utf-8", errors="ignore")


def percent_to_float(text: str) -> float:
    """把百分比文字轉成小數。

    輸入：text 是像 '74.16' 或 '74.16%' 的文字。
    輸出：0.7416 這類可直接寫入 Excel 百分比欄位的浮點數。
    """

    # 移除百分號與空白。
    cleaned = text.replace("%", "").strip()
    # 百分比除以 100 後回傳。
    return float(cleaned) / 100.0


def normalize_experiment_name(raw_name: str) -> str:
    """把 summary 中的實驗目錄字串正規化成資料夾名稱。

    輸入：raw_name 可以是純資料夾名稱、Windows 完整路徑或 WSL/Unix 路徑。
    輸出：實驗資料夾 basename。
    """

    # 去掉前後空白與可能包住路徑的引號。
    cleaned = raw_name.strip().strip("'\"")
    # 若是 Windows 路徑，PureWindowsPath 可以正確取出最後一段。
    windows_name = PureWindowsPath(cleaned).name
    # 若是 Unix/WSL 路徑，Path 可以正確取出最後一段。
    unix_name = Path(cleaned).name
    # 選擇較短但非空的 basename；通常兩者會相同。
    return unix_name if len(unix_name) <= len(windows_name) else windows_name


def extract_seed(experiment_name: str) -> str:
    """從實驗目錄名稱擷取 seed 標籤。

    輸入：experiment_name 是實驗資料夾 basename。
    輸出：像 seed42 的字串；找不到時回傳 unknown_seed。
    """

    # 從資料夾名稱搜尋 seed 後面的數字。
    match = re.search(r"seed(\d+)", experiment_name)
    # 找到時回傳 seedXX，找不到時給固定佔位字。
    return f"seed{match.group(1)}" if match else "unknown_seed"


def parse_experiment_names(summary_path: Path) -> list[str]:
    """解析 summary 檔內「彙整的實驗目錄」清單。

    輸入：summary_path 是 *_summary.txt 路徑。
    輸出：summary 中列出的實驗資料夾 basename 清單。
    """

    # 讀取 summary 檔內容。
    text = read_text(summary_path)
    # 切成一行一行方便掃描。
    lines = text.splitlines()
    # 建立輸出清單。
    names: list[str] = []
    # 標記目前是否已進入「彙整的實驗目錄」區塊。
    in_section = False
    # 逐行掃描 summary。
    for line in lines:
        # 找到區塊標題後開始收集。
        if "彙整的實驗目錄" in line:
            in_section = True
            continue
        # 如果還沒進入區塊，就略過目前行。
        if not in_section:
            continue
        # 空行代表實驗目錄清單結束。
        if not line.strip():
            break
        # 每個實驗目錄行通常以「-」開頭。
        match = re.match(r"\s*-\s+(.+?)\s*$", line)
        # 如果符合格式，就取出 basename 後加入清單。
        if match:
            names.append(normalize_experiment_name(match.group(1)))
    # 回傳解析出的實驗目錄名稱。
    return names


def metric_block(section: str, metric: str) -> str:
    """從一段文字中切出單一 metric 區塊。

    輸入：section 是累計統計文字，metric 是 Emotion / Cause / Pair(m*)。
    輸出：該 metric 從標題到下一個 metric 標題前的文字；找不到時回傳空字串。
    """

    # 建立目前 metric 標題的正則表達式。
    current_pattern = re.compile(rf"^\s*{re.escape(metric)}:\s*$", re.MULTILINE)
    # 在 section 中尋找目前 metric 標題。
    current_match = current_pattern.search(section)
    # 找不到標題時回傳空字串。
    if current_match is None:
        return ""
    # 設定區塊起點為目前標題位置。
    start = current_match.start()
    # 預設區塊終點為整段文字結尾。
    end = len(section)
    # 逐一找其他 metric 標題，取目前標題後最近的一個作為終點。
    for next_metric in METRIC_NAMES:
        # 略過自己。
        if next_metric == metric:
            continue
        # 建立下一個 metric 標題的正則表達式。
        next_pattern = re.compile(rf"^\s*{re.escape(next_metric)}:\s*$", re.MULTILINE)
        # 搜尋下一個 metric 標題。
        next_match = next_pattern.search(section, current_match.end())
        # 如果找到且位置更近，就更新終點。
        if next_match is not None and next_match.start() < end:
            end = next_match.start()
    # 回傳切出的區塊。
    return section[start:end]


def parse_summary_means(summary_path: Path) -> dict[str, SummaryMetric]:
    """解析 summary 檔的三次平均 Mean ± Std。

    輸入：summary_path 是 aggregate_multi_seed.py 產生的 *_summary.txt。
    輸出：metric 名稱到 SummaryMetric 的對應表。
    """

    # 讀取 summary 檔內容。
    text = read_text(summary_path)
    # 找出 Mean ± Std 區塊的起點。
    marker = "10 折累計統計結果 (Mean ± Std):"
    # 如果找不到該區塊，就回傳空字典。
    if marker not in text:
        return {}
    # 只解析 marker 後方文字。
    section = text.split(marker, 1)[1]
    # 建立輸出字典。
    results: dict[str, SummaryMetric] = {}
    # 逐一解析五個 metric。
    for metric in METRIC_NAMES:
        # 切出單一 metric 區塊。
        block = metric_block(section, metric)
        # 區塊不存在時略過。
        if not block:
            continue
        # 解析 P mean/std。
        p_match = re.search(r"P \(micro\):\s*([\d.]+)%\s*±\s*([\d.]+)%", block)
        # 解析 R mean/std。
        r_match = re.search(r"R \(micro\):\s*([\d.]+)%\s*±\s*([\d.]+)%", block)
        # 解析 F1 mean/std。
        f1_match = re.search(r"F1 \(micro\):\s*([\d.]+)%\s*±\s*([\d.]+)%", block)
        # 三個都存在才建立 SummaryMetric。
        if p_match and r_match and f1_match:
            results[metric] = SummaryMetric(
                p_mean=percent_to_float(p_match.group(1)),
                p_std=percent_to_float(p_match.group(2)),
                r_mean=percent_to_float(r_match.group(1)),
                r_std=percent_to_float(r_match.group(2)),
                f1_mean=percent_to_float(f1_match.group(1)),
                f1_std=percent_to_float(f1_match.group(2)),
            )
    # 回傳所有可解析的三次平均結果。
    return results


def find_number(block: str, pattern: str) -> int:
    """從 metric 區塊中擷取整數。

    輸入：block 是 metric 區塊文字，pattern 是含一個擷取群組的正則式。
    輸出：擷取到的整數；找不到時丟出 ValueError。
    """

    # 在 block 中尋找指定格式。
    match = re.search(pattern, block)
    # 找不到就回報明確錯誤。
    if match is None:
        raise ValueError(f"找不到整數欄位: {pattern}")
    # 回傳第一個擷取群組的整數值。
    return int(match.group(1))


def find_percent(block: str, pattern: str) -> float:
    """從 metric 區塊中擷取百分比並轉成小數。

    輸入：block 是 metric 區塊文字，pattern 是含一個百分比擷取群組的正則式。
    輸出：0.0 到 1.0 附近的小數值；找不到時丟出 ValueError。
    """

    # 在 block 中尋找指定百分比格式。
    match = re.search(pattern, block)
    # 找不到就回報明確錯誤。
    if match is None:
        raise ValueError(f"找不到百分比欄位: {pattern}")
    # 將百分比轉為小數後回傳。
    return percent_to_float(match.group(1))


def parse_counts_file(counts_path: Path) -> dict[str, MetricCounts]:
    """解析 counts_summary_detailed 的 10 折累計統計結果。

    輸入：counts_path 是某個 seed 實驗目錄內的 counts_summary_detailed 檔案。
    輸出：metric 名稱到 MetricCounts 的對應表。
    """

    # 讀取 detailed counts 檔案。
    text = read_text(counts_path)
    # 找出最後一個「10 折累計統計結果」，避免前面若有重複文字時抓錯區塊。
    marker_index = text.rfind("10 折累計統計結果")
    # 找不到累計區塊時丟出錯誤。
    if marker_index < 0:
        raise ValueError(f"找不到 10 折累計統計結果區塊: {counts_path}")
    # 只解析最後一個累計區塊後方文字。
    section = text[marker_index:]
    # 建立輸出字典。
    results: dict[str, MetricCounts] = {}
    # 逐一解析五個 metric。
    for metric in METRIC_NAMES:
        # 切出目前 metric 的文字區塊。
        block = metric_block(section, metric)
        # 若缺少某個 metric，就略過讓呼叫端可看出缺資料。
        if not block:
            continue
        # 解析 Total TP。
        total_tp = find_number(block, r"Total TP .*?:\s*(\d+)")
        # 解析 Total Pred。
        total_pred = find_number(block, r"Total Pred .*?:\s*(\d+)")
        # 解析 Total GT。
        total_gt = find_number(block, r"Total GT .*?:\s*(\d+)")
        # 解析 P micro。
        p_micro = find_percent(block, r"P \(micro\):\s*([\d.]+)%")
        # 解析 R micro。
        r_micro = find_percent(block, r"R \(micro\):\s*([\d.]+)%")
        # 解析 F1 micro。
        f1_micro = find_percent(block, r"F1 \(micro\):\s*([\d.]+)%")
        # 解析 Std P；舊檔若沒有 Std 就補 0。
        std_p = find_percent(block, r"Std P:\s*([\d.]+)%") if "Std P" in block else 0.0
        # 解析 Std R；舊檔若沒有 Std 就補 0。
        std_r = find_percent(block, r"Std R:\s*([\d.]+)%") if "Std R" in block else 0.0
        # 解析 Std F1；舊檔若沒有 Std 就補 0。
        std_f1 = find_percent(block, r"Std F1:\s*([\d.]+)%") if "Std F1" in block else 0.0
        # 儲存目前 metric 的計數結果。
        results[metric] = MetricCounts(metric, total_tp, total_pred, total_gt, p_micro, r_micro, f1_micro, std_p, std_r, std_f1)
    # 回傳五個 metric 的解析結果。
    return results


def infer_experiment_root_names(result_dir_name: str) -> list[str]:
    """由 results 資料夾名稱推測實驗根目錄名稱。

    輸入：result_dir_name，例如 results_ep_split10_t1v1te1_u7_aligned_disjoint_2019。
    輸出：可能的實驗根目錄名稱，例如 ep_split10_t1v1te1_u7_aligned_disjoint_2019。
    """

    # 如果名稱以 results_ 開頭，就移除此前綴。
    base_name = result_dir_name[len("results_") :] if result_dir_name.startswith("results_") else result_dir_name
    # 建立候選根目錄名稱清單。
    names = [base_name]
    # 英文 t9 在不同機器上可能叫 t9 或 t9te1，所以互相補一個候選。
    if base_name.endswith("_t9"):
        names.append(f"{base_name}te1")
    # 如果目前是 t9te1，也補一個 t9 候選。
    if base_name.endswith("_t9te1"):
        names.append(base_name[: -len("te1")])
    # 回傳去重後的候選名稱。
    return list(dict.fromkeys(names))


def windows_and_wsl_drive_roots(drive: str, root_name: str) -> list[Path]:
    """產生單一磁碟下的 Windows 與 WSL 候選根目錄。

    輸入：drive 是 D/F/I/H 這類磁碟代號，root_name 是 ep_* 根目錄名稱。
    輸出：例如 D:/ep_xxx 與 /mnt/d/ep_xxx 兩種 Path。
    """

    # 將磁碟代號轉成大寫，供 Windows 路徑使用。
    drive_upper = drive.rstrip(":\\/").upper()
    # 將磁碟代號轉成小寫，供 WSL /mnt 路徑使用。
    drive_lower = drive_upper.lower()
    # 回傳 Windows 與 WSL 兩種候選根目錄。
    return [Path(f"{drive_upper}:/") / root_name, Path("/mnt") / drive_lower / root_name]


def unique_paths(paths: Iterable[Path]) -> list[Path]:
    """保留順序並移除重複路徑。

    輸入：paths 是 Path 可迭代物件。
    輸出：去重後的 Path 清單。
    """

    # 建立已看過路徑的集合。
    seen: set[str] = set()
    # 建立輸出清單。
    output: list[Path] = []
    # 逐一處理輸入路徑。
    for path in paths:
        # 用字串作為去重鍵值。
        key = str(path)
        # 已看過就略過。
        if key in seen:
            continue
        # 標記此路徑已看過。
        seen.add(key)
        # 加入輸出清單。
        output.append(path)
    # 回傳去重結果。
    return output


def candidate_experiment_roots(result_dir: Path, extra_roots: list[str], search_drives: list[str]) -> list[Path]:
    """建立搜尋實驗資料夾時要嘗試的根目錄清單。

    輸入：result_dir 是 results 資料夾，extra_roots 是使用者額外指定根目錄，search_drives 是磁碟清單。
    輸出：依優先序排列的候選根目錄清單。
    """

    # 從 results 資料夾名稱推測 ep_* 實驗根目錄。
    root_names = infer_experiment_root_names(result_dir.name)
    # 先放使用者明確指定的根目錄。
    candidates = [Path(root) for root in extra_roots]
    # 把目前工作目錄當作直接放實驗目錄的候選。
    candidates.append(Path.cwd())
    # 把 results 同層的 ep_* 目錄加入候選。
    candidates.extend(result_dir.parent / root_name for root_name in root_names)
    # 把目前工作目錄底下的 ep_* 目錄加入候選。
    candidates.extend(Path.cwd() / root_name for root_name in root_names)
    # 依 D/F/I/H 等磁碟產生 Windows 與 WSL 候選路徑。
    for drive in search_drives:
        for root_name in root_names:
            candidates.extend(windows_and_wsl_drive_roots(drive, root_name))
    # 回傳去重後的候選清單。
    return unique_paths(candidates)


def resolve_result_dir(raw_dir: str) -> Path:
    """解析 results 資料夾路徑，並處理 t9/t9te1 名稱差異。

    輸入：raw_dir 是命令列傳入的 results 資料夾路徑或名稱。
    輸出：存在時回傳實際 Path；找不到時仍回傳原始 Path 供錯誤訊息使用。
    """

    # 先把使用者輸入轉成 Path。
    path = Path(raw_dir)
    # 如果路徑存在，直接回傳。
    if path.is_dir():
        return path
    # 若是相對路徑，試著從目前工作目錄尋找。
    cwd_path = Path.cwd() / raw_dir
    # 如果目前工作目錄底下存在，回傳該路徑。
    if cwd_path.is_dir():
        return cwd_path
    # 如果名稱以 _t9 結尾，嘗試補成 _t9te1。
    if raw_dir.endswith("_t9"):
        alt = Path(f"{raw_dir}te1")
        # 先查相對路徑。
        if alt.is_dir():
            return alt
        # 再查目前工作目錄底下。
        if (Path.cwd() / alt).is_dir():
            return Path.cwd() / alt
    # 找不到時回傳原始 path，讓主流程可輸出 missing 狀態。
    return path


def resolve_experiment_dir(experiment_name: str, result_dir: Path, extra_roots: list[str], search_drives: list[str]) -> Path | None:
    """從多個可能根目錄中找到實驗資料夾。

    輸入：experiment_name 是 summary 列出的實驗資料夾名稱，result_dir 是 summary 所在 results 資料夾。
    輸出：找到時回傳實驗資料夾 Path；找不到時回傳 None。
    """

    # 如果 summary 本身列的是可用完整路徑，就直接回傳。
    direct_path = Path(experiment_name)
    # 只有路徑存在時才採用。
    if direct_path.is_dir():
        return direct_path
    # 取得候選根目錄。
    roots = candidate_experiment_roots(result_dir, extra_roots, search_drives)
    # 逐一嘗試 root / experiment_name。
    for root in roots:
        # 組合候選實驗資料夾。
        candidate = root / experiment_name
        # 找到存在的資料夾就回傳。
        if candidate.is_dir():
            return candidate
    # 全部找不到就回傳 None。
    return None


def find_counts_file(experiment_dir: Path) -> Path | None:
    """在實驗資料夾中尋找 counts_summary_detailed 檔案。

    輸入：experiment_dir 是單一 seed 實驗資料夾。
    輸出：找到時回傳檔案 Path；找不到時回傳 None。
    """

    # 逐一嘗試可能檔名。
    for filename in COUNTS_FILENAMES:
        # 組合候選檔案路徑。
        candidate = experiment_dir / filename
        # 找到檔案就回傳。
        if candidate.is_file():
            return candidate
    # 沒有任何候選存在就回傳 None。
    return None


def collect_summary_contexts(result_dirs: list[Path], summary_glob: str) -> list[SummaryContext]:
    """收集所有 results 資料夾內的 summary 檔上下文。

    輸入：result_dirs 是 results 資料夾清單，summary_glob 是要抓取的 summary 檔名模式。
    輸出：SummaryContext 清單。
    """

    # 建立輸出清單。
    contexts: list[SummaryContext] = []
    # 逐一處理 results 資料夾。
    for result_dir in result_dirs:
        # 資料夾不存在就略過，主流程會另外統計 warning。
        if not result_dir.is_dir():
            continue
        # 依檔名排序，讓輸出穩定。
        for summary_path in sorted(result_dir.glob(summary_glob)):
            # 解析 summary 檔列出的三個 seed 實驗目錄。
            experiment_names = parse_experiment_names(summary_path)
            # 解析 summary 檔內的三次平均 Mean ± Std。
            summary_means = parse_summary_means(summary_path)
            # 加入輸出清單。
            contexts.append(SummaryContext(result_dir.name, summary_path, experiment_names, summary_means))
    # 回傳所有 summary 上下文。
    return contexts


def build_seed_rows(contexts: list[SummaryContext], extra_roots: list[str], search_drives: list[str]) -> list[SeedMetricRow]:
    """由 summary 上下文建立各 seed 的 metric 資料列。

    輸入：contexts 是 summary 清單，extra_roots/search_drives 控制實驗目錄搜尋位置。
    輸出：SeedMetricRow 清單，每個 seed 會展開成五個 metric rows。
    """

    # 建立輸出列清單。
    rows: list[SeedMetricRow] = []
    # 逐一處理 summary。
    for context in contexts:
        # summary 所在的 results 資料夾。
        result_dir = context.summary_path.parent
        # 若 summary 沒列出實驗目錄，就建立一列狀態列方便追蹤。
        if not context.experiment_names:
            rows.append(SeedMetricRow(context.result_group, context.summary_path.name, "", "", "", "", "", None, "missing_experiment_list"))
            continue
        # 逐一處理 summary 中列出的 seed 實驗。
        for experiment_name in context.experiment_names:
            # 擷取 seed 標籤。
            seed = extract_seed(experiment_name)
            # 嘗試定位實際實驗資料夾。
            experiment_dir = resolve_experiment_dir(experiment_name, result_dir, extra_roots, search_drives)
            # 找不到實驗資料夾時，五個 metric 都輸出 missing 狀態。
            if experiment_dir is None:
                for metric in METRIC_NAMES:
                    rows.append(SeedMetricRow(context.result_group, context.summary_path.name, experiment_name, seed, "", "", metric, None, "missing_experiment_dir"))
                continue
            # 嘗試在實驗資料夾中找到 counts_summary_detailed。
            counts_path = find_counts_file(experiment_dir)
            # 找不到 counts 檔時，五個 metric 都輸出 missing 狀態。
            if counts_path is None:
                for metric in METRIC_NAMES:
                    rows.append(SeedMetricRow(context.result_group, context.summary_path.name, experiment_name, seed, str(experiment_dir), "", metric, None, "missing_counts_file"))
                continue
            # 嘗試解析 counts 檔。
            try:
                counts_by_metric = parse_counts_file(counts_path)
            # 解析失敗時仍保留錯誤狀態，方便回頭修檔。
            except Exception as exc:  # noqa: BLE001
                for metric in METRIC_NAMES:
                    rows.append(SeedMetricRow(context.result_group, context.summary_path.name, experiment_name, seed, str(experiment_dir), str(counts_path), metric, None, f"parse_error: {exc}"))
                continue
            # 逐一輸出五個 metric。
            for metric in METRIC_NAMES:
                # 取出目前 metric 的計數。
                counts = counts_by_metric.get(metric)
                # 若缺少 metric，標記為 missing_metric。
                status = "ok" if counts is not None else "missing_metric"
                # 建立 seed-level row。
                rows.append(SeedMetricRow(context.result_group, context.summary_path.name, experiment_name, seed, str(experiment_dir), str(counts_path), metric, counts, status))
    # 回傳所有 seed-level rows。
    return rows


def safe_sheet_name(raw_name: str, used_names: set[str]) -> str:
    """建立 Excel 可用且不重複的工作表名稱。

    輸入：raw_name 是原始名稱，used_names 是已使用名稱集合。
    輸出：長度不超過 31 且不含非法字元的 sheet 名稱。
    """

    # 移除 Excel sheet name 不允許的字元。
    cleaned = re.sub(r"[\\/*?:\[\]]", "_", raw_name)
    # 將太長的 results_ / ep_ 前綴壓短。
    cleaned = cleaned.replace("results_ep_", "").replace("aligned_disjoint_2019", "aligned")
    # 最多保留 31 字元。
    base = cleaned[:31] or "sheet"
    # 先使用 base 作為候選。
    candidate = base
    # 建立流水號。
    index = 2
    # 若重複就加上 _2、_3 等後綴。
    while candidate in used_names:
        # 後綴文字。
        suffix = f"_{index}"
        # 留出後綴長度後再截斷。
        candidate = f"{base[:31 - len(suffix)]}{suffix}"
        # 遞增流水號。
        index += 1
    # 記錄此名稱已被使用。
    used_names.add(candidate)
    # 回傳安全 sheet 名稱。
    return candidate


def column_letter(column_number: int) -> str:
    """把 1-based 欄號轉成 Excel 欄名。

    輸入：column_number 是 1 起算的欄位編號。
    輸出：A、B、AA 這類 Excel 欄名。
    """

    # 建立輸出字串。
    letters = ""
    # 持續除以 26 直到欄號歸零。
    while column_number:
        # Excel 欄位是 1-based，所以先扣 1。
        column_number, remainder = divmod(column_number - 1, 26)
        # 把目前餘數轉成 A-Z 並接到前面。
        letters = chr(65 + remainder) + letters
    # 回傳欄名。
    return letters


def formula_join(sheet_name: str, column: str, rows: list[int]) -> str:
    """建立跨列公式參照清單。

    輸入：sheet_name 是來源分頁名，column 是欄名，rows 是列號清單。
    輸出：像 'sheet'!I2,'sheet'!I7 的公式片段。
    """

    # 先把 sheet 名稱中的單引號替換成 Excel 公式可接受的兩個單引號。
    escaped_sheet_name = sheet_name.replace("'", "''")
    # Excel sheet 名稱若含特殊字元需要用單引號包住。
    quoted_sheet = f"'{escaped_sheet_name}'"
    # 將每個列號轉成 sheet!cell 參照後以逗號串起來。
    return ",".join(f"{quoted_sheet}!{column}{row}" for row in rows)


def seed_sheet_headers() -> list[str]:
    """回傳 seed-level 分頁表頭。

    輸入：無。
    輸出：各 results 分頁要使用的欄位名稱清單。
    """

    # 回傳固定欄位順序。
    return [
        "result_group",
        "summary_file",
        "experiment_name",
        "seed",
        "metric",
        "total_tp",
        "total_pred",
        "total_gt",
        "precision_formula",
        "recall_formula",
        "f1_formula",
        "precision_from_file",
        "recall_from_file",
        "f1_from_file",
        "std_precision_from_file",
        "std_recall_from_file",
        "std_f1_from_file",
        "counts_file",
        "experiment_dir",
        "status",
    ]


def seed_row_values(row: SeedMetricRow, excel_row: int) -> list[object]:
    """把 SeedMetricRow 轉成可寫入工作表的一列資料。

    輸入：row 是 seed-level 資料，excel_row 是即將寫入的 Excel 列號。
    輸出：依 seed_sheet_headers() 排列的欄位值。
    """

    # 如果沒有成功解析 counts，就輸出空白數值與狀態。
    if row.counts is None:
        return [row.result_group, row.summary_file, row.experiment_name, row.seed, row.metric, "", "", "", "", "", "", "", "", "", "", "", "", row.counts_file, row.experiment_dir, row.status]
    # P 公式使用 Total TP / Total Pred。
    p_formula = f"=IF(G{excel_row}=0,0,F{excel_row}/G{excel_row})"
    # R 公式使用 Total TP / Total GT。
    r_formula = f"=IF(H{excel_row}=0,0,F{excel_row}/H{excel_row})"
    # F1 公式使用 2PR/(P+R)。
    f1_formula = f"=IF(I{excel_row}+J{excel_row}=0,0,2*I{excel_row}*J{excel_row}/(I{excel_row}+J{excel_row}))"
    # 回傳完整資料列。
    return [
        row.result_group,
        row.summary_file,
        row.experiment_name,
        row.seed,
        row.metric,
        row.counts.total_tp,
        row.counts.total_pred,
        row.counts.total_gt,
        p_formula,
        r_formula,
        f1_formula,
        row.counts.p_micro,
        row.counts.r_micro,
        row.counts.f1_micro,
        row.counts.std_p,
        row.counts.std_r,
        row.counts.std_f1,
        row.counts_file,
        row.experiment_dir,
        row.status,
    ]


def group_rows_by_result(rows: list[SeedMetricRow]) -> dict[str, list[SeedMetricRow]]:
    """依 results 資料夾分組 seed-level rows。

    輸入：rows 是所有 seed-level 資料列。
    輸出：result_group 到 rows 的對應表。
    """

    # 建立分組字典。
    grouped: dict[str, list[SeedMetricRow]] = {}
    # 逐一放入對應 result_group。
    for row in rows:
        grouped.setdefault(row.result_group, []).append(row)
    # 回傳分組結果。
    return grouped


def write_xlsx(output_path: Path, contexts: list[SummaryContext], rows: list[SeedMetricRow]) -> None:
    """寫出真正有多分頁的 xlsx 檔。

    輸入：output_path 是 .xlsx 路徑，contexts 是 summary 上下文，rows 是 seed-level 資料列。
    輸出：在磁碟建立 Excel 活頁簿。
    """

    # 延遲匯入 openpyxl，讓沒有安裝時仍可使用 CSV fallback。
    from openpyxl import Workbook
    # 匯入 Font 以設定表頭粗體。
    from openpyxl.styles import Font

    # 建立新活頁簿。
    workbook = Workbook()
    # 移除預設空白分頁。
    workbook.remove(workbook.active)
    # 建立 sheet 名稱去重集合。
    used_sheet_names: set[str] = set()
    # 建立 result_group 到 sheet_name 的對照。
    result_to_sheet: dict[str, str] = {}
    # 建立 summary/metric 到 seed row numbers 的對照，供平均公式使用。
    summary_metric_rows: dict[tuple[str, str, str], list[int]] = {}
    # 依 results 資料夾分組 rows。
    grouped_rows = group_rows_by_result(rows)
    # 逐一建立四個 results 分頁。
    for result_group, group_rows in grouped_rows.items():
        # 建立安全 sheet 名稱。
        sheet_name = safe_sheet_name(result_group, used_sheet_names)
        # 記錄對照。
        result_to_sheet[result_group] = sheet_name
        # 建立工作表。
        worksheet = workbook.create_sheet(sheet_name)
        # 寫入表頭。
        worksheet.append(seed_sheet_headers())
        # 將表頭設為粗體。
        for cell in worksheet[1]:
            cell.font = Font(bold=True)
        # 逐列寫入 seed-level rows。
        for seed_row in group_rows:
            # Excel 下一列列號。
            excel_row = worksheet.max_row + 1
            # 寫入資料列。
            worksheet.append(seed_row_values(seed_row, excel_row))
            # 若此列解析成功，記錄 row number 供平均分頁引用。
            if seed_row.counts is not None:
                key = (seed_row.result_group, seed_row.summary_file, seed_row.metric)
                summary_metric_rows.setdefault(key, []).append(excel_row)
        # 將百分比欄位設定為兩位小數百分比。
        for percent_column in range(9, 18):
            for cell in worksheet.iter_cols(min_col=percent_column, max_col=percent_column, min_row=2):
                for item in cell:
                    item.number_format = "0.00%"
        # 調整常用欄寬，讓檔案打開時可讀性較好。
        for column, width in {"A": 38, "B": 70, "C": 70, "D": 10, "E": 12, "R": 70, "S": 70, "T": 24}.items():
            worksheet.column_dimensions[column].width = width
    # 建立三次平均彙整分頁。
    summary_sheet = workbook.create_sheet(safe_sheet_name("三次平均彙整", used_sheet_names))
    # 寫入平均分頁表頭。
    summary_sheet.append([
        "result_group",
        "summary_file",
        "metric",
        "seed_count",
        "precision_mean_formula",
        "recall_mean_formula",
        "f1_mean_formula",
        "precision_std_formula",
        "recall_std_formula",
        "f1_std_formula",
        "precision_mean_from_summary",
        "recall_mean_from_summary",
        "f1_mean_from_summary",
        "precision_std_from_summary",
        "recall_std_from_summary",
        "f1_std_from_summary",
    ])
    # 將平均分頁表頭設為粗體。
    for cell in summary_sheet[1]:
        cell.font = Font(bold=True)
    # 逐一處理 summary context。
    for context in contexts:
        # 取得對應的 seed-level sheet 名稱。
        sheet_name = result_to_sheet.get(context.result_group, context.result_group)
        # 逐一輸出五個 metric 的平均列。
        for metric in METRIC_NAMES:
            # 取得此 summary/metric 在 seed 分頁上的三個列號。
            source_rows = summary_metric_rows.get((context.result_group, context.summary_path.name, metric), [])
            # 取得 summary 檔本身解析出的 Mean ± Std。
            parsed_summary = context.summary_means.get(metric)
            # 建立 P 平均公式。
            p_mean_formula = f"=AVERAGE({formula_join(sheet_name, 'I', source_rows)})" if source_rows else ""
            # 建立 R 平均公式。
            r_mean_formula = f"=AVERAGE({formula_join(sheet_name, 'J', source_rows)})" if source_rows else ""
            # 建立 F1 平均公式。
            f1_mean_formula = f"=AVERAGE({formula_join(sheet_name, 'K', source_rows)})" if source_rows else ""
            # 建立 P 標準差公式；使用 STDEV 提高舊版 Excel/WPS/LibreOffice 相容性。
            p_std_formula = f"=STDEV({formula_join(sheet_name, 'I', source_rows)})" if len(source_rows) > 1 else ""
            # 建立 R 標準差公式；STDEV 與 STDEV.S 一樣是樣本標準差。
            r_std_formula = f"=STDEV({formula_join(sheet_name, 'J', source_rows)})" if len(source_rows) > 1 else ""
            # 建立 F1 標準差公式；三列以上才計算。
            f1_std_formula = f"=STDEV({formula_join(sheet_name, 'K', source_rows)})" if len(source_rows) > 1 else ""
            # 寫入三次平均列。
            summary_sheet.append([
                context.result_group,
                context.summary_path.name,
                metric,
                len(source_rows),
                p_mean_formula,
                r_mean_formula,
                f1_mean_formula,
                p_std_formula,
                r_std_formula,
                f1_std_formula,
                parsed_summary.p_mean if parsed_summary else "",
                parsed_summary.r_mean if parsed_summary else "",
                parsed_summary.f1_mean if parsed_summary else "",
                parsed_summary.p_std if parsed_summary else "",
                parsed_summary.r_std if parsed_summary else "",
                parsed_summary.f1_std if parsed_summary else "",
            ])
    # 將平均分頁百分比欄設定為兩位小數百分比。
    for percent_column in range(5, 17):
        for cell in summary_sheet.iter_cols(min_col=percent_column, max_col=percent_column, min_row=2):
            for item in cell:
                item.number_format = "0.00%"
    # 設定平均分頁欄寬。
    for column, width in {"A": 38, "B": 70, "C": 12}.items():
        summary_sheet.column_dimensions[column].width = width
    # 確保輸出資料夾存在。
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # 儲存 Excel 活頁簿。
    workbook.save(output_path)


def write_csv_fallback(output_path: Path, contexts: list[SummaryContext], rows: list[SeedMetricRow], csv_dir: Path | None = None) -> Path:
    """在沒有 openpyxl 時輸出多個 CSV 檔作為 fallback。

    輸入：output_path 是原本希望輸出的 xlsx 路徑，contexts/rows 是彙整資料，csv_dir 是可選輸出資料夾。
    輸出：CSV 資料夾路徑；每個 CSV 對應一個原本的分頁。
    """

    # 若使用者有指定 csv_dir 就使用該資料夾，否則用 xlsx 檔名 stem 建立 fallback 資料夾。
    fallback_dir = csv_dir if csv_dir is not None else output_path.parent / f"{output_path.stem}_csv_sheets"
    # 確保 fallback 資料夾存在。
    fallback_dir.mkdir(parents=True, exist_ok=True)
    # 建立 result_group 到 CSV 檔名 stem 的對照。
    used_names: set[str] = set()
    # 依 results 資料夾分組。
    grouped_rows = group_rows_by_result(rows)
    # 寫出每個 results 分頁的 CSV。
    for result_group, group_rows in grouped_rows.items():
        # 建立安全檔名。
        sheet_name = safe_sheet_name(result_group, used_names)
        # 組合 CSV 檔路徑。
        csv_path = fallback_dir / f"{sheet_name}.csv"
        # 開啟 CSV 檔。
        with csv_path.open("w", encoding="utf-8-sig", newline="") as file_obj:
            # 建立 CSV writer。
            writer = csv.writer(file_obj)
            # 寫入表頭。
            writer.writerow(seed_sheet_headers())
            # 寫入每一列；CSV 中的公式以文字保留。
            for index, seed_row in enumerate(group_rows, start=2):
                writer.writerow(seed_row_values(seed_row, index))
    # 寫出 summary 平均 CSV；CSV 無法可靠跨檔參照，所以放 summary 檔解析值。
    summary_csv_path = fallback_dir / "三次平均彙整.csv"
    # 開啟 summary CSV。
    with summary_csv_path.open("w", encoding="utf-8-sig", newline="") as file_obj:
        # 建立 CSV writer。
        writer = csv.writer(file_obj)
        # 寫入表頭。
        writer.writerow(["result_group", "summary_file", "metric", "precision_mean", "recall_mean", "f1_mean", "precision_std", "recall_std", "f1_std"])
        # 逐一寫入 summary 的 Mean ± Std。
        for context in contexts:
            for metric in METRIC_NAMES:
                parsed_summary = context.summary_means.get(metric)
                if parsed_summary is None:
                    writer.writerow([context.result_group, context.summary_path.name, metric, "", "", "", "", "", ""])
                else:
                    writer.writerow([context.result_group, context.summary_path.name, metric, parsed_summary.p_mean, parsed_summary.r_mean, parsed_summary.f1_mean, parsed_summary.p_std, parsed_summary.r_std, parsed_summary.f1_std])
    # 回傳 fallback 目錄。
    return fallback_dir


def main() -> int:
    """主流程：掃描 summary、解析 counts，最後輸出 xlsx 或 CSV fallback。

    輸入：命令列參數。
    輸出：成功回傳 0；strict 模式下有缺檔或錯誤則回傳 1。
    """

    # 解析命令列參數。
    args = parse_args()
    # 解析 results 資料夾，並處理 t9/t9te1 差異。
    result_dirs = [resolve_result_dir(raw_dir) for raw_dir in args.result_dirs]
    # 找出不存在的 results 資料夾。
    missing_result_dirs = [path for path in result_dirs if not path.is_dir()]
    # 收集 summary 上下文。
    contexts = collect_summary_contexts(result_dirs, args.summary_glob)
    # 建立 seed-level rows。
    rows = build_seed_rows(contexts, args.extra_experiment_roots, args.search_drives)
    # 統計缺失或錯誤列。
    bad_rows = [row for row in rows if row.status != "ok"]
    # 設定輸出路徑。
    output_path = Path(args.output)
    # 如果沒有 summary，就提示使用者。
    if not contexts:
        print("找不到任何 summary 檔，請確認 --result-dirs 與 --summary-glob。")
        return 1
    # 嘗試輸出 xlsx。
    try:
        write_xlsx(output_path, contexts, rows)
        print(f"已輸出 Excel 分頁報表: {output_path}")
        if args.also_csv_dir:
            csv_dir = write_csv_fallback(output_path, contexts, rows, Path(args.also_csv_dir))
            print(f"已同步輸出 CSV 分頁資料夾: {csv_dir}")
    # openpyxl 不存在時輸出 CSV fallback。
    except ModuleNotFoundError as exc:
        if exc.name != "openpyxl":
            raise
        fallback_dir = write_csv_fallback(output_path, contexts, rows)
        print("找不到 openpyxl，無法建立真正有分頁的 .xlsx。")
        print("已改輸出多個 CSV 檔，每個 CSV 對應一個分頁。")
        print(f"CSV fallback 資料夾: {fallback_dir}")
        print("若要輸出單一多分頁 Excel 檔，請先執行: pip install openpyxl")
    # 印出 results 資料夾缺失狀態。
    for missing_dir in missing_result_dirs:
        print(f"警告：找不到 results 資料夾: {missing_dir}")
    # 印出 counts 或實驗資料夾缺失狀態。
    if bad_rows:
        print(f"警告：共有 {len(bad_rows)} 列不是 ok，請查看輸出檔 status 欄位。")
    # strict 模式下只要有缺資料就回傳 1。
    if args.strict and (missing_result_dirs or bad_rows):
        return 1
    # 一般模式成功回傳 0。
    return 0


# 讓此檔案可直接以 python utils/export_counts_summary_workbook.py 執行。
if __name__ == "__main__":
    sys.exit(main())