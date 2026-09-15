#!/usr/bin/env python3
"""計算多個亂數種子下，已選中偽標註文檔的品質指標

本腳本會讀取每個實驗資料夾中的
`pseudo_labeled_samples_fold{fold}_round{round}.json`，
並依照指定的目標元件與統計模式，輸出兩種不同層級的品質結果：

1. 文檔級錯誤率：
   只要某篇文檔在指定目標元件上有任一[MASK]預測錯誤，
   該篇文檔就視為錯誤
2. [MASK]級準確率：
   逐一比較所有局部[MASK]是否正確，
   統計整體正確比例

支援的目標元件如下：
1. emotion：情緒[MASK]
2. cause：原因[MASK]
3. pair：配對[MASK]

本腳本不會重新推論模型，
只利用既有的 pseudo JSON 與 ground truth 欄位完成統計
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from statistics import mean, stdev
from typing import Callable


# 用來從 multi-seed summary 檔案中抓出實際實驗資料夾名稱
SUMMARY_EXPERIMENT_PATTERN = re.compile(r"^\s*-\s*(prompt_ECPE_few_shot_ST_[^\s]+)\s*$")

# 用來從實驗資料夾名稱中取出 seed 編號，例如 seed20、seed42
SEED_PATTERN = re.compile(r"seed(\d+)")


class TargetComponent(str, Enum):
    """可分析的目標元件列舉。

    說明：
    1. EMOTION：每個子句的情緒[MASK]
    2. CAUSE：每個子句的原因[MASK]
    3. PAIR：每個子句的配對[MASK]

    備註：
    列舉值保留英文，是為了維持命令列參數相容性，
    例如 `--target emotion`、`--target cause`、`--target pair`
    """

    EMOTION = "emotion"
    CAUSE = "cause"
    PAIR = "pair"


# 指定每個目標元件在三元組中的位置
# 模板順序固定為：emotion、cause、pair
TARGET_INDEX = {
    TargetComponent.EMOTION: 0,
    TargetComponent.CAUSE: 1,
    TargetComponent.PAIR: 2,
}


# 將英文目標名稱映射成報告中顯示的繁體中文
TARGET_DISPLAY_LABEL = {
    TargetComponent.EMOTION: "情緒",
    TargetComponent.CAUSE: "原因",
    TargetComponent.PAIR: "配對",
}


@dataclass
class RoundResult:
    """描述單一 seed、單一 round 的統計結果

    欄位說明：
    1. seed_name：亂數種子名稱，例如 seed20
    2. round_id：自訓練輪次編號
    3. selected_docs：該輪被選中的未標註文檔數量
    4. all_correct_docs：在指定目標元件上整篇文檔完全正確的文檔數量
    5. error_docs：在指定目標元件上至少有一個[MASK]錯誤的文檔數量
    6. error_rate：文檔級錯誤率，計算式為 error_docs / selected_docs
    7. correct_slots：指定目標元件中，所有局部[MASK]的正確數量
    8. total_slots：指定目標元件中，所有局部[MASK]的總數量
    9. token_accuracy：[MASK]級正確率，計算式為 correct_slots / total_slots
    """

    seed_name: str
    round_id: int
    selected_docs: int
    all_correct_docs: int
    error_docs: int
    error_rate: float
    correct_slots: int
    total_slots: int
    token_accuracy: float


def parse_args() -> argparse.Namespace:
    """解析命令列參數

    輸入參數：
    1. 此函式直接從命令列讀取參數

    主要命令列選項：
    1. --summary：multi-seed summary 檔案路徑
    2. --experiments-root：實驗資料夾的根目錄
    3. --fold：要分析的 fold 編號
    4. --round-start：起始 round
    5. --round-end：結束 round
    6. --target：要分析的目標元件，emotion / cause / pair
    7. --metric：要輸出的指標類型，doc_error / token_accuracy / both
    8. --output：選擇性輸出檔案路徑

    回傳值：
    1. argparse.Namespace：解析後的參數物件
    """

    # 建立命令列解析器，說明本腳本用途
    parser = argparse.ArgumentParser(
        description="計算多個亂數種子下，已選中文檔的文檔級錯誤率與[MASK]級正確率"
    )

    # 指定 multi-seed summary 檔案路徑
    parser.add_argument("--summary", required=True, help="multi-seed 彙整檔案路徑")

    # 指定實際實驗資料夾所在的根目錄
    parser.add_argument(
        "--experiments-root",
        required=True,
        help="包含各 seed 實驗資料夾的根目錄",
    )

    # 指定要分析哪一折
    parser.add_argument("--fold", type=int, required=True, help="要分析的 fold 編號")

    # 指定起始 round，包含此 round
    parser.add_argument("--round-start", type=int, default=1, help="起始 round（包含）")

    # 指定結束 round，包含此 round
    parser.add_argument("--round-end", type=int, default=5, help="結束 round（包含）")

    # 指定要分析 emotion、cause 或 pair
    parser.add_argument(
        "--target",
        type=str,
        default=TargetComponent.PAIR.value,
        choices=[component.value for component in TargetComponent],
        help="要分析的目標元件：emotion、cause 或 pair",
    )

    # 指定輸出哪種統計視角
    parser.add_argument(
        "--metric",
        type=str,
        default="doc_error",
        choices=["doc_error", "token_accuracy", "both"],
        help="要輸出的統計型態：文檔級錯誤率、[MASK]級正確率，或兩者皆輸出",
    )

    # 若有需要，可將報告寫入指定檔案
    parser.add_argument("--output", default=None, help="選擇性輸出報告檔案路徑")

    # 回傳解析完成的參數結果
    return parser.parse_args()


def read_text(path: Path) -> str:
    """以 UTF-8 讀取純文字檔案內容

    輸入參數：
    1. path：要讀取的檔案路徑

    回傳值：
    1. str：檔案完整文字內容
    """

    # 以 UTF-8 讀取文字，若遇到少數異常字元則忽略，避免整體解析失敗
    return path.read_text(encoding="utf-8", errors="ignore")


def extract_experiment_names(summary_path: Path) -> list[str]:
    """從 multi-seed summary 檔案中擷取所有實驗資料夾名稱

    輸入參數：
    1. summary_path：multi-seed summary 檔案路徑

    回傳值：
    1. list[str]：從 summary 中抓出的實驗資料夾名稱列表

    例外：
    1. 若完全找不到任何實驗名稱，會拋出 ValueError
    """

    # 準備一個串列，用來累積所有抓到的實驗名稱
    experiment_names: list[str] = []

    # 逐行讀取 summary 檔案內容
    for line in read_text(summary_path).splitlines():
        # 嘗試用正則表達式匹配實驗資料夾名稱
        match = SUMMARY_EXPERIMENT_PATTERN.match(line)

        # 若匹配成功，就把資料夾名稱加入結果列表
        if match:
            experiment_names.append(match.group(1))

    # 若完全沒有抓到任何實驗資料夾，表示 summary 格式可能不符合預期
    if not experiment_names:
        raise ValueError(f"在 summary 檔案中找不到任何實驗資料夾名稱：{summary_path}")

    # 回傳所有抓到的實驗名稱
    return experiment_names


def extract_seed(experiment_name: str) -> str:
    """從實驗資料夾名稱中擷取 seed 字串

    輸入參數：
    1. experiment_name：單一實驗資料夾名稱

    回傳值：
    1. str：例如 seed20、seed42 的字串

    例外：
    1. 若資料夾名稱中找不到 seed 編號，會拋出 ValueError
    """

    # 在實驗名稱中尋找 seed 數字
    match = SEED_PATTERN.search(experiment_name)

    # 如果找不到，代表資料夾命名不符合預期
    if match is None:
        raise ValueError(f"無法從實驗名稱中解析 seed 編號：{experiment_name}")

    # 將數字重新組成 seedXX 的格式後回傳
    return f"seed{match.group(1)}"


def seed_sort_key(seed_name: str) -> tuple[int, str]:
    """產生 seed 排序用的鍵值

    輸入參數：
    1. seed_name：例如 seed20、seed42 的字串

    回傳值：
    1. tuple[int, str]：
       第一個元素是 seed 的數值部分，第二個元素是原字串

    用途：
    1. 讓 seed20、seed42、seed60 能按照數字大小排序
    2. 若某名稱無法解析出數字，則排在最後面
    """

    # 嘗試在 seed 名稱中找出數字部分
    match = SEED_PATTERN.search(seed_name)

    # 若成功解析，就回傳數字與原字串組成的排序鍵
    if match:
        return (int(match.group(1)), seed_name)

    # 若失敗，使用很大的數字讓它排序到最後
    return (10**9, seed_name)


def resolve_experiment_map(summary_path: Path, experiments_root: Path) -> dict[str, Path]:
    """建立 seed 名稱到實驗資料夾路徑的對照表

    輸入參數：
    1. summary_path：multi-seed summary 檔案路徑
    2. experiments_root：所有實驗資料夾所在的根目錄

    回傳值：
    1. dict[str, Path]：鍵為 seed 名稱，值為該 seed 對應的實驗資料夾路徑

    例外：
    1. 若某個實驗資料夾在磁碟上不存在，會拋出 FileNotFoundError
    2. 若 summary 中出現重複 seed，會拋出 ValueError
    """

    # 建立最終要回傳的 seed -> 資料夾路徑對照表
    experiment_map: dict[str, Path] = {}

    # 從 summary 中逐一取出實驗名稱
    for experiment_name in extract_experiment_names(summary_path):
        # 解析出該實驗的 seed 名稱
        seed_name = extract_seed(experiment_name)

        # 將實驗名稱接到根目錄下，形成完整資料夾路徑
        experiment_dir = experiments_root / experiment_name

        # 若資料夾不存在，就立即拋出錯誤，避免後續分析得到不完整結果
        if not experiment_dir.is_dir():
            raise FileNotFoundError(f"找不到實驗資料夾：{experiment_dir}")

        # 若同一個 seed 已經出現過，代表 summary 有重複資料
        if seed_name in experiment_map:
            raise ValueError(f"summary 中出現重複 seed：{seed_name}")

        # 將 seed 與資料夾路徑記錄起來
        experiment_map[seed_name] = experiment_dir

    # 回傳完整對照表
    return experiment_map


def find_single_pseudo_results_dir(experiment_dir: Path) -> Path:
    """在單一實驗資料夾下找出唯一的 pseudo_results 子目錄

    輸入參數：
    1. experiment_dir：單一 seed 的實驗資料夾路徑

    回傳值：
    1. Path：唯一的 pseudo_results_* 目錄路徑

    例外：
    1. 若完全找不到 pseudo_results 目錄，會拋出 FileNotFoundError
    2. 若找到多個 pseudo_results 目錄，會拋出 ValueError
    """

    # 收集所有符合 pseudo_results_* 模式的子目錄
    pseudo_dirs = sorted(path for path in experiment_dir.glob("pseudo_results_*") if path.is_dir())

    # 若一個都沒有，表示該實驗資料夾缺少必要輸出
    if not pseudo_dirs:
        raise FileNotFoundError(f"在實驗資料夾中找不到 pseudo_results_* 目錄：{experiment_dir}")

    # 若超過一個，代表資料夾結構不符合本腳本預期
    if len(pseudo_dirs) > 1:
        raise ValueError(
            f"實驗資料夾中存在多個 pseudo_results_* 目錄：{experiment_dir}，"
            f"找到的目錄為 {[path.name for path in pseudo_dirs]}"
        )

    # 回傳唯一找到的 pseudo_results 目錄
    return pseudo_dirs[0]


def load_selected_doc_count(json_path: Path) -> int:
    """讀取 selected pseudo JSON，回傳文檔數量

    輸入參數：
    1. json_path：selected pseudo JSON 檔案路徑

    回傳值：
    1. int：JSON 中記錄的文檔筆數

    例外：
    1. 若 JSON 最外層不是 list，會拋出 ValueError

    備註：
    1. 這個函式主要作為簡單工具保留，
       目前主流程實際上使用 `load_selected_records` 讀入完整資料
    """

    # 將 JSON 文字載入為 Python 物件
    records = json.loads(read_text(json_path))

    # 預期最外層必須是 list，否則格式不符
    if not isinstance(records, list):
        raise ValueError(f"selected pseudo JSON 的最外層不是列表：{json_path}")

    # 回傳列表長度，也就是文檔筆數
    return len(records)


def load_selected_records(json_path: Path) -> list[dict]:
    """讀取 selected pseudo JSON，回傳完整記錄列表

    輸入參數：
    1. json_path：selected pseudo JSON 檔案路徑

    回傳值：
    1. list[dict]：每筆文檔對應的 JSON 記錄列表

    例外：
    1. 若 JSON 最外層不是 list，會拋出 ValueError
    """

    # 讀取並解析 JSON 檔案內容
    records = json.loads(read_text(json_path))

    # 預期最外層為列表，因為每個元素代表一篇文檔
    if not isinstance(records, list):
        raise ValueError(f"selected pseudo JSON 的最外層不是列表：{json_path}")

    # 回傳完整記錄列表，供後續逐筆分析
    return records


def compute_target_stats(
    records: list[dict],
    target: TargetComponent,
    json_path: Path,
) -> tuple[int, int, int]:
    """計算指定目標元件的文檔級與[MASK]統計量

    輸入參數：
    1. records：selected pseudo JSON 讀出的完整文檔記錄列表
    2. target：要分析的目標元件，emotion / cause / pair
    3. json_path：目前分析中的 JSON 檔案路徑，用於錯誤訊息顯示

    回傳值：
    1. tuple[int, int, int]：
       - 第 1 個值：all_correct_docs，整篇文檔在該目標元件完全正確的文檔數
       - 第 2 個值：correct_slots，局部[MASK]預測正確的總數
       - 第 3 個值：total_slots，局部[MASK]總數

    例外：
    1. 若記錄中缺少 `pseudo_label_ids` 或 `gt_label_ids`，會拋出 ValueError
    2. 若偽標籤與 ground truth 長度不同，會拋出 ValueError
    3. 若長度不能被 3 整除，表示不符合 emotion/cause/pair 三元組結構，會拋出 ValueError
    """

    # 取得目標元件在三元組中的索引位置
    target_index = TARGET_INDEX[target]

    # 初始化三個統計量
    all_correct_docs = 0
    correct_slots = 0
    total_slots = 0

    # 逐篇文檔處理
    for record in records:
        # 取出偽標籤序列與 ground truth 序列
        pseudo_label_ids = record.get("pseudo_label_ids")
        gt_label_ids = record.get("gt_label_ids")

        # 若缺少關鍵欄位，就直接報錯，避免靜默產生錯誤統計
        if not isinstance(pseudo_label_ids, list) or not isinstance(gt_label_ids, list):
            raise ValueError(
                "selected pseudo JSON 記錄缺少 pseudo_label_ids 或 gt_label_ids："
                f"{json_path} doc_id={record.get('doc_id')}"
            )

        # 偽標籤與 ground truth 長度必須一致，否則無法逐位比較
        if len(pseudo_label_ids) != len(gt_label_ids):
            raise ValueError(
                "selected pseudo JSON 記錄中的標籤長度不一致："
                f"{json_path} doc_id={record.get('doc_id')} "
                f"pseudo={len(pseudo_label_ids)} gt={len(gt_label_ids)}"
            )

        # 由於模板固定為 emotion/cause/pair 三元組，長度必須能被 3 整除
        if len(pseudo_label_ids) % 3 != 0:
            raise ValueError(
                "selected pseudo JSON 記錄長度無法被 3 整除，不符合三元組結構："
                f"{json_path} doc_id={record.get('doc_id')} len={len(pseudo_label_ids)}"
            )

        # 只擷取指定目標元件的位置，例如只取 emotion 或只取 cause
        pseudo_target_ids = pseudo_label_ids[target_index::3]
        gt_target_ids = gt_label_ids[target_index::3]

        # 累加這篇文檔在該目標元件上的[MASK]總數
        total_slots += len(gt_target_ids)

        # 逐一比較每個[MASK]是否正確，累加正確數量
        correct_slots += sum(
            1 for pseudo_id, gt_id in zip(pseudo_target_ids, gt_target_ids) if pseudo_id == gt_id
        )

        # 只有在所有該目標元件[MASK]都正確時，這篇文檔才算全對
        if pseudo_target_ids == gt_target_ids:
            all_correct_docs += 1

    # 回傳文檔級與[MASK]級統計結果
    return all_correct_docs, correct_slots, total_slots


def collect_round_results(
    summary_path: Path,
    experiments_root: Path,
    fold: int,
    round_start: int,
    round_end: int,
    target: TargetComponent,
) -> dict[int, list[RoundResult]]:
    """蒐集指定 fold 與 round 範圍內的統計結果

    輸入參數：
    1. summary_path：multi-seed summary 檔案路徑
    2. experiments_root：所有實驗資料夾的根目錄
    3. fold：要分析的 fold 編號
    4. round_start：起始 round
    5. round_end：結束 round
    6. target：要分析的目標元件

    回傳值：
    1. dict[int, list[RoundResult]]：
       鍵是 round 編號，值是該 round 下各 seed 的統計結果列表
    """

    # 先解析出每個 seed 對應的實驗資料夾
    experiment_map = resolve_experiment_map(summary_path, experiments_root)

    # 以 round 為鍵，累積所有 seed 的結果
    round_results: dict[int, list[RoundResult]] = {}

    # 依照 seed 編號順序逐一處理每個實驗
    for seed_name in sorted(experiment_map.keys(), key=seed_sort_key):
        # 找出該 seed 實驗中唯一的 pseudo_results 目錄
        pseudo_dir = find_single_pseudo_results_dir(experiment_map[seed_name])

        # 逐一分析指定範圍內的 round
        for round_id in range(round_start, round_end + 1):
            # 組出該 fold / round 的 selected pseudo JSON 路徑
            json_path = pseudo_dir / f"pseudo_labeled_samples_fold{fold}_round{round_id}.json"

            # 若檔案不存在，表示該 round 沒有可分析資料，直接跳過
            if not json_path.is_file():
                continue

            # 讀取完整文檔記錄
            records = load_selected_records(json_path)

            # 被選中文檔數就是記錄數量
            selected_docs = len(records)

            # 計算文檔級與[MASK]級統計量
            all_correct_docs, correct_slots, total_slots = compute_target_stats(records, target, json_path)

            # 保險檢查：完全正確文檔數不可能大於總文檔數
            if all_correct_docs > selected_docs:
                raise ValueError(
                    "整篇完全正確的文檔數不可能大於被選中文檔數："
                    f"seed={seed_name}, fold={fold}, round={round_id}, "
                    f"correct={all_correct_docs}, selected={selected_docs}"
                )

            # 文檔級錯誤數 = 總文檔數 - 完全正確文檔數
            error_docs = selected_docs - all_correct_docs

            # 文檔級錯誤率；若 selected_docs 為 0，則定義為 0.0
            error_rate = (error_docs / selected_docs) if selected_docs else 0.0

            # [MASK]級正確率；若 total_slots 為 0，則定義為 0.0
            token_accuracy = (correct_slots / total_slots) if total_slots else 0.0

            # 把這個 seed / round 的結果封裝成 RoundResult，放入 round_results
            round_results.setdefault(round_id, []).append(
                RoundResult(
                    seed_name=seed_name,
                    round_id=round_id,
                    selected_docs=selected_docs,
                    all_correct_docs=all_correct_docs,
                    error_docs=error_docs,
                    error_rate=error_rate,
                    correct_slots=correct_slots,
                    total_slots=total_slots,
                    token_accuracy=token_accuracy,
                )
            )

    # 回傳所有 round 的蒐集結果
    return round_results


def format_percent(value: float) -> str:
    """把 0 到 1 的比例轉成百分比字串

    輸入參數：
    1. value：比例值，例如 0.9314

    回傳值：
    1. str：百分比字串，例如 93.14%
    """

    # 乘上 100 後格式化到小數點後兩位
    return f"{value * 100:.2f}%"


def build_metric_section(
    round_results: dict[int, list[RoundResult]],
    seed_names: list[str],
    title: str,
    value_getter: Callable[[RoundResult], float],
    numerator_getter: Callable[[RoundResult], int],
    denominator_getter: Callable[[RoundResult], int],
    mean_header: str,
    std_header: str,
) -> list[str]:
    """建立單一指標區塊的報告文字

    輸入參數：
    1. round_results：各 round 的統計結果
    2. seed_names：要顯示的 seed 名稱順序
    3. title：本區塊標題。
    4. value_getter：從 RoundResult 取出主要比例值的函式
    5. numerator_getter：從 RoundResult 取出分子數值的函式
    6. denominator_getter：從 RoundResult 取出分母數值的函式
    7. mean_header：平均欄位標題
    8. std_header：標準差欄位標題

    回傳值：
    1. list[str]：可直接拼接進最終報告的多行文字列表
    """

    # 初始化區塊文字列表
    lines: list[str] = []

    # 組出表頭欄位名稱
    header = ["輪次"] + seed_names + [mean_header, std_header]

    # 先放入區塊標題與表頭
    lines.append(title)
    lines.append("  " + "\t".join(header))

    # 依 round 順序輸出每一列
    for round_id in sorted(round_results.keys()):
        # 先把該 round 的結果依 seed 建成字典，方便查表
        results_by_seed = {result.seed_name: result for result in round_results[round_id]}

        # 每列第一格先放 round 名稱，例如 R1
        row_cells = [f"R{round_id}"]

        # 用來收集該 round 可用的比例值，以便計算平均與標準差
        available_values: list[float] = []

        # 依指定順序逐一填入每個 seed 的欄位
        for seed_name in seed_names:
            # 若某 seed 在該 round 沒資料，就填 --
            result = results_by_seed.get(seed_name)
            if result is None:
                row_cells.append("--")
                continue

            # 取出主要比例值、分子與分母
            value = value_getter(result)
            numerator = numerator_getter(result)
            denominator = denominator_getter(result)

            # 把比例值加入列表，供後續計算平均與標準差
            available_values.append(value)

            # 以「百分比 (分子/分母)」格式輸出該 seed 的結果
            row_cells.append(f"{format_percent(value)} ({numerator}/{denominator})")

        # 若這一輪至少有一個 seed 有資料，就計算平均與標準差
        if available_values:
            row_cells.append(format_percent(mean(available_values)))
            row_cells.append(format_percent(stdev(available_values)) if len(available_values) > 1 else "0.00%")
        else:
            # 若完全沒有可用資料，就以 -- 補齊
            row_cells.extend(["--", "--"])

        # 將這一列加入區塊文字
        lines.append("  " + "\t".join(row_cells))

    # 回傳整個區塊的文字內容
    return lines


def build_per_seed_average_section(
    round_results: dict[int, list[RoundResult]],
    seed_names: list[str],
    value_getter: Callable[[RoundResult], float],
    title: str,
) -> list[str]:
    """建立各 seed 跨 round 平均值的報告區塊

    輸入參數：
    1. round_results：各 round 的統計結果
    2. seed_names：要顯示的 seed 順序
    3. value_getter：從 RoundResult 取出要平均的比例值函式
    4. title：本區塊標題

    回傳值：
    1. list[str]：可直接拼接進最終報告的多行文字列表
    """

    # 區塊第一行先保留空白，讓視覺上與前一段分開
    lines = ["", title]

    # 逐一計算每個 seed 在所有可用 round 上的平均值
    for seed_name in seed_names:
        # 收集這個 seed 在所有 round 的數值
        values = [
            value_getter(result)
            for results in round_results.values()
            for result in results
            if result.seed_name == seed_name
        ]

        # 若這個 seed 完全沒有資料，就輸出 --
        if not values:
            lines.append(f"  - {seed_name}: --")
            continue

        # 否則輸出平均百分比
        lines.append(f"  - {seed_name}: {format_percent(mean(values))}")

    # 回傳整個區塊內容
    return lines


def build_report(
    summary_path: Path,
    experiments_root: Path,
    fold: int,
    round_results: dict[int, list[RoundResult]],
    target: TargetComponent,
    metric: str,
) -> str:
    """建立完整文字報告。

    輸入參數：
    1. summary_path：multi-seed summary 檔案路徑
    2. experiments_root：實驗根目錄
    3. fold：目前分析的 fold 編號
    4. round_results：各 round 的統計結果
    5. target：目標元件
    6. metric：要輸出的指標模式，doc_error、token_accuracy 或 both

    回傳值：
    1. str：完整的報告文字內容
    """

    # 取得所有出現過的 seed 名稱，並依 seed 編號排序
    seed_names = sorted(
        {result.seed_name for results in round_results.values() for result in results},
        key=seed_sort_key,
    )

    # 取得目前目標元件的中文顯示名稱
    target_label = TARGET_DISPLAY_LABEL[target]

    # 初始化報告文字列表
    lines: list[str] = []

    # 報告標題區
    lines.append("=" * 78)
    lines.append(f"已選中文檔 {target_label} 指標報告")
    lines.append("=" * 78)
    lines.append(f"彙整檔案：{summary_path}")
    lines.append(f"實驗根目錄：{experiments_root}")
    lines.append(f"分析 fold：{fold}")
    lines.append(f"目標元件：{target_label}（命令列值：{target.value}）")
    lines.append("")

    # 說明區，交代每個指標的意義
    lines.append("指標定義：")
    lines.append("  - 單位：已選中的未標註文檔")
    lines.append(f"  - 文檔級正確：某篇文檔的所有 {target_label} [MASK]都正確，才算整篇正確")
    lines.append("  - 文檔級錯誤率 = 錯誤文檔數 / 已選中文檔數")
    lines.append(f"  - [MASK]級正確率 = 正確 {target_label} [MASK]數 / 總 {target_label} [MASK]數")
    lines.append("")

    # 若使用者要求輸出文檔級錯誤率，就建立對應區塊
    if metric in ("doc_error", "both"):
        lines.extend(
            build_metric_section(
                round_results=round_results,
                seed_names=seed_names,
                title=f"各 round 的已選中文檔 {target_label} 文檔級錯誤率：",
                value_getter=lambda result: result.error_rate,
                numerator_getter=lambda result: result.error_docs,
                denominator_getter=lambda result: result.selected_docs,
                mean_header="平均錯誤率",
                std_header="錯誤率標準差",
            )
        )
        lines.extend(
            build_per_seed_average_section(
                round_results=round_results,
                seed_names=seed_names,
                value_getter=lambda result: result.error_rate,
                title="各 seed 在可用 round 上的文檔級錯誤率平均：",
            )
        )

    # 若同時輸出兩種指標，中間多留一個空白段落
    if metric == "both":
        lines.append("")

    # 若使用者要求輸出[MASK]級正確率，就建立對應區塊
    if metric in ("token_accuracy", "both"):
        lines.extend(
            build_metric_section(
                round_results=round_results,
                seed_names=seed_names,
                title=f"各 round 的已選中文檔 {target_label} [MASK]級正確率：",
                value_getter=lambda result: result.token_accuracy,
                numerator_getter=lambda result: result.correct_slots,
                denominator_getter=lambda result: result.total_slots,
                mean_header="平均[MASK]正確率",
                std_header="[MASK]正確率標準差",
            )
        )
        lines.extend(
            build_per_seed_average_section(
                round_results=round_results,
                seed_names=seed_names,
                value_getter=lambda result: result.token_accuracy,
                title="各 seed 在可用 round 上的[MASK]級正確率平均：",
            )
        )

    # 把所有段落合併成單一字串並回傳
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    """主程式入口

    輸入參數：
    1. 此函式會自行呼叫 `parse_args()` 解析命令列參數

    回傳值：
    1. 結果會直接印出到標準輸出，必要時也會寫入檔案

    主要流程：
    1. 解析命令列參數
    2. 檢查輸入路徑與 round 範圍是否合法
    3. 蒐集各 round 統計結果
    4. 建立並輸出報告
    5. 若指定 `--output`，則一併將報告寫入檔案
    """

    # 先解析所有命令列參數
    args = parse_args()

    # 將字串路徑轉成 Path 物件，方便後續檔案操作
    summary_path = Path(args.summary)
    experiments_root = Path(args.experiments_root)

    # 驗證 summary 檔案是否存在
    if not summary_path.is_file():
        raise FileNotFoundError(f"找不到 summary 檔案：{summary_path}")

    # 驗證實驗根目錄是否存在
    if not experiments_root.is_dir():
        raise FileNotFoundError(f"找不到實驗根目錄：{experiments_root}")

    # 驗證 round 起訖範圍是否合理
    if args.round_start > args.round_end:
        raise ValueError("round_start 不可大於 round_end")

    # 蒐集指定 fold 與 round 範圍內的統計結果
    round_results = collect_round_results(
        summary_path=summary_path,
        experiments_root=experiments_root,
        fold=args.fold,
        round_start=args.round_start,
        round_end=args.round_end,
        target=TargetComponent(args.target),
    )

    # 若完全沒有任何結果，表示輸入的 fold 或 round 可能不正確
    if not round_results:
        raise RuntimeError(
            "找不到任何 round 結果，請檢查 fold 編號、round 範圍、summary 檔案與實驗根目錄是否正確"
        )

    # 依照使用者指定的模式建立完整報告內容
    report = build_report(
        summary_path=summary_path,
        experiments_root=experiments_root,
        fold=args.fold,
        round_results=round_results,
        target=TargetComponent(args.target),
        metric=args.metric,
    )

    # 將報告輸出到終端
    print(report, end="")

    # 若使用者有指定輸出檔案，就將報告寫入磁碟
    if args.output:
        output_path = Path(args.output)

        # 若父資料夾不存在，就先建立
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 以 UTF-8 寫入報告內容
        output_path.write_text(report, encoding="utf-8")

        # 額外在終端提示輸出位置
        print(f"\n報告已寫入：{output_path}")


# 只有在直接執行此腳本時，才啟動主程式
if __name__ == "__main__":
    main()