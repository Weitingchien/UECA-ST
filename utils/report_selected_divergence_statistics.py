#!/usr/bin/env python3
"""統計指定 multi-seed summary 中 selected 文檔的散度分佈與文檔與真實答案完全相同的數量"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, median


MATCH_MODE_ALL = "all"
MATCH_MODE_CAUSE_PAIR = "cause_pair"
MATCH_MODE_PAIR = "pair"


# 這個正則表達式用來從 summary.txt 的條列內容中抓出實驗資料夾名稱
SUMMARY_EXPERIMENT_PATTERN = re.compile(r"^\s*-\s*(prompt_ECPE_few_shot_ST_[^\s]+)\s*$")

# 這個正則表達式用來從實驗名稱中抓出 seed 編號，例如 seed20、seed42
SEED_PATTERN = re.compile(r"seed(\d+)")


@dataclass
class DocRecord:
    """保存單一未標註文檔在某一個 seed/fold/round 單位中的必要資訊

    參數說明:
    - doc_id:
      文檔編號，型別是字串
      例如: "1713"
    - divergence_score:
      該文檔在 NeST divergence 檔中的散度值，型別是浮點數
      例如: 0.00021834703511558473
    - exact_match:
      這篇文檔的 pseudo_label_tokens 是否與 gt_label_tokens 完全一致，型別是布林值
      例如: True 或 False
    - selected:
      這篇文檔是否在該 round 被選入自訓練，型別是布林值
      例如: True 或 False
    """

    doc_id: str
    divergence_score: float
    exact_match: bool
    selected: bool


@dataclass
class StatsBlock:
    """保存一組文檔的統計結果

    欄位說明:
    - count:
      這一組文檔的數量，型別是整數
    - exact_count:
      這一組文檔中，pseudo_label_tokens 與 gt_label_tokens 完全一致的數量，型別是整數
    - mean_score:
      divergence 的平均值
      如果這一組是空的，則為 None
    - median_score:
      divergence 的中位數
      如果這一組是空的，則為 None
    """

    count: int
    exact_count: int
    mean_score: float | None
    median_score: float | None

    @property
    def exact_rate(self) -> float:
        """回傳文檔其偽標籤與真實答案完全相同的比例

        回傳值:
        - 如果 count == 0，回傳 0.0
        - 否則回傳 exact_count / count，型別是浮點數
        """

        return 0.0 if self.count == 0 else self.exact_count / self.count


@dataclass
class ScoreBucket:
    """累積一組文檔的散度與 exact-match 結果，方便之後一次做統計"""

    # scores 會保存這組文檔的所有散度值
    scores: list[float] = field(default_factory=list)

    # exact_count 會保存這組文檔中全文全對的數量
    exact_count: int = 0

    def add_records(self, records: list[DocRecord]) -> None:
        """把多筆 DocRecord 加進目前這個 bucket

        參數:
        - records:
          DocRecord 的串列
          例如: [DocRecord(...), DocRecord(...)]

        回傳值:
        - None
        """

        for record in records:
            self.scores.append(record.divergence_score)
            if record.exact_match:
                self.exact_count += 1

    def summarize(self) -> StatsBlock:
        """把目前累積的資料整理成 StatsBlock

        回傳值:
        - StatsBlock(count=..., exact_count=..., mean_score=..., median_score=...)
        """

        if not self.scores:
            return StatsBlock(count=0, exact_count=0, mean_score=None, median_score=None)

        return StatsBlock(
            count=len(self.scores),
            exact_count=self.exact_count,
            mean_score=mean(self.scores),
            median_score=median(self.scores),
        )


@dataclass
class GroupAccumulator:
    """同時累積 selected 與 low/middle/high 三段統計的容器"""

    # unit_count 保存實際累積了多少個 seed/fold/round 單位，避免把缺失 round 當成 0 納入平均
    unit_count: int = 0

    # selected_all 保存「所有被 selected 的文檔」的分數與全文全對數量
    selected_all: ScoreBucket = field(default_factory=ScoreBucket)

    # selected_low 保存「被 selected 且位於 low divergence 區間」的文檔統計
    selected_low: ScoreBucket = field(default_factory=ScoreBucket)

    # selected_middle 保存「被 selected 且位於 middle 區間」的文檔統計
    selected_middle: ScoreBucket = field(default_factory=ScoreBucket)

    # selected_high 保存「被 selected 且位於 high divergence 區間」的文檔統計
    selected_high: ScoreBucket = field(default_factory=ScoreBucket)

    # low_all 保存整個 low divergence 區間的所有文檔統計，不限 selected
    low_all: ScoreBucket = field(default_factory=ScoreBucket)

    # middle_all 保存整個 middle 區間的所有文檔統計，不限 selected
    middle_all: ScoreBucket = field(default_factory=ScoreBucket)

    # high_all 保存整個 high divergence 區間的所有文檔統計，不限 selected
    high_all: ScoreBucket = field(default_factory=ScoreBucket)

    def add_unit(
        self,
        records: list[DocRecord],
        low_records: list[DocRecord],
        middle_records: list[DocRecord],
        high_records: list[DocRecord],
    ) -> None:
        """把一個 seed/fold/round 單位的資料累積進來

        參數:
        - records:
          該單位的所有文檔
        - low_records:
          排序後前 k 筆 low divergence 文檔
        - middle_records:
          排序後介於 low 與 high 之間的文檔
        - high_records:
          排序後最後 k 筆 high divergence 文檔

        回傳值:
        - None
        """

        self.unit_count += 1

        selected_records = [record for record in records if record.selected]
        selected_low_records = [record for record in low_records if record.selected]
        selected_middle_records = [record for record in middle_records if record.selected]
        selected_high_records = [record for record in high_records if record.selected]

        self.selected_all.add_records(selected_records)
        self.selected_low.add_records(selected_low_records)
        self.selected_middle.add_records(selected_middle_records)
        self.selected_high.add_records(selected_high_records)

        self.low_all.add_records(low_records)
        self.middle_all.add_records(middle_records)
        self.high_all.add_records(high_records)


@dataclass
class UnitResult:
    """保存單一 seed/fold/round 單位的最終統計結果"""

    seed_name: str
    fold_id: int
    round_id: int
    selected_all: StatsBlock
    selected_low: StatsBlock
    selected_middle: StatsBlock
    selected_high: StatsBlock
    low_all: StatsBlock
    middle_all: StatsBlock
    high_all: StatsBlock


def parse_args() -> argparse.Namespace:
    """解析命令列參數

    參數來源:
    - 這個函式不直接接收 Python 參數
    - 它會從命令列讀取使用者輸入

    回傳值:
    - argparse.Namespace
      內含以下欄位:
      - summary: 單一 summary.txt 路徑字串
      - experiments_root: 實驗根目錄字串
      - fold_start: 起始 fold，整數
      - fold_end: 結束 fold，整數
      - round_start: 起始 round，整數
      - round_end: 結束 round，整數
      - quantile: low/high 區間比例，浮點數，例如 0.1
      - output_dir: 輸出資料夾路徑字串
    """

    parser = argparse.ArgumentParser(
        description=(
            "Compute selected divergence statistics, selected exact-match counts, "
            "and low/middle/high divergence summaries for one multi-seed summary file."
        )
    )
    parser.add_argument("--summary", required=True, help="One multi-seed summary file")
    parser.add_argument("--experiments-root", required=True, help="Experiments root directory")
    parser.add_argument("--fold-start", type=int, default=1, help="Start fold (inclusive)")
    parser.add_argument("--fold-end", type=int, default=10, help="End fold (inclusive)")
    parser.add_argument("--round-start", type=int, default=1, help="Start round (inclusive)")
    parser.add_argument("--round-end", type=int, default=5, help="End round (inclusive)")
    parser.add_argument(
        "--quantile",
        type=float,
        default=0.10,
        help="Bottom/top quantile used to define low/high groups (default: 0.10)",
    )
    parser.add_argument(
        "--output-dir",
        default="png/selected_divergence_statistics",
        help="Directory used to save the generated txt report",
    )
    parser.add_argument(
        "--debug-selected-rounds",
        type=int,
        nargs="*",
        default=[],
        help=(
            "Print per-seed/fold selected-doc divergence details for the specified rounds "
            "(for example: --debug-selected-rounds 1)"
        ),
    )
    parser.add_argument(
        "--skip-missing-rounds",
        action="store_true",
        help=(
            "Skip fold/round units whose divergence or prediction files are missing. "
            "Useful when self-training stops early and later rounds are not generated."
        ),
    )
    parser.add_argument(
        "--match-mode",
        choices=(MATCH_MODE_ALL, MATCH_MODE_CAUSE_PAIR, MATCH_MODE_PAIR),
        default=MATCH_MODE_ALL,
        help=(
            "How to decide exact match. "
            "'all' requires [MASK_e]/[MASK_c]/[MASK_p] all correct; "
            "'cause_pair' only checks [MASK_c] and [MASK_p], ignoring [MASK_e]; "
            "'pair' only checks [MASK_p], ignoring [MASK_e] and [MASK_c]."
        ),
    )
    return parser.parse_args()


def read_text(path: Path) -> str:
    """讀取文字檔內容並回傳字串"""

    return path.read_text(encoding="utf-8", errors="ignore")


def read_json(path: Path):
    """讀取 JSON 檔後回傳 Python 物件"""

    return json.loads(read_text(path))


def extract_experiment_names(summary_path: Path) -> list[str]:
    """從 summary.txt 中抽出三個 seed 對應的實驗資料夾名稱"""

    experiment_names: list[str] = []
    for line in read_text(summary_path).splitlines():
        match = SUMMARY_EXPERIMENT_PATTERN.match(line)
        if match:
            experiment_names.append(match.group(1))

    if not experiment_names:
        raise ValueError(f"No experiment directories found in summary: {summary_path}")

    return experiment_names


def extract_seed(experiment_name: str) -> str:
    """從實驗資料夾名稱中抽出 seed 名稱，例如 seed20"""

    match = SEED_PATTERN.search(experiment_name)
    if match is None:
        raise ValueError(f"Cannot parse seed from experiment name: {experiment_name}")
    return f"seed{match.group(1)}"


def seed_sort_key(seed_name: str) -> tuple[int, str]:
    """讓 seed20、seed42、seed60 能用數字順序排序"""

    match = SEED_PATTERN.search(seed_name)
    if match:
        return (int(match.group(1)), seed_name)
    return (10**9, seed_name)


def output_prefix_from_summary(summary_path: Path) -> str:
    """回傳輸出檔案用的前綴

    注意:
    - 這裡只去掉 `_summary.txt`
    - 因此輸出前綴會保留 `_CE`，符合使用者要求
    """

    name = summary_path.name
    if name.endswith("_summary.txt"):
        return name[: -len("_summary.txt")]
    return summary_path.stem


def resolve_experiment_map(summary_path: Path, experiments_root: Path) -> dict[str, Path]:
    """把 summary.txt 中的三個實驗資料夾解析成 `{seed_name: experiment_dir}`"""

    experiment_map: dict[str, Path] = {}

    for experiment_name in extract_experiment_names(summary_path):
        seed_name = extract_seed(experiment_name)
        experiment_dir = experiments_root / experiment_name

        if not experiment_dir.is_dir():
            raise FileNotFoundError(f"Experiment directory not found: {experiment_dir}")

        if seed_name in experiment_map:
            raise ValueError(f"Duplicate seed in summary: {seed_name}")

        experiment_map[seed_name] = experiment_dir

    return experiment_map


def find_single_pseudo_results_dir(experiment_dir: Path) -> Path:
    """找到該實驗資料夾下唯一的 `pseudo_results_*` 資料夾"""

    pseudo_dirs = sorted(path for path in experiment_dir.glob("pseudo_results_*") if path.is_dir())

    if not pseudo_dirs:
        raise FileNotFoundError(f"No pseudo_results_* directory found under: {experiment_dir}")

    if len(pseudo_dirs) > 1:
        raise ValueError(
            f"Multiple pseudo_results_* directories found under {experiment_dir}: "
            f"{[path.name for path in pseudo_dirs]}"
        )

    return pseudo_dirs[0]


def _extract_tokens_for_match_mode(
    pseudo_label_tokens: list,
    gt_label_tokens: list,
    match_mode: str,
) -> tuple[list, list]:
    """依據比對模式取出需要檢查的 token 序列。"""

    if match_mode == MATCH_MODE_ALL:
        return pseudo_label_tokens, gt_label_tokens

    if match_mode == MATCH_MODE_CAUSE_PAIR:
        selected_indices = [index for index in range(len(gt_label_tokens)) if index % 3 in (1, 2)]
        return (
            [pseudo_label_tokens[index] for index in selected_indices],
            [gt_label_tokens[index] for index in selected_indices],
        )

    if match_mode == MATCH_MODE_PAIR:
        selected_indices = [index for index in range(len(gt_label_tokens)) if index % 3 == 2]
        return (
            [pseudo_label_tokens[index] for index in selected_indices],
            [gt_label_tokens[index] for index in selected_indices],
        )

    raise ValueError(f"Unsupported match mode: {match_mode}")


def describe_match_mode(match_mode: str) -> str:
    """回傳報告與 log 可讀的比對模式描述"""

    if match_mode == MATCH_MODE_ALL:
        return "pseudo_label_tokens 與 gt_label_tokens 全部位置完全一致"
    if match_mode == MATCH_MODE_CAUSE_PAIR:
        return "只檢查 [MASK_c] 與 [MASK_p] 是否與 gt_label_tokens 一致，忽略 [MASK_e]"
    if match_mode == MATCH_MODE_PAIR:
        return "只檢查 [MASK_p] 是否與 gt_label_tokens 一致，忽略 [MASK_e] 與 [MASK_c]"
    raise ValueError(f"Unsupported match mode: {match_mode}")


def exact_match_from_prediction(record: dict, path: Path, match_mode: str) -> bool:
    """判斷某篇文檔的偽標籤是否符合指定比對模式"""

    if not record.get("ground_truth_available", False):
        raise ValueError(f"Ground truth unavailable in file: {path} doc_id={record.get('doc_id')}")

    pseudo_label_tokens = record.get("pseudo_label_tokens")
    gt_label_tokens = record.get("gt_label_tokens")

    if not isinstance(pseudo_label_tokens, list) or not isinstance(gt_label_tokens, list):
        raise ValueError(f"Missing token lists in file: {path} doc_id={record.get('doc_id')}")

    if len(pseudo_label_tokens) != len(gt_label_tokens):
        raise ValueError(
            f"Pseudo/GT token length mismatch in file: {path} doc_id={record.get('doc_id')} "
            f"pseudo={len(pseudo_label_tokens)} gt={len(gt_label_tokens)}"
        )

    pseudo_eval_tokens, gt_eval_tokens = _extract_tokens_for_match_mode(
        pseudo_label_tokens,
        gt_label_tokens,
        match_mode,
    )
    return pseudo_eval_tokens == gt_eval_tokens


def load_doc_records(divergence_path: Path, prediction_path: Path, match_mode: str) -> list[DocRecord]:
    """讀取 divergence 與 prediction 檔，組成 DocRecord 清單。"""

    divergence_data = read_json(divergence_path)
    prediction_data = read_json(prediction_path)

    prediction_map: dict[str, dict] = {}
    for record in prediction_data:
        doc_id = str(record.get("doc_id"))
        if doc_id in prediction_map:
            raise ValueError(f"Duplicate doc_id in prediction file: {prediction_path} doc_id={doc_id}")
        prediction_map[doc_id] = record

    records: list[DocRecord] = []
    for sample in divergence_data.get("samples", []):
        doc_id = str(sample.get("doc_id"))
        if doc_id not in prediction_map:
            raise ValueError(f"doc_id {doc_id} from divergence file not found in prediction file: {prediction_path}")

        prediction_record = prediction_map[doc_id]
        records.append(
            DocRecord(
                doc_id=doc_id,
                divergence_score=float(sample["divergence_score"]),
                exact_match=exact_match_from_prediction(prediction_record, prediction_path, match_mode),
                selected=bool(sample.get("selected", False)),
            )
        )

    if not records:
        raise ValueError(f"No samples found in divergence file: {divergence_path}")

    return records


def split_records_by_quantile(
    records: list[DocRecord],
    quantile: float,
) -> tuple[list[DocRecord], list[DocRecord], list[DocRecord]]:
    """把排序後的 records 切成 low / middle / high 三段

    回傳值:
    - (low_records, middle_records, high_records)
      三者都是 `list[DocRecord]`
    """

    group_size = int(len(records) * quantile)

    if group_size <= 0:
        raise ValueError(f"Quantile {quantile} is too small for {len(records)} records")

    if group_size * 2 > len(records):
        raise ValueError(f"Quantile {quantile} is too large for {len(records)} records")

    low_records = records[:group_size]
    high_records = records[-group_size:]
    middle_records = records[group_size : len(records) - group_size]
    return low_records, middle_records, high_records


def build_stats(records: list[DocRecord]) -> StatsBlock:
    """把一組 DocRecord 轉成 StatsBlock"""

    bucket = ScoreBucket()
    bucket.add_records(records)
    return bucket.summarize()


def build_unit_result(
    seed_name: str,
    fold_id: int,
    round_id: int,
    records: list[DocRecord],
    low_records: list[DocRecord],
    middle_records: list[DocRecord],
    high_records: list[DocRecord],
) -> UnitResult:
    """把單一 seed/fold/round 的資料整理成 UnitResult"""

    selected_all_records = [record for record in records if record.selected]
    selected_low_records = [record for record in low_records if record.selected]
    selected_middle_records = [record for record in middle_records if record.selected]
    selected_high_records = [record for record in high_records if record.selected]

    return UnitResult(
        seed_name=seed_name,
        fold_id=fold_id,
        round_id=round_id,
        selected_all=build_stats(selected_all_records),
        selected_low=build_stats(selected_low_records),
        selected_middle=build_stats(selected_middle_records),
        selected_high=build_stats(selected_high_records),
        low_all=build_stats(low_records),
        middle_all=build_stats(middle_records),
        high_all=build_stats(high_records),
    )


def format_score(value: float | None) -> str:
    """把浮點數格式化成可讀文字；空值則顯示 N/A。"""

    if value is None:
        return "N/A"
    return f"{value:.6f}"


def format_percent(value: float) -> str:
    """把比例格式化成百分比字串。"""

    return f"{value * 100:.2f}%"


def q_label(quantile: float) -> str:
    """把 quantile 轉成適合放進檔名的字串，例如 0.1 -> q010。"""

    return f"q{int(round(quantile * 100)):03d}"


def build_selected_group_debug_lines(
    title: str,
    records: list[DocRecord],
) -> list[str]:
    """建立某一區間內 selected 文檔的 debug 文字。"""

    lines = [f"  {title} (selected={len(records)}):"]
    if not records:
        lines.append("    - none")
        return lines

    for record in records:
        lines.append(
            "    - "
            f"doc_id={record.doc_id} "
            f"divergence={record.divergence_score:.12f} "
            f"exact_match={record.exact_match}"
        )
    return lines


def build_selected_round_debug_lines(
    seed_name: str,
    fold_id: int,
    round_id: int,
    records: list[DocRecord],
    low_records: list[DocRecord],
    middle_records: list[DocRecord],
    high_records: list[DocRecord],
    quantile: float,
) -> list[str]:
    """建立指定 round 的 selected 文檔散度值與切分結果文字。"""

    group_size = len(low_records)
    selected_records = [record for record in records if record.selected]
    selected_low_records = [record for record in low_records if record.selected]
    selected_middle_records = [record for record in middle_records if record.selected]
    selected_high_records = [record for record in high_records if record.selected]

    lines: list[str] = []

    lines.append("=" * 90)
    lines.append(f"DEBUG selected divergence values | {seed_name} fold{fold_id} round{round_id}")
    lines.append("=" * 90)
    lines.append(
        "  "
        f"total_records={len(records)} selected_records={len(selected_records)} "
        f"quantile={quantile:.2f} group_size={group_size}"
    )

    low_boundary_left = low_records[-1]
    low_boundary_right = middle_records[0] if middle_records else high_records[0]
    high_boundary_left = middle_records[-1] if middle_records else low_records[-1]
    high_boundary_right = high_records[0]

    low_boundary_tied = low_boundary_left.divergence_score == low_boundary_right.divergence_score
    high_boundary_tied = high_boundary_left.divergence_score == high_boundary_right.divergence_score

    lines.append(
        "  low/middle boundary: "
        f"left(doc_id={low_boundary_left.doc_id}, divergence={low_boundary_left.divergence_score:.12f}) | "
        f"right(doc_id={low_boundary_right.doc_id}, divergence={low_boundary_right.divergence_score:.12f}) | "
        f"same_score={low_boundary_tied}"
    )
    lines.append(
        "  middle/high boundary: "
        f"left(doc_id={high_boundary_left.doc_id}, divergence={high_boundary_left.divergence_score:.12f}) | "
        f"right(doc_id={high_boundary_right.doc_id}, divergence={high_boundary_right.divergence_score:.12f}) | "
        f"same_score={high_boundary_tied}"
    )
    lines.append(
        "  tie handling: records are sorted only by divergence_score; "
        "if scores are equal, Python stable sort preserves the original JSON order."
    )

    lines.extend(build_selected_group_debug_lines("low group", selected_low_records))
    lines.extend(build_selected_group_debug_lines("middle group", selected_middle_records))
    lines.extend(build_selected_group_debug_lines("high group", selected_high_records))

    lines.append("  selected docs in sorted order:")
    if not selected_records:
        lines.append("    - none")
    else:
        for record in selected_records:
            if record in low_records:
                group_name = "low"
            elif record in middle_records:
                group_name = "middle"
            else:
                group_name = "high"
            lines.append(
                "    - "
                f"doc_id={record.doc_id} divergence={record.divergence_score:.12f} "
                f"group={group_name} exact_match={record.exact_match}"
            )

    lines.append("")
    return lines


def emit_selected_round_debug(
    seed_name: str,
    fold_id: int,
    round_id: int,
    records: list[DocRecord],
    low_records: list[DocRecord],
    middle_records: list[DocRecord],
    high_records: list[DocRecord],
    quantile: float,
) -> list[str]:
    """在 terminal 印出指定 round 的 selected 文檔散度值與切分結果。"""

    lines = build_selected_round_debug_lines(
        seed_name=seed_name,
        fold_id=fold_id,
        round_id=round_id,
        records=records,
        low_records=low_records,
        middle_records=middle_records,
        high_records=high_records,
        quantile=quantile,
    )

    for line in lines:
        print(line)

    return lines


def build_selected_round_lines(round_summary: GroupAccumulator) -> list[str]:
    """建立某個 round 的 selected 統計文字。"""

    selected_all = round_summary.selected_all.summarize()
    selected_low = round_summary.selected_low.summarize()
    selected_middle = round_summary.selected_middle.summarize()
    selected_high = round_summary.selected_high.summarize()

    lines: list[str] = []
    lines.append(
        "    selected all      : "
        f"exact={selected_all.exact_count}/{selected_all.count} "
        f"rate={format_percent(selected_all.exact_rate)} "
        f"mean={format_score(selected_all.mean_score)} "
        f"median={format_score(selected_all.median_score)}"
    )
    lines.append(
        "    selected low      : "
        f"exact={selected_low.exact_count}/{selected_low.count} "
        f"rate={format_percent(selected_low.exact_rate)} "
        f"mean={format_score(selected_low.mean_score)} "
        f"median={format_score(selected_low.median_score)}"
    )
    lines.append(
        "    selected middle   : "
        f"exact={selected_middle.exact_count}/{selected_middle.count} "
        f"rate={format_percent(selected_middle.exact_rate)} "
        f"mean={format_score(selected_middle.mean_score)} "
        f"median={format_score(selected_middle.median_score)}"
    )
    lines.append(
        "    selected high     : "
        f"exact={selected_high.exact_count}/{selected_high.count} "
        f"rate={format_percent(selected_high.exact_rate)} "
        f"mean={format_score(selected_high.mean_score)} "
        f"median={format_score(selected_high.median_score)}"
    )
    return lines


def build_baseline_round_lines(round_summary: GroupAccumulator) -> list[str]:
    """建立某個 round 的 low/middle/high 全體 baseline 文字。"""

    low_all = round_summary.low_all.summarize()
    middle_all = round_summary.middle_all.summarize()
    high_all = round_summary.high_all.summarize()

    lines: list[str] = []
    lines.append(
        "    low all           : "
        f"exact={low_all.exact_count}/{low_all.count} "
        f"rate={format_percent(low_all.exact_rate)} "
        f"mean={format_score(low_all.mean_score)} "
        f"median={format_score(low_all.median_score)}"
    )
    lines.append(
        "    middle all        : "
        f"exact={middle_all.exact_count}/{middle_all.count} "
        f"rate={format_percent(middle_all.exact_rate)} "
        f"mean={format_score(middle_all.mean_score)} "
        f"median={format_score(middle_all.median_score)}"
    )
    lines.append(
        "    high all          : "
        f"exact={high_all.exact_count}/{high_all.count} "
        f"rate={format_percent(high_all.exact_rate)} "
        f"mean={format_score(high_all.mean_score)} "
        f"median={format_score(high_all.median_score)}"
    )
    return lines


def build_seed_section(
    seed_name: str,
    experiment_dir: Path,
    round_accumulators: dict[int, GroupAccumulator],
    unit_results: list[UnitResult],
) -> list[str]:
    """建立單一 seed 的完整報告段落。"""

    lines: list[str] = []
    lines.append("=" * 90)
    lines.append(seed_name)
    lines.append("=" * 90)
    lines.append(f"experiment directory: {experiment_dir}")
    lines.append("")

    lines.append("按 round 彙整 (跨所有 fold 合併):")
    available_round_ids = [
        round_id for round_id in sorted(round_accumulators.keys()) if round_accumulators[round_id].unit_count > 0
    ]
    if not available_round_ids:
        lines.append("  (指定 round 範圍內無可用資料)")
    for round_id in available_round_ids:
        lines.append(f"  round{round_id} (units={round_accumulators[round_id].unit_count}):")
        lines.extend(build_selected_round_lines(round_accumulators[round_id]))
        lines.extend(build_baseline_round_lines(round_accumulators[round_id]))
        lines.append("")

    lines.append("詳細結果 (每個 fold / round):")
    lines.append(
        "  fold round | sel_exact/sel_cnt rate sel_mean sel_median | "
        "sel_low exact/cnt mean median | sel_mid exact/cnt mean median | sel_high exact/cnt mean median"
    )
    lines.append("  " + "-" * 132)

    for unit in sorted(unit_results, key=lambda item: (item.fold_id, item.round_id)):
        lines.append(
            f"  {unit.fold_id:>4} {unit.round_id:>5} | "
            f"{unit.selected_all.exact_count:>3}/{unit.selected_all.count:<3} "
            f"{format_percent(unit.selected_all.exact_rate):>8} "
            f"{format_score(unit.selected_all.mean_score):>10} "
            f"{format_score(unit.selected_all.median_score):>10} | "
            f"{unit.selected_low.exact_count:>3}/{unit.selected_low.count:<3} "
            f"{format_score(unit.selected_low.mean_score):>10} "
            f"{format_score(unit.selected_low.median_score):>10} | "
            f"{unit.selected_middle.exact_count:>3}/{unit.selected_middle.count:<3} "
            f"{format_score(unit.selected_middle.mean_score):>10} "
            f"{format_score(unit.selected_middle.median_score):>10} | "
            f"{unit.selected_high.exact_count:>3}/{unit.selected_high.count:<3} "
            f"{format_score(unit.selected_high.mean_score):>10} "
            f"{format_score(unit.selected_high.median_score):>10}"
        )

    lines.append("")
    lines.append("  baseline (全體文檔，不限 selected):")
    lines.append(
        "  fold round | low_all exact/cnt mean median | "
        "middle_all exact/cnt mean median | high_all exact/cnt mean median"
    )
    lines.append("  " + "-" * 106)

    for unit in sorted(unit_results, key=lambda item: (item.fold_id, item.round_id)):
        lines.append(
            f"  {unit.fold_id:>4} {unit.round_id:>5} | "
            f"{unit.low_all.exact_count:>3}/{unit.low_all.count:<3} "
            f"{format_score(unit.low_all.mean_score):>10} "
            f"{format_score(unit.low_all.median_score):>10} | "
            f"{unit.middle_all.exact_count:>3}/{unit.middle_all.count:<4} "
            f"{format_score(unit.middle_all.mean_score):>10} "
            f"{format_score(unit.middle_all.median_score):>10} | "
            f"{unit.high_all.exact_count:>3}/{unit.high_all.count:<3} "
            f"{format_score(unit.high_all.mean_score):>10} "
            f"{format_score(unit.high_all.median_score):>10}"
        )

    lines.append("")
    return lines


def build_cross_seed_round_mean_lines(
    seed_round_accumulators: dict[str, dict[int, GroupAccumulator]],
    round_start: int,
    round_end: int,
) -> list[str]:
    """建立跨 seed 的 round 均值摘要文字。"""

    lines: list[str] = []
    lines.append("跨 seed round 均值摘要 (selected exact-match rate):")

    for round_id in range(round_start, round_end + 1):
        all_rates: list[float] = []
        low_rates: list[float] = []
        mid_rates: list[float] = []
        high_rates: list[float] = []

        for seed_name in sorted(seed_round_accumulators.keys(), key=seed_sort_key):
            round_accumulator = seed_round_accumulators[seed_name][round_id]
            if round_accumulator.unit_count == 0:
                continue

            selected_all = round_accumulator.selected_all.summarize()
            selected_low = round_accumulator.selected_low.summarize()
            selected_middle = round_accumulator.selected_middle.summarize()
            selected_high = round_accumulator.selected_high.summarize()

            all_rates.append(selected_all.exact_rate)
            low_rates.append(selected_low.exact_rate)
            mid_rates.append(selected_middle.exact_rate)
            high_rates.append(selected_high.exact_rate)

        if not all_rates:
            continue

        lines.append(
            f"round{round_id}: "
            f"all={format_percent(mean(all_rates))} "
            f"low={format_percent(mean(low_rates))} "
            f"mid={format_percent(mean(mid_rates))} "
            f"high={format_percent(mean(high_rates))}"
        )

    return lines


def build_cross_seed_round_pooled_lines(
    seed_round_accumulators: dict[str, dict[int, GroupAccumulator]],
    round_start: int,
    round_end: int,
) -> list[str]:
    """建立跨 seed 合併分子分母摘要文字。"""

    lines: list[str] = []
    lines.append("跨 seed round 合併分子分母摘要 (selected exact-match rate):")

    for round_id in range(round_start, round_end + 1):
        pooled_exact = {"all": 0, "low": 0, "mid": 0, "high": 0}
        pooled_count = {"all": 0, "low": 0, "mid": 0, "high": 0}

        for seed_name in sorted(seed_round_accumulators.keys(), key=seed_sort_key):
            round_accumulator = seed_round_accumulators[seed_name][round_id]
            if round_accumulator.unit_count == 0:
                continue

            selected_all = round_accumulator.selected_all.summarize()
            selected_low = round_accumulator.selected_low.summarize()
            selected_middle = round_accumulator.selected_middle.summarize()
            selected_high = round_accumulator.selected_high.summarize()

            pooled_exact["all"] += selected_all.exact_count
            pooled_exact["low"] += selected_low.exact_count
            pooled_exact["mid"] += selected_middle.exact_count
            pooled_exact["high"] += selected_high.exact_count

            pooled_count["all"] += selected_all.count
            pooled_count["low"] += selected_low.count
            pooled_count["mid"] += selected_middle.count
            pooled_count["high"] += selected_high.count

        if pooled_count["all"] == 0:
            continue

        all_rate = 0.0 if pooled_count["all"] == 0 else pooled_exact["all"] / pooled_count["all"]
        low_rate = 0.0 if pooled_count["low"] == 0 else pooled_exact["low"] / pooled_count["low"]
        mid_rate = 0.0 if pooled_count["mid"] == 0 else pooled_exact["mid"] / pooled_count["mid"]
        high_rate = 0.0 if pooled_count["high"] == 0 else pooled_exact["high"] / pooled_count["high"]

        lines.append(
            f"round{round_id}: "
            f"all={pooled_exact['all']}/{pooled_count['all']} rate={format_percent(all_rate)} "
            f"low={pooled_exact['low']}/{pooled_count['low']} rate={format_percent(low_rate)} "
            f"mid={pooled_exact['mid']}/{pooled_count['mid']} rate={format_percent(mid_rate)} "
            f"high={pooled_exact['high']}/{pooled_count['high']} rate={format_percent(high_rate)}"
        )

    return lines


def build_round_coverage_lines(
    seed_round_accumulators: dict[str, dict[int, GroupAccumulator]],
    round_start: int,
    round_end: int,
) -> list[str]:
    """建立各 seed 在每個 round 的實際覆蓋情況摘要。"""

    lines: list[str] = []
    lines.append("可用 round 覆蓋摘要:")
    for seed_name in sorted(seed_round_accumulators.keys(), key=seed_sort_key):
        coverage_parts: list[str] = []
        for round_id in range(round_start, round_end + 1):
            unit_count = seed_round_accumulators[seed_name][round_id].unit_count
            coverage_parts.append(f"round{round_id}={unit_count} folds")
        lines.append(f"  - {seed_name}: " + ", ".join(coverage_parts))
    return lines


def build_report(
    summary_path: Path,
    experiments_root: Path,
    match_mode: str,
    quantile: float,
    fold_start: int,
    fold_end: int,
    round_start: int,
    round_end: int,
    experiment_map: dict[str, Path],
    seed_unit_results: dict[str, list[UnitResult]],
    seed_round_accumulators: dict[str, dict[int, GroupAccumulator]],
    debug_report_lines: list[str],
    skipped_missing_units: list[str],
) -> str:
    """組合整份文字報告。"""

    lines: list[str] = []
    lines.append("選中未標註文檔的 divergence 統計報告")
    lines.append("=" * 90)
    lines.append("分析設定:")
    lines.append(f"  - summary file: {summary_path}")
    lines.append(f"  - experiments root: {experiments_root}")
    lines.append(f"  - fold 範圍: fold{fold_start} ~ fold{fold_end}")
    lines.append(f"  - round 範圍: round{round_start} ~ round{round_end}")
    lines.append(f"  - low/high 定義: 各 seed-fold-round 內，最低/最高 {format_percent(quantile)} 散度文檔")
    lines.append("  - middle 定義: 不介於 low 與 high 的其餘文檔")
    lines.append("  - selected 定義: divergence 檔中 selected=True 的文檔")
    lines.append(f"  - exact match 定義: {describe_match_mode(match_mode)}")
    if skipped_missing_units:
        lines.append(f"  - 缺失檔案已略過: {len(skipped_missing_units)} 個 seed/fold/round 單位")
    lines.append("")
    lines.append("summary 內的 experiment directories:")
    for seed_name in sorted(experiment_map.keys(), key=seed_sort_key):
        lines.append(f"  - {seed_name}: {experiment_map[seed_name]}")
    lines.append("")

    lines.extend(
        build_round_coverage_lines(
            seed_round_accumulators=seed_round_accumulators,
            round_start=round_start,
            round_end=round_end,
        )
    )
    lines.append("")

    for seed_name in sorted(experiment_map.keys(), key=seed_sort_key):
        lines.extend(
            build_seed_section(
                seed_name=seed_name,
                experiment_dir=experiment_map[seed_name],
                round_accumulators=seed_round_accumulators[seed_name],
                unit_results=seed_unit_results[seed_name],
            )
        )

    lines.append("=" * 90)
    lines.extend(
        build_cross_seed_round_mean_lines(
            seed_round_accumulators=seed_round_accumulators,
            round_start=round_start,
            round_end=round_end,
        )
    )
    lines.append("")
    lines.extend(
        build_cross_seed_round_pooled_lines(
            seed_round_accumulators=seed_round_accumulators,
            round_start=round_start,
            round_end=round_end,
        )
    )
    lines.append("")

    if debug_report_lines:
        lines.append("=" * 90)
        lines.append("指定 round 的 selected divergence debug 明細")
        lines.append("=" * 90)
        lines.extend(debug_report_lines)
        lines.append("")

    return "\n".join(lines) + "\n"


def main() -> int:
    """主程式入口。

    回傳值:
    - 0 代表成功完成
    """

    # 先解析命令列參數，取得 summary 路徑、fold 範圍、round 範圍、quantile 與輸出資料夾等設定
    args = parse_args()
    debug_selected_rounds = set(args.debug_selected_rounds)

    # 檢查 quantile 是否落在合法範圍內，避免 low/high 區間太小或互相重疊
    if not 0 < args.quantile < 0.5:
        # 如果 quantile 不合法，就立刻拋出例外並停止程式
        raise ValueError("--quantile must be between 0 and 0.5")

    # 把 summary 參數轉成 Path 物件，後續才能方便做路徑串接與讀檔
    summary_path = Path(args.summary)

    # 把 experiments root 轉成 Path 物件，後續用來定位三個 seed 的實驗資料夾
    experiments_root = Path(args.experiments_root)

    # 把輸出資料夾參數轉成 Path 物件，稍後會在這裡建立報告檔
    output_dir = Path(args.output_dir)

    # 解析 summary.txt 中列出的實驗目錄，整理成 {seed_name: experiment_dir} 的對照表
    experiment_map = resolve_experiment_map(summary_path, experiments_root)

    # 用來保存每個 seed 底下所有 fold / round 單位的細部結果
    seed_unit_results: dict[str, list[UnitResult]] = {}

    # 用來保存每個 seed 依 round 聚合後的統計累積器
    seed_round_accumulators: dict[str, dict[int, GroupAccumulator]] = {}

    # 用來保存指定 round 的 debug 明細，最後會附加進輸出報告
    debug_report_lines: list[str] = []

    # 若啟用略過缺失 round，這裡會記錄被跳過的 seed/fold/round 單位
    skipped_missing_units: list[str] = []

    # 依照 seed20、seed42、seed60 的數字順序逐一處理每個 seed
    for seed_name in sorted(experiment_map.keys(), key=seed_sort_key):
        # 找出這個 seed 對應實驗資料夾下唯一的 pseudo_results_* 目錄
        pseudo_dir = find_single_pseudo_results_dir(experiment_map[seed_name])

        # 準備一個串列，保存這個 seed 的所有單位結果
        unit_results: list[UnitResult] = []

        # 先為每個 round 建立一個累積器，之後可把不同 fold 的同一 round 統計加總在一起
        round_accumulators: dict[int, GroupAccumulator] = {
            # round_id 是 round 編號，值則是對應的 GroupAccumulator 物件
            round_id: GroupAccumulator() for round_id in range(args.round_start, args.round_end + 1)
        }

        # 依序走訪指定範圍內的每個 fold
        for fold_id in range(args.fold_start, args.fold_end + 1):
            # 在每個 fold 裡，再依序走訪指定範圍內的每個 self-training round
            for round_id in range(args.round_start, args.round_end + 1):
                # 組出這個 fold / round 對應的 divergence JSON 檔路徑
                divergence_path = pseudo_dir / f"nest_divergence_scores_fold{fold_id}_round{round_id}.json"

                # 組出這個 fold / round 對應的 all_unlabeled_pseudo_predictions JSON 檔路徑
                prediction_path = pseudo_dir / f"all_unlabeled_pseudo_predictions_fold{fold_id}_round{round_id}.json"

                missing_parts: list[str] = []
                if not divergence_path.is_file():
                    missing_parts.append("divergence")
                if not prediction_path.is_file():
                    missing_parts.append("prediction")
                if missing_parts:
                    if args.skip_missing_rounds:
                        skipped_missing_units.append(
                            f"{seed_name} fold{fold_id} round{round_id}: missing {', '.join(missing_parts)}"
                        )
                        continue
                    if "divergence" in missing_parts:
                        raise FileNotFoundError(f"Missing divergence file: {divergence_path}")
                    raise FileNotFoundError(f"Missing prediction file: {prediction_path}")

                # 讀入這個單位的所有文檔，並依 divergence 由小到大排序，方便之後切 low / middle / high
                records = sorted(
                    load_doc_records(divergence_path, prediction_path, args.match_mode),
                    key=lambda record: record.divergence_score,
                )

                # 依 quantile 把排序後的 records 切成 low、middle、high 三段
                low_records, middle_records, high_records = split_records_by_quantile(records, args.quantile)

                if round_id in debug_selected_rounds:
                    debug_lines = emit_selected_round_debug(
                        seed_name=seed_name,
                        fold_id=fold_id,
                        round_id=round_id,
                        records=records,
                        low_records=low_records,
                        middle_records=middle_records,
                        high_records=high_records,
                        quantile=args.quantile,
                    )
                    debug_report_lines.extend(debug_lines)

                # 把這個 seed / fold / round 單位的 selected 與 low/middle/high 統計整理成 UnitResult
                unit_result = build_unit_result(
                    seed_name=seed_name,
                    fold_id=fold_id,
                    round_id=round_id,
                    records=records,
                    low_records=low_records,
                    middle_records=middle_records,
                    high_records=high_records,
                )

                # 把單位結果加入 unit_results，之後可輸出每個 fold / round 的細部表格
                unit_results.append(unit_result)

                # 同時把這個單位的資料累積到對應 round 的聚合器中，方便輸出跨 fold 的 round 摘要
                round_accumulators[round_id].add_unit(records, low_records, middle_records, high_records)

        # 這個 seed 的所有單位結果整理完成後，存入總字典
        seed_unit_results[seed_name] = unit_results

        # 這個 seed 的各 round 聚合統計整理完成後，也存入總字典
        seed_round_accumulators[seed_name] = round_accumulators

    # 把所有 seed 的統計結果組合成最終的文字報告內容
    report = build_report(
        summary_path=summary_path,
        experiments_root=experiments_root,
        match_mode=args.match_mode,
        quantile=args.quantile,
        fold_start=args.fold_start,
        fold_end=args.fold_end,
        round_start=args.round_start,
        round_end=args.round_end,
        experiment_map=experiment_map,
        seed_unit_results=seed_unit_results,
        seed_round_accumulators=seed_round_accumulators,
        debug_report_lines=debug_report_lines,
        skipped_missing_units=skipped_missing_units,
    )

    # 若輸出資料夾不存在，就先建立，parents=True 代表連同上層不存在的資料夾一起建立
    output_dir.mkdir(parents=True, exist_ok=True)

    # 根據 summary 檔名推導輸出報告的檔名前綴
    prefix = output_prefix_from_summary(summary_path)

    # 非預設比對模式時，把模式名稱寫進輸出檔名，避免覆蓋原本的全文比對報告
    match_suffix = "" if args.match_mode == MATCH_MODE_ALL else f"_match_{args.match_mode}"

    # 組合最終輸出檔路徑，檔名會包含 round 範圍與 quantile 標記
    output_path = output_dir / (
        f"{prefix}_selected_divergence_statistics_"
        f"round{args.round_start}_{args.round_end}_{q_label(args.quantile)}{match_suffix}.txt"
    )

    # 把完整報告內容寫入 txt 檔，使用 UTF-8 編碼避免中文亂碼
    output_path.write_text(report, encoding="utf-8")

    # 在終端印出最終輸出檔位置，方便使用者確認報告存在哪裡
    print(f"Report written to: {output_path}")

    # 主程式正常結束時回傳 0，表示執行成功
    return 0


if __name__ == "__main__":
    # 只有在直接執行這支腳本時才會進入這裡；若是被其他模組 import，則不會自動執行 main()
    # raise SystemExit(main()) 的作用是把 main() 的整數回傳值轉成程式結束代碼
    raise SystemExit(main())