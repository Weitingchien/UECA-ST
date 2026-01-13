"""彙整 eval_UECA_CE_v2 產生的跨折評估計數結果。"""
# 匯入 __future__ 提供的型別註解支援，允許向前參照型別名稱。
from __future__ import annotations

# 匯入 argparse 以處理命令列參數。
import argparse

# 匯入 pathlib 以便操作檔案與目錄路徑。
import pathlib

# 匯入 re 以進行正則表達式比對。
import re

# 匯入 math 以計算標準差
import math

# 匯入 Dict、Tuple 與 List 以撰寫明確的型別提示。
from typing import Dict, Tuple, List

# 依照紀錄檔中的指標名稱建立對應的正則表達式。
_LINE_PATTERNS: Dict[str, re.Pattern[str]] = {
    # Emotion 指標的三個計數。
    "Emotion": re.compile(
        r"Emotion:\s*預測正確的數量:(\d+)\s+預測出情緒的數量:(\d+)\s+實際正確情緒的數量:(\d+)"
    ),
    # Cause 指標的三個計數。
    "Cause": re.compile(
        r"Cause:\s*預測正確的數量:(\d+)\s+預測出原因的數量:(\d+)\s+實際正確原因的數量:(\d+)"
    ),
    # Pair 方式一的三個計數。
    "Pair (m1)": re.compile(
        r"Pair方式一:\s*預測正確:\s*(\d+)\s+預測出配對:\s*(\d+)\s+實際正確配對的數量:(\d+)"
    ),
    # Pair 方式二的三個計數。
    "Pair (m2)": re.compile(
        r"Pair方式二:\s*預測正確:\s*(\d+)\s+預測出配對:\s*(\d+)\s+實際正確配對的數量:(\d+)"
    ),
    # Pair 方式三的三個計數。
    "Pair (m3)": re.compile(
        r"Pair方式三:\s*預測正確:\s*(\d+)\s+預測出配對:\s*(\d+)\s+實際正確配對的數量:(\d+)"
    ),
}

# 定義總表格的欄位寬度與對齊資訊，方便統一排版。
_COLUMN_SPECS = [
    {"header": "Metric", "subheader": "", "width": 17, "data_align": "<"},
    {"header": "Total TP", "subheader": "(預測正確的數量)", "width": 33, "data_align": ">"},
    {"header": "Total Pred", "subheader": "(預測出情緒/原因/組合的數量)", "width": 38, "data_align": ">"},
    {"header": "Total GT", "subheader": "(實際正確情緒/原因/組合的數量)", "width": 38, "data_align": ">"},
    {"header": "Avg TP", "subheader": "", "width": 8, "data_align": ">"},
    {"header": "Avg Pred", "subheader": "", "width": 10, "data_align": ">"},
    {"header": "Avg GT", "subheader": "", "width": 8, "data_align": ">"},
    {"header": "P (micro)", "subheader": "", "width": 12, "data_align": ">"},
    {"header": "R (micro)", "subheader": "", "width": 12, "data_align": ">"},
    {"header": "F1 (micro)", "subheader": "", "width": 12, "data_align": ">"},
]

_COLUMN_SEPARATOR = " | "
_TABLE_WIDTH = sum(spec["width"] for spec in _COLUMN_SPECS) + len(_COLUMN_SEPARATOR) * (len(_COLUMN_SPECS) - 1)


def _format_row(values: Tuple[str, ...], aligns: Tuple[str, ...]) -> str:
    """根據欄位寬度與對齊設定格式化單行文字。"""
    cells = []
    for value, spec, align in zip(values, _COLUMN_SPECS, aligns):
        cells.append(f"{value:{align}{spec['width']}}")
    return _COLUMN_SEPARATOR.join(cells)


class MetricTally:
    """維護單一指標的累計值與折數。"""

    # 宣告允許的屬性，避免動態新增屬性造成額外開銷。
    __slots__ = ("true_positives", "predicted_positives", "ground_truth", "folds", "fold_scores")

    def __init__(self) -> None:
        # 初始化真陽性總數為 0。
        self.true_positives = 0
        # 初始化預測為正的總數為 0。
        self.predicted_positives = 0
        # 初始化實際為正的總數為 0。
        self.ground_truth = 0
        # 初始化已處理的折數為 0。
        self.folds = 0
        # 儲存每折的 (precision, recall, f1) 以計算標準差
        self.fold_scores: List[Tuple[float, float, float]] = []

    def add(self, tp: int, predicted: int, ground_truth: int) -> None:
        # 累加真陽性數量。
        self.true_positives += tp
        # 累加預測為正的數量。
        self.predicted_positives += predicted
        # 累加實際為正的數量。
        self.ground_truth += ground_truth
        # 累計已處理的折數。
        self.folds += 1
        # 計算此折的 P/R/F1 並儲存
        p = _safe_div(tp, predicted)
        r = _safe_div(tp, ground_truth)
        f1 = _f1(p, r)
        self.fold_scores.append((p, r, f1))

    def averages(self) -> Tuple[float, float, float]:
        # 在沒有任何折數時回傳零值以避免除以零。
        if self.folds == 0:
            return 0.0, 0.0, 0.0
        # 回傳每折平均的真陽性、預測為正與實際為正數量。
        return (
            self.true_positives / self.folds,
            self.predicted_positives / self.folds,
            self.ground_truth / self.folds,
        )

    def micro_scores(self) -> Tuple[float, float, float]:
        # 以總累計值計算 micro Precision。
        precision = _safe_div(self.true_positives, self.predicted_positives)
        # 以總累計值計算 micro Recall。
        recall = _safe_div(self.true_positives, self.ground_truth)
        # 由 precision 與 recall 推導 micro F1。
        f1 = _f1(precision, recall)
        # 回傳 micro precision、recall 與 F1。
        return precision, recall, f1

    def std_scores(self) -> Tuple[float, float, float]:
        """計算各折 P/R/F1 的標準差。"""
        if len(self.fold_scores) < 2:
            return 0.0, 0.0, 0.0
        
        # 分離 P, R, F1 列表
        p_list = [s[0] for s in self.fold_scores]
        r_list = [s[1] for s in self.fold_scores]
        f1_list = [s[2] for s in self.fold_scores]
        
        def _std(values: List[float]) -> float:
            """ 計算樣本標準差 (sample std, N-1) """
            n = len(values)
            if n < 2:
                return 0.0
            mean = sum(values) / n
            variance = sum((x - mean) ** 2 for x in values) / (n - 1)
            return math.sqrt(variance)
        
        return _std(p_list), _std(r_list), _std(f1_list)


def _safe_div(num: int, denom: int) -> float:
    # 若分母為零則直接回傳 0.0，避免 ZeroDivisionError
    # 以條件運算回傳除法結果或 0.0
    return num / denom if denom else 0.0


def _f1(precision: float, recall: float) -> float:
    # 當 precision 與 recall 皆為零時直接回傳 0.0
    # 以標準 F1 公式計算結果
    return (2.0 * precision * recall / (precision + recall)) if (precision + recall) else 0.0


def parse_counts(lines: str) -> Tuple[Dict[str, MetricTally], list]:
    # 先為每個指標建立一個統計物件
    tallies = {name: MetricTally() for name in _LINE_PATTERNS}
    # 紀錄每折的詳細計數，結構為 [(fold_name, {metric: (tp, pred, gt)}), ...]
    fold_details = []
    current_fold = None
    current_fold_data = {}
    # 逐行解析輸入文字
    for line in lines.splitlines():
        # 嘗試從行中找到 Fold 識別符
        fold_match = re.search(r"=== Fold (\d+) ===", line)
        if fold_match:
            # 如果找到新的 fold，先儲存前一個 fold 的資料
            if current_fold_data:
                fold_details.append((current_fold, current_fold_data.copy()))
            current_fold = f"Fold {fold_match.group(1)}"
            current_fold_data = {}
        # 嘗試將目前這行與每一個指標的正則表達式比對
        for name, pattern in _LINE_PATTERNS.items():
            # 檢查是否符合指標的格式
            match = pattern.search(line)
            if match:
                # 解析出真陽性、預測正例與實際正例數量
                tp, predicted, gt = map(int, match.groups())
                # 將本行的計數累加到對應指標。
                tallies[name].add(tp, predicted, gt)
                # 將本折的計數紀錄到 fold_details
                current_fold_data[name] = (tp, predicted, gt)
                # 找到對應指標後即可離開內層迴圈
                break
    # 紀錄最後一個 fold。
    if current_fold_data:
        fold_details.append((current_fold, current_fold_data))
    # 回傳累計完成的指標資料與各折詳細資訊。
    return tallies, fold_details


def format_report(tallies: Dict[str, MetricTally], fold_details: list = None) -> str:
    # 建立輸出行的暫存串列。
    lines = []
    
    # 若有 fold_details，則先輸出各折的詳細資訊。
    if fold_details:
        lines.append("各 Fold 詳細計數與評分：")
        lines.append("=" * 160)
        
        # 輸出每折的計數與評分。
        for fold_name, fold_data in fold_details:
            lines.append(f"\n{fold_name}:")
            # 對每個指標輸出詳細資訊。
            for name in _LINE_PATTERNS:
                if name in fold_data:
                    tp, pred, gt = fold_data[name]
                    # 計算此折此指標的 P/R/F1。
                    precision = _safe_div(tp, pred)
                    recall = _safe_div(tp, gt)
                    f1 = _f1(precision, recall)
                    lines.append(
                        f"  {name:<15} | TP={tp:3d} Pred={pred:3d} GT={gt:3d} |"
                        f" P={precision*100:6.2f}% R={recall*100:6.2f}% F1={f1*100:6.2f}%"
                    )
        
        lines.append("\n" + "=" * 160)
        lines.append("")  # 空行分隔。
    
    # 輸出 10 折累計結果（列表式排版）
    lines.append("10 折累計統計結果：")
    lines.append("")
    
    # 依序處理每個指標。
    for name in _LINE_PATTERNS:
        # 取出對應的統計物件。
        tally = tallies[name]
        # 計算平均值。
        avg_tp, avg_pred, avg_gt = tally.averages()
        # 計算 micro precision、recall 與 F1。
        precision, recall, f1 = tally.micro_scores()
        
        # 輸出各指標的詳細資訊
        lines.append(f"{name}:")
        lines.append(f"  Total TP (預測正確的數量): {tally.true_positives}")
        lines.append(f"  Total Pred (預測出的數量): {tally.predicted_positives}")
        lines.append(f"  Total GT (實際正確的數量): {tally.ground_truth}")
        lines.append(f"  Avg TP: {avg_tp:.2f} | Avg Pred: {avg_pred:.2f} | Avg GT: {avg_gt:.2f}")
        lines.append(f"  P (micro): {precision * 100:.2f}% | R (micro): {recall * 100:.2f}% | F1 (micro): {f1 * 100:.2f}%")
        # 計算並輸出標準差
        std_p, std_r, std_f1 = tally.std_scores()
        lines.append(f"  Std P: {std_p * 100:.2f}% | Std R: {std_r * 100:.2f}% | Std F1: {std_f1 * 100:.2f}%")
        lines.append("")
    
    # 附加公式說明
    lines.append("=" * 80)
    lines.append("指標計算公式說明：")
    lines.append("")
    lines.append("● 各指標含義：")
    lines.append("  - Total TP (預測正確的數量)：所有 10 折中，模型預測正確的樣本總數")
    lines.append("  - Total Pred (預測出的數量)：所有 10 折中，模型判定為正例的樣本總數")
    lines.append("  - Total GT (實際正確的數量)：所有 10 折中，真實標籤為正例的樣本總數")
    lines.append("")
    lines.append("● 計算公式（Micro-averaged）：")
    lines.append("  - Precision (P) = Total TP / Total Pred")
    lines.append("  - Recall (R)    = Total TP / Total GT")
    lines.append("  - F1            = 2 × P × R / (P + R)")
    lines.append("")
    lines.append("● 各指標說明：")
    lines.append("  - Emotion: 情緒句的識別性能")
    lines.append("  - Cause:   原因句的識別性能")
    lines.append("  - Pair (m1): 情緒-原因配對的識別性能（只比較位置）")
    lines.append("  - Pair (m2): 情緒-原因配對的識別性能（比較位置 + 原因情緒標籤）")
    lines.append("  - Pair (m3): 情緒-原因配對的識別性能（比較位置 + 原因情緒標籤 + 主要情緒標籤）")
    lines.append("=" * 80)
    
    # 以換行字元串接所有輸出行。
    # 回傳整理好的多行文字。
    return "\n".join(lines)


def main() -> None:
    # 建立命令列參數解析器並設定說明文字。
    parser = argparse.ArgumentParser(description="彙整跨折的評估計數。")
    # 加入 input 參數以取得 test_results.txt 路徑。
    parser.add_argument("--input", required=True, type=pathlib.Path, help="test_results.txt 的路徑")
    # 加入 output 參數以指定輸出檔案（可選）。
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        help="選擇性地將報告寫入檔案；若未提供則只印出結果。",
    )
    # 解析命令列參數。
    args = parser.parse_args()

    # 讀取輸入檔案內容。
    text = args.input.read_text(encoding="utf-8")
    # 解析並累計所有指標的計數與各折詳細資訊。
    tallies, fold_details = parse_counts(text)

    # 如果有指標沒有任何資料，代表輸入可能缺漏。
    missing = [name for name, tally in tallies.items() if tally.folds == 0]
    if missing:
        # 將缺漏的指標名稱串成字串，方便錯誤訊息輸出。
        printable = ", ".join(missing)
        # 提醒使用者輸入內容不完整。
        raise ValueError(f"找不到以下指標的紀錄：{printable}，請檢查輸入檔案。")

    # 格式化統計結果（包含各折詳細計數）。
    report = format_report(tallies, fold_details)
    # 印出結果供使用者查看。
    print(report)

    # 若使用者指定了輸出檔案路徑。
    if args.output:
        # 將結果寫入指定的輸出檔案。
        args.output.write_text(report + "\n", encoding="utf-8")


# 僅在直接執行此模組時呼叫 main 函式。
if __name__ == "__main__":
    main()
