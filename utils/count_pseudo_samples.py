"""統計各折各輪偽標籤樣本的文檔數量。"""

import json
import argparse
import pathlib
from typing import Dict, Tuple


def count_samples_per_fold_and_round(pseudo_dir: pathlib.Path) -> Dict[Tuple[int, int], int]:
    """
    掃描 pseudo_dir 目錄下所有 pseudo_labeled_samples_fold{i}_round{j}.json 檔案，
    統計每個檔案中的文檔數量。
    
    Args:
        pseudo_dir: 偽標籤結果所在的目錄
        
    Returns:
        字典，鍵為 (fold, round)，值為該輪選中的文檔數量
    """
    results = {}
    
    # 掃描目錄下所有符合模式的 JSON 檔案
    for json_file in sorted(pseudo_dir.glob("pseudo_labeled_samples_fold*.json")):
        # 解析檔名以提取 fold 與 round 編號
        # 格式: pseudo_labeled_samples_fold{i}_round{j}.json
        filename = json_file.stem  # 去掉 .json 副檔名
        
        try:
            # 使用正則表達式提取 fold 與 round 編號
            import re
            match = re.search(r"fold(\d+)_round(\d+)", filename)
            if not match:
                print(f"警告：無法解析檔名 {filename}，跳過")
                continue
            
            fold = int(match.group(1))
            round_num = int(match.group(2))
        except (ValueError, AttributeError):
            print(f"警告：無法解析檔名 {filename}，跳過")
            continue
        
        # 讀取 JSON 檔案並計算文檔數量
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # 檔案內容應該是一個列表，每個元素是一個文檔的偽標籤資訊
            if isinstance(data, list):
                doc_count = len(data)
            else:
                print(f"警告：{json_file.name} 的格式不是列表，跳過")
                continue
            
            results[(fold, round_num)] = doc_count
        except json.JSONDecodeError:
            print(f"警告：{json_file.name} 的 JSON 格式無效，跳過")
            continue
        except Exception as e:
            print(f"警告：讀取 {json_file.name} 時出錯：{e}")
            continue
    
    return results


def format_report(results: Dict[Tuple[int, int], int]) -> str:
    """
    將統計結果格式化為易讀的報告。
    
    Args:
        results: 統計結果字典
        
    Returns:
        格式化後的報告字串
    """
    lines = []
    
    lines.append("各折各輪偽標籤樣本文檔數量統計")
    lines.append("=" * 80)
    lines.append("")
    
    # 找出所有的 fold 與 round 編號
    if not results:
        lines.append("未找到任何偽標籤樣本檔案")
        return "\n".join(lines)
    
    all_folds = sorted(set(fold for fold, _ in results.keys()))
    all_rounds = sorted(set(round_num for _, round_num in results.keys()))
    
    # 建立表頭
    header = "Fold \\ Round" + " | " + " | ".join(f"Round {r}" for r in all_rounds)
    lines.append(header)
    lines.append("-" * len(header))
    
    # 按 fold 輸出各行
    for fold in all_folds:
        row_parts = [f"Fold {fold}"]
        for round_num in all_rounds:
            if (fold, round_num) in results:
                row_parts.append(f"{results[(fold, round_num)]:6d}")
            else:
                row_parts.append("  N/A ")
        lines.append(" | ".join(row_parts))
    
    lines.append("")
    lines.append("=" * 80)
    lines.append("詳細統計：")
    lines.append("")
    
    # 按 fold 輸出詳細資訊
    for fold in all_folds:
        lines.append(f"Fold {fold}:")
        fold_totals = []
        for round_num in all_rounds:
            if (fold, round_num) in results:
                count = results[(fold, round_num)]
                fold_totals.append(count)
                lines.append(f"  Round {round_num}: {count:4d} 篇文檔")
            else:
                lines.append(f"  Round {round_num}: N/A")
        
        if fold_totals:
            avg_count = sum(fold_totals) / len(fold_totals)
            total_count = sum(fold_totals)
            lines.append(f"  小計: {total_count} 篇 | 平均每輪: {avg_count:.1f} 篇")
        lines.append("")
    
    # 計算全局統計
    lines.append("=" * 80)
    lines.append("全局統計：")
    lines.append("")
    
    total_all = sum(results.values())
    avg_all = total_all / len(results) if results else 0
    
    lines.append(f"總選中文檔數: {total_all}")
    lines.append(f"平均每輪選中: {avg_all:.2f} 篇")
    
    # 按 round 統計
    round_stats = {}
    for round_num in all_rounds:
        round_totals = [
            results[(fold, round_num)]
            for fold in all_folds
            if (fold, round_num) in results
        ]
        if round_totals:
            round_stats[round_num] = {
                "total": sum(round_totals),
                "avg": sum(round_totals) / len(round_totals),
            }
    
    lines.append("")
    lines.append("各輪統計（跨所有折）：")
    for round_num in sorted(round_stats.keys()):
        stats = round_stats[round_num]
        lines.append(
            f"  Round {round_num}: 總計 {stats['total']:4d} 篇 | 平均 {stats['avg']:.2f} 篇/折"
        )
    
    return "\n".join(lines)


def main() -> None:
    """主程式進入點。"""
    parser = argparse.ArgumentParser(description="統計各折各輪偽標籤樣本的文檔數量")
    parser.add_argument(
        "--pseudo_dir",
        type=pathlib.Path,
        required=True,
        help="偽標籤結果所在的目錄（通常是 pseudo_results_TIMESTAMP/）",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        help="可選：將報告輸出至檔案",
    )
    
    args = parser.parse_args()
    
    # 驗證目錄存在
    if not args.pseudo_dir.exists():
        print(f"錯誤：目錄不存在 {args.pseudo_dir}")
        return
    
    # 統計文檔數量
    print(f"掃描目錄：{args.pseudo_dir}")
    results = count_samples_per_fold_and_round(args.pseudo_dir)
    
    if not results:
        print("未找到任何偽標籤樣本檔案")
        return
    
    print(f"找到 {len(results)} 個檔案")
    
    # 生成報告
    report = format_report(results)
    
    # 輸出報告
    print("\n" + report)
    
    # 若指定輸出檔案，則寫入
    if args.output:
        args.output.write_text(report, encoding="utf-8")
        print(f"\n報告已保存至：{args.output}")


if __name__ == "__main__":
    main()
