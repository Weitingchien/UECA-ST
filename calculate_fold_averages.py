#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
計算多折交叉驗證的平均評估指標

功能：
1. 解析評估結果檔案
2. 計算各項指標的平均值
3. 生成格式化的統計報告

使用方式：
python calculate_fold_averages.py --file test_results.txt
或
python calculate_fold_averages.py --file test_results.txt --output average_results.txt
"""

import argparse
import sys
from typing import List, Dict, Tuple


class MetricCalculator:
    def __init__(self):
        self.fold_results = []
        
    def parse_results_file(self, file_path: str) -> None:
        """解析評估結果檔案"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except FileNotFoundError:
            print(f"錯誤：找不到檔案 {file_path}")
            sys.exit(1)
        except Exception as e:
            print(f"讀取檔案時發生錯誤：{e}")
            sys.exit(1)
            
        # 逐行解析，找到每個 fold 的結果
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # 找到 fold 開始的標記
            if line.startswith("=== Fold ") and line.endswith(" ==="):
                # 提取 fold 編號
                fold_num = int(line.split("Fold ")[1].split(" ===")[0])
                
                # 找到指標表格的開始位置
                metrics_found = False
                j = i + 1
                while j < len(lines):
                    current_line = lines[j].strip()
                    
                    # 找到指標表格的標題行
                    if "Metric" in current_line and "Precision" in current_line:
                        # 跳過分隔線
                        j += 1
                        if j < len(lines) and "---" in lines[j]:
                            j += 1
                        
                        # 開始讀取指標數據
                        fold_data = {'fold': fold_num}
                        
                        # 讀取 5 行指標數據
                        metric_keys = ['emotion', 'cause', 'pair_m1', 'pair_m2', 'pair_m3']
                        metric_names = ['Emotion', 'Cause', 'Pair (m1)', 'Pair (m2)', 'Pair (m3)']
                        
                        for k, metric_key in enumerate(metric_keys):
                            if j < len(lines):
                                metric_line = lines[j].strip()
                                if any(name in metric_line for name in metric_names):
                                    # 解析這一行的數據
                                    precision, recall, f1 = self._extract_metrics_from_line(metric_line)
                                    fold_data[metric_key] = {
                                        'precision': precision,
                                        'recall': recall,
                                        'f1': f1
                                    }
                                    j += 1
                                else:
                                    break
                        
                        # 檢查是否成功解析了所有指標
                        if len(fold_data) == 6:  # fold + 5 個指標
                            self.fold_results.append(fold_data)
                            metrics_found = True
                        break
                    
                    j += 1
                
                if not metrics_found:
                    print(f"警告：無法找到 Fold {fold_num} 的指標數據")
            
            i += 1
        
        if not self.fold_results:
            print("錯誤：無法解析結果檔案，請檢查檔案格式")
            print("檔案應該包含類似以下格式的內容：")
            print("=== Fold 1 ===")
            print("Metric       | Precision    | Recall       | F1-Score")
            print("Emotion      | 74.36%       | 68.72%       | 71.43%")
            sys.exit(1)
            
        print(f"成功解析 {len(self.fold_results)} 個 fold 的結果")
    
    def _extract_metrics_from_line(self, line: str) -> Tuple[float, float, float]:
        """從指標行中提取 Precision, Recall, F1-Score"""
        # 移除指標名稱，只保留數字部分
        parts = line.split('|')
        
        if len(parts) >= 4:
            try:
                # 提取並清理百分比數字
                precision_str = parts[1].strip().replace('%', '')
                recall_str = parts[2].strip().replace('%', '')
                f1_str = parts[3].strip().replace('%', '')
                
                precision = float(precision_str)
                recall = float(recall_str)
                f1 = float(f1_str)
                
                return precision, recall, f1
            except (ValueError, IndexError) as e:
                print(f"警告：無法解析行：{line}")
                print(f"錯誤詳情：{e}")
                return 0.0, 0.0, 0.0
        else:
            print(f"警告：行格式不正確：{line}")
            return 0.0, 0.0, 0.0
    
    def calculate_averages(self) -> Dict:
        """計算各項指標的平均值"""
        if not self.fold_results:
            raise ValueError("沒有資料可以計算平均值")
            
        num_folds = len(self.fold_results)
        averages = {}
        
        # 定義要計算的指標類別
        metrics = ['emotion', 'cause', 'pair_m1', 'pair_m2', 'pair_m3']
        
        for metric in metrics:
            precision_sum = sum(fold[metric]['precision'] for fold in self.fold_results)
            recall_sum = sum(fold[metric]['recall'] for fold in self.fold_results)
            f1_sum = sum(fold[metric]['f1'] for fold in self.fold_results)
            
            averages[metric] = {
                'precision': precision_sum / num_folds,
                'recall': recall_sum / num_folds,
                'f1': f1_sum / num_folds
            }
            
        return averages
    
    def calculate_standard_deviation(self, averages: Dict) -> Dict:
        """計算標準差"""
        import math
        
        num_folds = len(self.fold_results)
        std_devs = {}
        
        metrics = ['emotion', 'cause', 'pair_m1', 'pair_m2', 'pair_m3']
        
        for metric in metrics:
            avg_precision = averages[metric]['precision']
            avg_recall = averages[metric]['recall']
            avg_f1 = averages[metric]['f1']
            
            precision_var = sum((fold[metric]['precision'] - avg_precision) ** 2 for fold in self.fold_results) / num_folds
            recall_var = sum((fold[metric]['recall'] - avg_recall) ** 2 for fold in self.fold_results) / num_folds
            f1_var = sum((fold[metric]['f1'] - avg_f1) ** 2 for fold in self.fold_results) / num_folds
            
            std_devs[metric] = {
                'precision': math.sqrt(precision_var),
                'recall': math.sqrt(recall_var),
                'f1': math.sqrt(f1_var)
            }
            
        return std_devs
    
    def generate_report(self, averages: Dict, std_devs: Dict = None) -> str:
        """生成格式化的報告"""
        report = []
        report.append("=" * 80)
        report.append(f"{len(self.fold_results)} 折交叉驗證 - 平均評估結果")
        report.append("=" * 80)
        report.append("")
        
        # 表頭
        if std_devs:
            report.append(f"{'指標':<15} | {'Precision':<18} | {'Recall':<18} | {'F1-Score':<18}")
            report.append("-" * 80)
        else:
            report.append(f"{'指標':<15} | {'Precision':<12} | {'Recall':<12} | {'F1-Score':<12}")
            report.append("-" * 60)
        
        # 指標名稱對應
        metric_names = {
            'emotion': 'Emotion',
            'cause': 'Cause',
            'pair_m1': 'Pair (方式一)',
            'pair_m2': 'Pair (方式二)',
            'pair_m3': 'Pair (方式三)'
        }
        
        for metric, name in metric_names.items():
            avg = averages[metric]
            if std_devs:
                std = std_devs[metric]
                report.append(f"{name:<15} | {avg['precision']:6.2f}±{std['precision']:5.2f}% | {avg['recall']:6.2f}±{std['recall']:5.2f}% | {avg['f1']:6.2f}±{std['f1']:5.2f}%")
            else:
                report.append(f"{name:<15} | {avg['precision']:10.2f}% | {avg['recall']:10.2f}% | {avg['f1']:10.2f}%")
        
        if std_devs:
            report.append("-" * 80)
        else:
            report.append("-" * 60)
        
        # 添加詳細的 fold 結果
        report.append("")
        report.append("各 Fold 詳細結果：")
        report.append("")
        
        for fold_data in self.fold_results:
            fold_num = fold_data['fold']
            report.append(f"Fold {fold_num}:")
            for metric, name in metric_names.items():
                data = fold_data[metric]
                report.append(f"  {name:<15}: P={data['precision']:6.2f}%, R={data['recall']:6.2f}%, F1={data['f1']:6.2f}%")
            report.append("")
        
        # 添加最佳表現摘要
        report.append("最佳表現摘要：")
        best_f1_scores = {}
        for metric, name in metric_names.items():
            best_f1 = max(fold[metric]['f1'] for fold in self.fold_results)
            best_fold = next(fold['fold'] for fold in self.fold_results if fold[metric]['f1'] == best_f1)
            best_f1_scores[name] = (best_f1, best_fold)
            report.append(f"  {name:<15}: {best_f1:6.2f}% (Fold {best_fold})")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def save_report(self, report: str, output_file: str, output_dir: str = None) -> None:
        """儲存報告到檔案"""
        # 如果指定了輸出資料夾，則組合完整路徑
        if output_dir:
            import os
            # 確保輸出資料夾存在
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                print(f"建立輸出資料夾：{output_dir}")
            
            # 組合完整路徑
            full_path = os.path.join(output_dir, output_file)
        else:
            full_path = output_file
            
        try:
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"報告已儲存到：{full_path}")
        except Exception as e:
            print(f"儲存報告時發生錯誤：{e}")


def main():
    parser = argparse.ArgumentParser(
        description="計算多折交叉驗證的平均評估指標",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用範例：
  python calculate_fold_averages.py --file test_results.txt
  python calculate_fold_averages.py --file test_results.txt --output average_results.txt
  python calculate_fold_averages.py --file test_results.txt --output average_results.txt --output_dir reports/
  python calculate_fold_averages.py --file test_results.txt --output average_results.txt --output_dir reports/ --std
        """
    )
    
    parser.add_argument('--file', '-f', required=True, help='評估結果檔案路徑')
    parser.add_argument('--output', '-o', help='輸出檔案名稱（可選，預設只顯示在螢幕上）')
    parser.add_argument('--output_dir', '-d', help='輸出資料夾路徑（可選，預設為當前目錄）')
    parser.add_argument('--std', action='store_true', help='是否計算標準差')
    
    args = parser.parse_args()
    
    # 創建計算器實例
    calculator = MetricCalculator()
    
    # 解析結果檔案
    print(f"正在解析檔案：{args.file}")
    calculator.parse_results_file(args.file)
    
    # 計算平均值
    print("正在計算平均值...")
    averages = calculator.calculate_averages()
    
    # 計算標準差（如果需要）
    std_devs = None
    if args.std:
        print("正在計算標準差...")
        std_devs = calculator.calculate_standard_deviation(averages)
    
    # 生成報告
    print("正在生成報告...")
    report = calculator.generate_report(averages, std_devs)
    
    # 顯示報告
    print(report)
    
    # 儲存報告（如果指定輸出檔案）
    if args.output:
        calculator.save_report(report, args.output, args.output_dir)


if __name__ == "__main__":
    main()
