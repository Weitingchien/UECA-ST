#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
自動化評估腳本
訓練完成後，執行此腳本可自動完成以下三個步驟：
1. eval_UECA_CE_v2.py 或 eval_UECA_EC_v2.py - 計算各 fold 的評估指標 (根據資料夾名稱自動判斷)
2. calculate_fold_averages.py - 計算平均值和標準差
3. utils/aggregate_eval_counts.py - 彙整詳細計數統計

使用方式：
    python run_evaluation.py --gt_dir 資料集名稱/ --pred_dir 訓練時產生的資料夾名稱/

範例：
    # CE 任務 (資料夾結尾為 _CE)
    python run_evaluation.py --gt_dir split10_home_train1_test1_val1_unlabeled7_disjoint/ --pred_dir prompt_ECPE_few_shot_ST_2025_..._CE/
    
    # EC 任務 (資料夾結尾為 _EC)
    python run_evaluation.py --gt_dir split10_home_train1_test1_val1_unlabeled7_disjoint/ --pred_dir prompt_ECPE_few_shot_ST_2025_..._EC/

自動判斷規則：
    - 資料夾名稱以 _CE 結尾 → 使用 eval_UECA_CE_v2.py
    - 資料夾名稱以 _EC 結尾 → 使用 eval_UECA_EC_v2.py
"""

import argparse
import subprocess
import sys
import os


def run_command(cmd, description):
    """執行命令並顯示進度"""
    print(f"\n{'='*60}")
    print(f"[Step] {description}")
    print(f"{'='*60}")
    print(f"執行指令: {' '.join(cmd)}\n")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"\n✓ {description} 完成")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {description} 失敗 (返回碼: {e.returncode})")
        return False
    except FileNotFoundError:
        print(f"\n✗ 找不到 Python 或腳本檔案")
        return False


def detect_task_type(pred_dir):
    """
    根據資料夾名稱自動判斷任務類型 (CE 或 EC)
    
    規則：
        - 資料夾名稱包含 _EC（且不含 _CE）→ EC 任務
        - 資料夾名稱包含 _CE → CE 任務
        - 資料夾名稱以 _CE 或 _EC 結尾 → 對應任務
        - 其他情況 → 預設為 CE 任務
    
    Returns:
        str: 'CE' 或 'EC'
    """
    # 取得資料夾名稱（不含路徑）
    folder_name = os.path.basename(pred_dir.rstrip('/').rstrip('\\'))
    
    # 優先檢查結尾（向後相容原有邏輯）
    if folder_name.endswith('_EC'):
        return 'EC'
    elif folder_name.endswith('_CE'):
        return 'CE'
    # 若結尾不符合，則檢查是否包含 _EC 或 _CE
    elif '_EC' in folder_name and '_CE' not in folder_name:
        return 'EC'
    elif '_CE' in folder_name:
        return 'CE'
    else:
        # 預設使用 CE
        print(f"⚠ 無法從資料夾名稱判斷任務類型，預設使用 CE")
        return 'CE'


def main():
    parser = argparse.ArgumentParser(
        description='自動化執行訓練後的三個評估步驟',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
    python run_evaluation.py --gt_dir split10_home_train1_test1_val1_unlabeled7_disjoint/ --pred_dir prompt_ECPE_few_shot_ST_2025_11_27.../
    
這個腳本會自動執行:
    1. python eval_UECA_CE_v2.py 或 eval_UECA_EC_v2.py --gt_dir ... --pred_dir ... (根據資料夾名稱自動判斷)
    2. python calculate_fold_averages.py --file .../test_results.txt --output average_results.txt --output_dir ... --std
    3. python utils/aggregate_eval_counts.py --input .../test_results.txt --output .../counts_summary_detailed.txt
        """
    )
    parser.add_argument('--gt_dir', type=str, required=True,
                        help='資料集名稱 (Ground Truth 目錄)，例如: split10_home_train1_test1_val1_unlabeled7_disjoint/')
    parser.add_argument('--pred_dir', type=str, required=True,
                        help='訓練時產生的資料夾名稱')
    parser.add_argument('--skip_eval', action='store_true',
                        help='跳過 eval_UECA_CE_v2.py (若已執行過)')
    parser.add_argument('--skip_avg', action='store_true',
                        help='跳過 calculate_fold_averages.py')
    parser.add_argument('--skip_counts', action='store_true',
                        help='跳過 aggregate_eval_counts.py')
    parser.add_argument('--aggregate_seeds', type=str, nargs='+', default=None,
                        help='(Step 4) 彙整多個不同 seed 實驗的結果，傳入多個目錄路徑')
    
    args = parser.parse_args()
    
    # 確保路徑格式正確
    gt_dir = args.gt_dir.rstrip('/').rstrip('\\')
    pred_dir = args.pred_dir.rstrip('/').rstrip('\\')
    
    # 檢查目錄是否存在
    if not os.path.isdir(pred_dir):
        print(f"✗ 錯誤: 找不到預測結果目錄 '{pred_dir}'")
        sys.exit(1)
    
    print("\n" + "="*60)
    print("自動化評估腳本")
    print("="*60)
    print(f"Ground Truth 目錄: {gt_dir}")
    print(f"預測結果目錄: {pred_dir}")
    
    # 自動判斷任務類型
    task_type = detect_task_type(pred_dir)
    eval_script = f'eval_UECA_{task_type}_v2.py'
    print(f"偵測到任務類型: {task_type} → 使用 {eval_script}")
    print("="*60)
    
    success_count = 0
    total_steps = 3
    
    # Step 1: eval_UECA_CE_v2.py 或 eval_UECA_EC_v2.py
    if not args.skip_eval:
        cmd1 = [
            sys.executable, eval_script,
            '--gt_dir', gt_dir,
            '--pred_dir', pred_dir
        ]
        if run_command(cmd1, f"Step 1/3: 計算各 fold 評估指標 ({eval_script})"):
            success_count += 1
    else:
        print("\n[跳過] Step 1: eval_UECA_CE_v2.py")
        success_count += 1
    
    # Step 2: calculate_fold_averages.py
    if not args.skip_avg:
        test_results_path = os.path.join(pred_dir, 'test_results.txt')
        if not os.path.isfile(test_results_path):
            print(f"\n✗ 找不到 {test_results_path}，請先執行 Step 1")
        else:
            cmd2 = [
                sys.executable, 'calculate_fold_averages.py',
                '--file', test_results_path,
                '--output', 'average_results.txt',
                '--output_dir', pred_dir,
                '--std'
            ]
            if run_command(cmd2, "Step 2/3: 計算平均值和標準差 (calculate_fold_averages.py)"):
                success_count += 1
    else:
        print("\n[跳過] Step 2: calculate_fold_averages.py")
        success_count += 1
    
    # Step 3: aggregate_eval_counts.py
    if not args.skip_counts:
        test_results_path = os.path.join(pred_dir, 'test_results.txt')
        counts_output_path = os.path.join(pred_dir, 'counts_summary_detailed.txt')
        if not os.path.isfile(test_results_path):
            print(f"\n✗ 找不到 {test_results_path}，請先執行 Step 1")
        else:
            cmd3 = [
                sys.executable, 'utils/aggregate_eval_counts.py',
                '--input', test_results_path,
                '--output', counts_output_path
            ]
            if run_command(cmd3, "Step 3/3: 彙整詳細計數統計 (aggregate_eval_counts.py)"):
                success_count += 1
    else:
        print("\n[跳過] Step 3: aggregate_eval_counts.py")
        success_count += 1
    
    # Step 4: (可選) 彙整多個 seed 實驗結果
    if args.aggregate_seeds is not None:
        total_steps += 1  # 動態增加步驟數
        print("\n" + "="*60)
        print("[Step 4] 彙整多個 seed 實驗結果")
        print("="*60)
        
        # 使用預設的輸出路徑: multi_seed_results/multi_seed_summary.txt
        cmd4 = [
            sys.executable, 'utils/aggregate_multi_seed.py',
            '--dirs'] + args.aggregate_seeds
        if run_command(cmd4, "Step 4: 彙整多個 seed 實驗結果 (aggregate_multi_seed.py)"):
            success_count += 1
    
    # 總結
    print("\n" + "="*60)
    print("執行完成")
    print("="*60)
    print(f"成功: {success_count}/{total_steps} 步驟")
    
    if success_count == total_steps:
        print("\n✓ 所有步驟都已成功完成！")
        print(f"\n輸出檔案位於: {pred_dir}/")
        print(f"  - test_results.txt          (各 fold 詳細結果)")
        print(f"  - average_results.txt       (平均值與標準差)")
        print(f"  - counts_summary_detailed.txt (詳細計數統計)")
        if args.aggregate_seeds is not None:
            print(f"  - multi_seed_summary.txt    (多 seed 彙整結果)")
    else:
        print("\n⚠ 部分步驟失敗，請檢查錯誤訊息")
    
    return 0 if success_count == total_steps else 1


if __name__ == '__main__':
    sys.exit(main())
