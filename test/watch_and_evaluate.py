#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
自動監控並評估腳本
當偵測到新的 prompt_ECPE_few_shot_ST_* 資料夾時，自動執行評估流程。

使用方式：
    python watch_and_evaluate.py --gt_dir split10_home_train1_test1_val1_unlabeled7_disjoint/
    
    # 指定監控間隔（預設 10 秒）
    python watch_and_evaluate.py --gt_dir split10_few_shot_ST/ --interval 30
    
    # 僅掃描一次（不持續監控）
    python watch_and_evaluate.py --gt_dir split10_few_shot_ST/ --once

功能：
    - 持續監控 UECA_ST 資料夾
    - 偵測到新的 prompt_ECPE_few_shot_ST_* 資料夾時自動執行 run_evaluation.py
    - 記錄已處理過的資料夾，避免重複執行
    - 支援 CE 和 EC 兩種任務類型（自動判斷）
"""

import os
import sys
import time
import json
import argparse
import subprocess
from datetime import datetime
from pathlib import Path


# 設定檔路徑（記錄已處理過的資料夾）
PROCESSED_FOLDERS_FILE = '.processed_folders.json'


def load_processed_folders():
    """載入已處理過的資料夾列表"""
    if os.path.exists(PROCESSED_FOLDERS_FILE):
        try:
            with open(PROCESSED_FOLDERS_FILE, 'r', encoding='utf-8') as f:
                return set(json.load(f))
        except (json.JSONDecodeError, IOError):
            return set()
    return set()


def save_processed_folders(processed):
    """儲存已處理過的資料夾列表"""
    with open(PROCESSED_FOLDERS_FILE, 'w', encoding='utf-8') as f:
        json.dump(list(processed), f, indent=2)


def find_new_folders(watch_dir, processed_folders):
    """
    搜尋新的 prompt_ECPE_few_shot_ST_* 資料夾
    
    判斷條件：
    1. 名稱以 prompt_ECPE_few_shot_ST_ 開頭
    2. 是資料夾（不是檔案）
    3. 尚未被處理過
    4. 包含 test_text_result_fold*.txt 檔案（表示訓練已完成）
    """
    new_folders = []
    
    for item in os.listdir(watch_dir):
        item_path = os.path.join(watch_dir, item)
        
        # 檢查是否為資料夾
        if not os.path.isdir(item_path):
            continue
        
        # 檢查名稱是否符合格式
        if not item.startswith('prompt_ECPE_few_shot_ST_'):
            continue
        
        # 檢查是否已處理過
        if item in processed_folders:
            continue
        
        # 檢查是否包含測試結果檔案（表示訓練已完成）
        has_test_results = any(
            f.startswith('test_text_result_fold') and f.endswith('.txt')
            for f in os.listdir(item_path)
            if os.path.isfile(os.path.join(item_path, f))
        )
        
        if has_test_results:
            new_folders.append(item)
    
    return new_folders


def run_evaluation(gt_dir, pred_dir):
    """執行評估腳本"""
    print(f"\n{'='*70}")
    print(f"[自動評估] 偵測到新資料夾: {pred_dir}")
    print(f"[自動評估] 開始時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")
    
    cmd = [
        sys.executable, 'run_evaluation.py',
        '--gt_dir', gt_dir,
        '--pred_dir', pred_dir
    ]
    
    try:
        result = subprocess.run(cmd, check=True)
        print(f"\n✓ 評估完成: {pred_dir}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ 評估失敗: {pred_dir} (返回碼: {e.returncode})")
        return False
    except Exception as e:
        print(f"\n✗ 發生錯誤: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='自動監控並評估新產生的資料夾',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
    # 持續監控
    python watch_and_evaluate.py --gt_dir split10_home_train1_test1_val1_unlabeled7_disjoint/
    
    # 指定監控間隔為 30 秒
    python watch_and_evaluate.py --gt_dir split10_few_shot_ST/ --interval 30
    
    # 僅掃描一次，不持續監控
    python watch_and_evaluate.py --gt_dir split10_few_shot_ST/ --once
    
    # 重置已處理記錄（重新處理所有資料夾）
    python watch_and_evaluate.py --gt_dir split10_few_shot_ST/ --reset
        """
    )
    parser.add_argument('--gt_dir', type=str, required=True,
                        help='資料集名稱 (Ground Truth 目錄)')
    parser.add_argument('--watch_dir', type=str, default='.',
                        help='要監控的目錄 (預設: 當前目錄)')
    parser.add_argument('--interval', type=int, default=10,
                        help='監控間隔秒數 (預設: 10 秒)')
    parser.add_argument('--once', action='store_true',
                        help='僅掃描一次，不持續監控')
    parser.add_argument('--reset', action='store_true',
                        help='重置已處理記錄')
    
    args = parser.parse_args()
    
    # 切換到監控目錄
    watch_dir = os.path.abspath(args.watch_dir)
    os.chdir(watch_dir)
    
    # 重置已處理記錄
    if args.reset:
        if os.path.exists(PROCESSED_FOLDERS_FILE):
            os.remove(PROCESSED_FOLDERS_FILE)
            print("✓ 已重置處理記錄")
    
    # 載入已處理過的資料夾
    processed_folders = load_processed_folders()
    
    print("="*70)
    print("自動監控評估腳本")
    print("="*70)
    print(f"監控目錄: {watch_dir}")
    print(f"Ground Truth 目錄: {args.gt_dir}")
    print(f"監控間隔: {args.interval} 秒")
    print(f"已處理資料夾數: {len(processed_folders)}")
    print(f"模式: {'單次掃描' if args.once else '持續監控'}")
    print("="*70)
    
    if not args.once:
        print("\n按 Ctrl+C 停止監控\n")
    
    try:
        while True:
            # 搜尋新資料夾
            new_folders = find_new_folders(watch_dir, processed_folders)
            
            if new_folders:
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 發現 {len(new_folders)} 個新資料夾")
                
                for folder in new_folders:
                    success = run_evaluation(args.gt_dir, folder)
                    
                    # 無論成功或失敗都標記為已處理，避免重複嘗試
                    processed_folders.add(folder)
                    save_processed_folders(processed_folders)
                    
                    if success:
                        print(f"  ✓ {folder} - 評估完成")
                    else:
                        print(f"  ✗ {folder} - 評估失敗（已標記為已處理）")
            else:
                if not args.once:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] 沒有新資料夾，等待中...", end='\r')
            
            # 單次模式則結束
            if args.once:
                if not new_folders:
                    print("沒有找到新的未處理資料夾")
                break
            
            # 等待下一次掃描
            time.sleep(args.interval)
            
    except KeyboardInterrupt:
        print("\n\n監控已停止")
        print(f"共處理 {len(processed_folders)} 個資料夾")


if __name__ == '__main__':
    main()
