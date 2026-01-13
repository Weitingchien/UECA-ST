#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Read UECA test dump file (test_predictions.txt) and compute baseline
6-class micro Precision/Recall/F1 (exclude null class) to verify metrics
independently, matching data_process.acc_prf logic.

Expected dump format (as produced by UECA_CE_emotion_classification_baseline.py):
- Header lines, then repeated blocks of:
  # Sample <idx>
  DocID: <doc_id>, SentID: <sent_id>
  True: <true_idx> (<true_name>)
  Pred: <pred_idx> (<pred_name>) | token: <tok>
  Input IDs: [...]
  Decoded: ...
  ------------------------------------------------------------

Usage:
    wsl.exe python3 compute_weighted_f1_from_dump.py --file emotion_baseline_checkpoints/test_predictions.txt [--report]
"""

import argparse
import os
import re
from typing import List, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

# ^錨定行首、\s*: 0個或多個空白、\d+: 1個或多個數字(樣本編號)
BLOCK_START_RE = re.compile(r"^#\s*Sample\s*\d+")
# \s*: 允許冒號後的空白、(\d+): 捕捉數字、\s*\(: 允許數字後的空白和左括號
TRUE_RE = re.compile(r"^True:\s*(\d+)\s*\(")
PRED_RE = re.compile(r"^Pred:\s*(\d+)\s*\(")
# 當檔案格式沒有括號時，仍能抓到數字，會匹配True: 3，沒有後續的(也可以
TRUE_FALLBACK_RE = re.compile(r"^True:\s*(\d+)")
PRED_FALLBACK_RE = re.compile(r"^Pred:\s*(\d+)")


def parse_dump(path: str) -> Tuple[List[int], List[int]]:
    trues: List[int] = []
    preds: List[int] = []

    if not os.path.exists(path):
        raise FileNotFoundError(f"Dump file not found: {path}")

    with open(path, 'r', encoding='utf-8') as f:
        lines = [ln.rstrip('\n') for ln in f]

    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        if BLOCK_START_RE.match(line):
            # Expect the block to contain True and Pred lines shortly after
            # Scan next ~10 lines to be robust to minor format changes
            true_val = None
            pred_val = None
            for j in range(1, 12):
                if i + j >= n:
                    break
                l2 = lines[i + j]
                m_t = TRUE_RE.match(l2) or TRUE_FALLBACK_RE.match(l2)
                if m_t and true_val is None:
                    true_val = int(m_t.group(1))
                m_p = PRED_RE.match(l2) or PRED_FALLBACK_RE.match(l2)
                if m_p and pred_val is None:
                    pred_val = int(m_p.group(1))
                if true_val is not None and pred_val is not None:
                    break
            if true_val is not None and pred_val is not None:
                trues.append(true_val)
                preds.append(pred_val)
            i += j  # advance some
        else:
            i += 1

    if not trues:
        raise ValueError("No samples parsed from dump; check the file format or path.")

    return trues, preds


def scan_emotion_logs_for_predictions() -> List[str]:
    """掃描 emotion_logs_UECA_EE/ 資料夾下的子資料夾，找尋 test_predictions.txt 檔案"""
    emotion_logs_dir = "emotion_logs_UECA_EE"
    prediction_files = []
    
    if not os.path.exists(emotion_logs_dir):
        print(f"Warning: {emotion_logs_dir} directory not found")
        return prediction_files
    
    for item in os.listdir(emotion_logs_dir):
        item_path = os.path.join(emotion_logs_dir, item)
        if os.path.isdir(item_path):
            test_pred_file = os.path.join(item_path, "test_predictions.txt")
            if os.path.exists(test_pred_file):
                prediction_files.append(test_pred_file)
    
    return sorted(prediction_files)


def compute_metrics_for_file(file_path: str, show_report: bool = False) -> dict:
    """計算單一檔案的指標並回傳結果字典"""
    try:
        y_true, y_pred = parse_dump(file_path)
        y_true_np = np.asarray(y_true)
        y_pred_np = np.asarray(y_pred)
        
        # baseline6: replicate acc_prf logic (6-class micro, exclude 6)
        labels6 = [0, 1, 2, 3, 4, 5]
        tp_sum = fp_sum = fn_sum = 0
        for lab in labels6:
            tp = int(np.sum((y_true_np == lab) & (y_pred_np == lab)))
            fp = int(np.sum((y_true_np != lab) & (y_pred_np == lab)))
            fn = int(np.sum((y_true_np == lab) & (y_pred_np != lab)))
            tp_sum += tp
            fp_sum += fp
            fn_sum += fn
        
        p = tp_sum / (tp_sum + fp_sum + 1e-8)
        r = tp_sum / (tp_sum + fn_sum + 1e-8)
        f1 = 2 * p * r / (p + r + 1e-8)
        
        # Optional: accuracy only on subset where true in 0..5
        mask_true6 = np.isin(y_true_np, labels6)
        acc6 = float(np.mean(y_pred_np[mask_true6] == y_true_np[mask_true6])) if mask_true6.any() else 0.0
        
        results = {
            'file': file_path,
            'samples': len(y_true),
            'accuracy': acc6,
            'precision': p,
            'recall': r,
            'f1': f1,
            'tp_sum': tp_sum,
            'fp_sum': fp_sum,
            'fn_sum': fn_sum
        }
        
        if show_report:
            target_names = ['快樂','悲傷','厭惡','驚訝','恐懼','憤怒','無']
            print(f"\n{file_path} - Per-class report:")
            print(classification_report(y_true_np, y_pred_np, target_names=target_names, digits=4, zero_division=0))
        
        return results
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return {'file': file_path, 'error': str(e)}


def main():
    ap = argparse.ArgumentParser(description="Compute baseline 6-class micro metrics from UECA test dump")
    ap.add_argument('--file', '-f', type=str, help='Path to specific test_predictions.txt (if not provided, scan emotion_logs_UECA_EE/)')
    ap.add_argument('--report', action='store_true', help='Print per-class classification_report')
    ap.add_argument('--scan', action='store_true', help='Force scan emotion_logs_UECA_EE/ even if --file is provided')
    args = ap.parse_args()

    if args.file and not args.scan:
        # 單一檔案模式
        results = compute_metrics_for_file(args.file, args.report)
        if 'error' not in results:
            print("Verification Metrics (baseline 6-class micro, exclude null):")
            print(f"File: {results['file']}")
            print(f"Samples: {results['samples']}")
            print(f"Accuracy (true in 0..5): {results['accuracy']:.4f}")
            print(f"Precision: {results['precision']:.4f} ({results['tp_sum']}/{results['tp_sum'] + results['fp_sum']})")
            print(f"Recall: {results['recall']:.4f} ({results['tp_sum']}/{results['tp_sum'] + results['fn_sum']})")
            print(f"F1-Score: {results['f1']:.4f}")
    else:
        # 掃描模式
        prediction_files = scan_emotion_logs_for_predictions()
        
        if not prediction_files:
            print("No test_predictions.txt files found in emotion_logs_UECA_EE/ subdirectories")
            return
        
        print(f"Found {len(prediction_files)} test_predictions.txt files:")
        print("=" * 80)
        
        all_results = []
        for file_path in prediction_files:
            results = compute_metrics_for_file(file_path, args.report)
            all_results.append(results)
            
            if 'error' not in results:
                print(f"\n📁 {results['file']}")
                print(f"   Samples: {results['samples']}")
                print(f"   Accuracy: {results['accuracy']:.4f}")
                print(f"   Precision: {results['precision']:.4f} ({results['tp_sum']}/{results['tp_sum'] + results['fp_sum']})")
                print(f"   Recall: {results['recall']:.4f} ({results['tp_sum']}/{results['tp_sum'] + results['fn_sum']})")
                print(f"   F1-Score: {results['f1']:.4f}")
            else:
                print(f"\n❌ {results['file']}: {results['error']}")
        
        # 摘要統計
        valid_results = [r for r in all_results if 'error' not in r]
        if valid_results:
            print("\n" + "=" * 80)
            print("📊 [COLING 2022] UECA-Prompt (EE):")
            print("=" * 80)
            accuracies = [r['accuracy'] for r in valid_results]
            precisions = [r['precision'] for r in valid_results]
            recalls = [r['recall'] for r in valid_results]
            f1s = [r['f1'] for r in valid_results]
            
            print(f"Files processed: {len(valid_results)}/{len(all_results)}")
            print(f"Average Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
            print(f"Average Precision: {np.mean(precisions):.4f} ± {np.std(precisions):.4f}")
            print(f"Average Recall: {np.mean(recalls):.4f} ± {np.std(recalls):.4f}")
            print(f"Average F1-Score: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
            print(f"Best F1: {max(f1s):.4f} ({valid_results[f1s.index(max(f1s))]['file']})")


if __name__ == '__main__':
    main()