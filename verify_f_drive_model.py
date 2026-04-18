"""
驗證 F 槽 .pth 模型完整性腳本
=================================
對指定的實驗資料夾：
  1. 用 torch.load 載入每個 fold 的 .pth（驗檔案完整）
  2. 跑推論產出新的 text_result.txt（驗模型能正常 forward）
  3. 逐行比對新舊 text_result.txt（驗結果一致）

用法：
    python verify_f_drive_model.py \
      --exp_dir "/mnt/f/experiments/ep_split10_home_t1te1v1_u7_disjoint/prompt_ECPE_few_shot_ST_2025_..." \
      --dataset split10_home_train1_test1_val1_unlabeled7_disjoint/ \
      --device 0

    # 只驗一個 fold（快速測試）：
    python verify_f_drive_model.py \
      --exp_dir "/mnt/f/experiments/ep_split10_home_t1te1v1_u7_disjoint/prompt_ECPE_few_shot_ST_2025_..." \
      --dataset split10_home_train1_test1_val1_unlabeled7_disjoint/ \
      --start_fold 1 --end_fold 1 \
      --device 0
"""

# ============================================================
# 匯入標準函式庫
# ============================================================
import argparse   # 命令列參數解析器
import os          # 作業系統路徑操作（拼接路徑、檢查檔案是否存在等）
import sys         # 系統層級操作（取得 Python 可執行檔路徑、程式退出碼等）
import tempfile    # 建立暫存檔案/目錄（本腳本未直接使用，保留以備後續擴充）
import filecmp     # 檔案比對工具（本腳本改用逐行比對，保留以備後續擴充）
import glob        # 用萬用字元模式搜尋檔案（例如 fold*.pth）

# ============================================================
# 匯入第三方函式庫
# ============================================================
# 載入 prompt_bert 類別定義，torch.load 需要它才能反序列化模型
# （因為模型是用 torch.save(model) 存整個物件，而非只存 state_dict，
#   所以 pickle 反序列化時需要在當前命名空間中找到 prompt_bert 這個類別）
import torch          # PyTorch 深度學習框架
import torch.nn       # PyTorch 神經網路模組（提供 nn.Module 基底類別）
from transformers import BertTokenizer, BertForMaskedLM  # HuggingFace 的 BERT 分詞器與遮罩語言模型


# ============================================================
# prompt_bert 類別定義
# ============================================================
# 此類別必須定義在本腳本中，因為 UECA_CE_few_shot_ST_nest.py 的頂層有
# argparse.parse_args()，直接 import 會觸發參數解析錯誤。
# torch.load 反序列化整個模型物件時，需要在當前命名空間中找到此類別。
class prompt_bert(torch.nn.Module):
    def __init__(self, bert_path='./bert-base-chinese'):
        super(prompt_bert, self).__init__()                        # 呼叫父類別 nn.Module 的初始化
        self.bert = BertForMaskedLM.from_pretrained(bert_path)     # 從預訓練權重載入 BERT 遮罩語言模型
        self.tokenizer = BertTokenizer.from_pretrained(bert_path)  # 從預訓練權重載入 BERT 分詞器
        self.bert.resize_token_embeddings(len(self.tokenizer))     # 調整 embedding 大小以匹配分詞器詞彙量

    def forward(self, x_bert, labels):
        """前向傳播：輸入 token ids 與標籤，回傳 loss 和 logits"""
        output = self.bert(x_bert, labels=labels)       # 將輸入送進 BertForMaskedLM，計算 MLM loss 與 logits
        loss, logits = output.loss, output.logits       # 取出 loss（交叉熵損失）和 logits（每個位置的詞彙機率分佈）
        return loss, logits                             # 回傳給呼叫端

    def get_cls_embeddings(self, x_bert):
        """取得 [CLS] token 的隱藏狀態向量（用於 NeST 的 KNN 計算）"""
        outputs = self.bert.bert(x_bert, output_hidden_states=True, return_dict=True)  # 取得 BERT 編碼器所有層的輸出
        return outputs.last_hidden_state[:, 0, :]       # 回傳最後一層、第 0 個位置（[CLS]）的向量，shape=(batch, hidden_dim)


# ============================================================
# 輔助函式
# ============================================================

def count_pth_files(exp_dir):
    """列出並計算實驗資料夾中的 .pth 檔案"""
    pth_files = sorted(glob.glob(os.path.join(exp_dir, 'fold*.pth')))  # 搜尋所有 fold*.pth 並排序
    return pth_files  # 回傳排序後的檔案路徑清單


def detect_test_model_type(exp_dir):
    """自動偵測原始 text_result.txt 是用哪種模型產生的
    
    背景說明: 
      UECA_CE_few_shot_ST_nest.py 在訓練完成後會用模型跑測試集並儲存 text_result.txt
      如果有進行 self-training，最終的 text_result.txt 是用 self_training_best.pth 產生的;
      如果只有初始監督訓練，則是用 fold{i}.pth 產生的
      
    偵測方法：
      方法 1: 讀取 run_console_output*.txt (訓練時的完整 console log) ，
              搜尋 '載入測試用模型:' 這行，判斷路徑是否包含 'self_training_models'
      方法 2: 若找不到 console log，則檢查資料夾中是否存在 self_training_models/ 子目錄
    """
    # === 方法 1：從 console log 偵測 ===
    console_logs = glob.glob(os.path.join(exp_dir, 'run_console_output*.txt'))  # 搜尋所有 console log 檔案
    for log_file in console_logs:          # 逐一處理每個 log 檔
        try:
            with open(log_file, 'r', encoding='utf-8') as f:  # 以 UTF-8 編碼開啟
                for line in f:             # 逐行掃描 log 內容
                    # 若該行包含「載入測試用模型:」且路徑中有「self_training_models」
                    if '載入測試用模型:' in line and 'self_training_models' in line:
                        return 'self_training'  # 確認是 self-training 模型
                    # 若該行包含「載入測試用模型:」和「fold」但不含「self_training」
                    if '載入測試用模型:' in line and 'fold' in line and 'self_training' not in line:
                        return 'initial'        # 確認是初始監督模型
        except Exception:
            pass  # 忽略讀取錯誤（例如檔案損壞或權限問題）
    
    # === 方法 2：從資料夾結構推測 ===
    st_dir = os.path.join(exp_dir, 'self_training_models')  # 組合 self_training_models 子目錄路徑
    if os.path.isdir(st_dir):              # 若該子目錄存在
        st_files = glob.glob(os.path.join(st_dir, 'fold*_self_training_best.pth'))  # 搜尋 ST 模型檔案
        if st_files:                       # 若找到至少一個 ST 模型
            return 'self_training'         # 推測為 self-training 模型
    
    return 'initial'  # 預設回傳 initial (找不到任何 self-training 相關證據) 


def compare_text_results(original_dir, new_dir, start_fold, end_fold):
    """逐行比對原始 (F 槽) 與新產出 (暫存目錄) 的 text_result.txt
    
    Args:
        original_dir: 原始實驗資料夾路徑（包含 fold{i}_text_result.txt)
        new_dir:      新推論結果的暫存目錄路徑
        start_fold:   起始 fold 編號
        end_fold:     結束 fold 編號
    
    Returns:
        bool: 所有 fold 是否全部一致
    """
    print(f"\n{'='*60}")     # 印出分隔線
    print("Step 3: 比對推論結果")
    print(f"{'='*60}")
    
    all_match = True  # 追蹤所有 fold 是否全部通過，預設為 True
    for fold in range(start_fold, end_fold + 1):  # 遍歷每個 fold
        # 組合原始檔案與新檔案的完整路徑
        original_file = os.path.join(original_dir, f'fold{fold}_text_result.txt')  # F 槽上的原始結果
        new_file = os.path.join(new_dir, f'fold{fold}_text_result.txt')            # 新推論產出的結果
        
        # 檢查原始檔案是否存在
        if not os.path.exists(original_file):
            print(f"  Fold {fold:2d}: ⚠ 原始檔不存在: {original_file}")
            all_match = False  # 標記為不一致
            continue           # 跳過此 fold，繼續下一個
        # 檢查新推論結果是否存在
        if not os.path.exists(new_file):
            print(f"  Fold {fold:2d}: ✗ 新推論結果不存在: {new_file}")
            all_match = False
            continue
        
        # 讀取兩個檔案的所有行
        with open(original_file, 'r', encoding='utf-8') as f1, \
             open(new_file, 'r', encoding='utf-8') as f2:
            orig_lines = f1.readlines()  # 原始檔案的所有行（含換行符）
            new_lines = f2.readlines()   # 新檔案的所有行（含換行符）
        
        # 先比較行數是否相同
        if len(orig_lines) != len(new_lines):
            print(f"  Fold {fold:2d}: ✗ 行數不同 (原始: {len(orig_lines)}, 新: {len(new_lines)})")
            all_match = False
            continue
        
        # 逐行比對內容（去除首尾空白後比較）
        diff_count = 0  # 記錄不同的行數
        for i, (ol, nl) in enumerate(zip(orig_lines, new_lines)):  # 同時遍歷兩個檔案的對應行
            if ol.strip() != nl.strip():   # 去掉頭尾空白後比對
                diff_count += 1            # 累加差異計數
                if diff_count <= 3:        # 最多顯示前 3 處差異（避免輸出過多）
                    print(f"         第 {i+1} 行不同: 原始='{ol.strip()}' vs 新='{nl.strip()}'")
        
        # 輸出此 fold 的比對結果
        if diff_count == 0:
            print(f"  Fold {fold:2d}: ✓ 完全一致 ({len(orig_lines)} 行)")  # 全部相同
        else:
            print(f"  Fold {fold:2d}: ✗ 共 {diff_count} 行不同")           # 有差異
            all_match = False  # 標記為不一致
    
    return all_match  # 回傳最終結果


# ============================================================
# 主程式
# ============================================================

def main():
    # ========== 命令列參數定義 ==========
    parser = argparse.ArgumentParser(description='驗證 F 槽 .pth 模型完整性')
    parser.add_argument('--exp_dir', type=str, required=True,
                        help='F 槽上的實驗資料夾路徑 (包含 fold*.pth 和 fold*_text_result.txt) ')
    parser.add_argument('--dataset', type=str, default='split10_home_train1_test1_val1_unlabeled7_disjoint/',
                        help='資料集路徑 (相對於工作目錄，需包含 fold{i}_test.json 等檔案) ')
    parser.add_argument('--start_fold', type=int, default=1)    # 起始 fold 編號 (預設 1)
    parser.add_argument('--end_fold', type=int, default=10)     # 結束 fold 編號 (預設 10)
    parser.add_argument('--device', type=str, default='0')      # GPU 裝置編號 (預設 '0')
    parser.add_argument('--test_model_type', type=str, default=None,
                        choices=['initial', 'self_training'],
                        help='指定用哪種模型驗證 (預設 None=自動偵測：從 console log 判斷)')
    parser.add_argument('--skip_inference', action='store_true',
                        help='跳過推論步驟，只做 Step 1 的 torch.load 驗證')
    args = parser.parse_args()  # 解析命令列參數，結果存入 args 物件
    
    # 去除路徑尾端的斜線，確保路徑格式一致
    exp_dir = args.exp_dir.rstrip('/').rstrip('\\')
    
    # 檢查實驗資料夾是否存在
    if not os.path.isdir(exp_dir):
        print(f"✗ 找不到資料夾: {exp_dir}")
        sys.exit(1)  # 以錯誤碼 1 退出程式
    
    # ======================================================
    # 偵測 test_model_type（決定要載入哪種 .pth 來跑推論）
    # ======================================================
    # 原因：原始的 text_result.txt 可能由 fold{i}.pth（初始監督模型）或
    #        self_training_models/fold{i}_self_training_best.pth（ST 最佳模型）產生。
    #        如果用錯模型，推論結果會不一致，導致誤判為「檔案損壞」。
    if args.test_model_type is None:           # 使用者未手動指定
        detected_type = detect_test_model_type(exp_dir)  # 呼叫自動偵測函式
        print(f"\n自動偵測: text_result.txt 由 {detected_type} 模型產生")
        test_model_type = detected_type        # 使用偵測結果
    else:                                      # 使用者有手動指定
        test_model_type = args.test_model_type
        print(f"\n手動指定: 使用 {test_model_type} 模型驗證")
    
    # ======================================================
    # Step 1: 驗證 .pth 檔案可正常載入
    # ======================================================
    # 目的：確認 F 槽上的模型檔案沒有在複製過程中損壞，能被 torch.load 正常反序列化。
    print(f"\n{'='*60}")
    print("Step 1: 驗證 .pth 檔案載入")
    print(f"{'='*60}")
    print(f"模型目錄: {exp_dir}")              # 印出模型所在目錄
    print(f"驗證模型類型: {test_model_type}\n")  # 印出要驗證的模型類型
    
    pth_ok = True  # 追蹤所有 .pth 是否都能載入，預設為 True
    
    def verify_pth(path, label):
        """嘗試用 torch.load 載入單一 .pth 檔案，印出結果
        
        Args:
            path:  .pth 檔案的完整路徑
            label: 顯示用的標籤文字 (例如 'Fold  1 (ST best)') 
        """
        nonlocal pth_ok  # 允許修改外層函式的 pth_ok 變數
        if not os.path.exists(path):              # 檢查檔案是否存在
            print(f"  {label}: ✗ 檔案不存在")
            pth_ok = False                        # 標記為失敗
            return
        file_size_mb = os.path.getsize(path) / (1024 * 1024)  # 計算檔案大小（MB）
        try:
            # 將模型載入到 CPU（避免 GPU 記憶體不足）
            model = torch.load(path, map_location=torch.device('cpu'))
            if hasattr(model, 'eval'):            # 確認載入的物件是 nn.Module
                model.eval()                      # 切換到評估模式 (關閉 dropout 等) 
            param_count = sum(p.numel() for p in model.parameters())  # 計算模型總參數量
            print(f"  {label}: ✓ 載入成功 | {file_size_mb:.1f} MB | {param_count:,} 參數")
            del model                             # 釋放記憶體
        except Exception as e:                    # 捕獲所有載入錯誤
            print(f"  {label}: ✗ 載入失敗 | {file_size_mb:.1f} MB | {e}")
            pth_ok = False
    
    # 根據模型類型，驗證對應的 .pth 檔案
    for fold in range(args.start_fold, args.end_fold + 1):  # 遍歷每個 fold
        if test_model_type == 'self_training':
            # Self-training 模型路徑：{exp_dir}/self_training_models/fold{i}_self_training_best.pth
            st_path = os.path.join(exp_dir, 'self_training_models',
                                   f'fold{fold}_self_training_best.pth')
            verify_pth(st_path, f'Fold {fold:2d} (ST best)')
        else:
            # 初始監督模型路徑：{exp_dir}/fold{i}.pth
            init_path = os.path.join(exp_dir, f'fold{fold}.pth')
            verify_pth(init_path, f'Fold {fold:2d} (initial)')
    
    if not pth_ok:                                # 若有任何 .pth 載入失敗
        print("\n✗ 部分 .pth 檔案有問題，中止驗證")
        sys.exit(1)                               # 以錯誤碼退出
    
    if args.skip_inference:                       # 若使用者指定跳過推論
        print("\n✓ 所有 .pth 檔案載入正常（已跳過推論驗證）")
        sys.exit(0)                               # 以成功碼退出
    
    # ======================================================
    # Step 2: 用 .pth 跑推論，輸出到暫存目錄
    # ======================================================
    # 目的：實際載入模型並在測試集上跑推論，產出 text_result.txt，
    #        確認模型不僅能載入，還能正確執行 forward pass。
    print(f"\n{'='*60}")
    print("Step 2: 載入模型跑推論")
    print(f"{'='*60}")
    
    # 從 exp_dir 路徑中擷取實驗資料夾名稱（prompt_ECPE_... 那一段）作為子目錄
    exp_folder_name = os.path.basename(exp_dir)  # 取得路徑最後一層的資料夾名稱
    # 在腳本所在目錄下建立暫存目錄，結構為 _verify_tmp_output/{exp_folder_name}/
    tmp_base = os.path.join(os.path.dirname(os.path.abspath(__file__)),  # 取得本腳本所在的絕對路徑目錄
                            '_verify_tmp_output')                        # 暫存根目錄名稱
    tmp_dir = os.path.join(tmp_base, exp_folder_name)                   # 加上實驗名稱子目錄
    os.makedirs(tmp_dir, exist_ok=True)  # 遞迴建立目錄（若已存在則不報錯）
    print(f"推論結果暫存至: {tmp_dir}\n")
    
    import subprocess  # 匯入子行程模組（用於呼叫 UECA_CE_few_shot_ST_nest.py）
    
    # 組合要執行的命令列指令
    # 透過 subprocess 呼叫 UECA_CE_few_shot_ST_nest.py 的 --test_only 模式
    cmd = [
        sys.executable, 'UECA_CE_few_shot_ST_nest.py',  # 使用當前 Python 直譯器執行訓練腳本
        '--test_only', 'True',               # 僅測試模式（不訓練）
        '--checkpoint', 'True',              # 啟用從檢查點載入模型
        '--checkpointpath', exp_dir,         # 指定檢查點路徑為 F 槽的實驗資料夾
        '--experiment_output_dir', tmp_dir,  # 將推論結果輸出到暫存目錄（而非預設的 save_path）
        '--dataset', args.dataset,           # 指定資料集路徑（需包含 fold{i}_test.json）
        '--start_fold', str(args.start_fold),  # 起始 fold
        '--end_fold', str(args.end_fold),      # 結束 fold
        '--device', args.device,             # GPU 裝置編號
        '--test_model_type', test_model_type,  # 指定載入模型類型（initial 或 self_training）
    ]
    
    print(f"使用模型類型: {test_model_type}")          # 印出使用的模型類型
    print(f"執行指令:\n  {' '.join(cmd)}\n")          # 印出完整的命令列指令
    result = subprocess.run(cmd)                      # 以子行程方式執行指令，等待完成
    
    if result.returncode != 0:                        # 檢查子行程的返回碼
        print(f"\n✗ 推論執行失敗 (返回碼: {result.returncode})")
        sys.exit(1)                                   # 推論失敗，以錯誤碼退出
    
    # ======================================================
    # Step 3: 比對結果
    # ======================================================
    # 目的：逐行比對 F 槽上原始的 text_result.txt 與新推論產出的 text_result.txt，
    #        確認兩者完全一致，證明模型檔案在複製過程中沒有損壞。
    all_match = compare_text_results(exp_dir, tmp_dir, args.start_fold, args.end_fold)
    
    # ======================================================
    # 總結：輸出最終驗證結果
    # ======================================================
    print(f"\n{'='*60}")
    if all_match:
        print("✓ 驗證通過！F 槽模型完整，推論結果與原始完全一致。")
    else:
        print("✗ 驗證未通過，部分結果不一致。")
    print(f"{'='*60}")
    
    # 詢問使用者是否清理暫存目錄
    cleanup = input("\n是否刪除暫存目錄? (y/n): ").strip().lower()  # 等待使用者輸入
    if cleanup == 'y':
        import shutil                                    # 匯入高階檔案操作模組
        shutil.rmtree(tmp_dir, ignore_errors=True)       # 遞迴刪除暫存目錄及其所有內容
        print("已清理暫存目錄")
    else:
        print(f"暫存結果保留在: {tmp_dir}")              # 告知使用者暫存目錄位置
    
    sys.exit(0 if all_match else 1)  # 驗證全部通過回傳 0，否則回傳 1


# 當此檔案作為主程式直接執行時（而非被 import），呼叫 main()
if __name__ == '__main__':
    main()
