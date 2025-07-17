import json
import random
import os

# --- 設定 ---
# 原始資料集的路徑
source_dataset_path = 'split10/' 
# 新的 few-shot 資料集要儲存的路徑
output_dataset_path = 'split10_few_shot/' 
# 要縮減的比例
sample_ratio = 0.1 

# 確保輸出檔案夾存在
os.makedirs(output_dataset_path, exist_ok=True)

print(f"將從 '{source_dataset_path}' 創建 {sample_ratio*100}% 的 few-shot 資料集，並儲存至 '{output_dataset_path}'")

# 假設有 10 個 fold
for i in range(1, 11):
    train_file_name = f'fold{i}_train.json'
    input_file_path = os.path.join(source_dataset_path, train_file_name)

    if not os.path.exists(input_file_path):
        print(f"警告：找不到檔案 {input_file_path}")
        continue

    # 讀取原始的訓練檔案
    with open(input_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 計算取樣數量
    num_samples = int(len(data) * sample_ratio)
    
    # 隨機且不重複地抽取 10% 的資料
    # 使用 random.sample 確保公平抽樣
    few_shot_data = random.sample(data, num_samples)

    # 定義新的 few-shot 檔案名稱
    output_file_name = f'fold{i}_train_10_percent.json'
    output_file_path = os.path.join(output_dataset_path, output_file_name)

    # 將抽樣後的資料寫入新的 JSON 檔案
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(few_shot_data, f, ensure_ascii=False, indent=4)
    
    print(f"已生成: {output_file_name}，包含 {len(few_shot_data)} / {len(data)} 筆資料")

    # 同時，將 test 檔案也複製過去，方便管理
    test_file_name = f'fold{i}_test.json'
    source_test_path = os.path.join(source_dataset_path, test_file_name)
    dest_test_path = os.path.join(output_dataset_path, test_file_name)
    if os.path.exists(source_test_path):
        import shutil
        shutil.copy(source_test_path, dest_test_path)

print("\nFew-shot 資料集創建完成！")