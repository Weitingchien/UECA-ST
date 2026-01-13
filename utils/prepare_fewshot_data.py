import argparse
import json
import os
import random

def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a few-shot subset for each fold by random sampling."
    )
    parser.add_argument(
        "--source",
        dest="source_dataset_path",
        default="split10/",
        help="Input dataset folder containing fold{i}_train.json and fold{i}_test.json."
    )
    parser.add_argument(
        "--output",
        dest="output_dataset_path",
        default="split10_few_shot/",
        help="Destination folder to store the few-shot dataset files."
    )
    parser.add_argument(
        "--sample-ratio",
        type=float,
        default=0.1,
        help="Portion of samples to keep in each training fold (0 < ratio ≤ 1)."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Optional random seed for reproducible sampling."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    source_dataset_path = args.source_dataset_path
    output_dataset_path = args.output_dataset_path
    sample_ratio = args.sample_ratio

    if not 0 < sample_ratio <= 1:
        raise ValueError("--sample-ratio must be within (0, 1].")

    if args.seed is not None:
        random.seed(args.seed)

    # 確保輸出檔案夾存在
    os.makedirs(output_dataset_path, exist_ok=True)

    print(
        f"將從 '{source_dataset_path}' 創建 {sample_ratio * 100:.1f}% 的 few-shot 資料集，"
        f"並儲存至 '{output_dataset_path}'"
    )

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
    
        if num_samples <= 0:
            print(f"警告: fold{i} 的樣本數 {len(data)} 過少，無法依比例 {sample_ratio} 取得樣本。")
            continue

        # 隨機且不重複地抽取資料
        few_shot_data = random.sample(data, num_samples)

        # 定義新的 few-shot 檔案名稱
        output_file_name = f'fold{i}_train.json'
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


if __name__ == "__main__":
    main()