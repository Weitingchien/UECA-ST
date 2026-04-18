import json
import os

base_dir = "split10_train1_test1_val1_unlabeled7_disjoint"
files = ["fold1_train.json", "fold1_test.json", "fold1_val.json", "fold1_unlabeled.json"]

print(f"{'File':<25} {'Count':<10}")
print("-" * 35)

total = 0
counts = {}

for f in files:
    path = os.path.join(base_dir, f)
    try:
        with open(path, 'r', encoding='utf-8') as fp:
            data = json.load(fp)
            count = len(data)
            counts[f] = count
            total += count
            print(f"{f:<25} {count:<10}")
    except Exception as e:
        print(f"{f:<25} Error: {e}")

print("-" * 35)
print(f"{'Total':<25} {total:<10}")

if total > 0:
    print("\nRatios:")
    for f in files:
        print(f"{f:<25} {counts[f]/total:.4f} ({counts[f]})")
