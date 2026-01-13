"""
快速測試 CE 與 EC 版本 MyDataset 的差異
直接 import 兩個腳本的 MyDataset 類來比較實際輸出
使用 split10_home_train1_test1_val1_unlabeled7_disjoint/fold1_train.json 真實資料
"""

import sys
import json
import os

# 模擬命令列參數，避免 argparse 報錯
sys.argv = ['test', '--test_only', 'True']

# 動態載入模組
import importlib.util

def load_module(module_name, file_path):
    """動態載入模組"""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

def test_real_mydataset():
    """使用實際的 MyDataset 類比較 CE 與 EC 版本"""
    
    # 使用真實的測試資料
    real_data_path = './split10_home_train1_test1_val1_unlabeled7_disjoint/fold1_train.json'
    
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained('./bert-base-chinese')
    
    # 讀取原始資料以顯示 pairs 資訊
    with open(real_data_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    # 取第一個短一點的樣本 (doc_id="2104"，11個子句)
    test_doc = raw_data[15]  # 第15個樣本: doc_id="2104", pairs=[[8,5], [8,6], [8,7]]
    
    print("=" * 70)
    print(f"測試樣本: doc_id={test_doc['doc_id']}, doc_len={test_doc['doc_len']}")
    print(f"pairs = {test_doc['pairs']}")
    if test_doc['pairs']:
        for emo_idx, cause_idx in test_doc['pairs']:
            print(f"  → 句子{emo_idx} 是情緒句，句子{cause_idx} 是原因句")
    print("-" * 70)
    for i, clause in enumerate(test_doc['clauses'], 1):
        role = []
        for emo_idx, cause_idx in test_doc['pairs']:
            if i == emo_idx:
                role.append("情緒句")
            if i == cause_idx:
                role.append("原因句")
        role_str = f" ← {', '.join(role)}" if role else ""
        print(f"句子{i}: {clause['clause']}{role_str}")
    print("=" * 70)
    
    # 創建只包含這一筆樣本的臨時 JSON
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
        json.dump([test_doc], f, ensure_ascii=False)
        temp_json_path = f.name
    
    try:
        # 載入 CE 版本
        print("\n【載入 CE 版本 MyDataset】")
        ce_module = load_module('ueca_ce', './UECA_CE_few_shot_ST_nest.py')
        ce_dataset = ce_module.MyDataset(temp_json_path, tokenizer=tokenizer)
        
        # 清除快取並載入 EC 版本
        print("\n【載入 EC 版本 MyDataset】")
        if 'ueca_ec' in sys.modules:
            del sys.modules['ueca_ec']
        ec_module = load_module('ueca_ec', './UECA_EC_few_shot_ST_nest.py')
        ec_dataset = ec_module.MyDataset(temp_json_path, tokenizer=tokenizer)
        
        # 比較 label (即 [MASK] 位置應預測的 token ids)
        print("\n" + "=" * 70)
        print("【比較 CE vs EC 的 [MASK] 預測目標】")
        print("=" * 70)
        
        ce_labels = ce_dataset.label[0]
        ec_labels = ec_dataset.label[0]
        
        # 取得非 -100 的 label (即 [MASK] 位置)
        ce_mask_tokens = ce_labels[ce_labels != -100]
        ec_mask_tokens = ec_labels[ec_labels != -100]
        
        # 解碼成文字
        num_clauses = test_doc['doc_len']
        ce_decoded = tokenizer.convert_ids_to_tokens(ce_mask_tokens[:num_clauses*3].tolist())
        ec_decoded = tokenizer.convert_ids_to_tokens(ec_mask_tokens[:num_clauses*3].tolist())
        
        # 逐句比較
        print("\n格式: [情緒?] [原因?] [配對編號]")
        print("-" * 70)
        for i in range(num_clauses):
            start = i * 3
            end = start + 3
            ce_tokens = ce_decoded[start:end]
            ec_tokens = ec_decoded[start:end]
            diff_marker = " ← 差異!" if ce_tokens != ec_tokens else ""
            print(f"句子{i+1}: CE={ce_tokens}  EC={ec_tokens}{diff_marker}")
        
        print("\n" + "=" * 70)
        print("【驗證結論】")
        print("-" * 70)
        if test_doc['pairs']:
            for emo_idx, cause_idx in test_doc['pairs']:
                print(f"CE: 原因句(句子{cause_idx}) 第3個MASK 應預測 '{emo_idx}' (指向情緒句)")
                print(f"EC: 情緒句(句子{emo_idx}) 第3個MASK 應預測 '{cause_idx}' (指向原因句)")
        print("=" * 70)
        
    finally:
        os.unlink(temp_json_path)


if __name__ == "__main__":
    test_real_mydataset()
