import argparse
import json
import os
from typing import List, Dict, Any

from transformers import BertTokenizer


def build_mask_document(doc: Dict[str, Any]) -> str:
    """Replicate MyDataset 的 mask_full_document 建構方式。"""
    clauses = doc.get("clauses", [])
    segments: List[str] = []
    for idx, clause in enumerate(clauses, start=1):
        clause_text = clause.get("clause", "")
        # 與 UECA_CE 系列腳本一致：子句序號 + 內容 + 三個 [MASK] + [SEP]
        segments.append(f" {idx} {clause_text} [MASK] [MASK] [MASK] [SEP]")
    return "".join(segments).strip()


def inspect_file(json_path: str, tokenizer: BertTokenizer, num_samples: int, show_doc: bool) -> None:
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"找不到資料檔案: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        docs = json.load(f)

    if not isinstance(docs, list):
        raise ValueError("資料檔案格式錯誤：應為文件列表")

    cls_id = tokenizer.cls_token_id
    total = 0
    with_cls = 0
    overflow = 0

    print(f"檢查檔案: {json_path}")
    print(f"BERT 模型: {tokenizer.name_or_path}")
    print("-" * 60)

    for doc in docs:
        mask_text = build_mask_document(doc)
        raw_text = mask_text
        # 先用未截斷的編碼計算實際長度，與正式資料處理流程一致
        full_ids = tokenizer.encode_plus(mask_text, return_tensors="pt")["input_ids"][0]
        if len(full_ids) > 512:
            overflow += 1
            continue

        encoded = tokenizer.encode_plus(
            mask_text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            pad_to_max_length=True,
        )["input_ids"][0]

        first_id = int(encoded[0].item())
        has_cls = first_id == cls_id
        total += 1
        if has_cls:
            with_cls += 1

        if total <= num_samples:
            tokens_preview = tokenizer.convert_ids_to_tokens(encoded[:8].tolist())
            print(f"doc_id={doc.get('doc_id', 'N/A')}, first_id={first_id}, has_cls={has_cls}")
            print("  preview:", tokens_preview)
            if show_doc:
                print("  raw:", raw_text)
                print("  encoded tokens:")
                print("   ", tokenizer.convert_ids_to_tokens(encoded.tolist()))

    print("-" * 60)
    print(f"可檢查樣本數: {total}")
    print(f"含 [CLS] 的樣本: {with_cls} ({with_cls / total * 100:.2f}% )")
    print(f"因長度 >512 而被跳過的樣本: {overflow}")

    if with_cls == total and total > 0:
        print("結果：所有輸入序列皆以 [CLS] 開頭。")
    elif total == 0:
        print("結果：沒有任何樣本可供檢查，請確認資料檔。")
    else:
        missing = total - with_cls
        print(f"結果：找到 {missing} 筆未以 [CLS] 開頭的樣本，請再檢查資料流程。")


def main():
    parser = argparse.ArgumentParser(description="檢查 UECA 資料是否包含 [CLS] token")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="split10_few_shot_ST",
        help="資料夾路徑，例如 split10_few_shot_ST",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "test", "unlabeled"],
        help="要檢查的資料 split",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=1,
        help="折數 (用於拼出資料檔名，例如 fold1_train.json)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="輸出細節的樣本數 (其餘僅統計)",
    )
    parser.add_argument(
        "--show-doc",
        action="store_true",
        help="輸出樣本的完整模板文字與全部 token 序列",
    )
    parser.add_argument(
        "--bert-path",
        type=str,
        default="./bert-base-chinese",
        help="BERT tokenizer 路徑",
    )

    args = parser.parse_args()

    filename = f"fold{args.fold}_{args.split}.json"
    json_path = os.path.join(args.dataset_path, filename)

    tokenizer = BertTokenizer.from_pretrained(args.bert_path)
    inspect_file(json_path, tokenizer, num_samples=args.num_samples, show_doc=args.show_doc)


if __name__ == "__main__":
    main()
