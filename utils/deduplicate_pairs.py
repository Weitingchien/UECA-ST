import argparse
import json
from collections import Counter
from pathlib import Path


def unique_pairs(pairs):
    seen = set()
    result = []
    for pair in pairs:
        tpl = tuple(pair)
        if tpl not in seen:
            seen.add(tpl)
            result.append(list(tpl))
    return result


def collect_multi_emotion_pairs(doc):
    clause_map = {str(clause.get("clause_id")): clause for clause in doc.get("clauses", [])}
    pairs = unique_pairs(doc.get("pairs", []) or [])

    expanded = []
    for emotion_idx, cause_idx in pairs:
        clause = clause_map.get(str(emotion_idx))
        category = (clause.get("emotion_category") if clause else "") or ""
        tokens = [token.strip() for token in category.split("&") if token.strip()]
        if not tokens:
            expanded.append([emotion_idx, cause_idx])
        else:
            for token in tokens:
                expanded.append([emotion_idx, cause_idx])
    return expanded


def process_file(source_path: Path, dest_path: Path, split_multi_emotion: bool):
    with source_path.open(encoding="utf-8") as handle:
        docs = json.load(handle)

    dup_counter = Counter()
    updated_docs = []

    for doc in docs:
        original_pairs = doc.get("pairs", []) or []
        deduped_pairs = unique_pairs(original_pairs)

        if split_multi_emotion:
            normalized_pairs = collect_multi_emotion_pairs(doc)
        else:
            normalized_pairs = deduped_pairs

        if len(original_pairs) != len(normalized_pairs):
            dup_counter[len(original_pairs) - len(normalized_pairs)] += 1

        doc = dict(doc)
        doc["pairs"] = normalized_pairs
        updated_docs.append(doc)

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with dest_path.open("w", encoding="utf-8") as handle:
        json.dump(updated_docs, handle, ensure_ascii=False, indent=2)

    return dup_counter


def main():
    parser = argparse.ArgumentParser(description="Create a normalized copy of ECPE split files")
    parser.add_argument("--source", default="split10", help="Path to the original split directory")
    parser.add_argument("--destination", default="split10_pairs_normalized", help="Directory for normalized output")
    parser.add_argument(
        "--split-multi-emotion",
        action="store_true",
        help="Expand pairs when emotion category contains multiple labels separated by '&'",
    )
    args = parser.parse_args()

    source_root = Path(args.source)
    destination_root = Path(args.destination)

    if not source_root.exists():
        raise FileNotFoundError(f"Source directory not found: {source_root}")

    aggregate_counter = Counter()
    json_files = sorted(source_root.glob("*.json"))

    for path in json_files:
        relative = path.relative_to(source_root)
        destination_path = destination_root / relative
        counter = process_file(path, destination_path, split_multi_emotion=args.split_multi_emotion)
        aggregate_counter.update(counter)
        print(f"Processed {relative}")

    if aggregate_counter:
        print("\nSummary of pair adjustments (original minus normalized per document):")
        for diff, count in sorted(aggregate_counter.items()):
            print(f"  removed {diff} duplicate entries in {count} documents")
    else:
        print("No duplicate pairs were removed.")


if __name__ == "__main__":
    main()
