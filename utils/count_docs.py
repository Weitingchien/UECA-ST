from pathlib import Path
from collections import Counter
import hashlib
import re


HEADER_RE = re.compile(r"^\s*(\d+)\s+(\d+)\s*$")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def parse_ecpe_file(path: str):
    path = Path(path)
    lines = path.read_text(encoding="utf-8").splitlines()

    blocks = []
    i = 0

    while i < len(lines):
        while i < len(lines) and not lines[i].strip():
            i += 1

        if i >= len(lines):
            break

        header = lines[i].strip()
        m = HEADER_RE.fullmatch(header)
        if not m:
            raise ValueError(f"Invalid header at line {i + 1}: {lines[i]!r}")

        doc_id = m.group(1)
        doc_len = int(m.group(2))

        pair_idx = i + 1
        sent_start = i + 2
        sent_end = sent_start + doc_len

        if pair_idx >= len(lines):
            raise ValueError(f"Missing pair line after doc {doc_id} at line {i + 1}")

        if sent_end > len(lines):
            raise ValueError(
                f"Doc {doc_id} declares doc_len={doc_len}, "
                f"but file ends before enough sentence lines."
            )

        pair_line = lines[pair_idx].strip()
        sentence_lines = lines[sent_start:sent_end]

        text_only = []
        text_with_labels = []

        for offset, sent_line in enumerate(sentence_lines, start=sent_start + 1):
            parts = sent_line.split(",", 3)
            if len(parts) < 4:
                raise ValueError(
                    f"Invalid sentence line at line {offset}: {sent_line!r}"
                )

            sent_id, emo_label, cau_label, text = parts
            text_only.append(text.strip())
            text_with_labels.append(
                f"{sent_id.strip()},{emo_label.strip()},{cau_label.strip()},{text.strip()}"
            )

        blocks.append(
            {
                "doc_id": doc_id,
                "doc_len": doc_len,
                "header_line": i + 1,
                "pair_line": pair_line,
                "text_only_hash": sha256_text("\n".join(text_only)),
                "full_content_hash": sha256_text(
                    pair_line + "\n" + "\n".join(text_with_labels)
                ),
            }
        )

        i = sent_end

    return blocks


def report(name: str, path: str, expected_docs: int | None = None):
    path = Path(path)
    blocks = parse_ecpe_file(path)

    doc_ids = [b["doc_id"] for b in blocks]
    id_counter = Counter(doc_ids)

    text_hashes = [b["text_only_hash"] for b in blocks]
    full_hashes = [b["full_content_hash"] for b in blocks]

    duplicate_ids = {k: v for k, v in id_counter.items() if v > 1}

    print("=" * 80)
    print(name)
    print(f"path: {path.resolve()}")
    print(f"file_size_bytes: {path.stat().st_size}")
    print(f"physical_document_blocks: {len(blocks)}")
    print(f"unique_doc_ids: {len(set(doc_ids))}")
    print(f"duplicate_doc_id_count: {len(duplicate_ids)}")
    print(f"duplicate_doc_blocks_extra: {sum(v - 1 for v in duplicate_ids.values())}")
    print(f"unique_text_only_documents: {len(set(text_hashes))}")
    print(f"unique_full_content_documents: {len(set(full_hashes))}")

    if expected_docs is not None:
        print(f"paper_expected_docs: {expected_docs}")

        candidates = {
            "physical_document_blocks": len(blocks),
            "unique_doc_ids": len(set(doc_ids)),
            "unique_text_only_documents": len(set(text_hashes)),
            "unique_full_content_documents": len(set(full_hashes)),
        }

        matched = [k for k, v in candidates.items() if v == expected_docs]

        if matched:
            print(f"matched_paper_count_by: {matched}")
        else:
            print("matched_paper_count_by: NONE")

    if duplicate_ids:
        print("first_10_duplicate_doc_ids:")
        for doc_id, count in list(duplicate_ids.items())[:10]:
            print(f"  {doc_id}: {count}")


if __name__ == "__main__":
    report(
        "EN-ECPE",
        "domains/Englishnovel_multiple/enecpe_num.txt",
        expected_docs=1226,
    )

    report(
        "RECCON",
        "domains/Englishnovel_multiple/reccon_num.txt",
        expected_docs=780,
    )