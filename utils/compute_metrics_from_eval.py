import argparse
import re
from pathlib import Path


def parse_counts(line: str, label: str) -> tuple[int, int, int]:
    """Extract correct, predicted, and ground-truth counts for a given label."""
    patterns = {
        "Emotion": r"Emotion:\s*[^:]*?數量:(\d+)\s+預測出情緒的數量:(\d+)\s+實際正確情緒的數量:(\d+)",
        "Cause": r"Cause:\s*[^:]*?數量:(\d+)\s+預測出原因的數量:(\d+)\s+實際正確原因的數量:(\d+)",
        "Pair": r"Pair:\s*[^:]*?數量:(\d+)\s+預測出組合的數量:(\d+)\s+實際正確組合的數量:(\d+)",
    }
    match = re.search(patterns[label], line)
    if not match:
        raise ValueError(f"Unable to parse counts for {label}. Line: {line.strip()}")
    correct, predicted, actual = map(int, match.groups())
    return correct, predicted, actual


def compute_metrics(correct: int, predicted: int, actual: int) -> dict[str, float]:
    """Compute precision, recall, and F1 based on counts."""
    precision = correct / predicted if predicted else 0.0
    recall = correct / actual if actual else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def describe_metrics(label: str, correct: int, predicted: int, actual: int) -> None:
    """Print the numerator/denominator relationships in a human-friendly format."""
    metrics = compute_metrics(correct, predicted, actual)
    precision = metrics["precision"]
    recall = metrics["recall"]
    f1 = metrics["f1"]

    print(f"{label}:")
    print(f"  Precision = {correct} / {predicted} = {precision:.4f}")
    print(f"  Recall    = {correct} / {actual} = {recall:.4f}")
    denom = predicted + actual
    print(f"  F1        = 2 * {correct} / ({predicted} + {actual}) = {f1:.4f}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect numerator/denominator for test evaluation metrics.")
    parser.add_argument("evaluation_file", type=Path, help="Path to fold{i}_test_evaluation.txt")
    args = parser.parse_args()

    if not args.evaluation_file.is_file():
        raise FileNotFoundError(f"File not found: {args.evaluation_file}")

    lines = args.evaluation_file.read_text(encoding="utf-8").splitlines()

    label_counts = {}
    for label in ("Emotion", "Cause", "Pair"):
        matching_line = next((line for line in lines if line.startswith(label)), None)
        if not matching_line:
            raise ValueError(f"Missing summary line for {label} in {args.evaluation_file}")
        label_counts[label] = parse_counts(matching_line, label)

    print(f"Evaluation file: {args.evaluation_file}")
    for label, counts in label_counts.items():
        describe_metrics(label, *counts)


if __name__ == "__main__":
    main()
