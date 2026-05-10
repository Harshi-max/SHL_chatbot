from typing import Iterable


def recall_at_k(predicted: Iterable[str], expected: Iterable[str], k: int = 10) -> float:
    pred_list = list(predicted)[:k]
    expected_set = set(expected)
    if not expected_set:
        return 0.0
    hits = sum(1 for item in pred_list if item in expected_set)
    return hits / len(expected_set)


if __name__ == "__main__":
    print(recall_at_k(["A", "B", "C"], ["B", "D"], 10))
