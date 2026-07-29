from collections import Counter


def confusion_counts(labels: tuple[int | None, ...]) -> dict[str, int]:
    return dict(Counter("invalid" if label is None else str(label) for label in labels))
