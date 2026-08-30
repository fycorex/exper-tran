"""Pure helpers for the controlled 50-image confirmation run."""

from collections.abc import Sequence


def batch_slices(count: int, batch_size: int = 8) -> tuple[tuple[int, int], ...]:
    if count < 1 or batch_size < 1:
        raise ValueError("count and batch_size must be positive")
    return tuple(
        (start, min(start + batch_size, count))
        for start in range(0, count, batch_size)
    )


def conditional_hits(
    proxy_masks: Sequence[Sequence[bool]],
    target_masks: Sequence[Sequence[bool]],
) -> tuple[int, int]:
    if len(proxy_masks) != len(target_masks):
        raise ValueError("Proxy and target batches must align")
    numerator = 0
    denominator = 0
    for proxy, target in zip(proxy_masks, target_masks, strict=True):
        if len(proxy) != len(target):
            raise ValueError("Proxy and target masks must align")
        numerator += sum(bool(p) and bool(t) for p, t in zip(proxy, target, strict=True))
        denominator += sum(bool(value) for value in proxy)
    return numerator, denominator
