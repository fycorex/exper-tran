from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class PeakMemory:
    allocated_gb: float
    reserved_gb: float


def reset_peak_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def peak_memory() -> PeakMemory:
    gib = 1024**3
    if not torch.cuda.is_available():
        return PeakMemory(0.0, 0.0)
    return PeakMemory(
        torch.cuda.max_memory_allocated() / gib,
        torch.cuda.max_memory_reserved() / gib,
    )
