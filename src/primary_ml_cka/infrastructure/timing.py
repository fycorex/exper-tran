from dataclasses import dataclass, field
from time import perf_counter


@dataclass(slots=True)
class Timer:
    started: float = field(default_factory=perf_counter)

    def elapsed(self) -> float:
        return perf_counter() - self.started
