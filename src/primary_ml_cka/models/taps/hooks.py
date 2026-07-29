from dataclasses import dataclass

import torch


@dataclass(slots=True)
class InputCapture:
    value: torch.Tensor | None = None

    def __call__(self, _module: torch.nn.Module, arguments: tuple[object, ...]) -> None:
        if not arguments or not isinstance(arguments[0], torch.Tensor):
            raise RuntimeError("Tap hook did not receive a tensor input")
        self.value = arguments[0]
