import torch


def require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for real experiment commands")
    return torch.device("cuda")
