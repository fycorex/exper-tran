import pytest
import torch


def pytest_sessionstart(session: pytest.Session) -> None:
    if not torch.cuda.is_available():
        raise pytest.UsageError("CUDA is mandatory; CPU test execution is forbidden")
    torch.set_default_device("cuda")


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    torch.set_default_device("cpu")
