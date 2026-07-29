import pytest


@pytest.mark.gpu
@pytest.mark.smoke
def test_proxy_adapter_contract_requires_real_gpu_run() -> None:
    pytest.skip("Executed by `attack smoke` after per-pair model loading")
