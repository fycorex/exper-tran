import pytest


@pytest.mark.gpu
@pytest.mark.smoke
def test_target_adapter_contract_requires_real_gpu_run() -> None:
    pytest.skip(
        "Target contract is black-box generation only; executed during screening/evaluation"
    )
