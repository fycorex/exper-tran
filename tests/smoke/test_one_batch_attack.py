import pytest


@pytest.mark.gpu
@pytest.mark.smoke
def test_one_batch_attack_requires_validated_taps() -> None:
    pytest.skip("Executed by `attack smoke` only after correctness and tap gates")
