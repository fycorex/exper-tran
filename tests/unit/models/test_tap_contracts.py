from primary_ml_cka.domain.types import TapContract


def test_tap_contract_is_explicit_and_frozen() -> None:
    contract = TapContract("model", "revision", "path", "extract", "mean", "l2", "bf16", "pending")
    assert contract.module_path == "path"
    assert contract.status == "pending"
