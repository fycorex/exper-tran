from primary_ml_cka.reporting.summaries import LambdaCandidate, select_positive_lambda


def test_selection_uses_only_ordered_representation_criteria() -> None:
    candidates = (
        LambdaCandidate("P06", 0, 100, 100, 100, 0),
        LambdaCandidate("P06", 0.1, 1, 0.4, 0.6, 2),
        LambdaCandidate("P06", 1, 1, 0.5, 0.5, 1),
    )
    assert select_positive_lambda(candidates).lambda_cka == 1
