from types import MethodType, SimpleNamespace

import torch
import torch.nn.functional as functional

from primary_ml_cka.models.proxies.generative import GenerativeProxy


def test_generative_proxy_selects_configured_target_answer() -> None:
    proxy = GenerativeProxy.__new__(GenerativeProxy)
    proxy.class_margin = 2.0
    proxy.rank_weight = 1.0
    proxy.suppression_weight = 1.0

    def fake_answer_score(
        self: GenerativeProxy,
        images: torch.Tensor,
        prompt: str,
        answer: str,
    ) -> tuple[torch.Tensor, str, object]:
        label = int(answer)
        mask = SimpleNamespace(
            answer_token_ids=(100 + label,),
            label_positions=(label,),
        )
        score = torch.full((images.shape[0],), float(label), device=images.device)
        return score, f"rendered:{answer}", mask

    proxy._answer_score = MethodType(fake_answer_score, proxy)
    images = torch.rand(2, 3, 16, 16)
    output = proxy.target_loss(images, human_target_label=3, prompt="prompt")

    expected_logits = torch.arange(1, 11, dtype=torch.float32).repeat(2, 1)
    expected_targets = torch.full((2,), 2, dtype=torch.long)
    assert torch.allclose(
        output.target_nll,
        functional.cross_entropy(expected_logits, expected_targets),
    )
    assert output.answer_token_ids == (103,)
    assert output.label_positions == (3,)
    assert output.rendered_prompt == "rendered:3"
