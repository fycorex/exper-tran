# Gemma clean-anchored token CKA smoke

This is a three-step, eight-image P22 smoke test (Gemma E4B proxy to Gemma
E2B target) for the corrected contrastive token-CKA path. Eight target-class
references are softly aligned to each clean source image in the frozen proxy
feature space before target CKA is computed. The attack remains proxy-only.

The first `alpha=1` diagnostic completed end to end but failed the mechanism
check: source CKA decreased while aligned-target CKA also decreased. The
fail-fast mechanism gate was then added and the smoke was rerun with `alpha=4`.

| alpha | proxy strict | proxy free | TASR | ASR | source CKA drop | target CKA gain | mechanism |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 0/8 | 1/8 | 0/8 | 1/8 | +0.2621 | -0.0112 | failed target attraction |
| 4 | 1/8 | 2/8 | 0/8 | 0/8 | +0.1764 | +0.0446 | passed both components |

This smoke validates correspondence, gradients, serialization, proxy gates,
and target evaluation. Three attack steps are intentionally insufficient for
the normal 8/8 proxy promotion gate, so its TASR is not a full-budget attack
result.

Reproduce from the repository root:

```bash
PYTHONPATH=src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  .venv-primary-ml-cka/bin/python \
  experiments/2026-08-qwen-transfer-diagnostics/src/run_sweep.py \
  --config experiments/2026-08-qwen-transfer-diagnostics/config/gemma_clean_anchor_cka_smoke.yaml \
  --output-dir outputs/proxy_selector_cka_v2 \
  --fail-on-error
```

The detailed untracked runtime artifacts are written under
`outputs/proxy_selector_cka_v2/diagnostics/gemma_clean_anchor_cka_smoke` and
`outputs/proxy_selector_cka_v2/logs/P22/gemma_clean_anchor_cka_smoke`.
