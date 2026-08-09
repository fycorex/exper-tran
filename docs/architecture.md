# Architecture

## Threat Model

Attack generation is proxy-only. Its process loads one proxy and has no target
model, target encoder, target logits, target embeddings, or target gradients.
The target label generator is used for canonical clean screening and frozen-PNG
evaluation. Only decoded text crosses that boundary.

After attacks and target outputs are frozen, `models/analysis/` may load target
visual encoders sequentially to compute model-similarity CKA. No module under
`attack/` imports analysis or target code. On the 16GB A4000, proxy and target
models are never resident at the same time.

## Dependency Boundaries

```text
domain → config/data/prompts/infrastructure → models → attack → experiment
       → evaluation/reporting
```

`tests/integration/test_import_boundaries.py` enforces that attack modules do
not import target generation, target analysis, or target backends.

## Attack Data Flow

For each logical batch, the proxy produces per-image token tensors:

```text
H_clean     = proxy visual tokens of canonical clean images
H_adv       = proxy visual tokens of current adversarial images
H_reference = proxy visual-token bank of target-class references
```

CLIP/SigLIP use native text encoders and logit scales. Generative proxies
teacher-force exact answers `"1"`–`"10"`; mean answer-token log probabilities
become closed-set logits. Only frozen clean/adversarial PNG paths cross into
target evaluation.
