# Architecture

## Threat Model

The attack is proxy-only. During optimization, the process loads one proxy and
has no target model, target encoder, target logits, target embeddings, or target
gradients. The target is a black-box label generator used only for:

1. clean screening before attack generation; and
2. evaluation after adversarial PNGs are frozen.

`models/targets/` exposes decoded text through `TargetGenerator`. Local
Transformers generation is an experimental stand-in for a remote closed API;
`BlackBoxTargetAPI` supports an actual external service with the same narrow
interface.

## Dependency Boundaries

The package follows:

```text
domain → config/data/prompts/infrastructure → models → attack → experiment
       → evaluation/reporting
```

Attack modules cannot import target generation, vLLM, or target-model modules.
`tests/integration/test_import_boundaries.py` enforces this at source level.

## Attack Data Flow

For each batch, the proxy produces:

```text
Z_clean     = proxy image embeddings of clean source images
Z_adv       = proxy image embeddings of current adversarial images
Z_reference = proxy image embeddings of configured target-class references
```

The proxy also produces ten class logits. CLIP/SigLIP use their native text
encoder and logit scale. Generative proxies teacher-force exact answers
`"1"`–`"10"`; sequence log-probabilities become class logits.

Only frozen PNG paths cross from attack generation into target evaluation.
