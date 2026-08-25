# Semantic Contrastive V3 — eight-image summary

## P20 results

| Loss | Proxy | TASR | ASR | Target gain | Source drop | Gap gain |
|---|---:|---:|---:|---:|---:|---:|
| cls_only | 8/8 | 2/8 | 6/8 | -0.0171 | n/a | n/a |
| semantic_only | 5/8 | 2/8 | 5/8 | 0.1752 | n/a | n/a |
| cls_plus_semantic | 8/8 | 1/8 | 5/8 | 0.1743 | n/a | n/a |
| contrastive_only | 8/8 | 0/8 | 5/8 | -0.4283 | 0.8225474655628204 | 0.39425233006477356 |
| cls_plus_contrastive | 8/8 | 4/8 | 7/8 | -0.3958 | 0.7796188816428185 | 0.3838227093219757 |
| mean_reference_only | 8/8 | 0/8 | 4/8 | -0.3943 | 0.7481198757886887 | 0.3538491725921631 |

## Diagnostic answers

1. CLS is not universally helpful: CLS-only reached 2/8 TASR; adding CLS to target-only semantic reduced 2/8 to 1/8, while adding CLS to prototype contrastive increased 0/8 to 4/8.
2. Source-target contrastive only reached strict proxy 8/8 but 0/8 TASR. Its value appears in interaction with CLS, where the combined arm reached 4/8.
3. Prototype-only and mean-reference-only both reached 0/8 TASR; prototype had 5/8 ASR versus 4/8 for mean-reference. No TASR advantage is established between the two representation-only modes.
4. CLS pixel gradients are not extremely small. The detailed combined run records target-token, CE, margin, total-CLS, and representation gradients at steps 0/25/50/99.
5. The initial CLS/representation gradient cosine is weakly positive rather than strongly conflicting; see gradient_trace.csv for its evolution.
6. P20 uses Qwen Vision Encoder block 17 of 24, valid-token mean pooling, with no target-model representation access.
7. Depth materially changes semantic separability: source-target prototype cosine was 0.952 (layer 12), 0.929 (layer 17), and 0.99996 (layer 23).

These are eight-image diagnostic results, not final statistical claims.
