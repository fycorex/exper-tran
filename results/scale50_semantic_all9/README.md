# Controlled 50-image semantic attack results

This bundle archives the completed nine-pair, 50-image semantic-loss scale-up.
All pairs use the same all-model-consensus clean cohort. The attack uses 100
steps, epsilon 16/255, initial auxiliary/classification gradient ratio 0.3,
and a fixed 48-reference semantic target bank. P22 uses batch size four to fit
the A4000; the other pairs use batch size eight.

| Pair | Proxy hits | TASR | ASR |
| --- | ---: | ---: | ---: |
| P02 Qwen4B to GemmaE4B | 50/50 | 0/50 | 1/50 |
| P06 CLIP-L to InternVL2B | 50/50 | 0/50 | 2/50 |
| P11 SigLIP2 to GemmaE2B | 50/50 | 0/50 | 0/50 |
| P14 Qwen2B to Qwen4B | 48/50 | 13/50 | 21/50 |
| P20 Qwen4B to Qwen2B | 50/50 | 29/50 | 31/50 |
| P16 InternVL2B to InternVL4B | 50/50 | 3/50 | 6/50 |
| P21 InternVL4B to InternVL2B | 50/50 | 3/50 | 10/50 |
| P19 GemmaE2B to GemmaE4B | 50/50 | 10/50 | 16/50 |
| P22 GemmaE4B to GemmaE2B | 50/50 | 2/50 | 21/50 |

`scale_50_semantic_all9.csv` is the primary aggregate table. `attack_logs/`
and `target_evaluations/` contain all 69 batches, including per-image proxy
masks, generation outputs, TASR/ASR inputs, attack parameters, CKA diagnostics,
runtime, and VRAM measurements. Generated PNGs and ImageNet inputs are omitted.

Reproduce from the repository root with:

```bash
bash scripts/run_all9_semantic_scale50.sh outputs/proxy_selector_cka_v2_scale50
python3 scripts/archive_completed_experiment_results.py
```
