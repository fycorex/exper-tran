# Proxy-selector PPT figures

Presentation-ready figures generated from the corrected all-nine, eight-image
controlled experiment.

## Figures

- `01_controlled_8_sources.png`: the shared clean source cohort.
- `02_p20_targeted_transfer_example.png`: a successful Qwen 4B to Qwen 2B
  targeted-transfer example, with the perturbation amplified eight times for
  visibility.
- `03_all9_semantic_tasr_asr.png`: all-nine TASR and ASR comparison for the
  semantic auxiliary objective.
- `04_cka_vs_targeted_transfer.png`: pair-level global CKA versus TASR and
  target-class gap closure.

The quantitative figures use the semantic-only selector results under
`outputs/proxy_selector_cka_v2/diagnostics/selector_analysis_all9v2_common48_rho03/`.
They describe one controlled 8→7 class transition on eight images; the nine
model pairs are the statistical units in the CKA relationship plot.

## Reproduce

From the repository root:

```bash
.venv-primary-ml-cka/bin/python scripts/make_proxy_selector_ppt_figures.py
```

Generated files are written to this directory. Exact source-image and example
selection metadata is saved in `figure_metadata.json`.
