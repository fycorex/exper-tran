# Proxy Prototype Transfer

## Question

Can a per-image proxy prototype contrastive objective transfer a targeted
vehicle-class attack from CLIP ViT-L/14 to an unseen CLIP ViT-B/32 target more
reliably than centered batch CKA?

## Method

The attack minimizes the existing ten-class proxy text loss plus a configurable
prototype loss. Target and source prototypes are detached normalized means of
proxy image embeddings. Target references are retained only when the proxy
classifies them as the configured target class. The target model is never
loaded during optimization.

Target evaluation is allowed only when every frozen PNG:

- has the configured target as proxy text argmax;
- increases target-prototype cosine similarity relative to its clean image;
- is closer to the target prototype than the source prototype; and
- satisfies the configured pixel and PNG constraints.

## Inputs and environment

Use the repository's pinned `.venv-primary-ml-cka`, local Hugging Face cache,
and ImageNet data configured in `configs/data/imagenet_vehicle10.yaml`. CUDA is
mandatory. Parameters are tracked in `config/scan.yaml`.

## Reproduction

```bash
source .venv-primary-ml-cka/bin/activate
export HF_HOME="$PWD/.hf-cache"
export IMAGENET_ROOT="$PWD/data/imagenet_vehicle_official"
python -m primary_ml_cka.cli.main diagnostics prototype-scan
```

Results are written under
`outputs/primary_ml_cka_v1/evaluation/prototype_transfer/`. Conclusions must
be based on the recorded hit counts and denominators; incomplete runs are not
reported as successful.

Cross-model CKA is an evaluation-only diagnostic over matched image rows:

```text
proxy representations: [N_images, D_proxy]
target representations: [N_images, D_target]
```

Run it on existing frozen PNGs with:

```bash
python -m primary_ml_cka.cli.main diagnostics cross-model-cka
```

## Findings

All positive prototype weights passed the proxy gate on 8/8 images, but target
TASR remained 0/8. For the same eight ordered images, clean proxy-target CKA
was 0.972199. Adversarial CKA ranged from 0.919963 to 0.962558 and decreased
for every scanned weight. Thus the attack fitted proxy-specific directions
instead of preserving shared proxy-target structure.

## Research basis

- Kornblith et al., *Similarity of Neural Network Representations Revisited*
  (ICML 2019), defines CKA across matched observations.
- Inkawhich et al., *Feature Space Perturbations Yield More Transferable
  Adversarial Examples* (CVPR 2019), motivates intermediate feature alignment.
- Li et al., *Towards Transferable Targeted Attack* (CVPR 2020), combines
  target attraction with movement away from the source class.
- Wei et al., *Enhancing the Self-Universality for Transferable Targeted
  Attacks* (CVPR 2023), motivates consistency across global and local views.
