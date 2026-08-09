# Experiment Specification

## Proxy Objective

The attack canvas is RGB `224×224` in `[0,1]`. Source and target labels come
from `configs/data/imagenet_vehicle10.yaml`. For target index \(t\), ten proxy
class logits \(\ell\), margin \(m=2\), weight \(\alpha=1\), and temperature
\(\tau=0.5\):

\[
L_{cls}=CE(\ell,t)+\alpha\,softplus\!\left(
\frac{m-(\ell_t-\max_{j\ne t}\ell_j)}{\tau}\right).
\]

The former mean-other-probability term was removed because it equals
\((1-p_t)/9\) and duplicates the cross-entropy signal. Generative proxy logits
are mean answer-token log probabilities, so labels with different token lengths
are comparable. Their ten-way softmax is a closed-set normalized proxy
probability, not a free-generation probability.

## Data and Canonical Input

The configured allocation uses 48 target-class training references, 50 source
validation candidates, up to 32 main images, and the next up to 16 disjoint
confirmation images. A balanced calibration bank contains five validation
images from each of the ten classes.

`data prepare` materializes one deterministic bicubic-stretched RGB 224×224 PNG
per candidate, reference, and calibration image. Clean screening, proxy attack,
target clean evaluation, and target adversarial evaluation all start from these
same canonical PNGs. Model-native resizing and normalization happen only after
this shared input boundary.

## Per-image Proxy Attack CKA

Attack CKA is proxy-only and centers over real visual token positions, never
over batch rows. For adversarial tokens \(H_i^{adv}\), canonical clean tokens
\(H_i^{clean}\), and \(K\) target-reference token sets:

\[
L_{CKA}=\frac1B\sum_i\left[CKA(H_i^{adv},H_i^{clean})-
\frac1K\sum_j CKA(H_i^{adv},H_j^{target})\right].
\]

The minimized total objective is \(L_{cls}+\lambda L_{CKA}\). CLIP and SigLIP2
expose final-layer spatial patch tokens; Qwen, InternVL, and Gemma expose their
documented visual token taps. Global pooled embeddings are not used in attack
CKA.

## Strong Proxy Gate and Target Evaluation

The target receives only frozen PNGs and decoded-text prompts. Every image in a
logical batch must satisfy all four proxy conditions: target top-1, logit margin
at least 2, closed-set target probability at least 0.9, and strict proxy free
generation equal to the target label. A batch that is not 8/8 is recorded as
`proxy_target_not_reached` and is never sent to transfer evaluation.

Reports retain proxy hit counts, eligible batches/images, and clean-conditioned
TASR so proxy attack failure cannot be mistaken for transfer failure.

## Post-attack Model Similarity

`analysis model-similarity` runs only after adversarial PNGs and target outputs
are frozen. It loads one model at a time, extracts representations for identical
balanced calibration images, and computes global CKA for each pair. For every
attacked clean image, its eight nearest calibration neighbors in proxy space
define local CKA. Outputs include pair/lambda TASR, per-image target hits, and
descriptive global/local correlations. Target representations live in a
separate post-attack module and cannot enter the attack graph.
