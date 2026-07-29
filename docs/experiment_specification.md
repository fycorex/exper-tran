# Experiment Specification

## Proxy Objective

The attack canvas is RGB `224×224` in `[0,1]`. Source and target human labels
are required settings in `configs/data/imagenet_vehicle10.yaml`; the target
class index is always derived with `human_label_to_index`. For target index
\(t\), ten proxy class logits \(\ell\), and probabilities \(p\):

\[
L_{cls}=CE(\ell,t)
+\alpha\,softplus(\max_{j\ne t}\ell_j-\ell_t+m)
+\beta\,\frac{1}{9}\sum_{j\ne t}p_j.
\]

Defaults are margin \(m=2\), \(\alpha=1\), and \(\beta=1\). This pushes the
target class above every alternative while explicitly suppressing the mean
probability of the other nine classes.

Candidate discovery, clean screening, reference selection, proxy loss, and
TASR/ASR evaluation all consume the same data configuration. Changing a target
there does not require a source-code change.

## Data Allocation

All counts are required in `configs/data/imagenet_vehicle10.yaml`; the Python
schema provides no duplicate defaults. The configured allocation is:

- 48 target-class ImageNet training images as proxy CKA references;
- 50 source-class ImageNet validation candidates for clean screening;
- up to 32 clean-valid candidates for the main attack and evaluation;
- the next up to 16 clean-valid candidates for disjoint confirmation.

Main and confirmation counts are truncated to complete attack batches. This
experiment does not train model parameters and does not use the unlabeled
ImageNet test split.

Reference allocation is also disjoint: main batches consume references 0–31,
while confirmation batches consume references 32–47.

## Proxy CKA

CKA is computed across batch rows using proxy image representations:

\[
L_{CKA}=CKA(Z_{adv},Z_{clean})-CKA(Z_{adv},Z_{reference}).
\]

The minimized objective is:

\[
L_{total}=L_{cls}+\lambda L_{CKA}.
\]

The CKA kernel accepts any equal batch size of at least two. The checked-in
experiment configuration uses batch size 8, but it is a runtime setting rather
than a mathematical hardcode.

## Target Evaluation

The target receives the fixed prompt and returns decoded text. The evaluator
keeps raw output, parses only an exact integer on the first non-empty line, and
reports clean-conditioned TASR/ASR with hit counts and denominators. Target
outputs never influence lambda selection.

Target evaluation is gated on the frozen adversarial PNGs. Every image must
have the configured target class strictly above each of the other nine proxy
classes. Rows that do not reach `batch_size / batch_size` are recorded as
`proxy_target_not_reached` and are never sent to the target.
