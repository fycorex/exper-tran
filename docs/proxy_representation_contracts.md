# Proxy Representation Contracts

- **CLIP**: final `vision_model.last_hidden_state` patch tokens, excluding CLS.
- **SigLIP2**: all final `vision_model.last_hidden_state` spatial patch tokens.
- **Qwen 3.5**: `model.get_image_features(...).pooler_output`, the post-merger
  projected visual soft tokens passed to the language model.
- **InternVL 3.5**: `model.get_image_features(...).pooler_output`, the
  post-pixel-shuffle projected visual soft tokens passed to the language model.
- **Gemma 4**: pooled visual soft tokens from `model.embed_vision`, using the
  official differentiable Torchvision processor. Per-image checkpointing keeps
  one logical eight-image batch within A4000 memory.

Every accepted tap record includes pinned revision, module path, shape, dtype,
mask, pooling, normalization, finite nonzero input gradient, and a free-generation
probe. Target taps are prohibited during attack generation; post-attack analysis
representations are recorded separately.

## Target token correspondence

Legacy token-CKA runs compared the same spatial indices from unrelated source
and target-reference images. Those results remain reproducible under
`target_cka_mode=spatial_index_legacy`, but that mode assumes a semantic
correspondence that the data do not guarantee.

An exploratory alternative is `target_cka_mode=clean_anchor_soft`. For each clean
source token, cosine-softmax matching retrieves tokens from every target-class
reference in the same frozen proxy space. The aligned references are averaged
into a detached per-source prototype. Source repulsion remains exact
`adv_i`/`clean_i` token CKA; target attraction is CKA against the clean-anchored
prototype. Reference-token permutations therefore do not change the target.
This mode is not a silent correction or replacement for the original loss;
experiments must name the selected correspondence explicitly. The controlled
rho/alpha transfer search retains `spatial_index_legacy` to test the original
token-level contrastive CKA hypothesis without changing its definition.
