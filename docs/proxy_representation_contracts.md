# Proxy Representation Contracts

- **CLIP**: final `vision_model.last_hidden_state` patch tokens, excluding CLS.
- **SigLIP2**: all final `vision_model.last_hidden_state` spatial patch tokens.
- **Qwen 3.5**: final visual patch tokens after `model.visual.merger.norm`,
  before merger projection.
- **InternVL 3.5**: final normalized InternViT patch tokens, excluding CLS.
- **Gemma 4**: pooled visual soft tokens from `model.embed_vision`, using the
  official differentiable Torchvision processor. Per-image checkpointing keeps
  one logical eight-image batch within A4000 memory.

Every accepted tap record includes pinned revision, module path, shape, dtype,
mask, pooling, normalization, finite nonzero input gradient, and a free-generation
probe. Target taps are prohibited during attack generation; post-attack analysis
representations are recorded separately.
