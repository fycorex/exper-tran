# Proxy Representation Contracts

- **CLIP**: checkpoint-native projected image feature from
  `get_image_features`, with native preprocessing and L2 normalization.
- **SigLIP2**: checkpoint-native projected image feature from
  `get_image_features`, with native preprocessing and L2 normalization.
- **Qwen 3.5**: final proxy visual patch tokens after
  `model.visual.merger.norm`, before merger projection.
- **InternVL 3.5**: final normalized proxy InternViT patch tokens from
  `model.vision_tower.layernorm`, excluding CLS.
- **Gemma 4 proxy**: blocked until proxy-native differentiable preprocessing
  and a finite image-gradient tap are validated.

Every accepted tap record includes the pinned revision, module path, tensor
shape, dtype, mask, pooling rule, normalization, and finite nonzero
input-gradient result. A target model tap is prohibited.
