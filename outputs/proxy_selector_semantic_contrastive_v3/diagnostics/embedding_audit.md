# Embedding audit

| Model | Component | Layer | Module path | Token shape | Pooling | Embedding shape |
|---|---|---:|---|---|---|---|
| Qwen/Qwen3.5-2B | vision_encoder | 23/24 | `model.model.visual.blocks.23` | `[1, 256, 1024]` | masked mean over valid visual tokens | `[1, 1024]` |
| Qwen/Qwen3.5-4B | vision_encoder | 23/24 | `model.model.visual.blocks.23` | `[1, 256, 1024]` | masked mean over valid visual tokens | `[1, 1024]` |
| OpenGVLab/InternVL3_5-2B-HF | vision_encoder | 23/24 | `model.model.vision_tower.encoder.layer.23` | `[1, 1024, 1024]` | masked mean over valid visual tokens | `[1, 1024]` |
| OpenGVLab/InternVL3_5-4B-HF | vision_encoder | 23/24 | `model.model.vision_tower.encoder.layer.23` | `[1, 1024, 1024]` | masked mean over valid visual tokens | `[1, 1024]` |
| google/gemma-4-E2B-it | vision_encoder | 15/16 | `model.model.vision_tower.encoder.layers.15` | `[1, 2520, 768]` | masked mean over valid visual tokens | `[1, 768]` |
| google/gemma-4-E4B-it | vision_encoder | 15/16 | `model.model.vision_tower.encoder.layers.15` | `[1, 2520, 768]` | masked mean over valid visual tokens | `[1, 768]` |
| openai/clip-vit-large-patch14 | vision_encoder | 23/24 | `model.vision_model.encoder.layers.23` | `[1, 256, 1024]` | masked mean over real visual patch tokens | `[1, 1024]` |
| google/siglip2-so400m-patch14-384 | vision_encoder | 26/27 | `model.vision_model.encoder.layers.26` | `[1, 729, 1152]` | masked mean over real visual patch tokens | `[1, 1152]` |
