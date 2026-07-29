from functools import partial

from primary_ml_cka.data.preprocessing import resize_crop_normalize

CLIP_PREPROCESS = partial(
    resize_crop_normalize,
    size=224,
    mean=(0.48145466, 0.4578275, 0.40821073),
    std=(0.26862954, 0.26130258, 0.27577711),
)
