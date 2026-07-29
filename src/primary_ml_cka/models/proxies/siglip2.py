from functools import partial

from primary_ml_cka.data.preprocessing import resize_crop_normalize

SIGLIP2_PREPROCESS = partial(
    resize_crop_normalize,
    size=384,
    mean=(0.5, 0.5, 0.5),
    std=(0.5, 0.5, 0.5),
)
