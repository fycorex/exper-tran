from dataclasses import dataclass

ALL_RESULTS_COLUMNS = (
    "pair_id",
    "exp_type",
    "proxy_model",
    "target_model",
    "source_human_label",
    "target_human_label",
    "proxy_revision",
    "target_revision",
    "proxy_tap_status",
    "proxy_tap_path",
    "phase",
    "batch_id",
    "source_image_ids",
    "target_reference_ids",
    "lambda",
    "seed",
    "steps",
    "clean_valid_count",
    "targeted_hit_count",
    "tasr_percent",
    "untargeted_hit_count",
    "asr_percent",
    "proxy_target_nll",
    "proxy_target_probability",
    "proxy_target_hit_count",
    "proxy_target_hit_denominator",
    "proxy_target_all_hit",
    "proxy_min_target_logit_margin",
    "proxy_max_other_probability",
    "proxy_target_probability_margin",
    "proxy_classification_ce",
    "proxy_rank_loss",
    "proxy_other_suppression_loss",
    "loss_cka",
    "loss_total",
    "cka_clean_reference",
    "cka_adv_source",
    "cka_adv_reference",
    "reference_cka_gain",
    "source_cka_drop",
    "proxy_representation_shift",
    "grad_ml_l1",
    "grad_cka_weighted_l1",
    "grad_component_cosine",
    "linf_float",
    "linf_png",
    "elapsed_seconds",
    "peak_allocated_vram_gb",
    "peak_reserved_vram_gb",
    "status",
    "failure_reason",
)


@dataclass(frozen=True, slots=True)
class ResultRow:
    values: tuple[object, ...]

    def __post_init__(self) -> None:
        if len(self.values) != len(ALL_RESULTS_COLUMNS):
            raise ValueError("Result row does not match required schema")
