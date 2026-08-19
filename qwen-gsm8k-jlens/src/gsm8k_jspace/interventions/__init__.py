from gsm8k_jspace.interventions.mean_replace import MeanReplaceController
from gsm8k_jspace.interventions.no_op import NoOpController


class BaselineController:
    def reset_example(self, example_id: str, prompt_length: int) -> None:
        return None

    def before_generation(self) -> None:
        return None

    def after_generation(self) -> None:
        return None

    def __enter__(self) -> "BaselineController":
        return self

    def __exit__(self, *exc) -> None:
        return None

    def summary(self) -> dict:
        return {"method": "baseline", "layers": []}


def build_controller(
    condition: str,
    *,
    jlens=None,
    lens_model=None,
    layers: list[int] | None = None,
    spec=None,
    compute_device: str = "cpu",
    log_path=None,
):
    if condition == "baseline":
        return BaselineController()
    if condition == "no_op":
        return NoOpController(lens_model, layers or [])
    if condition == "intervention":
        return MeanReplaceController(
            jlens,
            lens_model,
            layers or [],
            spec,
            compute_device=compute_device,
            log_path=log_path,
        )
    raise ValueError(f"unknown condition {condition!r}")
