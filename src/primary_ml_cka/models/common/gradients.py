import torch


def assert_frozen(module: torch.nn.Module) -> None:
    trainable = [name for name, parameter in module.named_parameters() if parameter.requires_grad]
    if trainable:
        raise RuntimeError(f"Expected frozen module; trainable parameters: {trainable[:5]}")


def assert_input_gradient(loss: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
    gradient = torch.autograd.grad(loss, images, only_inputs=True)[0]
    if not torch.isfinite(gradient).all() or gradient.abs().sum() == 0:
        raise RuntimeError("Input-pixel gradient is non-finite or zero")
    return gradient


def assert_parameter_gradients_none(module: torch.nn.Module) -> None:
    populated = [
        name for name, parameter in module.named_parameters() if parameter.grad is not None
    ]
    if populated:
        raise RuntimeError(f"Frozen parameters acquired gradients: {populated[:5]}")
