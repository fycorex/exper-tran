import torch


def linear_cka(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("CKA inputs must be two-dimensional")
    if x.shape[0] != y.shape[0]:
        raise ValueError(
            f"CKA inputs must have the same batch size, got {x.shape[0]} and {y.shape[0]}"
        )
    if x.shape[0] < 2:
        raise ValueError("CKA requires at least two image rows")
    x = x.float()
    y = y.float()
    if not torch.isfinite(x).all() or not torch.isfinite(y).all():
        raise ValueError("CKA inputs must be finite")
    x_centered = x - x.mean(dim=0, keepdim=True)
    y_centered = y - y.mean(dim=0, keepdim=True)
    if x.shape[0] < min(x.shape[1], y.shape[1]):
        xx = x_centered @ x_centered.T
        yy = y_centered @ y_centered.T
        numerator = (xx * yy).sum()
    else:
        cross = x_centered.T @ y_centered
        xx = x_centered.T @ x_centered
        yy = y_centered.T @ y_centered
        numerator = cross.square().sum()
    denominator = torch.linalg.matrix_norm(xx) * torch.linalg.matrix_norm(yy)
    if not torch.isfinite(denominator) or denominator <= 0:
        raise ValueError("CKA denominator must be finite and positive")
    return numerator / (denominator + 1e-12)
