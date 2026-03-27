import torch
import torch.nn.functional as F
from typing import Callable, Optional
from nnunetv2.utilities.connected_components import get_voronoi, get_cc

class _BaseConnectedComponentLoss(torch.nn.Module):
    """Shared forward logic for losses operating on connected components."""

    def __init__(self, metric, activation) -> None:
        super().__init__()

        self.metric = metric
        self.activation = activation

    def forward(self, y_pred, y):
        self._validate_inputs(y_pred, y)

        y_one_hot = self._one_hot_encode(y_pred, y)
        components = self._compute_components(y_one_hot)

        self._validate_components(y_pred, y_one_hot, components)

        return binary_cc(
            y_pred=y_pred,
            y=y_one_hot,
            components=components,
            metric=self.metric,
            masking_fn=self._masking_fn,
            activation=self.activation,
        )

    # Helpers
    def _validate_inputs(self, y_pred: torch.Tensor, y: torch.Tensor) -> None:
        cls_name = self.__class__.__name__

        assert (
            y_pred.ndim == 5 and y_pred.shape[1] == 2
        ), f"Expected y_pred with shape [B,2,H,W,D], but got {tuple(y_pred.shape)}"
        assert list(y.shape) == [
            y_pred.shape[0],
            1,
            *y_pred.shape[2:],
        ], f"Expected y with shape ({tuple(y_pred.shape)}) [B,1,H,W,D], but got {tuple(y.shape)}"
        assert y.dtype == torch.int16, f"Expected y.dtype=torch.int16, but got {y.dtype}"

        assert (
            y_pred.is_cuda
        ), f"{cls_name} expects CUDA tensors for y_pred, but got device {y_pred.device}."
        assert y.is_cuda, f"{cls_name} expects CUDA tensors for y, but got device {y.device}."
        assert (
            y.device == y_pred.device
        ), f"y and y_pred must reside on the same CUDA device, but got y on {y.device} and y_pred on {y_pred.device}."

    def _one_hot_encode(self, y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        y_idx = y[:, 0].long()  # [B,*]
        return (
            F.one_hot(y_idx, num_classes=y_pred.shape[1])
            .movedim(-1, 1)
            .float()
        )  # [B,2,*]

    def _validate_components(
        self, y_pred: torch.Tensor, y: torch.Tensor, components: torch.Tensor
    ) -> None:
        assert y_pred.shape == y.shape, "y_pred and one-hot y must match"
        assert components.dtype == torch.int64
        expected_shape = [y_pred.shape[0], *y_pred.shape[2:]]
        assert (
            list(components.shape) == expected_shape
        ), f"Expected connected components with shape [B,H,W,D], but got {tuple(components.shape)}"

    # Abstract hooks
    def _compute_components(self, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _masking_fn(
        self, component_id: int, components: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError


class CCMetrics(_BaseConnectedComponentLoss):
    def __init__(self, metric, activation) -> None:
        super().__init__(metric=metric, activation=activation)

    def _compute_components(self, y: torch.Tensor) -> torch.Tensor:
        voronoi = get_voronoi(y, do_bg=False)
        return voronoi[:, 0, ...]

    def _masking_fn(
        self, component_id: int, components: torch.Tensor
    ) -> torch.Tensor:
        return components == component_id


class BlobLoss(_BaseConnectedComponentLoss):
    def __init__(self, metric, activation) -> None:
        super().__init__(metric=metric, activation=activation)

    def _compute_components(self, y: torch.Tensor) -> torch.Tensor:
        connected_components = get_cc(y, do_bg=False)
        return connected_components[:, 0, ...]

    def _masking_fn(
        self, component_id: int, components: torch.Tensor
    ) -> torch.Tensor:
        return (components == component_id) | (components == 0)


def per_channel_cc(
    y_pred: torch.Tensor,  # [2, *spatial]
    y: torch.Tensor,  # [2, *spatial]
    components: torch.Tensor,  # [,2 *spatial], integer component IDs
    metric: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    masking_fn: Callable[[int, torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """
    Compute metric on each connected component in one channel,
    but wrap each component's computation in a checkpoint to reduce memory.
    """
    # define pure function for checkpointing
    def _component_score(pred: torch.Tensor, true: torch.Tensor, mask: torch.Tensor):
        assert pred.shape == true.shape
        assert true.ndim == 4
        assert true.shape[0] == 2

        assert mask.ndim == 3
        assert list(mask.shape) == list(true.shape)[1:]
        assert mask.dtype == torch.bool
        
        masked_pred = pred * mask
        masked_true = true * mask

        return metric(masked_pred, masked_true, mask)

    max_id = components.max()
    if max_id == 0:
        # fallback: treat whole channel as one component
        mask = torch.ones_like(y[0], dtype=torch.bool)
        return _component_score(y_pred, y, mask)

    ids = torch.unique(components)
    ids = ids[ids > 0]

    scores: list[torch.Tensor] = []
    for comp_id in ids:
        mask = masking_fn(comp_id, components)
        # checkpoint the component-level forward
        score: torch.Tensor = torch.utils.checkpoint.checkpoint(
            _component_score, y_pred, y, mask, use_reentrant=False
        )
        assert score.ndim == 0, "metric_fn should return scalar functions"
        scores.append(score)

    return torch.stack(scores, dim=0).mean()


def binary_cc(
    y_pred: torch.Tensor,  # [B, C, *spatial]
    y: torch.Tensor,  # [B, C, *spatial]
    components: torch.Tensor,  # [B, *spatial], integer CC labels
    metric: Callable,  # fn(masked_pred, masked_true, mask) -> scalar tensor
    masking_fn: Callable[[int, torch.Tensor], torch.Tensor],
    activation: Optional[Callable],
) -> torch.Tensor:
    """Reduce a metric over connected components and channels to obtain a batch score.

    Args:
        y_pred: Logits or probabilities shaped ``[B, 2, H, W, D]``.
        y: One-hot ground truth tensor with the same shape as ``y_pred``.
        components: Integer labels describing connected components per channel, shape [B, H, W, D].
        metric: Callable applied to each component; should accept ``(masked_pred, masked_true, mask)`` tensors.
        activation: Optional callable applied to ``y_pred`` before metric evaluation.
    """
    # Sanity checks
    assert y_pred.shape == y.shape, "All inputs must match in shape"
    assert y.ndim == 5, "Expect [B, 2, H, W, D]"
    assert y.shape[1] == 2, "This is for binary inputs only"

    assert y.dtype in (torch.float16, torch.bfloat16, torch.float32)
    assert y_pred.dtype in (torch.float16, torch.bfloat16, torch.float32)

    # Optional activation
    if activation is not None:
        y_pred = activation(y_pred)

    B = y_pred.shape[0]
    sample_scores: list[torch.Tensor] = []

    for i in range(B):
        score = per_channel_cc(y_pred[i], y[i], components[i], metric, masking_fn)
        sample_scores.append(score)

    scores = torch.stack(sample_scores, dim=0)

    return scores.mean()
