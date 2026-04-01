import os
import torch
import torch.nn.functional as F
from typing import Callable, Literal, Optional
from torch.func import vmap
from nnunetv2.utilities.connected_components import get_voronoi, get_cc

VectorizedLossMode = Literal["dice", "ce", "dice_ce"]


def _should_compile_vectorized_reduction() -> bool:
    if not torch.cuda.is_available():
        return False
    if os.name == "nt":
        return False
    if "nnUNet_compile" not in os.environ:
        return True
    return os.environ["nnUNet_compile"].lower() in ("true", "1", "t")

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

    # Helpers -----------------------------------------------------------------
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

    # Abstract hooks ----------------------------------------------------------
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


class _BaseVectorizedLoss(_BaseConnectedComponentLoss):
    """Specialized reduction path that avoids per-component masking loops."""

    def __init__(self, activation, mode: VectorizedLossMode) -> None:
        super().__init__(metric=None, activation=activation)
        if mode not in ("dice", "ce", "dice_ce"):
            raise ValueError(f"Unsupported vectorized loss mode: {mode}")
        self.mode = mode

    def forward(self, y_pred, y):
        self._validate_inputs(y_pred, y)

        y_one_hot = self._one_hot_encode(y_pred, y)
        components = self._compute_components(y_one_hot)

        self._validate_components(y_pred, y_one_hot, components)

        return self._vectorized_loss(y_pred, y, components)

    def _vectorized_loss(
        self,
        y_pred: torch.Tensor,
        y: torch.Tensor,
        components: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError


class VectorizedCCLoss(_BaseVectorizedLoss):
    """Vectorized connected-component loss for binary instance segmentation.

    This is the fast path for Voronoi-based CC supervision. Each voxel belongs
    to exactly one connected component, so we can flatten the volume and use a
    segmented reduction instead of looping over component ids and building a
    full-volume mask for every component.

    The implementation supports the following static modes:
    - ``"dice"``: foreground Dice loss only
    - ``"ce"``: mean cross-entropy over each component
    - ``"dice_ce"``: sum of the two terms above

    Notes
    -----
    ``components`` is expected to be a dense Voronoi map with ids in
    ``{0, 1, ..., K}``, where ``0`` denotes the empty-component fallback case
    and ``1..K`` are the foreground instances. Reduction is vmapped over the
    batch and uses fixed ``K + 1`` bins per sample, with bin ``0`` reserved for
    background / empty handling.
    """
    def __init__(self, activation, mode: VectorizedLossMode = "dice_ce") -> None:
        super().__init__(activation=activation, mode=mode)

    def _compute_components(self, y: torch.Tensor) -> torch.Tensor:
        voronoi = get_voronoi(y, do_bg=False)
        return voronoi[:, 0, ...]

    def _masking_fn(
        self, component_id: int, components: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError("VectorizedCCLoss does not use masking_fn")

    def _vectorized_loss(
        self,
        y_pred: torch.Tensor,
        y: torch.Tensor,
        components: torch.Tensor,
    ) -> torch.Tensor:
        return _vectorized_cc_loss(y_pred, y, components, self.activation, self.mode)


class VectorizedBlobLoss(_BaseVectorizedLoss):
    """Vectorized blob loss for binary instance segmentation.

    This is the fast path for connected-component blob supervision. Unlike the
    CC/Voronoi case, the loss region for component ``k`` is not just that
    component, but ``background union component_k``. Because background voxels
    participate in every component loss, the reduction is split into:
    - one shared background bin
    - one bin per foreground component

    Final per-component Dice / CE terms are then assembled algebraically as:
    ``background contribution + component contribution``.

    The implementation supports the following static modes:
    - ``"dice"``: foreground Dice loss only
    - ``"ce"``: mean cross-entropy over each blob region
    - ``"dice_ce"``: sum of the two terms above

    This keeps the same semantics as the original BlobLoss implementation while
    avoiding per-component boolean mask materialization.
    """
    def __init__(self, activation, mode: VectorizedLossMode = "dice_ce") -> None:
        super().__init__(activation=activation, mode=mode)

    def _compute_components(self, y: torch.Tensor) -> torch.Tensor:
        connected_components = get_cc(y, do_bg=False)
        return connected_components[:, 0, ...]

    def _masking_fn(
        self, component_id: int, components: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError("VectorizedBlobLoss does not use masking_fn")

    def _vectorized_loss(
        self,
        y_pred: torch.Tensor,
        y: torch.Tensor,
        components: torch.Tensor,
    ) -> torch.Tensor:
        return _vectorized_blob_loss(y_pred, y, components, self.activation, self.mode)


def _scatter_segment_sum(
    values: torch.Tensor,
    segment_ids: torch.Tensor,
    num_segments: int,
) -> torch.Tensor:
    out = torch.zeros(num_segments, device=values.device, dtype=values.dtype)
    return torch.scatter_add(out, 0, segment_ids, values)


def _combine_vectorized_loss_terms(
    *,
    mode: VectorizedLossMode,
    intersection: Optional[torch.Tensor] = None,
    pred_sum: Optional[torch.Tensor] = None,
    true_sum: Optional[torch.Tensor] = None,
    ce_sum: Optional[torch.Tensor] = None,
    count: Optional[torch.Tensor] = None,
    eps: float = 1e-7,
) -> torch.Tensor:
    if mode == "dice":
        assert intersection is not None and pred_sum is not None and true_sum is not None
        return 1.0 - (2.0 * intersection / (pred_sum + true_sum).clamp_min(eps))
    if mode == "ce":
        assert ce_sum is not None and count is not None
        return ce_sum / count.clamp_min(1.0)

    assert intersection is not None and pred_sum is not None and true_sum is not None
    assert ce_sum is not None and count is not None
    dice_loss = 1.0 - (2.0 * intersection / (pred_sum + true_sum).clamp_min(eps))
    ce_loss = ce_sum / count.clamp_min(1.0)
    return dice_loss + ce_loss


def _whole_volume_fallback_loss(
    *,
    mode: VectorizedLossMode,
    foreground_pred: torch.Tensor,
    foreground_true: torch.Tensor,
    ce_map: Optional[torch.Tensor] = None,
    eps: float = 1e-7,
) -> torch.Tensor:
    if mode == "dice":
        return foreground_pred.new_tensor(1.0)
    if mode == "ce":
        assert ce_map is not None
        return ce_map.mean()

    assert ce_map is not None
    return ce_map.mean() + 1.0


# Keep the wrapper eager: it performs CuPy work and extracts a Python scalar for
# num_component_slots. Meaning it inserts graph breaks and thus compiling it is
# not worth it.
@torch.compiler.disable
def _vectorized_cc_loss(
    y_pred: torch.Tensor,
    y: torch.Tensor,
    components: torch.Tensor,
    activation: Optional[Callable],
    mode: VectorizedLossMode,
) -> torch.Tensor:
    eps = 1e-7
    probs = activation(y_pred) if activation is not None else y_pred
    probs_f32 = probs.to(torch.float32)
    target = y[:, 0].long()
    num_component_slots = max(1, int(components.max().item()))
    foreground_pred = probs_f32[:, 1]
    foreground_true = (target == 1).to(torch.float32)
    ce_map = -torch.log(probs_f32.clamp_min(eps)).gather(1, target.unsqueeze(1)).squeeze(1)
    return _compiled_vectorized_cc_reduction(
        foreground_pred,
        foreground_true,
        ce_map,
        components,
        mode,
        num_component_slots,
    )


def _vectorized_cc_reduction(
    foreground_pred: torch.Tensor,
    foreground_true: torch.Tensor,
    ce_map: torch.Tensor,
    components: torch.Tensor,
    mode: VectorizedLossMode,
    num_component_slots: int,
) -> torch.Tensor:
    eps = 1e-7

    def per_sample_loss(
        sample_foreground_pred: torch.Tensor,
        sample_foreground_true: torch.Tensor,
        sample_ce_map: torch.Tensor,
        sample_components: torch.Tensor,
    ) -> torch.Tensor:
        component_ids = sample_components.long().reshape(-1)
        num_slots_with_bg = num_component_slots + 1

        def segment_sum(values: torch.Tensor) -> torch.Tensor:
            return _scatter_segment_sum(
                values.reshape(-1),
                component_ids,
                num_slots_with_bg,
            )

        intersection = segment_sum(sample_foreground_pred * sample_foreground_true)[1:]
        pred_sum = segment_sum(sample_foreground_pred)[1:]
        true_sum = segment_sum(sample_foreground_true)[1:]
        ce_sum = segment_sum(sample_ce_map)[1:]
        count = segment_sum(torch.ones_like(sample_foreground_pred))[1:]
        component_loss = _combine_vectorized_loss_terms(
            mode=mode,
            intersection=intersection,
            pred_sum=pred_sum,
            true_sum=true_sum,
            ce_sum=ce_sum,
            count=count,
            eps=eps,
        )

        valid_components = count > 0
        valid_components_f = valid_components.to(component_loss.dtype)
        cc_loss = (
            (component_loss * valid_components_f).sum()
            / valid_components_f.sum().clamp_min(1.0)
        )
        fallback = _whole_volume_fallback_loss(
            mode=mode,
            foreground_pred=sample_foreground_pred,
            foreground_true=sample_foreground_true,
            ce_map=sample_ce_map,
            eps=eps,
        )
        return torch.where(valid_components.any(), cc_loss, fallback)

    per_sample = vmap(per_sample_loss, in_dims=(0, 0, 0, 0))(
        foreground_pred, foreground_true, ce_map, components
    )
    return per_sample.mean()


# Keep the wrapper eager: it performs CuPy work and extracts a Python scalar for
# num_component_slots. Meaning it inserts graph breaks and thus compiling it is
# not worth it.
@torch.compiler.disable
def _vectorized_blob_loss(
    y_pred: torch.Tensor,
    y: torch.Tensor,
    components: torch.Tensor,
    activation: Optional[Callable],
    mode: VectorizedLossMode,
) -> torch.Tensor:
    eps = 1e-7
    probs = activation(y_pred) if activation is not None else y_pred
    probs_f32 = probs.to(torch.float32)
    target = y[:, 0].long()
    num_component_slots = max(1, int(components.max().item()))
    foreground_pred = probs_f32[:, 1]
    foreground_true = (target == 1).to(torch.float32)
    ce_map = -torch.log(probs_f32.clamp_min(eps)).gather(1, target.unsqueeze(1)).squeeze(1)
    return _compiled_vectorized_blob_reduction(
        foreground_pred,
        foreground_true,
        ce_map,
        components,
        mode,
        num_component_slots,
    )


def _vectorized_blob_reduction(
    foreground_pred: torch.Tensor,
    foreground_true: torch.Tensor,
    ce_map: torch.Tensor,
    components: torch.Tensor,
    mode: VectorizedLossMode,
    num_component_slots: int,
) -> torch.Tensor:
    eps = 1e-7

    def per_sample_loss(
        sample_foreground_pred: torch.Tensor,
        sample_foreground_true: torch.Tensor,
        sample_ce_map: torch.Tensor,
        sample_components: torch.Tensor,
    ) -> torch.Tensor:
        component_ids = sample_components.long().reshape(-1)
        num_slots_with_bg = num_component_slots + 1

        def segment_sum(values: torch.Tensor) -> torch.Tensor:
            return _scatter_segment_sum(
                values.reshape(-1),
                component_ids,
                num_slots_with_bg,
            )

        intersection_bins = segment_sum(sample_foreground_pred * sample_foreground_true)
        pred_bins = segment_sum(sample_foreground_pred)
        true_bins = segment_sum(sample_foreground_true)
        ce_bins = segment_sum(sample_ce_map)
        count_bins = segment_sum(torch.ones_like(sample_foreground_pred))

        intersection = intersection_bins[0] + intersection_bins[1:]
        pred_sum = pred_bins[0] + pred_bins[1:]
        true_sum = true_bins[0] + true_bins[1:]
        ce_sum = ce_bins[0] + ce_bins[1:]
        count = count_bins[0] + count_bins[1:]
        component_loss = _combine_vectorized_loss_terms(
            mode=mode,
            intersection=intersection,
            pred_sum=pred_sum,
            true_sum=true_sum,
            ce_sum=ce_sum,
            count=count,
            eps=eps,
        )

        valid_components = count_bins[1:] > 0
        valid_components_f = valid_components.to(component_loss.dtype)
        blob_loss = (
            (component_loss * valid_components_f).sum()
            / valid_components_f.sum().clamp_min(1.0)
        )
        fallback = _whole_volume_fallback_loss(
            mode=mode,
            foreground_pred=sample_foreground_pred,
            foreground_true=sample_foreground_true,
            ce_map=sample_ce_map,
            eps=eps,
        )
        return torch.where(valid_components.any(), blob_loss, fallback)

    per_sample = vmap(per_sample_loss, in_dims=(0, 0, 0, 0))(
        foreground_pred, foreground_true, ce_map, components
    )
    return per_sample.mean()


if _should_compile_vectorized_reduction():
    _compiled_vectorized_cc_reduction = torch.compile(_vectorized_cc_reduction)
    _compiled_vectorized_blob_reduction = torch.compile(_vectorized_blob_reduction)
else:
    _compiled_vectorized_cc_reduction = _vectorized_cc_reduction
    _compiled_vectorized_blob_reduction = _vectorized_blob_reduction


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
    # -- sanity checks --
    assert y_pred.shape == y.shape, "All inputs must match in shape"
    assert y.ndim == 5, "Expect [B, 2, H, W, D]"
    assert y.shape[1] == 2, "This is for binary inputs only"

    assert y.dtype in (torch.float16, torch.bfloat16, torch.float32)
    assert y_pred.dtype in (torch.float16, torch.bfloat16, torch.float32)

    # -- optional activation --
    if activation is not None:
        y_pred = activation(y_pred)

    B = y_pred.shape[0]
    sample_scores: list[torch.Tensor] = []

    # Instead of vmap, loop over the batch
    for i in range(B):
        score = per_channel_cc(y_pred[i], y[i], components[i], metric, masking_fn)
        sample_scores.append(score)

    # Stack the results into a tensor (if that's what vmap was producing)
    scores = torch.stack(sample_scores, dim=0)

    return scores.mean()
