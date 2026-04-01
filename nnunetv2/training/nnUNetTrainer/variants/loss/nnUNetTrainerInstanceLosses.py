from __future__ import annotations

from typing import Mapping, Sequence, Tuple

import numpy as np
import torch
from torch import nn

from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.instance_losses import (
    VectorizedBlobLoss,
    VectorizedCCLoss,
)
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.git_logging import log_git_context
from nnunetv2.utilities.helpers import softmax_helper_dim1


class LossMixer(nn.Module):
    """Combine multiple loss modules into a weighted sum.

    Notes
    -----
    * Components are stored in a :class:`torch.nn.ModuleDict` so that they are moved
      correctly when ``LossMixer.to(device)`` is called.
    * The ``forward`` call accepts the default nnU-Net loss signature ``(y_pred, y)``
      and forwards it to every registered loss module.
    """

    def __init__(self, components: Sequence[Tuple[str, nn.Module, float]]):
        super().__init__()
        if not components:
            raise ValueError("LossMixer requires at least one component")

        names = [name for name, _, _ in components]
        if len(set(names)) != len(names):
            raise ValueError("LossMixer component names must be unique")

        self.components = nn.ModuleDict({name: module for name, module, _ in components})
        self.weights = {name: float(weight) for name, _, weight in components}

    def forward(self, y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        total = torch.zeros((), device=y_pred.device, dtype=y_pred.dtype)
        for name, module in self.components.items():
            weight = self.weights[name]
            if weight == 0:
                continue
            value = module(y_pred, y)
            total = total + weight * value
        return total


def integrate_deep_supervision(trainer: nnUNetTrainer, loss: nn.Module) -> nn.Module:
    """Wrap a loss with nnU-Net's deep supervision helper if required."""

    if not trainer.enable_deep_supervision:
        return loss

    deep_supervision_scales = trainer._get_deep_supervision_scales()
    weights = np.array([1 / (2**i) for i in range(len(deep_supervision_scales))], dtype=np.float32)
    if trainer.is_ddp and not trainer._do_i_compile():
        weights[-1] = 1e-6
    else:
        weights[-1] = 0

    weights /= weights.sum()
    return DeepSupervisionWrapper(loss, weights)


def _build_global_dc_ce_loss(trainer: nnUNetTrainer) -> nn.Module:
    loss = DC_and_CE_loss(
        {
            'batch_dice': trainer.configuration_manager.batch_dice,
            'smooth': 0,
            'do_bg': False,
            'ddp': trainer.is_ddp,
        },
        {},
        weight_ce=1,
        weight_dice=1,
        ignore_label=trainer.label_manager.ignore_label,
        dice_class=MemoryEfficientSoftDiceLoss,
    )

    if hasattr(loss, "dc") and trainer._do_i_compile():
        loss.dc = torch.compile(loss.dc)  # type: ignore[attr-defined]
    return loss


def _assert_instance_loss_prerequisites(trainer: nnUNetTrainer):
    if trainer.label_manager.has_regions:
        raise AssertionError("Instance-level losses require label-based training without regions")
    if trainer.label_manager.has_ignore_label:
        raise AssertionError("Instance-level losses do not support ignore labels")
    if trainer.label_manager.num_segmentation_heads != 2:
        raise AssertionError("Blob/CC instance losses require a binary segmentation (background + foreground)")


class _InstanceTrainerBase(nnUNetTrainer):
    include_global_component: bool
    use_instance_loss: bool
    global_component_weight: float = 1.0
    instance_component_weight: float = 1.0

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        
        assert hasattr(
            type(self), "include_global_component"
        ), f"{type(self).__name__} must define class attribute 'include_global_component'"
        assert hasattr(
            type(self), "use_instance_loss"
        ), f"{type(self).__name__} must define class attribute 'use_instance_loss'"
        assert isinstance(
            self.include_global_component, bool
        ), (
            f"'include_global_component' on {type(self).__name__} must be a bool, "
            f"got {type(self.include_global_component).__name__}"
        )
        assert isinstance(
            self.use_instance_loss, bool
        ), (
            f"'use_instance_loss' on {type(self).__name__} must be a bool, "
            f"got {type(self.use_instance_loss).__name__}"
        )
        
        log_git_context(self)

    def _build_instance_loss(self) -> nn.Module:
        raise NotImplementedError
    
    def _build_global_loss(self) -> nn.Module:
        return _build_global_dc_ce_loss(self)

    def _build_loss(self) -> nn.Module:  # type: ignore[override]
        _assert_instance_loss_prerequisites(self)

        components: list[tuple[str, nn.Module, float]] = []

        if self.use_instance_loss:
            instance_loss = self._build_instance_loss().to(self.device)
            components.append(("instance", instance_loss, self.instance_component_weight))

        if self.include_global_component:
            global_loss = self._build_global_loss().to(self.device)
            components.append(("global", global_loss, self.global_component_weight))

        assert components, "At least one loss component must be active"

        if len(components) == 1:
            final_loss = components[0][1]
        else:
            final_loss = LossMixer(components)

        final_loss = integrate_deep_supervision(self, final_loss)
        final_loss = final_loss.to(self.device)
        return final_loss


class nnUNetTrainerCCDiceCE(_InstanceTrainerBase):
    include_global_component = False
    use_instance_loss = True

    def _build_instance_loss(self) -> nn.Module:
        cc_loss = VectorizedCCLoss(activation=softmax_helper_dim1, mode="dice_ce")
        return cc_loss


class nnUNetTrainerBlobDiceCE(_InstanceTrainerBase):
    include_global_component = False
    use_instance_loss = True

    def _build_instance_loss(self) -> nn.Module:
        blob_loss = VectorizedBlobLoss(activation=softmax_helper_dim1, mode="dice_ce")
        return blob_loss


class nnUNetTrainerGlobalCCDiceCE(_InstanceTrainerBase):
    include_global_component = True
    use_instance_loss = True

    def _build_instance_loss(self) -> nn.Module:
        cc_loss = VectorizedCCLoss(activation=softmax_helper_dim1, mode="dice_ce")
        return cc_loss


class nnUNetTrainerGlobalBlobDiceCE(_InstanceTrainerBase):
    include_global_component = True
    use_instance_loss = True

    def _build_instance_loss(self) -> nn.Module:
        blob_loss = VectorizedBlobLoss(activation=softmax_helper_dim1, mode="dice_ce")
        return blob_loss

class nnUNetTrainerDiceCEBaseline(_InstanceTrainerBase):
    include_global_component = True
    use_instance_loss = False
