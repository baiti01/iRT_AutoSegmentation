#!/usr/bin/env python
# -*- coding:utf-8 -*-
# AUTHOR: Ti Bai
# EMAIL: tibaiw@gmail.com
# AFFILIATION: MAIA Lab | UT Southwestern Medical Center
# DATETIME: 10/08/2021 7:02 PM

import inspect
import warnings
from typing import Callable, Optional, Union, List

import torch
from monai.losses import DiceLoss
from monai.networks import one_hot
from monai.utils import LossReduction
from torch.nn.modules.loss import _Loss


class DiceLossCustomized(DiceLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input: torch.Tensor, target: torch.Tensor, reduce_axis=None) -> torch.Tensor:
        """
         Args:
             input: the shape should be BNH[WD], where N is the number of classes.
             target: the shape should be BNH[WD] or B1H[WD], where N is the number of classes.

         Raises:
             AssertionError: When input and target (after one hot transform if set)
                 have different shapes.
             ValueError: When ``self.reduction`` is not one of ["mean", "sum", "none"].

         Example:
             >>> from monai.losses.dice import *  # NOQA
             >>> import torch
             >>> from monai.losses.dice import DiceLoss
             >>> B, C, H, W = 7, 5, 3, 2
             >>> input = torch.rand(B, C, H, W)
             >>> target_idx = torch.randint(low=0, high=C - 1, size=(B, H, W)).long()
             >>> target = one_hot(target_idx[:, None, ...], num_classes=C)
             >>> self = DiceLoss(reduction='none')
             >>> loss = self(input, target)
             >>> assert np.broadcast_shapes(loss.shape, input.shape) == input.shape
         """
        if self.sigmoid:
            input = torch.sigmoid(input)

        n_pred_ch = input.shape[1]
        if self.softmax:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `softmax=True` ignored.")
            else:
                input = torch.softmax(input, 1)

        if self.other_act is not None:
            input = self.other_act(input)

        if self.to_onehot_y:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `to_onehot_y=True` ignored.")
            else:
                target = one_hot(target, num_classes=n_pred_ch)

        if not self.include_background:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `include_background=False` ignored.")
            else:
                # if skipping background, removing first channel
                target = target[:, 1:]
                input = input[:, 1:]

        if target.shape != input.shape:
            raise AssertionError(f"ground truth has different shape ({target.shape}) from input ({input.shape})")

        # reducing only spatial dimensions (not batch nor channels)
        if reduce_axis is None:
            reduce_axis: List[int] = torch.arange(2, len(input.shape)).tolist()
        if self.batch:
            # reducing spatial dimensions and batch
            reduce_axis = [0] + reduce_axis

        intersection = torch.sum(target * input, dim=reduce_axis)

        if self.squared_pred:
            target = torch.pow(target, 2)
            input = torch.pow(input, 2)

        ground_o = torch.sum(target, dim=reduce_axis)
        pred_o = torch.sum(input, dim=reduce_axis)

        denominator = ground_o + pred_o

        if self.jaccard:
            denominator = 2.0 * (denominator - intersection)

        f: torch.Tensor = 1.0 - (2.0 * intersection + self.smooth_nr) / (denominator + self.smooth_dr)

        if self.reduction == LossReduction.MEAN.value:
            f = torch.mean(f)  # the batch and channel average
        elif self.reduction == LossReduction.SUM.value:
            f = torch.sum(f)  # sum over the batch and channel dims
        elif self.reduction == LossReduction.NONE.value:
            # If we are not computing voxelwise loss components at least
            # make sure a none reduction maintains a broadcastable shape
            # broadcast_shape = list(f.shape[0:2]) + [1] * (len(input.shape) - 2)
            broadcast_shape = list(f.shape) + [1] * (len(input.shape) - len(f.shape))
            f = f.view(broadcast_shape)
        else:
            raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')

        return f


class ClassSpatialMaskedLoss(_Loss):
    """
    This is a wrapper class for the loss functions.  It allows for additional
    weighting masks to be applied to both input and target.

    See Also:
        - :py:class:`monai.losses.MaskedDiceLoss`
    """

    def __init__(self, loss: Union[Callable, _Loss], *loss_args, **loss_kwargs) -> None:
        """
        Args:
            loss: loss function to be wrapped, this could be a loss class or an instance of a loss class.
            loss_args: arguments to the loss function's constructor if `loss` is a class.
            loss_kwargs: keyword arguments to the loss function's constructor if `loss` is a class.
        """
        super().__init__()
        self.loss = loss(*loss_args, **loss_kwargs) if inspect.isclass(loss) else loss
        if not callable(self.loss):
            raise ValueError("The loss function is not callable.")

    def forward(self, input: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None, reduce_axis=None):
        """
        Args:
            input: the shape should be BNH[WD].
            target: the shape should be BNH[WD].
            mask: the shape should be BNH[WD] or 1NH[WD].
        """

        if mask is None:
            warnings.warn("No mask value specified for the MaskedLoss.")
            return self.loss(input, target)

        if input.dim() != mask.dim():
            warnings.warn(f"Dim of input ({input.shape}) is different from mask ({mask.shape}).")
        if input.shape[0] != mask.shape[0] and mask.shape[0] != 1:
            raise ValueError(f"Batch size of mask ({mask.shape}) must be one or equal to input ({input.shape}).")
        if target.dim() > 1:
            if input.shape[1:] != mask.shape[1:]:
                warnings.warn(
                    f"Spatial size and channel size of input ({input.shape}) is different from mask ({mask.shape}).")
        return self.loss(torch.sigmoid(input) * mask, target * mask, reduce_axis=reduce_axis)


class ClassSpatialMaskedDiceLoss(DiceLossCustomized):
    """
    Add an additional `masking` process before `DiceLoss`, accept a binary mask ([0, 1]) indicating a region,
    `input` and `target` will be masked by the region: region with mask `1` will keep the original value,
    region with `0` mask will be converted to `0`. Then feed `input` and `target` to normal `DiceLoss` computation.
    This has the effect of ensuring only the masked region contributes to the loss computation and
    hence gradient calculation.

    """

    def __init__(self, *args, **kwargs) -> None:
        """
        Args follow :py:class:`monai.losses.DiceLoss`.
        """
        super().__init__(*args, **kwargs)
        self.spatial_weighted = ClassSpatialMaskedLoss(loss=super().forward)

    def forward(self, input: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None, reduce_axis=None):
        """
        Args:
            input: the shape should be BNH[WD].
            target: the shape should be BNH[WD].
            mask: the shape should B1H[WD] or 1NH[WD].
        """
        return self.spatial_weighted(input=input, target=target, mask=mask, reduce_axis=reduce_axis)
