import torch, torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mmseg.models.losses import DiceLoss
from torch.cuda.amp import autocast

from . import OPENOCC_LOSS
from .base_loss import BaseLoss
@OPENOCC_LOSS.register_module()
class DepthLoss(BaseLoss):
    def __init__(self, weight = 1.0, input_dict = None, max_depth=60):
        super().__init__()
        self.loss_func = self.loss
        self.loss_weight = weight
        self.max_depth = max_depth
        if input_dict is None:
            self.input_dict = {
                'depths': 'depths',
                'gt_depths': 'gt_depths',
            }
        else:
            self.input_dict = input_dict
    
    def loss(self, depths, gt_depths):
        loss = 0.0
    
        for pred, gt in zip(depths, gt_depths[0]): 
            pred = pred.permute(0, 2, 3, 1).contiguous().reshape(-1)
            gt = gt.reshape(-1)
            gt = torch.tensor(gt, dtype=torch.float32, device=pred.device)
            fg_mask = torch.logical_and(
                gt > 0.0, torch.logical_not(torch.isnan(pred))
            )
            gt = gt[fg_mask]
            pred = pred[fg_mask]
            pred = torch.clip(pred, 0.0, self.max_depth)
            with autocast(enabled=False):
                error = torch.abs(pred - gt).sum()
                _loss = (
                    error
                    / max(1.0, len(gt) * len(depths))
                    * self.loss_weight
                )
            loss = loss + _loss
        return loss