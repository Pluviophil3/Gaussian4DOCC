from mmengine.registry import MODELS
from mmengine.model import BaseModule
from mmcv.cnn import Scale
import torch.nn as nn, torch
import torch.nn.functional as F
from .utils import linear_relu_ln, GaussianPrediction
from ...utils.safe_ops import safe_sigmoid


@MODELS.register_module()
class GaussianFusion(BaseModule):
    def forward(
        self,
        instance_feature: torch.Tensor,
        prev_embed: torch.Tensor,
    ):
        output = torch.cat([instance_feature, prev_embed], dim = -1)
        return output


