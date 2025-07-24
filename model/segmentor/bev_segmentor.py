
from mmseg.models import SEGMENTORS
from mmseg.models import build_backbone
import copy
import torch
import os.path
from .base_segmentor import CustomBaseSegmentor
from ..utils import utils
import torch.nn as nn

@SEGMENTORS.register_module()
class BEVSegmentor(CustomBaseSegmentor):

    def __init__(
        self,
        freeze_img_backbone=False,
        freeze_img_neck=False,
        img_backbone_out_indices=[1, 2, 3],
        extra_img_backbone=None,
        # use_post_fusion=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # self.fp16_enabled = False
        self.freeze_img_backbone = freeze_img_backbone
        self.freeze_img_neck = freeze_img_neck
        self.img_backbone_out_indices = img_backbone_out_indices
        self.depth_branch = DenseDepthNet(embed_dims = 128)
        # self.use_post_fusion = use_post_fusion

        if freeze_img_backbone:
            self.img_backbone.requires_grad_(False)
        if freeze_img_neck:
            self.img_neck.requires_grad_(False)
        if extra_img_backbone is not None:
            self.extra_img_backbone = build_backbone(extra_img_backbone)
        self.pc_range = [-50.0, -50.0, -5.0, 50.0, 50.0, 3.0]
        self.scale_range = [0.08, 0.64]
        self.xyz_coordinate = 'cartesian'
        self.phi_activation = 'sigmoid'

    def extract_img_feat(self, imgs, **kwargs):
        """Extract features of images."""
        B, N, C, H, W = imgs.size()
        imgs = imgs.reshape(B * N, C, H, W)
        img_feats_backbone = self.img_backbone(imgs)
        if isinstance(img_feats_backbone, dict):
            img_feats_backbone = list(img_feats_backbone.values())
        img_feats = []
        for idx in self.img_backbone_out_indices:
            img_feats.append(img_feats_backbone[idx])
        img_feats = self.img_neck(img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped
    
    def forward_extra_img_backbone(self, imgs, **kwargs):
        """Extract features of images."""
        B, N, C, H, W = imgs.size()
        imgs = imgs.reshape(B * N, C, H, W)
        img_feats_backbone = self.extra_img_backbone(imgs)

        if isinstance(img_feats_backbone, dict):
            img_feats_backbone = list(img_feats_backbone.values())

        img_feats_backbone_reshaped = []
        for img_feat_backbone in img_feats_backbone:
            BN, C, H, W = img_feat_backbone.size()
            img_feats_backbone_reshaped.append(
                img_feat_backbone.view(B, int(BN / B), C, H, W))
        return img_feats_backbone_reshaped
    

    def obtain_history_rep(self, pre_imgs, prev_metas, metas, kwargs):
        train = self.training
        self.eval()
        os.environ['eval'] = 'true'
        with torch.no_grad():
            prev_rep = None
            for k in list(prev_metas.keys()):
                if isinstance(prev_metas[k], torch.Tensor):
                    prev_metas[k] = prev_metas[k].cuda()
            results = {
                'imgs': [pre_imgs, pre_imgs],
                'metas': [prev_metas, prev_metas],
                'prev_rep': prev_rep,
                'rep_only': True
            }
            results.update(kwargs)
            imgs_feats = self.extract_img_feat(pre_imgs)
            results['ms_img_feats'] = [imgs_feats, imgs_feats]
            outs = self.lifter(**results)
            results.update(outs)
            outs = self.encoder(**results)
            prev_rep = outs['anchor']
            if train:
                self.train()
                os.environ['eval'] = 'false'
            prev_rep = utils.align(self.pc_range, prev_rep, metas, prev_metas)
            return prev_rep, imgs_feats

    def forward(self,
                imgs=None,
                metas=None,
                points=None,
                extra_backbone=False,
                occ_only=False,
                rep_only=False,
                **kwargs,
        ):
        """Forward training function.
        """
        if extra_backbone:
            return self.forward_extra_img_backbone(imgs=imgs)
        imgs, prev_imgs = imgs
        metas, prev_metas = metas
        prev_rep, pre_img_feats = self.obtain_history_rep(prev_imgs, prev_metas, metas, kwargs)
        #prev_rep = None
        results = {
            'imgs': [prev_imgs, imgs],
            'metas': [prev_metas, metas],
            'points': points,
            'prev_rep': prev_rep
        }
        results.update(kwargs)
        img_feats = self.extract_img_feat(imgs)
        if self.training:
            results['depths'] = self.depth_branch(img_feats, metas.get('focal'))
        results['ms_img_feats'] = [pre_img_feats, img_feats]
    
        
        outs = self.lifter(**results)
        results.update(outs)
        outs = self.encoder(**results)
        if rep_only:
            return outs['representation']
        results.update(outs)
        if occ_only and hasattr(self.head, "forward_occ"):
            outs = self.head.forward_occ(**results)
        else:
            outs = self.head(**results)
        results.update(outs)
        return results

class DenseDepthNet(nn.Module):
    def __init__(
        self,
        embed_dims=128,
        num_depth_layers=3,
        equal_focal=100,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.equal_focal = equal_focal
        self.num_depth_layers = num_depth_layers

        self.depth_layers = nn.ModuleList()
        for i in range(num_depth_layers):
            self.depth_layers.append(
                nn.Conv2d(embed_dims, 1, kernel_size=1, stride=1, padding=0)
            )

    def forward(self, feature_maps, focal=None):
        if focal is None:
            focal = self.equal_focal
        else:
            focal = focal.reshape(-1)
        depths = []
        for i, feat in enumerate(feature_maps[: self.num_depth_layers]):
            # print(feat.shape)
            depth = self.depth_layers[i](feat.flatten(end_dim=1).float()).exp()
            depth = (depth.T * focal / self.equal_focal).T
            depths.append(depth)
        return depths
