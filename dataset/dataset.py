import os
from copy import deepcopy
import numpy as np
from pyquaternion import Quaternion
from torch.utils.data import Dataset

import mmengine
from . import OPENOCC_DATASET, OPENOCC_TRANSFORMS
from .utils import get_img2global, get_lidar2global, custom_collate_fn_temporal
from nuscenes.can_bus.can_bus_api import NuScenesCanBus

@OPENOCC_DATASET.register_module()
class NuScenesDataset(Dataset):

    def __init__(
        self,
        data_root=None,
        imageset=None,
        data_aug_conf=None,
        pipeline=None,
        vis_indices=None,
        num_samples=0,
        phase='train'
    ):
        self.nusc_can = NuScenesCanBus(dataroot=data_root)
        self.data_path = data_root
        data = mmengine.load(imageset)
        self.scene_infos = data['infos']
        self.keyframes = data['metadata']
        self.keyframes = sorted(self.keyframes, key=lambda x: x[0] + "{:0>3}".format(str(x[1])))
        
        for idx in range(len(self.keyframes) - 1, -1, -1):
            if idx == 0 or self.keyframes[idx][0] != self.keyframes[idx - 1][0]:
                self.keyframes[idx] = (self.keyframes[idx], self.keyframes[idx][1])
            else:
                self.keyframes[idx] = (self.keyframes[idx], self.keyframes[idx - 1][1])

        self.data_aug_conf = data_aug_conf
        self.test_mode = (phase != 'train')
        self.pipeline = []
        
        for t in pipeline:
            self.pipeline.append(OPENOCC_TRANSFORMS.build(t))

        self.sensor_types = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 
            'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
        if vis_indices is not None:
            if len(vis_indices) > 0:
                vis_indices = [i % len(self.keyframes) for i in vis_indices]
                self.keyframes = [self.keyframes[idx] for idx in vis_indices]
            elif num_samples > 0:
                vis_indices = np.random.choice(len(self.keyframes), num_samples, False)
                self.keyframes = [self.keyframes[idx] for idx in vis_indices]
        elif num_samples > 0:
            vis_indices = np.random.choice(len(self.keyframes), num_samples, False)
            self.keyframes = [self.keyframes[idx] for idx in vis_indices]

    def _sample_augmentation(self):
        H, W = self.data_aug_conf["H"], self.data_aug_conf["W"]
        fH, fW = self.data_aug_conf["final_dim"]
        if not self.test_mode:
            resize = np.random.uniform(*self.data_aug_conf["resize_lim"])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = (
                int(
                    (1 - np.random.uniform(*self.data_aug_conf["bot_pct_lim"]))
                    * newH
                )
                - fH
            )
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.data_aug_conf["rand_flip"] and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.data_aug_conf["rot_lim"])
        else:
            resize = max(fH / H, fW / W)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = (
                int((1 - np.mean(self.data_aug_conf["bot_pct_lim"])) * newH)
                - fH
            )
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0
        return resize, resize_dims, crop, flip, rotate


    def __getitem__(self, index):
        scene_token, index, prev_index = self.keyframes[index][0][0], self.keyframes[index][0][1], self.keyframes[index][1]
        info = deepcopy(self.scene_infos[scene_token][index])
        input_dict = self.get_data_info(info)
        pre_info = deepcopy(self.scene_infos[scene_token][prev_index])
        pre_input_dict = self.get_data_info(pre_info)
        if self.data_aug_conf is not None:
            input_dict["aug_configs"] = self._sample_augmentation()
        for t in self.pipeline:
            input_dict = t(input_dict)
        if self.data_aug_conf is not None:
            pre_input_dict["aug_configs"] = self._sample_augmentation()
        for t in self.pipeline:
            pre_input_dict = t(pre_input_dict)
        if os.environ['eval'] == 'false':
            return_dict =  ({
                'can_bus':input_dict['can_bus'],
                'img': input_dict['img'],
                'projection_mat': input_dict['projection_mat'],
                'image_wh': input_dict['image_wh'],
                'occ_label': input_dict['occ_label'],
                'occ_xyz': input_dict['occ_xyz'],
                'occ_cam_mask': input_dict['occ_cam_mask'],
                'gt_depths': input_dict['gt_depths'],
            },
            {
                'can_bus':pre_input_dict['can_bus'],
                'img': pre_input_dict['img'],
                'projection_mat': pre_input_dict['projection_mat'],
                'image_wh': pre_input_dict['image_wh'],
                'occ_label': pre_input_dict['occ_label'],
                'occ_xyz': pre_input_dict['occ_xyz'],
                'occ_cam_mask': pre_input_dict['occ_cam_mask'],
            }
            )
        else:
            return_dict =  ({
                'can_bus':input_dict['can_bus'],
                'img': input_dict['img'],
                'projection_mat': input_dict['projection_mat'],
                'image_wh': input_dict['image_wh'],
                'occ_label': input_dict['occ_label'],
                'occ_xyz': input_dict['occ_xyz'],
                'occ_cam_mask': input_dict['occ_cam_mask'],
            },
            {
                'can_bus':pre_input_dict['can_bus'],
                'img': pre_input_dict['img'],
                'projection_mat': pre_input_dict['projection_mat'],
                'image_wh': pre_input_dict['image_wh'],
                'occ_label': pre_input_dict['occ_label'],
                'occ_xyz': pre_input_dict['occ_xyz'],
                'occ_cam_mask': pre_input_dict['occ_cam_mask'],
            }
            )
        return return_dict
    
    def get_data_info(self, info):
        image_paths = []
        lidar2img_rts = []
        img2lidar_rts = []
        cam_intrinsics = []
        cam2ego_rts = []
        ego2image_rts = []

        lidar2ego_r = Quaternion(info['data']['LIDAR_TOP']['calib']['rotation']).rotation_matrix
        lidar2ego = np.eye(4)
        lidar2ego[:3, :3] = lidar2ego_r
        lidar2ego[:3, 3] = np.array(info['data']['LIDAR_TOP']['calib']['translation']).T
        ego2lidar = np.linalg.inv(lidar2ego)

        lidar2global = get_lidar2global(info['data']['LIDAR_TOP']['calib'], info['data']['LIDAR_TOP']['pose'])
        ego2global = np.eye(4)
        ego2global[:3, :3] = Quaternion(info['data']['LIDAR_TOP']['pose']['rotation']).rotation_matrix
        ego2global[:3, 3] = np.asarray(info['data']['LIDAR_TOP']['pose']['translation']).T

        for cam_type in self.sensor_types:
            image_paths.append(os.path.join(self.data_path, info['data'][cam_type]['filename']))

            img2global = get_img2global(info['data'][cam_type]['calib'], info['data'][cam_type]['pose'])
            lidar2img = np.linalg.inv(img2global) @ lidar2global
            img2lidar = np.linalg.inv(lidar2global) @ img2global

            cam2ego_r = Quaternion(info['data'][cam_type]['calib']['rotation']).rotation_matrix
            cam2ego = np.eye(4)
            cam2ego[:3, :3] = cam2ego_r
            cam2ego[:3, 3] = np.array(info['data'][cam_type]['calib']['translation']).T

            intrinsic = info['data'][cam_type]['calib']['camera_intrinsic']
            viewpad = np.eye(4)
            viewpad[:3, :3] = intrinsic

            lidar2img_rts.append(lidar2img)
            img2lidar_rts.append(img2lidar)
            cam_intrinsics.append(viewpad)
            cam2ego_rts.append(cam2ego)
            ego2image_rts.append(np.linalg.inv(img2global) @ ego2global)
        
        can_bus = self.get_can_bus(info)

        input_dict =dict(
            sample_idx=info["token"],
            pts_filename=os.path.join(self.data_path, info['data']['LIDAR_TOP']['filename']),
            occ_path=info["occ_path"],
            timestamp=info["timestamp"] / 1e6,
            can_bus = can_bus,
            ego2global=ego2global,
            lidar2global=lidar2global,
            img_filename=image_paths,
            lidar2img=np.asarray(lidar2img_rts),
            img2lidar=np.asarray(img2lidar_rts),
            cam_intrinsic=np.asarray(cam_intrinsics),
            ori_intrinsic=np.array(cam_intrinsics).copy(),
            ego2lidar=ego2lidar,
            cam2ego=np.asarray(cam2ego_rts),
            ego2img=np.asarray(ego2image_rts))

        return input_dict

    def get_can_bus(self,info):
        can_bus = []
        if 'occ_path' in info:
            scene = info['occ_path'].split('/')[1]
            try:
                pose_list = self.nusc_can.get_messages(scene, 'pose')
            except:
                return np.zeros(18)
        else:
            return np.zeros(18)  # server scenes do not have can bus information.
        sample_timestamp = info['timestamp']
        # during each scene, the first timestamp of can_bus may be large than the first sample's timestamp
        last_pose = pose_list[0]
        for i, pose in enumerate(pose_list):
            if pose['utime'] > sample_timestamp:
                break
            last_pose = pose
        _ = last_pose.pop('utime')  # useless
        pos = last_pose.pop('pos')
        rotation = last_pose.pop('orientation')
        can_bus.extend(pos)
        can_bus.extend(rotation)
        for key in last_pose.keys():
            can_bus.extend(pose[key])  # 16 elements
        can_bus.extend([0., 0.])
        can_bus = np.array(can_bus)
        return can_bus

    def __len__(self):
        return len(self.keyframes)