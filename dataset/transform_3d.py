import os
import torch
import numpy as np
from numpy import random
import mmcv
from PIL import Image
import math
from .file_client import FileClient
from . import OPENOCC_TRANSFORMS
@OPENOCC_TRANSFORMS.register_module()
class LoadPointsFromFile(object):
    """Load Points From File.

    Load sunrgbd and scannet points from file.

    Args:
        coord_type (str): The type of coordinates of points cloud.
            Available options includes:
            - 'LIDAR': Points in LiDAR coordinates.
            - 'DEPTH': Points in depth coordinates, usually for indoor dataset.
            - 'CAMERA': Points in camera coordinates.
        load_dim (int): The dimension of the loaded points.
            Defaults to 6.
        use_dim (list[int]): Which dimensions of the points to be used.
            Defaults to [0, 1, 2]. For KITTI dataset, set use_dim=4
            or use_dim=[0, 1, 2, 3] to use the intensity dimension.
        shift_height (bool): Whether to use shifted height. Defaults to False.
        use_color (bool): Whether to use color features. Defaults to False.
        file_client_args (dict): Config dict of file clients, refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details. Defaults to dict(backend='disk').
    """

    def __init__(self,
                 coord_type,
                 load_dim=6,
                 use_dim=[0, 1, 2],
                 shift_height=False,
                 use_color=False,
                 file_client_args=dict(backend='disk')):
        self.shift_height = shift_height
        self.use_color = use_color
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        assert max(use_dim) < load_dim, \
            f'Expect all used dimensions < {load_dim}, got {use_dim}'
        assert coord_type in ['CAMERA', 'LIDAR', 'DEPTH']

        self.coord_type = coord_type
        self.load_dim = load_dim
        self.use_dim = use_dim
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def _load_points(self, pts_filename):
        """Private function to load point clouds data.

        Args:
            pts_filename (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        if self.file_client is None:
            self.file_client = FileClient(**self.file_client_args)
        try:
            pts_bytes = self.file_client.get(pts_filename)
            points = np.frombuffer(pts_bytes, dtype=np.float32)
        except ConnectionError:
            mmcv.check_file_exist(pts_filename)
            if pts_filename.endswith('.npy'):
                points = np.load(pts_filename)
            else:
                points = np.fromfile(pts_filename, dtype=np.float32)

        return points
    def __call__(self, results):
        """Call function to load points data from file.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the point clouds data. \
                Added key and value are described below.

                - points (:obj:`BasePoints`): Point clouds data.
        """
        pts_filename = results['pts_filename']
        points = self._load_points(pts_filename)
        points = points.reshape(-1, self.load_dim)
        points = points[:, self.use_dim]
        attribute_dims = None

        if self.shift_height:
            floor_height = np.percentile(points[:, 2], 0.99)
            height = points[:, 2] - floor_height
            points = np.concatenate(
                [points[:, :3],
                 np.expand_dims(height, 1), points[:, 3:]], 1)
            attribute_dims = dict(height=3)

        if self.use_color:
            assert len(self.use_dim) >= 6
            if attribute_dims is None:
                attribute_dims = dict()
            attribute_dims.update(
                dict(color=[
                    points.shape[1] - 3,
                    points.shape[1] - 2,
                    points.shape[1] - 1,
                ]))
        from .points import get_points_type
        points_class = get_points_type(self.coord_type)
        points = points_class(
            points, points_dim=points.shape[-1], attribute_dims=attribute_dims)
        results['points'] = points

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__ + '('
        repr_str += f'shift_height={self.shift_height}, '
        repr_str += f'use_color={self.use_color}, '
        repr_str += f'file_client_args={self.file_client_args}, '
        repr_str += f'load_dim={self.load_dim}, '
        repr_str += f'use_dim={self.use_dim})'
        return repr_str



@OPENOCC_TRANSFORMS.register_module()
class DefaultFormatBundle(object):
    """Default formatting bundle.

    It simplifies the pipeline of formatting common fields, including "img",
    "proposals", "gt_bboxes", "gt_labels", "gt_masks" and "gt_semantic_seg".
    These fields are formatted as follows.

    - img: (1)transpose, (2)to tensor, (3)to DataContainer (stack=True)
    - proposals: (1)to tensor, (2)to DataContainer
    - gt_bboxes: (1)to tensor, (2)to DataContainer
    - gt_bboxes_ignore: (1)to tensor, (2)to DataContainer
    - gt_labels: (1)to tensor, (2)to DataContainer
    - gt_masks: (1)to tensor, (2)to DataContainer (cpu_only=True)
    - gt_semantic_seg: (1)unsqueeze dim-0 (2)to tensor,
                       (3)to DataContainer (stack=True)
    """

    def __init__(self, ):
        return

    def __call__(self, results):
        """Call function to transform and format common fields in results.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data that is formatted with
                default bundle.
        """
        if 'img' in results:
            if isinstance(results['img'], list):
                # process multiple imgs in single frame
                imgs = [img.transpose(2, 0, 1) for img in results['img']] #h, w, c = (864, 1600, 3)
                # len(results) = 6
                imgs = np.ascontiguousarray(np.stack(imgs, axis=0))
            else:
                imgs = np.ascontiguousarray(results['img'].transpose(2, 0, 1))
            results['img'] = torch.from_numpy(imgs)
        return results

    def __repr__(self):
        return self.__class__.__name__


@OPENOCC_TRANSFORMS.register_module()
class NuScenesAdaptor(object):
    def __init__(self, num_cams, use_ego=False):
        self.num_cams = num_cams
        self.projection_key = 'ego2img' if use_ego else 'lidar2img'
        self.T_key = 'ego2global' if use_ego else 'lidar2global'
        pass

    def __call__(self, input_dict):
        input_dict["projection_mat"] = np.float32( 
            np.stack(input_dict[self.projection_key])
        )
        input_dict["image_wh"] = np.ascontiguousarray( 
            np.array(input_dict["img_shape"], dtype=np.float32)[:, :2][:, ::-1]
        )
        return input_dict


@OPENOCC_TRANSFORMS.register_module()
class ResizeCropFlipImage(object):
    def __call__(self, results):
        aug_configs = results.get("aug_configs")
        if aug_configs is None:
            return results
        resize, resize_dims, crop, flip, rotate = aug_configs
        imgs = results["img"]
        N = len(imgs)
        new_imgs = []
        for i in range(N):
            img = Image.fromarray(np.uint8(imgs[i]))
            img, ida_mat = self._img_transform(
                img,
                resize=resize,
                resize_dims=resize_dims,
                crop=crop,
                flip=flip,
                rotate=rotate,
            )
            mat = np.eye(4)
            mat[:3, :3] = ida_mat
            new_imgs.append(np.array(img).astype(np.float32))
            results["lidar2img"][i] = mat @ results["lidar2img"][i]
            results["ego2img"][i] = mat @ results["ego2img"][i]
            if "cam_intrinsic" in results:
                results["cam_intrinsic"][i][:3, :3] *= resize

        results["img"] = new_imgs
        results["img_shape"] = [x.shape[:2] for x in new_imgs]
        return results

    def _get_rot(self, h):
        return torch.Tensor(
            [
                [np.cos(h), np.sin(h)],
                [-np.sin(h), np.cos(h)],
            ]
        )

    def _img_transform(self, img, resize, resize_dims, crop, flip, rotate):
        ida_rot = torch.eye(2)
        ida_tran = torch.zeros(2)
        # adjust image
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)

        # post-homography transformation
        ida_rot *= resize
        ida_tran -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            ida_rot = A.matmul(ida_rot)
            ida_tran = A.matmul(ida_tran) + b
        A = self._get_rot(rotate / 180 * np.pi)
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        ida_rot = A.matmul(ida_rot)
        ida_tran = A.matmul(ida_tran) + b
        ida_mat = torch.eye(3)
        ida_mat[:2, :2] = ida_rot
        ida_mat[:2, 2] = ida_tran
        return img, ida_mat


@OPENOCC_TRANSFORMS.register_module()
class NormalizeMultiviewImage(object):
    """Normalize the image.
    Added key is "img_norm_cfg".
    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, results):
        """Call function to normalize images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """
        results["img"] = [
            mmcv.imnormalize(img, self.mean, self.std, self.to_rgb)
            for img in results["img"]
        ]
        results["img_norm_cfg"] = dict(
            mean=self.mean, std=self.std, to_rgb=self.to_rgb
        )
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(mean={self.mean}, std={self.std}, to_rgb={self.to_rgb})"
        return repr_str


@OPENOCC_TRANSFORMS.register_module()
class PhotoMetricDistortionMultiViewImage:
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.
    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)
    8. randomly swap channels
    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(
        self,
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18,
    ):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def __call__(self, results):
        """Call function to perform photometric distortion on images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Result dict with images distorted.
        """
        imgs = results["img"]
        new_imgs = []
        for img in imgs:
            assert img.dtype == np.float32, (
                "PhotoMetricDistortion needs the input image of dtype np.float32,"
                ' please set "to_float32=True" in "LoadImageFromFile" pipeline'
            )
            # random brightness
            if random.randint(2):
                delta = random.uniform(
                    -self.brightness_delta, self.brightness_delta
                )
                img += delta

            # mode == 0 --> do random contrast first
            # mode == 1 --> do random contrast last
            mode = random.randint(2)
            if mode == 1:
                if random.randint(2):
                    alpha = random.uniform(
                        self.contrast_lower, self.contrast_upper
                    )
                    img *= alpha

            # convert color from BGR to HSV
            img = mmcv.bgr2hsv(img)

            # random saturation
            if random.randint(2):
                img[..., 1] *= random.uniform(
                    self.saturation_lower, self.saturation_upper
                )

            # random hue
            if random.randint(2):
                img[..., 0] += random.uniform(-self.hue_delta, self.hue_delta)
                img[..., 0][img[..., 0] > 360] -= 360
                img[..., 0][img[..., 0] < 0] += 360

            # convert color from HSV to BGR
            img = mmcv.hsv2bgr(img)

            # random contrast
            if mode == 0:
                if random.randint(2):
                    alpha = random.uniform(
                        self.contrast_lower, self.contrast_upper
                    )
                    img *= alpha

            # randomly swap channels
            if random.randint(2):
                img = img[..., random.permutation(3)]
            new_imgs.append(img)
        results["img"] = new_imgs
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(\nbrightness_delta={self.brightness_delta},\n"
        repr_str += "contrast_range="
        repr_str += f"{(self.contrast_lower, self.contrast_upper)},\n"
        repr_str += "saturation_range="
        repr_str += f"{(self.saturation_lower, self.saturation_upper)},\n"
        repr_str += f"hue_delta={self.hue_delta})"
        return repr_str


@OPENOCC_TRANSFORMS.register_module()
class LoadMultiViewImageFromFiles(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool, optional): Whether to convert the img to float32.
            Defaults to False.
        color_type (str, optional): Color type of the file.
            Defaults to 'unchanged'.
    """

    def __init__(self, to_float32=False, color_type='unchanged'):
        self.to_float32 = to_float32
        self.color_type = color_type

    def __call__(self, results):
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data.
                Added keys and values are described below.

                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        filename = results['img_filename']
        # img is of shape (h, w, c, num_views)
        img = np.stack(
            [mmcv.imread(name, self.color_type) for name in filename], axis=-1)
        if self.to_float32:
            img = img.astype(np.float32)
        results['filename'] = filename
        # unravel to list, see `DefaultFormatBundle` in formatting.py
        # which will transpose each image separately and then stack into array
        results['img'] = [img[..., i] for i in range(img.shape[-1])]
        # print(results['img'][0].shape)
        
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32}, '
        repr_str += f"color_type='{self.color_type}')"
        return repr_str

@OPENOCC_TRANSFORMS.register_module()
class MultiScaleDepthMapGenerator(object):
    def __init__(self, downsample=1, max_depth=60):
        if not isinstance(downsample, (list, tuple)):
            downsample = [downsample]
        self.downsample = downsample
        self.max_depth = max_depth

    def __call__(self, input_dict):
        points = input_dict["points"].tensor[..., :3, None].cpu().numpy()
        gt_depth = []
        for i, lidar2img in enumerate(input_dict["lidar2img"]):
            H, W = input_dict["img_shape"][i][:2]
            pts_2d = (
                np.squeeze(lidar2img[:3, :3] @ points, axis=-1)
                + lidar2img[:3, 3]
            )
            pts_2d[:, :2] /= pts_2d[:, 2:3]
            U = np.round(pts_2d[:, 0]).astype(np.int32)
            V = np.round(pts_2d[:, 1]).astype(np.int32)
            depths = pts_2d[:, 2]
            mask = np.logical_and.reduce(
                [
                    V >= 0,
                    V < H,
                    U >= 0,
                    U < W,
                    depths >= 0.1,
                    depths <= self.max_depth,
                ]
            )
            V, U, depths = V[mask], U[mask], depths[mask]
            sort_idx = np.argsort(depths)[::-1]
            V, U, depths = V[sort_idx], U[sort_idx], depths[sort_idx]

            for j, downsample in enumerate(self.downsample):
                if len(gt_depth) < j + 1:
                    gt_depth.append([])
                h, w = (int(H / downsample), int(W / downsample))
                u = np.floor(U / downsample).astype(np.int32)
                v = np.floor(V / downsample).astype(np.int32)
                depth_map = np.ones([h, w], dtype=np.float32) * -1
                depth_map[v, u] = depths
                gt_depth[j].append(depth_map)
                #print(f"depth_map.shape = {depth_map.shape}")
        input_dict["gt_depths"] = [np.stack(x) for x in gt_depth]
        return input_dict
        
@OPENOCC_TRANSFORMS.register_module()
class LoadPointFromFile(object):

    def __init__(self, pc_range, num_pts, use_ego=False):
        self.use_ego = use_ego
        self.pc_range = pc_range
        self.num_pts = num_pts
        pass

    def __call__(self, results):
        pts_path = results['pts_filename']
        scan = np.fromfile(pts_path, dtype=np.float32)
        scan = scan.reshape((-1, 5))[:, :4]
        scan[:, 3] = 1.0 # n, 4
        if self.use_ego:
            ego2lidar = results['ego2lidar']
            lidar2ego = np.linalg.inv(ego2lidar)
            scan = lidar2ego[None, ...] @ scan[..., None]
            scan = np.squeeze(scan, axis=-1)
        scan = scan[:, :3] # n, 3

        ### filter
        norm = np.linalg.norm(scan, 2, axis=-1)
        mask = (scan[:, 0] > self.pc_range[0]) & (scan[:, 0] < self.pc_range[3]) & \
            (scan[:, 1] > self.pc_range[1]) & (scan[:, 1] < self.pc_range[4]) & \
            (scan[:, 2] > self.pc_range[2]) & (scan[:, 2] < self.pc_range[5]) & \
            (norm > 1.0)
        scan = scan[mask]

        ### append
        if scan.shape[0] < self.num_pts:
            multi = int(math.ceil(self.num_pts * 1.0 / scan.shape[0])) - 1
            scan_ = np.repeat(scan, multi, 0)
            scan_ = scan_ + np.random.randn(*scan_.shape) * 0.3
            scan_ = scan_[np.random.choice(scan_.shape[0], self.num_pts - scan.shape[0], False)]
            scan_[:, 0] = np.clip(scan_[:, 0], self.pc_range[0], self.pc_range[3])
            scan_[:, 1] = np.clip(scan_[:, 1], self.pc_range[1], self.pc_range[4])
            scan_[:, 2] = np.clip(scan_[:, 2], self.pc_range[2], self.pc_range[5])
            scan = np.concatenate([scan, scan_], 0)
        else:
            scan = scan[np.random.choice(scan.shape[0], self.num_pts, False)]
        
        scan[:, 0] = (scan[:, 0] - self.pc_range[0]) / (self.pc_range[3] - self.pc_range[0])
        scan[:, 1] = (scan[:, 1] - self.pc_range[1]) / (self.pc_range[4] - self.pc_range[1])
        scan[:, 2] = (scan[:, 2] - self.pc_range[2]) / (self.pc_range[5] - self.pc_range[2])
        results['anchor_points'] = scan
        
        return results
    
    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        return repr_str


@OPENOCC_TRANSFORMS.register_module()
class LoadPseudoPointFromFile(object):

    def __init__(self, datapath, pc_range, num_pts, is_ego=True, use_ego=False):
        self.datapath = datapath
        self.is_ego = is_ego
        self.use_ego = use_ego
        self.pc_range = pc_range
        self.num_pts = num_pts
        pass

    def __call__(self, results):
        pts_path = os.path.join(self.datapath, f"{results['sample_idx']}.npy")
        scan = np.load(pts_path)
        if self.is_ego and (not self.use_ego):
            ego2lidar = results['ego2lidar']
            scan = np.concatenate([scan, np.ones_like(scan[:, :1])], axis=-1)
            scan = ego2lidar[None, ...] @ scan[..., None] # p, 4, 1
            scan = np.squeeze(scan, axis=-1)

        if (not self.is_ego) and self.use_ego:
            ego2lidar = results['ego2lidar']
            lidar2ego = np.linalg.inv(ego2lidar)
            scan = np.concatenate([scan, np.ones_like(scan[:, :1])], axis=-1)
            scan = lidar2ego[None, ...] @ scan[..., None]
            scan = np.squeeze(scan, axis=-1)
        
        scan = scan[:, :3] # n, 3

        ### filter
        norm = np.linalg.norm(scan, 2, axis=-1)
        mask = (scan[:, 0] > self.pc_range[0]) & (scan[:, 0] < self.pc_range[3]) & \
            (scan[:, 1] > self.pc_range[1]) & (scan[:, 1] < self.pc_range[4]) & \
            (scan[:, 2] > self.pc_range[2]) & (scan[:, 2] < self.pc_range[5]) & \
            (norm > 1.0)
        scan = scan[mask]

        ### append
        if scan.shape[0] < self.num_pts:
            multi = int(math.ceil(self.num_pts * 1.0 / scan.shape[0])) - 1
            scan_ = np.repeat(scan, multi, 0)
            scan_ = scan_ + np.random.randn(*scan_.shape) * 0.3
            scan_ = scan_[np.random.choice(scan_.shape[0], self.num_pts - scan.shape[0], False)]
            scan_[:, 0] = np.clip(scan_[:, 0], self.pc_range[0], self.pc_range[3])
            scan_[:, 1] = np.clip(scan_[:, 1], self.pc_range[1], self.pc_range[4])
            scan_[:, 2] = np.clip(scan_[:, 2], self.pc_range[2], self.pc_range[5])
            scan = np.concatenate([scan, scan_], 0)
        else:
            scan = scan[np.random.choice(scan.shape[0], self.num_pts, False)]
        
        scan[:, 0] = (scan[:, 0] - self.pc_range[0]) / (self.pc_range[3] - self.pc_range[0])
        scan[:, 1] = (scan[:, 1] - self.pc_range[1]) / (self.pc_range[4] - self.pc_range[1])
        scan[:, 2] = (scan[:, 2] - self.pc_range[2]) / (self.pc_range[5] - self.pc_range[2])
        results['anchor_points'] = scan
        
        return results
    
    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        return repr_str


@OPENOCC_TRANSFORMS.register_module()
class LoadOccupancySurroundOcc(object):

    def __init__(self, occ_path, semantic=False, use_ego=False, use_sweeps=False):
        self.occ_path = occ_path
        self.semantic = semantic
        self.use_ego = use_ego
        assert semantic and (not use_ego)
        self.use_sweeps = use_sweeps

        xyz = self.get_meshgrid([-50, -50, -5.0, 50, 50, 3.0], [200, 200, 16], 0.5)
        self.xyz = np.concatenate([xyz, np.ones_like(xyz[..., :1])], axis=-1) # x, y, z, 4

    def get_meshgrid(self, ranges, grid, reso):
        xxx = torch.arange(grid[0], dtype=torch.float) * reso + 0.5 * reso + ranges[0]
        yyy = torch.arange(grid[1], dtype=torch.float) * reso + 0.5 * reso + ranges[1]
        zzz = torch.arange(grid[2], dtype=torch.float) * reso + 0.5 * reso + ranges[2]

        xxx = xxx[:, None, None].expand(*grid)
        yyy = yyy[None, :, None].expand(*grid)
        zzz = zzz[None, None, :].expand(*grid)

        xyz = torch.stack([
            xxx, yyy, zzz
        ], dim=-1).numpy()
        return xyz # x, y, z, 3

    def __call__(self, results):
        label_file = os.path.join(self.occ_path, results['pts_filename'].split('/')[-1]+'.npy')
        if os.path.exists(label_file):
            label = np.load(label_file)

            new_label = np.ones((200, 200, 16), dtype=np.int64) * 17
            new_label[label[:, 0], label[:, 1], label[:, 2]] = label[:, 3]

            mask = new_label != 0

            results['occ_label'] = new_label if self.semantic else new_label != 17
            results['occ_cam_mask'] = mask
        elif self.use_sweeps:
            new_label = np.ones((200, 200, 16), dtype=np.int64) * 17
            mask = new_label != 0
            results['occ_label'] = new_label if self.semantic else new_label != 17
            results['occ_cam_mask'] = mask
        else:
            raise NotImplementedError
        
        if not self.use_ego:
            occ_xyz = self.xyz[..., :3]
        else:
            ego2lidar = np.linalg.inv(results['ego2lidar']) # 4, 4
            occ_xyz = ego2lidar[None, None, None, ...] @ self.xyz[..., None] # x, y, z, 4, 1
            occ_xyz = np.squeeze(occ_xyz, -1)[..., :3]
        results['occ_xyz'] = occ_xyz
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        return repr_str

@OPENOCC_TRANSFORMS.register_module()
class LoadOccupancyKITTI360(object):

    def __init__(self, occ_path, semantic=False):
        self.occ_path = occ_path
        self.semantic = semantic

        xyz = self.get_meshgrid([0.0, -25.6, -2.0, 51.2, 25.6, 4.4], [256, 256, 32], 0.2)
        self.xyz = np.concatenate([xyz, np.ones_like(xyz[..., :1])], axis=-1) # x, y, z, 4

    def get_meshgrid(self, ranges, grid, reso):
        xxx = torch.arange(grid[0], dtype=torch.float) * reso + 0.5 * reso + ranges[0]
        yyy = torch.arange(grid[1], dtype=torch.float) * reso + 0.5 * reso + ranges[1]
        zzz = torch.arange(grid[2], dtype=torch.float) * reso + 0.5 * reso + ranges[2]

        xxx = xxx[:, None, None].expand(*grid)
        yyy = yyy[None, :, None].expand(*grid)
        zzz = zzz[None, None, :].expand(*grid)

        xyz = torch.stack([
            xxx, yyy, zzz
        ], dim=-1).numpy()
        return xyz # x, y, z, 3

    def __call__(self, results):        
        occ_xyz = self.xyz[..., :3]
        results['occ_xyz'] = occ_xyz

        ## read occupancy label
        label_path = os.path.join(
            self.occ_path, results['sequence'], "{}_1_1.npy".format(results['token']))
        label = np.load(label_path).astype(np.int64)

        results['occ_cam_mask'] = (label != 255)
        results['occ_label'] = label
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        return repr_str
