import os
import torch
import numpy as np
from glob import glob
from pathlib import Path
from typing import Optional, Tuple
from PIL import Image
from .dataset import RangeDataset, RangeLoader, point_cloud_to_range_image
import torch.nn.functional as F

COLOR_MAP = {
    0:  [0,   0,   0],    # unlabeled -> black
    1:  [255, 0,   0],    # e.g. "car"
    2:  [0,   255, 0],    # e.g. "bicycle"
    3:  [0,   0,   255],  # e.g. "motorcycle"
    4:  [255, 255, 0],    # e.g. "truck"
    5:  [255, 0,   255],  # e.g. "other-vehicle"
    6:  [0,   255, 255],  # e.g. "person"
    7:  [128, 0,   0],    # e.g. "bicyclist"
    8:  [0,   128, 0],    # e.g. "motorcyclist"
    9:  [0,   0,   128],  # e.g. "road"
    10: [128, 128, 0],    # e.g. "parking"
    11: [128, 0,   128],  # e.g. "sidewalk"
    12: [0,   128, 128],  # e.g. "other-ground"
    13: [128, 128, 128],  # e.g. "building"
    14: [64,  0,   0],    # e.g. "fence"
    15: [0,   64,  0],    # e.g. "vegetation"
    16: [0,   0,   64],   # e.g. "trunk"
    17: [64,  64,  0],    # e.g. "terrain"
    18: [64,  0,   64],   # e.g. "pole"
    19: [0,   64,  64],   # e.g. "traffic-sign"
}

LEARNING_MAP = {
    0: 0, 1: 0,
    10: 1, 11: 2, 13: 5, 15: 3, 16: 5, 18: 4,
    20: 5, 30: 6, 31: 7, 32: 8, 40: 9, 44: 10,
    48: 11, 49: 12, 50: 13, 51: 14, 52: 0, 60: 9,
    70: 15, 71: 16, 72: 17, 80: 18, 81: 19, 99: 0,
    252: 1, 253: 7, 254: 6, 255: 8, 256: 5, 257: 5,
    258: 4, 259: 5
}


def colorize_range_img(range_img, vmin=0.0, vmax=1.0, cmap_name="jet"):
    """
    Convert a single-channel range image (tensor or NumPy array)
    into a color PNG using a matplotlib colormap.
    """
    import matplotlib.pyplot as plt
    from matplotlib import cm

    if isinstance(range_img, torch.Tensor):
        range_img = range_img.detach().cpu().numpy()

    clipped = np.clip(range_img, vmin, vmax)
    normed = (clipped - vmin) / (vmax - vmin + 1e-8)

    color_map = cm.get_cmap(cmap_name)(normed)  # shape (W,H,4)
    color_map = (color_map[..., :3] * 255).astype(np.uint8)

    # If your array is (W,H), transpose for PIL which expects (H,W).
    color_map = np.transpose(color_map, (1, 0, 2))  # => (H,W,3)

    return Image.fromarray(color_map, 'RGB')

def colorize_label_img(label_img_np, color_map_dict, assume_bgr=True):
    """
    label_img_np: (H, W) NumPy array of integer label IDs.
    color_map_dict: { label_id: [B, G, R] } or [R, G, B].
    assume_bgr: If True, interpret the dictionary as BGR and convert to RGB for PIL.

    returns: PIL.Image in RGB mode.
    """
    H, W = label_img_np.shape
    color_image = np.zeros((H, W, 3), dtype=np.uint8)

    for row in range(H):
        for col in range(W):
            lbl_id = label_img_np[row, col]
            # If the label ID is not in the dictionary, default to black
            color_bgr = color_map_dict[lbl_id]
            
            # Convert BGR -> RGB if needed
            if assume_bgr:
                color_rgb = color_bgr[::-1]  # reverse [B, G, R] -> [R, G, B]
            else:
                color_rgb = color_bgr

            color_image[row, col] = color_rgb

    return Image.fromarray(color_image, mode='RGB')

class point_cloud_to_range_image_KITTI_vanilla(point_cloud_to_range_image):
    """
    Same projection class as in your KITTI360/SemanticKITTI code.
    """
    def __init__(self, 
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.H = 64
        self.proj_fov_up = 3.0 / 180.0 * np.pi
        self.proj_fov_down = -25.0 / 180.0 * np.pi
        self.fov = self.proj_fov_up - self.proj_fov_down
        self.height = np.zeros(self.H)

    def get_row_inds(self, pc):
        point_range = np.linalg.norm(pc[:,:3], axis=1, ord=2) + 1e-8
        zen = np.arcsin(pc[:,2] / point_range)

        row_inds = self.H - 1.0 + 0.5 - (zen - self.proj_fov_down) / self.fov * self.H
        row_inds = np.round(row_inds).astype(np.int32)
        row_inds[row_inds >= self.H] = self.H - 1
        row_inds[row_inds < 0] = 0
        return row_inds

    def get_col_inds(self, pc):
        """
        Typically the horizontal angle is -atan2(y, x). 
        Range: [-pi, pi], shift to [0, 2*pi], then scale to [0, width).
        """
        azi = -np.arctan2(pc[:,1], pc[:,0])
        azi[azi < 0] += 2.0 * np.pi
        col_inds = np.floor(azi / (2.0 * np.pi) * self.width).astype(np.int32)
        col_inds[col_inds >= self.width] = self.width - 1
        col_inds[col_inds < 0] = 0
        return col_inds
    
    def project_points_labels(self, pc, labels=None):
        """
        pc: N x 4 (x, y, z, intensity) or N x 3
        labels: N (semantic label) or None

        Returns:
        range_image: (H, W, 2) => [range, intensity]
        label_image: (H, W) => integer labels
        """
        N = pc.shape[0]
        row_inds = self.get_row_inds(pc)  # shape (N,)
        col_inds = self.get_col_inds(pc)  # shape (N,)

        # Distance
        ranges = np.linalg.norm(pc[:, :3], axis=1)

        # Prepare buffers
        range_image = np.zeros((self.H, self.width, 2), dtype=np.float32)
        depth_buffer = np.zeros((self.H, self.width), dtype=np.float32)
        label_image = None
        if labels is not None:
            label_image = np.zeros((self.H, self.width), dtype=np.int32)

        for i in range(N):
            r  = ranges[i]
            rr = row_inds[i]
            cc = col_inds[i]
            
            # If there's no label array, treat new_label as 0 or ignore
            new_label = labels[i] if (labels is not None) else 0

            old_label = 0
            if label_image is not None:
                old_label = label_image[rr, cc]
            
            # The existing range in the pixel
            old_range = depth_buffer[rr, cc]

            # -------------
            # LOGIC FLOW:
            # -------------
            if old_range == 0.0:
                # Pixel empty => store the new point/label no matter what
                depth_buffer[rr, cc] = r
                range_image[rr, cc, 0] = r
                if pc.shape[1] == 4:  # intensity
                    range_image[rr, cc, 1] = pc[i, 3]
                if label_image is not None:
                    label_image[rr, cc] = new_label
            else:
                # Pixel already has a point (and label)
                if old_label == 0 and new_label != 0:
                    # Case A: existing is unlabeled, new is labeled => override
                    depth_buffer[rr, cc] = r
                    range_image[rr, cc, 0] = r
                    if pc.shape[1] == 4:  # intensity
                        range_image[rr, cc, 1] = pc[i, 3]
                    label_image[rr, cc] = new_label

                elif old_label != 0 and new_label == 0:
                    # Case B: existing is labeled, new is unlabeled => ignore new
                    pass

                elif old_label != 0 and new_label != 0:
                    # Case C: both old and new are labeled => keep the *closer* one
                    if r < old_range:
                        depth_buffer[rr, cc] = r
                        range_image[rr, cc, 0] = r
                        if pc.shape[1] == 4:  # intensity
                            range_image[rr, cc, 1] = pc[i, 3]
                        label_image[rr, cc] = new_label
                else:
                    # Case D: both old and new are unlabeled => just keep whichever is closer
                    if r < old_range:
                        depth_buffer[rr, cc] = r
                        range_image[rr, cc, 0] = r
                        if pc.shape[1] == 4:
                            range_image[rr, cc, 1] = pc[i, 3]
                        if label_image is not None:
                            label_image[rr, cc] = new_label

        return range_image, label_image


    def to_pc_torch(self, range_images):
        """
        Convert range images to point clouds (same logic as your KITTI code).
        range_images: Bx2xWxH -> point_cloud: BxNx4 (if remission is present)
        """
        device = range_images.device
        batch_size, channels, width_dim, height_dim = range_images.shape

        # Extract point range and remission
        if self.log:
            point_range = 2**(range_images[:, 0, :, :] * 6) - 1
        elif self.inverse:
            point_range = 1/torch.max(
                range_images[:, 0, :, :], 
                torch.Tensor([0.0001]).to(device)
            )
        else:
            point_range = range_images[:, 0, :, :] * self.std + self.mean

        if channels > 1:
            remission = range_images[:, 1, :, :].reshape(batch_size, -1)

        r_true = point_range 
        zen = (height_dim - 0.5 - torch.arange(0, height_dim, device=device)) / height_dim * self.fov + self.proj_fov_down
        z = (r_true * torch.sin(zen[None, None, :])).reshape(batch_size, -1)
        xy_norm = r_true * torch.cos(zen[None, None, :])

        azi = (width_dim - 0.5 - torch.arange(0, width_dim, device=device)) / width_dim * 2. * torch.pi - torch.pi
        x = (xy_norm * torch.cos(azi[None, :, None])).reshape(batch_size, -1)
        y = (xy_norm * torch.sin(azi[None, :, None])).reshape(batch_size, -1)

        if channels > 1:
            point_cloud = torch.stack([x, y, z, remission], dim=2)
        else:
            point_cloud = torch.stack([x, y, z], dim=2)

        return point_cloud


class SemanticKITTIDataset(RangeDataset):
    """
    Dataset class for SemanticKITTI that also loads labels and
    projects them to the range image.
    """
    def __init__(self, 
                 SEMANTIC_KITTI_path, 
                 train=True, 
                 width=1024, 
                 grid_sizes=[1, 1024, 1024], 
                 train_sequences=[0,1,2,3,4,5,6,7,9,10],
                 valid_sequences=[8],
                 pc_range=[-25.6, -25.6, -3., 25.6, 25.6, 1.], 
                 log=False,
                 inverse=False,
                 **kwargs):
        super().__init__(**kwargs)

        self.SEMANTIC_KITTI_path = SEMANTIC_KITTI_path
        self.train = train
        self.width = width
        self.grid_sizes = grid_sizes
        self.pc_range = pc_range
        self.log = log
        self.inverse = inverse
        
        # Typically, for SemanticKITTI:
        #   * train: sequences 00–07, 09, 10
        #   * valid/test: sequence 08
        all_files = glob(
            os.path.join(SEMANTIC_KITTI_path, 'sequences/*/velodyne/*.bin')
        )
        def get_sequence_id(f):
            # Example path: /path/to/sequences/08/velodyne/XXXX.bin
            # `os.path.dirname(f)` = /path/to/sequences/08/velodyne
            # `os.path.dirname(os.path.dirname(f))` = /path/to/sequences/08
            # We take the basename of that = '08', then convert to int.
            seq_str = os.path.basename(
                os.path.dirname(
                    os.path.dirname(f)
                )
            )
            return int(seq_str)
        if train:
            self.file_paths = [
                f for f in all_files 
                if get_sequence_id(f) in train_sequences
            ]
        else:
            self.file_paths = [
                f for f in all_files 
                if get_sequence_id(f) in valid_sequences
            ]
        self.file_paths.sort()

        # Our range image converter
        self.to_range_image = point_cloud_to_range_image_KITTI_vanilla(
            width=width, 
            grid_sizes=grid_sizes, 
            pc_range=pc_range, 
            log=log,
            inverse=inverse
        )

    def get_pts(self, pts_path):
        """
        Return Nx4 array: x, y, z, intensity
        """
        return np.fromfile(pts_path, dtype=np.float32).reshape(-1, 4)
    
    def get_label_path(self, pts_path):
        """
        Derive label file from velodyne path. Example:
          .../sequences/00/velodyne/000123.bin
        => .../sequences/00/labels/000123.label
        """
        seq_id = pts_path.split('/')[-3]  # e.g. "00"
        file_id = Path(pts_path).stem     # e.g. "000123"
        label_path = os.path.join(
            self.SEMANTIC_KITTI_path, 
            "sequences", 
            seq_id, 
            "labels", 
            file_id + ".label"
        )
        return label_path

    def get_labels(self, label_path):
        """
        Load label file: Nx1 of type uint32
        - The lower 16 bits is the semantic label
        - The upper 16 bits is the instance ID (if you need it)
        """
        if not os.path.exists(label_path):
            # If you don't have labels (e.g., for test set), handle gracefully
            return None
        raw_labels = np.fromfile(label_path, dtype=np.uint32)
        sem_labels = raw_labels & 0xFFFF
        mapped_labels = np.zeros_like(sem_labels, dtype=np.int32)
        for i, old_id in enumerate(sem_labels):
            mapped_labels[i] = LEARNING_MAP[old_id]
        # instance_labels = raw_labels >> 16
        return mapped_labels

    def get_pth_path(self, pts_path):
        """
        Optional: caching location for the data
        """
        new_path = pts_path.replace('dataset', 'dataset_range_vanilla_wlabel').replace('.bin', '.pth')
        return new_path

    def __getitem__(self, idx):
        pts_path = self.file_paths[idx]
        lbl_path = self.get_label_path(pts_path)
        pts = self.get_pts(pts_path)
        lbl = self.get_labels(lbl_path)  # Nx1 array of semantic labels

        # Project into range image
        range_img, label_img = self.to_range_image.project_points_labels(pts, lbl)
        
        # Optional normalization:
        range_img = self.to_range_image.normalize(range_img)
        
        range_img_torch = torch.from_numpy(range_img).permute(2, 1, 0).float()
        # -----------------------------------------------------
        # 2) Integer label image -> One-hot
        # -----------------------------------------------------
        # label_img originally (H, W), each pixel an integer 0..num_classes-1
        label_img_torch = torch.from_numpy(label_img).long()  # shape (H, W)

        # Convert to one-hot => shape (H, W, num_classes), then permute to (num_classes, W, H).
        # For example, if num_classes=20:
        one_hot_label = F.one_hot(label_img_torch, num_classes=20)          # (H, W, 20)
        label_img_torch_one_hot = one_hot_label.permute(2, 1, 0).float()             # (20, W, H)
        label_img_torch = label_img_torch.unsqueeze(0).permute(0, 2, 1).float()
        # If you already have an attribute like `self.num_classes`, use that:
        # one_hot_label = F.one_hot(label_img_torch, num_classes=self.num_classes).permute(2, 1, 0).float()

        ret = {
            'jpg': range_img_torch,     # shape (2, W, H) 
            'origin_sem': label_img_torch,       # shape (1, W, H)
            'onehot_sem': label_img_torch_one_hot,
            'pts_path': pts_path,
            'lbl_path': lbl_path
        }
        return ret


class SemanticKITTILoader(RangeLoader):
    """
    DataLoader class for SemanticKITTI, now returning range images + label images.
    """
    def __init__(self, 
                 SEMANTIC_KITTI_path, 
                 used_feature=2, 
                 width=1024, 
                 grid_sizes=[1, 1024, 1024], 
                 pc_range=[-25.6, -25.6, -3., 25.6, 25.6, 1.], 
                 log=False, 
                 inverse=False, 
                 downsample=None,
                 inpainting=None,
                 coord=False,
                 **kwargs):
        super().__init__(**kwargs)

        self.train_dataset = SemanticKITTIDataset(
            SEMANTIC_KITTI_path, 
            train=True,
            used_feature=used_feature,
            width=width, 
            grid_sizes=grid_sizes,
            pc_range=pc_range, 
            log=log,
            inverse=inverse,
            downsample=downsample,
            inpainting=inpainting,
            coord=coord
        )

        self.test_dataset = SemanticKITTIDataset(
            SEMANTIC_KITTI_path, 
            train=False,
            used_feature=used_feature, 
            width=width, 
            grid_sizes=grid_sizes, 
            pc_range=pc_range,
            log=log,
            inverse=inverse,
            downsample=downsample,
            inpainting=inpainting,
            coord=coord
        )
