import time

import numpy as np
import torch
import yaml


def normalize_image(rgb):
    """Normalize image values from [0, 255] to [0, 1]."""
    if torch.is_tensor(rgb):
        return rgb.float() / 255
    if isinstance(rgb, np.ndarray):
        return rgb.astype(float) / 255
    raise TypeError(f"Unsupported input rgb type: {type(rgb)}")


def channels_first(rgb):
    """Convert arrays/tensors from (..., H, W, C) to (..., C, H, W)."""
    if not (isinstance(rgb, np.ndarray) or torch.is_tensor(rgb)):
        raise TypeError(f"Unsupported input rgb type {type(rgb)}")

    if rgb.ndim < 3:
        raise ValueError(
            f"Input rgb must contain at least 3 dims, but had {rgb.ndim} dims."
        )

    ordering = list(range(rgb.ndim))
    ordering[-2], ordering[-1], ordering[-3] = ordering[-3], ordering[-2], ordering[-1]

    if isinstance(rgb, np.ndarray):
        return np.ascontiguousarray(rgb.transpose(*ordering))
    return rgb.permute(*ordering).contiguous()


def scale_intrinsics(intrinsics, h_ratio, w_ratio):
    """Scale intrinsics for resized frames."""
    if isinstance(intrinsics, np.ndarray):
        scaled_intrinsics = intrinsics.astype(np.float32).copy()
    elif torch.is_tensor(intrinsics):
        scaled_intrinsics = intrinsics.to(torch.float).clone()
    else:
        raise TypeError(f"Unsupported input intrinsics type {type(intrinsics)}")

    if not (intrinsics.shape[-2:] == (3, 3) or intrinsics.shape[-2:] == (4, 4)):
        raise ValueError(
            "intrinsics must have shape (*, 3, 3) or (*, 4, 4), "
            f"but had shape {intrinsics.shape} instead"
        )

    scaled_intrinsics[..., 0, 0] *= w_ratio  # fx
    scaled_intrinsics[..., 1, 1] *= h_ratio  # fy
    scaled_intrinsics[..., 0, 2] *= w_ratio  # cx
    scaled_intrinsics[..., 1, 2] *= h_ratio  # cy
    return scaled_intrinsics


def relative_transformation(
    trans_01: torch.Tensor, trans_02: torch.Tensor, orthogonal_rotations: bool = False
) -> torch.Tensor:
    """Compute the relative homogeneous transform between trans_01 and trans_02."""
    if orthogonal_rotations:
        rot = trans_01[..., :3, :3]
        trans = trans_01[..., :3, 3:4]
        trans_10 = torch.eye(4, dtype=trans_01.dtype, device=trans_01.device)
        if trans_01.dim() == 3:
            trans_10 = trans_10.unsqueeze(0).repeat(trans_01.shape[0], 1, 1)
        trans_10[..., :3, :3] = rot.transpose(-1, -2)
        trans_10[..., :3, 3:4] = -rot.transpose(-1, -2) @ trans
    else:
        trans_10 = torch.inverse(trans_01)
    return torch.matmul(trans_10, trans_02)


def measure_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        print(f"Done! Execution time of {func.__name__} function: {elapsed_time:.2f} seconds")
        return result

    return wrapper


def as_intrinsics_matrix(intrinsics):
    """Get matrix representation of intrinsics."""
    K = np.eye(3)
    K[0, 0] = intrinsics[0]
    K[1, 1] = intrinsics[1]
    K[0, 2] = intrinsics[2]
    K[1, 2] = intrinsics[3]
    return K


def load_dataset_config(path, default_path=None):
    """Load dataset config with optional inheritance."""
    with open(path, "r") as f:
        cfg_special = yaml.full_load(f)

    inherit_from = cfg_special.get("inherit_from")

    if inherit_from is not None:
        cfg = load_dataset_config(inherit_from, default_path)
    elif default_path is not None:
        with open(default_path, "r") as f:
            cfg = yaml.full_load(f)
    else:
        cfg = dict()

    update_recursive(cfg, cfg_special)
    return cfg


def update_recursive(dict1, dict2):
    """Update config dictionaries recursively."""
    for k, v in dict2.items():
        if k not in dict1:
            dict1[k] = dict()
        if isinstance(v, dict):
            update_recursive(dict1[k], v)
        else:
            dict1[k] = v
