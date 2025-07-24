import torch
import numpy as np
import torch.nn.functional as F


def list_2_tensor(lst, key, tensor: torch.Tensor):
    values = []

    for dct in lst:
        values.append(dct[key])
    if isinstance(values[0], (np.ndarray, list)):
        rst = np.stack(values, axis=0)
    elif isinstance(values[0], torch.Tensor):
        rst = torch.stack(values, dim=0)
    else:
        raise NotImplementedError
    
    return tensor.new_tensor(rst)


def get_rotation_matrix(tensor):
    assert tensor.shape[-1] == 4

    tensor = F.normalize(tensor, dim=-1)
    mat1 = torch.zeros(*tensor.shape[:-1], 4, 4, dtype=tensor.dtype, device=tensor.device)
    mat1[..., 0, 0] = tensor[..., 0]
    mat1[..., 0, 1] = - tensor[..., 1]
    mat1[..., 0, 2] = - tensor[..., 2]
    mat1[..., 0, 3] = - tensor[..., 3]
    
    mat1[..., 1, 0] = tensor[..., 1]
    mat1[..., 1, 1] = tensor[..., 0]
    mat1[..., 1, 2] = - tensor[..., 3]
    mat1[..., 1, 3] = tensor[..., 2]

    mat1[..., 2, 0] = tensor[..., 2]
    mat1[..., 2, 1] = tensor[..., 3]
    mat1[..., 2, 2] = tensor[..., 0]
    mat1[..., 2, 3] = - tensor[..., 1]

    mat1[..., 3, 0] = tensor[..., 3]
    mat1[..., 3, 1] = - tensor[..., 2]
    mat1[..., 3, 2] = tensor[..., 1]
    mat1[..., 3, 3] = tensor[..., 0]

    mat2 = torch.zeros(*tensor.shape[:-1], 4, 4, dtype=tensor.dtype, device=tensor.device)
    mat2[..., 0, 0] = tensor[..., 0]
    mat2[..., 0, 1] = - tensor[..., 1]
    mat2[..., 0, 2] = - tensor[..., 2]
    mat2[..., 0, 3] = - tensor[..., 3]
    
    mat2[..., 1, 0] = tensor[..., 1]
    mat2[..., 1, 1] = tensor[..., 0]
    mat2[..., 1, 2] = tensor[..., 3]
    mat2[..., 1, 3] = - tensor[..., 2]

    mat2[..., 2, 0] = tensor[..., 2]
    mat2[..., 2, 1] = - tensor[..., 3]
    mat2[..., 2, 2] = tensor[..., 0]
    mat2[..., 2, 3] = tensor[..., 1]

    mat2[..., 3, 0] = tensor[..., 3]
    mat2[..., 3, 1] = tensor[..., 2]
    mat2[..., 3, 2] = - tensor[..., 1]
    mat2[..., 3, 3] = tensor[..., 0]

    mat2 = torch.conj(mat2).transpose(-1, -2)
    
    mat = torch.matmul(mat1, mat2)
    return mat[..., 1:, 1:]

def align(pc_range_in, prev_rep, metas, prev_metas):
        """
        Align previous frame's Gaussian representation `prev_rep` with current frame through translation and rotation.
        """
        # 1. Get current and previous frame positions (x, y, z), ensure correct dimensions

        cur_xyz = torch.tensor(metas['can_bus'][0][:3], device=prev_rep.device)
        prev_xyz = torch.tensor(prev_metas['can_bus'][0][:3], device=prev_rep.device)
        # 2. Calculate translation vector from previous frame to current frame
        translation = cur_xyz - prev_xyz  # (3,)
        # 3. Ensure pc_range dimensions match
        pc_range = torch.tensor(pc_range_in, device=prev_rep.device)  # (6,)
        pc_min = pc_range[:3]  # (3,)
        pc_max = pc_range[3:]  # (3,)

        # 4. Normalize translation to adapt to `prev_rep`
        translation_norm = translation / (pc_max - pc_min)  # (3,) / (3,)

        # 5. Translate Gaussian centers in `prev_rep`
        prev_rep[..., :3] += translation_norm  # Only translate means (first three dimensions)

        # 6. Rotate `prev_rep` to align with current frame
        cur_orientation = torch.tensor(metas['can_bus'][0][3:7], device=prev_rep.device)  # (4,)
        prev_orientation = torch.tensor(prev_metas['can_bus'][0][3:7], device=prev_rep.device)  # (4,)
        # Calculate relative rotation quaternion
        prev_orientation_inv = quaternion_inverse(prev_orientation)
        relative_rotation = quaternion_multiply(cur_orientation, prev_orientation_inv)

        # Apply rotation transformation to `r` (quaternion part) of prev_rep
        prev_rep[..., 6:10] = quaternion_multiply(relative_rotation, prev_rep[..., 6:10])

        return prev_rep

def quaternion_inverse(q):
    """Calculate quaternion inverse (conjugate quaternion)."""
    q_inv = q.clone()
    q_inv[..., 1:] = -q_inv[..., 1:]  # Negate x, y, z
    return q_inv


def quaternion_multiply(q1, q2):
    """Quaternion multiplication q1 * q2."""
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return torch.stack([w, x, y, z], dim=-1)