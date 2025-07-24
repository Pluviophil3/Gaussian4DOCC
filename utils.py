import torch
import numpy as np
def inside_voxel(gaussian, occ):
    means = gaussian.means[0].detach().cpu().numpy()
    voxels = occ[0].cpu().to(torch.int)
    sems = gaussian.semantics[0].detach().cpu().numpy() # g, 18
    pred = np.argmax(sems, axis=-1)
    total = 0
    count = 0
    count2 = 0
    for coord, current_pred in zip(means, pred):
        x = int((coord[0] + 50) * 2)
        y = int((coord[1] + 50) * 2)
        z = int((coord[2] + 5) * 2)
        total += 1
        if voxels[x, y, z] != 17:
            count += 1
            if current_pred == voxels[x, y, z]:
                count2 += 1
    return count, count2, total