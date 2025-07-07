from typing import Tuple

import torch

from batchgeneratorsv2.transforms.spatial.mirroring import MirrorTransform



class MirrorTransform_checkLR(MirrorTransform):
    def __init__(self, allowed_axes: Tuple[int, ...]):
        super().__init__(allowed_axes=allowed_axes)
        # Swap dictionary: {left_label: right_label, right_label: left_label}
        self.swap_dict = {
            3: 4, 4: 3,
            5: 6, 6: 5,
            11: 19, 19: 11,
            12: 20, 20: 12,
            13: 21, 21: 13,
            14: 22, 22: 14,
            15: 23, 23: 15,
            16: 24, 24: 16,
            17: 25, 25: 17,
            18: 26, 26: 18,
            27: 35, 35: 27,
            28: 36, 36: 28,
            29: 37, 37: 29,
            30: 38, 38: 30,
            31: 39, 39: 31,
            32: 40, 40: 32,
            33: 41, 41: 33,
            34: 42, 42: 34,
            }


    def _apply_to_segmentation(self, segmentation: torch.Tensor, **params) -> torch.Tensor:
        if len(params['axes']) == 0:
            return segmentation
        axes = [i + 1 for i in params['axes']]
        
        # relabel segmentation if flip LR is applied
        if 2 in axes:
            relabeled_seg = segmentation.clone()
            for old_label, new_label in self.swap_dict.items():
                relabeled_seg[segmentation == old_label] = new_label
            segmentation = relabeled_seg
            
        # Flip the segmentation tensor along the specified axes        
        return torch.flip(segmentation, axes)


class MirrorTransform_checkLR2(MirrorTransform_checkLR):
    def __init__(self, allowed_axes: Tuple[int, ...]):
        super().__init__(allowed_axes=allowed_axes)


    def _apply_to_segmentation(self, segmentation: torch.Tensor, **params) -> torch.Tensor:
        if len(params['axes']) == 0:
            return segmentation
        axes = [i + 1 for i in params['axes']]
        
        # check len axes is odd, and relabel in this case
        if len(axes) % 2 == 1:
            relabeled_seg = segmentation.clone()
            for old_label, new_label in self.swap_dict.items():
                relabeled_seg[segmentation == old_label] = new_label
            segmentation = relabeled_seg
            
        # Flip the segmentation tensor along the specified axes        
        return torch.flip(segmentation, axes)






if __name__ == '__main__':
    from time import time
    import numpy as np
    import os

    os.environ['OMP_NUM_THREADS'] = '1'
    torch.set_num_threads(1)

    mbt = MirrorTransform((0, 1, 2))

    times_torch = []
    for _ in range(100):
        data_dict = {'image': torch.ones((2, 128, 192, 64))}
        st = time()
        out = mbt(**data_dict)
        times_torch.append(time() - st)
    print('torch', np.mean(times_torch))

    from batchgenerators.transforms.spatial_transforms import MirrorTransform as BGMirror

    gnt_bg = BGMirror((0, 1, 2))
    times_bg = []
    for _ in range(100):
        data_dict = {'data': np.ones((1, 2, 128, 192, 64))}
        st = time()
        out = gnt_bg(**data_dict)
        times_bg.append(time() - st)
    print('bg', np.mean(times_bg))