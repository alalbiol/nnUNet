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
            11: 21, 21: 11,
            12: 22, 22: 12,
            13: 23, 23: 13,
            14: 24, 24: 14,
            15: 25, 25: 15,
            16: 26, 26: 16,
            17: 27, 27: 17,
            18: 28, 28: 18,
            41: 31, 31: 41,
            42: 32, 32: 42,
            43: 33, 33: 43,
            44: 34, 34: 44,
            45: 35, 35: 45,
            46: 36, 36: 46,
            47: 37, 37: 47,
            48: 38, 38: 48,
            103: 104, 104: 103,
            111: 121, 121: 111,
            112: 122, 122: 112,
            113: 123, 123: 113,
            114: 124, 124: 114,
            115: 125, 125: 115,
            116: 126, 126: 116,
            117: 127, 127: 117,
            118: 128, 128: 118,
            141: 131, 131: 141,
            142: 132, 132: 142,
            143: 133, 133: 143,
            144: 134, 134: 144,
            145: 135, 135: 145,
            146: 136, 136: 146,
            147: 137, 137: 147,
            148: 138, 138: 148,
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