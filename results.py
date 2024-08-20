import os
import numpy as np
import cv2


class Results(object):
    def __init__(self, mask_dir):
        self.mask_dir = mask_dir

    def _read_mask(self, sequence, frame_id):
        try:
            mask_path = os.path.join(self.mask_dir, sequence, f'{frame_id}.png')
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = mask.astype(bool).astype(np.uint8)
            return mask
        except IOError as err:
            print(sequence + " frame %s not found!" % frame_id)
            print("The frames have to be indexed PNG files placed inside the corespondent sequence folder.\nThe indexes have to match with the initial frame.")
            print("IOError: " + err.strerror)
            exit()

    def read_masks(self, sequence, masks_id):
        mask_0 = self._read_mask(sequence, masks_id[0])
        masks = np.zeros((len(masks_id), *mask_0.shape))
        for ii, m in enumerate(masks_id):
            masks[ii, ...] = self._read_mask(sequence, m)
        return masks
