import os
from collections import defaultdict
import numpy as np
import cv2


class MVPS(object):
    SUBSET_OPTIONS = ['train', 'val']

    def __init__(self, root, subset='val', sequences='all', resolution='480p'):
        if subset not in self.SUBSET_OPTIONS:
            raise ValueError(f'Subset should be in {self.SUBSET_OPTIONS}')

        self.subset = subset
        self.root = root
        self.img_path = os.path.join(self.root, 'Keyframes', resolution)
        self.mask_path = os.path.join(self.root, 'Annotations', resolution)
        self.imagesets_path = os.path.join(self.root, 'ImageSets')

        self._check_directories()

        if sequences == 'all':
            with open(os.path.join(self.imagesets_path, f'{self.subset}.txt'), 'r') as f:
                tmp = f.readlines()
            sequences_names = [x.strip() for x in tmp]
            sequences_names = list(set(sequences_names))
            sequences_names.sort()
        else:
            sequences_names = sequences if isinstance(sequences, list) else [sequences]
        self.sequences = defaultdict(dict)

        for seq in sequences_names:
            image_files = os.listdir(os.path.join(self.img_path, seq))
            images = []
            for image_file in image_files:
                if image_file.endswith('.jpg'):
                    images.append(os.path.join(self.img_path, seq, image_file))
            images.sort()
            if len(images) == 0:
                raise FileNotFoundError(f'Images for sequence {seq} not found.')
            self.sequences[seq]['images'] = images
            mask_files = os.listdir(os.path.join(self.mask_path, seq))
            masks = []
            for mask_file in mask_files:
                if mask_file.endswith('.png'):
                    masks.append(os.path.join(self.mask_path, seq, mask_file))
            masks.sort()
            if len(images) != len(masks):
                raise FileNotFoundError(f'Number of images and masks for sequence {seq} does not match.')
            self.sequences[seq]['masks'] = masks

    def _check_directories(self):
        if not os.path.exists(self.root):
            raise FileNotFoundError('MVPS not found in the specified directory')
        if not os.path.exists(os.path.join(self.imagesets_path, f'{self.subset}.txt')):
            raise FileNotFoundError(f'Subset sequences list for {self.subset} not found')
        if not os.path.exists(self.mask_path):
            raise FileNotFoundError('Annotations folder not found')

    def _get_all_elements(self, sequence, obj_type):
        obj = cv2.imread(self.sequences[sequence][obj_type][0], cv2.IMREAD_GRAYSCALE)
        all_objs = np.zeros((len(self.sequences[sequence][obj_type]), *obj.shape))
        obj_id = []
        for i, obj in enumerate(self.sequences[sequence][obj_type]):
            all_objs[i, ...] = cv2.imread(obj, cv2.IMREAD_GRAYSCALE)
            obj_id.append(os.path.splitext(os.path.basename(obj))[0])
        return all_objs, obj_id

    def get_all_masks(self, sequence):
        masks, masks_id = self._get_all_elements(sequence, 'masks')
        masks = masks.astype(bool).astype(np.uint8)
        return masks, masks_id

    def get_sequences(self):
        for seq in self.sequences:
            yield seq