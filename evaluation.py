from tqdm import tqdm
import warnings
import numpy as np
from mvps import MVPS
from metrics import db_eval_boundary, db_eval_iou
from results import Results


def db_statistics(per_frame_values, mean_only=False):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        M = np.nanmean(per_frame_values)
        if mean_only:
            return M
        O = np.nanmean(per_frame_values > 0.5)

    N_bins = 4
    ids = np.round(np.linspace(1, len(per_frame_values), N_bins + 1) + 1e-10) - 1
    ids = ids.astype(np.uint8)

    D_bins = [per_frame_values[ids[i]:ids[i + 1] + 1] for i in range(0, 4)]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        D = np.nanmean(D_bins[0]) - np.nanmean(D_bins[3])

    return M, O, D


class MVPSEvaluation(object):
    def __init__(self, mvps_root, subset, include_first_last=False, sequences='all'):
        self.mvps_root = mvps_root
        self.include_first_last = include_first_last
        self.dataset = MVPS(root=mvps_root, subset=subset, sequences=sequences)

    @staticmethod
    def _evaluate(all_gt_masks, all_res_masks):
        J = db_eval_iou(all_gt_masks, all_res_masks)
        F = db_eval_boundary(all_gt_masks, all_res_masks)
        return J, F

    def evaluate(self, mask_path):
        # Containers
        metrics_res = {'seq_names': []}
        metrics_res['J'] = {"M": [], "R": [], "D": []}
        metrics_res['F'] = {"M": [], "R": [], "D": []}

        # Sweep all sequences
        results = Results(mask_dir=mask_path)
        for seq in tqdm(list(self.dataset.get_sequences())):
            all_gt_masks, all_masks_id = self.dataset.get_all_masks(seq)
            if not self.include_first_last:
                all_gt_masks, all_masks_id = all_gt_masks[1:-1, :, :], all_masks_id[1:-1]
            all_res_masks = results.read_masks(seq, all_masks_id)
            J, F = self._evaluate(all_gt_masks, all_res_masks)
            metrics_res['seq_names'].append(seq)
            JM, JR, JD = db_statistics(J)
            metrics_res['J']["M"].append(JM)
            metrics_res['J']["R"].append(JR)
            metrics_res['J']["D"].append(JD)
            FM, FR, FD = db_statistics(F)
            metrics_res['F']["M"].append(FM)
            metrics_res['F']["R"].append(FR)
            metrics_res['F']["D"].append(FD)
        return metrics_res
