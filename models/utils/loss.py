# Imports
from torch import nn
import torch.nn.functional as F
import numpy as np
from scipy.spatial import cKDTree as KDTree

# ~~~~~~~~~~~~~~~
# LOSS FUNCTIONS
# ~~~~~~~~~~~~~~~

class DecoderLoss(nn.Module):
    """ 
    Decoder main Loss function.
    
    Args:
        -- config_loss (dict): Configuration dictionary for loss function
    """
    def __init__(self, config_loss:dict):
        super().__init__()
        self.loss_name = config_loss["loss_name"]
    # ----------------
    
    def forward(self, y_hat, y_gt):
        if self.loss_name == 'L2':
            return F.mse_loss(y_hat, y_gt)

        elif self.loss_name == 'L1':
            return F.l1_loss(y_hat, y_gt)
# =============================


def compute_chamfer(gt_coords, coords_pred_on_surface=None, observed_coords=None, pred_sdf=None, level=1e-4):
    """ 
    Compute training/validation chamfer loss between ground truth and predicted coordinates.
    """
    if gt_coords is not None and not isinstance(gt_coords, np.ndarray):
        # assuming pytorch tensor then
        gt_coords = gt_coords.detach().cpu().numpy()

    if observed_coords is not None and not isinstance(observed_coords, np.ndarray):
        # assuming pytorch tensor then
        observed_coords = observed_coords.detach().cpu().numpy()
        pred_sdf = pred_sdf.detach().cpu().numpy()

    if pred_sdf is not None:
        mask_pred = np.abs(pred_sdf) <= level
        if np.sum(mask_pred) > 0:
            coords_pred_on_surface = observed_coords[mask_pred]
        else:
            # TODO, maybe we can handle this differently? None of the predicted SDF values are close to 0
            # hence no on surface coordinates. We also predict SDF for coords away from the surface. Computing then
            # chamfer less always results in values further off. BUt maybe we should just take the coords based
            # on SDF (hence observed) and compute chamfer
            # print("Warning - Chamfer_loss - predicted sdf ")
            return -1

    if coords_pred_on_surface is not None and not isinstance(coords_pred_on_surface, np.ndarray):
        coords_pred_on_surface = coords_pred_on_surface.detach().cpu().numpy()

    # one direction
    gen_points_kd_tree = KDTree(coords_pred_on_surface)
    one_distances, one_vertex_ids = gen_points_kd_tree.query(gt_coords)
    gt_to_pred_chamfer = np.mean(np.square(one_distances))

    # other direction
    gt_points_kd_tree = KDTree(gt_coords)
    two_distances, two_vertex_ids = gt_points_kd_tree.query(coords_pred_on_surface)
    pred_to_gt_chamfer = np.mean(np.square(two_distances))
    chamfer_loss = gt_to_pred_chamfer + pred_to_gt_chamfer
    # print("chamfer loss ", coords_pred.shape, coords_gt.shape)
    return chamfer_loss
# ----------------------------
