from scipy.stats import median_abs_deviation
import torch
import numpy as np

def L1_MAD_weighted(x_pert,x_orig):
    
    MAD = median_abs_deviation(x_orig.cpu().numpy(),axis = 0)
    MAD = np.where(np.abs(MAD) < 0.0001, np.array(0.01), MAD)
    diff = torch.abs(x_orig - x_pert)
    return (diff / torch.tensor(MAD, device=x_orig.device)).sum(dim = 1)


def adv_loss(lamb,
            adv_logits,
            y_target,
            x_orig,
            x_pert):

    sq_diff = lamb * (adv_logits - y_target) ** 2
    dist_loss = L1_MAD_weighted(x_pert,x_orig)
    
    return (sq_diff + dist_loss).mean()