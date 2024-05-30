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

class MinMaxScalerTorch:

    def __init__(self, X) -> None:
        
        self.X = X
        self.mean = None
        self.max = None
        self.min = None

    def fit(self):
        self.min = self.X.min(dim=0).values
        self.max = self.X.max(dim=0).values
        self.mean = self.X.mean(dim=0)

    def transform(self, X):    
        assert self.min != None, "Scaler not fitted"
        assert self.max != None, "Scaler not fitted"
        assert self.mean != None, "Scaler not fitted" 
    
        XScaled = (X - self.min) / (self.max - self.min)
        return XScaled

    def inverseTransform(self, XScaled):
        assert self.min != None, "Scaler not fitted"
        assert self.max != None, "Scaler not fitted"
        assert self.mean != None, "Scaler not fitted" 

        XUnscaled = XScaled * (self.max - self.min) + self.min
        return XUnscaled