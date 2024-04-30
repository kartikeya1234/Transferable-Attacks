from torch.utils.data import Dataset

class CustomDataset(Dataset):
    
    """CustomDataset

    Args:
        X (torch.tensor): Features
        Y (torch.tensor): Labels
    """
    
    def __init__(self, X, Y):
        
        self.X  = X
        self.Y = Y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]