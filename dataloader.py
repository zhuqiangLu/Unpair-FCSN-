import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import h5py


class FeatureVectorLoader(Dataset):
    def __init__(self, data_root):
        self.data = h5py.file(data_root, 'r')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return 
