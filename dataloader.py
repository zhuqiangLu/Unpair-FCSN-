import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import h5py


class FeatureVectorData(Dataset):
    def __init__(self, data_root):

        self.data = h5py.File(data_root, 'r')
        self.keys = self.data.keys()
        self._data = [i for i in range(len(self.data))]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        idx += 1
        key = 'video_{}'.format(idx)
        video = self.data[key]
        features = torch.t(torch.tensor(video['features']))
        print(features.shape)
        gtscore = torch.tensor(video['gtscore'][()], dtype=torch.long)
        print(gtscore.shape)
        return features


def get_dataloader(path, batch_size=1):
    train_dataset = FeatureVectorData(path)
    loader = DataLoader(train_dataset, batch_size=batch_size)
    return loader


if __name__ == "__main__":
    data_root = 'dataset/eccv16_dataset_summe_google_pool5.h5'
    loader = get_dataloader(data_root)
    for i, batch in enumerate(loader):
        print(i)
        print(batch.shape)
