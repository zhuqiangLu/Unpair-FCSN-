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
        # print(features.shape)
        gtsummary = torch.tensor(video['user_summary'][()], dtype=torch.long)
        # print(gtscore.shape)
        return features, gtsummary


def get_dataloader(path, ratio, batch_size=1):

    dataset = FeatureVectorData(path)

    train_len = int(len(dataset) * ratio)
    test_len = len(dataset) - train_len

    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, (train_len, test_len))
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    return train_loader, test_dataset


if __name__ == "__main__":
    data_root = 'FeatureVectors/eccv16_dataset_summe_google_pool5.h5'
    train_loader, test_dataset = get_dataloader(data_root, ratio=1)
    for i, feature, gt in enumerate(train_loader):
        print(i)
        print(batch.shape)
