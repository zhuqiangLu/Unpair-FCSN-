import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import h5py
import random


class TrainData(Dataset):
    def __init__(self, data_root, videos):

        self.size = int(len(videos)/2)
        self.V = random.sample(videos, self.size)
        videos = [v for v in videos if v not in self.V]
        self.S = random.sample(videos, self.size)

        self.data = h5py.File(data_root, 'r')
        self.keys = self.data.keys()

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        #idx += 1

        video = self.data[self.V[idx]]
        features = torch.t(torch.tensor(video['features'][()]))

        summary = self.data[self.S[idx]]
        gt = torch.tensor(summary['gt_score'][()], dtype=torch.float)
        print(self.V[idx], self.S[idx])
        return features, gt


class TestData(Dataset):
    def __init__(self, data_root, videos):
        self.videos = videos
        self.data = h5py.File(data_root, 'r')
        self.keys = self.data.keys()

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        #idx += 1
        key = self.videos[idx]
        video = self.data[key]
        features = torch.t(torch.tensor(video['features'][()]))
        # print(features.shape)
        gt = torch.tensor(video['gt_score'][()], dtype=torch.float)
        # print(gtscore.shape)
        return features, gt


class LoadersFactory():
    def __init__(self, paths, ratio=0.8):
        self.paths = paths
        self.ratio = ratio
        self.test_pair = dict()  # the test pair is fixed
        self.train_pair = dict()

        for path in paths:
            keys = [key for key in h5py.File(path, 'r').keys()]
            test_vids = random.sample(
                keys, int(len(keys) * (1-self.ratio)))
            self.test_pair[path] = test_vids
            self.train_pair[path] = [
                key for key in keys if key not in test_vids]

    def get_train_loaders(self, batch_size=1):
        loaders = dict()
        for k, v in self.train_pair.items():
            loaders[k] = DataLoader(TrainData(k, v))
        return loaders

    def get_test_data(self):
        loaders = dict()
        for k, v in self.test_pair.items():
            loaders[k] = DataLoader(TestData(k, v))
        return loaders


if __name__ == "__main__":
    data_root = 'generated_data/summe.h5'
    # train_loader, test_dataset = get_dataloader(data_root, ratio=0.8)
    # for i, feature, gt in enumerate(train_loader):
    #     print(i)
    #     print(batch.shape)

    # keys = [key for key in h5py.File(data_root, 'r').keys()]

    # Train = random.sample(keys, int(len(keys)*0.8))
    # Test = [key for key in keys if key not in Train]

    # print("train", Train)
    # print("test", Test)

    # print("testdata")
    # test_dataset = TestData(data_root, Test)
    # test_loader = DataLoader(test_dataset, batch_size=1)
    # for i, batch in enumerate(test_loader):
    #     print(i)
    #     print(batch[0].shape)
    #     print(batch[1].shape)

    # print("traindata")
    # train_dataset = TrainData(data_root, Train)
    # train_loader = DataLoader(train_dataset, batch_size=1)
    # for i, batch in enumerate(train_loader):
    #     print(i)
    #     print(batch[0].shape)
    #     print(batch[1].shape)

    roots = ['generated_data/summe.h5', 'generated_data/tvsum.h5']
    factory = LoadersFactory(roots)
    loaders = factory.get_train_loaders()
    for k, v in loaders.items():
        print(k)
        for i, batch in enumerate(v):
            print(i)
            print(batch[0].shape)
            print(batch[1].shape)
