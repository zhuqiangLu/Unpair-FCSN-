import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import h5py
import random
from utils import construct_S
import numpy as np
class MultiTrainData(Dataset):
    def __init__(self, roots, video_sets):
        self.V = list()
        self.S = list()
        self.data_readers = list()
        self.video_sets = video_sets
        
        for root in roots:
            self.data_readers.append(h5py.File(root, 'r'))

        self.shuffle()

    
    def __len__(self):
        return self.size

    
    def shuffle(self):
        self.size = 0
        self.avail_data = list()
        for video_set in self.video_sets:

            tmp_size = int(len(video_set)/2)
            tmp_V = random.sample(video_set, tmp_size)
            remain_set = [v for v in video_set if v not in tmp_V]
            tmp_S = random.sample(remain_set, tmp_size)
            self.V.append(tmp_V)
            self.S.append(tmp_S)
            self.size += tmp_size
            self.avail_data.append(tmp_size)

    def __getitem__(self, idx):

        dice = [i for i in range(len(self.data_readers)) if self.avail_data[i] != 0]

        d = random.sample(dice, 1)[0]
        
        reader = self.data_readers[d]
        V = self.V[d]
        S = self.S[d]
        
        self.V[d] = self.V[d][1:]
        self.S[d] = self.S[d][1:]
        self.avail_data[d]-=1
        

        V_name = V[0]
        S_name = S[0]

        video_info_v = reader[V_name]
        features = (torch.tensor(video_info_v['features'][()]))

        video_info_s = reader[S_name] 
        summary = None
        while summary is None:
            summary = construct_S(video_info_s)
            S_name = random.sample(self.video_sets[d], 1)[0]
            video_info_s = reader[S_name]

        
        return features, summary
        



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
        features = (torch.tensor(video['features'][()]))
        #gt = video['gt_scores'][()]
        cps = np.array(video['change_points'][()], np.int32)
        picks = video['picks'][()]
        n_frame_per_seg = video['n_frame_per_seg'][()]
        gt_seg_scores = video['seg_scores'][()]
        n_frame = video['n_frame'][()]
        return features, gt_seg_scores, cps, picks, n_frame_per_seg, n_frame


class LoadersFactory():
    def __init__(self, paths, ratios):
        self.paths = paths
        self.ratios = ratios
        self.test_pair = dict()  # the test pair is fixed
        self.train_pair = dict()

        for i in range(len(paths)):
            path = paths[i]
            ratio = ratios[i]
            keys = [key for key in h5py.File(path, 'r').keys()]
            test_vids = random.sample(
                keys, int(len(keys) * round(1-ratio, 2)))
            self.test_pair[path] = test_vids
            self.train_pair[path] = [
                key for key in keys if key not in test_vids]
        
    def get_train_loaders(self, batch_size=1):
        # loaders = dict()
        # for k, v in self.train_pair.items():
        #     loaders[k] = DataLoader(TrainData(k, v))
        # return loaders
        roots = list()
        videos = list()
        for k,v in self.train_pair.items():
            roots.append(k)
            videos.append(v)
        return DataLoader(MultiTrainData(roots, videos))

    def get_test_loaders(self):
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

    roots = ['generated_data/summe.h5', 'generated_data/tvsum.h5', 'generated_data/ovp.h5','generated_data/youtube.h5']
    ratios = [0.8, 0.8, 1.0, 1.0]
    factory = LoadersFactory(roots, ratios)
    loaders = factory.get_test_loaders()
    print(len(loaders))
    print(loaders.keys())
    for k, v in loaders.items():
        print(k)
        for i, video_info in enumerate(v):
            print(i)
            print(video_info[0].shape)
            
    # print(len(loader))
    # print(len(loader))
    # from tqdm import tqdm
    # for i,batch in enumerate(loader):
    #     i = 1
    #     print(i)
    #     print(batch[0].shape)
    #     print(batch[1].shape)

    
