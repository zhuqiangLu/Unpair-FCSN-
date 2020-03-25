
import numpy as np
import cv2
from cv2 import VideoWriter, VideoWriter_fourcc, imread, resize
import h5py
import scipy.io
import os
import torch
from FeatureExtractor import FeatureExtractor
from cpd_auto import cpd_auto
from tqdm import tqdm, trange

'''
GLOBAL variables
'''
feature_extractor = FeatureExtractor()

def _test_samples(samples, fps):
    '''
        write the sample in a form of video
    '''
    print('start restoring sample')
    writer = cv2.VideoWriter(
        './test.mp4', VideoWriter_fourcc(*'MP4V'), fps, (224, 224))
    for i in range(samples.shape[0]):
        # reshape the image
        frame = np.zeros((224, 224, 3), dtype=np.uint8)
        for j in range(3):
            frame[:, :, j] = samples[i, j, :, :]
        writer.write(frame)

    writer.release()


def downsample_video(video_path, n_sample, bar_descrip='test',image_shape=(224, 224)):
    '''
        sample T frame from the video
    '''
    video = cv2.VideoCapture(video_path)
    n_frame = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)


    # sample shape is [n, c, d,d]
    down_video = np.zeros(
        (n_frame, 3, image_shape[0], image_shape[1]), dtype=np.uint8)

    # record the selected frames to select gt
    picks = [int(i * (n_frame / n_sample)) for i in range(n_sample)]
    
    with trange(n_frame) as t:
        for i in t:
            target_frame = int(i * (n_frame / n_sample))

            t.set_description(bar_descrip)

            _,frame = video.read()

            # resize
            # frame shape is [d,d,c]
            frame = resize(frame, image_shape)

            # reshape to [c, d,d] and stack to sample
            for j in range(frame.shape[-1]):
                down_video[i, j, :, :] = frame[:, :, j]

    video.release()

    return down_video, picks, n_frame, fps

def down_features(features, picks):
    '''
        donw sample the feature vector
        feature: [c, n]
    '''
    n_sample = len(picks)
    n_channel = features.shape[0]
    down_features = np.zeros((n_channel, n_sample), dtype=features.dtype)
    for i in range(n_sample):
       
        down_features[:,i] = features[:,picks[i]]
    
    return down_features



def feature_scaling(arr):
    '''
        the expected shape of arr is [n_frame, n_users]
        rescale the importance score in (0, 1)
    '''
    rescale_arr = np.zeros(arr.shape, dtype=np.float32)
    for i in range(arr.shape[1]):

        min_score = min(arr[:, i])
        max_score = max(arr[:, i])
        epsilon = 1e-5
        rescale_arr[:, i] = (arr[:, i]-min_score) / \
            (max_score-min_score + epsilon)
    return rescale_arr


def downsample_gt(gt, indeces):
    '''
        downsample the gt according to index
        gt : (n_frame, n_users)
        indeces: list()
    '''

    n_frame = len(indeces)
    n_user = gt.shape[1]
    down_gt = np.zeros((n_frame, n_user))

    for i in range(n_user):
        for j in range(n_frame):
            down_gt[j, i] = gt[indeces[j], i]
    return down_gt

def segment_video(features, n_frame, fps):

    K = np.dot(features.T, features) # K -> [N, N]
    ncp = int(int(n_frame//fps))
    print(ncp)
    ncp =30
    cps,cost = cpd_auto(K, ncp, 1) # cps is a (n_cps,) np array
    print(cps)
    
    #reform the cps as cps[i] = (cp_start, cp_end)
    cps = [0]+ cps.tolist() + [n_frame]
    n_seg = len(cps)-1
    reform_cps = np.zeros((n_seg, 2), dtype=np.uint8)
    n_frame_per_seg = list()
    for i in range(n_seg):
        reform_cps[i,:] = np.array([cps[i], cps[i+1]-1])
        n_frame_per_seg.append(cps[i+1]-cps[i])

    return reform_cps, n_frame_per_seg


def get_features(video,C=1024, T=320):
    '''
        to handle the case the exceeding max gpu memory
        video: [n, c, d,d]
    '''
    n_frame = video.shape[0]
    n_frame_per_run = 1000
    n_iter = n_frame//n_frame_per_run
    s = 0
    features = None
    for i in range(n_iter):
        in_tensor = torch.Tensor(video[s:s+n_frame_per_run,:,:,:])
        out = feature_extractor(in_tensor).cpu().data.numpy()
        if features is None:
            features = out
        else:
            features = np.concatenate((features, out), axis=1)
        s+=n_frame_per_run
    in_tensor = torch.Tensor(video[s:,:,:,:])
    out = feature_extractor(in_tensor).cpu().data
    features = np.concatenate((features, out), axis=1)
    return features


def gen_summe(T=320):
    '''
        assume exists
            ./generated_data 
            ./RawVideos/summe/videos
            ./RawVideos/summe/GT
    '''
    # prepare for paths
    cur = os.getcwd()
    gen_path = os.path.join(cur, 'generated_data')
    save_path = os.path.join(gen_path, 'summe.h5')
    vid_path = os.path.join(cur, 'RawVideos/summe/videos')
    gt_path = os.path.join(cur, 'RawVideos/summe/GT')

    # create generated_data dir
    if not os.path.exists(gen_path):
        os.mkdir(gen_path)

    # init save h5
    save_h5 = h5py.File(save_path, 'w')

    # get all videos
    all_files = os.listdir(vid_path)
    vid_names = [f for f in all_files if f.endswith('mp4')]

    counter = 1
    for vid_name in vid_names:
        # get gt data
        gt = scipy.io.loadmat(os.path.join(
            gt_path, vid_name.replace('.mp4', '.mat')))

        # gt_scores = gt['gt_score']  # shape (N, 1)

        user_scores = gt['user_score']  # shape(n_frame, n_user)

        user_score_rescale = feature_scaling(user_scores)

        gt_scores = gt['gt_score']

        # get video path
        video_path = os.path.join(vid_path, vid_name)

        # create h5 group
        vid_group = save_h5.create_group('video_{}'.format(counter))

        vid_group['video_name'] = np.string_(vid_name)
        
        # downsample the video and gts
        down_video, picks, n_frame, fps = downsample_video(video_path, T, 'summe {}'.format(vid_name))
        #_test_samples(down_video, fps)
        
        vid_group['picks'] = np.array(picks)
        vid_group['fps'] = fps
        vid_group['n_frame'] = n_frame

        # _test_samples(samples)
        vid_group['gt_score'] = downsample_gt(gt_scores, picks)
        vid_group['user_score'] = downsample_gt(user_score_rescale, picks)
        
        # extract feature
        #features = feature_extractor(torch.Tensor(down_video)).cpu().data #[C, N]
        features  = get_features(down_video)
        

        vid_group['features'] = down_features(features, picks)
        
        # segment video using feature vector
        cps, n_frame_per_seg = segment_video(features, n_frame, fps)
        vid_group['change_points'] =cps
        vid_group['n_frame_per_seg'] = n_frame_per_seg
        
        # d
        break
        counter += 1


if __name__ == "__main__":

    # save_path = os.path.join(os.getcwd(), 'generated_data')

    # dataset_name = 'test'
    # save_h5 = h5py.File('{}.h5'.format(dataset_name), 'w')

    # video_dir = './RawVideos/summe/videos/'
    # video_type = 'mp4'
    # all_files = os.listdir(video_dir)
    # fnames = [f for f in all_files if f.endswith(video_type)]
    # video_path = os.path.join(video_dir, fnames[0])
    # print(video_path)
    gen_summe()
    #print(get_features(np.zeros((4473, 3, 224,224))).shape)#should be (1024,4473)

    # test for downsample
    # a = np.random.randint(1, 10, (10, 2))
    # b = [1, 3, 9]
    # print(a, b)
    # print(downsample_gt(a, b))
