
import numpy as np
import cv2
from cv2 import VideoWriter, VideoWriter_fourcc, imread, resize
import h5py
import scipy.io
import os
import torch
from PIL import Image
from torchvision import models, transforms
from FeatureExtractor import FeatureExtractor
from cpd_auto import cpd_auto
from tqdm import tqdm, trange
import pandas as pd
import re
from utils import score_shot
'''
GLOBAL variables
'''
feature_extractor = FeatureExtractor()

#normalize, rescale and everything
preprocessor = transforms.Compose([
    transforms.Resize(224),
    # transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


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


def video_to_feature(video_path, bar_descrip='test', image_shape=(224, 224), position=0):
    '''
        sample T frame from the video
    '''
    video = cv2.VideoCapture(video_path)
    n_frame = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)

    features = None

    fail_frame_count = 0
    with trange(n_frame) as t:
        for i in t:
            t.set_description(bar_descrip)

            ret, frame = video.read()

            if ret:
                #image = Image.fromarray(frame).resize((image_shape), resample=Image.BILINEAR)
                image = preprocessor(Image.fromarray(frame)).unsqueeze(0)
            else:
                fail_frame_count += 1

            image_feature = feature_extractor(image).cpu().detach().numpy()

            if features is None:
                features = image_feature
            else:
                features = np.concatenate((features, image_feature), axis=1)

    video.release()
    print('fail frame count {}'.format(fail_frame_count))

    return features, features.shape[1], fps


def pick_features(features, fps):
    '''
        donw sample the feature vector
        feature: [c, n]
        the expected T of the output is a multiple of 32
    '''

    n_frame = features.shape[1]  # n_frame = fps * length
    n_channel = features.shape[0]

    n_sample = round((n_frame/fps) * 2)  # uniformally downsample to 2 fps
    n_sample -= n_sample % 32
    down_features = np.zeros((n_channel, n_sample), dtype=features.dtype)
    picks = [int(i * (n_frame / n_sample)) for i in range(n_sample)]

    for i in range(len(picks)):

        down_features[:, i] = features[:, picks[i]]

    return down_features, picks


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

    K = np.dot(features.T, features)  # K -> [N, N]
    ncp = int(int(n_frame//fps)/4)
    cps, cost = cpd_auto(K, ncp, 1)  # cps is a (n_cps,) np array

    # reform the cps as cps[i] = (cp_start, cp_end)
    cps = [0] + cps.tolist() + [n_frame]
    cps.sort()

    n_seg = len(cps)-1
    reform_cps = np.zeros((n_seg, 2), dtype=np.uint32)
    n_frame_per_seg = list()
    for i in range(n_seg):
        reform_cps[i, :] = np.array([cps[i], cps[i+1]-1])
        n_frame_per_seg.append(cps[i+1]-cps[i])

    return reform_cps, n_frame_per_seg


def get_features(video, C=1024, T=320):
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
        in_tensor = torch.Tensor(video[s:s+n_frame_per_run, :, :, :])
        out = feature_extractor(in_tensor).cpu().data.numpy()
        if features is None:
            features = out
        else:
            features = np.concatenate((features, out), axis=1)
        s += n_frame_per_run
    in_tensor = torch.Tensor(video[s:, :, :, :])
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

        user_scores = gt['user_score']  # shape(n_frame, n_user)

        user_score_rescale = feature_scaling(user_scores)

        gt_scores = gt['gt_score']

        # get video path
        video_path = os.path.join(vid_path, vid_name)

        # create h5 group
        vid_group = save_h5.create_group('video_{}'.format(counter))

        vid_group['video_name'] = np.string_(vid_name)

        # downsample the video and gts
        features, n_frame, fps = video_to_feature(
            video_path, 'summe {}'.format(vid_name))

        down_features, picks = pick_features(features, fps)

        #_test_samples(down_video, fps)

        vid_group['features'] = down_features
        vid_group['picks'] = np.array(picks)
        vid_group['fps'] = fps
        vid_group['n_frame'] = n_frame

        # _test_samples(samples)
        vid_group['gt_score'] = downsample_gt(gt_scores, picks)
        vid_group['user_score'] = downsample_gt(user_score_rescale, picks)

        # segment video using feature vector
        cps, n_frame_per_seg = segment_video(features, n_frame, fps)
        vid_group['change_points'] = cps
        vid_group['n_frame_per_seg'] = n_frame_per_seg

        counter += 1


def gen_tvsum():
    '''
        |-tvsum
        |   |-data
        |   |   |-ydata-tvsum50-anno.tsv 
        |   |   |-ydata-tvsum50-info.tsv
        |   |-matlab
        |   |   |-some stuff
        |   |-thumbnail
        |   |   |-thumbnails
        |   |-video
    '''

    # prepare for paths
    cur = os.getcwd()
    gen_path = os.path.join(cur, 'generated_data')
    save_path = os.path.join(gen_path, 'tvsum.h5')
    vid_path = os.path.join(cur, 'RawVideos/tvsum/video')
    gt_file = os.path.join(cur, 'RawVideos/tvsum/data/ydata-tvsum50-anno.tsv')

    # create generated_data dir
    if not os.path.exists(gen_path):
        os.mkdir(gen_path)

    # init save h5
    save_h5 = h5py.File(save_path, 'w')

    # get all videos
    all_files = os.listdir(vid_path)
    vid_names = [f for f in all_files if f.endswith('mp4')]

    counter = 0
    gt = pd.read_csv(gt_file, sep="\t")
    raw_scores = dict()

    for row in gt.itertuples():

        vid_name = row[1]
        anno = list(map(int, row[3].split(',')))

        if vid_name not in raw_scores:
            raw_scores[vid_name] = list()

        raw_scores[vid_name].append(anno)

    for k, v in raw_scores.items():
        raw_scores[k] = np.array(v, dtype=np.uint8).T

    for vid_name, user_scores in raw_scores.items():

        # create h5 group
        vid_group = save_h5.create_group('video_{}'.format(counter))
        vid_group['video_name'] = np.string_('{}.mp4'.format(vid_name))

        # get video path
        video_path = os.path.join(vid_path, '{}.mp4'.format(vid_name))

        user_score_rescale = feature_scaling(user_scores)
        gt_scores = np.mean(user_score_rescale, axis=1).reshape(
            user_score_rescale.shape[0], 1)

        # downsample the video and gts
        features, n_frame, fps = video_to_feature(
            video_path, 'tvsum {}'.format('{}.mp4'.format(vid_name)))

        down_features, picks = pick_features(features, fps)

        #_test_samples(down_video, fps)

        vid_group['features'] = down_features
        vid_group['picks'] = np.array(picks)
        vid_group['fps'] = fps
        vid_group['n_frame'] = n_frame

        # _test_samples(samples)
        vid_group['gt_score'] = downsample_gt(gt_scores, picks)
        vid_group['user_score'] = downsample_gt(user_score_rescale, picks)

        # segment video using feature vector
        cps, n_frame_per_seg = segment_video(features, n_frame, fps)
        vid_group['change_points'] = cps
        vid_group['n_frame_per_seg'] = n_frame_per_seg

        counter += 1


def gen_ovp():
    '''
        |-ovp
        |   |-database
        |   |   |-v1.mpg
        |   |-UserSummary 
        |   |   |-v1
        |   |   |   |-user1
        |   |   |   |   |-frame*.jpeg
    '''

    # prepare for paths
    cur = os.getcwd()
    gen_path = os.path.join(cur, 'generated_data')
    save_path = os.path.join(gen_path, 'ovp.h5')
    vid_path = os.path.join(cur, 'RawVideos/ovp/database')
    gt_path = os.path.join(cur, 'RawVideos/ovp/UserSummary')

    # create generated_data dir
    if not os.path.exists(gen_path):
        os.mkdir(gen_path)

    # init save h5
    save_h5 = h5py.File(save_path, 'w')

    # get all videos
    all_files = os.listdir(vid_path)
    vid_names = [f for f in all_files if f.endswith('mpg')]

    counter = 0
    for vid_name in vid_names:

        # find the gt dir
        gt_dir = os.path.join(gt_path, vid_name.split('.')[0])

        raw_gt = list()
        n_user = 0
        for user in os.listdir(gt_dir):
            
            gt_frame_dir = os.path.join(gt_dir, user)
            gt_frames = list()

            if not os.path.isdir(gt_frame_dir):
                    continue
            else:
                n_user += 1

            for gt_frame in os.listdir(gt_frame_dir):
                if gt_frame.endswith('jpeg'):
                    gt_frames = gt_frames + re.findall('\d+', gt_frame)
            raw_gt.append(list(map(int, gt_frames)))

        raw_gt = np.array(raw_gt)

        # get video path
        video_path = os.path.join(vid_path, vid_name)

        # create h5 group
        vid_group = save_h5.create_group('video_{}'.format(counter))

        vid_group['video_name'] = np.string_(vid_name)

        # downsample the video and gts
        features, n_frame, fps = video_to_feature(
            video_path, 'ovp {}'.format(vid_name))

        down_features, picks = pick_features(features, fps)
        # process the raw gt

        gt = np.zeros((n_frame, n_user), dtype=np.uint8)
        for i in range(n_user):
            gt[raw_gt[i], i] = 1
        
        user_score_rescale = gt

        # segment video using feature vector
        cps, n_frame_per_seg = segment_video(features, n_frame, fps)
        
        gt_scores = np.mean(user_score_rescale, axis=1).reshape(
            user_score_rescale.shape[0], 1)

        print(np.array(n_frame_per_seg))
        # score the segment with gt_scores
        seg_scores = score_shot(cps, gt_scores, None, np.array(n_frame_per_seg), rescale=False)
        
        
        #_test_samples(down_video, fps)

        vid_group['features'] = down_features
        vid_group['picks'] = np.array(picks)
        vid_group['fps'] = fps
        vid_group['n_frame'] = n_frame

        # _test_samples(samples)
        vid_group['gt_score'] = downsample_gt(gt_scores, picks)
        vid_group['user_score'] = downsample_gt(user_score_rescale, picks)

        vid_group['change_points'] = cps
        vid_group['n_frame_per_seg'] = n_frame_per_seg
        vid_group['seg_scores'] = seg_scores
        

        counter += 1


def gen_youtube():
    '''
        |-youtube
        |   |-database
        |   |   |-v1.mpg
        |   |-UserSummary 
        |   |   |-v1
        |   |   |   |-user1
        |   |   |   |   |-frame*.jpeg
    '''

    # prepare for paths
    cur = os.getcwd()
    gen_path = os.path.join(cur, 'generated_data')
    save_path = os.path.join(gen_path, 'youtube.h5')
    vid_path = os.path.join(cur, 'RawVideos/youtube/database')
    gt_path = os.path.join(cur, 'RawVideos/youtube/UserSummary')

    # create generated_data dir
    if not os.path.exists(gen_path):
        os.mkdir(gen_path)

    # init save h5
    save_h5 = h5py.File(save_path, 'w')

    # get all videos
    all_files = os.listdir(vid_path)
    vid_names = [f for f in all_files if (f.endswith('flv') or f.endswith('avi'))]

    counter = 0

    for vid_name in vid_names:
        
        # find the gt dir
        gt_dir = os.path.join(gt_path, vid_name.split('.')[0])

        try:
            raw_gt = list()
            n_user = 0
            for user in os.listdir(gt_dir):
                
                gt_frame_dir = os.path.join(gt_dir, user)
                gt_frames = list()
                
                if not os.path.isdir(gt_frame_dir):
                    continue
                else:
                    n_user += 1

                for gt_frame in os.listdir(gt_frame_dir):
                    
                    if gt_frame.endswith('jpeg') or gt_frame.endswith('jpg'):
                        gt_frames = gt_frames + re.findall('\d+', gt_frame)
                raw_gt.append(list(map(int, gt_frames)))

            raw_gt = np.array(raw_gt)

            # get video path
            video_path = os.path.join(vid_path, vid_name)
            
            # downsample the video and gts
            features, n_frame, fps = video_to_feature(
                video_path, 'youtube {}'.format(vid_name))

            down_features, picks = pick_features(features, fps)
            # process the raw gt

            gt = np.zeros((n_frame, n_user), dtype=np.uint8)
            
            for i in range(n_user):
                gt[raw_gt[i], i] = 1
            
            user_score_rescale = gt
            
            gt_scores = np.mean(user_score_rescale, axis=1).reshape(
                user_score_rescale.shape[0], 1)

        except:
            print("{} fails".format(vid_name))
        
        

        # create h5 group
        vid_group = save_h5.create_group('video_{}'.format(counter))

        vid_group['video_name'] = np.string_(vid_name)
        #_test_samples(down_video, fps)

        vid_group['features'] = down_features
        vid_group['picks'] = np.array(picks)
        vid_group['fps'] = fps
        vid_group['n_frame'] = n_frame

        # _test_samples(samples)
        vid_group['gt_score'] = downsample_gt(gt_scores, picks)
        vid_group['user_score'] = downsample_gt(user_score_rescale, picks)

        # segment video using feature vector
        cps, n_frame_per_seg = segment_video(features, n_frame, fps)
        vid_group['change_points'] = cps
        vid_group['n_frame_per_seg'] = n_frame_per_seg

        counter += 1



def add_seg_score_tvsum():
    '''
        add seg score for ovp and youtube
    '''
    cur = os.getcwd()
    gen_path = os.path.join(cur, 'generated_data')

    tvsum_save_path = os.path.join(gen_path, 'tvsum.h5')
    tvsum_gt_file = os.path.join(cur, 'RawVideos/tvsum/data/ydata-tvsum50-anno.tsv')

    tvsum_h5 = h5py.File(tvsum_save_path, 'a')

    gt = pd.read_csv(tvsum_gt_file, sep="\t")
    raw_scores = dict()

    for row in gt.itertuples():

        vid_name = row[1]
        anno = list(map(int, row[3].split(',')))

        if vid_name not in raw_scores:
            raw_scores[vid_name] = list()

        raw_scores[vid_name].append(anno)

    for k, v in raw_scores.items():
        
        vid_name = "{}.mp4".format(k)
        for key in tvsum_h5.keys():
            if tvsum_h5[key]['video_name'][()].decode("utf-8") == vid_name:
                user_score_rescale = feature_scaling(np.array(v, dtype=np.uint8).T)
                gt_scores = np.mean(user_score_rescale, axis=1).reshape(user_score_rescale.shape[0], 1)
                cps = tvsum_h5[key]['change_points'][()]
                n_frame_per_seg = tvsum_h5[key]['n_frame_per_seg'][()]
                seg_scores = score_shot(cps, gt_scores, None, np.array(n_frame_per_seg), rescale=True)
                tvsum_h5[key]['seg_scores'] = seg_scores

    

def add_seg_score_summe():
    cur = os.getcwd()
    gen_path = os.path.join(cur, 'generated_data')
    save_path = os.path.join(gen_path, 'summe.h5')
    vid_path = os.path.join(cur, 'RawVideos/summe/videos')
    gt_path = os.path.join(cur, 'RawVideos/summe/GT')

    # get all videos
    all_files = os.listdir(vid_path)
    vid_names = [f for f in all_files if f.endswith('mp4')]

    # init save h5
    summe_h5 = h5py.File(save_path, 'a')

    counter = 1
    for vid_name in vid_names:

        # get gt data
        gt = scipy.io.loadmat(os.path.join(
            gt_path, vid_name.replace('.mp4', '.mat')))

        gt_scores = gt['gt_score']

    
        for key in summe_h5.keys():
            if summe_h5[key]["video_name"][()].decode('utf-8') == vid_name:
                cps = summe_h5[key]['change_points'][()]
                n_frame_per_seg = summe_h5[key]['n_frame_per_seg'][()]
                seg_scores = score_shot(cps, gt_scores, None, np.array(n_frame_per_seg), rescale=True)
                summe_h5[key]["seg_scores"] = seg_scores


def add_seg_score_youtube():
    # prepare for paths
    cur = os.getcwd()
    gen_path = os.path.join(cur, 'generated_data')
    save_path = os.path.join(gen_path, 'youtube.h5')
    vid_path = os.path.join(cur, 'RawVideos/youtube/database')
    gt_path = os.path.join(cur, 'RawVideos/youtube/UserSummary')

    youtube_h5 = h5py.File(save_path, 'a')

    for key in youtube_h5.keys():
        vid_name = youtube_h5[key]['video_name'][()].decode('utf-8')
         # find the gt dir
        gt_dir = os.path.join(gt_path, vid_name.split('.')[0])
        raw_gt = list()
        n_user = 0
        for user in os.listdir(gt_dir):
            
            gt_frame_dir = os.path.join(gt_dir, user)
            gt_frames = list()
            
            if not os.path.isdir(gt_frame_dir):
                continue
            else:
                n_user += 1

            for gt_frame in os.listdir(gt_frame_dir):
                
                if gt_frame.endswith('jpeg') or gt_frame.endswith('jpg'):
                    gt_frames = gt_frames + re.findall('\d+', gt_frame)
            raw_gt.append(list(map(int, gt_frames)))

        raw_gt = np.array(raw_gt)

        n_frame = youtube_h5[key]['n_frame'][()]
        gt = np.zeros((n_frame, n_user), dtype=np.uint8)
            
        for i in range(n_user):
            gt[raw_gt[i], i] = 1
        
        
        user_score_rescale = gt

        gt_scores = np.mean(user_score_rescale, axis=1).reshape(
                user_score_rescale.shape[0], 1)
    
        cps = youtube_h5[key]['change_points'][()]
        
        n_frame_per_seg = youtube_h5[key]['n_frame_per_seg'][()]
        seg_scores = score_shot(cps, gt_scores, None, np.array(n_frame_per_seg), rescale=False)
        
        youtube_h5[key]['seg_scores'] = seg_scores


            

                




if __name__ == "__main__":
    # gen_summe()
    # gen_tvsum()
    # gen_ovp()
    # gen_youtube()
    #add_seg_score_summe()
    add_seg_score_youtube()
