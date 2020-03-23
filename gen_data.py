
import numpy as np
import cv2
from cv2 import VideoWriter, VideoWriter_fourcc, imread, resize
import h5py
import scipy.io
import os

from FeatureExtractor import FeatureExtractor
from cpd_auto import cpd_auto


def _test_samples(samples):
    '''
        write the sample in a form of video
    '''
    print('start restoring sample')
    writer = cv2.VideoWriter(
        './test.mp4', VideoWriter_fourcc(*'MP4V'), 10, (224, 224))
    for i in range(samples.shape[0]):
        # reshape the image
        frame = np.zeros((224, 224, 3), dtype=np.uint8)
        for j in range(3):
            frame[:, :, j] = samples[i, j, :, :]
        writer.write(frame)

    writer.release()


def downsample_video(video_path, n_sample, image_shape=(224, 224)):
    '''
        sample T frame from the video
    '''
    video = cv2.VideoCapture(video_path)
    n_frame = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # sample shape is [n, c, d,d]
    samples = np.zeros(
        (n_sample, 3, image_shape[0], image_shape[1]), dtype=np.uint8)

    # record the selected frames to select gt
    selected = list()
    for i in range(n_sample):
        target_frame = int(i * (n_frame / n_sample))

        selected.append(target_frame)

        # jump to the desire frame and read
        video.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        _, frame = video.read()

        # resize
        # frame shape is [d,d,c]
        frame = resize(frame, image_shape)

        # reshape to [c, d,d] and stack to sample
        for j in range(frame.shape[-1]):
            samples[i, j, :, :] = frame[:, :, j]

    video.release()
    return samples, selected


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

    n_frames = len(indeces)
    n_users = gt.shape[1]
    down_gt = np.zeros((n_frames, n_users))

    for i in range(n_users):
        for j in range(n_frames):
            down_gt[j, i] = gt[indeces[j], i]
    return down_gt


def gen_summe():
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
        samples, indeces = downsample_video(video_path, 320)

        # _test_samples(samples)
        vid_group['gt_score'] = downsample_gt(gt_scores, indeces)
        vid_group['user_score'] = downsample_gt(user_score_rescale, indeces)

        # extract feature
        vid_group['features'] = FeatureExtractor(samples)

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

    # test for downsample
    # a = np.random.randint(1, 10, (10, 2))
    # b = [1, 3, 9]
    # print(a, b)
    # print(downsample_gt(a, b))
