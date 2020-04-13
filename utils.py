import numpy as np


def f_score(gt_seg_idx, pred_seg_idx, n_frame_per_seg, epsilon=1e-3):
    '''
        gt_seg_idx: the idx of the gt segment, (n_gt_idx, )
        pred_seg_idx: the idx of the pred segment, (n_pred_idx, )
        n_frame_per_seg: (n_cp)
    '''

    pred_length = 0
    gt_length = 0
    n_cp = n_frame_per_seg.shape[0]

    for gt_idx in gt_seg_idx:
        gt_length += n_frame_per_seg[gt_idx]

    for pred_idx in pred_seg_idx:
        pred_length += n_frame_per_seg[pred_idx]

    overlap_idx = list()
    l_pred_seg_idx = list(pred_seg_idx)
    l_gt_seg_idx = list(gt_seg_idx)
    # find overlap
    for pred_idx in l_pred_seg_idx:
        if pred_idx in l_gt_seg_idx:
            overlap_idx.append(pred_idx)

    overlap = 0
    for idx in overlap_idx:
        overlap += n_frame_per_seg[idx]

    P = overlap / pred_length
    R = overlap / gt_length

    F = (2*P*R)/(P+R+epsilon)
    return F


def score_shot(cps, frame_scores, picks, n_frame_per_seg, rescale=True):
    '''
        cps: change points, expected shape (n_cp, 2)
        frame_scores: the scores of the each frames, expect shape (n_selected_frame, 1)
        picks: the idx of the selected frame in the original video, expected shape (n_selected_frame,)
        n_frame_per_seg: the number of frame per segment, expected shape (n_seg, )
    '''

    # flatten
    frame_scorse = frame_scores.reshape(-1)

    n_cp = cps.shape[0]
    seg_scores = np.zeros(n_cp, dtype=np.float32)

    n_selected_frame = frame_scores.shape[0]

    for i in range(n_selected_frame):
        # if picks is none, then the n_select_frame = n_frame
        if picks is None:
            idx = i
        else:
            idx = picks[i]
        score = frame_scores[i]

        for j in range(n_cp):
            cp = cps[j]
            if idx in range(cp[0], cp[1]+1):
                seg_scores[j] += score

    if rescale:
        n_seg = n_frame_per_seg.shape[0]
        for i in range(n_seg):
            seg_scores[i] /= n_frame_per_seg[i]
    return seg_scores


def knapsack(pred_seg_scores, n_frames_per_seg, n_selected_frames):
    '''
        pred_seg_scores: the score of each segment, type: list
        n_frames_per_seg: the number of frame that each segment contains, type: list
        n_selected_frames: the number of frame needs to be selected from the original video, type:int

    '''

    n_seg = len(pred_seg_scores)
    #_selected = np.zeros((n_seg+1, n_selected_frames+1))
    selected = [[[] for j in range(n_selected_frames+1)]
                for i in range(n_seg+1)]

    scores = np.zeros((n_seg+1, n_selected_frames+1))

    for i in range(n_seg+1):
        for j in range(n_selected_frames+1):

            if i == 0 or j == 0:
                continue
            elif (j >= n_frames_per_seg[i-1]):
                score_include = pred_seg_scores[i-1] + \
                    scores[i-1][j-n_frames_per_seg[i-1]]
                score_exclude = scores[i-1][j]
                if score_include > score_exclude:
                    scores[i][j] = score_include
                    selected[i][j] = selected[i-1][j-n_frames_per_seg[i-1]]+[i]
                else:
                    scores[i][j] = score_exclude
                    selected[i][j] = selected[i-1][j]

            else:
                scores[i][j] = scores[i-1][j]  # i-th segment is too large
                selected[i][j] = selected[i-1][j]

    # pick the one with highest score by rows
    indeces = np.argmax(scores, axis=1)

    best = None
    best_score = 0
    i = 0
    # then find the highest of each row
    for index in indeces:

        if best_score < scores[i][index]:

            best = selected[i][index]

        i += 1

    best = [i-1 for i in best]  # fix the index
    ''' 
        besdt is a list, containing the index of the selected item
    '''

    return np.array(best)


'''
the following methods are redundent
'''


def get_keyshot(video_info, pred_scores):
    n_frames = video_info['n_frames']
    cps = video_info['change_points']
    n_frames_per_seg = video_info['n_frame_per_seg']
    pred_scores = np.array(pred_scores.cpu().data[0])  # shape [S, T]
    # run argmax to get scores
    pred_scores = np.argmax(pred_scores, axis=0)  # shape [1, T]
    pred_scores = upsample(pred_scores, n_frames)  # shape [1, n_frames]

    # prepare for the knapsack
    pred_seg_scores = np.array(
        [pred_scores[:, cp[0]:cp[1]+1].mean() for cp in cps])

    selected = knapsack(pred_seg_scores, n_frames_per_seg,
                        int(n_frames * 0.15))

    # label
    pred_label = np.zeros(n_frames)
    for i in selected:
        pred_label[cps[i][0]:cps[i][1]+1] = 1
    '''
        pred_label -> a list
        selected -> a list
    '''
    return pred_label, selected


def upsample(pred_scores, n_frames):
    ''' the expected shape of pred_scores is [c, t]'''
    n_scores = pred_scores.shape[0]
    up_scores = np.zeros((n_scores, n_frames))
    length = pred_scores.shape[-1]
    ratio = n_frames // length
    reminder = n_frames % length

    # try dynamic upsampling
    l = 0
    for i in range(length):
        if reminder > 0:
            up_scores[:, l:l + ratio+1]\
                = np.ones((n_scores, ratio+1), dtype=int) * pred_scores[:, i]
            reminder -= 1
            l += ratio+1
        else:
            up_scores[:, l:l + ratio]\
                = np.ones((n_scores, ratio), dtype=int) * pred_scores[:, i]
            l += ratio
    return up_scores


if __name__ == '__main__':
    # pred_seg_scores = np.array([1.20, 1.00, 6.0])
    # n_frames_per_seg = np.array([3, 2, 1])
    # n_selected_frames = 4
    # best = (knapsack(pred_seg_scores, n_frames_per_seg, n_selected_frames))
    # print(best.shape)

    # cps = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    # frame_scores = np.array([2, 3, 1, 0])
    # picks = np.array([2, 4, 6, 8])
    # n_frame_per_seg = np.array([2, 2, 2, 2])
    # seg_scores = score_shot(cps, frame_scores, picks, n_frame_per_seg,)
    # print(seg_scores.shape)

    gt_seg_idx = np.array([1])
    pred_seg_idx = np.array([0, 2, 3])
    n_frame_per_seg = np.array([2, 3, 4, 3])
    f = f_score(gt_seg_idx, pred_seg_idx, n_frame_per_seg)
    print(f)
