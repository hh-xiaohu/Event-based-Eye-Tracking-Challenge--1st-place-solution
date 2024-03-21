import h5py
import os
import numpy as np
import pdb
import sys
import torch
import imageio
import cv2
import pandas as pd


def get_target(filename):
    with open(filename, "r") as f:
        # target is at the frequency of 100 Hz. It will be downsampled to 20 Hz in the target transformation
        target = np.array(
            [list(map(float, line.strip('()\n').split(', '))) for line in f.readlines()], np.float32)
    return target


def generate_clip(events, ordering='txyp', clip_size=(16, 2, 128, 128), split_by='time'):
    '''
    Generaete clip from events.
    parameter:
        events: ndarray of shape [num, channel].
        ordering: ordering of the events tuple inside of events.
        clip_size: the size of generated clip [time, channel, x, y].
            channel of output clip: 2:only frequency add;
                                    3:contain average time (no polarity);
                                    4:contain average time (polarity)
        split_by: 'time' or 'number' decide how to split the events into even bins.
    Returns:
        (TxHxWxC)(clip_size), numpy array of n rate-coded frames with channels p
    '''

    x_index = ordering.find("x")
    y_index = ordering.find("y")
    t_index = ordering.find("t")
    p_index = ordering.find("p")

    t, x, y, p = np.split(events[:, (t_index, x_index, y_index, p_index)], 4, axis=1)
    T, C, H, W = clip_size

    if split_by == 'time':
        split_weight = t * 0.99 * T
    elif split_by == 'number':
        split_weight = np.arange(0, 1, 1 / events.shape[0]) * T

    x = x.astype(np.uint32)
    y = y.astype(np.uint32)
    p = p.astype(np.bool8)
    split_index = split_weight.astype(np.uint32)

    clip_cnt_pos = np.zeros((T * H * W,), dtype=np.float32)
    clip_cnt_neg = np.zeros((T * H * W,), dtype=np.float32)
    np.add.at(clip_cnt_pos, x[p] + W * y[p] + H * W * split_index[p], 1)
    np.add.at(clip_cnt_neg, x[~p] + W * y[~p] + H * W * split_index[~p], 1)
    clip = [clip_cnt_pos, clip_cnt_neg]

    if C == 3:
        clip_time_no_pol = np.zeros((T * H * W,), dtype=np.float32)
        p = p.astype(np.int32)
        p[p == 0] = -1
        weight = p * (split_weight - split_index)
        np.add.at(clip_time_no_pol, x + W * y + H * W * split_index, weight)
        clip.append(clip_time_no_pol)
    elif C == 4:
        clip_time_pos = np.zeros((T * H * W,), dtype=np.float32)
        clip_time_neg = np.zeros((T * H * W,), dtype=np.float32)
        weight = split_weight - split_index
        np.add.at(clip_time_pos, x[p] + W * y[p] + H * W * split_index[p], weight[p])
        np.add.at(clip_time_neg, x[~p] + W * y[~p] + H * W * split_index[~p], weight[~p])
        clip.append(clip_time_pos)
        clip.append(clip_time_neg)

    clip = np.stack(clip, -1).reshape((T, H, W, C))
    #
    # clip = np.divide(clip,
    #                  np.amax(clip, axis=(1, 2), keepdims=True),
    #                  out=np.zeros_like(clip),
    #                  where=clip != 0)
    return clip

def event_to_image(events, repeat=4, sum_bins=2, amp=1, show_int=True):
    '''
    Visualization events
    :param events: a tensor with shape (B, 1, T, H, W)
    :return: color images with shape (B, T//sum_bins, 3, repeat*H, repeat*W)
        if show_int=True:
                R   G   B   events_value  |  R   G   B   events_value
                255 180 180    +1         |  180 180 255    -1
                255 160 160    +2         |  160 160 255    -2
                255 130 130    +3         |  130 130 255    -3
                255 100 100    +4         |  100 100 255    -4
                255 60  60     +5         |  60  60  255    -5
                255 0   0      >= +6      |  0   0   255    <= -6
        else:
            GB(R) = 255 - int(90 * value) if abs(value) < 1 else 255 - min(int(57 + 33 * value), 255)

    '''
    if show_int:
        events = torch.round(events * amp)
    else:
        events = events
    B, T, H, W = events.shape
    assert T % sum_bins == 0, 'T % sum_bins should be an integer'
    # events = events[:, 8:8+sum_bins].sum(1)
    events = torch.reshape(events, (B, T // sum_bins, sum_bins, H, W)).sum(2)
    events = torch.reshape(events, (B * T // sum_bins, H, W))
    image = torch.zeros((B * T // sum_bins, 3, H, W)).cuda() + 255

    if show_int:
        color_list = [165, 132, 99, 66, 33, 0]
        # color_list = [0] * 6
        # process positive events
        for i in range(1, len(color_list)+1):
            index = (events == i) if i != len(color_list) else (events >= i)
            image[:, 1][index] = color_list[i-1]
            image[:, 2][index] = color_list[i-1]
        # process negtive events
        for i in range(1, len(color_list)+1):
            index = (events == -i) if i != len(color_list) else (events <= -i)
            image[:, 0][index] = color_list[i-1]
            image[:, 1][index] = color_list[i-1]
    else:
        raise ValueError

    image = torch.clamp(image, 0, 255).byte()
    image = torch.from_numpy(image.cpu().numpy().repeat(repeat, 2).repeat(repeat, 3))
    return image


def write_array_to_txt(filename, array):
    with open(filename, 'w') as file:
        for item in array:
            file.write(f"({', '.join(map(str, item))})\n")

def vis_xytp_array(events, height, width, bins, repeat=1):
    """
        events: N * 4 
        return: T*C*H*W
    """
    t, x, y, p = events[:, 0], events[:, 1], events[:, 2], events[:, 3]
    t = t - t[0]
    t = (t / (t[-1] - t[0])).astype(float)
    events = np.stack([t, x, y, p], axis=-1).astype(float)
    events_voxel = generate_clip(events, clip_size=(bins, 2, height, width))
    events_voxel = torch.FloatTensor(events_voxel).permute(3, 0, 1, 2)
    events_voxel = events_voxel[0, :, :, :] - events_voxel[1, :, :, :]
    image_output = event_to_image(events_voxel.unsqueeze(0), repeat=repeat, sum_bins=1, amp=1)
    image = image_output.permute(0, 2, 3, 1).detach().cpu().numpy()[:, :, :, ::-1].astype(np.uint8)
    return image

if __name__ == "__main__":
    dtype = np.dtype([("t", int), ("x", int), ("y", int), ("p", int)])
    fps = 5
    data_dir = "/xxx/event_data/test"
    submission_file = "./log/submission.csv"
    df = pd.read_csv(submission_file)

    # pdb.set_trace()
    nums = 0
    dirs = ['1_1', '2_2', '3_1', '4_2', '5_2', '6_4', '7_5', '8_2', '8_3', '10_2', '11_3', '12_4']
    for seq in dirs:
        # pdb.set_trace()
        file_name = os.path.join(data_dir, seq, "label_zeros.txt")
        target = get_target(file_name)
        rows = target[::5]
        num = rows.shape[0]
        rows = df.iloc[nums:nums+num, 1:3]
        rows = np.pad(rows, ((0, 0), (0, 1)), mode='constant', constant_values=0)
        rows[:, 0] = rows[:, 0] * 8
        rows[:, 1] = rows[:, 1] * 8
        target[::5] = rows
        write_array_to_txt(file_name.replace("label_zeros.txt", "label_pred.txt"), target)
        # os.rename(file_name, file_name.replace("label_.txt", "label.txt"))
        nums += num

    os.makedirs(f"./vis_test_{fps}", exist_ok=True)
    for seq in os.listdir(data_dir):
        file_name = os.path.join(data_dir, seq, f"{seq}.h5")
        with h5py.File(file_name, "r") as f:
            events = f["events"][:].astype(dtype)
        gt = get_target(file_name.replace(f"{seq}.h5", "label_pred.txt"))

        gt = gt[::5]

        events = np.frombuffer(events, dtype=dtype)
        events = events.view('i8').reshape(-1, 4)
        voxels = vis_xytp_array(events, 480, 640, gt.shape[0])
        
        voxels_ = voxels.copy()
        for i in range(gt.shape[0]):
            cv2.circle(voxels_[i], (int(gt[i][0]) , int(gt[i][1])), 8, (0, 255, 0), -1)
            cv2.circle(voxels_[i], (int(gt[i][0]) , int(gt[i][1])), 80, (0, 255, 0), 2)
            # cv2.imwrite(f'./vis_test/{seq}_{i}.jpg', voxels_[i])
            # pdb.set_trace()
        imageio.mimsave(f'./vis_test_{fps}/{seq}.gif', voxels_, duration=1000/fps)