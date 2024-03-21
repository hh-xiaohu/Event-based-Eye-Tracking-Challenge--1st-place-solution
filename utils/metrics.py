import torch
import torch.nn as nn
import numpy as np
import pdb


import numpy as np
import pdb

def process_array(arr, targets):
    arr = arr.numpy().reshape(-1, 3)
    targets = targets.numpy().reshape(-1, 3)

    zero_indices = np.where(arr[:, 2] == 0)[0]
    sequences1 = np.array_split(arr, zero_indices[1:], axis=0)
    sequences2 = np.array_split(targets, zero_indices[1:], axis=0)
    
    new_arrays = []
    new_targets = []
    for index in range(len(sequences1)):
        unique_vals = np.unique(sequences1[index][:, 2])
        new_arr = []
        new_target = []
        for val in unique_vals:
            mask = sequences1[index][:, 2] == val
            mean_values = np.mean(sequences1[index][mask, :2], axis=0)
            new_arr.append(np.concatenate((mean_values, [val])))
            # pdb.set_trace()
            new_target.append(np.concatenate((np.mean(sequences2[index][mask, :3], axis=0), [val])))
        new_arr = np.array(new_arr)
        new_target = np.array(new_target)
        # new_arr[:, 2] = [new_val_map[val] for val in new_arr[:, 2]]
        new_arrays.append(new_arr)
        new_targets.append(new_target)
    return new_arrays, new_targets




def p_acc(target, prediction, width_scale, height_scale, pixel_tolerances=[1,3,5,10]):
    """
    Calculate the accuracy of prediction
    :param target: (N, seq_len, 2) tensor, seq_len could be 1
    :param prediction: (N, seq_len, 2) tensor
    :return: a dictionary of p-total correct and batch size of this batch
    """
    # flatten the N and seqlen dimension of target and prediction
    target = target.reshape(-1, 2)
    prediction = prediction.reshape(-1, 2)

    dis = target - prediction
    dis[:, 0] *= width_scale
    dis[:, 1] *= height_scale
    dist = torch.norm(dis, dim=-1)

    total_correct = {}
    for p_tolerance in pixel_tolerances:
        total_correct[f'p{p_tolerance}'] = torch.sum(dist < p_tolerance)

    bs_times_seqlen = target.shape[0]
    return total_correct, bs_times_seqlen


def p_acc_wo_closed_eye(target, prediction, width_scale, height_scale, pixel_tolerances=[1,3,5,10]):
    """
    Calculate the accuracy of prediction, with p tolerance and only calculated on those with fully opened eyes
    :param target: (N, seqlen, 3) tensor
    :param prediction: (N, seqlen, 2) tensor, the last dimension is whether the eye is closed
    :return: a dictionary of p-total correct and batch size of this batch
    """
    # flatten the N and seqlen dimension of target and prediction
    target = target.reshape(-1, 3)
    prediction = prediction.reshape(-1, 2)

    dis = target[:,:2] - prediction
    dis[:, 0] *= width_scale
    dis[:, 1] *= height_scale
    dist = torch.norm(dis, dim=-1)
    # check if there is nan in dist
    assert torch.sum(torch.isnan(dist)) == 0

    eye_closed = target[:,2] # 1 is closed eye
    # get the total number frames of those with fully opened eyes
    total_open_eye_frames = torch.sum(eye_closed == 0)

    # get the indices of those with closed eyes
    eye_closed_idx = torch.where(eye_closed == 1)[0]
    dist[eye_closed_idx] = np.inf
    total_correct = {}
    for p_tolerance in pixel_tolerances:
        total_correct[f'p{p_tolerance}'] = torch.sum(dist < p_tolerance)
        assert total_correct[f'p{p_tolerance}'] <= total_open_eye_frames

    return total_correct, total_open_eye_frames.item()


def px_euclidean_dist(target, prediction, width_scale, height_scale):
    """
    Calculate the total pixel euclidean distance between target and prediction
    in a batch over the sequence length
    :param target: (N, seqlen, 3) tensor
    :param prediction: (N, seqlen, 2) tensor
    :return: a dictionary of p-total correct and batch size of this batch
    """
    # flatten the N and seqlen dimension of target and prediction
    target = target.reshape(-1, 3)[:, :2]
    prediction = prediction.reshape(-1, 2)

    dis = target - prediction
    dis[:, 0] *= width_scale
    dis[:, 1] *= height_scale
    dist = torch.norm(dis, dim=-1)

    total_px_euclidean_dist = torch.sum(dist)
    sample_numbers = target.shape[0]
    return total_px_euclidean_dist, sample_numbers


# class weighted_MSELoss(nn.Module):
#     def __init__(self, weights, reduction='mean'):
#         super().__init__()
#         self.reduction = reduction
#         self.weights = weights
#         self.mseloss = nn.MSELoss(reduction='none')
        
#     def forward(self, inputs, targets):
#         batch_loss = self.mseloss(inputs, targets) * self.weights
#         if self.reduction == 'mean':
#             return torch.mean(batch_loss)
#         elif self.reduction == 'sum':
#             return torch.sum(batch_loss)
#         else:
#             return batch_loss

class weighted_MSELoss(nn.Module):
    def __init__(self, weights, reduction='mean'):
        super().__init__()
        
    def forward(self, inputs, targets):
        loss = ((inputs - targets) ** 2).sum(dim=2).sqrt().mean()
        # loss += 1e-4 * ((inputs[:-1] - inputs[1:]) ** 2).sum(dim=2).sqrt().mean()
        # pdb.set_trace()
        # mask = ((targets[:, :-1] - targets[:, 1:]) ** 2).sum(dim=2).sqrt() < 0.05 
        # loss += 1e-3 * (((inputs[:, :-1] - inputs[:, 1:]) ** 2).sum(dim=2).sqrt()[mask].mean())
        # loss += ((inputs[:, 1:] - inputs[:, :-1]) ** 2).sum(dim=2).sqrt().mean()
        
        return loss