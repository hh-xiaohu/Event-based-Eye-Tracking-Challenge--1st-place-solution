import torch
import os
from utils.metrics import p_acc, p_acc_wo_closed_eye, px_euclidean_dist, process_array
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import pdb
import numpy as np
def train_epoch(model, train_loader, criterion, optimizer, lr_scheduler, epoch, args):
    model.train()
    total_loss = 0.0
    total_p_corr_all = {f'p{p}_all':0 for p in args.pixel_tolerances}
    total_p_error_all  = {f'error_all':0}  # averaged euclidean distance
    total_samples_all, total_sample_p_error_all  = 0, 0

    iters = len(train_loader)
    for i, (inputs, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs.to(args.device))
        #taking only the last frame's label, and first two dim are coordinate, last is open or close so discarded
        targets = targets.to(args.device)
        loss = criterion(outputs, targets[:,:, :2]) 

        loss.backward()
        optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step(epoch + i / iters)
        total_loss += loss.item()

        # calculate pixel tolerated accuracy
        p_corr, batch_size = p_acc(targets[:, :, :2], outputs[:, :, :], \
                                width_scale=args.sensor_width*args.spatial_factor, \
                                height_scale=args.sensor_height*args.spatial_factor, \
                                    pixel_tolerances=args.pixel_tolerances)
        total_p_corr_all = {f'p{k}_all': (total_p_corr_all[f'p{k}_all'] + p_corr[f'p{k}']).item() for k in args.pixel_tolerances}
        total_samples_all += batch_size

        # calculate averaged euclidean distance
        p_error_total, bs_times_seqlen = px_euclidean_dist(targets[:, :, :3], outputs[:, :, :], \
                                width_scale=args.sensor_width*args.spatial_factor, \
                                height_scale=args.sensor_height*args.spatial_factor)
        total_p_error_all = {f'error_all': (total_p_error_all[f'error_all'] + p_error_total).item()}
        total_sample_p_error_all += bs_times_seqlen
    
    metrics = {'tr_p_acc_all': {f'tr_p{k}_acc_all': (total_p_corr_all[f'p{k}_all']/total_samples_all) for k in args.pixel_tolerances},
               'tr_p_error_all': {f'tr_p_error_all': (total_p_error_all[f'error_all']/total_sample_p_error_all)}}
    
    return model, total_loss / len(train_loader), metrics


def validate_epoch(model, val_loader, criterion, args):
    model.eval()
    total_loss = 0.0
    total_p_corr_all = {f'p{p}_all':0 for p in args.pixel_tolerances}
    total_p_error_all  = {f'error_all':0}
    total_samples_all, total_sample_p_error_all  = 0, 0
    outputs_list = []
    targets_list = []
    with torch.no_grad():
        for inputs, targets in val_loader:
            outputs = model(inputs.to(args.device))
            targets = targets.to(args.device)
            try:
                loss = criterion(outputs, targets[:,:, :2]) 
            except:
                pdb.set_trace()
            total_loss += loss.item()
            outputs_list.append(outputs.detach().cpu())
            targets_list.append(targets.detach().cpu())
            # # calculate pixel tolerated accuracy
            # p_corr, batch_size = p_acc(targets[:, :, :2], outputs[:, :, :], \
            #                         width_scale=args.sensor_width*args.spatial_factor, \
            #                         height_scale=args.sensor_height*args.spatial_factor, \
            #                             pixel_tolerances=args.pixel_tolerances)
            # total_p_corr_all = {f'p{k}_all': (total_p_corr_all[f'p{k}_all'] + p_corr[f'p{k}']).item() for k in args.pixel_tolerances}
            # total_samples_all += batch_size

            # # calculate averaged euclidean distance
            # p_error_total, bs_times_seqlen = px_euclidean_dist(targets[:, :, :3], outputs[:, :, :], \
            #                         width_scale=args.sensor_width*args.spatial_factor, \
            #                         height_scale=args.sensor_height*args.spatial_factor)
            # total_p_error_all = {f'error_all': (total_p_error_all[f'error_all'] + p_error_total).item()}
            # total_sample_p_error_all += bs_times_seqlen

    outputs_list = torch.cat(outputs_list, dim=0)
    targets_list = torch.cat(targets_list, dim=0)
    
    outputs_list = torch.cat([outputs_list, targets_list[:, :, -1].unsqueeze(-1)], dim=2)
    # pdb.set_trace()
    outputs, targets = process_array(outputs_list, targets_list[:, :, :3])
    outputs = np.concatenate(outputs, axis=0)
    # pdb.set_trace()
    targets = np.concatenate(targets, axis=0)
    outputs, targets = torch.tensor(outputs), torch.tensor(targets)
    p_corr, batch_size = p_acc(targets[:, :2], outputs[:, :2], \
                        width_scale=args.sensor_width*args.spatial_factor, \
                        height_scale=args.sensor_height*args.spatial_factor, \
                            pixel_tolerances=args.pixel_tolerances)
    total_p_corr_all = {f'p{k}_all': (total_p_corr_all[f'p{k}_all'] + p_corr[f'p{k}']).item() for k in args.pixel_tolerances}
    total_samples_all += batch_size
    p_error_total, bs_times_seqlen = px_euclidean_dist(targets[:, :3], outputs[:, :2], \
                        width_scale=args.sensor_width*args.spatial_factor, \
                        height_scale=args.sensor_height*args.spatial_factor)
    total_p_error_all = {f'error_all': (total_p_error_all[f'error_all'] + p_error_total).item()}
    total_sample_p_error_all += bs_times_seqlen
    metrics = {'val_p_acc_all': {f'val_p{k}_acc_all': (total_p_corr_all[f'p{k}_all']/total_samples_all) for k in args.pixel_tolerances},
                'val_p_error_all': {f'val_p_error_all': (total_p_error_all[f'error_all']/total_sample_p_error_all)}}
    
    return total_loss / len(val_loader), metrics


def top_k_checkpoints(args):
    """
    only save the top k model checkpoints with the lowest validation loss.
    """
    # list all files ends with .pth in artifact_uri
    model_checkpoints = [f for f in os.listdir(args.log_dir) if f.endswith(".pth")]

    # but only save at most args.save_k_best models checkpoints
    if len(model_checkpoints) > args.save_k_best:
        # sort all model checkpoints by validation loss in ascending order
        model_checkpoints = sorted([f for f in os.listdir(args.log_dir) if f.startswith("model_best_ep")], \
                                    key=lambda x: float(x.split("_")[-1][:-4]))
        # delete the model checkpoint with the largest validation loss
        os.remove(os.path.join(args.log_dir, model_checkpoints[-1]))


