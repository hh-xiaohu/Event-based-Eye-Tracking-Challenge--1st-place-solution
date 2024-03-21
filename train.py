"""
This script implements a PyTorch deep learning training pipeline for an eye tracking application.
It includes a main function to pass in arguments, train and validation functions, and uses MLflow as the logging library.
The script also supports fine-grained deep learning hyperparameter tuning using argparse and JSON configuration files.
All hyperparameters are logged with MLflow.

Author: Zuowen Wang
Affiliation: Insitute of Neuroinformatics, University of Zurich and ETH Zurich
Email: wangzu@ethz.ch
"""

import argparse, json, yaml, os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.training_utils import train_epoch, validate_epoch, top_k_checkpoints
from utils.metrics import weighted_MSELoss
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from dataset.ThreeET_plus import ThreeETplus_Eyetracking
from dataset.custom_transforms import ScaleLabel, NormalizeLabel, \
    TemporalSubsample, NormalizeLabel, SliceLongEventsToShort, \
    EventSlicesToMap, SliceByTimeEventsTargets, \
    Jitter
    

import tonic.transforms as transforms
from tonic import SlicedDataset, DiskCachedDataset
import pdb
import importlib
import logging
import determined as det
import time
import shutil
def log_setting(config):
    logger = logging.getLogger()
    ch = logging.StreamHandler() # Adding Terminal Logger

    # Adding File Logger
    log_dir = os.path.join(config['log_dir'], config['model'], config['run_name'])
    os.makedirs(log_dir, exist_ok=True)
    # 将submission文件复制到该文件夹下，用shutil.copyfile
    shutil.copy2("submission.csv", log_dir)

    config['log_dir'] = log_dir

    fh = logging.FileHandler(filename=os.path.join(log_dir, 'logger.txt'))
    fh.setFormatter(logging.Formatter("%(asctime)s  : %(message)s", "%b%d-%H:%M"))

    if not logger.handlers:
        logger.addHandler(ch)
        logger.addHandler(fh)

    logger.setLevel(logging.INFO)
    logger.info(yaml.dump(config, sort_keys=False, default_flow_style=False))
    return logger

def train(model, train_loader, val_loader, criterion, optimizer, args, logger):
    best_val_loss = float("inf")
    # scheduler = optim.lr_scheduler.StepLR(optimizer, 100, 0.5, -1)
    lr_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=8, T_mult=2, eta_min=1e-6, last_epoch=-1, verbose=True)
    # Training loop
    with det.core.init() as core_context:
        for epoch in range(args.num_epochs):

            model, train_loss, metrics = train_epoch(model, train_loader, criterion, optimizer, lr_scheduler, epoch, args)
            core_context.train.report_training_metrics(steps_completed=epoch, metrics={f"train_loss": train_loss})
            for k, v in metrics['tr_p_acc_all'].items():
                core_context.train.report_training_metrics(steps_completed=epoch, metrics={k[3:]: v})
            core_context.train.report_training_metrics(steps_completed=epoch, metrics={"p_error_all": metrics["tr_p_error_all"]["tr_p_error_all"]})

            if args.val_interval > 0 and (epoch + 1) % args.val_interval == 0:
                val_loss, val_metrics = validate_epoch(model, val_loader, criterion, args)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    # save the new best model to MLflow artifact with 3 decimal places of validation loss in the file name
                    torch.save(model.state_dict(), os.path.join(args.log_dir, \
                                f"model_best_ep{epoch}_val_loss_{val_loss:.4f}.pth"))
                    
                    # DANGER Zone, this will delete files (checkpoints) in MLflow artifact
                    top_k_checkpoints(args)
                    
                logger.info(f"[Validation] at Epoch {epoch+1}/{args.num_epochs}: Val Loss: {val_loss:.4f}")

                core_context.train.report_validation_metrics(steps_completed=epoch, metrics={f"val_loss": val_loss})
                for k, v in val_metrics['val_p_acc_all'].items():
                    core_context.train.report_validation_metrics(steps_completed=epoch, metrics={k[4:]: v})
                core_context.train.report_validation_metrics(steps_completed=epoch, metrics={f"p_error_all": val_metrics['val_p_error_all']['val_p_error_all']})

            # Print progress
            logger.info(f"Epoch {epoch+1}/{args.num_epochs}: Train Loss: {train_loss:.4f}")

            # scheduler.step()
    return model


def main(args):
    # Load hyperparameters from JSON configuration file
    # if args.config_file:
        # with open(os.path.join('./configs', args.config_file), 'r') as f:
        #     config = json.load(f)
        # Overwrite hyperparameters with command-line arguments

        # for key, value in vars(args).items():
        #     if value is not None:
        #         config[key] = value

        # if args.timestamp is None:
        #     config['timestamp'] = time.strftime('%y%m%d%H%M%S', time.localtime(time.time())) # Add timestamp with format mouth-day-hour-minute
        # else:
        #     config['timestamp'] = args.timestamp

        # logger = log_setting(config)
        # args = argparse.Namespace(**config)
    # else:
    #     raise ValueError("Please provide a JSON configuration file.")

    with open(os.path.join('./configs', args.config_file), 'r') as f:
        # print(f"Loading config from {args.config_file}")
        config = json.load(f)
    # Overwrite hyperparameters with command-line arguments
    for key, value in vars(args).items():
        if value is not None:
            config[key] = value
    args = argparse.Namespace(**config)
    logger = log_setting(config)
    args = argparse.Namespace(**config)

    # Define your model, optimizer, and criterion
    model = importlib.import_module(f"model.{args.model}").Model(args).to(args.device)
    model = nn.DataParallel(model)
    if args.spatial_factor > 0.125:
        model.load_state_dict(torch.load(args.checkpoint))
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if args.loss == "mse":
        criterion = nn.MSELoss()
    elif args.loss == "weighted_mse":
        criterion = weighted_MSELoss(weights=torch.tensor((args.sensor_width/args.sensor_height, 1)).to(args.device), \
                                        reduction='mean')
    else:
        raise ValueError("Invalid loss name")

    factor = args.spatial_factor # spatial downsample factor
    temp_subsample_factor = args.temporal_subsample_factor # downsampling original 100Hz label to 20Hz

    # First we define the label transformations
    label_transform = transforms.Compose([
        ScaleLabel(factor),
        TemporalSubsample(temp_subsample_factor),
        NormalizeLabel(pseudo_width=640*factor, pseudo_height=480*factor)
    ])

    # Then we define the raw event recording and label dataset, the raw events spatial coordinates are also downsampled
    train_data_orig = ThreeETplus_Eyetracking(save_to=args.data_dir, split="train", \
                    transform=transforms.Downsample(spatial_factor=factor), 
                    target_transform=label_transform, dataset=args.dataset)
    val_data_orig = ThreeETplus_Eyetracking(save_to=args.data_dir, split="val", \
                    transform=transforms.Downsample(spatial_factor=factor),
                    target_transform=label_transform, dataset=args.dataset)

    slicing_time_window = args.train_length*int(10000/temp_subsample_factor) #microseconds
    train_stride_time = int(10000/temp_subsample_factor*args.train_stride) #microseconds
    valid_stride_time = int(10000/temp_subsample_factor*args.val_stride) #microseconds
    train_slicer=SliceByTimeEventsTargets(slicing_time_window, overlap=slicing_time_window-train_stride_time, \
                    seq_length=args.train_length, seq_stride=args.train_stride, include_incomplete=True)
    val_slicer=SliceByTimeEventsTargets(slicing_time_window, overlap=slicing_time_window-valid_stride_time, \
                    seq_length=args.val_length, seq_stride=args.val_stride, include_incomplete=True)

    post_slicer_transform = transforms.Compose([
        SliceLongEventsToShort(time_window=int(10000/temp_subsample_factor), overlap=0, include_incomplete=True),
        EventSlicesToMap(sensor_size=(int(640*factor), int(480*factor), 2), \
                                n_time_bins=args.n_time_bins, per_channel_normalize=args.voxel_grid_ch_normaization,
                                map_type=args.map_type)
    ])
    # train_data = SlicedDataset(train_data_orig, train_slicer, transform=post_slicer_transform)
    # val_data = SlicedDataset(val_data_orig, val_slicer, transform=post_slicer_transform)
    # pdb.set_trace()
    if args.dataset == "t":
        train_data = SlicedDataset(train_data_orig, train_slicer, transform=post_slicer_transform, metadata_path=f"{args.metadata_dir}/3et_train_tl_{args.train_length}_ts{args.train_stride}_ch{args.n_time_bins}_t{args.map_type}")
        val_data = SlicedDataset(val_data_orig, val_slicer, transform=post_slicer_transform, metadata_path=f"{args.metadata_dir}/3et_val_vl_{args.val_length}_vs{args.val_stride}_ch{args.n_time_bins}_t{args.map_type}")
        train_data = DiskCachedDataset(train_data, 
                                   cache_path=f"{args.cache_dir}/train_tl_{args.train_length}_ts{args.train_stride}_ch{args.n_time_bins}_t{args.map_type}",
                                   transforms=Jitter())
        val_data = DiskCachedDataset(val_data, cache_path=f"{args.cache_dir}/val_vl_{args.val_length}_vs{args.val_stride}_ch{args.n_time_bins}_t{args.map_type}",
                                   transforms=None)
    else:
        train_data = SlicedDataset(train_data_orig, train_slicer, transform=post_slicer_transform, metadata_path=f"{args.metadata_dir}/3et_train_tl_{args.train_length}_ts{args.train_stride}_{args.dataset}")
        val_data = SlicedDataset(val_data_orig, val_slicer, transform=post_slicer_transform, metadata_path=f"{args.metadata_dir}/3et_val_vl_{args.val_length}_vs{args.val_stride}_{args.dataset}")
        train_data = DiskCachedDataset(train_data, 
                                   cache_path=f"{args.cache_dir}/train_tl_{args.train_length}_ts{args.train_stride}_{args.dataset}",
                                   transforms=Jitter())
        val_data = DiskCachedDataset(val_data, cache_path=f"{args.cache_dir}/val_vl_{args.val_length}_vs{args.val_stride}_{args.dataset}",
                                   transforms=None)

 
    # Finally we wrap the dataset with pytorch dataloader
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, \
                                num_workers=4, pin_memory=False)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, \
                            num_workers=4)

    # Train your model
    model = train(model, train_loader, val_loader, criterion, optimizer, args, logger)

    # Save your model for the last epoch
    torch.save(model.state_dict(), os.path.join(args.log_dir, f"model_last_epoch{args.num_epochs}.pth"))

    from test import test
    model.eval()
    test(args, model)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # training management arguments     
    
    # a config file 
    parser.add_argument("--config_file", 
                        default="sliced_baseline.json", 
                        help="path to JSON configuration file")
    parser.add_argument("--run_name", type=str, help="name of the run")
    # training hyperparameters
    parser.add_argument("--lr", type=float, help="learning rate")
    parser.add_argument("--num_epochs", type=int, help="number of epochs")
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--train_length", type=int)
    parser.add_argument("--val_length", type=int)
    parser.add_argument("--test_length", type=int)
    parser.add_argument("--train_stride", type=int)
    parser.add_argument("--val_stride", type=int)
    parser.add_argument("--test_stride", type=int)
    parser.add_argument("--n_time_bins", type=int)
    parser.add_argument("--map_type", type=str)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--spatial_factor", type=float)
    args = parser.parse_args()

    main(args)
