import argparse
import logging
import pprint

import wandb

import torch
# from torch.nn import functional as F
# import torchvision
# import torchvision.transforms as transforms
# from torchvision.transforms import _transforms_video as transforms_video

from action_recognition.utils import setup_logging
from action_recognition.experiment import ExperimentConfig, run_experiment
from action_recognition.models import I3D

# Initiate Logger
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cage", default=-1, type=int)
    args = parser.parse_args()
    # Setup Experiment Config
    config = ExperimentConfig()

    # Setup Logging Group
    config.wandb_group = "5min_data_3D_nomix_balancevideo"
    if args.cage >= 0:
        config.split_by = f"cage_{args.cage}"

    # Setup Random Seed
    config.random_seed = 666

    # Setup dataset
    # config.dataset_root = "./data/breakfast"
    # config.split_by = 'cv_0'
    # config.dataset_root = './data/mouse_video/crop_preprocessed'
    config.dataset_artifact = 'mouse_cropped'
    # config.output_type = "video"
    config.output_type = "random_frame"
    config.frames_per_clip = 1
    # config.frame_rate = 3
    config.num_sample_per_clip = 1
    config.extract_groom = True
    config.mix_clip = 0

    # I3D
    config.batch_size = 16
    config.frames_per_clip = 16
    config.output_type = "video"

    config.train_sampler_config = {
        "log_weight": False,
        "balance_label": True,
        "balance_clip": True,
        "balance_src": False,
    }

    config.valid_set_args = {"num_sample_per_clip": 1}
    config.test_set_args = {"num_sample_per_clip": 1}

    # Set Transformations
    # config.train_transform = (
    #     transforms_video.ToTensorVideo(),
    #     # augmentation.VideoClipResize(224),  # not square
    #     # transforms_video.CenterCropVideo(224),
    #     transforms_video.RandomResizedCropVideo(112),
    #     augmentation.RandomVerticalFlipVideo(),
    #     transforms_video.RandomHorizontalFlipVideo(),
    #     transforms_video.NormalizeVideo(
    #         augmentation.kinetics400_transform_dict["mean"],
    #         augmentation.kinetics400_transform_dict["std"]),
    # )
    # config.valid_transform = (
    #     transforms_video.ToTensorVideo(),
    #     augmentation.VideoClipResize(112),  # not square
    #     transforms_video.CenterCropVideo(112),
    #     transforms_video.NormalizeVideo(
    #         augmentation.kinetics400_transform_dict["mean"],
    #         augmentation.kinetics400_transform_dict["std"]),
    # )
    # config.test_transform = (
    #     transforms_video.ToTensorVideo(),
    #     augmentation.VideoClipResize(112),  # not square
    #     transforms_video.CenterCropVideo(112),
    #     transforms_video.NormalizeVideo(
    #         augmentation.kinetics400_transform_dict["mean"],
    #         augmentation.kinetics400_transform_dict["std"]),
    # )

    # Set Model
    # config.model = torchvision.models.resnet18
    # config.model = torchvision.models.resnet101
    # config.model = torchvision.models.resnet152
    config.model = I3D
    # config.model = torchvision.models.video.r2plus1d_18
    config.model_args = {
        "num_classes": 2,
        "dropout_prob": 0.5,
        "state_dict_path": 'action_recognition/models/i3d_weights/model_RGB_400.pth',
        # "freeze": True,
        # "finetune_layers": ('mixed_5b', 'mixed_5c', 'conv3d_0c_1x1'),
        # "pretrained": True,
    }
    # config.save_path = './output/best_cv0_model.pth'

    # Set Batch Size
    # config.batch_size = 16 * 16 // config.frames_per_clip  # for full GPU usage
    # config.batch_size = 64

    # Set Number of Epochs
    config.num_epochs = 50
    config.samples_per_epoch = 2048

    # Set Loss function
    config.loss_function = torch.nn.CrossEntropyLoss(reduction='sum')
    # config.loss_function = torch.nn.CrossEntropyLoss(
    #     weight=torch.as_tensor([20., 1.]), reduction='sum')

    # Set Optimizer
    # config.optimizer = torch.optim.SGD
    config.optimizer = torch.optim.Adam
    config.optimizer_args = {
        "lr": 0.001,
        # "momentum": 0.9,
        # "momentum": 0.7,
        # "nesterov": True,
        # "weight_decay": 5e-4,
    }

    # Set lr_scheduler
    # config.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR
    config.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR
    # config.lr_scheduler = torch.optim.lr_scheduler
    config.lr_scheduler_args = {
        # "milestones": [3, 5, 7, 9, 11, 13, 15],
        # "gamma": 0.2,
        "gamma": 0.98,
    }

    # debug config
    # config.samples_per_epoch = config.batch_size * 5
    # config.valid_every_epoch = 1
    # config.num_epochs = 2

    # Init logging
    log_file_path = f'./output/{config.cur_time}.log'
    # setup_logging(log_file_path, "DEBUG")
    setup_logging(log_file_path, "INFO")
    logger.info("Experiment Config:\n%s", pprint.pformat(config.to_dict()))

    # Init wandb
    wandb.init(
        entity=config.wandb_repo, project=config.wandb_project,
        name=config.wandb_name or config.cur_time, group=config.wandb_group,
        dir=config.wandb_dir, config=config.to_dict(), job_type='train',
    )
    wandb.tensorboard.patch(pytorch=True)
    config.update_sweep_dict(wandb.config.as_dict())

    # Run Experiment
    training_history, test_report = run_experiment(config)
    logger.info("Test Result:\n%s", pprint.pformat(test_report))

    # Save log file to wandb
    wandb.save(log_file_path)
