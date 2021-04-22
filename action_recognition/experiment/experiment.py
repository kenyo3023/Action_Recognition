import os
import json
import logging

# from typing import Dict

import wandb

import torch
from torch.nn.parallel import DataParallel
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.models.resnet import ResNet

from action_recognition.training.training import train_model
from action_recognition.training.checkpoint import Checkpointer
from action_recognition.evaluate.evaluate import evaluate_model
from action_recognition.experiment.config import ExperimentConfig
from action_recognition.datasets.baseline_dataset import MouseClipDataset
from action_recognition.models.model_utils import init_xavier_weights
from action_recognition.utils import setup_random_seed

# Initiate Logger
logger = logging.getLogger(__name__)


def run_experiment(config: ExperimentConfig, save_model=True):  # pylint: disable=too-many-statements, too-many-branches, too-many-locals
    # Check Pytorch Version Before Running
    logger.info('Torch Version: %s', torch.__version__)  # type: ignore
    logger.info('Cuda Version: %s', torch.version.cuda)  # type: ignore

    if config.random_seed is not None:
        setup_random_seed(config.random_seed)

    # Initialize Writer
    writer_dir = f"{config.tensorboard_log_root}/{config.cur_time}/"
    writer = SummaryWriter(log_dir=writer_dir)

    # Initialize Device
    if isinstance(config.gpu_device_id, list):
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, config.gpu_device_id))
    elif config.gpu_device_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(config.gpu_device_id)

    replica = torch.cuda.device_count()
    logger.info('Device Counts: %s', replica)
    wandb.log({"Model batch size": config.batch_size * replica}, 0)
    # device = torch.device(f"cuda:{config.gpu_device_id}")

    # Initialize Dataset and Split into train/valid/test DataSets
    dataset_dict = {
        "output_type": config.output_type,
        "frames_per_clip": config.frames_per_clip,
        "step_between_clips": config.step_between_clips,
        "frame_rate": config.frame_rate,
        "num_sample_per_clip": config.num_sample_per_clip,
    }

    if config.dataset_artifact:
        dataset = MouseClipDataset.from_wandb_artifact(
            config.dataset_artifact,
            split_by=config.split_by,
            mix_clip=config.mix_clip,
            no_valid=config.no_valid,
            extract_groom=config.extract_groom,
            exclude_5min=config.exclude_5min,
            exclude_2_mouse=config.exclude_2_mouse,
            exclude_fpvid=config.exclude_fpvid,
            exclude_2_mouse_valid=config.exclude_2_mouse_valid,
            **dataset_dict)
    elif config.dataset_root in ['./data/breakfast', './data/mpii']:
        dataset = MouseClipDataset.from_annotation_list(dataset_root=config.dataset_root, **dataset_dict)
    else:
        metadata_path = (
            config.metadata_path
            if config.metadata_path is not None
            else os.path.join(config.dataset_root, "metadata.pth")
        )
        dataset = MouseClipDataset.from_ds_folder(
            dataset_root=config.dataset_root,
            metadata_path=metadata_path,
            extract_groom=config.extract_groom,
            **dataset_dict)

    train_set = dataset.get_split("train", config.split_by, config.transform_size, {})
    valid_set = dataset.get_split("valid", config.split_by, config.transform_size, config.valid_set_args)
    test_set = dataset.get_split("test", config.split_by, config.transform_size, config.test_set_args)

    logger.info('Train Transform:\n%s', train_set.transform)
    logger.info('Valid Transform:\n%s', valid_set.transform)
    logger.info('Test Transform:\n%s', test_set.transform)

    dataloaders = {
        "train": DataLoader(
            train_set, config.batch_size * replica,
            sampler=train_set.get_sampler("train", config.train_sampler_config, config.samples_per_epoch),
            num_workers=config.num_worker, pin_memory=True, drop_last=True),
        "valid": DataLoader(
            valid_set, config.batch_size * replica,
            sampler=valid_set.get_sampler("valid", config.valid_sampler_config),
            num_workers=config.num_worker, pin_memory=True, drop_last=False),
        "test": DataLoader(
            test_set, config.batch_size * replica,
            sampler=test_set.get_sampler("test", config.test_sampler_config),
            num_workers=config.num_worker, pin_memory=True, drop_last=False),
    }

    # initialize model
    if config.model is not None:
        model = config.model(**config.model_args)
        if isinstance(model, ResNet):
            model.fc = torch.nn.Linear(model.fc.in_features, len(set(dataset.labels)))

        if config.xavier_init:
            model = init_xavier_weights(model)
        # Make wandb Track the model
        wandb.watch(model, "parameters")

        logger.info('Model: %s', model.__class__.__name__)
        # Log total parameters in the model
        pytorch_total_params = sum(p.numel() for p in model.parameters())
        logger.info('Model params: %s', pytorch_total_params)
        pytorch_total_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info('Model params trainable: %s', pytorch_total_params_trainable)

        model_structure_str = "Model Structue:\n"
        for name, param in model.named_parameters():
            model_structure_str += f"\t{name}: {param.requires_grad}, {param.numel()}\n"
        # logger.info(model_structure_str)

        model = model.cuda()
        if replica > 1:
            model = DataParallel(model)
    else:
        logger.critical("Model not chosen in config!")
        return None

    if isinstance(config.loss_function, torch.nn.Module):
        config.loss_function = config.loss_function.cuda()

    optimizer = config.optimizer(params=model.parameters(), **config.optimizer_args)
    logger.info("Optimizer: %s\n%s", config.optimizer.__name__, config.optimizer_args)

    if config.lr_scheduler is not None:
        lr_scheduler = config.lr_scheduler(optimizer, **config.lr_scheduler_args)
        logger.info("LR Scheduler: %s\n%s", config.lr_scheduler.__name__, config.lr_scheduler_args)
    else:
        lr_scheduler = None
        logger.info("No LR Scheduler")

    logger.info("Training Started!")
    ckpter = Checkpointer(config.best_metric, save_path=config.save_path)

    training_history, total_steps = train_model(
        model=model,
        optimizer=optimizer,
        dataloaders=dataloaders,
        writer=writer,
        num_epochs=config.num_epochs,
        loss_function=config.loss_function,
        lr_scheduler=lr_scheduler,
        valid_every_epoch=config.valid_every_epoch,
        ckpter=ckpter,
    )
    logger.info("Training Complete!")

    if ckpter is not None:
        ckpter.load_best_model(model)

    logger.info("Testing Started!")
    test_report = evaluate_model(
        model, dataloaders['test'], "Testing", total_steps, writer, config.loss_function)
    logger.info("Testing Complete!")

    if save_model:
        train_artifact = wandb.Artifact(f'run_{wandb.run.id}_model', 'model')
        model_tmp_path = os.path.join(wandb.run.dir, f'best_valid_{ckpter.name}_model.pth')
        torch.save(model.module if isinstance(model, DataParallel) else model, model_tmp_path)
        train_artifact.add_file(model_tmp_path)
        with train_artifact.new_file('split_data.json') as f:
            json.dump(dataset.split_data, f)
        wandb.run.log_artifact(train_artifact)

    return training_history, test_report
