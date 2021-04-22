import logging

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Callable, Optional, Dict, Tuple, Any, Type, List, Union

import yaml

import torch
import torch.nn as nn
import torchvision

from action_recognition import models

# Initiate Logger
logger = logging.getLogger(__name__)


def transform_dict(config_dict: Dict, expand: bool = True):
    """
    General function to transform any dictionary into wandb config acceptable format
    (This is mostly due to datatypes that are not able to fit into YAML format which makes wandb angry)
    The expand argument is used to expand iterables into dictionaries
    So the configs can be used when comparing results across runs
    """
    ret: Dict[str, Any] = {}
    for k, v in config_dict.items():
        if v is None or isinstance(v, (int, float, str)):
            ret[k] = v
        elif isinstance(v, (list, tuple, set)):
            # Need to check if item in iterable is YAML-friendly
            t = transform_dict(dict(enumerate(v)), expand)
            # Transform back to iterable if expand is False
            ret[k] = t if expand else [t[i] for i in range(len(v))]
        elif isinstance(v, dict):
            ret[k] = transform_dict(v, expand)
        else:
            # Transform to YAML-friendly (str) format
            # Need to handle both Classes, Callables, Object Instances
            # Custom Classes might not have great __repr__ so __name__ might be better in these cases
            vname = v.__name__ if hasattr(v, '__name__') else v.__class__.__name__
            ret[k] = f"{v.__module__}:{vname}"
    return ret


def dfac_cur_time():
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def update_dict_by_key_value(argdict, key, value):
    now_dict = argdict
    *dict_keys, last_key = key.split('.')
    for k in dict_keys:
        now_dict.setdefault(k, {})
        if now_dict[k] is None:
            now_dict[k] = {}
        now_dict = now_dict[k]
    now_dict[last_key] = value
    return argdict


ARGS_TYPE = Dict[str, Any]


@dataclass
class WandbConfig:
    # Logging Related
    cur_time: str = field(default_factory=dfac_cur_time)
    # WandB setting
    wandb_repo: str = "donny"
    wandb_project: str = "video_classification"
    wandb_group: str = "test"
    wandb_name: str = ''

    wandb_dir: str = "./output/wandb/"


@dataclass  # pylint: disable=too-many-instance-attributes
class ExperimentConfig(WandbConfig):
    # GPU Device Setting
    gpu_device_id: Union[None, int, List[int]] = None
    tensorboard_log_root: str = "./output/tensorboard/"

    # Set random seed. Set to None to create new Seed
    random_seed: Optional[int] = None

    # Dataset Config
    dataset_root: str = "./data/clipped_database/preprocessed"
    dataset_artifact: str = ""
    output_type: str = "random_frame"
    split_by: str = "random"
    frames_per_clip: int = 1
    step_between_clips: int = 1
    frame_rate: Optional[int] = None
    metadata_path: Optional[str] = None
    num_sample_per_clip: int = 1

    # mouse dataset args
    mix_clip: int = 0
    no_valid: bool = False
    extract_groom: bool = True
    exclude_5min: bool = False
    exclude_2_mouse: bool = False
    exclude_fpvid: bool = True
    exclude_2_mouse_valid: bool = False

    # Over-write arguments for valid set and test set
    valid_set_args: ARGS_TYPE = field(default_factory=dict)
    test_set_args: ARGS_TYPE = field(default_factory=dict)

    # sampler
    train_sampler_config: ARGS_TYPE = field(default_factory=dict)
    valid_sampler_config: ARGS_TYPE = field(default_factory=dict)
    test_sampler_config: ARGS_TYPE = field(default_factory=dict)

    # Transform Function
    transform_size: int = 224
    train_transform: Optional[Tuple[Callable, ...]] = None
    valid_transform: Optional[Tuple[Callable, ...]] = None
    test_transform: Optional[Tuple[Callable, ...]] = None

    # Increase dataloader worker to increase throughput
    num_worker: int = 16

    # Training Related
    batch_size: int = 64

    # Default Don't Select Model
    model: Optional[Union[Type[torch.nn.Module], Callable[..., torch.nn.Module]]] = None
    model_args: ARGS_TYPE = field(default_factory=dict)
    xavier_init: bool = False

    # Default Cross Entropy loss
    loss_function: nn.Module = field(default_factory=lambda: nn.CrossEntropyLoss(reduction='sum'))

    # Default Select Adam as Optimizer
    optimizer: Type[torch.optim.Optimizer] = torch.optim.Adam  # type: ignore
    optimizer_args: ARGS_TYPE = field(default_factory=lambda: {"lr": 1e-2})

    # Default adjust learning rate
    lr_scheduler: Optional[Type[torch.optim.lr_scheduler._LRScheduler]] = None  # pylint: disable=protected-access
    lr_scheduler_args: ARGS_TYPE = field(default_factory=dict)

    # Set number of epochs to train
    num_epochs: int = 20
    samples_per_epoch: Optional[int] = None

    valid_every_epoch: int = 1
    save_path: Optional[str] = None
    best_metric: str = 'Accuracy'

    def to_dict(self, expand: bool = True):
        return transform_dict(asdict(self), expand)

    def update_value(self, key, value):
        first_key, *rest = key.split('.')
        if key in ["model", "model.class"]:
            self.model = getattr(models, value, None) \
                or getattr(torchvision.models, value, None) \
                or getattr(torchvision.models.video, value)
        elif key.startswith('model.'):
            self.model_args = update_dict_by_key_value(self.model_args, key[len('model.'):], value)
        elif key.startswith('train_sampler_config.'):
            self.train_sampler_config = update_dict_by_key_value(
                self.train_sampler_config, key[len('train_sampler_config.'):], value)
        elif key in ["optimizer", "optimizer.class"]:
            assert hasattr(torch.optim, value)
            self.optimizer = getattr(torch.optim, value)
        elif key.startswith('optimizer.'):
            self.optimizer_args = update_dict_by_key_value(self.optimizer_args, key[len('optimizer.'):], value)
        elif key in ["lr_scheduler", "lr_scheduler.class"]:
            assert hasattr(torch.optim.lr_scheduler, value)
            self.lr_scheduler = getattr(torch.optim.lr_scheduler, value)
        elif key == "lr_scheduler.step":
            self.lr_scheduler_args['milestones'] = list(range(value, self.num_epochs, value))
        elif key.startswith('lr_scheduler.'):
            self.lr_scheduler_args = update_dict_by_key_value(self.lr_scheduler_args, key[len('lr_scheduler.'):], value)
        elif key == 'frames_per_clip':
            self.frames_per_clip = value
            self.batch_size = 16 * 16 // value
        elif hasattr(self, key):
            assert getattr(self, key) is None or isinstance(getattr(self, key), type(value)), (key, value)
            setattr(self, key, value)
        elif hasattr(self, first_key) and len(rest) > 0:
            assert isinstance(getattr(self, first_key), dict)
            setattr(
                self, first_key, update_dict_by_key_value(
                    getattr(self, first_key), key[len(first_key) + 1:], value)
            )
        else:
            raise NotImplementedError(f"Unknown key={key}")

    def update_sweep_dict(self, wandb_config: Dict[str, Any]):
        SWEEP_ARG_PREFIX = "WS_"
        for k, v in wandb_config.items():
            if k.startswith("WS_BASE"):
                sp_value = v['value']
                if isinstance(sp_value, list):
                    for item in sp_value:
                        self.merge_with_yaml(item)
                elif sp_value:
                    self.merge_with_yaml(sp_value)
            elif k.startswith(SWEEP_ARG_PREFIX):
                sp_name = k[len(SWEEP_ARG_PREFIX):]
                # wandb.config.as_dict() returns Dict[k, Dict[str, v]]
                # https://github.com/wandb/client/blob/master/wandb/wandb_config.py#L321
                sp_value = v['value']
                self.update_value(sp_name, sp_value)

    def update_dict_value(self, dictionary, prefix=''):
        for k, v in dictionary.items():
            if isinstance(v, dict):
                self.update_dict_value(v, prefix=prefix + k + '.')
            else:
                self.update_value(prefix + k, v)

    def merge_with_yaml(self, yaml_path):
        with open(yaml_path, 'r') as f:
            cfg_dict = yaml.safe_load(f)
        if "__BASE__" in cfg_dict:
            if isinstance(cfg_dict['__BASE__'], list):
                for path in cfg_dict['__BASE__']:
                    self.merge_with_yaml(path)
            else:
                assert isinstance(cfg_dict['__BASE__'], str)
                self.merge_with_yaml(cfg_dict['__BASE__'])
            cfg_dict.pop("__BASE__")
        self.update_dict_value(cfg_dict)
