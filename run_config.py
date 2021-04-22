import os
import argparse
import logging
import pprint

import wandb

from action_recognition.utils import setup_logging
from action_recognition.experiment import ExperimentConfig, run_experiment

# Initiate Logger
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", nargs='*')
    parser.add_argument("--base", nargs='*', help="Find config name under config/base/")
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("--name", type=str, default="")
    parser.add_argument("--group", type=str, default="")
    parser.add_argument("--lr", default=None, type=float)
    parser.add_argument("--batch_size", default=None, type=int)
    parser.add_argument("--epoch", default=None, type=int)
    parser.add_argument("--cage", default=-1, type=int)
    parser.add_argument("--split", default='', type=str)
    args = parser.parse_args()
    # Setup Experiment Config
    config = ExperimentConfig()

    # Setup Random Seed
    config.random_seed = 666

    if args.config:
        for cfg in args.config:
            config.merge_with_yaml(cfg)
    if args.base:
        config_base_root = "configs/base/"
        for base_name in args.base:
            config_path = os.path.join(config_base_root, f'{base_name}.yaml')
            assert os.path.exists(config_path), f"{base_name}.yaml not found under {config_base_root}"
            config.merge_with_yaml(config_path)
    if args.name:
        config.wandb_name = args.name
    if args.group:
        config.wandb_group = args.group

    if args.lr is not None:
        config.optimizer_args['lr'] = args.lr
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.epoch is not None:
        config.num_epochs = args.epoch

    if args.cage >= 0:
        config.split_by = f"cage_{args.cage}"
    if args.split:
        config.split_by = args.split

    config.num_worker = 8
    logging_level = "INFO"
    # debug config
    if args.debug:
        config.wandb_group = "Debug"
        # os.environ["WANDB_MODE"] = "dryrun"
        logging_level = "DEBUG"
        config.samples_per_epoch = config.batch_size * 5
        config.valid_every_epoch = 1
        config.num_epochs = 2

    # Init logging
    log_file_path = f'./output/{config.cur_time}.log'
    # setup_logging(log_file_path, "DEBUG")
    setup_logging(log_file_path, logging_level)
    logger.info("Experiment Config:\n%s", pprint.pformat(config.to_dict()))

    # Init wandb
    wandb.init(
        entity=config.wandb_repo, project=config.wandb_project,
        name=config.wandb_name or config.cur_time, group=config.wandb_group,
        dir=config.wandb_dir, config=config.to_dict(), job_type='train',
        sync_tensorboard=True,
    )
    # wandb.tensorboard.patch(pytorch=True)
    # config.update_sweep_dict(wandb.config.as_dict())

    # Run Experiment
    training_history, test_report = run_experiment(config)
    logger.info("Test Result:\n%s", pprint.pformat(test_report))

    # Save log file to wandb
    wandb.save(log_file_path)
