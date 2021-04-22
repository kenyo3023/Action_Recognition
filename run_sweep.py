import logging
import pprint

import wandb

from action_recognition.utils import setup_logging
from action_recognition.experiment import ExperimentConfig, run_experiment
# from action_recognition.datasets import augmentation


# Initiate Logger
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    # Setup Experiment Config
    config = ExperimentConfig()

    config.num_worker = 8

    # Init logging
    log_file_path = f'./output/{config.cur_time}.log'
    setup_logging(log_file_path, "INFO")

    # Init wandb
    wandb.init(
        entity=config.wandb_repo, project=config.wandb_project,
        name=f'{config.cur_time}', group=config.wandb_group,
        dir=config.wandb_dir, config=config.to_dict()
    )
    wandb.tensorboard.patch(pytorch=True)
    config.update_sweep_dict(wandb.config.as_dict())
    logger.info("Experiment Config:\n%s", pprint.pformat(config.to_dict()))

    # Run Experiment
    training_history, test_report = run_experiment(config, save_model=False)
    logger.info("Test Result:\n%s", pprint.pformat(test_report))

    # Save log file to wandb
    wandb.save(log_file_path)
