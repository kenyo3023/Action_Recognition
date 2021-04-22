import logging
import pprint
import os
import collections
import multiprocessing

import wandb

from action_recognition.utils import setup_logging
from action_recognition.experiment import ExperimentConfig, run_experiment
from action_recognition.evaluate.evaluate_video import expand_nested_str_dict
from action_recognition.evaluate.meters import AverageDictMeter
# from action_recognition.datasets import augmentation


# Initiate Logger
logger = logging.getLogger(__name__)


Worker = collections.namedtuple("Worker", ("queue", "process"))
WorkerInitData = collections.namedtuple(
    "WorkerInitData", ("split", "sweep_id", "sweep_run_name", "config")
)
WorkerDoneData = collections.namedtuple(
    "WorkerDoneData", ("train_history", "test_result")
)


def reset_wandb_env():
    exclude = {
        "WANDB_PROJECT",
        "WANDB_ENTITY",
        "WANDB_API_KEY",
    }
    for k, _ in os.environ.items():
        if k.startswith("WANDB_") and k not in exclude:
            del os.environ[k]


def run_single_split(worker_q, res_q):
    reset_wandb_env()
    worker_data: WorkerInitData = worker_q.get()
    # Setup Experiment Config
    config = ExperimentConfig()

    config.num_worker = 8
    config.update_sweep_dict(worker_data.config)
    config.split_by = f'cage_{worker_data.split}'

    # Init logging
    log_file_path = f'./output/{config.cur_time}.log'
    setup_logging(log_file_path, "INFO")

    # Init wandb
    wandb.init(
        entity=config.wandb_repo, project=config.wandb_project,
        name="{}-{}".format(worker_data.sweep_run_name, worker_data.split),
        group=worker_data.sweep_run_name, job_type=f'sweep-{worker_data.sweep_id}',
        dir=config.wandb_dir, config=config.to_dict()
    )
    wandb.tensorboard.patch(pytorch=True)
    logger.info("Experiment Config:\n%s", pprint.pformat(config.to_dict()))

    # Run Experiment
    train_history, test_report = run_experiment(config, save_model=False)
    logger.info("Test Result:\n%s", pprint.pformat(test_report))

    # Save log file to wandb
    wandb.save(log_file_path)

    res_q.put(WorkerDoneData(train_history, test_report))


if __name__ == "__main__":
    splits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    try:
        result_q: multiprocessing.Queue = multiprocessing.Queue()
        workers = []
        for split in splits:
            q: multiprocessing.Queue = multiprocessing.Queue()
            p = multiprocessing.Process(
                target=run_single_split, kwargs=dict(worker_q=q, res_q=result_q)
            )
            p.start()
            workers.append(Worker(queue=q, process=p))

        sweep_run = wandb.init()
        sweep_id = sweep_run.sweep_id or "unknown"
        sweep_url = sweep_run.get_sweep_url()
        project_url = sweep_run.get_project_url()
        sweep_group_url = "{}/groups/{}".format(project_url, sweep_id)
        sweep_run.notes = sweep_group_url
        sweep_run.save()
        sweep_run_name = sweep_run.name or sweep_run.id or "unknown"

        metrics = AverageDictMeter('Testing.')
        for split, worker in zip(splits, workers):
            worker.queue.put(
                WorkerInitData(
                    split=split,
                    sweep_id=sweep_id,
                    sweep_run_name=sweep_run_name,
                    config=wandb.config.as_dict())
            )
            result: WorkerDoneData = result_q.get()
            worker.process.join()
            metrics.update(expand_nested_str_dict(result.test_result), 1)

        wandb.log(metrics.summary())
        wandb.join()
    except KeyboardInterrupt:
        for i, worker in enumerate(workers):
            worker.process.terminate()
            print(f"Stopped worker: {i}")

    print("*" * 40)
    print("Sweep URL:       ", sweep_url)
    print("Sweep Group URL: ", sweep_group_url)
    print("*" * 40)
