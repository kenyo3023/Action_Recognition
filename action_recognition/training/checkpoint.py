import logging

import torch

from action_recognition.utils import safe_dir

# Initiate Logger
logger = logging.getLogger(__name__)


def save_model(model, path, **kwargs):
    if isinstance(model, torch.nn.parallel.DataParallel):
        model = model.module
    # model_dict = model.state_dict()
    save_format = {
        # "state_dict": model_dict,
        "model": model,
    }
    save_format.update(kwargs)
    torch.save(save_format, path)


def load_from_path(model, path):
    if isinstance(model, torch.nn.parallel.DataParallel):
        model = model.module
    model.load_state_dict(torch.load(path, map_location='cuda'))


class Checkpointer:
    def __init__(self, name, save_path=None):
        self.name = name
        self.minimize = self.is_minimize(name)
        self.value = float('inf') if self.minimize else float('-inf')
        self.model_dict = None
        if save_path is not None:
            safe_dir(save_path, with_filename=True)
        self.save_path = save_path

    @staticmethod
    def is_minimize(name: str):
        if 'loss' in name.lower():
            return True
        if 'accuracy' in name.lower():
            return False
        raise NotImplementedError(f"unknown name {name}")

    def is_better(self, new_val):
        if self.minimize:
            return new_val < self.value
        else:
            return new_val > self.value

    def update_with_metric(self, metric_value, model: torch.nn.Module):
        if isinstance(model, torch.nn.parallel.DataParallel):
            model = model.module
        if self.is_better(metric_value):
            self.value = metric_value
            self.model_dict = {k: v.cpu() for k, v in model.state_dict().items()}
            if self.save_path:
                save_model(model, self.save_path, metric_name=self.name, metric_value=metric_value)

    def load_best_model(self, model):
        if self.model_dict is None:
            logger.info("Not resuming cause no model is logged")
            return
        logger.info('Resuming best model from ckpt by metric %s, value: %s', self.name, self.value)
        if isinstance(model, torch.nn.parallel.DataParallel):
            model = model.module
        model.load_state_dict(self.model_dict)
        if torch.cuda.is_available():
            model = model.cuda()
