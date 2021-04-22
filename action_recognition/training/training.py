import logging

from typing import List, Dict, Tuple, Optional, Union, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm, trange

from action_recognition.training.checkpoint import Checkpointer
from action_recognition.evaluate.evaluate import evaluate_model
from action_recognition.evaluate.meters import AverageMeter
from action_recognition.evaluate.metrics import multiclass_roc_auc_score, sensitivity_specificity_support_with_avg
from action_recognition.utils.wandb_utils import log_classification_report

# Initiate Logger
logger = logging.getLogger(__name__)


def train_model(model: nn.Module,  # pylint: disable=too-many-locals, too-many-statements, too-many-branches
                optimizer: torch.optim.Optimizer,  # type: ignore
                dataloaders: Dict[str, DataLoader],
                writer: SummaryWriter,
                num_epochs: int = 100,
                loss_function: Union[nn.Module, Callable] = F.cross_entropy,
                lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,  # pylint: disable=protected-access
                valid_every_epoch: int = 5,
                ckpter: Optional[Checkpointer] = None) -> Tuple[List[Dict], int]:

    # Remember total instances trained for plotting
    total_steps = 0

    # Save Per Epoch Progress
    result: List[Dict] = []
    accuracy_meter = AverageMeter()
    loss_meter = AverageMeter()

    epochs = trange(1, num_epochs + 1, dynamic_ncols=True)
    for epoch in epochs:
        epochs.set_description(f'Training Epoch: {epoch}')

        # Set model to Training Mode
        model.train()
        accuracy_meter.reset()
        loss_meter.reset()

        y_pred, y_truth = [], []
        y_pred_prob = []

        training_data = tqdm(dataloaders['train'], dynamic_ncols=True, leave=False)
        for data, target, _video_idx in training_data:
            # Move data to device, model shall already be at device
            data = data.cuda()
            target = target.cuda()

            # Run batch data through model
            output = model(data)

            # Get and Sum up Batch Loss
            batch_loss = loss_function(output, target)
            batch_size = len(data)
            loss_meter.update(batch_loss.item(), batch_size)

            # Increment Correct Count and Total Count
            prediction = output.max(1, keepdim=True)[1]
            correct_cnt = prediction.eq(target.view_as(prediction)).sum().item()
            accuracy_meter.update(correct_cnt, batch_size)

            # Back Propagation the Loss
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            # Append Prediction Results
            y_truth.append(target.cpu())
            y_pred.append(prediction.reshape(-1).cpu())
            y_pred_prob.append(F.softmax(output).cpu().detach())

            training_data.set_description(
                f'Train loss: {loss_meter.avg:.3f} '
                f'Accuracy: {accuracy_meter.avg * 100:.2f}%')

            # Write Progress to Tensorboard
            total_steps += 1
            writer.add_scalar('BATCH/Training Loss',
                              loss_meter.avg, total_steps)
            writer.add_scalar('BATCH/Training Accuracy',
                              accuracy_meter.avg, total_steps)

        # Merge results from each batch
        y_truth = np.concatenate(y_truth)
        y_pred = np.concatenate(y_pred)
        y_pred_prob = np.concatenate(y_pred_prob)

        # Get unique y values
        unique_y = np.unique(np.concatenate([y_truth, y_pred])).tolist()
        # unique_y = np.unique(y_truth).tolist()

        # Print Evaluation Metrics and log to wandb
        report = log_classification_report(
            "Training", total_steps, writer, loss_meter.avg,
            classification_report(
                y_true=y_truth, y_pred=y_pred, labels=unique_y,
                output_dict=True,
            ),
            sensitivity_specificity_support_with_avg(y_truth, y_pred, unique_y),
            multiclass_roc_auc_score(y_truth, y_pred_prob, unique_y),
            confusion_matrix(y_truth, y_pred, labels=unique_y),
            cid2name=None
        )

        # Log per epoch metric
        writer.add_scalar('Epoch', epoch, total_steps)

        do_validation = (epoch % valid_every_epoch == 0)

        per_epoch_metric = {
            "train": {
                "Accuracy": accuracy_meter.avg,
                "Loss": loss_meter.avg,
            }
        }
        per_epoch_metric['train'].update(report)

        if do_validation:
            # Start Validation
            epochs.set_description(f'Validating Epoch: {epoch}')
            per_epoch_metric['valid'] = evaluate_model(
                model, dataloaders['valid'], "Validation", total_steps, writer, loss_function)

        if lr_scheduler is not None:
            lr_scheduler.step()  # type: ignore
            last_lr = lr_scheduler.get_last_lr()  # type: ignore
            logger.debug("lr_scheduler Last lr: %s", last_lr)
            writer.add_scalar('lr_scheduler', last_lr[0], total_steps)

        if ckpter is not None and 'valid' in per_epoch_metric:  # TODO: match per_epoch_metric with ckpt.name
            ckpter.update_with_metric(per_epoch_metric['valid'][ckpter.name], model)

        result.append(per_epoch_metric)

    return result, total_steps
