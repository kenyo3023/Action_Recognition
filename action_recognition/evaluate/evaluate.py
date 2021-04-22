import logging

from typing import Dict, Union, Callable

import numpy as np

from sklearn.metrics import classification_report, confusion_matrix
# from scipy.special import softmax

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

from action_recognition.evaluate.metrics import multiclass_roc_auc_score, sensitivity_specificity_support_with_avg
from action_recognition.utils.wandb_utils import log_classification_report
from action_recognition.evaluate.meters import AverageMeter

# Initiate Logger
logger = logging.getLogger(__name__)


def evaluate_model(
        model: nn.Module,
        test_loader: DataLoader,
        wandb_name: str,
        wandb_step: int,
        writer: SummaryWriter,
        loss_function: Union[nn.Module, Callable] = F.cross_entropy,
) -> Dict:

    # Set model to Eval Mode (For Correct Dropout and BatchNorm Behavior)
    model.eval()

    loss_meter = AverageMeter()
    accuracy_meter = AverageMeter()
    # Save Predictions, Predicted probability and Truth Data for Evaluation Report
    y_pred, y_truth = [], []
    y_pred_prob = []

    with torch.no_grad():
        testing_data = tqdm(test_loader, dynamic_ncols=True, leave=False)
        for data, target, _video_idx in testing_data:
            # Move data to device, model shall already be at device
            data = data.cuda()
            target = target.cuda()

            # Run batch data through model
            output = model(data)
            prediction = output.max(1, keepdim=True)[1]

            # Get and Sum up Batch Loss
            batch_loss = loss_function(output, target)
            batch_size = len(data)
            loss_meter.update(batch_loss.item(), batch_size)

            # Increment Correct Count and Total Count
            correct_cnt = prediction.eq(target.view_as(prediction)).sum().item()
            accuracy_meter.update(correct_cnt, batch_size)

            # TODO: add to meters
            # Append Prediction Results
            y_truth.append(target.cpu())
            y_pred.append(prediction.reshape(-1).cpu())
            y_pred_prob.append(F.softmax(output).cpu().detach())

    # Merge results from each batch
    y_truth = np.concatenate(y_truth)
    y_pred = np.concatenate(y_pred)
    y_pred_prob = np.concatenate(y_pred_prob)

    # Get unique y values
    unique_y = np.unique(np.concatenate([y_truth, y_pred])).tolist()
    # unique_y = np.unique(y_truth).tolist()

    # Print Evaluation Metrics and log to wandb
    # TODO: Add name mapping for categories
    report = log_classification_report(
        wandb_name, wandb_step, writer, loss_meter.avg,
        classification_report(
            y_true=y_truth, y_pred=y_pred, labels=unique_y,
            output_dict=True,
        ),
        sensitivity_specificity_support_with_avg(y_truth, y_pred, unique_y),
        multiclass_roc_auc_score(y_truth, y_pred_prob, unique_y),
        confusion_matrix(y_truth, y_pred, labels=unique_y),
        cid2name=None
    )

    return report
