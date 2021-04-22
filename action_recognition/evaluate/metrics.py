from typing import List

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder

from imblearn.metrics import sensitivity_specificity_support


def multiclass_roc_auc_score(y_truth, y_pred, labels) -> List[float]:
    oh = OneHotEncoder(categories=[labels], sparse=False)
    oh.fit([[lab] for lab in labels])
    y_truth = oh.transform(y_truth.reshape(-1, 1))
    return [
        roc_auc_score(y_truth[:, i], y_pred[:, i]) if len(np.unique(y_truth[:, i])) > 1 else -1
        for i, _cn in enumerate(oh.categories_[0])]


def sensitivity_specificity_support_with_avg(y_truth, y_pred, labels):
    return sensitivity_specificity_support(y_truth, y_pred, labels=labels), \
        sensitivity_specificity_support(y_truth, y_pred, labels=labels, average="macro"), \
        sensitivity_specificity_support(y_truth, y_pred, labels=labels, average="weighted")
