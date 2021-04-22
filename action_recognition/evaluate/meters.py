from typing import Dict, Any

# import numpy as np


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.correct = 0
        self.total = 0

    # trace loss and accuracy
    def update(self, corr, n):
        self.correct += corr
        self.total += n

    @property
    def avg(self) -> float:
        return self.correct / self.total

    def summary(self) -> Dict[str, Any]:
        return {
            "Accuracy": self.avg,
            "Correct": self.correct,
            "Total": self.total,
        }


class AverageDictMeter:
    def __init__(self, prefix='', main=''):
        self.meters = {}
        self.prefix = prefix
        self.main = main
        self.reset()

    def reset(self):
        for k in self.meters:
            self.meters[k].reset()

    def update(self, corr: Dict[str, Any], n):
        for k in corr:
            if not self.main:
                self.main = k
            meter = self.meters.get(k, AverageMeter())
            meter.update(corr[k], n)
            self.meters[k] = meter

    @property
    def avg(self) -> Dict[str, float]:
        return self.meters[self.main].avg

    def summary(self):
        return {f'{self.prefix}{k}': self.meters[k].avg for k in self.meters}
