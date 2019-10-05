import types
import math
import torch
from torch._six import inf
from bisect import bisect_right
from functools import partial

from torch.optim.lr_scheduler import _LRScheduler

from .optimizer import Optimizer


class WaveLR(_LRScheduler):

    def __init__(self, optimizer, base_lr, disinflation_max=9, down_rate=0.7, up_rate=1.1, last_epoch=-1):
        self.base_lr = base_lr
        self.disinflation_max = disinflation_max
        self.disinflation_count = 0
        self.down_rate = down_rate
        self.up_rate = up_rate
        self.last_loss = 9999
        super(WaveLR, self).__init__(optimizer, last_epoch)

    def get_lr(self, loss):
        d_loss = loss - self.last_loss
        self.last_loss = loss
        if d_loss < 0:
            self.base_lr *= self.down_rate
            self.calm_count = 0
            return self.base_lr
        else:
            self.disinflation_count += 1
            if self.disinflation_count > self.disinflation_max:
                self.base_lr *= self.up_rate

        return self.base_lr
