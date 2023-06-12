import math
from copy import deepcopy
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts, CyclicLR


class WarmupCosineStepSchedule(LambdaLR):
    """ Linear warmup and then cosine decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps`.
        Decreases learning rate from 1. to 0. over remaining
            `total_steps - warmup_steps` steps following a cosine curve.
        If `cycles` (default=0.5) is different from default, learning rate
            follows cosine function after warmup.
    """

    def __init__(self, optimizer, warmup_steps, total_steps, cycles=.5, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.cycles = cycles
        super(WarmupCosineStepSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        # progress after warmup
        progress = float(step - self.warmup_steps) / float(max(1, self.total_steps - self.warmup_steps))
        return max(0.0, 0.5 * (1. + math.cos(math.pi * float(self.cycles) * 2.0 * progress)))
