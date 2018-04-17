import torch
import numpy as np
from enum import Enum, auto
from torch.autograd import Variable
import torch.nn.functional as F

class LossL1(object):

    def __init__(self):
        pass

    def __call__(self, trainQ_func, trainQ, targetQ):
        loss = F.smooth_l1_loss(trainQ, targetQ)
        return loss

class LossAutoencoder(object):

    def __init__(self):
        self.mse_gain = 0.1

    def __call__(self, trainQ_func, trainQ, targetQ):
        loss = F.smooth_l1_loss(trainQ, targetQ)
        lossMSE = F.mse_loss(trainQ_func.reconst, trainQ_func.x)
        lossMSE /= lossMSE
        lossMSE *= loss * self.mse_gain
        loss += lossMSE
        return loss
