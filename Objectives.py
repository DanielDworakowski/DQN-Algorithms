import torch
import numpy as np
from torch.autograd import Variable
#
# Bellman error objective, standard Q learning.
class Objective(object):

    def __init__(self, tensorCfg):
        self.toTensorImg, self.toTensor, self.use_cuda = tensorCfg
        self.objective = self.bellmanError

    def bellmanError(self, trainNet, targetNet, samples, gamma):
        obs_batch, act_batch, rew_batch, next_obs_batch, done_mask = samples
        #
        # Convert everything to be tensors, send to the GPU as needed.
        notDoneMask = (done_mask == False).astype(np.uint8)
        nextValidObs = next_obs_batch.compress(notDoneMask, axis=0)
        act = Variable(self.toTensor(act_batch))
        rew = Variable(self.toTensor(rew_batch))
        obs = Variable(self.toTensorImg(obs_batch))
        expectedQ = Variable(self.toTensor(np.zeros((obs_batch.shape[0]), dtype=np.float32)))
        nextValidObs = Variable(self.toTensorImg(nextValidObs), volatile = True)
        notDoneTensor = Variable(self.toTensor(notDoneMask))
        #
        # Forward through both networks.
        trainQ = torch.gather(trainNet(obs), 1, act.unsqueeze_(1))
        #
        # Not all have a next observation -> some finished.
        targetQValid, _ = targetNet(nextValidObs).max(1)
        expectedQ[notDoneTensor] = targetQValid
        #
        # Calculate the belman error.
        expectedQ.volatile = False
        expectedQ = expectedQ.mul_(gamma) + rew
        return trainQ, expectedQ

    def __call__(self, *kwargs):
        return self.objective(*kwargs)