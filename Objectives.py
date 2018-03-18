import torch
import numpy as np
from enum import Enum, auto
from torch.autograd import Variable
#
# Enumeration of options.
class Objective_type(Enum):
    DQN_VANILLA = auto()
    DDQN = auto()

#
# Bellman error objective, standard Q learning.
class Objective(object):

    def __init__(self, tensorCfg, objtype = Objective_type.DQN_VANILLA):
        self.toTensorImg, self.toTensor, self.use_cuda = tensorCfg
        self.objective = self.bellmanError
        #
        # Target fn options.
        targetFnOpt = {
            Objective_type.DQN_VANILLA: self._dqn_target,
            Objective_type.DDQN: self._ddqn_target
        }
        self.targetFn = targetFnOpt[objtype]

    def _dqn_target(self, targetNet, trainNet, obs):
        targetQValid, _ = targetNet(obs).max(1)
        return targetQValid

    def _ddqn_target(self, targetNet, trainNet, obs):
        _, act_plus = trainNet(obs).max(1, keepdim=True)
        return torch.gather(targetNet(obs), 1, act_plus)

    def bellmanError(self, trainNet, targetNet, samples, gamma):
        obs_batch, act_batch, rew_batch, next_obs_batch, done_mask = samples
        #
        # Convert everything to be tensors, send to the GPU as needed.
        notDoneMask = (done_mask == False).astype(np.uint8)
        nextValidObs = torch.Tensor(0)
        rew = Variable(self.toTensor(rew_batch), requires_grad=False)
        act = Variable(self.toTensor(act_batch))
        obs = Variable(self.toTensorImg(obs_batch))
        expectedQ = Variable(self.toTensor(np.zeros((obs_batch.shape[0]), dtype=np.float32)), volatile=True)
        notDoneTensor = Variable(self.toTensor(notDoneMask))
        #
        # Forward through both networks.
        trainQ = torch.gather(trainNet(obs), 1, act.unsqueeze_(1))
        #
        # Forward the valid observations
        notDoneIdx = np.argwhere(notDoneMask).squeeze(1)
        if notDoneIdx.size > 0:
            nextValidObs = next_obs_batch.take(np.argwhere(notDoneMask).squeeze(1), axis=0)
            nextValidObs = Variable(self.toTensorImg(nextValidObs), volatile = True)
            #
            # Not all have a next observation -> some finished.
            expectedQ[notDoneTensor] = self.targetFn(targetNet, trainNet, nextValidObs)
        #
        # Calculate the belman error.
        expectedQ.volatile = False
        expectedQ = expectedQ.mul_(gamma) + rew
        return trainQ, expectedQ

    def __call__(self, *kwargs):
        return self.objective(*kwargs)