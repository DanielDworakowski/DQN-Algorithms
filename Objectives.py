import torch
from torch.autograd import Variable

def bellmanError(trainNet, targetNet, samples, gamma):
    obs_batch, act_batch, rew_batch, next_obs_batch, done_mask = samples
    #
    # Convert everything to be tensors, send to the GPU as needed.
    notDoneMask = (done_mask == False).astype(np.uint8)
    nextValidObs = next_obs_batch.compress(notDoneMask, axis=0)
    act = Variable(toTensor(act_batch))
    rew = Variable(toTensor(rew_batch))
    obs = Variable(toTensorImg(obs_batch))
    expectedQ = Variable(toTensor(np.zeros((batch_size), dtype=np.float32)))
    nextValidObs = Variable(toTensorImg(nextValidObs), volatile = True)
    notDoneTensor = Variable(toTensor(notDoneMask))
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