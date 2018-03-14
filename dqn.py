import sys
import copy
import random
import itertools
import gym.spaces
import Objectives
import Exploration
import numpy as np
import TensorConfig
from tqdm import tqdm
from dqn_utils import *
import torch.nn.functional as F
from collections import namedtuple
from torch.autograd import Variable
from tensorboardX import SummaryWriter
import torchvision.transforms as transforms
#
# Visualize a batch.
def visobs(obs):
    from PIL import Image
    obsBatch = obs.data.cpu().numpy()
    for i, obs in enumerate(obsBatch):
        for plane in obs:
            img = Image.fromarray(plane*255)
            img.show()
        if i == 5:
            sys.exit(0)
#
# Logging configuration.
def logEpochTensorboard(logger, model, epochSummary, t):
    # logger.add_scalar('%s_loss'%epochSummary['phase'], epochSummary['loss'], epochSummary['epoch'])
    # logger.add_scalar('%s_acc'%epochSummary['phase'], epochSummary['acc'], epochSummary['epoch'])
    # labels = epochSummary['data']['label']
    # for i in range(epochSummary['data']['label'].shape[0]):
    #     logger.add_image('{}_image_i-{}_epoch-{}_pre-:{}_label-{}'.format(epochSummary['phase'], i, epochSummary['epoch'], epochSummary['pred'][i], int(labels[i])), epochSummary['data']['img'][i]*math.sqrt(0.06342617) + 0.59008044, epochSummary['epoch'])
    for key in epochSummary:
        logger.add_scalar(key, epochSummary[key], t)
    for name, param in model.named_parameters():
        logger.add_histogram(name, param.clone().cpu().data.numpy(), t)
#
# Write everything as needed.
def closeTensorboard(logger):
    logger.close()
#
# When not using tensor board.
def doNothing(logger = None, model = None, tmp = None, tmp1 = None):
    pass
#
# Training fn.
def learn(env,
          q_func,
          optimizer,
          lr_schedule,
          explorer,
          tensorCfg,
          batch_size=32,
          gamma=0.99,
          learning_starts=50000,
          learning_freq=4,
          target_update_freq=10000,
          grad_norm_clipping=10,
          useTB=False):

    assert type(env.observation_space) == gym.spaces.Box
    assert type(env.action_space)      == gym.spaces.Discrete
    #
    # Environment information.
    nAct = env.action_space.n
    #
    # Information for tensor configuration.
    toTensorImg, toTensor, use_cuda = tensorCfg
    #
    # Logging setup.
    logger = None
    logEpoch = doNothing
    closeLogger = doNothing
    LOG_EVERY_N_STEPS = 10000
    PROGRESS_UPDATE_FREQ = 100
    if useTB:
        logger = SummaryWriter()
        logEpoch = logEpochTensorboard
        closeLogger = closeTensorboard
    #
    # Construct support objects for learning.
    num_param_updates = 0
    mean_episode_reward = -float('nan')
    best_mean_episode_reward = -float('inf')
    trainQ_func = q_func
    targetQ_func = copy.deepcopy(trainQ_func).eval()
    objective = Objectives.Objective(tensorCfg)
    runningLoss = 0
    #
    # Send networks to CUDA.
    if use_cuda:
        trainQ_func.cuda()
        targetQ_func.cuda()
    #
    # Training loop.
    pbar = None
    for t in itertools.count():
        #
        # Check if we are done.
        if explorer.shouldStop():
            break
        #
        # Exploration.
        # explorer.explore(t, trainQ_func)
        explorer.explore(explorer.stepSize())
        #
        # Learning gating.
        if (explorer.numSteps() > learning_starts and t % learning_freq == 0 and explorer.can_sample(batch_size)):
        # if (explorer.numSteps() > learning_starts and t % 1 == 0 and explorer.can_sample(batch_size)):
            #
            # Update as many times as we would have updated if everything was serial.
            for i in range(explorer.stepSize()):
            # for i in range(1):
                # print(explorer.stepSize())
                #
                # Sample from replay buffer.
                sample = explorer.sample(batch_size)
                #
                # Get the objective information (bellman eq).
                trainQ, targetQ = objective(trainQ_func, targetQ_func, sample, gamma)
                #
                # Calculate Huber loss.
                loss = F.smooth_l1_loss(trainQ, targetQ)
                runningLoss += loss.data[0]
                #
                # Optimize the model.
                optimizer.zero_grad()
                loss.backward()
                #
                # Clip the gradient.
                # torch.nn.utils.clip_grad_norm(trainQ_func.parameters(), grad_norm_clipping)
                optimizer.step()
                lr_schedule.step(t)
                num_param_updates += 1
                #
                # Update the target network as needed (target_update_freq).
                if num_param_updates % target_update_freq == 0:
                    targetQ_func.load_state_dict(trainQ_func.state_dict())
                    targetQ_func.eval()
                    if use_cuda:
                        targetQ_func.cuda()
        #
        # Logging.
        mean_episode_reward = explorer.getRewards()
        best_mean_episode_reward = max(best_mean_episode_reward, mean_episode_reward)
        if t % (LOG_EVERY_N_STEPS // explorer.stepSize()) == 0:
            print("Timestep %d" % (t,))
            print("mean reward (100 episodes) %f" % mean_episode_reward)
            print("best mean reward %f" % best_mean_episode_reward)
            print("episodes %d" % explorer.getNumEps())
            print("Exploration %f" % explorer.epsilon())
            print("learning_rate ", lr_schedule.get_lr())
            if pbar is not None:
                pbar.close()
            sys.stdout.flush()
            pbar = tqdm(total=LOG_EVERY_N_STEPS)
            summary = {
                'Mean reward (100 episodes)': mean_episode_reward,
                'Best mean reward': best_mean_episode_reward,
                'Episodes': explorer.getNumEps(),
                'Learning rate': lr_schedule.get_lr()[0],
                'Exploration': explorer.epsilon(),
                'Train loss': runningLoss / LOG_EVERY_N_STEPS,
            }
            logEpoch(logger, trainQ_func, summary, t)
            runningLoss = 0
        if t % PROGRESS_UPDATE_FREQ == 0:
            pbar.update(PROGRESS_UPDATE_FREQ * explorer.stepSize())
    #
    # Close logging (TB))
    closeLogger(logger)