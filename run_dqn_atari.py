import argparse
import ConfigureEnv
import os.path as osp
from dqn_utils import *
from gym import wrappers
from ConfigureEnv import *
from atari_wrappers import *
import torch.multiprocessing as mp
#
# Parse the input arguments.
def getInputArgs():
    parser = argparse.ArgumentParser('Configuration options.')
    parser.add_argument('--useTB', dest='useTB', default=False, action='store_true', help='Whether or not to log to Tesnor board.')
    parser.add_argument('--replaySize', dest='replaySize', default=1000000, help='Whether or not to log to Tesnor board.')
    parser.add_argument('--frame_history_len', dest='frameHistLen', default=4, help='Whether or not to log to Tesnor board.')

    args = parser.parse_args()
    return args

def atari_learn(num_timesteps, args):
    import dqn
    import gym
    import torch
    import random
    import Objectives
    import numpy as np
    import Exploration
    import TensorConfig
    import deepMindModel
    #
    # This is just a rough estimate
    num_iterations = float(num_timesteps) / 4.0

    lr_multiplier = 1.0
    lr_schedule = PiecewiseSchedule([
                                         (0,                   1e-4 * lr_multiplier),
                                         (num_iterations / 10, 1e-4 * lr_multiplier),
                                         (num_iterations / 2,  5e-5 * lr_multiplier),
                                    ],
                                    outside_value=5e-5 * lr_multiplier)
    lr_schedule = PiecewiseSchedule([
                                    ],
                                    outside_value=1 * lr_multiplier)

    explorationSched = PiecewiseSchedule(
        [
            (0, 1.0),
            (1e6, 0.1),
            (num_iterations / 2, 0.01),
        ], outside_value=0.01)
    #
    # Environment config
    seed = 0
    replay_buffer = ReplayBuffer(args.replaySize, args.frameHistLen)
    tensorCfg = TensorConfig.getTensorConfiguration()
    env = configureEnv(seed)
    model = deepMindModel.atari_model(env.action_space.n)
    explorer = Exploration.EpsilonGreedy(explorationSched, TensorConfig.TensorConfig(), replay_buffer, env, model)
    # parallelCfg = Exploration.ExploreParallelCfg()
    # parallelCfg.model = model
    # parallelCfg.exploreSched = explorationSched
    # parallelCfg.numFramesInBuffer = args.replaySize
    # explorer = Exploration.ParallelExplorer(parallelCfg)
    print('Set seeds!')
    setRandomSeeds(seed)
    #
    # Create the model.
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-4, weight_decay=0)
    #
    # Exploration scheduler.
    def sched(epoch):
        return lr_schedule.value(epoch)
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = sched)
    #
    # Learn.
    dqn.learn(
        env,
        q_func = model,
        optimizer = optimizer,
        lr_schedule = schedule,
        explorer = explorer,
        tensorCfg = tensorCfg,
        batch_size = 32,
        gamma = 0.99,
        learning_starts = 50000,
        learning_freq = 4,
        target_update_freq = 10000,
        grad_norm_clipping = 10,
        useTB = args.useTB
    )
    env.close()

def setRandomSeeds(seed):
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def main():
    args = getInputArgs()
    #
    # The learning fn.
    atari_learn(num_timesteps=2e7, args=args)

if __name__ == "__main__":
    mp.set_start_method('forkserver')
    main()
