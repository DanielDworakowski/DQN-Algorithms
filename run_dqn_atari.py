import dqn
import gym
import torch
import random
import argparse
import numpy as np
import deepMindModel
import os.path as osp
from dqn_utils import *
from gym import wrappers
from atari_wrappers import *
#
# Parse the input arguments.
def getInputArgs():
    parser = argparse.ArgumentParser('Configuration options.')
    parser.add_argument('--useTB', dest='useTB', default=False, action='store_true', help='Whether or not to log to Tesnor board.')

    args = parser.parse_args()
    return args

def atari_learn(env, num_timesteps, args):
    # This is just a rough estimate
    num_iterations = float(num_timesteps) / 4.0

    lr_multiplier = 1.0
    lr_schedule = PiecewiseSchedule([
                                         (0,                   1e-4 * lr_multiplier),
                                         (num_iterations / 10, 1e-4 * lr_multiplier),
                                         (num_iterations / 2,  5e-5 * lr_multiplier),
                                    ],
                                    outside_value=5e-5 * lr_multiplier)

    model = deepMindModel.atari_model(env.action_space.n)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_multiplier, betas=(0.9, 0.999), eps=1e-4, weight_decay=0)
    def sched(epoch):
        return lr_schedule.value(epoch)
    # schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda epoch: lr_schedule.value(epoch))
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = sched)

    def stopping_criterion(env, t):
        # notice that here t is the number of steps of the wrapped env,
        # which is different from the number of steps in the underlying env
        return get_wrapper_by_name(env, "Monitor").get_total_steps() >= num_timesteps

    exploration_schedule = PiecewiseSchedule(
        [
            (0, 1.0),
            (1e6, 0.1),
            (num_iterations / 2, 0.01),
        ], outside_value=0.01)

    dqn.learn(
        env,
        q_func = model,
        optimizer = optimizer,
        lr_schedule = schedule,
        lr_scheduler = lr_schedule,
        exploration = exploration_schedule,
        stopping_criterion = stopping_criterion,
        replay_buffer_size = 1000000,
        batch_size = 32,
        gamma = 0.99,
        learning_starts = 50000,
        learning_freq = 4,
        frame_history_len = 4,
        target_update_freq = 10000,
        grad_norm_clipping = 10,
        useTB = args.useTB
    )
    env.close()

def setRandomSeeds(seed, env):
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    env.seed(seed)

def configureEnv(env):

    expt_dir = '/tmp/hw3_vid_dir2/'
    env = wrappers.Monitor(env, osp.join(expt_dir, "gym"), force=True)
    env = wrap_deepmind(env)
    return env

def main():
    args = getInputArgs()
    #
    # Environment config
    env = gym.make('PongNoFrameskip-v0')
    setRandomSeeds(0, env)
    env = configureEnv(env)
    #
    # The learning fn.
    atari_learn(env, num_timesteps=2e7, args=args)

if __name__ == "__main__":
    main()
