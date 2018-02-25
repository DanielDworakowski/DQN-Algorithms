import dqn
import gym
import torch
import random
import argparse
import numpy as np
import os.path as osp
from dqn_utils import *
from gym import wrappers
from atari_wrappers import *

def atari_learn(env, session, num_timesteps):
    # This is just a rough estimate
    num_iterations = float(num_timesteps) / 4.0

    lr_multiplier = 1.0
    lr_schedule = PiecewiseSchedule([
                                         (0,                   1e-4 * lr_multiplier),
                                         (num_iterations / 10, 1e-4 * lr_multiplier),
                                         (num_iterations / 2,  5e-5 * lr_multiplier),
                                    ],
                                    outside_value=5e-5 * lr_multiplier)
    
    model = atari_model(env.action_space.n)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_multiplier, betas=(0.9, 0.999), eps=1e-4, weight_decay=0)
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda epoch: lr_schedule(epoch))

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
        q_func = atari_model,
        optimizer_spec = optimizer,
        session = session,
        exploration = exploration_schedule,
        stopping_criterion = stopping_criterion,
        replay_buffer_size = 1000000,
        batch_size = 32,
        gamma = 0.99,
        learning_starts = 50000,
        learning_freq = 4,
        frame_history_len = 4,
        target_update_freq = 10000,
        grad_norm_clipping = 10
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

    # Run training 
    specs = gym.envs.registry
    env = gym.make('Pong-v0')
    setRandomSeeds(0, env)
    configureEnv(env)
    # env = get_env(task, seed)
    # tmp = specs.env_specs['Pong-v0'].timestep_limit
    atari_learn(env, session, num_timesteps=1e7)

if __name__ == "__main__":
    main()
