import os.path as osp
from gym import wrappers
from atari_wrappers import *

def configureEnv(seed, envName = 'PongNoFrameskip-v0', id='dqn'):
    expt_dir = '/tmp/%s/'%id
    env = gym.make(envName)
    env = wrappers.Monitor(env, osp.join(expt_dir, "gym"), force=True)
    env = wrap_deepmind(env)
    env.seed(seed)
    return env

class ConfigureEnv(object):
    getConfig = staticmethod(configureEnv)