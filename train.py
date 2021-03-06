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
    parser.add_argument('--expName', dest='expName', default='', type=str, help='What to prefix names with on TB.')
    parser.add_argument('--config', dest='configStr', default='DefaultConfig', type=str, help='Name of the config file to import.')
    parser.add_argument('--env', dest='envName', default='PongNoFrameskip-v4', type=str, help='Name of the environment to run.')
    parser.add_argument('--seed', dest='seed', default=1, type=int, help='Random seed.')
    args = parser.parse_args()
    return args
#
# Get the configuration, override as needed.
def getConfig(args, expName):
    print(args.configStr)
    config_module = __import__('config.' + args.configStr)
    configuration = getattr(config_module, args.configStr)
    conf = configuration.Config(args.seed, args.envName, expName)
    #
    # Modifications to the configuration happen here.
    conf.useTensorBoard = args.useTB
    return conf
#
# Setup learning.
def doRL(conf):
    import dqn
    dqn.learn(conf)

def main():
    args = getInputArgs()
    #
    # Get configuration.
    conf = getConfig(args, args.expName)
    #
    # The learning fn.
    doRL(conf)

if __name__ == "__main__":
    mp.set_start_method('forkserver')
    main()
