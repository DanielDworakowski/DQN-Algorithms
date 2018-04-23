#!/usr/bin/env python
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
#
# Parse the input arguments.
def getInputArgs():
    parser = argparse.ArgumentParser('Create visualization based on a network.')
    parser.add_argument('--runs', dest='runs', default='runFolder', type=str, help='Directory to the runs folder.')
    parser.add_argument('--smooth', dest='alpha', default=0, type=float, help='Exponential moving average smoothing alpha.')
    args = parser.parse_args()
    return args
#
# Exponential moving avg.
def numpy_ewma_vectorized_v2(data, alpha):
    alpha_rev = 1-alpha
    n = data.shape[0]
    pows = alpha_rev**(np.arange(n+1))

    scale_arr = 1/pows[:-1]
    offset = data[0]*pows[1:]
    pw0 = alpha*alpha_rev**(n-1)

    mult = data*pw0*scale_arr
    cumsums = mult.cumsum()
    out = offset + cumsums*scale_arr[::-1]
    return out
#
# Plot.
def gatherData(path, alpha):
    event_acc = EventAccumulator(path)
    event_acc.Reload()
    # Show all tags in the log file
    print(event_acc.Tags()['scalars'])
    if len(event_acc.Tags()['scalars']) == 0:
        return

    # E. g. get wall clock, number of steps and value for a scalar 'Accuracy'
    _, stp, bestMeanRwd = zip(*event_acc.Scalars('Best_mean_reward'))
    _, stp, mean100ep = zip(*event_acc.Scalars('Mean_reward__100_episodes_'))
    _, stp, loss = zip(*event_acc.Scalars('Train_loss'))
    bestMeanRwd = np.array(bestMeanRwd)
    mean100ep = np.array(mean100ep)
    loss = np.array(loss)
    stp = np.array(stp)
    print(path)
    # bestMeanRwd = numpy_ewma_vectorized_v2(bestMeanRwd, alpha)
    # loss = numpy_ewma_vectorized_v2(loss, alpha)
    # mean100ep = numpy_ewma_vectorized_v2(mean100ep, alpha)
    return (stp, bestMeanRwd, mean100ep, loss)
# 
# Make plots.
def makePlots(env, cfg, paths, args, ax):
    lossax, maxax, meanax = ax
    # 
    # Iterate and plot. 
    stps = []
    bestMeans = []
    mean100s = []
    losses = []
    maxSteps = -1
    maxIdx = -1
    for idx, path in enumerate(paths):
        stp, bestMeanRwd, mean100ep, loss = gatherData(path, args.alpha)
        stps.append(stp)
        bestMeans.append(bestMeanRwd)
        mean100s.append(mean100ep)
        losses.append(loss)
        if maxSteps < stp.shape[0]:
            maxSteps = stp.shape[0]
            maxIdx = idx
    # 
    # Pad everything.
    for idx in range(len(bestMeans)):
        padSize = maxSteps - bestMeans[idx].shape[0]
        bestMeans[idx] = np.pad(bestMeans[idx], (0, padSize), 'edge')
        mean100s[idx] = np.pad(mean100s[idx], (0, padSize), 'edge')
        losses[idx] = np.pad(losses[idx], (0, padSize), 'edge')
    # 
    # numpy each of them. 
    stps = np.array(stps[maxIdx])
    bestMeans = np.array(bestMeans)
    mean100s = np.array(mean100s)
    losses = np.array(losses)
    def getStats(data):
        maxs = np.max(data, axis=0)
        mins = np.min(data, axis=0)
        means = np.mean(data, axis=0)
        return maxs, mins, means
    bestStats = getStats(bestMeans)
    meanStats = getStats(mean100s)
    lossStats = getStats(losses)
    # 
    # Plot the plots.
    translator = {
        'DefaultConfig': 'DQN',
        'type.DDQN': 'DDQN',
        'QNEncodedConfig': 'Autoencoder Loss',
    }
    cfg = translator[cfg]
    maxax.plot(stps, bestStats[2], label = cfg)
    maxax.fill_between(stps, bestStats[1], bestStats[0],alpha=0.5)
    meanax.plot(stps, meanStats[2], label = cfg)
    meanax.fill_between(stps, meanStats[1], meanStats[0],alpha=0.5)
    lossax.plot(stps, lossStats[2], label = cfg)
    lossax.fill_between(stps, lossStats[1], lossStats[0],alpha=0.5)


#
# Iterate over directories.
def doPath(args):
    base = os.path.abspath(args.runs)   
    relatedPaths = defaultdict(lambda: defaultdict(list))
    for root, dirs, files in os.walk(base):
        dirs.sort()
        for direc in (dirs):
            # print(direc)
            env = direc[direc.find('_')+1:]
            env = env[:env.find('_')]
            cfg = direc[:direc.find('_seed')]
            cfg = cfg[cfg.rfind('_')+1:]
            # gatherData(root+'/'+direc, trainlax, trainAax, vallax, valAax, direc, args.alpha)
            relatedPaths[env][cfg].append(root+'/'+direc)
            # print(env)
    # print(relatedPaths)
    # 
    # Iterate and generate plots.
    for env in relatedPaths:
        loss = plt.figure()
        lossax = loss.gca()
        lossax.set_xlabel('Timestep')
        lossax.set_ylabel('Loss')
        lossax.set_title('Training loss: ' + env)
        maxReward = plt.figure()
        maxax = maxReward.gca()
        maxax.set_xlabel('Timestep')
        maxax.set_ylabel('Reward')
        maxax.set_title('Maximum Reward: ' + env)
        meanReward = plt.figure()
        meanax = meanReward.gca()
        meanax.set_xlabel('Timestep')
        meanax.set_ylabel('Loss')
        meanax.set_title('Mean Reward: ' + env)
        ax = (lossax, maxax, meanax)
        for cfg in relatedPaths[env]:
            # env, cfg = key
            paths = relatedPaths[env][cfg]
            makePlots(env, cfg, paths, args, ax)
        lossax.legend()
        maxax.legend()
        meanax.legend()
        loss.savefig(env + '_loss.png')
        maxReward.savefig(env + '_max.png')
        meanReward.savefig(env + '_mean.png')
    plt.show()
    # trainl.savefig('trainl.png')
    # vallax.legend(loc='lower left')
    # valAax.legend(loc='lower left')
    # trainlax.legend(loc='lower left')
    # trainAax.legend(loc='lower left')
    # trainA.savefig('trainA.png')
    # vall.savefig('vall.png')
    # valA.savefig('valA.png')

    # plt.show()


#
# main.
if __name__ == '__main__':
    args = getInputArgs()
    doPath(args)
