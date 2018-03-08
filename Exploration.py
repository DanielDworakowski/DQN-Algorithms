import torch
import random
import numpy as np
import ConfigureEnv
import TensorConfig
import torch.multiprocessing as mp
from dqn_utils import ReplayBuffer
from torch.autograd import Variable
from dqn_utils import get_wrapper_by_name
#
# When to stop
def stopping_criterion(env):
    # notice that here t is the number of steps of the wrapped env,
    # which is different from the number of steps in the underlying env
    return get_wrapper_by_name(env, "Monitor").get_total_steps()

class EpsilonGreedy(object):

    def __init__(self, exploreSched, tensorCfg, replay, env, model):
        self.toTensorImg, self.toTensor, self.use_cuda = tensorCfg.getConfig()
        self.replay_buffer = replay
        self.env = env
        self.model = model
        self.lastObs = env.reset()
        self.nAct = env.action_space.n
        self.exploreSched = exploreSched
        self.nSteps = 0
        # if self.use_cuda:
        #     model.cuda()

    def explore_nobuffer(self, t):
        self.nSteps += 1
        #
        # Epsilon greedy exploration.
        action = None
        if random.random() < self.exploreSched.value(t):
            action = np.random.randint(0, self.nAct, dtype=np.int_)
        else:
            obs = self.toTensorImg(np.expand_dims(self.replay_buffer.encode_recent_observation(), axis=0))
            #
            # Forward through network.
            _, action = self.model(Variable(obs, volatile=True)).max(1)
            # _, action = targetQ_func(Variable(obs, volatile=True)).max(1)
            action = action.data.cpu().numpy().astype(np.int_)
        self.lastObs, reward, done, info = self.env.step(action)
        #
        # Step and save transition.
        if done:
            self.lastObs = self.env.reset()
        return (self.lastObs, reward, done, info, action)

    def explore(self, t, *kwargs):
        #
        # Store the latest frame in the replay buffer.
        storeIndex = self.replay_buffer.store_frame(self.lastObs)
        self.lastObs, reward, done, info, action = self.explore_nobuffer(t)
        self.replay_buffer.store_effect(storeIndex, action, reward, done)

    def can_sample(self, batchSize):
        return self.replay_buffer.can_sample(batchSize)

    def sample(self, batchSize):
        return self.replay_buffer.sample(batchSize)

    def epsilon(self):
        return self.exploreSched.value(self.nSteps)

    def shouldStop(self):
        return stopping_criterion(self.env) >= 2e7

    def numSteps(self):
        return self.nSteps

    def stepSize(self):
        return 1

    def getRewards(self):
        # episode_rewards = get_wrapper_by_name(env, "Monitor").get_episode_rewards()
        return np.mean(get_wrapper_by_name(self.env, "Monitor").get_episode_rewards()[-100:])

    def getNumEps(self):
        return len(get_wrapper_by_name(self.env, "Monitor").get_episode_rewards())


class ExploreParallelCfg(object):
    numEnv = 10
    model = None
    exploreSched = None
    stackFrameLen = 4
    numFramesInBuffer = 1
    maxSteps = 2e7

class ExploreProcess(mp.Process):

    def __init__(self, coms, cfg, seed, procId, actionVec):
        super(ExploreProcess, self).__init__()
        self.com = coms
        self.model = cfg.model
        self.seed = seed
        self.procId = procId
        self.lastObs = None
        self.cfg = cfg
        self.env = ConfigureEnv.configureEnv(self.seed, 'dqn_' + str(procId))
        # self.replay_buffer = ReplayBuffer(self.cfg.stackFrameLen, self.cfg.stackFrameLen)
        # frameSize = self.env.observation_space.shape
        # self.stackedFrames = torch.ByteTensor(1, frameSize[0], frameSize[1], frameSize[2] * self.cfg.stackFrameLen)
        # self.stackedFrames.storage().share_memory_()
        #
        # Shared memory to transfer the action commands.
        self.actionVec = actionVec
        print('Initialized process ', procId)

    def run(self):
        #
        # For the first run, just setup a random action.
        self.lastObs = self.env.reset()
        obs, reward, done, info = self.env.step(0)
        self.com.send((self.lastObs, reward, done, info, 0, 0, 0))
        self.lastObs = obs
        #
        # Loop and do work.
        while True:
            #
            # Wait for actions.
            step = self.com.recv()
            action = self.actionVec.numpy().astype(np.int_)[self.procId]
            obs, reward, done, info = self.env.step(action)
            #
            # Step and save transition.
            if done:
                obs = self.env.reset()
            #
            # Store effects.
            lastRew = get_wrapper_by_name(self.env, "Monitor").get_episode_rewards()
            mean_episode_reward = 0
            if (len(lastRew) > 0):
                mean_episode_reward = np.mean(lastRew[-100:])
            self.com.send((self.lastObs, reward, done, info, action, mean_episode_reward, len(lastRew)))
            self.lastObs = obs

class ParallelExplorer(object):

    def __init__(self, cfg):
        #
        # This must be set in the main.
        # mp.set_start_method('forkserver')
        self.processes = []
        self.comms = []
        self.followup = []
        self.replayBuffers = []
        self.curThread = 0
        self.nThreads = cfg.numEnv
        self.meanRewards = [0] * self.nThreads
        self.numEps = [0] * self.nThreads
        self.nInBuffers = 0
        self.totSteps = 0
        self.maxBuffers = cfg.numFramesInBuffer
        self.exploreSched = cfg.exploreSched
        self.model = cfg.model
        self.actionVec = torch.LongTensor(self.nThreads)
        self.actionVec.storage().share_memory_()
        self.threads = np.arange(self.nThreads, dtype=np.int_)
        self.toTensorImg, self.toTensor, self.use_cuda = TensorConfig.getTensorConfiguration()
        # cfg.model.cuda()
        for idx in range(self.nThreads):
            print('Exploration: Actually set the seed properly.')
            sendP, subpipe = mp.Pipe()
            explorer = ExploreProcess(subpipe, cfg, idx, idx, self.actionVec)
            explorer.start()
            self.processes.append(explorer)
            self.comms.append(sendP)
            self.replayBuffers.append(ReplayBuffer(cfg.numFramesInBuffer // cfg.numEnv, cfg.stackFrameLen))
            self.followup.append(idx)
        self.nAct = self.processes[0].env.action_space.n

    def __del__(self):
        for proc in self.processes:
            proc.terminate()
            proc.join()

    def explore(self, nStep):
        #
        # Can only do at most nThreads steps at once.
        assert nStep <= self.nThreads
        #
        # Gather the responses from each.
        for pipeIdx in self.followup:
            ret = self.comms[pipeIdx].recv()
            obs, reward, done, info, action, meanReward, nEp = ret
            self.meanRewards[pipeIdx] = meanReward
            self.numEps[pipeIdx] = nEp
            storeIndex = self.replayBuffers[pipeIdx].store_frame(obs)
            self.replayBuffers[pipeIdx].store_effect(storeIndex, action, reward, done)
        #
        # We have finished following up.
        self.followup = []
        #
        # Keep track of the effective number of steps.
        curStep = self.totSteps
        self.totSteps += nStep
        #
        # Select each of the threads to use.
        thSelect = torch.from_numpy(np.random.choice(self.threads, nStep, replace=False))
        exploration = torch.from_numpy(np.random.uniform(size=nStep))
        randomActions = torch.from_numpy(np.random.randint(0, self.nAct, size=nStep, dtype=np.int_))
        self.actionVec.copy_(randomActions)
        runNetIdx = torch.from_numpy(self.threads[thSelect][exploration >= self.exploreSched.value(curStep)])
        obsList = []
        #
        # Ensure that we actually even want to do anything.
        if runNetIdx.shape[0] > 0:
            #
            # Build the batch of images.
            for idx in runNetIdx:
                obsList.append(self.replayBuffers[idx].encode_recent_observation())
            #
            # Forward through the network.
            obsStack = Variable(self.toTensorImg(np.array(obsList)), volatile=True)
            _, actions = self.model(obsStack).max(1)
            self.actionVec.scatter_(0, runNetIdx, actions.data.cpu())
        #
        # Notify all workers.
        for idx in thSelect:
            self.comms[idx].send(curStep)
            self.followup.append(idx)

    def close(self):
        for proc in self.processes:
            proc.terminate()
            proc.join()

    def can_sample(self, batchSize):
        #
        # Ensure that all can sample.
        ret = True
        for buf in self.replayBuffers:
            ret = ret and buf.can_sample(batchSize)
        return ret

    def sample(self, batchSize):
        bufferSamples = batchSize // self.nThreads
        extra = batchSize - self.nThreads * bufferSamples
        extraBuff = np.zeros(self.nThreads, dtype=np.int8)
        addList = np.random.choice(self.nThreads, replace=False)
        extraBuff[addList] = 1
        samplelist = []
        for threadIdx in range(self.nThreads):
            threadBatch = bufferSamples + extraBuff[threadIdx]
            samplelist.append(self.replayBuffers[threadIdx].sample(threadBatch))
        obs_batch, act_batch, rew_batch, next_obs_batch, done_mask = zip(*samplelist)
        obs_batch = np.concatenate(obs_batch)
        act_batch = np.concatenate(act_batch)
        rew_batch = np.concatenate(rew_batch)
        next_obs_batch = np.concatenate(next_obs_batch)
        done_mask = np.concatenate(done_mask)
        return obs_batch, act_batch, rew_batch, next_obs_batch, done_mask

    def epsilon(self):
        return self.exploreSched.value(self.totSteps)

    def shouldStop(self):
        return stopping_criterion(self.processes[0].env) >= 2e7

    def numSteps(self):
        return self.totSteps

    def stepSize(self):
        return self.nThreads

    def getRewards(self):
        # episode_rewards = get_wrapper_by_name(env, "Monitor").get_episode_rewards()
        return np.mean(np.array(self.meanRewards))

    def getNumEps(self):
        return np.mean(np.array(self.numEps))