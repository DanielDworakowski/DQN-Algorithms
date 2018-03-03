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
        self.last_obs = env.reset()
        self.nAct = env.action_space.n
        self.exploreSched = exploreSched
        self.nSteps = 0
        # if self.use_cuda:
        #     model.cuda()

    def explore(self, t, *kwargs):
        #
        # Store the latest frame in the replay buffer.
        storeIndex = self.replay_buffer.store_frame(self.last_obs)
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
        #
        # Step and save transition.
        self.last_obs, reward, done, info = self.env.step(action)
        self.replay_buffer.store_effect(storeIndex, action, reward, done)
        #
        # Reset as needed.
        if done:
            self.last_obs = self.env.reset()

    def can_sample(self, batchSize):
        return self.replay_buffer.can_sample(batchSize)

    def sample(self, batchSize):
        return self.replay_buffer.sample(batchSize)

    def epsilon(self, t):
        return self.exploreSched.value(t)

    def shouldStop(self):
        return stopping_criterion(self.env) >= 2e7

    def numSteps(self):
        return self.nSteps

    def stepSize(self):
        return 1

class ExploreParallelCfg(object):
    numEnv = 4
    model = None
    exploreSched = None
    stackFrameLen = 4
    numFramesInBuffer = 1
    maxSteps = 2e7

class ExploreProcess(mp.Process):

    def __init__(self, coms, cfg, seed, name):
        super(ExploreProcess, self).__init__()
        self.com = coms
        self.model = cfg.model
        self.seed = seed
        self.name = str(name)
        self.cfg = cfg
        self.env = ConfigureEnv.configureEnv(self.seed, self.name)
        self.replay_buffer = ReplayBuffer(self.cfg.numFramesInBuffer // self.cfg.numEnv, self.cfg.stackFrameLen)
        print('Initialized process ', name)

    def run(self):
        self.explorer = EpsilonGreedy(self.cfg.exploreSched, TensorConfig.TensorConfig(), self.replay_buffer, self.env, self.model)
        while True:
            step = self.com.get()
            self.explorer.explore(step)

class ParallelExplorer(object):

    def __init__(self, cfg):
        cfg.model.cuda()
        cfg.model.share_memory()
        #
        # This must be set in the main.
        # mp.set_start_method('forkserver')
        # ctx = mp.get_context('forkserver')
        # self.manager = ctx.Manager()
        self.processes = []
        self.send = []
        self.curThread = 0
        self.nThreads = cfg.numEnv
        self.nInBuffers = 0
        self.totSteps = 0
        self.maxBuffers = cfg.numFramesInBuffer
        self.exploreSched = cfg.exploreSched
        # cfg.model.cuda()
        for idx in range(self.nThreads):
            print('Exploration: Actually set the seed properly.')
            sendP = mp.Queue()
            explorer = ExploreProcess(sendP, cfg, idx, idx)
            explorer.start()
            self.processes.append(explorer)
            self.send.append(sendP)

    def explore(self, curStep, nStep):
        #
        # Iterate and get n more examples in each of the threads.
        for idx in range(nStep):
            #
            # Notify a thread that it should do work.
            self.send[self.curThread].put(curStep)
            self.curThread = (self.curThread + 1) % self.nThreads
            curStep += 1
        self.totSteps += nStep
        self.nInBuffers = min(self.totSteps, self.maxBuffers)

    def close(self):
        for proc in self.processes:
            proc.terminate()
            proc.join()

    def can_sample(self, batchSize):
        # return self.nInBuffers > batchSize
        return self.processes[0].replay_buffer.can_sample(batchSize)

    def sample(self, batchSize):
        return self.processes[0].replay_buffer.sample(batchSize)

    def epsilon(self, t):
        return self.exploreSched.value(t)

    def shouldStop(self):
        return stopping_criterion(self.processes[0].env) >= 2e7

    def numSteps(self):
        return self.totSteps

    def stepSize(self):
        return self.nThreads