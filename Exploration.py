import random 
import numpy as np
from dqn_utils import ReplayBuffer
from torch.autograd import Variable
from dqn_utils import get_wrapper_by_name
from torch.multiprocessing import Process, Pipe, Condition

# 
# When to stop
def stopping_criterion(env):
    # notice that here t is the number of steps of the wrapped env,
    # which is different from the number of steps in the underlying env
    return get_wrapper_by_name(env, "Monitor").get_total_steps()

class EpsilonGreedy(object):

    def __init__(self, exploreSched, tensorCfg, replay, env, model):
        self.toTensorImg, self.toTensor, self.use_cuda = tensorCfg
        self.replay_buffer = replay
        self.env = env
        self.model = model
        self.last_obs = env.reset()
        self.nAct = env.action_space.n
        self.exploreSched = exploreSched
        self.nSteps = 0

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
    envCfg = None
    numEnv = 4
    model = None
    exploreSched = None
    tensorCfg = None
    stackFrameLen = 4
    numFramesInBuffer = 1
    maxSteps = 2e7

class ExploreProcess(Process):

    def __init__(self, coms, cfg, seed, name):
        super(ExploreProcess, self).__init__()
        self.env = cfg.envCfg(seed, name)
        self.com = coms
        self.model = cfg.model
        self.replay_buffer = ReplayBuffer(cfg.numFramesInBuffer // cfg.numEnv, cfg.stackFrameLen)
        self.explorer = EpsilonGreedy(cfg.exploreSched, cfg.tensorCfg, self.replay_buffer, self.env, self.model)
        print('Initialized process ', name)

    def run(self):
        while True:
            step = self.com.recv()
            self.explorer.explore(step)

class ParallelExplorer(object):

    def __init__(self, cfg):
        cfg.model.share_memory()
        self.processes = []
        self.send = []
        self.curThread = 0
        self.nThreads = cfg.numEnv
        self.nInBuffers = 0
        self.totSteps = 0
        self.maxBuffers = cfg.numFramesInBuffer
        self.exploreSched = cfg.exploreSched
        for idx in range(self.nThreads):
            print('Exploration: Actually set the seed properly.')
            sendP, envEnd = Pipe()
            explorer = ExploreProcess(envEnd, cfg, idx, idx)
            explorer.start()
            self.processes.append(explorer)
            self.send.append(sendP)

    def explore(self, curStep, nStep):
        # 
        # Iterate and get n more examples in each of the threads. 
        for idx in range(nStep):
            # 
            # Notify a thread that it should do work.
            self.send[self.curThread].send(curStep)
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