import random 
import numpy as np
from torch.autograd import Variable

class EpsilonGreedy(object):

    def __init__(self, exploreSched, tensorCfg, replay, env):
        self.toTensorImg, self.toTensor, self.use_cuda = tensorCfg
        self.replay_buffer = replay
        self.env = env
        self.last_obs = env.reset()
        self.nAct = env.action_space.n
        self.exploreSched = exploreSched

    def explore(self, t, qf):
        #
        # Store the latest frame in the replay buffer.
        storeIndex = self.replay_buffer.store_frame(self.last_obs)
        #
        # Epsilon greedy exploration.
        action = None
        if random.random() < self.exploreSched.value(t):
            action = np.random.randint(0, self.nAct, dtype=np.int_)
        else:
            obs = self.toTensorImg(np.expand_dims(self.replay_buffer.encode_recent_observation(), axis=0))
            #
            # Forward through network.
            _, action = qf(Variable(obs, volatile=True)).max(1)
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
