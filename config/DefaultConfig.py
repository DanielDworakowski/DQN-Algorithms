import torch
import Objectives
import Exploration
import TensorConfig
from dqn_utils import *
from gym import wrappers
from ConfigureEnv import *
from atari_wrappers import *
from models import DeepMindModel

class DefaultConfig(object):

    def __init__(self, seed, objtype = Objectives.Objective_type.DQN_VANILLA):
        #
        # Whether to use TB.
        self.useTensorBoard = False
        #
        # Number of timsteps.
        self.num_timesteps=2e7
        num_iterations = float(self.num_timesteps) / 4.0
        #
        # Setup the environment.
        self.env = configureEnv(seed)
        #
        # Create the q_function model.
        self.q_func = DeepMindModel.atari_model(self.env.action_space.n)
        #
        # Create the optimizer.
        self.optimizer = torch.optim.Adam(self.q_func.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-4, weight_decay=0)
        #
        # Learning schedule.
        lr_multiplier = 1.0
        self.lr_schedule = PiecewiseSchedule([], outside_value=1 * lr_multiplier)
        #
        # Exploration Schedule.
        self.explorationSched = PiecewiseSchedule([
                                (0, 1.0),
                                (1e6, 0.1),
                                (num_iterations / 1.3, 0.02),
                            ], outside_value=0.02)
        #
        # Size of replay buffer.
        self.replaySize = int(1e6)
        #
        # History.
        self.frameHist = 4
        #
        # How to convert numpy to tensors (CUDA conversion and so on).
        self.tensorCfg = TensorConfig.getTensorConfiguration()
        #
        # Batch size for learning.
        self.batch_size = 32
        #
        # Discount.
        self.gamma = 0.99
        #
        # Gather data until.
        self.learning_starts = 50000
        #
        # How often to learn.
        self.learning_freq = 4
        #
        # When to switch targets.
        self.target_update_freq = 10000
        #
        # Maximum gradient size.
        self.grad_norm_clipping = 10
        #
        # Max steps in the env (different since env counts differently).
        self.maxSteps = 4e7
        #
        # Number of times to perform backprop / sample once learning starts (Parallel with replaybuffer.)
        self.nBprop = 1
        #
        # How often to print logs.
        self.logPeriod = 10000
        #
        # Objective function.
        self.objective = Objectives.Objective(self.tensorCfg, objtype)
        #
        # Exploration scheduler.
        def sched(epoch):
            return self.lr_schedule.value(epoch)
        self.schedule = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda = sched)
        #
        # Randomize.
        self.setRandomSeeds(seed)
    #
    # Explorer type (exploration policy).
    def getExplorer(self):
        #
        # Replay.
        replay_buffer = ReplayBuffer(self.replaySize, self.frameHist)
        explorer = Exploration.EpsilonGreedy(self.explorationSched, TensorConfig.TensorConfig(), replay_buffer, self.env, self.q_func, self.maxSteps)
        return explorer
    #
    # Set the random seeds.
    def setRandomSeeds(self, seed):
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
#
# Class to use the default configuration.
class Config(DefaultConfig):
    #
    # Initialize.
    def __init__(self, seed):
        super(Config, self).__init__(seed)