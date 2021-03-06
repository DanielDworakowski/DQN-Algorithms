import Exploration
from config.DefaultConfig import DefaultConfig
#
# Class to use the default configuration.
class Config(DefaultConfig):
    #
    # Setup parallel configuration for sequential with no real replay buffer.
    def __init__(self, seed, envName, expName):
        super(Config, self).__init__(seed, envName, expName=expName)
        self.parallelCfg = Exploration.ExploreParallelCfg()
        self.parallelCfg.model = self.q_func
        self.parallelCfg.exploreSched = self.explorationSched
        self.parallelCfg.numFramesPerBuffer = self.frameHist + 1
        self.parallelCfg.sampleLatest = True
        self.parallelCfg.numEnv = 32
        self.batch_size = self.parallelCfg.numEnv
        self.logPeriod = int(self.parallelCfg.numEnv * 100)
        #
        # Dont need to wait since we are going sequentially, but allow for some randomness.
        self.learning_starts = 50
        self.learning_freq = 1
        self.epsilonStepSize = 1 # To match single step learning frequency.
    #
    # Override the explorer configuration.
    def getExplorer(self):
        explorer = Exploration.ParallelExplorer(self.parallelCfg)
        return explorer