import Exploration
from config.DefaultConfig import DefaultConfig 
#
# Class to use the default configuration.
class Config(DefaultConfig):
    #
    # Setup parallel configuration for sequential with no real replay buffer.
    def __init__(self, seed):
        super(Config, self).__init__(seed)
        self.parallelCfg = Exploration.ExploreParallelCfg()
        self.parallelCfg.model = self.q_func
        self.parallelCfg.exploreSched = self.explorationSched
        self.parallelCfg.numFramesPerBuffer = self.frameHist 
        self.parallelCfg.numEnv = 32
    # 
    # Override the explorer configuration.
    def getExplorer(self):
        explorer = Exploration.ParallelExplorer(self.parallelCfg)
        return explorer