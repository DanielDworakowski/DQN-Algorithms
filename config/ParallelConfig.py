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
        self.parallelCfg.sampleLatest = True
        self.parallelCfg.numEnv = 32
        #
        # Dont need to wait since we are going sequentially, but allow for some randomness.  
        self.learning_starts = 500
    # 
    # Override the explorer configuration.
    def getExplorer(self):
        explorer = Exploration.ParallelExplorer(self.parallelCfg)
        return explorer