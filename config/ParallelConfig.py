import Exploration
from config.DefaultConfig import DefaultConfig 
#
# Class to use the default configuration.
class Config(DefaultConfig):
    #
    # Initialize.
    def __init__(self, seed):
        super(Config, self).__init__(seed)
        self.parallelCfg = Exploration.ExploreParallelCfg()
        self.parallelCfg.model = model
        self.parallelCfg.exploreSched = explorationSched
        self.parallelCfg.numFramesInBuffer = args.replaySize
    # 
    # Override the explorer configuration.
    def getExplorer(self):
        explorer = Exploration.ParallelExplorer(self.parallelCfg)   
        return explorer