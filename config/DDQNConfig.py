import Objectives
from config.DefaultConfig import DefaultConfig 
#
# Class to use the default configuration.
class Config(DefaultConfig):
    #
    # Initialize.
    def __init__(self, seed):
        super(Config, self).__init__(seed, Objectives.Objective_type.DDQN)