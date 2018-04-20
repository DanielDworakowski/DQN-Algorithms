import loss
import Objectives
from models import DeepMindModelEmbedding
from config.DefaultConfig import DefaultConfig
#
# Class to use the default configuration.
class Config(DefaultConfig):
    #
    # Initialize.
    def __init__(self, seed, envName, expName):
        super(Config, self).__init__(seed, envName=envName, cfg='QNEncodedConfig', expName=expName, model=DeepMindModelEmbedding.atari_model, objtype=Objectives.Objective_type.DQN_VANILLA)
        #
        # Create loss calculator.
        self.loss_calculator = loss.LossAutoencoder()