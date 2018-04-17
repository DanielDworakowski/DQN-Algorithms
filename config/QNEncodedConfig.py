import Objectives
import loss
from models import DeepMindModelEmbedding
from config.DefaultConfig import DefaultConfig
#
# Class to use the default configuration.
class Config(DefaultConfig):
    #
    # Initialize.
    def __init__(self, seed, expName):
        super(Config, self).__init__(seed, Objectives.Objective_type.DQN_VANILLA, expName=expName, model=DeepMindModelEmbedding.atari_model)
        #
        # Create loss calculator.
        self.loss_calculator = loss.LossAutoencoder()