import selfplay_mcts
import dual_net
import preprocessing
from tensorflow import gfile

EXAMPLES_PER_RECORD = 10000
WINDOW_SIZE = 125000000
working_dir = './temp'

import os
import time

##########################

dual_net.bootstrap(working_dir)
