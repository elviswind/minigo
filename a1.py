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
files = gfile.Glob('*.index')
mx = 0
file = ''
for f in files:
    t = os.path.getmtime(f)
    if t > mx:
        mx = t
        file = f

readouts = 400
network = dual_net.DualNetwork(file.split('.index')[0])
for i in range(50):
    player = selfplay_mcts.play(network, readouts, -1.0, 1)
    game_data = player.extract_data()
    tf_examples = preprocessing.make_dataset_from_selfplay(game_data)
    preprocessing.write_tf_examples(working_dir + '/selfplay-' + str(time.time()) + '.tfrecord.zz', tf_examples)
