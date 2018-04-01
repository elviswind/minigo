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
record_files = gfile.Glob(working_dir + '/selfplay-*.tfrecord.zz')
for i, example_batch in enumerate(preprocessing.shuffle_tf_examples(EXAMPLES_PER_RECORD, record_files)):
    print(i)
    output_record = working_dir + '/gathered-{}.tfrecord.zz'.format(str(i))
    preprocessing.write_tf_examples(
        output_record, example_batch, serialize=False)

tf_records = sorted(gfile.Glob(working_dir + '/gathered-*.tfrecord.zz'))
tf_records = tf_records[-1 * (WINDOW_SIZE // EXAMPLES_PER_RECORD):]

print("Training from:", tf_records[0], "to", tf_records[-1])
dual_net.train(working_dir, tf_records, 2)