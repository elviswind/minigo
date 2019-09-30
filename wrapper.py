import tensorflow as tf 
from . import dao 
from . import shipname
import os 
from . import dual_net
from . import utils

from absl import flags
import sys
flags.FLAGS(['--base_dir','.'])

MAX_GAMES_PER_GENERATION = 20000
WINDOW_SIZE = 125000000
EXAMPLES_PER_RECORD = 10000

def bootstrap(working_dir):
    bootstrap_name = shipname.generate(0)
    bootstrap_model_path = os.path.join('models', bootstrap_name)
    print("Bootstrapping with working dir {}\n Model 0 exported to {}".format(
        working_dir, bootstrap_model_path))

    utils.ensure_dir_exists(working_dir)
    utils.ensure_dir_exists(bootstrap_model_path)
    dual_net.bootstrap(working_dir)
    dual_net.export_model(working_dir, bootstrap_model_path)
    freeze_graph(bootstrap_model_path)

def freeze_graph(load_file):
    """ Loads a network and serializes just the inference parts for use by e.g. the C++ binary """
    n = dual_net.DualNetwork(load_file)
    out_graph = tf.graph_util.convert_variables_to_constants(
        n.sess, n.sess.graph.as_graph_def(), ["policy_output", "value_output"])
    with tf.gfile.GFile(os.path.join(load_file + '.pb'), 'wb') as f:
        f.write(out_graph.SerializeToString())

def get_models():
    all_models = tf.gfile.Glob(os.path.join('models', '*.meta'))
    model_filenames = [os.path.basename(m) for m in all_models]
    model_numbers_names = sorted([
        (shipname.detect_model_num(m), shipname.detect_model_name(m))
        for m in model_filenames])
    return model_numbers_names    
    
def selfplay(verbose=2):
    _, model_name = get_models()[-1]
    utils.ensure_dir_exists(os.path.join('data', model_name))
    games = tf.gfile.Glob(os.path.join('data', model_name, '*.zz'))
    if len(games) > MAX_GAMES_PER_GENERATION:
        print("{} has enough games ({})".format(model_name, len(games)))
        time.sleep(10 * 60)
        sys.exit(1)
    print("Playing a game with model {}".format(model_name))
    model_save_path = os.path.join('models', model_name)
    selfplay_dir = os.path.join('data', model_name)
    
    with utils.logged_timer("Loading weights from %s ... " % model_save_path):
        network = dual_net.DualNetwork(model_save_path)

    with utils.logged_timer("Playing game"):
        dao.play(network, selfplay_dir)

def train(working_dir):
    model_num, model_name = get_models()[-1]

    print("Training on gathered game data, initializing from {}".format(model_name))
    new_model_num = model_num + 1
    new_model_name = shipname.generate(new_model_num)
    print("New model will be {}".format(new_model_name))

    tf_records = sorted(tf.gfile.Glob(os.path.join(os.path.join('data', model_name), '*.tfrecord.zz')))
    tf_records = tf_records[-1 * (WINDOW_SIZE // EXAMPLES_PER_RECORD):]

    model_save_path = os.path.join('models', new_model_name)
    
    print("Training on:", tf_records[0], "to", tf_records[-1])
    with utils.logged_timer("Training"):
        dual_net.train(*tf_records, steps=5000)
    print("== Training done.  Exporting model to ", model_save_path)
    dual_net.export_model(working_dir, model_save_path)
    freeze_graph(model_save_path)

def run(n, path):
    dao.init(path)
    import os
    import numpy as np
    os.system('rmdir /S /Q temp')
    os.system('rmdir /S /Q models')
    os.system('rmdir /S /Q data')
    if os.path.exists("lasttime.npy"):
        os.remove("lasttime.npy") 
    os.mkdir('data')

    bootstrap('temp')
    selfplay()
    for i in range(n):
        train('temp')
        selfplay()

    return np.load('lasttime.npy', allow_pickle=True).tolist()