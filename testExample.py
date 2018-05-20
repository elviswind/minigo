import preprocessing
import os

fname = os.path.join('madeup', "1526716756-2-DESKTOP-39OPSJL.tfrecord.zz")

ds = preprocessing.get_input_tensors(128, [fname], filter_amount=1)
import tensorflow as tf

with tf.Session() as sess:
    print(sess.run(ds)[1]['pi_tensor'][0])



