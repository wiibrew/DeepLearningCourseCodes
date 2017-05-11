import numpy as np
import tensorflow as tf

import vgg16
import utils
from imagenet1000_clsid_to_human import labels

img1 = utils.load_image("./test_data/tiger.jpeg")
img2 = utils.load_image("./test_data/puzzle.jpeg")

batch1 = img1.reshape((1, 224, 224, 3))
batch2 = img2.reshape((1, 224, 224, 3))

batch = np.concatenate((batch1, batch2), 0)

def percent(v):
    return '%.2f%%' % (v * 100)

# with tf.Session(config=tf.ConfigProto(gpu_options=(tf.GPUOptions(per_process_gpu_memory_fraction=0.7)))) as sess:
with tf.device('/cpu:0'):
    with tf.Session() as sess:
        images = tf.placeholder("float", [2, 224, 224, 3])
        feed_dict = {images: batch}

        vgg = vgg16.Vgg16()
        with tf.name_scope("content_vgg"):
            vgg.build(images)

        prob = sess.run(vgg.prob, feed_dict=feed_dict)
        for i, p in enumerate(prob):
            v = sess.run(tf.nn.top_k(p, 5))
            print('-'*4)
            for j, k in enumerate(v.indices):
                print(labels[k], ':', percent(v.values[j]))
