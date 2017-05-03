import tensorflow as tf
import numpy as np


data = np.array([np.arange(64)], dtype=np.float32)
filter_w = np.array([np.arange(8)], dtype=np.float32)

data = data.reshape(2,4,4,2)
filter_w = filter_w.reshape(2,2,2,1)



result = tf.nn.conv2d(data, filter_w, strides=[1,1,1,1], padding="SAME")

SessConfig = tf.ConfigProto(
        device_count = {'GPU': 0}
    )
# Launch the graph
sess = tf.Session(config=SessConfig)

print(sess.run(result))



