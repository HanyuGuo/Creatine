import tensorflow as tf
import numpy as np


data = np.array([1.0,1,2,0,1,2,0,1,1,2,0,2,2,1,0,0,0,1,0,1,0,2,0,0,1,0,1,2,0,0,2,2,1,0,2,2,1,2,2,0,1,0,1,0,1,0,0,1,0,1,1,0,1,2,2,0,1,0,1,0,0,2,2,1,1,2,1,1,2,0,2,1,0,1,2, 1.0,1,2,0,1,2,0,1,1,2,0,2,2,1,0,0,0,1,0,1,0,2,0,0,1,0,1,2,0,0,2,2,1,0,2,2,1,2,2,0,1,0,1,0,1,0,0,1,0,1,1,0,1,2,2,0,1,0,1,0,0,2,2,1,1,2,1,1,2,0,2,1,0,1,2], dtype=np.float32)
filter_w = np.array([-1.0, 0,-1, 0, 1, 0, 0,-1, 0, 0, 1, 0, 1, 0, 1, 0,-1, 1, 1,-1, 1, 0, 0, 1, 0,-1, 0, 1,-1,-1, 0,-1, 1, 0,-1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0,-1,-1, 1, 1, 0,-1, 0], dtype=np.float32)

data = data.reshape(2,5,5,3)
filter_w = filter_w.reshape(3,3,3,2)



result = tf.nn.conv2d(data, filter_w, strides=[1,2,2,1], padding="SAME")

SessConfig = tf.ConfigProto(
        device_count = {'GPU': 0}
    )
# Launch the graph
sess = tf.Session(config=SessConfig)

print(sess.run(result))


