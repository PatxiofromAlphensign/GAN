import cgan_classifier

import tensorflow as tf
import numpy as np
from PIL import Image
import sys

def create_feeddict(x):
    tfx = [ i*x[0,:,:] for i in range(x.shape[0])]  # should have used the product of hxl 
    return {v:k for k, v in zip(np.random.rand(*(x.shape)), tfx)}

class DataLoad:
    def __init__(self, imgn):
        self.x = np.asarray(Image.open(imgn))
        self.tfx = tf.compat.v1.placeholder(tf.float32, shape=self.x.shape)
        self.x_dim = self.x.shape[0]
        self.y_dim = self.x.shape[1]
        self.z_dim = self.x.shape[2]
        x,y = self.x[0,:],self.x[:, 0]
        if x.shape[0] > y.shape[0]:
            self.prod  = x[:y.shape[0],:] * y
        else:
            self.prod = y[:x.shape[0],:] * x

        self.prod  = tf.constant(self.prod)
        self.size = np.prod(self.prod.shape)
        self.channel = 1

    def push(self, sess):
        fdict =create_feeddict(self.x)
        # MUST DO IT IN A LOOP
        sess.run(self.prod, feed_dict = fdict )    

class generator: # later inheretet the generator from tp1
    def __init__(self, x):
        self.x = x
        self.shape1,self.shape2 = self.x.shape
        self.dense = tf.keras.layers.Dense(self.shape2)

    def __call__(self, x):
        self.dense(x)
        pass # x ops here

            
if __name__ == "__main__":
    tf.compat.v1.disable_eager_execution()
    with tf.compat.v1.Session() as sess:
        d = DataLoad(sys.argv[1])
        cgan_classifier.CGAN_Classifier(generator, None, None, d)

