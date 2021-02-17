from cgan_classifier import CGAN_Classifier
from dataloading import DataLoad, generator 

import tensorflow as tf
import sys 
import numpy as np

from basic_utils import make2Dndarray

class classifier(CGAN_Classifier):
    # TODO : need to extend the classifier deriving from CGAN_Classifier
    def __init__(self, img):
        self.data = DataLoad(img)
        super().__init__(generator, discrimator, None,self.data)

    def __call__(self, x):
        self.G_sample()

class generator: # later inheretet the generator from tp1
    def __init__(self, x):
        self.x = x
        self.dense = tf.keras.layers.Dense(self.x.shape[-1])
        self.loss_v1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.x, labels=tf.ones_like(self.x)))
#        self.loss_v2 =  tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.x, labels=tf.ones_like(self.x))

    def __call__(self, x):
        self.dense(x)
        pass # x ops here

class discrimator: # later inheretet the generator from wtp1
    def __init__(self, x, reuse=False): 
        shape = x.shape[1:] if None in x.shape else x.shape
        rands = np.random.rand(*shape)
        xdata = np.arange(np.prod(shape)).reshape(*shape)
        expr = "".join([chr(i) for i in range(97, 97 +  len(x.shape)+1)])
        self.x = np.einsum(expr, [rands, xdata])

    def __call__(self, x):
        return x # for now


def main():
    tf.compat.v1.disable_eager_execution() 
    data = DataLoad(sys.argv[1])
    discrimator(data.x)
    x = tf.constant(data.x, dtype=tf.float32) #convert to float32 
    generator(x)

#classifier(sys.argv[1])
main()


