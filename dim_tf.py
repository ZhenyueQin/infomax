import os
os.system('nvcc --version')

import tensorflow as tf

import utilities
out_dir = utilities.init_logging('generated_outs')

from keras.datasets import cifar10


