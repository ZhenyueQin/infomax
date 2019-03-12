from infomax_tf_keras import DIM
from keras.datasets import cifar10
import argparse
from keras import backend as K
import numpy as np
import theano


(x_train, y_train), (x_test, y_test) = cifar10.load_data()

parser = argparse.ArgumentParser()
parser.add_argument('--alpha')
parser.add_argument('--beta')
parser.add_argument('--gamma')
parser.add_argument('--z_dim')
parser.add_argument('--feature_encoder_ite')
parser.add_argument('--vec_encoder_ite')
parser.add_argument('--mission')
opt = parser.parse_args()

if opt.alpha is None:
    opt.alpha = 0.5
else:
    opt.alpha = int(opt.alpha)
if opt.beta is None:
    opt.beta = 1.5
else:
    opt.beta = int(opt.beta)
if opt.gamma is None:
    opt.gamma = 0.01
else:
    opt.gamma = int(opt.gamma)
opt.z_dim = 256
opt.feature_encoder_ite = 3
opt.vec_encoder_ite = 2

print('arguments: ')
print(opt)

dim = DIM(x_train, x_test, opt)

if opt.mission == 'train_a_model':
    dim.train_a_model()
elif opt.mission == 'load_a_model':
    model_path = './generated_outs/2019-03-12T18-04-00/total_model.cifar10.weights'
    a_model = dim.load_a_model(model_path)
    print('a model: ', a_model)

    dim.sample_knn(a_model)





