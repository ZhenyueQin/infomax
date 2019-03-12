import tensorflow as tf
import os
import utilities
from keras.datasets import cifar10
import keras
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K
import numpy as np
import imageio
import datetime


def normalize_data(data):
    return data.astype('float32') / 255 - 0.5


class DIM:
    def __init__(self, x_train, x_test, args):
        self.x_train = normalize_data(x_train)
        self.x_test = normalize_data(x_test)
        self.args = args
        self.img_dim = x_train.shape[1]


    def get_feature_encoder(self):
        feature_enc = keras.Sequential()
        for i in range(self.args.feature_encoder_ite):
            feature_enc.add(keras.layers.Conv2D(int(self.args.z_dim / 2 ** (2 - i)), kernel_size=(3, 3), padding='SAME'))
            feature_enc.add(keras.layers.BatchNormalization())
            feature_enc.add(keras.layers.LeakyReLU(0.2))
            feature_enc.add(keras.layers.MaxPooling2D(2, 2))
        return feature_enc

    def get_vec_encoder(self):
        vec_enc = keras.Sequential()
        for i in range(self.args.vec_encoder_ite):
            vec_enc.add(keras.layers.Conv2D(self.args.z_dim, kernel_size=(3, 3), padding='SAME'))
            vec_enc.add(keras.layers.BatchNormalization())
            vec_enc.add(keras.layers.LeakyReLU(0.2))
        vec_enc.add(keras.layers.GlobalMaxPool2D())
        return vec_enc

    def get_mean_logvar(self, x_in):
        z_mean = keras.layers.Dense(self.args.z_dim)(x_in)
        z_log_var = keras.layers.Dense(self.args.z_dim)(x_in)
        return z_mean, z_log_var

    def reparameterization(self, args):
        z_mean, z_log_var = args
        u = tf.random_normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(z_log_var / 2) * u

    def shuffling(self, x):
        idxs = keras.backend.arange(0, keras.backend.shape(x)[0])
        idxs = keras.backend.tf.random_shuffle(idxs)
        return keras.backend.gather(x, idxs)

    def get_global_dis(self):
        return keras.Sequential([
            keras.layers.Dense(self.args.z_dim, activation='relu'),
            keras.layers.Dense(self.args.z_dim, activation='relu'),
            keras.layers.Dense(self.args.z_dim, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid')
        ])

    def get_local_dis(self):
        return keras.Sequential([
            keras.layers.Dense(self.args.z_dim, activation='relu'),
            keras.layers.Dense(self.args.z_dim, activation='relu'),
            keras.layers.Dense(self.args.z_dim, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid')
        ])

    def get_overall_model(self):
        x_in = Input(shape=(self.img_dim, self.img_dim, 3))

        feature_map = self.get_feature_encoder()(x_in)
        vec_enc = self.get_vec_encoder()(feature_map)
        z_mean, z_log_var = self.get_mean_logvar(vec_enc)

        z_samples = keras.layers.Lambda(self.reparameterization)([z_mean, z_log_var])
        prior_kl_loss = - 0.5 * tf.keras.backend.mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))

        z_shuffle = keras.layers.Lambda(self.shuffling)(z_samples)
        z_z_1 = keras.layers.Concatenate()([z_samples, z_samples])
        z_z_2 = keras.layers.Concatenate()([z_samples, z_shuffle])

        feature_map_shuffle = keras.layers.Lambda(self.shuffling)(feature_map)
        z_samples_repeat = keras.layers.RepeatVector(4 * 4)(z_samples)
        z_samples_map = keras.layers.Reshape((4, 4, self.args.z_dim))(z_samples_repeat)
        z_f_1 = keras.layers.Concatenate()([z_samples_map, feature_map])
        z_f_2 = keras.layers.Concatenate()([z_samples_map, feature_map_shuffle])

        z_z_1_scores = self.get_global_dis()(z_z_1)
        z_z_2_scores = self.get_local_dis()(z_z_2)
        global_info_loss = - tf.keras.backend.mean(tf.keras.backend.log(z_z_1_scores + 1e-6) + tf.keras.backend.log(1 - z_z_2_scores + 1e-6))

        z_f_1_scores = self.get_global_dis()(z_f_1)
        z_f_2_scores = self.get_global_dis()(z_f_2)
        local_info_loss = - tf.keras.backend.mean(tf.keras.backend.log(z_f_1_scores + 1e-6) + tf.keras.backend.log(1 - z_f_2_scores + 1e-6))

        model_train = Model(x_in, [z_z_1_scores, z_z_2_scores, z_f_1_scores, z_f_2_scores, z_mean, z_log_var])
        model_train.add_loss(self.args.alpha * global_info_loss + self.args.beta * local_info_loss + self.args.gamma * prior_kl_loss)
        model_train.compile(optimizer=Adam(1e-3))

        return model_train

    def train_a_model(self):
        self.out_dir = utilities.init_logging('generated_outs')
        model_train = self.get_overall_model()
        model_train.fit(self.x_train, epochs=50, batch_size=64)
        model_train.save_weights(os.path.join(self.out_dir, 'total_model.cifar10.weights'))

    def load_a_model(self, model_path):
        a_model = self.get_overall_model()
        import os
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
        a_model.load_weights(model_path)
        return a_model

    def sample_knn(self, model, bch_sz=50000):
        n = 10
        topn = 10
        figure1 = np.zeros((self.img_dim * n, self.img_dim * topn, 3))
        figure2 = np.zeros((self.img_dim * n, self.img_dim * topn, 3))

        zs = model.predict(self.x_train)[4]
        zs_ = zs / (zs ** 2).sum(1, keepdims=True) ** 0.5
        for i in range(n):
            one = i
            idxs = ((zs ** 2).sum(1) + (zs[one] ** 2).sum() - 2 * np.dot(zs, zs[one])).argsort()[:topn]
            for j, k in enumerate(idxs):
                digit = self.x_train[k]
                figure1[i * self.img_dim: (i + 1) * self.img_dim,
                j * self.img_dim: (j + 1) * self.img_dim] = digit
            idxs = np.dot(zs_, zs_[one]).argsort()[-n:][::-1]
            for j, k in enumerate(idxs):
                digit = self.x_train[k]
                figure2[i * self.img_dim: (i + 1) * self.img_dim,
                j * self.img_dim: (j + 1) * self.img_dim] = digit
        figure1 = (figure1 + 1) / 2 * 255
        figure1 = np.clip(figure1, 0, 255)
        figure2 = (figure2 + 1) / 2 * 255
        figure2 = np.clip(figure2, 0, 255)

        now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        imageio.imwrite(os.path.join('testing_models', str(now) + '_l2.png'), figure1)
        imageio.imwrite(os.path.join('testing_models', str(now) + '_cos.png'), figure2)
