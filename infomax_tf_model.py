import tensorflow as tf
from tensorflow.contrib.framework.python.ops import arg_scope
from keras.datasets import cifar10


def make_model(name, template, **kwargs):
    """Create model with fixed kwargs."""
    run = lambda *args, **kw: template(*args, **dict((k, v) for kws in (kw, kwargs) for k, v in kws.items()))
    return tf.make_template(name, run, unique_name_=name)


class Model(object):
    def __init__(self, args):
        self.args = args
        self.bch_sz = args.bch_sz
        self.img_sz = args.img_sz
        self.img_shape = self.img_sz ** 2
        self.z_dim = args.z_dim
        self.define_graph()
        self.fea_enc_ite = 3
        self.vec_enc_ite = 2

    def feature_map_encoding(self, x_in, dropout_p=0.5, activation='elu'):
        x_enc = x_in
        with arg_scope([tf.nn.conv2d], dropout_p=dropout_p, activation=activation):
            for i in range(self.fea_enc_ite):
                out_channel = int(self.z_dim / 2**(2-i))
                if i == 0:
                    x_enc = tf.nn.conv2d(x_enc, filter=[3, 3, 3, out_channel], strides=1, padding='SAME')
                else:
                    x_enc = tf.nn.conv2d(x_enc, filter=[3, 3, int(self.z_dim / 2**(2-(i+1))), out_channel], strides=1, padding='SAME')
                x_enc = tf.nn.batch_normalization(x_enc)
                x_enc = tf.nn.leaky_relu(x_enc, 0.2)
                x_enc = tf.nn.max_pool(x_enc, [2,2])
            return x_enc

    def vec_encoding(self, fea_x_in, dropout_p=0.5, activation='relu'):
        x_enc = fea_x_in
        with arg_scope([tf.nn.conv2d], dropout_p=dropout_p, activation=activation):
            for i in range(2):
                x_enc = tf.nn.conv2d(x_enc, filter=[3, 3, self.z_dim, self.z_dim], strides=1, padding='SAME')
                x_enc = tf.nn.batch_normalization(x_enc)
                x_enc = tf.nn.leaky_relu(x_enc, 0.2)
            z_mean = tf.keras.layers.Dense(self.z_dim)(x_enc)
            z_log_var = tf.keras.layers.Dense(self.z_dim)(x_enc)

            return z_mean, z_log_var

    def sampling(self, args):
        z_mean, z_log_var = args
        u = tf.random.normal(shape=z_mean.shape)
        return z_mean + tf.math.exp(z_log_var / 2) * u

    def sampling_enc_z(self, x_in):
        fea_x_enc = self.feature_map_encoding(x_in)
        vec_x_enc = self.vec_encoding(fea_x_enc)
        z_mean, z_log_var = self.sampling(vec_x_enc)
        return z_mean, z_log_var

    def shuffle_x(self, x_in):
        return tf.random.shuffle(x_in)

    def global_discriminator(self, x_in):
        z = x_in
        z_mean = tf.keras.layers.Dense(self.z_dim, activation='relu')(z)
        z_mean = tf.keras.layers.Dense(self.z_dim, activation='relu')(z)
        z_mean = tf.keras.layers.Dense(self.z_dim, activation='relu')(z)
        z_mean = tf.keras.layers.Dense(1, activation='sigmoid')(z)
        return z_mean

    def local_discriminator(self, x_in):
        z = x_in
        z_mean = tf.keras.layers.Dense(self.z_dim, activation='relu')(z)
        z_mean = tf.keras.layers.Dense(self.z_dim, activation='relu')(z)
        z_mean = tf.keras.layers.Dense(self.z_dim, activation='relu')(z)
        z_mean = tf.keras.layers.Dense(1, activation='sigmoid')(z)
        return z_mean

    def define_graph(self):
        self.x_in = tf.placeholder(
            tf.float32,
            shape=[self.bch_sz] + self.img_shape)

        fea_x_enc = self.feature_map_encoding(self.x_in)
        z_mean, z_log_var = self.vec_encoding(fea_x_enc)
        z_samples = self.sampling([z_mean, z_log_var])
        prior_kl_loss = - 0.5 * tf.math.reduce_mean(1 + z_log_var - tf.math.square(z_mean) - tf.math.exp(z_log_var))

        shuffle_concat_mean_log_var = self.shuffle_x(z_samples)
        shuffle_fea_x_enc = self.shuffle_x(fea_x_enc)

        z_z_1 = tf.concat([z_samples, z_samples])
        z_z_2 = tf.concat([z_samples, shuffle_concat_mean_log_var])

        z_samples_repeat = tf.keras.layers.RepeatVector(4 * 4)(z_samples)
        z_samples_map = tf.keras.layers.Reshape((4, 4, self.z_dim))(z_samples_repeat)

        z_f_1 = tf.concat([z_samples_map, fea_x_enc])
        z_f_2 = tf.concat([z_samples_map, shuffle_fea_x_enc])

        z_z_1_scores = self.global_discriminator(z_z_1)
        z_z_2_scores = self.global_discriminator(z_z_2)
        global_info_loss = - tf.reduce_mean(tf.math.log(z_z_1_scores + 1e-6) + tf.math.log(1 - z_z_2_scores + 1e-6))

        z_f_1_scores = self.local_discriminator(z_z_1)
        z_f_2_scores = self.local_discriminator(z_z_2)
        local_info_loss = - tf.reduce_mean(tf.math.log(z_f_1_scores + 1e-6) + tf.math.log(1 - z_f_2_scores + 1e-6))

        loss = self.args.alpha * global_info_loss + self.args.beta * local_info_loss + self.args.gamma * prior_kl_loss

        optimizer = tf.train.AdamOptimizer()
        opt_op = optimizer.minimize(loss)

    def train(self):
        with tf.train.MonitoredSession() as sess:
            for epoch in range(100):
                (x_train, y_train), (x_test, y_test) = cifar10.load_data()
                x_train = x_train.astype('float32') / 255 - 0.5
                x_test = x_test.astype('float32') / 255 - 0.5

                test_elbo, test_codes, test_samples = sess.run({self.x_in: x_train})
                print('Epoch', epoch, 'elbo', test_elbo)
                plot_codes(test_codes)
                plot_sample(test_samples)
                for _ in range(600):
                    sess.run(optimize, {data: mnist.train.next_batch(100)[0]})
