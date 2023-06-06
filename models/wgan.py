from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv1D
from keras.models import Sequential, Model
from tensorflow.keras.optimizers import RMSprop

import keras.backend as K
# import tensorflow as tf
import pandas as pd

# import keras
import numpy as np
# from tensorflow.python.keras.backend import set_session
# from tensorflow.python.keras.models import load_model

# session = tf.compat.v1.Session()
# init = tf.compat.v1.global_variables_initializer()
# session.run(init)

class WGAN():
    def __init__(self):
        self.no_snps = 1000

        self.img_shape = (self.no_snps, 1)
        self.latent_dim = 100

        # Following parameter and optimizer set as recommended in paper
        self.n_critic = 5
        self.clip_value = 0.01
        optimizer = RMSprop(lr=0.00005)



        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generated imgs
        z = Input(shape=(self.latent_dim,))
        # img = tf.reshape(self.generator(z), [ 1000, 1])
        img = self.generator(z)

        # Build and compile the critic
        self.critic = self.build_critic()
        self.critic.compile(loss=self.wasserstein_loss,
                            optimizer=optimizer,
                            metrics=['accuracy'])
        # For the combined model we will only train the generator
        self.critic.trainable = False

        # The critic takes generated images as input and determines validity
        valid = self.critic(img)

        # The combined model  (stacked generator and critic)
        self.combined = Model(z, valid)
        self.combined.compile(loss=self.wasserstein_loss,
                              optimizer=optimizer,
                              metrics=['accuracy'])

    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def build_generator(self):

        model = Sequential()
        # input layer
        model.add(Dense(256, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.01))
        model.add(BatchNormalization(momentum=0.8))
        # hidden layer 1
        model.add(Dense(256))  # size no_snps/1.2
        model.add(LeakyReLU(alpha=0.01))
        model.add(BatchNormalization(momentum=0.8))
        # hidden layer 2
        model.add(Dense(512))  # size no_snps/1.1
        model.add(LeakyReLU(alpha=0.01))
        model.add(BatchNormalization(momentum=0.8))
        # output layer size np_snps
        model.add(Dense(self.no_snps, activation='tanh'))
        # the latent vector was set with numpy.rand.normal setiing
        # the mean of the distribution to 0 and the standaard deviation as 1
        model.add(Reshape((self.no_snps, 1)))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        sample = model(noise)

        # return Model(noise, sample)
        return model

    def build_critic(self):
        import pdb; pdb.set_trace()

        model = Sequential()

        model.add(Conv1D(16, kernel_size=3, strides=1, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv1D(32, kernel_size=3, strides=1, padding="same"))
        # model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv1D(64, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv1D(128, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1))

        model.summary()

        img = Input(shape=(self.no_snps,))
        # validity = model(img)

        return model

    def train(self, dat, epochs, batch_size=128, sample_interval=50):

        # Load the dataset
        X_train = dat*0.5+ 0.5

        # Rescale -1 to 1


        # Adversarial ground truths
        valid = -np.ones((batch_size, 1))
        fake = np.ones((batch_size, 1))

        for epoch in range(epochs):

            for _ in range(self.n_critic):

                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Select a random batch of images
                idx = np.random.randint(0, X_train.shape[0], batch_size)
                # import pdb;
                # pdb.set_trace()

                imgs = np.reshape(X_train[idx], (  X_train[idx].shape[0], X_train[idx].shape[1], 1 ))

                # Sample noise as generator input
                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

                # Generate a batch of new images
                gen_imgs = self.generator.predict(noise)
                # Train the critic
                d_loss_real = self.critic.train_on_batch(imgs, valid)
                d_loss_fake = self.critic.train_on_batch(gen_imgs, fake)
                d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)

                # Clip critic weights
                for l in self.critic.layers:
                    weights = l.get_weights()
                    weights = [np.clip(w, -self.clip_value, self.clip_value) for w in weights]
                    l.set_weights(weights)

            # ---------------------
            #  Train Generator
            # ---------------------

            g_loss = self.combined.train_on_batch(noise, valid)



    def generate_samples(self, n_samples):
        samples = []
        noise = np.random.normal(0, 1, (n_samples, self.latent_dim))

        sample = self.generator.predict(noise) *0.5+ 0.5
        sample =np.around(sample, decimals=0)
        # samples.append(sample)
        return sample


if __name__ == '__main__':
    dataset = 'ASW'
    data_file = f'/home/bristena/syn_genomics/datasets/chr13/small_{dataset}.chr13.hap'
    dat = pd.read_csv(data_file, sep='\t', header=None)
    import pdb;

    pdb.set_trace()

    dat = dat.to_numpy()
    pos = dat[:, 0]
    dat = np.transpose(dat[:, 1:])

    wgan = WGAN()
    wgan.train(dat, epochs=100, batch_size=32, sample_interval=200)
    sample = np.transpose(wgan.generate_samples(dat.shape[0]))
    import pdb;

    pdb.set_trace()
    sample = np.insert(sample[0], 0, pos, axis=1)
    import pdb;

    pdb.set_trace()
    res_file = f'../syn_data/wgan_hap_out_{dataset}3.csv'
    np.savetxt(res_file, sample, delimiter='\t', fmt='%i')
    print(sample)
