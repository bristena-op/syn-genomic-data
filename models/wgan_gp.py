# Large amount of credit goes to:
# https://github.com/keras-team/keras-contrib/blob/master/examples/improved_wgan.py
# which I've used as a reference for this implementation

from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers.merge import _Merge
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv1D
from keras.models import Sequential, Model
from tensorflow.keras.optimizers import RMSprop
from functools import partial

import keras.backend as K

import pandas as pd
import sys

import numpy as np

class RandomWeightedAverage(_Merge):
    """Provides a (random) weighted average between real and generated image samples"""
    def _merge_function(self, inputs):
        alpha = K.random_uniform((32,1, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])

class WGANGP():
    def __init__(self):
        self.no_snps = 1000

        self.img_shape = (self.no_snps, 1)
        self.latent_dim = 100

        # Following parameter and optimizer set as recommended in paper
        self.n_critic = 5
        optimizer = RMSprop(lr=0.00005)

        # Build the generator and critic
        self.generator = self.build_generator()
        self.critic = self.build_critic()

        #-------------------------------
        # Construct Computational Graph
        #       for the Critic
        #-------------------------------

        # Freeze generator's layers while training critic
        self.generator.trainable = False

        # Image input (real sample)
        real_img = Input(shape=self.img_shape)

        # Noise input
        z_disc = Input(shape=(self.latent_dim,))
        # Generate image based of noise (fake sample)
        fake_img = self.generator(z_disc)

        # Discriminator determines validity of the real and fake images
        fake = self.critic(fake_img)
        valid = self.critic(real_img)

        # Construct weighted average between real and fake images
        interpolated_img = RandomWeightedAverage()([real_img, fake_img])
        # Determine validity of weighted sample
        validity_interpolated = self.critic(interpolated_img)

        # Use Python partial to provide loss function with additional
        # 'averaged_samples' argument
        partial_gp_loss = partial(self.gradient_penalty_loss,
                          averaged_samples=interpolated_img)
        partial_gp_loss.__name__ = 'gradient_penalty' # Keras requires function names

        self.critic_model = Model(inputs=[real_img, z_disc],
                            outputs=[valid, fake, validity_interpolated])
        self.critic_model.compile(loss=[self.wasserstein_loss,
                                              self.wasserstein_loss,
                                              partial_gp_loss],
                                        optimizer=optimizer,
                                        loss_weights=[1, 1, 10])
        #-------------------------------
        # Construct Computational Graph
        #         for Generator
        #-------------------------------

        # For the generator we freeze the critic's layers
        self.critic.trainable = False
        self.generator.trainable = True

        # Sampled noise for input to generator
        z_gen = Input(shape=(self.latent_dim,))
        # Generate images based of noise
        img = self.generator(z_gen)
        # Discriminator determines validity
        valid = self.critic(img)
        # Defines generator model
        self.generator_model = Model(z_gen, valid)
        self.generator_model.compile(loss=self.wasserstein_loss, optimizer=optimizer)


    def gradient_penalty_loss(self, y_true, y_pred, averaged_samples):
        """
        Computes gradient penalty based on prediction and weighted real / fake samples
        """
        gradients = K.gradients(y_pred, averaged_samples)[0]
        # compute the euclidean norm by squaring ...
        gradients_sqr = K.square(gradients)
        #   ... summing over the rows ...
        gradients_sqr_sum = K.sum(gradients_sqr,
                                  axis=np.arange(1, len(gradients_sqr.shape)))
        #   ... and sqrt
        gradient_l2_norm = K.sqrt(gradients_sqr_sum)
        # compute lambda * (1 - ||grad||)^2 still for each single sample
        gradient_penalty = K.square(1 - gradient_l2_norm)
        # return the mean as loss over all the batch samples
        return K.mean(gradient_penalty)


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

    def train(self,dat, epochs, batch_size, sample_interval=50):

        # Load the dataset
        X_train = dat * 0.5 + 0.5

        # Rescale -1 to 1

        import pdb;
        pdb.set_trace()
        # Adversarial ground truths
        valid = -np.ones((batch_size, 1))
        fake =  np.ones((batch_size, 1))
        dummy = np.zeros((batch_size, 1)) # Dummy gt for gradient penalty


        for epoch in range(epochs):

            for _ in range(self.n_critic):

                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Select a random batch of images
                idx = np.random.randint(0, X_train.shape[0], batch_size)
                # imgs = X_train[idx]
                imgs = np.reshape(X_train[idx], (  X_train[idx].shape[0], X_train[idx].shape[1], 1 ))

                # Sample generator input
                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
                # Train the critic
                d_loss = self.critic_model.train_on_batch([imgs, noise],
                                                                [valid, fake, dummy])

            # ---------------------
            #  Train Generator
            # ---------------------

            g_loss = self.generator_model.train_on_batch(noise, valid)

            # Plot the progress
            print ("%d [D loss: %f] [G loss: %f]" % (epoch, d_loss[0], g_loss))



    def generate_samples(self, n_samples):
        samples = []
        noise = np.random.normal(0, 1, (n_samples, self.latent_dim))

        sample = self.generator.predict(noise) *0.5+ 0.5
        sample =np.around(sample, decimals=0)
        # samples.append(sample)
        return sample


if __name__ == '__main__':
    dataset = 'CHB'
    data_file = f'/home/bristena/syn_genomics/datasets/chr13/small_{dataset}.chr13.hap'
    dat = pd.read_csv(data_file, sep='\t', header=None)
    # import pdb;

    # pdb.set_trace()

    dat = dat.to_numpy()
    pos = dat[:, 0]
    dat = np.transpose(dat[:, 1:])
    wgan = WGANGP()
    wgan.train(dat, epochs=100, batch_size=32, sample_interval=200)
    sample = np.transpose(wgan.generate_samples(dat.shape[0]))
    # import pdb;

    # pdb.set_trace()
    sample = np.insert(sample[0], 0, pos, axis=1)
    # import pdb;

    # pdb.set_trace()
    res_file = f'../syn_data/wgangp_hap_out_{dataset}3.csv'
    np.savetxt(res_file, sample, delimiter='\t', fmt='%i')
    print(sample)
