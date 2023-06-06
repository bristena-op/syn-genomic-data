from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras.objectives import binary_crossentropy
from keras.callbacks import LearningRateScheduler
from keras.utils.vis_utils import model_to_dot
from keras.callbacks import EarlyStopping, ModelCheckpoint

import keras.backend as K
from numpy.random import seed
import pandas as pd
import numpy as np

encoding_dim = 20
hidden_layer_dim = 20
latent_space_dim = 2

class VAE():
    def __init__(self, epochs, batch_size=20):
        self.nosnps = 1000
        self.input_layer = Input(shape=(self.nosnps,), name='input_layer')
        self.epochs = epochs
        self.batch_size = batch_size
        # self.autoencoder = self.build_vae()
        # self.autoencoder.compile(optimizer='adam',
        #                          loss='mean_squared_error',
        #                          metrics=['accuracy'])
        self.hidden_layer = Dense(hidden_layer_dim, activation='relu',
            name='hiden_layer')(self.input_layer)
        self.mu = Dense(latent_space_dim, activation='linear', name='mu')(self.hidden_layer)
        self.log_sigma = Dense(latent_space_dim, activation='linear', name='log_sigma')(self.hidden_layer)
        self.vae = self.build_vae()
        self.vae.compile(optimizer='adam', loss=self.vae_loss)
        self.decoder = self.aux_d()
    def build_vae(self):
        self.encoder = Model(self.input_layer, self.mu, name='encoder')
        # Sample from the output of the 2 dense layers
        sampleZ = Lambda(self.sample_z, name='sampleZ', output_shape=(latent_space_dim,))([self.mu, self.log_sigma])

        # Define decoder layers in VAE model
        self.decoder_hidden = Dense(hidden_layer_dim, activation='relu', name='decoder_hidden')
        self.decoder_out = Dense(self.nosnps, activation='sigmoid', name='decoder_out')

        h_p = self.decoder_hidden(sampleZ)
        output_layer = self.decoder_out(h_p)

        # VAE model, Unsupervised leraning for reconstruction of the input data
        vae = Model(self.input_layer, output_layer, name='vae')
        return vae
    def aux_d(self):
        d_in = Input(shape=(latent_space_dim,), name='decoder_input')
        d_h = self.decoder_hidden(d_in)
        d_out = self.decoder_out(d_h)

        decoder = Model(d_in, d_out, name='decoder')
        return decoder
    def vae_loss(self,y_true, y_pred):
        """ Calculate loss = reconstruction loss + KL loss for each data in minibatch """
        recon = K.sum(K.binary_crossentropy(y_pred, y_true), axis=1)
        kl = 0.5 * K.sum(K.exp(self.log_sigma) + K.square(self.mu) - 1. - self.log_sigma, axis=1)

        return recon + kl
    def fit(self, dat):
        import pdb;
        pdb.set_trace()

        self.dat = dat
        X_train = self.dat.to_numpy()

        self.vae.fit(X_train,X_train,
                             epochs=self.epochs,
                             batch_size=self.batch_size,
                             shuffle=True,
                             verbose=1)

    def sample_z(self,args):
        mu, log_sigma = args
        eps = K.random_normal(shape=(self.batch_size, latent_space_dim), mean=0., stddev=1.)
        return mu + K.exp(log_sigma / 2) * eps

    def generate_samples(self, n_samples):
        import pdb; pdb.set_trace()

        samples_enc = self.encoder.predict(self.dat.to_numpy())
        samples = np.around(self.decoder.predict(samples_enc)[:n_samples], decimals=0)
        samples = samples.astype(int)

        return pd.DataFrame(samples, columns=self.dat.columns)

