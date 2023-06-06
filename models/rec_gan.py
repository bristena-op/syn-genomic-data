from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam

import numpy as np
import pandas as pd
import tensorflow as tf
from recomb_model import RecombModel


class REGAN():
    def __init__(self, epochs, batch_size=32, sample_interval=50):
        self.no_snps = 1000
        # self.channels = 1
        self.img_shape = (self.no_snps)
        self.latent_dim = 600

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer='adam',
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        validity = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer='adam')


        self.epochs = epochs
        self.batch_size = batch_size
        self.sample_interval = sample_interval

        self.datatype=pd.DataFrame
        self.__name__="GAN"

    def build_generator(self):

        model = Sequential()
        # input layer
        model.add(Dense(256, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.01))
        model.add(BatchNormalization(momentum=0.8))
        # hidden layer 1
        model.add(Dense(512))  # size no_snps/1.2
        model.add(LeakyReLU(alpha=0.01))
        model.add(BatchNormalization(momentum=0.8))
        # hidden layer 2
        model.add(Dense(1024))  # size no_snps/1.1
        model.add(LeakyReLU(alpha=0.01))
        model.add(BatchNormalization(momentum=0.8))
        # output layer size np_snps
        model.add(Dense(self.no_snps, activation='tanh'))
        # the latent vector was set with numpy.rand.normal setiing
        # the mean of the distribution to 0 and the standaard deviation as 1


        model.summary()

        noise = Input(shape=(self.latent_dim,))
        sample = model(noise)

        # return Model(noise, sample)
        return model

    def build_discriminator(self):

        model = Sequential()

        # input layer
        model.add(Dense(1024, input_shape=( self.no_snps,)))
        model.add(LeakyReLU(alpha=0.01))
        # hidden layer 1
        model.add(Dense(512))  # size no_snps/2
        model.add(LeakyReLU(alpha=0.01))
        # hidden layer 2
        model.add(Dense(256))  # size no_snps/3
        model.add(LeakyReLU(alpha=0.01))
        # output layer size 1
        model.add(Dense(1, activation='sigmoid'))

        model.summary()
        # import pdb; pdb.set_trace()
        sample = Input(shape=(self.no_snps,))
        validity = model(sample)

        # return Model(sample, validity)
        return model
    def fit(self, dat):

        # Load the dataset
        recomb = RecombModel("../datasets/chr13/small_genotypes_chr13_CHB.txt", "../datasets/chr13/small_CHB.chr13.hap",
                             "../datasets/chr13/small_genetic_map_chr13_combined_b36.txt")
        dat = recomb.generate_samples(1000).to_numpy()
        X_train = dat-0.5 /0.5



        # Adversarial ground truths
        valid = np.ones((self.batch_size, 1))
        fake = np.zeros((self.batch_size, 1))

        for epoch in range(self.epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of images
            idx = np.random.randint(0, X_train.shape[0], self.batch_size)
            imgs = X_train[idx]

            noise = np.random.normal(0, 1, (self.batch_size, self.latent_dim))


            # Generate a batch of new images
            gen_imgs = self.generator.predict(noise)
            # import pdb; pdb.set_trace()
            # gen_imgs = np.around(gen_imgs, decimals=0)
            # import pdb;
            # pdb.set_trace()
            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            noise = np.random.normal(0, 1, (self.batch_size, self.latent_dim))

            # Train the generator (to have the discriminator label samples as valid)
            g_loss = self.combined.train_on_batch(noise, valid)

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
    def generate_samples(self, n_samples):
        samples = []
        noise = np.random.normal(0, 1, (n_samples, self.latent_dim))
        import pdb; pdb.set_trace()
        sample = self.generator.predict(noise)*0.5+ 0.5
        sample =np.around(sample, decimals=0)
        # samples.append(sample)
        return sample

    # def sample_data(self, epoch):
    #     r, c = 5, 5
    #     noise = np.random.normal(0, 1, (r * c, self.latent_dim))
    #     gen_imgs = self.generator.predict(noise)
    #
    #     # Rescale images 0 - 1
    #     gen_imgs = 0.5 * gen_imgs + 0.5
    #
    #     fig, axs = plt.subplots(r, c)
    #     cnt = 0
    #     for i in range(r):
    #         for j in range(c):
    #             axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
    #             axs[i, j].axis('off')
    #             cnt += 1
    #     fig.savefig("images/%d.png" % epoch)
    #     plt.close()



# if __name__ == '__main__':
#     dataset = 'ASW'
#     data_file = f'/home/bristena/syn_genomics/datasets/chr13/small_{dataset}.chr13.hap'
#     dat = pd.read_csv(data_file,sep= '\t', header=None)
#     # import pdb; pdb.set_trace()
#
#     dat = dat.to_numpy()
#     pos = dat[:, 0]
#     dat = np.transpose(dat[:, 1:])
#
#     gan = GAN()
#     gan.train(dat, epochs=8000, batch_size=32, sample_interval=200)
#     sample = np.transpose(gan.generate_samples(dat.shape[0]))
#     # import pdb; pdb.set_trace()
#     sample = np.insert(sample, 0, pos, axis = 1)
#     # import pdb; pdb.set_trace()
#     res_file = f'../syn_data/gan_out_hap_{dataset}.csv'
#     np.savetxt(res_file, sample , delimiter='\t', fmt = '%i')
#     print(sample[0])






