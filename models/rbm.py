from __future__ import print_function

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
# import tensorflow as tf
# tf.compat.v1.disable_eager_execution()

import numpy as np
import sys
import pandas as pd


def tf_xavier_init(fan_in, fan_out, *, const=1.0, dtype=np.float32):
    k = const * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random.uniform((fan_in, fan_out), minval=-k, maxval=k, dtype=dtype)
def sample_bernoulli(probs):
    return tf.nn.relu(tf.sign(probs - tf.random.uniform(tf.shape(probs))))

class RBM:
    def __init__(self,
                 n_visible,
                 n_hidden,
                 learning_rate=0.01,
                 momentum=0.95,
                 xavier_const=1.0,
                 n_epoches=2000,
                 batch_size=10,
                 shuffle=False,
                 verbose=True,
                 err_function='mse',
                 use_tqdm=False,
                 # DEPRECATED:
                 tqdm=None):
        if not 0.0 <= momentum <= 1.0:
            raise ValueError('momentum should be in range [0, 1]')

        if err_function not in {'mse', 'cosine'}:
            raise ValueError('err_function should be either \'mse\' or \'cosine\'')

        self._use_tqdm = use_tqdm
        self._tqdm = None

        if use_tqdm or tqdm is not None:
            from tqdm import tqdm
            self._tqdm = tqdm

        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.xavier_const = xavier_const
        self.n_epoches = n_epoches
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.verbose = verbose
        self.err_function = err_function


        # init = tf.global_variables_initializer()
        # config = tf.ConfigProto(device_count={'CPU': 10})
        # self.sess = tf.Session(config=config)
        # self.sess.run(init)

        self.datatype = pd.DataFrame

        self.__name__ = 'RBM'

    def _initialize_vars(self):
        hidden_p = tf.nn.sigmoid(tf.matmul(self.x, self.w) + self.hidden_bias)
        visible_recon_p = tf.nn.sigmoid(tf.matmul(sample_bernoulli(hidden_p), tf.transpose(self.w)) + self.visible_bias)
        hidden_recon_p = tf.nn.sigmoid(tf.matmul(visible_recon_p, self.w) + self.hidden_bias)

        positive_grad = tf.matmul(tf.transpose(self.x), hidden_p)
        negative_grad = tf.matmul(tf.transpose(visible_recon_p), hidden_recon_p)

        def f(x_old, x_new):
            return self.momentum * x_old + \
                   self.learning_rate * x_new * (1 - self.momentum) / tf.cast(tf.shape(x_new)[0],dtype=tf.float32)

        delta_w_new = f(self.delta_w, positive_grad - negative_grad)
        delta_visible_bias_new = f(self.delta_visible_bias, tf.reduce_mean(self.x - visible_recon_p, 0))
        delta_hidden_bias_new = f(self.delta_hidden_bias, tf.reduce_mean(hidden_p - hidden_recon_p, 0))

        update_delta_w = self.delta_w.assign(delta_w_new)
        update_delta_visible_bias = self.delta_visible_bias.assign(delta_visible_bias_new)
        update_delta_hidden_bias = self.delta_hidden_bias.assign(delta_hidden_bias_new)

        update_w = self.w.assign(self.w + delta_w_new)
        update_visible_bias = self.visible_bias.assign(self.visible_bias + delta_visible_bias_new)
        update_hidden_bias = self.hidden_bias.assign(self.hidden_bias + delta_hidden_bias_new)

        self.update_deltas = [update_delta_w, update_delta_visible_bias, update_delta_hidden_bias]
        self.update_weights = [update_w, update_visible_bias, update_hidden_bias]

        self.compute_hidden = tf.nn.sigmoid(tf.matmul(self.x, self.w) + self.hidden_bias)
        self.compute_visible = tf.nn.sigmoid(tf.matmul(self.compute_hidden, tf.transpose(self.w)) + self.visible_bias)
        self.compute_visible_from_hidden = tf.nn.sigmoid(tf.matmul(self.y, tf.transpose(self.w)) + self.visible_bias)

    def get_err(self, batch_x):
        return self.sess.run(self.compute_err, feed_dict={self.x: batch_x})

    def get_free_energy(self):
        pass

    def transform(self, batch_x):
        return self.sess.run(self.compute_hidden, feed_dict={self.x: batch_x})

    def transform_inv(self, batch_y):
        return self.sess.run(self.compute_visible_from_hidden, feed_dict={self.y: batch_y})

    def reconstruct(self, batch_x):
        return self.sess.run(self.compute_visible, feed_dict={self.x: batch_x})

    def partial_fit(self, batch_x):
        # import pdb; pdb.set_trace()

        self.sess.run(self.update_weights + self.update_deltas, feed_dict={self.x: batch_x})

    def fit(self,
            data_x):
        self.x = tf.placeholder(tf.float32, [None, self.n_visible])
        self.y = tf.placeholder(tf.float32, [None, self.n_hidden])

        self.w = tf.Variable(tf_xavier_init(self.n_visible, self.n_hidden, const=self.xavier_const), dtype=tf.float32)
        self.visible_bias = tf.Variable(tf.zeros([self.n_visible]), dtype=tf.float32)
        self.hidden_bias = tf.Variable(tf.zeros([self.n_hidden]), dtype=tf.float32)

        self.delta_w = tf.Variable(tf.zeros([self.n_visible, self.n_hidden]), dtype=tf.float32)
        self.delta_visible_bias = tf.Variable(tf.zeros([self.n_visible]), dtype=tf.float32)
        self.delta_hidden_bias = tf.Variable(tf.zeros([self.n_hidden]), dtype=tf.float32)

        self.update_weights = None
        self.update_deltas = None
        self.compute_hidden = None
        self.compute_visible = None
        self.compute_visible_from_hidden = None


        self._initialize_vars()

        assert self.update_weights is not None
        assert self.update_deltas is not None
        assert self.compute_hidden is not None
        assert self.compute_visible is not None
        assert self.compute_visible_from_hidden is not None

        if self.err_function == 'cosine':
            x1_norm = tf.nn.l2_normalize(self.x, 1)
            x2_norm = tf.nn.l2_normalize(self.compute_visible, 1)
            cos_val = tf.reduce_mean(tf.reduce_sum(tf.mul(x1_norm, x2_norm), 1))
            self.compute_err = tf.acos(cos_val) / tf.constant(np.pi)
        else:
            self.compute_err = tf.reduce_mean(tf.square(self.x - self.compute_visible))
        assert self.n_epoches > 0
        self.dat=data_x
        n_data = self.dat.shape[0]
        if self.batch_size > 0:
            n_batches = n_data // self.batch_size + (0 if n_data % self.batch_size == 0 else 1)
        else:
            n_batches = 1
        # import pdb; pdb.set_trace()
        if self.shuffle:
            data_x_cpy = self.dat.copy()
            inds = np.arange(n_data)
        else:
            data_x_cpy = self.dat

        errs = []
        self.sess = tf.Session()

        self.sess.run(tf.global_variables_initializer())
        for e in range(self.n_epoches):
            if self.verbose and not self._use_tqdm:
                print('Epoch: {:d}'.format(e))

            epoch_errs = np.zeros((n_batches,))
            epoch_errs_ptr = 0

            if self.shuffle:
                np.random.shuffle(inds)
                data_x_cpy = data_x_cpy[inds]

            r_batches = range(n_batches)

            if self.verbose and self._use_tqdm:
                r_batches = self._tqdm(r_batches, desc='Epoch: {:d}'.format(e), ascii=True, file=sys.stdout)

            for b in r_batches:
                batch_x = data_x_cpy[b * self.batch_size:(b + 1) * self.batch_size]
                # import pdb; pdb.set_trace()
                self.partial_fit(batch_x)
                batch_err = self.get_err(batch_x)
                epoch_errs[epoch_errs_ptr] = batch_err
                epoch_errs_ptr += 1

            if self.verbose:
                err_mean = epoch_errs.mean()
                if self._use_tqdm:
                    self._tqdm.write('Train error: {:.4f}'.format(err_mean))
                    self._tqdm.write('')
                else:
                    print('Train error: {:.4f}'.format(err_mean))
                    print('')
                sys.stdout.flush()

            errs = np.hstack([errs, epoch_errs])

        return errs

    def get_weights(self):
        return self.sess.run(self.w),\
            self.sess.run(self.visible_bias),\
            self.sess.run(self.hidden_bias)

    def save_weights(self, filename, name):
        saver = tf.train.Saver({name + '_w': self.w,
                                name + '_v': self.visible_bias,
                                name + '_h': self.hidden_bias})
        return saver.save(self.sess, filename)

    def set_weights(self, w, visible_bias, hidden_bias):
        self.sess.run(self.w.assign(w))
        self.sess.run(self.visible_bias.assign(visible_bias))
        self.sess.run(self.hidden_bias.assign(hidden_bias))

    def load_weights(self, filename, name):
        saver = tf.train.Saver({name + '_w': self.w,
                                name + '_v': self.visible_bias,
                                name + '_h': self.hidden_bias})
        saver.restore(self.sess, filename)
    def generate_samples(self, n_samples):
        # samples = []
        # s=[]
        # for i in range(n_samples):
        #     r = np.random.randint(0, self.dat.shape[0])
        #     s.append(r)
        # import pdb; pdb.set_trace()
        #     sample = np.reshape(self.dat.iloc[r], (1, -1))
        #     sample_rec = self.reconstruct( sample)
        #     if i ==0:
        #         samples = sample_rec
        #     else:
        #         samples = np.concatenate((samples, sample_rec), axis = 0)
        # sample_rec = self.reconstruct(self.dat)[:n_samples]
        sample_rec = np.around(self.reconstruct(self.dat)[:n_samples], decimals=0)
        sample_rec = sample_rec.astype(int)
        # import pdb; pdb.set_trace()
        return pd.DataFrame(sample_rec, columns=self.dat.columns)

# if __name__ == '__main__':
#     dataset = 'ASW'
#     data_file = f'/home/bristena/syn_genomics/datasets/chr13/small_{dataset}.chr13.hap'
#     dat = pd.read_csv(data_file,sep= '\t', header=None)
#     # import pdb; pdb.set_trace()
#
#     dat = dat.to_numpy()
#     pos = dat[:, 0]
#     dat = np.transpose(dat[:, 1:])
#     rbm = RBM(n_visible=1000, n_hidden=100, learning_rate=0.01,  momentum=0.95, err_function='mse', use_tqdm=False)
#     errs = rbm.fit(dat, n_epoches=1000, batch_size=32)
#     sample = np.transpose(rbm.generate_samples(dat, dat.shape[0]))
#     sample = np.around(sample, decimals=0)
#     import pdb; pdb.set_trace()
#
#     sample = sample.astype(int)
#
#     sample = np.insert(sample, 0, pos, axis=1)
#     res_file = f'../syn_data/rbm_out_hap_{dataset}.csv'
#     np.savetxt(res_file, sample, delimiter='\t', fmt='%i')
#     import pdb;
#
#     pdb.set_trace()
