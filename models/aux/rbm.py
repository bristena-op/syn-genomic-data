import tensorflow.compat.v1 as tf
import numpy as np
import pandas as pd

class RBM():
    def __init__(self, nv=1000*1, nh=512, cd_steps=3):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.W = tf.Variable(tf.random.truncated_normal((nv, nh)) * 0.01)
            self.bv = tf.Variable(tf.zeros((nv, 1)))
            self.bh = tf.Variable(tf.zeros((nh, 1)))

            self.cd_steps = cd_steps
            self.modelW = None

    def bernoulli(self, p):
        return tf.nn.relu(tf.sign(p - tf.random_uniform(p.shape)))

    def energy(self, v):
        b_term = tf.matmul(v, self.bv)
        linear_tranform = tf.matmul(v, self.W) + tf.squeeze(self.bh)
        h_term = tf.reduce_sum(tf.log(tf.exp(linear_tranform) + 1), axis=1)
        return tf.reduce_mean(-h_term - b_term)

    def sample_h(self, v):
        ph_given_v = tf.sigmoid(tf.matmul(v, self.W) + tf.squeeze(self.bh))
        return self.bernoulli(ph_given_v)

    def sample_v(self, h):
        pv_given_h = tf.sigmoid(tf.matmul(h, tf.transpose(self.W)) + tf.squeeze(self.bv))
        return self.bernoulli(pv_given_h)

    def gibbs_step(self, i, k, vk):
        hk = self.sample_h(vk)
        vk = self.sample_v(hk)
        return i + 1, k, vk

    def train(self, X, lr=0.01, batch_size=32, epochs=5):
        with self.graph.as_default():
            tf.disable_v2_behavior()
            tf_v = tf.placeholder(tf.float32, [batch_size, self.bv.shape[0]])
            v = tf.round(tf_v)
            vk = tf.identity(v)

            i = tf.constant(0)
            _, _, vk = tf.while_loop(cond=lambda i, k, *args: i <= k,
                                     body=self.gibbs_step,
                                     loop_vars=[i, tf.constant(self.cd_steps), vk],
                                     parallel_iterations=1,
                                     back_prop=False)

            vk = tf.stop_gradient(vk)
            loss = self.energy(v) - self.energy(vk)
            optimizer = tf.train.AdamOptimizer(lr).minimize(loss)
            init = tf.global_variables_initializer()

        with tf.Session(graph=self.graph) as sess:
            init.run()
            for epoch in range(epochs):
                losses = []
                for i in range(0, len(X) - batch_size, batch_size):
                    x_batch = X[i:i + batch_size]
                    l, _ = sess.run([loss, optimizer], feed_dict={tf_v: x_batch})
                    losses.append(l)
                print('Epoch Cost %d: ' % epoch, np.mean(losses), end='\r')
            self.modelW = self.W.eval()
        self.vk = vk

    def reconstruct(self, batch_x):
        return self.sess.run(self.compute_visible, feed_dict={self.x: batch_x})
if __name__ == '__main__':
    dataset = 'CHB'
    data_file = f'/home/bristena/syn_genomics/datasets/chr13/small_{dataset}.chr13.hap'
    dat = pd.read_csv(data_file,sep= '\t', header=None)
    # import pdb; pdb.set_trace()

    dat = dat.to_numpy()
    pos = dat[:, 0]
    dat = np.transpose(dat[:, 1:])
# data = np.random.permutation(data.train.images)

    rbm = RBM(cd_steps=3)
    import pdb;

    pdb.set_trace()

    rbm.train(X=dat, lr=0.001, epochs=5)
    sample = rbm.generate_samples(1)