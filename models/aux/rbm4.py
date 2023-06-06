from sklearn.neural_network import BernoulliRBM
import numpy as np
import pandas as pd


if __name__ == '__main__':
    dataset = 'CHB'
    data_file = f'/home/bristena/syn_genomics/datasets/chr13/small_{dataset}.chr13.hap'
    dat = pd.read_csv(data_file,sep= '\t', header=None)
    # import pdb; pdb.set_trace()

    dat = dat.to_numpy()
    pos = dat[:, 0]
    dat = np.transpose(dat[:, 1:])
    rbm = BernoulliRBM(n_components=1000, learning_rate=0.001, batch_size=32)
    syn = rbm.fit_transform(dat)
    import pdb;pdb.set_trace()
    print (syn)