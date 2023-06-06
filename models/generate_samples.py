import pandas as pd
import numpy as np



from models.rec_rbm import RecRBM
# from rec_gan import REGAN
from models.recomb_model import RecombModel
from models.rbm import RBM
from models.vae import VAE
from priv_eval.datagen import load_local_data_as_df

def gen_samples(model, dat, n_samples):
    pass
def gen_samples_recomb(model, dat, n_samples):
    pass

if __name__ == '__main__':
    dataset = 'CHB'
    data_file = f'datasets/chr13/small_{dataset}.chr13'
    # dat = pd.read_csv(data_file,sep= '\t', header=None)
    # import pdb; pdb.set_trace()
    RawDF, metadata = load_local_data_as_df(data_file)

    RawDF['ID'] = [f'ID{i}' for i in np.arange(len(RawDF))]
    RawDF = RawDF.set_index('ID')


    # model = VAE(100)
    # import pdb; pdb.set_trace()
    # model.fit(RawDF)
    # samples = model.generate_samples(100)
    import pdb; pdb.set_trace()





    # dat = dat.to_numpy()
    # pos = dat[:, 0]
    n_samples = 340
    # dat = dat[:,1:]
    # dat = np.transpose(dat)
    # dat = pd.DataFrame(data=dat[:,1:], index=dat[:,0])
    recomb = RecombModel('datasets/chr13/small_genetic_map_chr13_combined_b36.txt')
    recomb.fit(RawDF)
    recomb_data = recomb.generate_samples(10)
    resf_recomb = f'../syn_data/recomb_for_rbm_{dataset}_test1.csv'
    # np.savetxt(resf_recomb, recomb_data, delimiter='\t', fmt='%i')
    # # dat = pd.read_csv(resf_recomb, sep='\t', header=None)
    # import pdb; pdb.set_trace()
    # dat = recomb_data
    # dat = dat.to_numpy()
    # pos = dat[:, 0]
    # dat = np.transpose(dat[:, 1:])
    # import pdb; pdb.set_trace()


    # model = RecRBM(1000, 100, 0.01, 0.95, 1.0, 100, 32)
    # errs = model.fit(dat)
    # samples = model.generate_samples(n_samples)
    # import pdb;pdb.set_trace()
    # # samples = samples.astype(int)
    # # sample = np.insert(samples, 0, dat.index, axis=1)
    # res_file = f'syn_data/rbm_recomb_out_{dataset}_test1.csv'
    # np.savetxt(res_file, sample, delimiter='\t', fmt='%i')

    # model = RBM(1000, 100, 0.01, 0.95, 1.0, 2000, 32)
    # errs = model.fit(dat)
    # samples = model.generate_samples(n_samples)
    # import pdb; pdb.set_trace()
    # samples = samples.astype(int)
    # sample = np.insert(samples, 0, dat.index, axis=1)
    # res_file = f'syn_data/rbm_recomb_out_{dataset}_test1.csv'
    # np.savetxt(res_file, sample, delimiter='\t', fmt='%i')

    # model = REGAN(8000, 32, 200)
    # model.fit(dat)
    # import pdb; pdb.set_trace()
    #
    # samples = np.around(model.generate_samples(n_samples), decimals=0)
    # samples = samples.astype(int)
    # sample = np.insert(samples, 0, pos, axis=1)
    # res_file = f'../syn_data/gan_recomb_out_{dataset}_test.csv'
    # np.savetxt(res_file, sample, delimiter='\t', fmt='%i')