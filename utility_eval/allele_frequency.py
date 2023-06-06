import pandas as pd
import numpy as np

def af(data_file):
    dat = pd.read_csv(data_file,skiprows=1,sep= '\t', header=None)
    dat = dat.to_numpy()

    pos = dat[:, 0]
    dat = dat[:, 1:]
    maj_count = []
    min_count = []
    # print(dat.shape)
    for j in range(dat.shape[1]):
        item = dat[j, :]
        maj, min = 0,0
        unique, count = np.unique(item, return_counts=True)
        for i in range(len(unique)):
            if unique[i] ==0:
                maj+=count[i]*2
            elif unique[i] == 1:
                maj+=count[i]
                min+=count[i]
            elif unique[i]==2:
                min+=count[i]
        maj_count.append((pos[j],maj/(2*dat.shape[1])))
        min_count.append(min/(2*dat.shape[1]))

    print(maj_count)
def hap_af(data_file):
    dat = pd.read_csv(data_file,skiprows=1,sep= '\t', header=None)
    dat = dat.to_numpy()
    import pdb; pdb.set_trace()
    pos = dat[:, 0]
    dat = dat[:, 1:]
    print(np.unique(dat,return_counts=True))
    maj_count = []
    min_count = []
    # print(dat.shape)
    for j in range(dat.shape[1]):
        item = dat[j, :]
        maj, min = 0,0
        unique, count = np.unique(item, return_counts=True)
        for i in range(len(unique)):
            if unique[i] ==0:
                maj+=count[i]

            elif unique[i]==1:
                min+=count[i]
        maj_count.append((pos[j],maj/(dat.shape[1])))
        min_count.append(min/(dat.shape[1]))
    import pdb; pdb.set_trace()

    print(maj_count)




if __name__ == '__main__':
    # af('../datasets/chr13/small_genotypes_chr13_CEU.txt')
    # af('../syn_data/gan_out_CEU.csv')
    hap_af('../datasets/chr13/small_CHB.chr13.hap')
    hap_af('/Users/bristena/PycharmProjects/syn_genomic_data/syn_data/gan_out_hap_CHB.csv')
    hap_af('/Users/bristena/PycharmProjects/syn_genomic_data/syn_data/rbm_out_hap_CHB.csv')
    hap_af('/Users/bristena/PycharmProjects/syn_genomic_data/syn_data/wgan_hap_out_CHB.csv')
    hap_af('/Users/bristena/PycharmProjects/syn_genomic_data/syn_data/wgangp_hap_out_CHB.csv')
    hap_af('/Users/bristena/PycharmProjects/syn_genomic_data/syn_data/wgangp_dna_hap_out_CHB.csv')


    # hap_af('/Users/bristena/PycharmProjects/syn_genomic_data/syn_data/recomb_random_chr13_CEU.txt')