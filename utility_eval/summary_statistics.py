import allel
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

np.seterr(divide='ignore', invalid='ignore')

import seaborn as sns
sns.set_style('white')
sns.set_style('ticks')
sns.set_context('notebook')


def convert_h_2_g(data):
    for i in range(data.shape[0]):
        if i%2==0:
            aux = data[i]+data[i+1]
            if i==0:
                gen_data = np.array([aux])
            else:
                gen_data = np.concatenate((gen_data, [aux]), axis =0)
    return gen_data
def get_gen_arr(data):
    for i in range(data.shape[0]-1):
        if i%2==0:
            aux = np.concatenate((data[i].reshape(-1,1), data[i+1].reshape(-1,1)), axis = 1)
            if i==0:
                g_data = np.array([aux])
            else:


                g_data = np.concatenate((g_data, [aux]), axis =0)
    return g_data
def get_bi_genotype(data):
    for item in data:
        import pdb; pdb.set_trace()
        pass
def f_statistic(data):


    subprops = [[*range(170)]]
    a,b,c = allel.weir_cockerham_fst(data.T, subprops)
    fst = a/(a+b+c)
    return fst
def plot_genotype_frequency(r_data,s_data, title):
    vars = r_data.n_variants
    r_pc = r_data.count_het(axis=0)[:]*100/vars
    r_pc = r_pc.astype(np.float)
    # import pdb; pdb.set_trace()

    r_pc2 = np.vstack((r_pc,np.array(['r']*len(r_pc)) )).T
    vars = s_data.n_variants
    s_pc = s_data.count_het(axis=0)[:] * 100 / vars
    s_pc = s_pc.astype(np.float)
    s_pc2 =   np.vstack((s_pc,np.array(['s']*len(s_pc)) )).T
    fig, ax = plt.subplots(figsize=(12, 4))
    pc = np.concatenate([r_pc,s_pc])
    # X,Y = zip(*pc)
    left = np.arange(len(r_pc)+ len(s_pc))
    palette = sns.color_palette()
    pop2color = {'r': palette[0], 's': palette[1]}
    colors = [pop2color['r'] for i in range(len(r_pc))] + [pop2color['s'] for i in range(len(s_pc))]
    ax.bar(left, pc, color = colors)
    # ax.bar(left, s_pc, color = 'blue')

    ax.set_xlim(0, len(r_pc)+ len(s_pc))
    # ax.set_yticks([0, 5, 10, 15, 20, 25, 30, 35, 40])
    ax.set_xlabel('Sample index')
    ax.set_ylabel('Percent calls')
    ax.set_title(title)
    handles = [mpl.patches.Patch(color=palette[0]),
               mpl.patches.Patch(color=palette[1])]
    ax.legend(handles=handles, labels=['real', 'synthetic'], title='Population',
              bbox_to_anchor=(1, 1), loc='upper left')
    plt.show()
def plot_alternate_ac(r_data, s_data, model):
    r_all = r_data.count_alleles()
    s_all = s_data.count_alleles()
    axis_font = { 'size': '20'}
    jsfs = allel.joint_sfs(r_all[:,1], s_all[:,1])
    fig, ax = plt.subplots(figsize=(6, 6))
    allel.plot_joint_sfs(jsfs, ax=ax)
    x_lab = f'Alternate allele count, {model}'
    ax.set_ylabel('Alternate allele count, real', **axis_font)
    ax.set_xlabel(x_lab, **axis_font)
    # ax.set_title('Alternate allele correlation')
    plt.rc('axes', titlesize=15)

    plt.show()
def plot_reference_ac(r_data, s_data, model):
    r_all = r_data.count_alleles()
    s_all = s_data.count_alleles()
    jsfs = allel.joint_sfs(r_all[:,0], s_all[:,0])
    fig, ax = plt.subplots(figsize=(6, 6))
    allel.plot_joint_sfs(jsfs, ax=ax)
    x_lab = f'Reference allele count, {model}'
    ax.set_ylabel('Reference allele count, real')
    ax.set_xlabel(x_lab)
    ax.set_title('Reference allele correlation')
    plt.show()
def plot_heterozygosity(data_lst, data_labels):
    pc = np.array([])
    lengths = []
    for item in data_lst:
        vars = item.n_variants
        item_pc = item.count_het(axis=0)[:]*100/vars
        lengths.append(len(item_pc))
        if pc.size == 0:
            pc = item_pc
        else:
            pc = np.concatenate((pc, item_pc))

    # import pdb; pdb.set_trace()
    left = np.arange(pc.size)
    palette = sns.color_palette()
    colors = [palette[i] for i in range(len(data_lst)) for _ in range(lengths[i])]
    fig, ax = plt.subplots(figsize=(18, 4))
    # sns.despine(ax=ax, offset=10)

    ax.bar(left, pc,width=1.0, color = colors)
    ax.set_xlim(0, len(colors))
    # ax.set_yticks([0, 5, 10, 15, 20, 25, 30, 35, 40])
    ax.set_xlabel('Sample index')
    ax.set_ylabel('Percent calls')
    ax.set_title("Heterozygous")
    handles = [mpl.patches.Patch(color=palette[i]) for i in range(len(data_lst))]
    ax.legend(handles=handles, labels=data_labels, title='Population',
              bbox_to_anchor=(1, 1), loc='upper left')
    plt.show()



def maf(dat):
    maj_count = []

    for j in range(dat.shape[0]):
        item = dat[ j, :]
        maj, min = 0, 0
        unique, count = np.unique(item, return_counts=True)
        for i in range(len(unique)):
            if unique[i] == 0:
                maj += count[i]
        maj_count.append( maj / (dat.shape[1]))

    return maj_count
def plot_maf(data, data_labels):
    maf_lst = []
    print(len(data))
    fig, axis = plt.subplots(len(data)-1, 1, figsize=(18, 5),sharex=True)
    palette = sns.color_palette()
    x = np.arange(data[0].shape[0])

    for i in range(0, len(data)-1):
        ax=axis.flat[i]
        color1 = palette[len(data) - 1]
        color2 = palette[len(data)-i-2]
        maf_plot_helper(data, data_labels, i+1 ,ax, color1, color2)
        print(color1, color2)
    handles = [mpl.patches.Patch(color=palette[len(data) - i -1]) for i in range(len(data))]
    ax.legend(handles=handles, labels=data_labels, title='Samples',
              bbox_to_anchor=(1, 7), loc='upper left')
    plt.tight_layout()
    plt.show()
def maf_plot_helper(data,data_labels, i, ax, color1,color2):
    lst = ['(i) ', '(ii) ', '(iii) ', '(iv) ', '(v) ', 'f) ']
    sns.despine(ax=ax, offset=10)
    x = np.arange(data[0].shape[0])
    linestyles = ['-', '--', '-.', ':', '-']
    orig = maf(data[0])
    mac = maf(data[i])
    print(data_labels[i])
    ax.plot(x, orig, label=data_labels[0], linestyle=linestyles[0], color=color1 )
    ax.plot(x, mac, label=data_labels[i], linestyle=linestyles[i % 3 +1], color = color2)
    # Remove y- and x-label
    h = ax.set_ylabel(lst[i-1], fontweight='bold', labelpad=10)
    h.set_rotation(0)
    ax.set_xlabel('')
    ax.set_yticks([])
    ax.set_xlim(0, x.shape[0])




def plot_average_hudson_fst(r_data, s_data_list,data_labels, pos, blen=1000):
    ac1 = r_data.count_alleles()
    markers = ['o', 'v', 'p', '+', 's', 'x']
    palette = sns.color_palette()
    for i in range(len(s_data_list)):
        ac2 = s_data_list[i].count_alleles()
    # ac2 = s_data.count_alleles()
        fst, se, vb, _ = allel.average_hudson_fst(ac1, ac2, blen=blen)

        # use the per-block average Fst as the Y coordinate
        y = vb
        # import pdb; pdb.set_trace()
        # use the block centres as the X coordinate
        x = allel.moving_statistic(pos, statistic=lambda v: (v[0] + v[-1]) / 2, size=blen)

        label = f'Real vs {data_labels[i]}'
        # plot
        # fig, ax = plt.subplots()
        # sns.despine(ax=ax, offset=10)
        plt.scatter(x, y, marker = markers[i], label = label)
    plt.ylabel('$F_{ST}$')
    plt.xticks([], [])
    # plt.set_xlim(0, pos.max())
    plt.legend()
    plt.show()

def plot_hudson_fst(r_data, s_data_list, data_labels, pos, blen=1000):
    ac1 = r_data.count_alleles()
    markers = ['o', 'v', 'p', '+', 's']
    palette = sns.color_palette()
    for i in range(1,len(s_data_list)):
        ac2 = s_data_list[i].count_alleles()
        # ac2 = s_data.count_alleles()
        num, den = allel.hudson_fst(ac1, ac2)

        # use the per-block average Fst as the Y coordinate
        y = num/den
        # import pdb; pdb.set_trace()
        # use the block centres as the X coordinate
        x = np.arange(y.shape[0])

        label = f'Real vs {data_labels[i]}'
        # plot
        fig, ax = plt.subplots()
        sns.despine(ax=ax, offset=10)
        ax.plot(x, y, marker=markers[i], label=label)
    # plt.set_ylabel('$F_{ST}$')
    # ax.set_xlabel('Chromosome %s position (bp)' % chrom)
    # plt.set_xlim(0, pos.max())
    plt.legend()
    plt.show()

def plot_hw_heterozygosity(r_data, s_data, data_labels):

    x = np.arange(r_data.shape[0])

    af = r_data.count_alleles().to_frequencies()
    exp = allel.heterozygosity_expected(af, ploidy=2)
    palette = sns.color_palette()

    for i in range(len(s_data)):
        fig, ax = plt.subplots(figsize=(18, 4))
        sns.despine(ax=ax, offset=10)
        ax.plot(x, exp, label='Expected')
        obs = allel.heterozygosity_observed(s_data[i])
        label = f'Observed {data_labels[i]}'
        ax.plot(x, obs, label = label, color = palette[i+1])
        plt.legend()
    plt.show()
def ld_prune(gn, size, step, threshold=.1, n_iter=1):
    for i in range(n_iter):
        loc_unlinked = allel.locate_unlinked(gn, size=size, step=step, threshold=threshold)
        n = np.count_nonzero(loc_unlinked)
        n_remove = gn.shape[0] - n
        print('iteration', i+1, 'retaining', n, 'removing', n_remove, 'variants')
        gn = gn.compress(loc_unlinked, axis=0)
    return gn
def plot_pca_coordinates(data_lst,data_label, pc1, pc2):
    fig, ax = plt.subplots(figsize=(18, 4))
    sns.despine(ax=ax, offset=10)
    markers = ['o', 'v', 'p', '+', 's', 'x']
    palette = sns.color_palette()
    for i in range(len(data_lst)):
        # import pdb; pdb.set_trace()
        # d = np.sum(data_lst[i], axis=2)
        # d = ld_prune(d,  size=200, step=50, threshold=.1, n_iter=5)
        d = data_lst[i].to_n_alt()
        is_informative = np.any(d[:, 0, np.newaxis] != d, axis=1)
        d_inf =d.compress(is_informative, axis=0)
        coords, model = allel.pca(d_inf)
        x = coords[:, pc1]
        y = coords[:, pc2]

        ax.plot(x,y, marker = markers[i], color = palette[i], label = data_label[i], linestyle=' ')
    ax.set_xlabel('PC%s' % (pc1 + 1))
    ax.set_ylabel('PC%s' % (pc2 + 1))
    plt.legend()

    plt.show()
# def plot_ld_r(data_lst,data_label):
def plot_sfs(data_lst, data_labels):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.despine(ax=ax, offset=10)
    for i in range(len(data_lst)):
        ac = data_lst[i].count_alleles()
        is_biallelic_01 = ac.is_biallelic_01()[:]

        dat = ac.compress(is_biallelic_01, axis=0) [:,:2]
        sfs = allel.sfs_folded_scaled(dat)
        print(sfs)
        allel.plot_sfs_folded_scaled(sfs, ax=ax, label=data_labels[i], n=dat.sum(axis=1).max())

    ax.legend()
    ax.set_title('Scaled folded site frequency spectrum')
    # workaround bug in scikit-allel re axis naming
    # ax.set_ylim([0,200])
    ax.set_xlabel('minor allele frequency')
    plt.show()
def asarray_ndim(a, *ndims, **kwargs):
    """Ensure numpy array.
    Parameters
    ----------
    a : array_like
    *ndims : int, optional
        Allowed values for number of dimensions.
    **kwargs
        Passed through to :func:`numpy.array`.
    Returns
    -------
    a : numpy.ndarray
    """
    allow_none = kwargs.pop('allow_none', False)
    kwargs.setdefault('copy', False)
    if a is None and allow_none:
        return None
    a = np.array(a, **kwargs)
    if a.ndim not in ndims:
        if len(ndims) > 1:
            expect_str = 'one of %s' % str(ndims)
        else:
            # noinspection PyUnresolvedReferences
            expect_str = '%s' % ndims[0]
        raise TypeError('bad number of dimensions: expected %s; found %s' %
                        (expect_str, a.ndim))
    return a
def ensure_square(dist):
    from scipy.spatial.distance import squareform
    dist = asarray_ndim(dist, 1, 2)
    if dist.ndim == 1:
        dist = squareform(dist)
    else:
        if dist.shape[0] != dist.shape[1]:
            raise ValueError('distance matrix is not square')
    return dist

def plot_ld(real_data, gm_label,colorbar=True, ax=None, imshow_kwargs=None):
    fig, ax = plt.subplots(figsize=(5, 5))
    sns.despine(ax=ax, offset=10)
    # Get R_square values
    #
    data = allel.rogers_huff_r(real_data)**2
    # blank out lower triangle and flip up/down

    m_square = np.tril(data)[::-1,:]
    # import pdb; pdb.set_trace()
    # set up axes
    if ax is None:
        # make a square figure with enough pixels to represent each variant
        x = m_square.shape[0] / plt.rcParams['figure.dpi']
        x = max(m_square, plt.rcParams['figure.figsize'][0])
        fig, ax = plt.subplots(figsize=(x, x))
        fig.tight_layout(pad=0)

    # setup imshow arguments
    if imshow_kwargs is None:
        imshow_kwargs = dict()
    imshow_kwargs.setdefault('interpolation', 'none')
    imshow_kwargs.setdefault('cmap', 'hot_r')
    imshow_kwargs.setdefault('vmin', 0)
    imshow_kwargs.setdefault('vmax', 1)

    # plot as image
    im = ax.imshow(m_square, **imshow_kwargs)
    # tidy upgit
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f'{gm_label} data pairwise LD')
    for s in 'left','bottom', 'right':
        ax.spines[s].set_visible(False)
    if colorbar:
        plt.gcf().colorbar(im, shrink=.5, pad=0)
    plt.show()

    return ax
def plot_ld2(real_data,syn_data, gm_label,colorbar=True, ax=None, imshow_kwargs=None):
    fig, ax = plt.subplots(figsize=(5, 5))
    sns.despine(ax=ax, offset=10)
    # Get R_square values
    #
    r_data = allel.rogers_huff_r(real_data)**2
    s_data = allel.rogers_huff_r(syn_data)**2
    # blank out lower triangle and flip up/down

    m_square = np.tril(r_data)[::-1,:] + np.triu(s_data, 1)[::-1,:]
    # import pdb; pdb.set_trace()
    # set up axes
    if ax is None:
        # make a square figure with enough pixels to represent each variant
        x = m_square.shape[0] / plt.rcParams['figure.dpi']
        x = max(m_square, plt.rcParams['figure.figsize'][0])
        fig, ax = plt.subplots(figsize=(x, x))
        fig.tight_layout(pad=0)

    # setup imshow arguments
    if imshow_kwargs is None:
        imshow_kwargs = dict()
    imshow_kwargs.setdefault('interpolation', 'none')
    imshow_kwargs.setdefault('cmap', 'hot_r')
    imshow_kwargs.setdefault('vmin', 0)
    imshow_kwargs.setdefault('vmax', 1)

    # plot as image
    im = ax.imshow(m_square, **imshow_kwargs)
    # tidy up
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f'Pairwise LD - Real vs {gm_label} data')
    for s in 'left','bottom', 'right':
        ax.spines[s].set_visible(False)
    if colorbar:
        plt.gcf().colorbar(im, shrink=.5, pad=0)
    plt.savefig(f'ld_chb_real_v_{gm_label}.png')

    return ax

def plot_pair_distance(dist,gm_label, labels=None, colorbar=True, ax=None,
                           imshow_kwargs=None):

    dist_square = ensure_square(dist)
    # set up axes
    if ax is None:
        # make a square figure
        x = plt.rcParams['figure.figsize'][0]
        fig, ax = plt.subplots(figsize=(x, x))
        fig.tight_layout()

    # setup imshow arguments
    if imshow_kwargs is None:
        imshow_kwargs = dict()
    imshow_kwargs.setdefault('interpolation', 'none')
    imshow_kwargs.setdefault('cmap', 'rainbow')
    imshow_kwargs.setdefault('vmin', 0)
    imshow_kwargs.setdefault('vmax', 40)

    # plot as image
    im = ax.imshow(dist_square, **imshow_kwargs)

    # tidy up
    if labels:
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=90)
        ax.set_yticklabels(labels, rotation=0)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
    if colorbar:
        plt.gcf().colorbar(im, shrink=.5)
    ax.set_title(f'{gm_label} data')
    return ax
def pair_distance(data_lst, data_labels):

    for i in range(len(data_lst)):
        gn = data_lst[i].to_n_alt()
        dist = allel.pairwise_distance(gn, metric = 'euclidean')
        plot_pair_distance(dist, data_labels[i])
        plt.show()
if __name__ == "__main__":
    dataset = 'CEU'
    data_gn = []
    # get real data file
    r_file = f'../datasets/chr13/small_{dataset}.chr13.hap'
    r_data = pd.read_csv(r_file, sep='\t', header=None)
    r_data = r_data.to_numpy()
    # import pdb; pdb.set_trace()

    pos = r_data[:, 0]
    r_data = np.transpose(r_data[:, 1:201])
    r_gen_dat = get_gen_arr(r_data)
    r_g_arr = allel.GenotypeArray(np.transpose(r_gen_dat, axes=(1, 0, 2)))
    data_gn.append(r_data)
    plot_ld2(r_data,r_data, 'bla')
    # get synthetic data
    gan_models = ['rbm', 'gan', 'rbm_recomb','gan_recomb']  # , 'wgangp']
    s_lst = []
    s_data_lst = []

    # # allel.plot_pairwise_ld(r, imshow_kwargs={'cmap': 'hot_r'})
    # # # get synthetic recombination data
    recomb_file = f'../syn_data/recomb_hap_chr13_{dataset}.csv'
    rec_data = pd.read_csv(recomb_file, sep=' ', header=None)
    rec_data = rec_data.to_numpy()[:200]
    # import pdb; pdb.set_trace()
    rec_gen_dat = get_gen_arr(rec_data)
    rec_g_arr = allel.GenotypeArray(np.transpose(rec_gen_dat, axes=(1, 0, 2)))
    s_lst.append(rec_g_arr)
    data_gn.append(rec_data)
    rec = allel.rogers_huff_r(rec_data)**2
    # plot_ld(r_data,rec_data, 'bla')
    #
    #
    #
    for item in gan_models:
        s_file = f'../syn_data/{item}_out_hap_{dataset}.csv'
        s_data = pd.read_csv(s_file, sep='\t', header=None)
        s_data = s_data.to_numpy()
        pos = s_data[:, 0]
        s_data = np.transpose(s_data[:, 1:201])
        s_data_lst.append(s_data)
        s_gen_dat = get_gen_arr(s_data)
        s_g_arr = allel.GenotypeArray(np.transpose(s_gen_dat, axes=(1, 0, 2)))
        s_lst.append(s_g_arr)
    #
    # #
    #
    #
    data_lst = [r_g_arr] + s_lst
    data_labels = ['Real'] + ['Recomb'] + [s.upper() for s in gan_models[:-2]]+ ['Rec-RBM', 'Rec-GAN']
    #
    # pair_distance(data_lst, data_labels)



    #plot LD1
    data_gn = data_gn + s_data_lst
    for i in range(len(data_gn)):
    #     plot_ld(data_gn[i], data_labels[i])
        plot_ld2(data_gn[0], data_gn[i], data_labels[i])

        # #
    # # plot_sfs(data_lst, data_labels)
    # # plot_maf(data_lst, data_labels)
    # # plot_maf(data_lst, data_labels)
    # # plot_average_hudson_fst(r_g_arr, data_lst, data_labels,  np.arange(r_g_arr.shape[0]))
    # plot_alternate_ac(r_g_arr, r_g_arr, 'real')
    # # # plot_reference_ac(r_g_arr, r_g_arr, 'real')
    # #
    # for i in range(len(data_lst)-1):
    #     plot_alternate_ac(r_g_arr, data_lst[i+1], data_labels[i+1])
    #     # plot_reference_ac(r_g_arr, s_lst[i], gan_models[i])

    # plot_pca_coordinates(data_lst, data_labels, 0, 1)
    # plot_pca_coordinates(data_lst, data_labels, 1, 2)
    # plot_pca_coordinates(data_lst, data_labels, 2, 3)
    # plot_pca_coordinates(data_lst, data_labels, 0, 4)