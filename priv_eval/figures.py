import json
import matplotlib.pyplot as plt
import seaborn as sns

from numpy import mean, ceil, arange, median, sqrt
from pandas import DataFrame, to_numeric, pivot_table, concat, read_csv
from itertools import combinations, combinations_with_replacement
from scipy import stats
from math import log10
from husl import hex_to_husl

from glob import glob
from os import path
# from husl import hex_to_husl

GMS = [ 'RecombModel', 'RBM', 'RecRBM']
GMNAMES = {'RecombModel': 'Recomb',
           'RBM': 'RBM',
           'RecRBM': 'REC-RBM'
           }
FEATURESET = ['Naive', 'Histogram', 'Correlations', 'Ensemble', 'None']
ATTACKS = ['RandomForestClassifier', 'LogisticRegression', 'KNeighborsClassifier']
ACNAMES = {'RandomForestClassifier': '$\mathtt{RandForest}$',
           'LogisticRegression': '$\mathtt{LogReg}$',
           'KNeighborsClassifier': '$\mathtt{KNN}$'}


FSIZELABELS = 26
FSIZETICKS = 24
COLOURS = ['#BB5566',  '#004488', '#DDAA33', '#000000']
MARKERS = ['o', 'X', 'D', 'P', 'X']
cpalette = sns.color_palette(COLOURS)
cmap_sequential = sns.light_palette(hex_to_husl(COLOURS[1]), input="husl", as_cmap=True)
cmap_diverging = sns.diverging_palette(h_neg=227, h_pos=4, s=73, l=60, n=12, as_cmap=True)
cmap_diverging.set_bad('white')



def load_results(directory, dname, attack='MIA'):
    """
    Load results of privacy evaluation
    :param directory: str: path/to/result/files
    :param dname: str: name of dataset for which to load results
    :return: DataFrame: results
    """
    files = glob(path.join(directory, f'{dname}*.json'))

    resList = []

    for fname in files:
        if attack == 'MIA':
            gm = fname.split('/')[-1].split(attack)[0].split(dname)[-1]

            if gm in GMS :
                with open(fname) as file:
                    rd = json.load(file)
                # import pdb; pdb.set_trace()
                rdf = parse_results_mia(rd)
                rdf['GenerativeModel'] = gm
                rdf['Dataset'] = dname
                resList.append(rdf)

        # elif attack == 'MLE-AI':
        #     f = fname.split('/')[-1]
        #     gm = f.split(attack)[0].split(dname)[-1]
        #     sensitive = f.split(attack)[-1].split('.')[0]
        #
        #     if gm in GMS :
        #         with open(fname) as file:
        #             rd = json.load(file)
        #
        #         rdf = parse_results_attr_inf(rd)
        #         rdf['GenerativeModel'] = gm
        #         rdf['Dataset'] = dname
        #         rdf['SensitiveAttribute'] = sensitive
        #         resList.append(rdf)

        else:
            raise ValueError('Unknown attack type')
    # import pdb; pdb. set_trace()
    return concat(resList)

def parse_results_mia(resDict):
    """ Parse results from privacy evaluation under MIA and aggregate by target and test run"""
    dfList = []

    for am, res in resDict.items():
        # Aggregate data by target and test run and average
        resDF = DataFrame(res).groupby(['TargetID', 'TestRun']).agg(mean).reset_index()
        # import pdb; pdb.set_trace()
        # Get attack model details
        fset = [m for m in FEATURESET if m in am][0]
        amc = am.split(fset)[0]
        resDF['AttackClassifier'] = amc
        resDF['FeatureSet'] = fset

        dfList.append(resDF)

    results = concat(dfList)

    results['RecordPrivacyLossSyn'] = to_numeric(results['RecordPrivacyLossSyn'])
    results['RecordPrivacyLossRaw'] = to_numeric(results['RecordPrivacyLossRaw'])
    results['RecordPrivacyGain'] = to_numeric(results['RecordPrivacyGain'])
    results['ProbSuccess'] = to_numeric(results['ProbSuccess'])

    return results
def mia_res_by_target_figure(data, metric, targetID):
    fig, axis = plt.subplots(1, 3, figsize=(30, 4), sharey='all')
    pltdata = data[data['TargetID'] == targetID]
    # dname = 'adult'
    # ddata = dgroups.get_group(dname)
    agroups = pltdata.groupby(['AttackClassifier'])
    aname = 'RandomForestClassifier'
    adata = agroups.get_group(aname)

    ax = axis.flat[2]
    swarmplot(adata, metric, aname, ax, GMS)
    for i, (aname, adata) in enumerate(agroups):
        ax = axis.flat[i]

        if aname in ATTACKS:
            swarmplot(adata, metric, aname, ax, GMS)
    axis[0].set_ylabel('$\overline{\mathtt{PG}}_{\mathbf{t}}$', fontsize=FSIZELABELS)
    for tick in axis[0].yaxis.get_major_ticks():
        tick.label.set_fontsize(FSIZETICKS)

    fig.subplots_adjust(wspace=0.02)
    plt.show()
    return fig
def swarmplot(data, metric, aname,  ax, models):

    sns.swarmplot(data=data, y=metric,
                  x='GenerativeModel', hue='FeatureSet',
                  hue_order=FEATURESET, order=models,
                  ax=ax, dodge=True)

    # Remove legend
    ax.get_legend().remove()

    # Make title
    ax.set_title(f"CEU: {ACNAMES[aname]}", fontsize=FSIZELABELS)

    # Remove y- and x-label
    ax.set_ylabel('')
    ax.set_xlabel('')

    # Rename GMs
    ax.set_xticklabels([GMNAMES[gm.get_text()] for gm in ax.get_xticklabels()], fontsize=FSIZELABELS)

    # Resize y-tick labels
    ax.set_ylim(-0.05, 0.55)
    ax.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])

    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(FSIZETICKS)
def mia_res_all_figure(data, metric,targetID):
    fig, axis = plt.subplots(1, 3, figsize=(30, 4), sharey='all')
    pltdata = data[data['TargetID'] != targetID]
    # dname = 'adult'
    # ddata = dgroups.get_group(dname)
    agroups = pltdata.groupby(['AttackClassifier'])
    aname = 'RandomForestClassifier'
    adata = agroups.get_group(aname)

    ax = axis.flat[2]
    swarmplot(adata, metric, aname, ax, GMS)
    for i, (aname, adata) in enumerate(agroups):
        ax = axis.flat[i]

        if aname in ATTACKS:
            swarmplot(adata, metric, aname, ax, GMS)
    axis[0].set_ylabel('$\overline{\mathtt{PG}}_{\mathbf{t}}$', fontsize=FSIZELABELS)
    for tick in axis[0].yaxis.get_major_ticks():
        tick.label.set_fontsize(FSIZETICKS)

    fig.subplots_adjust(wspace=0.02)
    plt.show()
    return fig


if __name__=='__main__':
    resList = load_results("../priv_out", "small_CEU.chr13")
    mia_res_by_target_figure(resList, 'RecordPrivacyGain', 'Crafted')
    mia_res_all_figure(resList, 'RecordPrivacyGain', 'Crafted')