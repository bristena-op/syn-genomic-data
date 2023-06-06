"""
Command-line interface for running privacy evaluation under a membership inference adversary
"""

import json
from numpy import arange
from numpy.random import choice
from argparse import ArgumentParser

from priv_eval.datagen import load_s3_data_as_df, load_local_data_as_df
from priv_eval.utils import json_numpy_serialzer
from priv_eval.evaluation_framework import *

from priv_eval.feature_sets.independent_histograms import HistogramFeatureSet
from priv_eval.feature_sets.model_agnostic import NaiveFeatureSet, EnsembleFeatureSet
from priv_eval.feature_sets.bayes import CorrelationsFeatureSet

from models.rbm import RBM
from models.gan import GAN
from models.recomb_model import RecombModel
from models.rec_rbm import RecRBM

from priv_eval.mia_classifier import MIAttackClassifierLinearSVC, MIAttackClassifierLogReg, MIAttackClassifierRandomForest, MIAttackClassifierSVC, MIAttackClassifierKNN

from warnings import filterwarnings
filterwarnings('ignore')

from logging import getLogger
from logging.config import fileConfig

import os

cwd = path.dirname(__file__)
logconfig = path.join('logging_config.ini')
fileConfig(logconfig)
logger = getLogger()


if __name__ == "__main__":
    argparser = ArgumentParser()
    datasource = argparser.add_mutually_exclusive_group()
    datasource.add_argument('--s3name', '-S3', type=str, choices=['adult', 'census', 'credit', 'alarm', 'insurance'], help='Name of the dataset to run on')
    datasource.add_argument('--datapath', '-D', type=str, default = '../datasets/chr13/small_CEU.chr13', help='Relative path to cwd of a local data file')
    argparser.add_argument('--attack_model', '-AM', type=str, default='ANY', choices=['RandomForest', 'LogReg', 'LinearSVC', 'SVC', 'KNN', 'ANY'])
    argparser.add_argument('--runconfig', '-RC', default='runconfig_mia.json', type=str, help='Path relative to cwd of runconfig file')
    argparser.add_argument('--outdir', '-O', default='priv_out', type=str, help='Path relative to cwd for storing output files')
    args = argparser.parse_args()

    # Load runconfig
    with open(path.join(cwd,args.runconfig)) as f:
        runconfig = json.load(f)
    print('Runconfig:')
    print(runconfig)
    try:
        os.makedirs(args.outdir)
    except FileExistsError:
        # directory already exists
        pass
    # Load data
    if args.s3name is not None:
        RawDF, metadata = load_s3_data_as_df(args.s3name)
        dname = args.s3name
    else:
        RawDF, metadata = load_local_data_as_df(path.join(cwd, args.datapath))
        dname = args.datapath.split('/')[-1]
    RawDF['ID'] = [f'ID{i}' for i in arange(len(RawDF))]
    RawDF = RawDF.set_index('ID')
    import pdb; pdb.set_trace()

    print(f'Loaded data {dname}:')
    print(RawDF.info())
    # Randomly select nt target records T = (t_1, ..., t_(nt))
    targetIDs = choice(list(RawDF.index), size=runconfig['nTargets'], replace=False).tolist()
    Targets = RawDF.loc[targetIDs, :]

    # Drop targets from sample population
    RawDFdropT = RawDF.drop(targetIDs)

    # Add a crafted outlier target to the evaluation set
    targetCraft = craft_outlier(RawDF, runconfig['sizeTargetCraft'])
    targetIDs.extend(list(set(targetCraft.index)))
    Targets = Targets.append(targetCraft)

    # Sample adversary's background knowledge RawA
    rawAidx = choice(list(RawDFdropT.index), size=runconfig['sizeRawA'], replace=False).tolist()

    # Sample k independent target test sets
    rawTindices = [choice(list(RawDFdropT.index), size=runconfig['sizeRawT'], replace=False).tolist() for nr in range(runconfig['nIter'])]

    # List of candidate generative models to evaluate
    gmList = []
    for gm, paramsList in runconfig['generativeModels'].items():
        if gm == 'RecRBM':
            for params in paramsList:
                gmList.append(RecRBM(*params))
        elif gm == 'RBM':
            for params in paramsList:
                gmList.append(RBM(*params))
        elif gm == 'Recomb':
            for params in paramsList:
                gmList.append(RecombModel(*params))
        # elif gm == 'GAN':
        #     for params in paramsList:
        #         gmList.append(GAN(*params))
        # elif gm == 'PrivBayes':
        #     for params in paramsList:
        #         gmList.append(PrivBayes(*params))
        # elif gm == 'CTGAN':
        #     for params in paramsList:
        #         gmList.append(CTGAN(metadata, *params))
        else:
            raise ValueError(f'Unknown GM {gm}')

    for GenModel in gmList:
        print(f'----- {GenModel.__name__} -----')

        FeatureList = [NaiveFeatureSet(GenModel.datatype), HistogramFeatureSet(GenModel.datatype, metadata), CorrelationsFeatureSet(GenModel.datatype, metadata), EnsembleFeatureSet(GenModel.datatype, metadata)]

        prior = {LABEL_IN: runconfig['prior']['IN'], LABEL_OUT: runconfig['prior']['OUT']}

        if args.attack_model == 'RandomForest':
            AttacksList = [MIAttackClassifierRandomForest(metadata, prior, F) for F in FeatureList]
        elif args.attack_model == 'LogReg':
            AttacksList = [MIAttackClassifierLogReg(metadata, prior, F) for F in FeatureList]
        elif args.attack_model == 'LinearSVC':
            AttacksList = [MIAttackClassifierLinearSVC(metadata, prior, F) for F in FeatureList]
        elif args.attack_model == 'SVC':
            AttacksList = [MIAttackClassifierSVC(metadata, prior, F) for F in FeatureList]
        elif args.attack_model == 'KNN':
            AttacksList = [MIAttackClassifierKNN(metadata, prior, F) for F in FeatureList]
        elif args.attack_model == 'ANY':
            AttacksList = []
            for F in FeatureList:
                AttacksList.extend([MIAttackClassifierRandomForest(metadata, prior, F),
                                    MIAttackClassifierLogReg(metadata, prior, F),
                                    MIAttackClassifierKNN(metadata, prior, F)])
        else:
            raise ValueError(f'Unknown AM {args.attack_model}')

        # Run privacy evaluation under MIA adversary
        results = evaluate_mia(GenModel, AttacksList, RawDFdropT, Targets, targetIDs, rawAidx, rawTindices,
                               runconfig['sizeRawT'], runconfig['sizeSynT'], runconfig['nSynT'],
                               runconfig['nSynA'], runconfig['nShadows'], metadata)

        outfile = f"{dname}{GenModel.__name__}MIA2"

        with open(path.join(f'{args.outdir}', f'{outfile}.json'), 'w') as f:
            json.dump(results, f, indent=2, default=json_numpy_serialzer)