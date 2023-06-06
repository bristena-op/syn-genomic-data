"""
Command-line interface for running privacy evaluation under a membership inference adversary
"""

import json
from argparse import ArgumentParser
from pandas import Series
from numpy import arange
from numpy.random import choice

from priv_eval.utils import json_numpy_serialzer
from priv_eval.datagen import load_local_data_as_df, load_s3_data_as_df
from priv_eval.evaluation_framework import *

from models.rbm import RBM
from models.gan import GAN
from models.recomb_model import RecombModel
from models.rec_rbm import RecRBM


from warnings import filterwarnings
filterwarnings('ignore')

from logging import getLogger
from logging.config import fileConfig

cwd = path.dirname(__file__)

logconfig = path.join('logging_config.ini')
fileConfig(logconfig)
logger = getLogger()

PROCESSES = 16

if __name__ == "__main__":
    argparser = ArgumentParser()
    datasource = argparser.add_mutually_exclusive_group()
    datasource.add_argument('--s3name', '-S3', type=str, choices=['adult', 'census', 'credit', 'alarm', 'insurance'], help='Name of the dataset to run on')
    datasource.add_argument('--datapath', '-D', type=str, default = '../datasets/chr13/small_CHB.chr13', help='Relative path to cwd of a local data file')
    argparser.add_argument('--runconfig', '-RC', default='runconfig_mleai_pos.json', type=str, help='Path relative to cwd of runconfig file')
    argparser.add_argument('--outdir', '-O', default='priv_out', type=str, help='Path relative to cwd for storing output files')
    args = argparser.parse_args()

    # Load runconfig
    with open(path.join(cwd, args.runconfig)) as f:
        runconfig = json.load(f)
    print('Runconfig:')
    print(runconfig)

    # Load data
    if args.s3name is not None:
        rawDF, metadata = load_s3_data_as_df(args.s3name)
        dname = args.s3name
    else:
        rawDF, metadata = load_local_data_as_df(path.join(cwd, args.datapath))
        dname = args.datapath.split('/')[-1]
    rawDF['ID'] = [f'ID{i}' for i in arange(len(rawDF))]
    rawDF = rawDF.set_index('ID')

    print(f'Loaded data {dname}:')
    print(rawDF.info())

    # Randomly select nt target records T = (t_1, ..., t_(nt))
    targetIDs = choice(list(rawDF.index), size=runconfig['nTargets'], replace=False).tolist()
    targetRecords = rawDF.loc[targetIDs, :]

    # Drop targets from sample population
    rawWithoutTargets = rawDF.drop(targetIDs)

    # Add a crafted outlier target to the evaluation set
    targetCraft = craft_outlier(rawDF, runconfig['sizeTargetCraft'])
    targetIDs.extend(list(set(targetCraft.index)))
    targetRecords = targetRecords.append(targetCraft)

    # Sample adversary's background knowledge RawA (needed for prior calculation only)
    if runconfig['prior']['type'] == 'sampled':
        rawAidx = choice(list(rawWithoutTargets.index), size=runconfig['prior']['sizeRawA'], replace=False).tolist()
        rawA = rawWithoutTargets.loc[rawAidx, :]

    elif runconfig['prior']['type'] == 'uniform':
        sensitiveRange = (rawDF[runconfig['sensitiveAttribute']].min(), rawDF[runconfig['sensitiveAttribute']].max())
        rawA = Series(arange(*sensitiveRange), name=runconfig['sensitiveAttribute']).to_frame()

    else:
        raise ValueError(f"Unknown prior type: {runconfig['prior']['type']}")

    # Sample k independent target test sets RawT
    rawTindices = [choice(list(rawWithoutTargets.index), size=runconfig['sizeRawT'], replace=False).tolist() for nr in range(runconfig['nIter'])]

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
        else:
            raise ValueError(f'Unknown GM {gm}')

    for GenModel in gmList:
        print(f'----- {GenModel.__name__} -----')

        # Run privacy evaluation under AI adversary
        results = evaluate_ai(GenModel, rawWithoutTargets, targetRecords, targetIDs, rawA, rawTindices,
                              runconfig['sensitiveAttribute'], runconfig['sizeSynT'], runconfig['nSynT'], metadata)

        outfile = f"{dname}{GenModel.__name__}MLE-AI{runconfig['sensitiveAttribute']}"

        with open(path.join(f'{args.outdir}', f'{outfile}.json'), 'w') as f:
            json.dump(results, f, indent=2, default=json_numpy_serialzer)

