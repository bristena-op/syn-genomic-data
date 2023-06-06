#TODO: modify the code from NDSS submission to predict values rather that just calculate the coefficients for linear regression; add more prediction models.


from numpy.random import choice
import json
from os import path
import numpy as np
import random

from priv_eval.datagen import load_local_data_as_df, load_s3_data_as_df
from priv_eval.utils import json_numpy_serialzer


from priv_eval.genomic_inference_models import PubLDModel, RecombModel, DirectCondProbModel
from priv_eval.reconstruction import AttributeInferenceAttackLinearRegression


from logging import getLogger
from logging.config import fileConfig
cwd = path.dirname(__file__)

logconfig = path.join('logging_config.ini')
fileConfig(logconfig)
logger = getLogger()


def inference_real_data(realDF, metadata, nTargets, sensitiveAttribute, outdir ):

    results = { 'Attribute': sensitiveAttr,
        'LinReg':
                {
                    'Target': [],
                    'TrueValue': [],
                    'MLERawT': [],
                    'SigmaRawT': [],
                    'ProbCorrectRawT': [],
                    'PredictedValue': [],
                },
        'LDAF':{
                    'Target': [],
                    'TrueValue': [],
                    'MLERawT': [],
                    'SigmaRawT': [],
                    'ProbCorrectRawT': [],
                    'PredictedValue': [],
                },
        'Recomb':{
                    'Target': [],
                    'TrueValue': [],
                    'MLERawT': [],
                    'SigmaRawT': [],
                    'ProbCorrectRawT': [],
                    'PredictedValue': [],
                }
    }



    targetIDs = choice(list(realDF.index), size=nTargets, replace=False).tolist()
    targetRecords = realDF.loc[targetIDs, :]
    realWithoutTargets = realDF.drop(targetIDs)

    linReg_attack = AttributeInferenceAttackLinearRegression(sensitiveAttribute, metadata, realWithoutTargets)
    linReg_attack.train(realWithoutTargets)

    for tid in targetIDs:
        t = targetRecords.loc[[tid], :]
        targetAux = t.drop_duplicates().drop(sensitiveAttribute, axis=1)
        targetSecret = t.drop_duplicates().loc[tid, sensitiveAttribute]

        results['LinReg']['Target'].append(tid)
        results['LinReg']['TrueValue'].append(targetSecret)
        results['LinReg']['SigmaRawT'].append(linReg_attack.sigma)
        results['LinReg']['MLERawT'].extend(linReg_attack.attack(targetAux).tolist())
        results['LinReg']['ProbCorrectRawT'].extend(
            linReg_attack.get_likelihood(targetAux, targetSecret).tolist())
        results['LinReg']['PredictedValue'] .append(linReg_attack.get_predicted_value(targetAux, targetSecret))
    outfile = f"{dname}MLEAI"

    with open(path.join(f'{outdir}', f'{outfile}.json'), 'w') as f:
        json.dump(results, f, indent=2, default=json_numpy_serialzer)
if __name__=="__main__":
    datapath = '../datasets/chr13/small_CHB.chr13'
    outdir = '/Users/bristena/PycharmProjects/syn_genomic_data/priv_out'
    rawDF, metadata = load_local_data_as_df(path.join(cwd, datapath))
    dname = datapath.split('/')[-1]
    rawDF['ID'] = [f'ID{i}' for i in np.arange(len(rawDF))]
    rawDF = rawDF.set_index('ID')
    sensitiveAttr = random.choice(rawDF.columns)
    import pdb; pdb.set_trace()
    inference_real_data(rawDF,metadata,5, sensitiveAttr, outdir)