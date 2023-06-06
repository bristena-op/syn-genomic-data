"""Parent class for launching a membership inference attack on the output of a generative model"""
from pandas import DataFrame
from numpy import ndarray, concatenate, stack, array, round
from os import path

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import ShuffleSplit

from priv_eval.datagen import convert_df_to_array
from priv_eval.attack_model import PrivacyAttack

import logging
from logging.config import fileConfig
dirname = path.dirname(__file__)
logconfig = path.join(dirname, 'logging_config.ini')
fileConfig(logconfig)
logger = logging.getLogger(__name__)

LABEL_OUT = 0
LABEL_IN = 1


class MIAttackClassifier(PrivacyAttack):
    """"Parent class for membership inference attack on the output of a generative model using sklearn classifier"""
    def __init__(self, Distinguisher, metadata, priorProbabilities, FeatureSet=None):

        self.Distinguisher = Distinguisher
        self.FeatureSet = FeatureSet
        self.ImputerCat = SimpleImputer(strategy='most_frequent')
        self.ImputerNum = SimpleImputer(strategy='median')
        self.metadata = metadata

        self.priorProbabilities = priorProbabilities

        self.trained = False
        self.__name__ = f'{self.Distinguisher.__class__.__name__}{self.FeatureSet.__class__.__name__}'

    def _get_prior_probability(self, secret):
        """ Get prior probability of the adversary guessing the target's secret"""
        try:
            return self.priorProbabilities[secret]
        except:
            return 0

    def train(self, synA, labels):
        """Train a membership inference attack on a labelled training set"""
        # import pdb; pdb.set_trace()
        if self.FeatureSet is not None:
            synA = stack([self.FeatureSet.extract(s) for s in synA])
        else:
            if isinstance(synA[0], DataFrame):
                synA = [self._impute_missing_values(s) for s in synA]
                synA = stack([convert_df_to_array(s, self.metadata).flatten() for s in synA])
            else:
                synA = stack([s.flatten() for s in synA])
        if not isinstance(labels, ndarray):
            labels = array(labels)

        self.Distinguisher.fit(synA, labels)

        logger.debug('Finished training MIA distinguisher')
        self.trained = True

        del synA, labels

    def attack(self, synT):
        """Makes a guess about whether target data on which attack was trained was in the original data
       from which samples of synthetic dataset were produced"""
        assert self.trained, 'Attack must first be trained on some random data before can predict membership of target data'
        if self.FeatureSet is not None:
            synT = stack([self.FeatureSet.extract(s) for s in synT])
        else:
            if isinstance(synT[0], DataFrame):
                synT = stack([convert_df_to_array(s, self.metadata).flatten() for s in synT])
            else:
                synT = stack([s.flatten() for s in synT])

        return round(self.Distinguisher.predict(synT), 0).astype(int).tolist()

    def get_probability_of_success(self, synT, secret):
        """Calculate probability that attacker correctly predicts whether target was present in model's training data"""
        assert self.trained, 'Attack must first be trained on some random data before can predict membership of target data'
        if self.FeatureSet is not None:
            synT = stack([self.FeatureSet.extract(s) for s in synT])
        else:
            if isinstance(synT[0], DataFrame):
                synT = stack([convert_df_to_array(s, self.metadata).flatten() for s in synT])
            else:
                synT = stack([s.flatten() for s in synT])

        probs = self.Distinguisher.predict_proba(synT)

        return [p[s] for p,s in zip(probs, secret)]

    def _impute_missing_values(self, df):
        cat_cols = list(df.select_dtypes(['object', 'category']))
        if len(cat_cols) > 0:
            self.ImputerCat.fit(df[cat_cols])
            df[cat_cols] = self.ImputerCat.transform(df[cat_cols])

        num_cols = list(df.select_dtypes(['int', 'float']))
        if len(num_cols) > 0:
            self.ImputerNum.fit(df[num_cols])
            df[num_cols] = self.ImputerNum.transform(df[num_cols])

        return df


class MIAttackClassifierLinearSVC(MIAttackClassifier):

    def __init__(self, metadata, priorProbabilities, FeatureSet=None):
        super().__init__(SVC(kernel='linear', probability=True), metadata, priorProbabilities, FeatureSet)


class MIAttackClassifierSVC(MIAttackClassifier):

    def __init__(self, metadata, priorProbabilities, FeatureSet=None):
        super().__init__(SVC(probability=True), metadata, priorProbabilities, FeatureSet)


class MIAttackClassifierLogReg(MIAttackClassifier):

    def __init__(self, metadata, priorProbabilities, FeatureSet=None):
        super().__init__(LogisticRegression(), metadata, priorProbabilities, FeatureSet)


class MIAttackClassifierRandomForest(MIAttackClassifier):

    def __init__(self, metadata, priorProbabilities, FeatureSet=None):
        super().__init__(RandomForestClassifier(), metadata, priorProbabilities, FeatureSet)


class MIAttackClassifierKNN(MIAttackClassifier):

    def __init__(self, metadata, priorProbabilities, FeatureSet=None):
        super().__init__(KNeighborsClassifier(n_neighbors=5), metadata, priorProbabilities, FeatureSet)


class MIAttackClassifierMLP(MIAttackClassifier):

    def __init__(self, metadata, priorProbabilities, FeatureSet=None):
        super().__init__(MLPClassifier((200,), solver='lbfgs'), metadata, priorProbabilities, FeatureSet)


def generate_mia_shadow_data_shufflesplit(GenModel, target, rawA, sizeRaw, sizeSyn, numModels, numCopies):
    assert isinstance(rawA, GenModel.datatype), f"GM expectes datatype {GenModel.datatype} but got {type(rawA)}"
    assert isinstance(target, type(rawA)), f"Mismatch of datatypes between target record and raw data"

    kf = ShuffleSplit(n_splits=numModels, train_size=sizeRaw)
    synA, labels = [], []

    logger.debug(f'Start training {numModels} shadow models of class {GenModel.__class__.__name__}')
    # import pdb; pdb.set_trace()
    for train_index, _ in kf.split(rawA):

        # Fit GM to data without target's data
        if isinstance(rawA, DataFrame):
            rawAout = rawA.iloc[train_index]
        else:
            rawAout = rawA[train_index, :]
        GenModel.fit(rawAout)

        # Generate synthetic sample for data without target
        SynA_out = [GenModel.generate_samples(sizeSyn) for _ in range(numCopies)]
        synA.extend(SynA_out)
        labels.extend([LABEL_OUT for _ in range(numCopies)])

        # Insert targets into training data
        if isinstance(rawA, DataFrame):
            rawAin = rawAout.append(target)
        else:
            if len(target.shape) == 1:
                target = target.reshape(1, len(target))
            rawAin = concatenate([rawAout, target])

        # Fit generative model to data including target
        GenModel.fit(rawAin)

        # Generate synthetic sample for data including target
        synthetic_in = [GenModel.generate_samples(sizeSyn) for _ in range(numCopies)]
        synA.extend(synthetic_in)
        labels.extend([LABEL_IN for _ in range(numCopies)])

    return synA, labels


def generate_mia_shadow_data_allin(GenModel, target, rawA, nsamples, nsynthetic=10):
    assert isinstance(rawA, GenModel.datatype), f"GM expectes datatype {GenModel.datatype} but got {type(rawA)}"
    assert isinstance(target, type(rawA)), f"Mismatch of datatypes between target record and raw data"

    synA, labels = [], []

    logger.debug(f'Start training shadow model of class {GenModel.__class__.__name__} on data of size {len(rawA)}')

    # Fit GM to data without target's data
    GenModel.fit(rawA)

    # Generate synthetic sample for data without target
    synAout = [GenModel.generate_samples(nsamples) for _ in range(nsynthetic)]
    synA.extend(synAout)
    labels.extend([LABEL_OUT for _ in range(nsynthetic)])

    # Insert targets into training data
    if isinstance(rawA, DataFrame):
        rawAin = rawA.append(target)
    else:
        if len(target.shape) == 1:
            target = target.reshape(1, len(target))
        rawAin = concatenate([rawA, target])

    # Fit generative model to data including target
    GenModel.fit(rawAin)

    # Generate synthetic sample for data including target
    synAin = [GenModel.generate_samples(nsamples) for _ in range(nsynthetic)]
    synA.extend(synAin)
    labels.extend([LABEL_IN for _ in range(nsynthetic)])

    return synA, labels