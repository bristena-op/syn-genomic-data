from os import path
from pandas import DataFrame
from numpy import mean, concatenate, ndarray, ones, sqrt
from scipy.stats import norm
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression

from priv_eval.attack_model import PrivacyAttack
from priv_eval.datagen import convert_df_to_array, convert_series_to_array

import logging
from logging.config import fileConfig
dirname = path.dirname(__file__)
logconfig = path.join(dirname, '../logging_config.ini')
fileConfig(logconfig)
logger = logging.getLogger(__name__)


class AttributeInferenceAttack(PrivacyAttack):
    """A privacy attack that aims to reconstruct a sensitive attribute c given a partial target record T"""

    def __init__(self, RegressionModel, sensitiveAttribute, metadata, backgroundKnowledge):
        """
        Parent class for simple regression attribute inference attack

        :param RegressionModel: object: sklearn type regression model object
        :param sensitiveAttribute: string: name of a column in a DataFrame that is considered the unknown, sensitive attribute
        :param metadata: dict: schema for the data to be attacked
        :param backgroundKnowledge: pd.DataFrame: adversary's background knowledge dataset
        """

        self.sensitiveAttribute = sensitiveAttribute
        self.RegressionModel = RegressionModel
        self.metadata = metadata
        self.imputerCat = SimpleImputer(strategy='most_frequent')
        self.imputerNum = SimpleImputer(strategy='median')

        self.scaleFactor = None
        self.coefficients = None
        self.sigma = None

        # self.priorProbabilities = self._calculate_prior_probabilities(backgroundKnowledge, self.sensitiveAttribute)
        self.trained = False

        self.__name__ = f'{self.RegressionModel.__class__.__name__}'


    def _calculate_prior_probabilities(self, backgroundKnowledge, sensitiveAttribute):
        """

        :param backgroundKnowledge: pd.DataFrame: adversary's background knowledge dataset
        :param sensitiveAttribute: str: name of a column in the DataFrame that is considered sensitive
        :return: priorProb: dict: prior probabilities over sensitive attribute
        """

        return dict(backgroundKnowledge[sensitiveAttribute].value_counts(sort=False, dropna=False)/len(backgroundKnowledge))
    def _calculate_prior_probabilities_LD_AF(self, backgroundKnowledge, sensitiveAttribute):
        """

        :param backgroundKnowledge: pd.DataFrame: adversary's background knowledge dataset
        :param sensitiveAttribute: str: name of a column in the DataFrame that is considered sensitive
        :return: priorProb: dict: prior probabilities over sensitive attribute
        """

        return dict(backgroundKnowledge[sensitiveAttribute].value_counts(sort=False, dropna=False)/len(backgroundKnowledge))
    def _calculate_prior_probabilities_recomb(self, backgroundKnowledge, sensitiveAttribute):
        """

        :param backgroundKnowledge: pd.DataFrame: adversary's background knowledge dataset
        :param sensitiveAttribute: str: name of a column in the DataFrame that is considered sensitive
        :return: priorProb: dict: prior probabilities over sensitive attribute
        """

        return dict(backgroundKnowledge[sensitiveAttribute].value_counts(sort=False, dropna=False)/len(backgroundKnowledge))


    def get_prior_probability(self, sensitiveValue):
        try:
            return self.priorProbabilities[sensitiveValue]
        except:
            return 0


    def train(self, synT):
        """
        Train a MLE attack to reconstruct an unknown sensitive value from a vector of known attributes
        :param synT: type(DataFrame) A synthetic dataset of shape (n, k + 1)
        """

        # Split data into known and sensitive
        if isinstance(synT, DataFrame):
            assert self.sensitiveAttribute in list(synT), f'DataFrame only contains columns {list(synT)}'
            synAux = synT.drop(self.sensitiveAttribute, axis=1)
            synSensitive = synT[self.sensitiveAttribute]
            synAux = convert_df_to_array(synAux, self.metadata)
            synSensitive = convert_series_to_array(synSensitive, self.metadata)
        else:
            assert isinstance(synT, ndarray), f"Unknown data type {type(synT)}"
            # If input data is array assume that self.metadata is the schema of the array
            attrList = [c['name'] for c in self.metadata['columns']]
            sensitiveIdx = attrList.index(self.sensitiveAttribute)
            synAux = synT[:, [i for i in range(len(attrList)) if i != sensitiveIdx]]
            synSensitive = synT[:, sensitiveIdx]

        n, k = synAux.shape
        import pdb; pdb.set_trace()
        # Center independent variables for better regression performance
        self.scaleFactor = mean(synAux, axis=0)
        synAuxScaled = synAux - self.scaleFactor
        synAuxScaled = concatenate([ones((len(synAuxScaled), 1)), synAuxScaled], axis=1) # append all  ones for inclu intercept in beta vector

        # Get MLE for linear coefficients
        self.RegressionModel.fit(synAuxScaled, synSensitive)
        self.coefficients = self.RegressionModel.coef_
        self.sigma = sum((synSensitive - synAuxScaled.dot(self.coefficients))**2)/(n)

        logger.debug('Finished training regression model')
        self.trained = True

    def attack(self, targetAux):
        """Makes a guess about the target's secret attribute from the synthetic data"""
        assert self.trained, 'Attack must first be trained on some data before can predict sensitive target value'
        if isinstance(targetAux, DataFrame):
            targetAux = convert_df_to_array(targetAux, self.metadata) # extract attribute values for known attributes
        else:
            assert isinstance(targetAux, ndarray), f'Unknown data type {type(targetAux)}'
        targetAux_scaled = targetAux - self.scaleFactor
        targetAux_scaled = concatenate([ones((len(targetAux_scaled), 1)), targetAux_scaled], axis=1)

        return targetAux_scaled.dot(self.coefficients)

    def get_likelihood(self, targetAux, targetSensitive):
        assert self.trained, 'Attack must first be trained on some data before can predict sensitive target value'

        targetKnown = convert_df_to_array(targetAux, self.metadata) # extract attribute values for known attributes
        targetKnown_scaled = targetKnown - self.scaleFactor
        targetKnown_scaled = concatenate([ones((len(targetKnown_scaled), 1)), targetKnown_scaled], axis=1)
        pdfLikelihood = norm(loc=targetKnown_scaled.dot(self.coefficients), scale=sqrt(self.sigma))

        return pdfLikelihood.pdf(targetSensitive)
    def get_predicted_value(self, targetAux, targetSensitive):
        assert self.trained, 'Attack must first be trained on some data before can predict sensitive target value'

        targetKnown = convert_df_to_array(targetAux, self.metadata)  # extract attribute values for known attributes
        targetKnown_scaled = targetKnown - self.scaleFactor
        targetKnown_scaled = concatenate([ones((len(targetKnown_scaled), 1)), targetKnown_scaled], axis=1)

        target_secret = targetKnown_scaled.dot(self.coefficients)
        return target_secret
    def _impute_missing_values(self, df):
        catCols = list(df.select_dtypes(['object', 'category']))
        if len(catCols) > 0:
            self.imputerCat.fit(df[catCols])
            df[catCols] = self.imputerCat.transform(df[catCols])

        numCols = list(df.select_dtypes(['int', 'float']))
        if len(numCols) > 0:
            self.imputerNum.fit(df[numCols])
            df[numCols] = self.imputerNum.transform(df[numCols])

        return df


class AttributeInferenceAttackLinearRegression(AttributeInferenceAttack):
    """An AttributeInferenceAttack based on a simple Linear Regression model"""

    def __init__(self, sensitiveAttribute, metadata, backgroundKnowledge):
        super().__init__(LinearRegression(fit_intercept=False), sensitiveAttribute, metadata, backgroundKnowledge)

