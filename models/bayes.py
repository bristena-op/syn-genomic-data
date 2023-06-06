from pandas import DataFrame
from numpy import array_equal
import numpy as np
import pandas as pd

from models.data_synthesiser_utils.datatypes.constants import *
from models.data_synthesiser_utils.datatypes.FloatAttribute import FloatAttribute
from models.data_synthesiser_utils.datatypes.IntegerAttribute import IntegerAttribute
from models.data_synthesiser_utils.datatypes.SocialSecurityNumberAttribute import SocialSecurityNumberAttribute
from models.data_synthesiser_utils.datatypes.StringAttribute import StringAttribute
from models.data_synthesiser_utils.datatypes.DateTimeAttribute import is_datetime, DateTimeAttribute
from models.data_synthesiser_utils.datatypes.utils.AttributeLoader import parse_json
from models.data_synthesiser_utils.utils import *

class BayesianNet:
    """
    A BayesianNet model using non-private GreedyBayes to learn conditional probabilities
    """
    def __init__(self, category_threshold=5, histogram_bins=5, k=1, multiprocess=False):
        self.data_describer = DataDescriber(category_threshold, histogram_bins)

        self.k = k # maximum number of parents in Bayesian network
        self.bayesian_network = None
        self.conditional_probabilities = None
        self.multiprocess = multiprocess
        self.datatype = DataFrame

        self.trained = False

        self.__name__ = 'BayesianNet'

    def fit(self, data):
        assert isinstance(data, self.datatype), f'{self.__class__.__name__} expects {self.datatype} as input data but got {type(data)}'
        if self.trained:
            self.trained = False
            self.data_describer.data_description = {}
            self.bayesian_network = None
            self.conditional_probabilities = None

        self.data_describer.describe(data)

        encoded_df = DataFrame()
        for attr in self.data_describer.metadata['attribute_list_hist']:
            column = self.data_describer.attr_dict[attr]
            encoded_df[attr] = column.encode_values_into_bin_idx()
        if encoded_df.shape[1] < 2:
            raise Exception("BayesianNet requires at least 2 attributes(i.e., columns) in dataset.")

        if self.multiprocess:
            self.bayesian_network = self._greedy_bayes_multiprocess(encoded_df, self.k)
        else:
            self.bayesian_network = self._greedy_bayes_linear(encoded_df, self.k)
        self.conditional_probabilities = self._construct_conditional_probabilities(self.bayesian_network, encoded_df)

        self.trained = True

    def generate_samples(self, nsamples):
        # logger.debug(f'Generate synthetic dataset of size {nsamples}')
        assert self.trained, "Model must be fitted to some real data first"
        all_attributes = self.data_describer.metadata['attribute_list']
        synthetic_data = DataFrame(columns=all_attributes)

        # Get samples for attributes modelled in Bayesian net
        encoded_dataset = self._generate_encoded_dataset(nsamples)

        for attr in all_attributes:
            column = self.data_describer.attr_dict[attr]
            if attr in encoded_dataset:
                synthetic_data[attr] = column.sample_values_from_binning_indices(encoded_dataset[attr])
            else:
                # For attributes not in BN use independent attribute mode
                binning_indices = column.sample_binning_indices_in_independent_attribute_mode(nsamples)
                synthetic_data[attr] = column.sample_values_from_binning_indices(binning_indices)

        return synthetic_data

    def _generate_encoded_dataset(self, nsamples):
        encoded_df = DataFrame(columns=self._get_sampling_order(self.bayesian_network))

        bn_root_attr = self.bayesian_network[0][1][0]
        root_attr_dist = self.conditional_probabilities[bn_root_attr]
        encoded_df[bn_root_attr] = choice(len(root_attr_dist), size=nsamples, p=root_attr_dist)

        for child, parents in self.bayesian_network:
            child_conditional_distributions = self.conditional_probabilities[child]

            for parents_instance in child_conditional_distributions.keys():
                dist = child_conditional_distributions[parents_instance]
                parents_instance = list(eval(parents_instance))

                filter_condition = ''
                for parent, value in zip(parents, parents_instance):
                    filter_condition += f"(encoded_df['{parent}']=={value})&"

                filter_condition = eval(filter_condition[:-1])
                size = encoded_df[filter_condition].shape[0]
                if size:
                    encoded_df.loc[filter_condition, child] = choice(len(dist), size=size, p=dist)
                unconditioned_distribution = self.data_describer.data_description[child]['distribution_probabilities']
                encoded_df.loc[encoded_df[child].isnull(), child] = choice(len(unconditioned_distribution),
                                                                              size=encoded_df[child].isnull().sum(),
                                                                              p=unconditioned_distribution)
        encoded_df[encoded_df.columns] = encoded_df[encoded_df.columns].astype(int)
        return encoded_df

    def _get_sampling_order(self, bayesian_net):
        order = [bayesian_net[0][1][0]]
        for child, _ in bayesian_net:
            order.append(child)
        return order

    def _greedy_bayes_multiprocess(self, encoded_df, k=1):
        """Construct a Bayesian Network (BN) using greedy algorithm."""
        dataset = encoded_df.astype(str, copy=False)

        root_attribute = choice(dataset.columns)
        V = [root_attribute]
        rest_attributes = set(dataset.columns)
        rest_attributes.remove(root_attribute)
        bayesian_net = []
        while rest_attributes:
            parents_pair_list = []
            mutual_info_list = []

            num_parents = min(len(V), k)
            tasks = [(child, V, num_parents, split, dataset) for child, split in
                     product(rest_attributes, range(len(V) - num_parents + 1))]
            # with Pool() as pool:
            res_list = map(bayes_worker, tasks)

            for res in res_list:
                parents_pair_list += res[0]
                mutual_info_list += res[1]

            idx = mutual_info_list.index(max(mutual_info_list))

            bayesian_net.append(parents_pair_list[idx])
            adding_attribute = parents_pair_list[idx][0]
            V.append(adding_attribute)
            rest_attributes.remove(adding_attribute)

        return bayesian_net

    def _greedy_bayes_linear(self, encoded_df, k=1):
        """Construct a Bayesian Network (BN) using greedy algorithm."""
        dataset = encoded_df.astype(str, copy=False)

        root_attribute = choice(dataset.columns)
        V = [root_attribute]
        rest_attributes = set(dataset.columns)
        rest_attributes.remove(root_attribute)
        bayesian_net = []
        while rest_attributes:
            parents_pair_list = []
            mutual_info_list = []

            num_parents = min(len(V), k)
            for child, split in product(rest_attributes, range(len(V) - num_parents + 1)):
                task = (child, V, num_parents, split, dataset)
                res = bayes_worker(task)
                parents_pair_list += res[0]
                mutual_info_list += res[1]

            idx = mutual_info_list.index(max(mutual_info_list))

            bayesian_net.append(parents_pair_list[idx])
            adding_attribute = parents_pair_list[idx][0]
            V.append(adding_attribute)
            rest_attributes.remove(adding_attribute)

        return bayesian_net

class DataDescriber(object):
    def __init__(self, category_threshold, histogram_bins):
        self.category_threshold = category_threshold
        self.histogram_bins = histogram_bins

        self.attributes = {}
        self.data_description = {}
        self.metadata = {}

    def describe(self, df):
        attr_to_datatype = self.infer_attribute_data_types(df)
        self.attr_dict = self.represent_input_dataset_by_columns(df, attr_to_datatype)

        for column in self.attr_dict.values():
            column.infer_domain()
            column.infer_distribution()
            self.data_description[column.name] = column.to_json()
        self.metadata = self.describe_metadata(df)

    def infer_attribute_data_types(self, df):
        attr_to_datatype = {}
        # infer data types
        numerical_attributes = infer_numerical_attributes_in_dataframe(df)

        for attr in list(df):
            column_dropna = df[attr].dropna()

            # Attribute is either Integer or Float.
            if attr in numerical_attributes:

                if array_equal(column_dropna, column_dropna.astype(int, copy=False)):
                    attr_to_datatype[attr] = INTEGER
                else:
                    attr_to_datatype[attr] = FLOAT

            # Attribute is either String or DateTime
            else:
                attr_to_datatype[attr] = STRING

        return attr_to_datatype

    def represent_input_dataset_by_columns(self, df, attr_to_datatype):
        attr_to_column = {}

        for attr in list(df):
            data_type = attr_to_datatype[attr]
            is_categorical = self.is_categorical(df[attr])
            paras = (attr, df[attr], self.histogram_bins, is_categorical)
            if data_type is INTEGER:
                attr_to_column[attr] = IntegerAttribute(*paras)
            elif data_type is FLOAT:
                attr_to_column[attr] = FloatAttribute(*paras)
            elif data_type is DATETIME:
                attr_to_column[attr] = DateTimeAttribute(*paras)
            elif data_type is STRING:
                attr_to_column[attr] = StringAttribute(*paras)
            elif data_type is SOCIAL_SECURITY_NUMBER:
                attr_to_column[attr] = SocialSecurityNumberAttribute(*paras)
            else:
                raise Exception(f'The DataType of {attr} is unknown.')

        return attr_to_column

    def describe_metadata(self, df):
        nrecords, nfeatures = df.shape
        all_attributes = list(df)
        hist_attributes = []
        str_attributes = []

        for attr in all_attributes:
            if attr in self.data_description.keys():
                column = self.data_description[attr]
                if column is STRING and not column.is_categorical:
                    str_attributes.append(attr)
                else:
                    hist_attributes.append(attr)

        metadata = {'num_records': nrecords,
                    'num_attributes': nfeatures,
                    'attribute_list': all_attributes,
                    'attribute_list_hist': hist_attributes,
                    'attribute_list_str': str_attributes}
        return metadata

    def is_categorical(self, data):
        """
        Detect whether an attribute is categorical.
        """
        return data.dropna().unique().size <= self.category_threshold

if __name__ == '__main__':
    dataset = 'CHB'
    data_file = f'/home/bristena/syn_genomics/datasets/chr13/small_{dataset}.chr13.hap'
    dat = pd.read_csv(data_file,sep= '\t', header=None)
    # import pdb; pdb.set_trace()

    dat = dat.to_numpy()
    pos = dat[:, 0]
    dat = pd.DataFrame(np.transpose(dat[:, 1:]))
    # dat = dat.T
    import pdb; pdb.set_trace()
    bn = BayesianNet(45, 10)
    errs = bn.fit(dat)
    sample = np.transpose(bn.generate_samples(dat, dat.shape[0]))
    sample = np.around(sample, decimals=0)
    import pdb; pdb.set_trace()

    sample = sample.astype(int)

    sample = np.insert(sample, 0, pos, axis=1)
    res_file = f'../syn_data/bn_out_hap_{dataset}.csv'
    np.savetxt(res_file, sample, delimiter='\t', fmt='%i')
