from numpy import arange

from models.data_synthesiser_utils.datatypes.AbstractAttribute import AbstractAttribute
from models.data_synthesiser_utils.datatypes.utils.DataType import DataType


class FloatAttribute(AbstractAttribute):
    def __init__(self, name, data, histogram_size, is_categorical):
        super().__init__(name, data, histogram_size, is_categorical)
        self.is_numerical = True
        self.data_type = DataType.FLOAT

    def infer_domain(self, categorical_domain=None, numerical_range=None):
        super().infer_domain(categorical_domain, numerical_range)

    def infer_distribution(self):
        super().infer_distribution()

    def generate_values_as_candidate_key(self, n):
        return arange(self.min, self.max, (self.max - self.min) / n)

    def sample_values_from_binning_indices(self, binning_indices):
        return super().sample_values_from_binning_indices(binning_indices)
