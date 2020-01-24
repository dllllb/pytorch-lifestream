import numpy as np
import torch
import random
from sklearn.preprocessing import QuantileTransformer


class Dataset:

    def __init__(self, X_train, y_train, X_valid, y_valid,
                 random_state=None,
                 normalize=False,
                 quantile_transform=False,
                 output_distribution='normal',
                 quantile_noise=0):
        """
        Dataset is a dataclass that contains all training and evaluation data required for an experiment
        :param random_state: global random seed for an experiment
        :param normalize: standardize features by removing the mean and scaling to unit variance
        :param quantile_transform: transforms the features to follow a normal distribution.
        :param output_distribution: if quantile_transform == True, data is projected onto this distribution
            See the same param of sklearn QuantileTransformer
        :param quantile_noise: if specified, fits QuantileTransformer on data with added gaussian noise
            with std = :quantile_noise: * data.std ; this will cause discrete values to be more separable
            Please not that this transformation does NOT apply gaussian noise to the resulting data,
            the noise is only applied for QuantileTransformer
        """
        if random_state is not None:
            np.random.seed(random_state)
            torch.manual_seed(random_state)
            random.seed(random_state)

        self.X_train = X_train
        self.y_train = y_train
        self.X_valid = X_valid
        self.y_valid = y_valid

        if normalize:
            mean = np.mean(self.X_train, axis=0)
            std = np.std(self.X_train, axis=0)
            self.X_train = (self.X_train - mean) / std
            self.X_valid = (self.X_valid - mean) / std

        if quantile_transform:
            quantile_train = np.copy(self.X_train)
            if quantile_noise:
                stds = np.std(quantile_train, axis=0, keepdims=True)
                noise_std = quantile_noise / np.maximum(stds, quantile_noise)
                quantile_train += noise_std * np.random.randn(*quantile_train.shape)

            qt = QuantileTransformer(random_state=random_state, output_distribution=output_distribution).fit(quantile_train)
            self.X_train = qt.transform(self.X_train)
            self.X_valid = qt.transform(self.X_valid)