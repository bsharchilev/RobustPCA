import numpy as np

from sklearn.decomposition import PCA
from itertools import product

from abc import ABCMeta, abstractmethod

class OutlierDetectorRPCA(PCA):
    """
    Fits a PCA on a train set without ```num_outliers``` points identified from the train set. Exact
    method of detection should be implemented in ```fit```.

    Parameters
    -----------
    *args, **kwargs:
        All standard PCA parameters.

    num_outliers: int >= 0, Keyword argument
        Number of outliers to identify. If 0, the method reduces to ordinary PCA.

    Attributes
    -----------
    PCA_attributes:
        All standard PCA attributes.

    num_outliers\_: int >= 0
        Number of outliers identified. If 0, the method reduces to ordinary PCA.

    outlier_mask\_: array of bool, [n_samples_]
        Boolean mask of identified outliers in X used for fitting.
    """
    __metaclass__ = ABCMeta
    
    def __init__(self, *args, **kwargs):
        self.num_outliers_ = kwargs.pop('num_outliers')
        return PCA.__init__(self, *args, **kwargs)

    @abstractmethod
    def fit(self, X, y=None):
        pass

class GreedyRPCA(OutlierDetectorRPCA):
    """
    Fits a PCA on a train set without ```num_outliers``` points, which are identified greedily
    by, on each step, fitting a PCA and throwing out a point with the largest error.
    """
    def fit(self, X, y=None):
        outlier_flags = np.zeros(len(X), dtype=bool)
        for k in range(self.num_outliers_):
            X_reduced, non_outlier_flags = X[~outlier_flags, :], outlier_flags[~outlier_flags]
            PCA.fit(self, X_reduced)
            X_scores = PCA.score_samples(self, X_reduced)
            new_outlier_ind = np.argmin(X_scores)
            non_outlier_flags[new_outlier_ind] = True

            outlier_flags[~outlier_flags] = non_outlier_flags

        self.outlier_mask_ = outlier_flags
        return PCA.fit(self, X[~self.outlier_mask_, :])

class kOutliersRPCA(OutlierDetectorRPCA):
    """
    Fits a PCA on a train set without ```num_outliers``` points, which are identified all in one
    step by fitting a PCA and throwing ```num_outlier``` with the largest error.
    """
    def fit(self, X, y=None):
        PCA.fit(self, X)
        X_scores = PCA.score_samples(self, X)
        outlier_inds = np.argsort(X_scores)[:self.num_outliers_]

        self.outlier_mask_ = np.zeros(len(X), dtype=bool)
        self.outlier_mask_[outlier_inds] = True
        return PCA.fit(self, X[~self.outlier_mask_, :])

class IncrementalRPCA(OutlierDetectorRPCA):
    """
    Fits a PCA on a train set without ```num_outliers``` points, which are identified incrementally
    by, on each step, looping over the sample, throwing away each point successively, fitting PCA on
    the rest and choosing the point that yields the best log-likelihood.

    Note: this method is provably identical to GreedyRPCA, is included for research purposes and
    should not be used in production.
    """
    def fit(self, X, y=None):
        outlier_flags = np.zeros(len(X), dtype=bool)
        for k in range(self.num_outliers_):
            X_reduced, non_outlier_flags = X[~outlier_flags, :], outlier_flags[~outlier_flags]
            new_outlier_ind = self._find_outlier(X_reduced)
            
            non_outlier_flags[new_outlier_ind] = True
            outlier_flags[~outlier_flags] = non_outlier_flags

        self.outlier_mask_ = outlier_flags
        return PCA.fit(self, X[~self.outlier_mask_, :])

    def _find_outlier(self, X):
        indices = set(range(len(X)))
        log_likelihoods = np.array([])
        for i in range(len(X)):
            X_remain = X[list(indices - set([i])), :]
            PCA.fit(self, X_remain)
            log_likelihoods = np.append(log_likelihoods, PCA.score_samples(self, X_remain).sum())
        return np.argmax(log_likelihoods)

class CombinatorialRPCA(OutlierDetectorRPCA):
    """
    Loops through each combination of ```n_outliers``` points from X and labels as outliers the
    combination which yields the highest log-likelihood when removed.
    """
    def fit(self, X, y=None):
        index_space = [list(range(len(X)))] * self.num_outliers_
        all_indices = set(range(len(X)))
        best_indices, max_log_likelihood = [0] * self.num_outliers_, -np.inf
        for indices_to_remove in product(*index_space):
            indices_to_leave = list(all_indices - set(indices_to_remove))
            X_remain = X[indices_to_leave, :]
            PCA.fit(self, X_remain)

            current_log_likelihood = PCA.score_samples(self, X_remain).sum()
            if current_log_likelihood > max_log_likelihood:
                best_indices = list(indices_to_remove)
                max_log_likelihood = current_log_likelihood

        self.outlier_mask_ = np.zeros(len(X), dtype=bool)
        self.outlier_mask_[best_indices] = True
        return PCA.fit(self, X[~self.outlier_mask_, :])
