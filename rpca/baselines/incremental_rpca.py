import numpy as np
from sklearn.decomposition import PCA

class IncrementalRPCA(PCA):
    """
    Fits a PCA on a train set without ```num_outliers``` points, which are identified incrementally
    by, on each step, looping over the sample, throwing away each point successively, fitting PCA on
    the rest and choosing the point that yields the best log-likelihood.

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
    def __init__(self, *args, **kwargs):
        self.num_outliers_ = kwargs.pop('num_outliers')
        return PCA.__init__(self, *args, **kwargs)

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
