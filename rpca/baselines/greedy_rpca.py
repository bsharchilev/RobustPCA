import numpy as np
from sklearn.decomposition import PCA

class GreedyRPCA(PCA):
    """
    Fits a PCA on a train set without ```num_outliers``` points, which are identified greedily
    by, on each step, fitting a PCA and throwing out a point with the largest error.

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
            PCA.fit(self, X_reduced)
            X_scores = PCA.score_samples(self, X_reduced)
            new_outlier_ind = np.argmin(X_scores)
            non_outlier_flags[new_outlier_ind] = True

            outlier_flags[~outlier_flags] = non_outlier_flags

        self.outlier_mask_ = outlier_flags
        return PCA.fit(self, X[~self.outlier_mask_, :])
