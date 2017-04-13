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

    outlier_inds\_: array of int, [num_outliers]
        Indices of outliers in X used for fitting.
    """
    def __init__(self, *args, **kwargs):
        self.num_outliers_ = kwargs.pop('num_outliers')
        return super(PCA, self).__init__(*args, **kwargs)

    def fit(self, X, y=None):
        outlier_flags = np.zeros(len(X), dtype=bool)
        for k in range(self.num_outliers_):
            X_reduced, non_outlier_flags = X[~outlier_flags, :], outlier_flags[~outlier_flags]
            super(PCA, self).fit(X_reduced)
            X_scores = super(PCA, self).score_samples(X_reduced)
            new_outlier_ind = np.argmin(X_scores)
            non_outlier_flags[new_outlier_ind] = True
        self.outlier_inds_ = np.where(outlier_flags)[0]
        return super(PCA, self).fit(X[self.outlier_inds_, :])
