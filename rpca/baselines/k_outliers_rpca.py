import numpy as np
from sklearn.decomposition import PCA

class kOutliersRPCA(PCA):
    """
    Fits a PCA on a train set without ```num_outliers``` points, which are identified all in one
    step by fitting a PCA and throwing ```num_outlier``` with the largest error.

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
        PCA.fit(self, X)
        X_scores = PCA.score_samples(self, X)
        outlier_inds = np.argsort(X_scores)[:self.num_outliers_]

        self.outlier_mask_ = np.zeros(len(X), dtype=bool)
        self.outlier_mask_[outlier_inds] = True
        return PCA.fit(self, X[~self.outlier_mask_, :])
