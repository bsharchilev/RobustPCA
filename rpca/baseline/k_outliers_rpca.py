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

    outlier_inds\_: array of int, [num_outliers]
        Indices of outliers in X used for fitting.
    """
    def __init__(self, *args, **kwargs):
        self.num_outliers_ = kwargs.pop('num_outliers')
        return super(PCA, self).__init__(*args, **kwargs)

    def fit(self, X, y=None):
        super(PCA, self).fit(X)
        X_scores = super(PCA, self).score_samples(X)
        self.outlier_inds_ = np.argsort(-X_scores)[:self.num_outliers_]
        return super(PCA, self).fit(X[self.outlier_inds_, :])
