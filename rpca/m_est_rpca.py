import warnings
from sklearn.decomposition import PCA
from sklearn.decomposition.base import _BasePCA
from sklearn.utils import check_array
from sklearn.utils.extmath import svd_flip
from scipy.sparse import issparse
from scipy import linalg
import numpy as np

class MRobustPCA(_BasePCA):
    """
    An implementation of Robust PCA using M-estimator loss. A particular case with the Huber loss
    function is described in the paper:

        B.T. Polyak, M.V. Khlebnikov: Principal Component Analysis: Robust Variants, 2017.

    Parameters
    -----------
    n_components: int or None
        Number of components. If None, n_components = data dimensionality.

    loss : MLossFunction object
        Loss function to be optimized during fitting. See "Loss functions" for details.

    model : string {'first', 'second'} (default 'first')
        Statistical model to be used during fitting, according to the original method. First is
        based on the iid noise assumption, second estimates the noise covariance structure but
        assumes a single cluster center for all points. For more details, see the original paper.abs

    eps : float >= 0, optional (default 1e-6)
        Max relative error for the objective function optimised by IRLS. Used as a stopping
        criterion.

    max_iter : int >= 0, optional (default 100)
        Max number of IRLS iterations performed during optimisation. Used as a stopping crietrion.

    Attributes
    ----------
    components\_ : array, [n_components, n_features]
        Principal axes in feature space. The components are sorted by
        ``explained_variance_``.

    explained_variance\_ : array, [n_components]
        The amount of variance explained by each of the selected components.

    explained_variance_ratio\_ : array, [n_components]
        Percentage of variance explained by each of the selected components.
        If ``n_components`` is not set then all components are stored and the
        sum of explained variances is equal to 1.0.

    mean\_ : array, [n_features]
        Per-feature empirical mean, estimated from the training set.
        Equal to `X.mean(axis=1)`.

    n_components\_ : int
        The number of components. It equals the parameter
        n_components, or n_features if n_components is None.    

    weights\_ : array, [n_samples]
        Weights assigned to input samples during IRLS optimisation.

    n_iterations\_ : int
        Number of iterations performed during fitting.

    errors\_ : array, [n_iterations\_]
        Errors achieved by the method on each iteration.
    """
    def __init__(self,
                 n_components,
                 loss,
                 model='first',
                 eps=1e-6,
                 max_iter=100):
        self.n_components = n_components
        self.loss = loss
        self.model = model
        self.eps = eps
        self.max_iter = max_iter
        self.whiten = False # TODO: implement proper whitening

    def fit(self, X, y=None, weights_init=None):
        """
        Fit the model with X.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            Training data, where n_samples in the number of samples
            and n_features is the number of features.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        self._fit(X, weights_init)
        return self

    def fit_transform(self, X, y=None):
        """
        Fit to data, then transform it.

        Fits the model to X and returns a transformed version of X.
        
        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            Training data, where n_samples in the number of samples
            and n_features is the number of features.

        Returns
        -------
        X_new: array-like, shape (n_samples, n_components)
            Transformed array.
        """
        return super(MRobustPCA, self).fit_transform(X, y)

    def _fit(self, X, weights_init=None):
        # Raise an error for sparse input.
        # This is more informative than the generic one raised by check_array.
        if issparse(X):
            raise TypeError('MRobustPCA does not support sparse input.')

        X = check_array(X, dtype=[np.float64], ensure_2d=True,
                        copy=True)

        # Handle n_components==None
        if self.n_components is None:
            n_components = X.shape[1]
        else:
            n_components = self.n_components

        return self._fit_model(X, n_components, weights_init)

    def _fit_model(self, X, n_components, weights_init):
        vectorized_loss = np.vectorize(self.loss.__call__)
        vectorized_weights = np.vectorize(self.loss.weight)

        n_samples, n_features = X.shape
        if weights_init is not None:
            self.weights_ = weights_init
        else:
            self.weights_ = 1. / n_samples * np.ones(n_samples)
        self.errors_ = [np.inf]
        self.n_iterations_ = 0
        not_done_yet = True
        while not_done_yet:
            # Calculating components with current weights
            self.mean_ = np.average(X, axis=0, weights=self.weights_)
            X_centered = X - self.mean_
            U, S, V = linalg.svd(X_centered * np.sqrt(self.weights_.reshape(-1,1)))
            # U, V = svd_flip(U, V)
            self.components_ = V[:n_components, :]

            # Calculate current errors in different models
            if self.model == 'first':
                non_projected_metric = np.eye(n_features) - \
                        self.components_.T.dot(self.components_)
                errors_raw = np.sqrt(np.diag(X_centered.dot(non_projected_metric.dot(X_centered.T))))
            elif self.model == 'second':
                # Obtain inverse empirical covariance from the SVD
                R_inv = np.diag(1. / S**2.)
                inverse_cov = V.T.dot(R_inv.dot(V))
                errors_raw = np.sqrt(np.diag(X_centered.dot(inverse_cov.dot(X_centered.T))))
            else:
                raise ValueError('Model should be either \"first\" or \"second\".') 

            errors_loss = vectorized_loss(errors_raw)

            # New weights based on errors
            self.weights_ = vectorized_weights(errors_raw)
            self.weights_ /= self.weights_.sum()
            # Checking stopping criteria
            self.n_iterations_ += 1
            old_total_error = self.errors_[-1]
            total_error = errors_loss.sum()

            if not np.equal(total_error, 0.):
                rel_error = abs(total_error - old_total_error) / abs(total_error)
            else:
                rel_error = 0.

            print('[RPCA] Iteraton %d: error %f, relative error %f'%(self.n_iterations_,
                                                                     total_error,
                                                                     rel_error))
            self.errors_.append(total_error)
            not_done_yet = rel_error > self.eps and self.n_iterations_ < self.max_iter
        if rel_error > self.eps:
            warnings.warn('[RPCA] Did not reach desired precision after %d iterations; relative\
                          error %f instead of specified maximum %f'%(self.n_iterations_,
                                                                     rel_error,
                                                                     self.eps))
        # Get variance explained by singular values
        explained_variance_ = (S ** 2) / n_samples
        total_var = explained_variance_.sum()
        if not np.equal(total_var, 0.):
            explained_variance_ratio_ = explained_variance_ / total_var
        else:
            explained_variance_ratio_ = np.zeros_like(explained_variance_)
        self.n_samples_, self.n_features_ = n_samples, n_features
        self.n_components_ = n_components
        self.explained_variance_ = explained_variance_[:n_components]
        self.explained_variance_ratio_ = \
            explained_variance_ratio_[:n_components]

        self.errors_ = np.array(self.errors_[1:])
