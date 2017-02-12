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
        assumes a single cluster center for all points. For more details, see the original paper.

    copy : bool (default True)
        If False, data passed to fit are overwritten and running
        fit(X).transform(X) will not yield the expected results,
        use fit_transform(X) instead.

    whiten : bool, optional (default False)
        When True (False by default) the `components_` vectors are multiplied
        by the square root of n_samples and then divided by the singular values
        to ensure uncorrelated outputs with unit component-wise variances.

    tol : float >= 0, optional (default .0)
        Tolerance for singular values.

    eps : float >= 0, optional (default .0)
        Max relative error for the objective function optimised by IRLS. Used as a stopping
        criterion.

    max_iter : int >= 0, optional (default 100)
        Max number of IRLS iterations performed during optimisation. Used as a stopping crietrion.

    random_state : int or RandomState instance or None (default None)
        Pseudo Random Number generator seed control. If None, use the
        numpy.random singleton.

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
                 copy=True,
                 whiten=False,
                 tol=.0,
                 eps=.0,
                 max_iter=100,
                 random_state=None):
        self.n_components = n_components
        self.loss = loss
        self.model = model
        self.copy = copy
        self.whiten = whiten
        self.tol = tol
        self.eps = eps
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X, y=None):
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
        self._fit(X)
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
        return super().fit_transform(X, y)

    def _fit(self, X):
        # Raise an error for sparse input.
        # This is more informative than the generic one raised by check_array.
        if issparse(X):
            raise TypeError('MRobustPCA does not support sparse input.')

        X = check_array(X, dtype=[np.float64], ensure_2d=True,
                        copy=self.copy)

        # Handle n_components==None
        if self.n_components is None:
            n_components = X.shape[1]
        else:
            n_components = self.n_components

        if self.model == 'first':
            return self._fit_first_model(X, n_components)
        elif self.model == 'second':
            return self._fit_second_model(X, n_components)
        else:
            raise ValueError('Unknown model %s, should be \'first\' or \'second\''%(self.model))

    def _fit_first_model(self, X, n_components):
        vectorized_loss = np.vectorize(self.loss.__call__)
        vectorized_weights = np.vectorize(self.loss.weight)

        n_samples, n_features = X.shape
        self.weights_ = 1. / n_samples * np.ones(n_samples)
        self.errors_ = [np.inf]
        self.n_iterations_ = 0
        not_done_yet = True
        while not_done_yet:
            # Calculating components with current weights
            self.mean_ = np.average(X, axis=0, weights=self.weights_)
            X_centered = X - self.mean_
            U, S, V = linalg.svd(X_centered * np.sqrt(self.weights_.reshape(-1,1)))
            self.components_ = V[:n_components, :]

            # Calculate current errors
            non_projected_metric = np.eye(n_features) - \
                    self.components_.T.dot(self.components_)
            errors_raw = np.sqrt(np.diag(X_centered.dot(non_projected_metric.dot(X_centered.T))))
            errors_loss = vectorized_loss(errors_raw)

            # New weights based on errors
            self.weights_ = vectorized_weights(errors_raw)
            self.weights_ /= self.weights_.sum()
            # Checking stopping criteria
            iterations_done += 1
            old_total_error = self.errors_[-1]
            total_error = errors_loss.sum()

            #assert total_error <= old_total_error
            if not np.equal(total_error, 0.):
                rel_error = abs(total_error - old_total_error) / abs(total_error)
            else:
                rel_error = 0.

            print('[RPCA] Iteraton %d: error %f, relative error %f'%(iterations_done,
                                                                     total_error,
                                                                     rel_error))
            self.errors_.append(total_error)
            not_done_yet = rel_error > self.eps and iterations_done < self.max_iter
        if rel_error > self.eps:
            warnings.warn('[RPCA] Did not reach desired precision after %d iterations; relative\
                          error %f instead of specified maximum %f'%(iterations_done,
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

    def _fit_second_model(self, X, n_components):
        raise NotImplementedError('Only the first model is implemented yet.')
