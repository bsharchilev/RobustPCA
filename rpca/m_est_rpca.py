import warnings
from sklearn.decomposition import PCA
from sklearn.decomposition.base import _BasePCA
from sklearn.utils import check_array
from sklearn.utils.extmath import svd_flip
import numpy as np

class MRobustPCA(_BasePCA):
    """
    An implementation of Robust PCA using M-estimator loss. A particular case with the Huber loss
    function is described in the paper:

        B.T. Polyak, M.V. Khlebnikov: Principal Component Analysis: Robust Variants, 2017.

    Parameters:
    -----------
    n_components: int or None
        Number of components. If None, n_components = data dimensionality.

    loss : MLossFunction object
        Loss function to be optimized during fitting.

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

    eps : float >= 0, optional (default 0.01)
        Max relative error for the objective function optimised by IRLS. Used as a stopping
        criterion.

    max_iter : int >= 0, optional (default 100)
        Max number of IRLS iterations performed during optimisation. Used as a stopping crietrion.

    random_state : int or RandomState instance or None (default None)
        Pseudo Random Number generator seed control. If None, use the
        numpy.random singleton.

    Attributes
    ----------
    components_ : array, [n_components, n_features]
        Principal axes in feature space. The components are sorted by
        ``explained_variance_``.

    explained_variance_ : array, [n_components]
        The amount of variance explained by each of the selected components.

    explained_variance_ratio_ : array, [n_components]
        Percentage of variance explained by each of the selected components.
        If ``n_components`` is not set then all components are stored and the
        sum of explained variances is equal to 1.0.

    mean_ : array, [n_features]
        Per-feature empirical mean, estimated from the training set.
        Equal to `X.mean(axis=1)`.

    n_components_ : int
        The number of components. It equals the parameter
        n_components, or n_features if n_components is None.    

    weights_ : array, [n_samples]
        Weights assigned to input samples during IRLS optimisation.
    """
    def __init__(self,
                 n_components,
                 loss,
                 model='first',
                 copy=True,
                 whiten=False,
                 tol=.0,
                 eps=0.01,
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

    fit = PCA.fit
    #fit_transform = PCA.fit_transform # MAY CHANGE; ITERATIVE SVD MAY BE COMPUTATIONALLY BETTER

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
        return self._fit_first_model(X, n_components)

    def _fit_first_model(self, X, n_components):
        vectorized_loss = np.vectorize(self.loss.__call__)
        vectorized_weights = np.vectorize(self.loss.weight)

        n_samples, n_features = X.shape
        self.weights_ = 1. / n_samples * np.ones((n_samples, 1))
        total_error = np.inf
        iterations_done = 0
        not_done_yet = True
        while not_done_yet:
            # Calculating components with current weights
            self.mean_ = np.average(X, axis=0, weights=self.weights_)
            X_centered = X - self.mean_
            U, S, V = np.svd(X_scaled / np.sqrt(self.weights_))
            self.components_ = V[:n_components, :]

            # Calculate current errors
            non_projected_metric = np.eye(n_features) - \
                    self.components_.T.dot(self.components_)
            errors = vectorized_loss(np.sqrt((X_centered.dot(non_projected_metric)\
                                              * X_centered).sum(-1)))

            # New weights based on errors
            self.weights_ = vectorized_weights(errors)
            self.weights_ /= self.weights_.sum()

            # Checking stopping criteria
            iterations_done += 1
            old_total_error = total_error
            total_error = errors.sum()
            assert total_error <= old_total_error
            if not np.equal(total_error, 0.):
                rel_error = abs(total_error - old_total_error) / abs(total_error)
            else:
                rel_error = 0.

            print('[RPCA] Iteraton %d: error %f, relative error %f'%(iterations_done,
                                                                     total_error,
                                                                     rel_error))
            not_done_yet = rel_error > self.eps and iterations_done < self.max_iter
        if rel_error > self.eps:
            warnings.warn('[RPCA] Did not reach desired precision after %d iterations; relative
                          error %f instead of specified maximum %f'%(iterations_done,
                                                                     rel_error,
                                                                     self.eps))
        # Get variance explained by singular values
        explained_variance_ = (S ** 2) / n_samples
        total_var = explained_variance_.sum()
        explained_variance_ratio_ = explained_variance_ / total_var

        self.n_samples_, self.n_features_ = n_samples, n_features
        self.n_components_ = n_components
        self.explained_variance_ = explained_variance_[:n_components]
        self.explained_variance_ratio_ = \
            explained_variance_ratio_[:n_components]
