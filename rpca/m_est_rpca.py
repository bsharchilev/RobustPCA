from sklearn.decomposition.base import _BasePCA
from sklearn.utils import check_array
from abc import ABCMeta, abstractmethod

class MLossFunction:
    """
    An abstract class for a loss function suitable for an M-estimator solution using IRLS. Has to
    have a strict minimum at zero, be symmetric and positive definite, and increasing less than
    square.
    Programmatically, it has to implement a __call__(self, x) method, where x is a float, a
    diff(self, x) method returning its derivative at x, and (optionally) a weight(self,
    x)=diff(self, x)/x.
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def __call__(self, x):
        """
        Evaluates the function at x.

        Parameters:
        -----------
        x : float
            Point to evaluate the function at.

        Returns:
        --------
        y : float
            Value at point x.
        """

    @abstractmethod
    def diff(self, x):
        """
        Evaluates the function's derivative at x.

        Parameters:
        -----------
        x : float
            Point to evaluate the function derivative at.

        Returns:
        --------
        y : float
            Derivative value at point x.
        """

    def weight(self, x):
        """
        Evaluates the weight function (:=f(x)/x) induced by the loss function at x.

        Parameters:
        -----------
        x : float
            Point to evaluate the weight function at.

        Returns:
        --------
        y : float
            Weight function value at point x.
        """
        return self.diff(x) / float(x)

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
