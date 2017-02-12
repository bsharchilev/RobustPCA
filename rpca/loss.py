from abc import ABCMeta, abstractmethod
from numbers import Number

class MLoss:
    """
    An abstract class for a loss function suitable for an M-estimator solution using IRLS. Has to
    have a strict minimum at zero, be symmetric and positive definite, and increasing less than
    square.

    Programmatically, it has to implement a ``__call__(self, x)`` method, where `x` is a float, a
    ``diff(self, x)`` method returning its derivative at ``x``, and (optionally) a ``weight(self,
    x)=diff(self, x)/x``.
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def __call__(self, x):
        """
        Evaluates the function at x.

        Parameters
        ------------
        x : float
            Point to evaluate the function at.

        Returns
        ---------
        y : float
            Value at point x.
        """

    @abstractmethod
    def diff(self, x):
        """
        Evaluates the function's derivative at x.

        Parameters
        ------------
        x : float
            Point to evaluate the function derivative at.

        Returns
        ---------
        y : float
            Derivative value at point x.
        """

    def weight(self, x):
        """
        Evaluates the weight function (:=f(x)/x) induced by the loss function at x.

        Parameters
        ------------
        x : float
            Point to evaluate the weight function at.

        Returns
        ---------
        y : float
            Weight function value at point x.
        """
        return self.diff(x) / float(x)

class HuberLoss(MLoss):
    """
    Huber loss function: 
 
    .. math::
        f_\\delta (x) = \\begin{cases}
        \\frac{x^2}{2},\\,\\,\\,&\\vert x\\vert\\leq\\delta,\\\\
                \\delta\\left(\\vert x\\vert - \\frac{\\delta}{2} \\right),\\,\\,\\,&\\vert
        x\\vert\\geq\\delta
        \\end{cases}

    Parameters
    ------------
    delta : float >= 0
        Delta parameter in function definition.
    """
    def __init__(self, delta):
        assert isinstance(delta, Number)
        assert float(delta) >= 0, 'delta has to be non-negative.'

        self.delta = float(delta)
        self.delta_half_square = (self.delta ** 2) / 2.

    def __call__(self, x):
        x_flt = float(x)
        assert x_flt >= 0
        if x_flt <= self.delta:
            return (x_flt ** 2) / 2.
        else:
            return self.delta * x_flt - self.delta_half_square
    
    def diff(self, x):
        pass

    def weight(self, x):
        x_flt = float(x)
        assert x_flt >= 0
        if x_flt <= self.delta:
            return 1.
        else:
            return self.delta / x_flt
