"""
An implementation of Robust PCA using M-estimator loss. A particular case with the Huber loss
function is described in the paper: 
    
    B.T. Polyak, M.V. Khlebnikov: Principal Component Analysis: Robust Variants, 2017.

"""
from .m_est_rpca import *
import loss
