"""
The core GP-EnKF module that contains the implementations of prediction and update steps for all algorithms
"""

from .dual_gpenkf import DualGPEnKF
from .augmented_gpenkf import AugmentedGPEnKF
from .liuwest_dual_gpenkf import LiuWestDualGPEnKF
from .classic_gp import NormalGP

from .parameters import Parameters
