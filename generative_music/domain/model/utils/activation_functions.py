"""Activation Functions module for neural networks.

This module includes common activation functions: GELU, ReLU, Tanh, and Sigmoid.
"""
from enum import Enum


class ActivationFunctions(Enum):
    """This class is an enumeration of common activation functions.

    It includes GELU, ReLU, Tanh, and Sigmoid as available activation functions.
    This enumeration can be used to easily select and switch
    between activation functions in custom models and adaptations.
    """

    GELU = "gelu"
    RELU = "relu"
    TANH = "tanh"
    SIGMOID = "sigmoid"
