"""
neuron

Holds the activations for a specific neuron for a set of inputs
"""
import numpy as np

from kmeans.activation import Activation, LabelType


class Neuron(Activation):
    """All the activations for a specific neuron in a network

    Structurally this identical to a 1D Activation object;
    it's constructed by taking the data from an ActivationTable at 90'

    """

    def __init__(self, labels: LabelType, vector: np.array) -> None:
        if vector is not None and vector.shape is not () and vector.ndim is not 1:
            raise ValueError("Neurons must have 1D vectors")
        super(Neuron, self).__init__(labels, vector, None)
