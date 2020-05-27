""" Activation
provides a holder for a set of activations for a specific input,
taken from a neural network
"""
import json
from typing import Union, List, Tuple

import numpy as np

LabelType = Union[str, List[str]]
ActivationIndexType = Tuple[str, int]


class ActivationEncoder(json.JSONEncoder):
    """ custom json encoder for points. """

    def default(self, obj):  # pylint: disable=method-hidden
        """ encode """
        if not isinstance(obj, Activation):
            return json.JSONEncoder.default(self, obj)
        # Don't actually save the values
        return {'index': obj.index, 'label': obj.label, 'labels': obj.labels}


class Activation(object):
    """The N-d set of neuron activations for a single image

    Attributes:
        vector (np.array): an N-dimensional array representing a point in space.
        label (str): the assigned label for this data point
        labels ([str]): the assigned labels for this data point (if more than one)
        index (int, optional): the index from the activationTable representing
            which source data file this comes from.
    """
    _encoder = ActivationEncoder()

    @classmethod
    def from_json(cls, point_source, json_text):
        """Given a point source and a json file, recreate the Activation object"""
        data = json.loads(json_text)
        point = point_source.getPoint(data['index'])
        # Fairly limited amount of verification we can do
        assert point.label == data['label']
        return point

    def __init__(self, labels: LabelType, vector: np.array,
                 index: ActivationIndexType = None) -> None:
        """ Create a point with vector v and a classification label

        """
        self.vector = np.array(vector)
        if isinstance(labels, str):
            self.label = labels
            self.labels = [labels]
        else:
            if len(labels) > 0:
                self.label = labels[0]
            else:
                self.label = None
            self.labels = labels
        self.index = index
        self.__array_interface__ = self.vector.__array_interface__

    def hydrate(self, activation_table: 'ActivationTable') -> None:
        """ Given a point source, get the vector values.
        """
        point = activation_table.get_activation(self.index)
        self.vector = point.vector
        self.__array_interface__ = self.vector.__array_interface__

    def dessicate(self) -> None:
        """ Reduce memory usage by dumping the vector values.
        """
        self.vector = None

    def encode(self) -> str:
        """ Get the json encoded form of this point. """
        return Activation._encoder.encode(self)

    def dimensionality(self):
        """ get the number of dimensions to this point. """
        if self.vector.shape is ():
            return 0
        if len(self.vector.shape) is 1:
            return 1
        _, dim = self.vector.shape
        return dim

    def standardise(self):
        """ Scale the space in which the point exists such that the largest value is 1.0.
        """
        if self.vector.shape is ():
            return
        if self.dimensionality() != 1:
            # TODO: implement
            raise NotImplementedError
        max_value = 1.0 * max(self.vector)
        if max_value == 0.0:
            # Nothing to do
            return
        self.vector = self.vector.astype('float64') / max_value

    def __repr__(self):
        """ point representation string. """
        return 'Activation: {} (idx{})'.format(self.labels, self.index)

    def __eq__(self, other):
        """Override the default Equals behavior"""
        if isinstance(other, self.__class__):
            return self.index == other.index and \
                self.label == other.label and \
                self.labels == other.labels and \
                np.all(self.vector == other.vector)
        return NotImplemented

    def __ne__(self, other):
        """Define a non-equality test"""
        if isinstance(other, self.__class__):
            return not self.__eq__(other)
        return NotImplemented

    def __hash__(self):
        """Override the default hash behavior (that returns the id or the object)"""
        return hash(
            (tuple(
                self.vector), self.index, tuple(
                self.labels), self.label))
