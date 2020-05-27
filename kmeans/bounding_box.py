#!/usr/bin/env python3
""" Bounding box
Provides a variable dimensionaly bounding box
"""
import abc

import numpy as np

from kmeans.activation import Activation


class BoundingBox(metaclass=abc.ABCMeta):
    """
    Abstract data type that will represent a bounding box.
    """

    # Margins for rounding numbers to allow for float variance.
    # Used in Lowerbound to determine how close to 0 an activation needs to be
    # before it is treated as 0
    margin = 0.0001
    zero_margin = 0.0000001

    def __init__(self, ll: np.array=None, ur: np.array=None):
        """
        Base constructor. Not that we can't do much until we have a point added.
        """
        self.lower_left = ll
        self.upper_right = ur

    @abc.abstractmethod
    def expand(self, point: Activation) -> None:
        """
        Expand the bounding box so that it includes the supplied point
        :param point: to add
        """
        raise NotImplementedError("Use a subtype")

    @abc.abstractmethod
    def get_centre(self) -> np.array:
        """
        calculate the centre point of the bounding box
        :return: a numpy array
        """
        if self.lower_left is None:
            # Not yet initialised
            return None
        return np.mean((self.lower_left, self.upper_right), axis=0)

    @abc.abstractmethod
    def contains(self, point: Activation) -> bool:
        """
        test whether the given box contains a specific point
        :param point: the point to add
        :return: true iff the point is contained within the bounding box
        """
        raise NotImplementedError("Use a subtype")

    @abc.abstractmethod
    def escape_distance_L1(self, point: Activation) -> float:
        """
        calculate an approximation of how far outside the cluster a given point falls.
        this is defined as the sum of the distance from the closest limit to the point on each axis
        for all axes where the point falls outside the cluster.
        L1 distance or block distance (sum of errors)
        :param point: a point to test
        :return: the escape distance for the point
        """
        under = self.lower_left - point.vector

        under_distance = np.sum(np.where(under > 0, under, 0.0))

        over = point.vector - self.upper_right

        over_distance = np.sum(np.where(over > 0, over, 0.0))

        return under_distance + over_distance

    @abc.abstractmethod
    def _get_shortest_edge_vector(self, point:Activation):
        """
        Calculate the shortest vector to the edge of the bounding box from a _contained_ point.
        :param point:
        :return: a vector representing the distance, or None
        """
        if not self.contains(point):
            return None

        to_under = point.vector - self.lower_left
        to_upper = self.upper_right - point.vector
        # As we _know_ the point is contained, all values in to_under and to_upper are >=0
        return np.minimum(to_under, to_upper)

    def internal_distance_L1(self, point:Activation) -> float:
        """ Calculates the minimum distance from a _contained_ point to the
            edge of the bounding box using the L1 mestric
        """
        closest = self._get_shortest_edge_vector(point)
        if not closest:
            raise ValueError("Point must be contained")

        return np.sum(closest)

    def internal_distance_L2(self, point: Activation) -> float:
        """ Calculates the minimum distance from a _contained_ point to the
            edge of the bounding box using the L2 metric
        """
        closest = self._get_shortest_edge_vector(point)
        if not closest:
            raise ValueError("Point must be contained")

        return np.linalg.norm(closest)

    @abc.abstractmethod
    def escape_distance_L0(self, point: Activation) -> int:
        """
        calculates on how many dimensions a given point falls outside the cluster.
        this is defined as the sum of the distance from the closest limit to the point on each axis
        for all axes where the point falls outside the cluster.
        L0 distance or no. nonzero errors, equiv. to weight in binary vectors or Hamming distance
        :param point: a point to test
        :return: the escape distance for the point
        """
        under = self.lower_left - point.vector

        under_distance = np.sum(np.where(under > 0, 1.0, 0.0))

        over = point.vector - self.upper_right

        over_distance = np.sum(np.where(over > 0, 1.0, 0.0))

        return under_distance + over_distance

    @abc.abstractmethod
    def escape_distance_L2(self, point: Activation) -> float:
        """
        calculate how far outside the cluster a given point falls.
        this is defined as the sum of the distance from the closest limit to the point on each axis
        for all axes where the point falls outside the cluster.
        L2 distance (sum of errors) pythagorean distance
        :param point: a point to test
        :return: the escape distance for the point
        """
        under = self.lower_left - point.vector

        under_distance = np.where(under > 0, under, 0.0)

        over = point.vector - self.upper_right

        over_distance = np.where(over > 0, over, 0.0)

        return np.linalg.norm(under_distance + over_distance)

    def __eq__(self, other):
        """
        Equality test
        :param other: the object to test against
        :return: true iff the object is considered equal
        """
        if not isinstance(other, self.__class__):
            return False
        return np.all(self.lower_left == other.lower_left) and \
            np.all(self.upper_right == other.upper_right)

    def __hash__(self):
        """ Hash function for sets"""
        return hash((tuple(self.upper_right),
                     tuple(self.lower_left)))

    def __repr__(self):
        """ bounding box representation string"""
        return 'BoundingBox: {} to {}'.format(
            self.lower_left, self.upper_right)


class LowerDimensionedBoundingBox(BoundingBox):
    """ LowerDimensionedBoundingBox
    Defines a bounding box to contain a region of point space.
    These bounding boxes only exist in a subset of point space's dimensions,
    specifically those to which it has been shown a non-0 value in a point
    """

    def __init__(self) -> None:
        """ create an empty bounding box """
        super().__init__()
        self.upper_right = None
        # an array of all dimensions in which we have seen a non-zero
        # activations.
        self.filter = None
        # an array of all dimensions in which we have seen a zero activation.
        self.zero_filter = None
        self.count = 0

    def expand(self, activation: Activation):
        """ Increase the boundingbox to include the passed activation."""
        self.count += 1

        if self.lower_left is None:
            # Make sure that we actually have a box
            self.lower_left = np.where(
                activation.vector != 0, activation.vector - self.margin, 0.0)
            self.upper_right = np.where(
                activation.vector != 0, activation.vector + self.margin, 0.0)
            # self.filter = np.logical_or(activation.vector > self.zero_margin, activation.vector < -self.zero_margin)
            self.filter = activation.vector != 0

            self.zero_filter = np.logical_not(self.filter)
            return

        # Note: we assume all points have the same dimensionality.
        self.filter = np.logical_or(self.filter, (activation.vector != 0.0))
        self.zero_filter = np.logical_or(
            self.zero_filter, (activation.vector == 0.0))
        self.lower_left = np.min((self.lower_left, np.where(
            activation.vector != 0, activation.vector - self.margin, 0.0)), axis=0)
        self.upper_right = np.max((self.upper_right, np.where(
            activation.vector != 0, activation.vector + self.margin, 0.0)), axis=0)

    def get_centre(self) -> np.array:
        """ calculate the point at the centre of the bounding box."""
        if self.lower_left is None:
            return None
        return np.where(self.filter, np.mean(
            (self.lower_left, self.upper_right), axis=0), 0.0)

    def contains(self, activation: Activation) -> bool:
        """ Determine if a activation lies within the box.
        Note that a activation on the lower border is considered 'in',
        while one on the upper border is considered 'out'.
        """
        # if it has a value on an axis in which we don't exist, skip
        if np.any(np.where(self.filter, 0.0, activation.vector)):
            return False

        allowed_zeros = np.logical_and(
            self.zero_filter, activation.vector == 0.0)

        # Not that we take containment as inclusive on the lower bound only
        contain = np.logical_and(
            self.lower_left <= activation.vector, self.upper_right > activation.vector)

        # ignore filtered overlaps
        distance = np.size(
            self.filter) - np.sum(np.logical_or(np.where(self.filter, contain, True),
                                                allowed_zeros))

        return distance == 0

    def _get_shortest_edge_vector(self, point:Activation):
        """
        Calculate the shortest vector to the edge of the bounding box from a _contained_ point.
        :param point:
        :return: a vector representing the distance, or None
        """
        if not self.contains(point):
            return None

        to_under = np.where(self.filter, point.vector - self.lower_left, 0)
        to_upper = np.where(self.filter, self.upper_right - point.vector, 0)
        # As we _know_ the point is contained, all values in to_under and to_upper are >=0
        return np.minimum(to_under, to_upper)

    def escape_distance_L1(self, point: Activation) -> float:
        """L1 Norm
        """
        outside_distance = abs(np.sum(point.vector[self.zero_filter]))

        under = np.where(self.filter, self.lower_left - point.vector, 0.0)

        under_distance = np.sum(np.where(under > 0, under, 0.0))

        over = np.where(self.filter, point.vector - self.upper_right, 0.0)

        over_distance = np.sum(np.where(over > 0, over, 0.0))

        return outside_distance + under_distance + over_distance

    def escape_distance_L2(self, point: Activation) -> float:
        """L2 Norm - pythaogorean distance
        """
        mask = np.ones(len(self.filter), np.bool)
        mask[self.filter] = False
        outside_distance = np.where(mask, point.vector, 0.0)

        zero_values = np.logical_and(self.zero_filter, point.vector)

        under = np.where(zero_values, 0.0, self.lower_left - point.vector)

        under_distance = np.where(under > 0, under, 0.0)

        over = np.where(self.filter, point.vector - self.upper_right, 0.0)

        over_distance = np.where(over > 0, over, 0.0)

        return np.linalg.norm(
            outside_distance + under_distance + over_distance)

    def escape_distance_L0(self, point: Activation) -> int:
        """L0 Norm - no of non zero dimensions
        """
        mask = np.ones(len(self.filter), np.bool)
        mask[self.filter] = False
        # vector of dimensions which have always been zero
        outside_vector = point.vector[mask]
        outside_distance = np.sum(np.where(outside_vector != 0, 1, 0))

        zero_values = np.logical_and(self.zero_filter, point.vector)

        under = np.where(zero_values, 0.0, self.lower_left - point.vector)

        under_distance = np.sum(np.where(under > 0, 1.0, 0.0))

        over = np.where(self.filter, point.vector - self.upper_right, 0.0)

        over_distance = np.sum(np.where(over > 0, 1.0, 0.0))

        return outside_distance + under_distance + over_distance

    def escape_distance_zip(self, point) -> int:
        """ Determine the total distance by which a point falls outside the bounding box.
        """

        distance = 0.0

        for value, lower, upper, present, zero_ok in zip(point.vector,
                                                         self.lower_left,
                                                         self.upper_right,
                                                         self.filter,
                                                         self.zero_filter):
            if value < lower:
                if zero_ok and value == 0.0:
                    continue
                if not present:
                    distance += value
                    continue
                distance += (lower - value)

            if value > upper:
                distance += (value - upper)

            if not present:
                distance += value
                continue

        return distance

    def escape_count(self, activation) -> int:
        """ Determine the count of axes in which the given activation falls
        outside the bounding box.
        Note that a activation on the lower border is considered 'in',
        while one on the upper border is considered 'out'.
        L0 norm
        """
        # TODO: handle
        # if it has a value on an axis in which we don't exist, skip

        allowed_zeros = np.logical_and(
            self.zero_filter, activation.vector == 0.0)

        # Not that we take containment as inclusive on the lower bound only
        contain = np.logical_and(
            self.lower_left <= activation.vector, self.upper_right > activation.vector)

        # ignore filtered overlaps
        distance = np.size(self.filter) - \
            np.sum(np.logical_or(np.where(self.filter, contain, True),
                                 allowed_zeros))

        return distance

    def overlap(self, other):
        """ Determine if two boxes overlap """
        # Do these boxes overlap in every dimension?
        # if ah<=bl or bh<=al in every dimension, then no overlap
        overlaps = np.logical_or(self.upper_right <= other.lower_left,
                                 other.upper_right <= self.lower_left)
        filters = np.logical_and(self.filter, other.filter)
        return not np.any(np.logical_and(overlaps, filters))

    def __eq__(self, other):
        """ test for equality. """
        return np.all(self.upper_right == other.upper_right) and \
            np.all(self.lower_left == other.lower_left) and \
            np.all(self.filter == other.filter) and \
            np.all(self.zero_filter == self.zero_filter)

    def __hash__(self):
        """ Hash function for sets"""
        return hash((tuple(self.upper_right),
                     tuple(self.lower_left),
                     tuple(self.filter),
                     tuple(self.zero_filter)))


class PartialDataBoundingBox(BoundingBox):
    """
    A bounding box where some values may be None
    """

    def get_centre(self) -> np.array:
        result = np.zeros(len(self.lower_left), np.float)
        result += self.lower_left[self.mask]
        result += self.upper_right[self.mask]
        result /= 2.0
        result[self.mask] = None
        return result

    def contains(self, point: Activation) -> bool:
        if self.lower_left is None:
            # Not yet created, so...
            return False

        # Calculate those axes on which both the bounding box and the point
        # have values
        mask = self.mask.copy()
        mask[np.where(point.vector==None)] = False

        if not np.any(mask):
            # There is no axes of overlap, so assume false
            return False

        under = self.lower_left[mask] - point.vector[mask]
        if np.any(under > 0.0):
            return False

        over = point.vector[mask] - self.upper_right[mask]
        if np.any(over > 0.0):
            return False

        # Seems Good.
        return True

    def _get_shortest_edge_vector(self, point: Activation):
        """
        Calculate the shortest vector to the edge of the bounding box from a _contained_ point.
        :param point:
        :return: a vector representing the distance, or None
        """
        if not self.contains(point):
            return None

        to_under = np.where(self.mask, point.vector - self.lower_left, 0)
        to_upper = np.where(self.mask, self.upper_right - point.vector, 0)
        # As we _know_ the point is contained, all values in to_under and to_upper are >=0
        return np.minimum(to_under, to_upper)

    def escape_distance_L1(self, point: Activation) -> float:
        # Calculate those axes on which both the bounding box and the point
        # have values
        mask = self.mask.copy()
        mask[np.where(point.vector == None)] = False

        under = self.lower_left[mask] - point.vector[mask]
        under_distance = np.sum(np.where(under > 0, under, 0.0))

        over = point.vector[mask] - self.upper_right[mask]
        over_distance = np.sum(np.where(over > 0, over, 0.0))

        return under_distance + over_distance

    def escape_distance_L0(self, point: Activation) -> int:
        # Calculate those axes on which both the bounding box and the point
        # have values
        mask = self.mask.copy()
        mask[np.where(point.vector == None)] = False

        under = self.lower_left[mask] - point.vector[mask]
        under_distance = np.sum(np.where(under > 0, 1.0, 0.0))

        over = point.vector[mask] - self.upper_right[mask]
        over_distance = np.sum(np.where(over > 0, 1.0, 0.0))

        return under_distance + over_distance

    def escape_distance_L2(self, point: Activation) -> float:
        # Calculate those axes on which both the bounding box
        # and the point have values
        mask = self.mask.copy()
        mask[np.where(point.vector == None)] = False

        under = self.lower_left[mask] - point.vector[mask]
        under_distance = np.where(under > 0, under, 0.0)

        over = point.vector[mask] - self.upper_right[mask]
        over_distance = np.where(over > 0, over, 0.0)

        return np.linalg.norm(under_distance + over_distance)

    def expand(self, point: Activation):
        if self.lower_left is None:
            # This is the first point. Use margins so that we actually have a box
            # Of course, this point may be partial...
            self.lower_left = point.vector.copy()
            self.upper_right = point.vector.copy()
            self.mask = np.ones(len(self.lower_left), np.bool)
            self.mask[np.where(self.lower_left == None)] = False
            self.lower_left[self.mask] -= self.margin
            self.upper_right[self.mask] += self.margin
            return
        # first set the minimum of the intersections
        # TODO:: Don't go via lists.
        mask = self.mask.copy()
        mask[np.where(point.vector == None)] = False
        newmask = np.invert(self.mask.copy())
        newmask[np.where(point.vector == None)] = False
        oldmask = self.mask.copy()
        oldmask[np.where(point.vector != None)] = False

        overlap_min = np.array([min(x, y) if m else None for (x, y, m) in
                                zip(self.lower_left, point.vector, mask)])
        overlap_min[newmask] = point.vector[newmask]
        overlap_min[oldmask] = self.lower_left[oldmask]
        self.lower_left = overlap_min

        overlap_max = np.array([max(x, y) if m else None for (x, y, m) in
                                zip(self.upper_right, point.vector, mask)])
        overlap_max[newmask] = point.vector[newmask]
        overlap_max[oldmask] = self.upper_right[oldmask]
        self.upper_right = overlap_max
        self.mask[np.where(self.lower_left == None)] = False
