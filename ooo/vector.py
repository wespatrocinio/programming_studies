"""
Playing around with overload
"""

class Vector:
    """ Represent a vector in a multidimensional space. """

    def __init__(self, d):
        """
        Create a d-dimensional vector of zeros.
        
        d   Dimension of th vctor space (int)
        """
        self._coords = [0]*d

    def __len__(self):
        """ Return the dimension of the vector. """
        return len(self._coords)

    def __getitem__(self, k):
        """ Return k-th coordinate of the vector """
        return self._coords[k]

    def __setitem__(self, k, value):
        """ Set k-th coordinate of vector to given value. """
        self._coords[k] = value

    def __add__(self, other):
        """
        Return  sum of two vectors.

        other   Another Vector instance. Expected the same dimension.
        """
        if len( self) != len(other):
            raise ValueError('Dimensions must match.')
        result = Vector(len(self))
        for i in range(len(self)):
            result[i] = self[i] + other[i]
        return result

    def __eq__(self, other):
        """
        Return True if vector has same coords as other.
        
        other   Another Vector instance.
        """
        return self._coords == other._coords

    def __ne__(self, other):
        """
        Return True if vector differs from other. Rely on __eq__ method above.
        
        other   Another Vector instance.
        """
        return not self == other

    def __str__(self):
        """ Produce string representation of vector. """
        return '< {coords} >'.format(coords=self._coords)