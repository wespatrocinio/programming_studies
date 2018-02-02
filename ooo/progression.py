from abc import ABCMeta, abstractmethod

class Progression(metaclass=ABCMeta):
    """ Iterator producing a generic progression. """

    def __init__(self, start=0):
        """ Initialize self._current to the first value of the progression. """
        self._current = start

    @abstractmethod
    def _advance(self):
        """
        Update self._current to the next value.

        This must be overridden by a subclass to customize progression.

        By convention, if current is set to None, this designates the end of a finite progression.
        """
    
    def __next__(self):
        """ Return the next element, or else raise StopIteration error. """
        if self._current is None:
            raise StopIteration()
        else:
            answer = self._current
            self._advance()
            return answer
    
    def print_progression(self, n):
        """
        Print next n values of the progression.

        n   number of iterations
        """
        print(' '.join(str(next(self)) for j in range(n)))


class ArithmeticProgression(Progression):
    """ Iterator producing an arithmetic progression. """

    def __init__(self, increment=1, start=0):
        """
        Create a new arithmetic progression.

        increment   the fixed constant to add to each term (default 1)
        start       the first term of the progression (default 0)
        """
        super(ArithmeticProgression, self).__init__(start)
        self._increment = increment
    
    def _advance(self):
        """ Update current value by adding the fixed increment. """
        self._current += self._increment


class GeometricProgression(Progression):
    """ Iterator producing a geometric progression. """

    def __init__(self, base=2, start=1):
        """
        Create a new geometric progression.

        base    the fixed constant multiplied to each term (default 2)
        start   the first term of the progression (default 1)
        """
        super(GeometricProgression, self).__init__(start)
        self._base = base
    

    def _advance(self):
        """ Update current value by multiplying it by the base value. """
        self._current *= self._base


class FibonacciProgression(Progression):
    """ Iterator producing a generalized Fibonacci progression. """

    def __init__(self, first=0, second=1):
        """
        Create a new Fibonacci progression

        first   the first term of the progression (default 0)
        second  the first term of the progression (default 1)
        """
        super(FibonacciProgression, self).__init__(first) # start at first
        self._prev = second - first # fictitious value preceding the first

    def _advance(self):
        """ Update current value by taking sum of previous two. """
        self._prev, self._current = self._current, self._prev + self._current

if __name__ == '__main__' :
    print('Arithmetic progression with increment 5:')
    ArithmeticProgression(5).print_progression(10)
    print('Arithmetic progression with increment 5 and start 2:')
    ArithmeticProgression(5, 2).print_progression(10)
    print('Geometric progression with default base:')
    GeometricProgression().print_progression(10)
    print('Geometric progression with base 3:')
    GeometricProgression(3).print_progression(10)
    print('Fibonacci progression with default start values:')
    FibonacciProgression().print_progression(10)
    print('Fibonacci progression with start values 4 and 6:')
    FibonacciProgression(4, 6).print_progression(10)