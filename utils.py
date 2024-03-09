import math
from typing import Collection
from collections.abc import MutableSet
from icecream import ic
import numpy as np
from typing import Any, Callable
from functools import reduce

def list_transpose(ls, i, j):
    """Transposes the items in list `ls` at index `i` and `j`"""
    ls[i], ls[j] = ls[j], ls[i]

def linear_interpolated_binned_means(ls: list, n: int) -> list:
    """Divides the list `ls` in `n` many equal-sized bins, using linear
    interpolation where `bins` does not evenly divide `len(ls)`

    >>> linear_interpolated_binned_means([1,2,3,4], 3)
    [1.25, 2.5, 3.75]
    """
    bin_width = len(ls)/n
    return [
        sum(linear_interpolated_slice(ls, i*bin_width, (i+1)*bin_width))/bin_width for i in range(n)
    ]

def linear_interpolated_slice(ls, s, e):
    """Takes a slice from a list `ls` of numbers from `s` to `e`, where `s` and
    `e` can be floats. If `s` contains a fractional part, the first item in the
    slice will be `ls[floor(s)]` times one minus that fractional part. If `e`
    contains a fractional part, the last item in the slice will be
    `ls[floor(e)]` times that fractional part.
    >>> linear_interpolated_slice([8,2,1,2,4,9,3,7], 1.25, 5.75)
    [1.5, 1, 2, 4, 6.75]
    >>> linear_interpolated_slice([8,5,1,2,4,9,3,7], 1.25, 1.75)
    [2.5]
    """
    slice = []
    if math.floor(s) == math.floor(e):
        return [ls[math.floor(s)]*(e-s)]
    if s%1 != 0:
        slice += [
            (1 - (s%1))
            * ls[math.floor(s)]
        ]
    slice += ls[math.ceil(s):math.floor(e)]
    if e%1 != 0:
        slice += [
            (e%1)
            * ls[math.floor(e)]
        ]
    return slice

def nice_list(li: list[str]):
    """Takes a list of strings and formats it nicely as a string (with an
    Oxford Comma because I make the rules around here)
    
    >>> assortment = ['a large bear', 'the nuns', 'Brian Blessed', 'Liz Truss']
    >>> for i in range(4):
    ...     print(nice_list(assortment[3-i:4]))
    Liz Truss
    Brian Blessed and Liz Truss
    the nuns, Brian Blessed, and Liz Truss
    a large bear, the nuns, Brian Blessed, and Liz Truss
    """
    return ', '.join(li[:-2] + [f'{"," if len(li) > 2 else ""} and '.join(li[-2:])])


def collect(a, t: type[Collection], empty_if_none=False):
    if a is not None:
        return (
            a if isinstance(a, t)
            else t(a) 
            if isinstance(a, Collection) and not isinstance(a, str)
            else t([a])
        )
    elif empty_if_none or a is not None:
        return t()

class IDSet(MutableSet):
    def __init__(self, iterable=None):
        self._dict = {}
        if iterable:
            for it in iterable:
                self.add(it)

    def __contains__(self, other):
        return id(other) in self._dict
    
    def __iter__(self):
        return self._dict.values().__iter__()
    
    def __len__(self):
        return len(self._dict)
    
    def add(self, elem):
        self._dict[id(elem)] = elem
    
    def discard(self, other):
        """Removes `other` from the `IDSet`
        
        >>> a = [9, 99]
        >>> b = [9, 99]
        >>> s_a = IDSet([a])
        >>> print(s_a)
        IDSet([[9, 99]])
        >>> s_a.discard(b)
        >>> print(s_a)
        IDSet([[9, 99]])
        >>> s_a.discard(a)
        >>> print(s_a)
        IDSet([])
        """
        if id(other) in self._dict:
            del self._dict[id(other)]

    def __repr__(self):
        return f'IDSet({list(self._dict.values()) if self._dict else "[]"})'
    
    def array(self):
        """Returns 1-d numpy ndarray of elements
        
        >>> a = [9, 99]
        >>> b = [9, 99]
        >>> c = [7, 77, 777, 0]
        >>> d = [1, 2]
        >>> s_abc = IDSet([a, b, c])
        >>> s_abd = IDSet([a ,b, d])
        >>> s_abc.array()
        array([list([9, 99]), list([9, 99]), list([7, 77, 777, 0])], dtype=object)
        >>> s_abd.array()
        array([list([9, 99]), list([9, 99]), list([1, 2])], dtype=object)
        """
        t = tuple(self)
        a = np.empty(len(t), dtype=object)
        a[:] = t
        return a
    
def disjoin_tests(tests: Collection[Callable[[Any], bool]]):
    def test_disjunction(*args, **kwargs):
        return reduce(
            lambda a, b: a(*args, **kwargs) or b(*args, **kwargs), 
            tests, 
            False
        )
    return test_disjunction

def conjoin_tests(tests: Collection[Callable[[Any], bool]]):
    def test_conjunction(*args, **kwargs):
        return reduce(
            lambda a, b: a(*args, **kwargs) and b(*args, **kwargs), 
            tests, 
            True
        )
    return test_conjunction





def main():
    import doctest
    doctest.testmod()

if __name__ == '__main__':
    main()
