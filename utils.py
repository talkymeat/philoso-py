import math
from typing import Collection
from collections.abc import MutableSet
from icecream import ic
import numpy as np
from typing import Any, Callable
from functools import reduce
import torch


def scale_to_sum(arr: np.ndarray, _sum=1) -> np.ndarray:
    return arr/arr.sum() * _sum

def aeq(a, b, eta=1e-6):
    return abs(a-b) < eta

class InsufficientPostgraduateFundingError(Exception):
    pass

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

def _id(elem):
    if hasattr(elem, 'metadata') and 'id' in elem.metadata:
        return elem.metadata['id']
    return id(elem)

class IDSet(MutableSet):
    def __init__(self, iterable=None):
        self._dict = {}
        if iterable:
            for it in iterable:
                self.add(it)

    def __contains__(self, other):
        return _id(other) in self._dict
    
    def __iter__(self):
        return self._dict.values().__iter__()
    
    def __len__(self):
        return len(self._dict)
    
    def add(self, elem):
        self._dict[_id(elem)] = elem

    def __sub__(self, elem):
        return IDSet([v for v in self._dict.values() if id(v)!=id(elem)])
    
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
        if _id(other) in self._dict:
            del self._dict[_id(other)]
        # else:
        #     print(_id(other))
        #     print(self._dict.keys())

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
    
def disjoin_tests(*tests: Callable[[Any|None], bool]):
    """Takes a collection of tests (callables that return boolean) that are capable 
    of taking the same *args & **kwargs, and returns a fuction which takes the same 
    *args & **kwargs and returns `True` if at least one of the tests returns `True`
    for the given *args & **kwargs

    >>> def is_true():
    ...     return True
    >>> def is_false():
    ...     return False
    >>> test = disjoin_tests(is_false, is_true, is_false)
    >>> test()
    True
    >>> test = disjoin_tests(is_false, is_false, is_false)
    >>> test()
    False
    """
    def test_disjunction(*args, **kwargs):
        return reduce(
            lambda a, b: a or b(*args, **kwargs), 
            tests, 
            False
        )
    return test_disjunction

def conjoin_tests(*tests: Callable[[Any|None], bool]):
    """Takes a collection of tests (callables that return boolean) that are capable 
    of taking the same *args & **kwargs, and returns a fuction which takes the same 
    *args & **kwargs and returns `True` if at least one of the tests returns `True`
    for the given *args & **kwargs

    >>> def is_true():
    ...     return True
    >>> def is_false():
    ...     return False
    >>> test = conjoin_tests(is_false, is_true, is_false)
    >>> test()
    False
    >>> test = disjoin_tests(is_true, is_true, is_true)
    >>> test()
    True
    """
    def test_conjunction(*args, **kwargs):
        return reduce(
            lambda a, b: a and b(*args, **kwargs), 
            tests, 
            True
        )
    return test_conjunction

class HierarchicalDict(dict):
    def __getitem__(self, idx):
        if isinstance(idx, list):
            if len(idx)==1:
                return super().__getitem__(idx[0])
            elif not idx:
                raise IndexError(
                    'An empty list is an invalid index for ' +
                    'HierarchicalDict: Provide a Hashable object ' +
                    'or a list of one or more Hashables.'
                )
            else:
                pass


def _i(item):
    if isinstance(item, torch.Tensor) and np.prod(item.shape)==1:
        return item.item()
    elif isinstance(item, np.ndarray) and np.prod(item.shape)==1:
        return item.item()
    else:
        return item


def taper(x, scale=1):
    if np.abs(x) <= scale:
        return x
    return (scale * np.sign(x)) + (np.tanh((x - scale)/scale) * scale)

def simplify(ls_: list):
    if sum([oth==ls_[0] for oth in ls_[1:]]):
        return ls_[0]
    return ls_

def name_dict(*vals) -> dict:
    return {v.__name__: v for v in vals}

def main():
    import doctest
    doctest.testmod()

if __name__ == '__main__':
    main()
