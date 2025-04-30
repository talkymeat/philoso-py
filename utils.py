import math
from typing import Collection
from collections.abc import MutableSet
from icecream import ic
import numpy as np
from typing import Any, Callable
from functools import reduce
import torch
import re


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

def unfold_lists(source_order, *lists):
    """Combines two or more `lists` into a single list.
    All items in each list in `lists` appear in the same
    order in the returned list as they did in their original
    list, while the order in which the lists are folded together
    is given by `source_order`. Thus, if `lists` includes two
    lists of four elements each, say all `'A'` and all `'B'`:

    >>> a_s = ['A', 'A', 'A', 'A'] # lists[0]
    >>> b_s = ['B', 'B', 'B', 'B'] # lists[1]

    A `source_order` of alternating 0's and 1's, starting with 0,
    will return alternating `'A'`s and `'B'`s:

    >>> unfold_lists([0, 1, 0, 1, 0, 1, 0, 1], a_s, b_s)
    ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B']

    This also works using `True` and `False`, if `lists` is of
    length 2. Thus alternating `True, False` will return 
    alternating `'B', 'A'`:

    >>> unfold_lists([True, False, True, False, True, False, True, False], a_s, b_s)
    ['B', 'A', 'B', 'A', 'B', 'A', 'B', 'A']

    For each list `lists[i]` at index `i`, `i` must appear
    in `source_order` exactly `len(lists[i])` times. If
    if does not, or any value appears in `source_order` that
    is not an index of `lists`, a `ValueError` will be raised

    >>> unfold_lists([0, 0, 0, 0, 0, 1, 1, 1], a_s, b_s)
    Traceback (most recent call last):
        ...
    ValueError: List 0 is of length 4 but `0` appears 5 times in source_order: List 1 is of length 4 but `1` appears 3 times in source_order

    Any value appearing in `source_order` that is not an index
    of `lists` will also raise a ValueError

    >>> unfold_lists([0, 0, 0, 0, 2, 1, 1, 1, 1], a_s, b_s)
    Traceback (most recent call last):
        ...
    ValueError: 2 appears in source_order but there is no lists[2]

    If the elements of the lists in `lists` are all ints equal to
    the `i` of `list[i]`, any valid `source order` will return a 
    list identical to `source_order`. This allows us to randomly 
    generate a large number of tests:

    >>> for _ in range(10000):
    ...     zeros = [0] * np.random.randint(21)
    ...     ones = [1] * np.random.randint(21)
    ...     source_order = zeros + ones
    ...     np.random.shuffle(source_order)
    ...     assert unfold_lists(source_order, zeros, ones)==source_order, f'fail on {source_order}'
    >>> for _ in range(10000):
    ...     zeros = [0] * np.random.randint(21)
    ...     ones = [1] * np.random.randint(21)
    ...     twos = [2] * np.random.randint(21)
    ...     threes = [3] * np.random.randint(21)
    ...     source_order = zeros + ones + twos + threes
    ...     np.random.shuffle(source_order)
    ...     assert unfold_lists(source_order, zeros, ones, twos, threes)==source_order, f'fail on {source_order}'
    """
    # This will be easier to perform the needed calculations
    # on `source_order` if it's an np.ndarray 
    source_order = np.array(source_order, dtype=int)
    # if lists is actually just one list, return a copy,
    # as long as source_order has no errors - which in 
    # this case means it is a list of falsy values of
    # equal length to the one list in lists 
    if len(lists)==1 and not any(source_order) and len(source_order)==len(lists[0]):
        return list[lists[0]]
    # raise if, for some index `i` of lists, the length
    # of lists[i] is different from the number of occurences
    # of `i` in `source_order`
    # -- `zip(*np.unique(source_order, return_counts=True))` gives
    # tuples of an index i and its freqency of occurence
    # -- the first disjunct catches the case where an index i
    # occurs that isn't even in `lists`
    # -- the second disjunct catches the case where lists[i]
    # exists, but the count doesn't match the length
    # -- the walrus operator `:=` catches the list of booleans
    # that indicates which indices have problems, which can
    # then be used to generate an informative error message 
    if any(probs := [(i >= len(lists) or (len(lists[i]) != ct)) for i, ct in zip(*np.unique(source_order, return_counts=True))]):
        # Make a record, `errs` of all lists with problems
        errs = []
        # Note: probs is a list of bools that are True for
        # all indices where the corresponding index of `lists`
        # corresponds to a list with a problem 
        for i, prob in enumerate(probs):
            if prob:
                # We distinguish the case (if i <len(lists)) where 
                # the list exists but is under- or over-represented
                # in source_order, and (else) the case where it 
                # does not exist, but is represented spuriously in
                # source_order
                errs.append(
                    (
                        f"List {i} is of length {len(lists[i])} but " +
                        f"`{i}` appears {np.sum(source_order==i)} times" + 
                        " in source_order"
                    ) if i < len(lists) else (
                        f"{i} appears in source_order but there is no" +
                        f" lists[{i}]"
                    )
                )
        # When all the error messages are gathered, concatenate
        # them, colon-separated, and raise
        raise ValueError(': '.join(errs))
    # work from shallow copies of the lists, because the
    # list comprehension pops from lists, and if the same
    # lists are still needed elsewhere, it would be bad for
    # them to suddenly be empty 
    list_copies = [list(ls) for ls in lists]
    # each value of source_order is an index of `lists`: in
    # the returned list, each value is the head ideam of the
    # list in `lists` at that index, which is popped so it is
    # then removed from the (copy of the) list 
    return [list_copies[source].pop(0) for source in source_order]


def _i(item):
    if hasattr(item, 'item') and np.prod(item.shape)==1:
    # if isinstance(item, torch.Tensor) and np.prod(item.shape)==1:
    #     print('ok', item.item())
    #     return item.item()
    # elif isinstance(item, np.ndarray) and np.prod(item.shape)==1:
    #     print('np', item.item())
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

def torchify_dtype(dtype:str|type|np.dtype) -> torch.dtype:
    """Takes a string input and checks that it corresponds to a
    valid torch dtype: if it does, return the dtype: raise
    otherwise
    
    >>> torchify_dtype('float16')
    torch.float16
    >>> torchify_dtype('complex128')
    torch.complex128
    >>> torchify_dtype('float_16; print("banana"*2**100**100**100**100**100**100**100**100**100)')
    Traceback (most recent call last):
    ...
    ValueError: 'float_16; print("banana"*2**100**100**100**100**100**100**100**100**100)' is an invalid string
    >>> torchify_dtype('banana')
    Traceback (most recent call last):
    ...
    ValueError: 'banana' is not a valid torch dtype
    >>> torchify_dtype('Tensor')
    Traceback (most recent call last):
    ...
    ValueError: 'Tensor' is not a valid torch dtype
    """
    if not isinstance(dtype, str):
        dtype = str(np.dtype(dtype))
    dtype = dtype.split('.')[-1]
    if re.fullmatch(r'[a-zA-Z_]\w*', dtype):
        try:
            dt_ = eval(f'torch.{dtype}')
        except AttributeError:
            raise ValueError(f"'{dtype}' is not a valid torch dtype")
        if isinstance(dt_, torch.dtype):
            return dt_
        else:
            raise ValueError(f"'{dtype}' is not a valid torch dtype")
    else:
        raise ValueError(f"'{dtype}' is an invalid string")

def main():
    import doctest
    doctest.testmod()

if __name__ == '__main__':
    main()
