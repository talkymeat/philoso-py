import numpy as np
from trees import Tree


def no_coerce_array(_trees) -> np.ndarray:
    """Creates arrays for trees (from a single tree or a Sequence 
    containing trees) WITHOUT coercing the trees into np.ndarrays
    themselves. Numpy tends to coerce anything sequence-like into
    an array if it is passed to np.array.

    >>> from test_materials import tree_lists as tls
    >>> from trees import Tree
    >>> tarrays = [no_coerce_array(tl) for tl in tls]
    >>> all_trees = True
    >>> for tarray in tarrays:
    ...     print(tarray.shape)
    ...     for t in tarray:
    ...         all_trees = all_trees and isinstance(t, Tree)
    (3,)
    (2,)
    (2,)
    (2,)
    (2,)
    (2,)
    (1,)
    (1,)
    (1,)
    >>> all_trees
    True
    """
    if isinstance(_trees, Tree):
        _trees = [_trees]
    arr = np.empty((len(_trees),), dtype = 'object')
    arr[:] = _trees
    return arr


def main():
    import doctest
    doctest.testmod()

if __name__ == '__main__':
    main()