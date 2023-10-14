import math
from typing import Collection

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
    return ', '.join(li[:-2] + [f'{"," if len(li) > 2 else ""} and '.join(li[-2:])])


def collect(a, t: type[Collection], empty_if_none=False):
    if a:
        return (
            a if isinstance(a, t) 
            else t(a) 
            if isinstance(a, Collection) and not isinstance(a, str) 
            else t([a])
        )
    elif empty_if_none:
        return t()


def main():
    import doctest
    doctest.testmod()

if __name__ == '__main__':
    main()
