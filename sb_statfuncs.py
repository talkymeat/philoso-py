import numpy as np
import pandas as pd
from scipy import stats as st
from jsonable import SimpleJSONable, JSONableFunc
from collections.abc import Sequence
from typing import Callable

def _filter_col(col: Sequence) -> np.ndarray:
    col = np.array(col)
    return col[np.isfinite(col)]


@JSONableFunc
def mean(col): 
    """Calculates the mean of the non-nan, non-inf values in a column. 
    Has a `json` property which returns {"name": "mean"}

    Since a function decorated with a class cannot have doctests, this
    function is tested in `sb_stafunc_tests`.

    Parameters
    ----------
    col : Sequence[float|int|bool]
        column

    Returns
    -------
        float
    """
    col = _filter_col(col)
    if len(col):
        return col.mean() 
    return 0.0
    

@JSONableFunc
def mode(col):  
    """Calculates the mode of the non-nan, non-inf values in a column. 
    If there is more than one mode, calculates the mean of modes.
    Has a `json` property which returns {"name": "mode"}

    Since a function decorated with a class cannot have doctests, this
    function is tested in `sb_stafunc_tests`.

    Parameters
    ----------
    col : Sequence[float|int|bool]
        column

    Returns
    -------
        float
    """
    col = _filter_col(col)
    if len(col):
        return pd.Series(col).mode().mean() 
    return 0.0

@JSONableFunc
def std(col):  
    """Calculates the standard deviation of the non-nan, non-inf values 
    in a column. Has a `json` property which returns {"name": "std"}

    Since a function decorated with a class cannot have doctests, this
    function is tested in `sb_stafunc_tests`.

    Parameters
    ----------
    col : Sequence[float|int|bool]
        column

    Returns
    -------
        float
    """
    col = _filter_col(col)
    if len(col):
        return col.std() 
    return 0.0

@JSONableFunc
def nanage(col):  
    """Calculates the proportion of values in a column which are NaN. 
    Has a `json` property which returns {"name": "nanage"}

    Since a function decorated with a class cannot have doctests, this
    function is tested in `sb_stafunc_tests`.

    Parameters
    ----------
    col : Sequence[float|int|bool]
        column

    Returns
    -------
        float : between 0.0 and 1.0 inclusive
    """
    return np.isnan(col).mean()

@JSONableFunc
def infage(col):  
    """Calculates the proportion of values in a column which are inf or 
    -inf. Has a `json` property which returns {"name": "infage"}

    Since a function decorated with a class cannot have doctests, this
    function is tested in `sb_statfunc_tests`.

    Parameters
    ----------
    col : Sequence[float|int|bool]
        column

    Returns
    -------
        float : between 0.0 and 1.0 inclusive
    """
    return np.isinf(col).mean()

class Quantile(SimpleJSONable):
    """Callable class which calculates a quantile `q` of the non-nan,
    non-inf values of a column. `q` is fixed at initialisation time, 
    either directly, with parameter `q`, or indirectly, with parameters
    `n` and `i`, such that `q = i / (n-1)`. This later is useful where
    `n` evenly spaced quantiles from `q=0.0` (`min`) to `q=1.0` (`max`)
    are wanted. `n-1` is used here because the range is inclusive above
    and below, and `n` refers to the number of quantiles: thus `n = 9`
    should give 9 quantiles, spaced 1/8 apart.

    Parameters
    ----------
    q : float
        q of the quantile
    n : float|int
        If `q` is not given directly, `q = i / (n-1)`
    i  : float|int
        If `q` is not given directly, `q = i / (n-1)`
    """
    kwargs = ['q', 'n', 'i']
    __name__ = 'Quantile'

    def __init__(self, q: float=None, n: int=None, i: int=None, **kwargs):
        if n==0:
            raise ZeroDivisionError
        if n is None and i is None and q is not None:
            if q > 1.0:
                raise ValueError('q cannot be greater than 1.0')
            if q < 0.0:
                raise ValueError('q cannot be less than 0.0')
            self._q = q
            self.n = None
            self.i = None
        elif n is not None and i is not None and q is None:
            self._q = None
            self.n = n
            self.i = i
        else:
            vars_passed = {k: v for k, v in {'q': q, 'n': n, 'i': i}.items() if v is not None}
            raise ValueError(
                "A Quantile must be given values of `n` and `i` but not `q`, " +
                "or `q` only and not `n` and `i`. " +
                f"{vars_passed if vars_passed else 'No values'} were passed."
            )
        
    @classmethod
    def multi(cls, n: int) -> list["Quantile"]:
        """Class method which generates a list `n` of Quantiles, with
        equally spaced values of `q`, starting at `q = 0.0` and ending at
        `q = 1.0`, equivalent to `min` and `max` functions. If `n` is 
        odd, a Quantile with `q = 0.5` will also be in the list, equivalent
        to a `median` function.

        Parameters
        ----------
        n : int
            Number of Quantiles
        
        Returns
        -------
        list[Quantile]

        >>> for n in range(5):
        ...     Quantile.multi(n)
        []
        [Quantile(q=0.5)]
        [Quantile(n=2, i=0), Quantile(n=2, i=1)]
        [Quantile(n=3, i=0), Quantile(n=3, i=1), Quantile(n=3, i=2)]
        [Quantile(n=4, i=0), Quantile(n=4, i=1), Quantile(n=4, i=2), Quantile(n=4, i=3)]
        >>> Quantile.multi(-1)
        Traceback (most recent call last):
        ....
        ValueError: n should be nonnegative
        """
        match n:
            case 0:
                return []
            case 1:
                return [cls(0.5)]
            case n if n > 1:
                return [cls(i=i, n=n) for i in range(n)]
            case _:
                raise ValueError('n should be nonnegative')
        

    @property
    def q(self) -> float:
        """The q-value of the Quantile
        
        >>> [q.q for q in Quantile.multi(5)]
        [0.0, 0.25, 0.5, 0.75, 1.0]
        """
        if self.n is not None and self.i is not None and self._q is None:
            self._q = self.i/(self.n-1.0)
        return self._q

    def __call__(self, col: Sequence) -> float:
        """Calls the qth Quantile on the non-nan, non-inf members of
        col.
        
        Parameters
        ----------
        col : Sequence[float|int|bool]
            column

        Returns
        -------
            float : between 0.0 and 1.0 inclusive

        >>> from test_materials import col, nans, infs
        >>> np.random.shuffle(col)
        >>> [q(col) for q in Quantile.multi(5)] 
        [1.0, 8.75, 16.0, 32.5, 69.0]
        >>> [q(col+nans) for q in Quantile.multi(5)] 
        [1.0, 8.75, 16.0, 32.5, 69.0]
        >>> [q(col+infs) for q in Quantile.multi(5)] 
        [1.0, 8.75, 16.0, 32.5, 69.0]
        >>> [q(col+nans+infs) for q in Quantile.multi(5)] 
        [1.0, 8.75, 16.0, 32.5, 69.0]
        >>> [q(nans*4) for q in Quantile.multi(5)] 
        [0.0, 0.0, 0.0, 0.0, 0.0]
        >>> [q(infs*4) for q in Quantile.multi(5)] 
        [0.0, 0.0, 0.0, 0.0, 0.0]
        >>> [q(nans*2+infs*2) for q in Quantile.multi(5)] 
        [0.0, 0.0, 0.0, 0.0, 0.0]
        >>> [q([5.0]*16) for q in Quantile.multi(5)] 
        [5.0, 5.0, 5.0, 5.0, 5.0]
        """
        col = _filter_col(col)
        if len(col):
            return np.quantile(col, self.q)
        return 0.0
    
    @property
    def json(self) -> dict:
        """JSON representation of the Quantile: containing the name of
        the class as `'name'` and the params required to recreate the
        instance
        """
        if self.n is not None and self.i is not None:
            return {'name': self.__name__, 'n': self.n, 'i': self.i}
        return {'name': self.__name__, 'q': self.q}
    
    def __str__(self):
        if self.n is not None and self.i is not None:
            return f"Quantile(n={self.n}, i={self.i})"
        return f"Quantile(q={self.q})"
    
    def __repr__(self):
        return self.__str__()
    
    def __eq__(self, other):
        """Two Quantiles are equal if they have the same value of `q`

        >>> assert Quantile(0.5) == Quantile(n=9, i=4)
        """
        if issubclass(other.__class__, self.__class__):
            return self.q == other.q
        

    
def main():
    print('is this annoying?')
    import doctest
    print('is this annoying?')
    doctest.testmod()
    print('is this annoying?')

if __name__ == '__main__':
    main()