import numpy as np
from jsonable import SimpleJSONable, JSONableFunc
from collections.abc import Sequence
from typing import Callable

@JSONableFunc
def mean(col): 
    if len(col[np.isfinite(col)]) > 0:
        return col[np.isfinite(col)].mean() 
    return 0.0
    

@JSONableFunc
def mode(col): 
    if len(col[np.isfinite(col)]) > 0:
        return col[np.isfinite(col)].mode().mean() 
    return 0.0

@JSONableFunc
def std(col): 
    if len(col[np.isfinite(col)]) > 0:
        return col[np.isfinite(col)].std() 
    return 0.0

@JSONableFunc
def nanage(col): 
    return np.isnan(col).mean()

@JSONableFunc
def infage(col): 
    return np.isinf(col).mean()

class Quantile(SimpleJSONable):
    kwargs = ['q', 'n', 'i']

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
    def multi(cls, n: int) -> list[Callable]:
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
        if self.n is not None and self.i is not None and self._q is None:
            self._q = self.i/(self.n-1.0)
        return self._q

    def __call__(self, col: Sequence) -> float:
        col = np.array(col)
        return np.quantile(col[np.isfinite(col)], self.q)
    
    @property
    def json(self) -> dict:
        if self.n is not None and self.i is not None:
            return {'name': self.__name__, 'n': self.n, 'i': self.i}
        return {'name': self.__name__, 'q': self.q}
    
    @property
    def __name__(self):
        return 'Quantile'
    
    def __str__(self):
        if self.n is not None and self.i is not None:
            return f"Quantile(n={self.n}, i={self.i})"
        return f"Quantile(q={self.q})"
    
    def __repr__(self):
        return self.__str__()