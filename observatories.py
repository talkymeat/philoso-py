import pandas as pd
from typing import Protocol, runtime_checkable
from collections.abc import Callable, Mapping, Iterable
# import pandera as pa
# from pandera import Column, DataFrameSchema, Check, Index
from utils import nice_list
from functools import reduce
import random

class Observatory(Protocol):
    def __init__(self, ivs: Iterable[str] | str, dvs: Iterable[str] | str, sources: Mapping, *args, **kwargs):
        ...

    def __next__(self) -> pd.DataFrame:
        ...

    def target(self) -> pd.DataFrame:
        ...

    @property
    def ivs(self) -> list[str]:
        ...

@runtime_checkable
class TargetFunc(Protocol):
    def __call__(self, ivs: pd.Series) -> pd.Series:
        ...

@runtime_checkable
class GenFunc(Protocol):
    def __call__(self, obs_len: int, **kwargs) -> pd.Series:
        ...


class StaticObservatory:
    """Takes a pre-defined `pd.DataFrame` containing values of the IV's and DV's
    
    >>> a = [197, 191, 179, 167, 157, 149, 137, 127, 109, 103, 97, 83, 73, 67, 59, 47, 41, 31, 23, 17, 11, 5, 2]
    >>> b = [3, 7, 13, 19, 29, 37, 43, 53, 61, 71, 79, 89, 101, 107, 113, 131, 139, 151, 163, 173, 181, 193, 199]
    >>> df = pd.DataFrame({'A': a, 'B': b})
    >>> df['C'] = df.A * df.B
    >>> obsy = StaticObservatory(['A', 'B'], 'C', sources=df, obs_len=15)
    >>> i=0
    >>> while i<100:
    ...     ob = next(obsy)
    ...     t = obsy.target
    ...     if ((ob.B * ob.A) == t['C']).all():
    ...         pass
    ...     else:
    ...         print(pd.DataFrame({'AxB': (ob.B * ob.A), 'C': t}))
    ...     i+=1
    """

    def __init__(self, ivs: list[str] | str, dvs: list[str] | str, sources: pd.DataFrame, *args, obs_len: int=None, **kwargs):
        if obs_len is None and args and isinstance(args[0], int):
            obs_len = args[0]
        if isinstance(ivs, str):
            ivs=[ivs]
        if isinstance(dvs, str):
            dvs=[dvs]
        if pd.Series([var in sources for var in (ivs+dvs)]).all():
            self.sources = sources
            self.ivs = ivs
            self.dvs = dvs
        else:
            wrongness = []
            for vs in (ivs, dvs):
                bad_vs = [f"'{v}'" for v in vs if v not in sources]
                if bad_vs:
                    wrongness.append(
                        f"{'D' if vs is dvs else 'Ind'}ependent Variable" +
                        f"{'s' if len(bad_vs)>1 else ''} {nice_list(bad_vs)}"
                    )
            #print(f"{' and '.join(wrongness)} not in sources")
            raise ValueError(
                f"{' and '.join(wrongness)} not in sources"
            )
        if obs_len is not None and obs_len < sources.shape[0]:
            self._obs_len = obs_len
            self.sources['__sample__'] = pd.Series([True]*self._obs_len + [False]*(self.sources.shape[0]-self._obs_len))

    @property
    def obs_len(self):
        try:
            return self._obs_len
        except AttributeError:
            self.obs_len = self.sources.shape[0]
            return self._obs_len
        

    def __next__(self) -> pd.DataFrame:
        if '__sample__' in self.sources:
            self.sources.__sample__ = self.sources.__sample__.sample(frac=1)
            return self.sources[self.sources.__sample__][self.ivs].copy()
        else:
            return self.sources[self.ivs].copy()

    @property
    def target(self) ->  pd.DataFrame:
        if '__sample__' in self.sources:
            return self.sources[self.sources.__sample__][self.dvs].copy()
        else:
            return self.sources[self.dvs].copy()

# pandera this?
class FunctionObservatory:
    """An observatory that for scenarios where GP is to be tested to approximate
    a function: the observatory takes generator functions to generate (random)
    IV values, and the target function to generate DV values for the IV values

    >>> def fn(x: pd.Series = None):
    ...     return x**2 + 2*x + 1
    >>> fo = FunctionObservatory('x', 'y', {'x': lambda n: pd.Series([random.uniform(-10, 10) for i in range(n)]), 'y': fn}, 10)
    >>> x = next(fo)
    >>> fx = x.x**2 + 2*x.x + 1
    >>> x.x.dtype
    dtype('float64')
    >>> y = fo.target
    >>> type(x)
    <class 'pandas.core.frame.DataFrame'>
    >>> type(y)
    <class 'pandas.core.frame.DataFrame'>
    >>> (y.y == fx).all()
    True
    >>> def fn(x: pd.Series = None):
    ...     return 2*x
    >>> fo = FunctionObservatory('x', 'y', {'x': lambda n: pd.Series(range(n)), 'y': fn}, 4)
    >>> x = next(fo)
    >>> x
       x
    0  0
    1  1
    2  2
    3  3
    >>> fo.target
       y
    0  0
    1  2
    2  4
    3  6
    """
#----#----#----#---@#----#----#----#---@#----#----#----#---@#----#----#----#---@

    def __init__(
            self, 
            ivs: list[str] | str, 
            dvs: list[str] | str, 
            sources: Mapping[str, GenFunc | TargetFunc],
            obs_len: int,
            **kwargs):
        self.obs_len = obs_len
        if isinstance(ivs, str):
            ivs=[ivs]
        if isinstance(dvs, str):
            dvs=[dvs]
        if pd.Series([var in sources for var in (ivs+dvs)]).all():
            self.sources = sources
            self.ivs = ivs
            self.dvs = dvs
            self._iv_data = pd.DataFrame()
            self._dv_data = pd.DataFrame()
            self._dv_ready = False
            wrong_ivs = [
                iv for iv in ivs if not isinstance(self.sources[iv], GenFunc)
            ]
            wrong_dvs = [
                dv for dv in dvs if not isinstance(self.sources[dv], TargetFunc)
            ]
            if wrong_ivs + wrong_dvs:
                errs = []
                for name, functype, wrong, expln in [
                            (
                                'Independent Variable', 'GenFunc', wrong_ivs,
                                'Genfuncs generate pandas Series outputs, and' +
                                " only take an int argument, 'obs_len', " +
                                'giving the length of the Series.'
                            ), 
                            (
                                'Dependent Variable', 'TargetFunc', wrong_dvs,
                                "TargetFuncs take as input a pandas DataFrame" +
                                " 'ivs' with the Independent Variable values," +
                                ' and output a pandas Series with the ' +
                                'corresponding Dependent Variable values.'
                            )
                        ]:
                    errs += [
                        f'{name}{"s"*bool(wrong)} {nice_list(wrong)} should ' +
                        f'be Callables of type {functype}. {expln}'
                    ] if wrong else []
                raise ValueError(' Also, '.join(errs))
        else:
            wrongness = []
            for vs in (ivs, dvs):
                bad_vs = [f"'{v}'" for v in vs if v not in sources]
                if bad_vs:
                    wrongness.append(
                        f"{'D' if vs is dvs else 'Ind'}ependent Variable" +
                        f"{'s'*bool(bad_vs)} {nice_list(bad_vs)}"
                    )
            raise ValueError(
                f"{' and '.join(wrongness)} not in sources"
            )


    def __next__(self) -> pd.DataFrame:
        self._iv_data = pd.DataFrame({
            name: self.sources[name](self.obs_len) for name in self.ivs
        })
        return self._iv_data.copy()

    @property
    def target(self) -> pd.DataFrame:
        if not self._dv_ready:
            dvdic = {
                name: self.sources[name](self._iv_data) for name in self.dvs
            }
            self._dv_data = pd.DataFrame(dvdic)
            self._dv_ready = True
        return self._dv_data.copy()


class WorldObservatory:
    def __init__(self, ivs: list[str], dvs: list[str], **kwargs):
        pass

    def __next__(self) -> (bool, pd.DataFrame):
        pass

    def target(self) -> (bool, pd.DataFrame):
        pass


if __name__ == '__main__':
    import doctest
    doctest.testmod()