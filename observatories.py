import pandas as pd
import numpy as np
from typing import Protocol, runtime_checkable, TypeAlias
from collections.abc import Callable, Mapping, Iterable, Hashable
from typing import Any
# import pandera as pa
# from pandera import Column, DataFrameSchema, Check, Index
from utils import nice_list, collect
from functools import reduce
import random
from dataclasses import dataclass, field, astuple
from icecream import ic

@dataclass
class ArgsNKwargs:
    args: tuple = ()
    kwargs: dict = field(default_factory=dict)

    def apply(self, func: Callable) -> Any:
        return func(*self.args, **self.kwargs)
    
    @property
    def tuple(self):
        return astuple(self)

    def __eq__(self, obj: 'ArgsNKwargs') -> bool:
        """Two ArgsNKwargs objects are equal if they have the same args, 
        the exact same keys in kwargs, all mapping to the same values.

        >>> from logtools import Suple
        >>> eg_templates = Suple('a', 'z', None) * ('b', None) * (({'c': 3},), ({'c': 32},), ({'x': 33},), None) * (({'d': 4},), None)
        >>> eg_templates = [[tuple(filter(lambda x: p - isinstance(x, dict), eg)) for p in (1, 0)] for eg in eg_templates]
        >>> egs = Suple(*[ArgsNKwargs(eg[0], reduce(lambda a, b: {**a, **b}, eg[1], {})) for eg in eg_templates])
        >>> sum([(eg[0]==eg[1]) == bool(i%(len(egs)+1)) for i, eg in enumerate(egs**2)])
        0
        """
        args, kwargs = obj if isinstance(obj, tuple) else obj.tuple
        return self.args==args and self.kwargs==kwargs    

class TattleDict(dict):
    def __setitem__(self, __key: Any, __value: Any) -> None:
        """If the __value passed is equal to the value already mapped by __key,
        the dict is not changed and False is returned. Otherwise, makes the 
        change and returns False
        """
        self._changed = __key not in self or self[__key] != __value
        if self._changed:
            super().__setitem__(__key, __value)
        return self._changed
    
    def __delitem__(self, __key: Any) -> None:
        self._changed = True
        return super().__delitem__(__key)
    
    def __setitem__(self, __key: Any, __value: Any) -> None:
        """If the __value passed is equal to the value already mapped by __key,
        the dict is not changed and False is returned. Otherwise, makes the 
        change and returns False
        """
        self._changed = __key not in self or self[__key] != __value
        if self._changed:
            super().__setitem__(__key, __value)
        return self._changed
    
    def pop(self, _TattleDict__key: Hashable, *args, **kwargs) -> Any:
        self._changed = True
        return super().pop(_TattleDict__key, *args, **kwargs)
    
    @property
    def changed(self):
        """Returns True if the dict has changed since last time `changed` was called,
        and sets `_changed` to False: this will be set to True if a value is changed, 
        added, or removed"""
        ch = hasattr(self, '_changed') and self._changed
        self._changed = False
        return ch

AKs: TypeAlias = ArgsNKwargs|tuple[tuple, dict[str, Any]]

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
    def __call__(self, obs_len: int, *args, **kwargs) -> pd.Series:
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
        ivs=collect(ivs)
        dvs=collect(dvs)
        if '' in sources:
            for iv in ivs:
                sources[iv] = sources.get(iv, sources[''])
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
        self._dv_ready = False
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
    

class StaticFunctionObservatory(FunctionObservatory):
    def __init__(
            self, 
            ivs: list[str] | str, 
            dvs: list[str] | str, 
            sources: Mapping[str, GenFunc|TargetFunc],
            obs_len: int,
            iv_params: Mapping[str, ArgsNKwargs|tuple[tuple, dict]],
            **kwargs
        ):
        super().__init__(
            ivs=ivs, 
            dvs=dvs, 
            sources=sources,
            obs_len=obs_len,
            **kwargs
        )
        self.iv_params = TattleDict(iv_params)
        self.last_varkeys = {}
        self._dv_data = pd.DataFrame

    def set_obs_ranges(self, **iv_params: tuple[tuple, dict]|ArgsNKwargs):
        for k, v in iv_params.items():

            if k in self.iv_params:
                self.iv_params[k] = v
            else:
                raise ValueError(f'Variable {k} not in StaticFunctionObservatory')
            
    
    def __next__(self) -> pd.DataFrame:
        self.last_varkeys = {}
        iv_out = pd.DataFrame()
        for iv in self.ivs:
            # `nsel` = 'start, end, & length'
            param_tuple = (iv, self.arg_len, self.iv_params[iv]) 
            if param_tuple not in self._iv_data:
                self._dv_ready = False
                self.last_varkeys[iv] = param_tuple
                self._iv_data[param_tuple] = self.iv_params[iv].apply(self.sources[iv])
            iv_out[iv] = self._iv_data[param_tuple].copy()
        return iv_out

    @property
    def target(self) -> pd.DataFrame:
        if not self._dv_ready:
            ivdic = {
                iv: self._iv_data[self.last_varkeys[iv]] for iv in self.ivs
            }
            self._dv_data = pd.DataFrame({
                name: self.sources[name](**ivdic) for name in self.dvs
            })
            self._dv_ready = True
        return self._dv_data.copy()
        


class WorldObservatory:
    def __init__(self, ivs: list[str], dvs: list[str], **kwargs):
        pass

    def __next__(self) -> pd.DataFrame:
        pass

    def target(self) -> pd.DataFrame:
        pass


if __name__ == '__main__':
    import doctest
    doctest.testmod()