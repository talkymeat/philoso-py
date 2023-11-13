import pandas as pd
import numpy as np
from typing import Protocol, runtime_checkable, TypeAlias
from collections.abc import Callable, Mapping, Iterable, Hashable, Sequence
from typing import Any
# import pandera as pa
# from pandera import Column, DataFrameSchema, Check, Index
from utils import nice_list, collect
from dataclasses import dataclass, field, astuple
from icecream import ic
from frozendict import frozendict
from functools import reduce




def join_all(d: dict[str, Sequence|pd.Series|pd.DataFrame], how: str) -> pd.DataFrame:
    """Useful wee function that takes a dict and returns a new df such
    that the rows of the new df represent the cartesian product of the columns 
    of the old. Thus:

    ```
    {'x': [1,2], 'y': [3,4], 'z': [5,6]}
    ```

    --- becomes:
    ```
        x	y	z
    0	1	3	5
    1	1	3	6
    2	1	4	5
    3	1	4	6
    4	2	3	5
    5	2	3	6
    6	2	4	5
    7	2	4	6
    ```
    >>> df = {'x': [1,2], 'y': [3,4], 'z': [5,6]}
    >>> join_all(df, 'cross')
       x  y  z
    0  1  3  5
    1  1  3  6
    2  1  4  5
    3  1  4  6
    4  2  3  5
    5  2  3  6
    6  2  4  5
    7  2  4  6
    >>> join_all(df, 'left')
       x  y  z
    0  1  3  5
    1  2  4  6
    """
#----#----#----#---@#----#----#----#---@#----#----#----#---@#----#----#----#---@
    return reduce(
        lambda df1, df2: df1.merge(
            df2, 
            how=how, 
            left_index=(how != 'cross'), 
            right_index=(how != 'cross')
        ), 
        [
            v
            if isinstance(v, pd.DataFrame) 
            else pd.DataFrame({k: v})
            for k, v 
            in d.items()
        ]
    )

@dataclass(frozen=True)
class ArgsNKwargs:
    args: Sequence = ()
    kwargs: frozendict = frozendict()

    def apply(self, func: Callable) -> Any:
        return func(*self.args, **self.kwargs)
    
    def __post_init__(self):
        object.__setattr__(self, 'args', tuple(self.args))
        object.__setattr__(self, 'kwargs', frozendict(self.kwargs))
    
    @property
    def tuple(self):
        return astuple(self)
    
    def __hash__(self):
        return hash(self.tuple)
    
    def copy(self, *args, extend_args: bool=False, **kwargs):
        return ArgsNKwargs(
            args if args and not extend_args else self.args+args,
            {**self.kwargs, **kwargs}
        )

    def __eq__(self, obj: 'ArgsNKwargs') -> bool:
        """Two ArgsNKwargs objects are equal if they have the same args, 
        the exact same keys in kwargs, all mapping to the same values.

        >>> from m import MTuple
        >>> from functools import reduce
        >>> # this test generates a wide variety of example ArgsNKwargs, consisting of
        >>> # zero, one, or two args and zero, one, or two kwargs:
        >>> #     'a', 'z', or nothing
        >>> #     'b', or nothing
        >>> #     c=3, c=32, x=33, or nothing
        >>> #     d=4 or nothing
        >>> eg_templates = MTuple('a', 'z', None) * ('b', None) * (({'c': 3},), ({'c': 32},), ({'x': 33},), None) * (({'d': 4},), None)
        >>> eg_templates = [[tuple(filter(lambda x: p - isinstance(x, dict), eg)) for p in (1, 0)] for eg in eg_templates]
        >>> egs = MTuple(*[ArgsNKwargs(eg[0], reduce(lambda a, b: {**a, **b}, eg[1], {})) for eg in eg_templates])
        >>> # This compares all to all. The main diagonal should be True, the rest 
        >>> # False. `bool(i%(len(egs)+1))` if False on the main diagonal, True otherwise
        >>> # The sum should therefore be zero
        >>> sum([(eg[0]==eg[1]) == bool(i%(len(egs)+1)) for i, eg in enumerate(egs**2)])
        0
        """
        args, kwargs = obj if isinstance(obj, tuple) else obj.tuple
        return self.args==args and self.kwargs==kwargs    



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

    >>> from random import uniform
    >>> def fn(x: pd.Series = None):
    ...     return x**2 + 2*x + 1
    >>> fo = FunctionObservatory('x', 'y', {'x': lambda n: pd.Series([uniform(-10, 10) for i in range(n)]), 'y': fn}, 10)
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

    def __init__(
            self, 
            ivs: list[str] | str, 
            dvs: list[str] | str, 
            sources: Mapping[str, GenFunc | TargetFunc],
            obs_len: Mapping[str, int] | int,
            cartesian: bool = False,
            **kwargs):
        ivs=collect(ivs, list)
        dvs=collect(dvs, list)
        self._cartesian = cartesian
        if '' in sources:
            for iv in ivs:
                sources[iv] = sources.get(iv, sources[''])
        if isinstance(obs_len, int):
            self._def_obs_len = obs_len 
            self._obs_lens = frozendict()
        elif len(obs_len) == len(ivs) == 1:
            self._def_obs_len = obs_len.values()[0]
            self._obs_lens = frozendict()
        elif not cartesian:
            raise ValueError(
                "If you are not taking dv values for the cartesian product of" +
                " the IV values in your sample, then obs_len must be the same" +
                " for all IVs: therefore, just pass a single int to `obs_len`" +
                " when creating a FunctionObservatory"
            )
        else:
            self._def_obs_len = obs_len.get('', 0)
            if not self._def_obs_len:
                missing_obs_lens = set(ivs) - set(obs_len.keys())
                if missing_obs_lens:
                    raise ValueError(
                        f'IV{"s" if len(missing_obs_lens) > 1 else ""} ' +
                        f'{nice_list(missing_obs_lens)} missing from obs_len'
                    )
            else:
                del obs_len['']
            surplus_obs_lens = set(obs_len.keys()) - set(ivs)
            if surplus_obs_lens:
                raise ValueError(
                    f'obs_len{"s" if len(surplus_obs_lens) > 1 else ""} ' +
                    f'{nice_list(surplus_obs_lens)} not in IVs'
                )
            self._obs_lens = frozendict(obs_len)
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
    
    def obs_len(self, iv_name: str = None):
        if (iv_name is None or iv_name not in self._obs_lens) and not self._def_obs_len:
            raise ValueError(
                'This Observatory does not have a default `obs_len`: you must' +
                ' specify which iv you are retrieving the length of'
            )
#----#----#----#---@#----#----#----#---@#----#----#----#---@#----#----#----#---@
        return self._obs_lens.get(iv_name, self._def_obs_len) 
    
    def set_obs_len(self, *default, all_def: bool=False, **obs_lens):
        if len(default) > 1:
            raise TypeError(
                'StaticFunctionObservatory.set_obs_len takes only one ' +
                'positional argument'
            )
        elif default:
            self._def_obs_len = default[0]
        if all_def:
            self._obs_lens = frozendict()
        if obs_lens and not self.cartesian:
            raise ValueError(
                "If you are not taking dv values for the cartesian product of" +
                " the IV values in your sample, then obs_len must be the same" +
                " for all IVs: therefore, just pass a single int to `obs_len`" +
                " when creating a FunctionObservatory"
            )
        elif obs_lens:
            surplus_obs_lens = set(obs_lens.keys()) - set(self.ivs)
            if surplus_obs_lens:
                raise ValueError(
                    f'obs_len{"s" if len(surplus_obs_lens) > 1 else ""} ' +
                    f'{nice_list(surplus_obs_lens)} not in IVs'
                )
            remove_these = []
            for k, v in obs_lens.items():
                if v is None:
                    remove_these.append(k)
            for k, v in obs_lens.items():
                self._obs_lens = self._obs_lens.set(k, v)
            for ol in remove_these:
                self._obs_lens = self._obs_lens.delete(ol)

    def set_iv_data(self, iv_data: pd.DataFrame):
        self._iv_data = iv_data
        self._dv_ready = False

    @property
    def cartesian(self):
        return self._cartesian
    
    @cartesian.setter
    def cartesian(self, val):
        if not val and (
            not self._def_obs_len
            or sum([
                v!=self._def_obs_len for v in self._obs_lens.values()
            ])
            or sum([
                v!=self._obs_lens.value(0) for v in self._obs_lens.values()
            ])
        ):
            raise ValueError(
                'You cannot set StaticFunctionObservatory.cartesian to False' +
                ' unless all ivs have the same observation length'
            )
        self._cartesian = val

    def __next__(self) -> pd.DataFrame:
        self._iv_data = pd.DataFrame({
            name: self.sources[name](self.obs_len(name)) for name in self.ivs
        })
        self._dv_ready = False
        if self.cartesian:
            self._iv_data = join_all(self._iv_data, how='cross')
        return self._iv_data.copy()

    @property
    def target(self) -> pd.DataFrame:
        if not self._dv_ready:
            self._dv_data = pd.DataFrame()
            for name in self.dvs:
                self._dv_data[name] = self.sources[name](**self._iv_data) 
            self._dv_ready = True
        return self._dv_data.copy()
    

class StaticFunctionObservatory(FunctionObservatory):
    def __init__(
            self, 
            ivs: Sequence[str] | str, 
            dvs: Sequence[str] | str, 
            sources: Mapping[str, GenFunc|TargetFunc],
            obs_len: Mapping[str, int] | int,
            iv_params: Mapping[str, ArgsNKwargs|tuple[tuple, dict]],
            cartesian: bool = False,
            **kwargs
        ):
        """Memoized version of FunctionObservatory. Assumes that functions are
        pure - output will be same if args are same. If a random `GenFunc` is 
        used, the same output will be returned as long as it is given the same
        arguments.
        
        >>> sfo = example_sfo()
        >>> #ic.disable()
        >>> iv_vals = next(sfo)
        gen x
        gen y
        >>> list(iv_vals.x)
        [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]
        >>> list(iv_vals.y)
        [-4.0, -2.0, 0.0, 2.0, 4.0, 6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0]
        >>> h0, h1 = hash(tuple(iv_vals.x)), hash(tuple(iv_vals.y))
        >>> dv_vals = sfo.target
        called sumsq
        called diffcu
        >>> list(dv_vals.a)
        [20.0, 8.0, 4.0, 8.0, 20.0, 40.0, 25.0, 13.0, 9.0, 13.0, 25.0, 45.0, 32.0, 20.0, 16.0, 20.0, 32.0, 52.0, 41.0, 29.0, 25.0, 29.0, 41.0, 61.0]
        >>> list(dv_vals.b)
        [72.0, 16.0, 8.0, 0.0, -56.0, -208.0, 91.0, 35.0, 27.0, 19.0, -37.0, -189.0, 128.0, 72.0, 64.0, 56.0, 0.0, -152.0, 189.0, 133.0, 125.0, 117.0, 61.0, -91.0]
        >>> h2, h3 = hash(tuple(dv_vals.a)), hash(tuple(dv_vals.b))
        >>> iv_vals2 = next(sfo)
        >>> h0 == hash(tuple(iv_vals2.x)), h1 == hash(tuple(iv_vals2.y)) 
        (True, True)
        >>> dv_vals2 = sfo.target
        >>> h2 == hash(tuple(dv_vals2.a)), h3 == hash(tuple(dv_vals2.b)) 
        (True, True)
        >>> sfo.set_iv_param('y', mult=1)
        >>> iv_vals3 = next(sfo)
        gen y
        >>> list(iv_vals3.y)
        [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
        >>> h0 == hash(tuple(iv_vals3.x)), h1 == hash(tuple(iv_vals3.x)) 
        (True, False)
        >>> dv_vals3 = sfo.target
        called sumsq
        called diffcu
        >>> list(dv_vals3.a)
        [8.0, 5.0, 4.0, 5.0, 8.0, 13.0, 13.0, 10.0, 9.0, 10.0, 13.0, 18.0, 20.0, 17.0, 16.0, 17.0, 20.0, 25.0, 29.0, 26.0, 25.0, 26.0, 29.0, 34.0]
        >>> list(dv_vals3.b)
        [16.0, 9.0, 8.0, 7.0, 0.0, -19.0, 35.0, 28.0, 27.0, 26.0, 19.0, 0.0, 72.0, 65.0, 64.0, 63.0, 56.0, 37.0, 133.0, 126.0, 125.0, 124.0, 117.0, 98.0]
        >>> h0, h1 = hash(tuple(iv_vals3.x)), hash(tuple(iv_vals3.y))
        >>> h2, h3 = hash(tuple(dv_vals3.a)), hash(tuple(dv_vals3.b))
        >>> iv_vals4 = next(sfo)
        >>> h0 == hash(tuple(iv_vals4.x)), h1 == hash(tuple(iv_vals4.y)) 
        (True, True)
        >>> dv_vals4 = sfo.target
        >>> h2 == hash(tuple(dv_vals4.a)), h3 == hash(tuple(dv_vals4.b)) 
        (True, True)
        >>> sfo.set_obs_len(x=2, y=3)
        >>> iv_vals5 = next(sfo)
        gen x
        gen y
        >>> list(iv_vals5.x)
        [2.0, 2.0, 2.0, 5.0, 5.0, 5.0]
        >>> list(iv_vals5.y)
        [-2.0, 0.5, 3.0, -2.0, 0.5, 3.0]
        >>> dv_vals5 = sfo.target
        called sumsq
        called diffcu
        >>> list(dv_vals5.a)
        [8.0, 4.25, 13.0, 29.0, 25.25, 34.0]
        >>> list(dv_vals5.b)
        [16.0, 7.875, -19.0, 133.0, 124.875, 98.0]
        """
        super().__init__(
            ivs=ivs, 
            dvs=dvs, 
            sources=sources,
            obs_len=obs_len,
            cartesian=cartesian,
            **kwargs
        )
        self._iv_params  = frozendict(self.format_iv_params(**iv_params))
        self.last_varkeys = {}
        self._dv_data = {}
        self._iv_data = {}
        self._last_dv_out = pd.DataFrame()
        self._state_hash = self.calc_state_hash()
        self._iv_ready = False

    def calc_state_hash(self):
        return hash((self.cartesian, self._def_obs_len, self._obs_lens, self._iv_params ))
    
    def check_hash(self):
#----#----#----#---@#----#----#----#---@#----#----#----#---@#----#----#----#---@
        """If any of the parameters of the SFO are changed, iv and dv values 
        need to be recalculated: if nothing has changed, then the SFO can just
        return previously calculated values. For this reason, a hash of the SFO 
        state is stored, and can be recalculated and compared to the stored 
        value: if the hash is changed, the state has changed

        >>> #ic.disable()
        >>> sfo = example_sfo()
        >>> sfo._iv_ready # after __init__
        False
        >>> ivs_ = next(sfo)
        gen x
        gen y
        >>> sfo._iv_ready
        True
        >>> dvs_ = sfo.target
        called sumsq
        called diffcu
        >>> sfo._iv_ready
        True
        >>> ivs_ = next(sfo)
        >>> sfo._iv_ready
        True
        >>> dvs_ = sfo.target
        >>> sfo._iv_ready
        True
        >>> sfo.cartesian = True
        >>> sfo._iv_ready
        True
        >>> ivs_ = next(sfo)
        >>> sfo.cartesian = False
        Traceback (most recent call last):
            ....
        ValueError: You cannot set StaticFunctionObservatory.cartesian to False unless all ivs have the same observation length
        >>> sfo.set_obs_len(4, all_def=True)
        >>> sfo._def_obs_len
        4
        >>> sfo._obs_lens
        frozendict.frozendict({})
        >>> sfo.cartesian
        True
        >>> sfo._iv_ready # after obs len chg
        False
        >>> ivs_ = next(sfo)
        gen y
        >>> ivs_.shape
        (16, 2)
        >>> sfo._iv_ready
        True
        >>> dvs_ = sfo.target
        called sumsq
        called diffcu
        >>> sfo._iv_ready
        True
        >>> sfo.cartesian = False
        >>> sfo._iv_ready
        False
        >>> ivs_ = next(sfo)
        >>> sfo._iv_ready
        True
        >>> dvs_ = sfo.target
        called sumsq
        called diffcu
        >>> sfo._iv_ready
        True
        >>> ivs_.shape
        (4, 2)
        >>> dvs_.shape
        (4, 2)
        >>> sfo.set_iv_param('x', 'decl', 'something')
        >>> sfo._iv_ready
        False
        """
        h = self.calc_state_hash()
        if h != self._state_hash:
            self._state_hash = h
            self._iv_ready = False

    def format_iv_params(self, **iv_params: tuple[tuple, dict]|ArgsNKwargs):
        iv_p = {}
        for k, v in iv_params.items():
            if k in self.ivs:
                iv_p[k] = v if isinstance(v, ArgsNKwargs) else ArgsNKwargs(*v)
            else:
                raise ValueError(f'Variable {k} not in StaticFunctionObservatory')
        return iv_p
        
    def set_iv_param(self, iv: str, *args, extend_args: bool=False, **kwargs):
        self._iv_params = self._iv_params.set(
            iv, 
            self._iv_params[iv].copy(*args, extend_args=extend_args, **kwargs)
        )
        self.check_hash()

    @property
    def cartesian(self):
        return self._cartesian
    
    @cartesian.setter
    def cartesian(self, val):
        FunctionObservatory.cartesian.fset(self, val)
        self.check_hash()

    def set_obs_len(self, *default, all_def: bool = False, **obs_lens):
        super().set_obs_len(*default, all_def=all_def, **obs_lens)
        self.check_hash()
            
    def __next__(self) -> pd.DataFrame:
        self.check_hash()
        if self._iv_ready and hasattr(self, '_iv_df'):
            return self._iv_df.copy()
        iv_data = {}
        iv_raw = {}
        changed = False
        param_tuples = tuple([(iv, self.obs_len(iv), self._iv_params[iv]) for iv in self.ivs])
        for i, iv in enumerate(self.ivs):
            if not changed and self.cartesian and (iv,) + param_tuples in self._iv_data:
                iv_data[iv] = self._iv_data[(iv,) + param_tuples]
            else:
                if param_tuples[i] in self._iv_data:
                    iv_raw[iv] = self._iv_data[param_tuples[i]]
                else:
                    self._iv_data[
                        param_tuples[i]
                    ] = iv_raw[iv] = pd.Series(
                        self.sources[iv](
                            param_tuples[i][1], 
                            *param_tuples[i][2].args,
                            **param_tuples[i][2].kwargs
                        )
                    )
                if not self.cartesian:
                    self.last_varkeys[iv] = param_tuples[i]
                if not changed:
                    for missed_iv, pt in zip(self.ivs[:i], param_tuples[:i]):
                        iv_raw[missed_iv] = self._iv_data[pt]
                changed = True
        iv_vals = iv_data if not changed and self.cartesian else iv_raw
        join_type = 'cross' if changed and self.cartesian else 'left'
        iv_df = join_all(iv_vals, how=join_type)
        if changed:
            self._dv_ready = False
            if self.cartesian:
                for iv in iv_df:
                    self._iv_data[(iv,)+param_tuples] = iv_df[iv]
                    self.last_varkeys[iv] = (iv,)+param_tuples
        self._iv_df = iv_df
        self._iv_ready = True
        return iv_df.copy()
    #     """# possible states:
        
    #     C=Cartesian, H=cHanged, R=use iv_Raw, D=use iv_Data, L=Left join, X=cross join
    #     not cartesian (~C,~H)v(~C,H):
    #         all ivs in iv_raw, all same len, left join and return (R,L)
    #     cartesian:
    #         all ivs previously memoised with 'X' (C,~H):
    #             all ivs in iv_data, all same len, left join and return (D,L)
    #         some ivs had to be made fresh (C,H):
    #             all ivs in iv_raw, maybe diff lens, cross join, add to self._iv_data and and return (R,X)
    #     (~C&~K)->X, X<->~L
    #     (~C&K)->D, D<->~R
    #     """


    # def __next__(self) -> pd.DataFrame:
    #     self.last_varkeys = {} # should this be reset like this? is it needed?
    #     iv_data = {}
    #     changed = False
    #     for iv in self.ivs:
    #         param_tuple = (iv, self.obs_len(iv), self._iv_params [iv]) 
    #         if param_tuple not in self._iv_data:
    #             self._dv_ready = False
    #             changed = True
    #             self.last_varkeys[iv] = param_tuple
    #             iv_data[iv] = self.sources[iv](*param_tuple[1:])
    #         else:

    #         # self._iv_data[[col for col in self._iv_data.columns if param_tuple in col]]
    #         iv_out[iv] = self._iv_data[param_tuple].copy()
    #     return iv_out

    @property 
    def target(self) -> pd.DataFrame:
        if not self._dv_ready: 
            ivdf = pd.DataFrame({
                iv: self._iv_data[self.last_varkeys[iv]] for iv in self.ivs
            })
            self.dv_key = tuple(self.last_varkeys.items())
            dv_out = {}
            for name in self.dvs:
                self._dv_data[
                    (name,)+self.dv_key
                ] = dv_out[name] = self.sources[name](**ivdf) 
            self._last_dv_out = pd.DataFrame(dv_out)
            self._dv_ready = True
            # self._dv_data = pd.DataFrame({
            #     name: self.sources[name](**ivdic) for name in self.dvs
            # })
        return self._last_dv_out.copy()
    
    # self._dv_data[[col for col in self._dv_data.columns if self.dv_key in col]].copy()
        

class WorldObservatory:
    def __init__(self, ivs: list[str], dvs: list[str], **kwargs):
        pass

    def __next__(self) -> pd.DataFrame:
        pass

    def target(self) -> pd.DataFrame:
        pass


if __name__ == '__main__':
    import doctest
    def example_sfo():
        def u(obs_len: int, start: int, stop: int, mult: int=1, decl=None):
            if decl:
                print(decl)
            return np.linspace(start, stop, num=obs_len)*mult
        def sumsq(x, y, **kwargs):
            print('called sumsq')
            return x**2 + y**2
        def diffcu(x, y, **kwargs):
            print('called diffcu')
            return x**3 - y**3
        akx = ArgsNKwargs((2.0, 5.0), {'decl': 'gen x'})
        aky = ArgsNKwargs((-2.0, 3.0), {'mult': 2, 'decl': 'gen y'})
        src = {'x': u, 'y': u, 'a': sumsq, 'b': diffcu}
        obl = {'x': 4, 'y': 6}
        prm = {'x': akx, 'y': aky}
        sfo = StaticFunctionObservatory(
            ['x', 'y'], ['a', 'b'], src, obl, prm, cartesian=True
        )
        return sfo
    doctest.testmod(extraglobs={'example_sfo': example_sfo})