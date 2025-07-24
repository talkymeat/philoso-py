import numpy as np
from typing import Any, Protocol, Callable
from math import nextafter
from collections import defaultdict
from icecream import ic
import warnings
import re

# ic.disable()
Float = float|np.float16|np.float32|np.float64

def i(val):
    try:
        return val.item()
    except AttributeError:
        return val

def tanh_arg_extremum(float_class, max=True):
    """If `max==False`, finds the highest value of x in floating point number
    type `float_type` (float, no.float16, np.float32, etc), for which tanh(x) 
    rounds to -1, or if `max==True`, finds the lowest value of x for which 
    tanh(x) rounds to 1.

    >>> for float_type in (np.float16, np.float32, np.float64, float):
    ...     tanh_min = tanh_arg_extremum(float_type, max=False)
    ...     tanh_max = tanh_arg_extremum(float_type, max=True)
    ...     assert np.tanh(tanh_min) == -1.0
    ...     assert np.tanh(tanh_max) == 1.0
    ...     assert np.tanh(nextafter(tanh_min, 0)) != -1.0
    ...     assert np.tanh(nextafter(tanh_max, 0)) != 1.0
    """
    lo, hi, prev, current = np.array([0.0, 1000.0, 0.0, 1000.0], dtype=float_class) - (0 if max else 1000)
    target = 1.0 if max else -1.0
    while prev != current:
        if (np.tanh(current)==target) if max else (np.tanh(current)!=target):
            current, prev, hi = np.mean([current, lo]), current, current
        else:
            current, prev, lo = np.mean([current, hi]), current, current
    while np.tanh(current) != target:
        current = nextafter(current, current+target)
    while np.tanh(nextafter(current, current-target)) == target:
        current = nextafter(current, current-target)
    return current

def exp_arg_extremum(float_class: type, max=True):
    """If `max==False`, finds the highest value of x, given the local standard 
    for floats, for which tanh(x) rounds to -1, or if `max==True`, finds the 
    lowest value of x for which tanh(x) rounds to 1.

    >>> for float_type in (np.float16, np.float32, np.float64, float):
    ...     exp_min = exp_arg_extremum(float_type, max=False)
    ...     exp_max = exp_arg_extremum(float_type, max=True)
    ...     assert np.exp(exp_min) == 0.0, (np.exp(exp_min), exp_min, float_type)
    ...     with warnings.catch_warnings():
    ...         warnings.filterwarnings('ignore')
    ...         assert np.exp(exp_max) == np.inf, np.exp(exp_max)
    ...     assert np.exp(nextafter(exp_min, 0)) != 0.0
    ...     assert np.exp(nextafter(exp_max, 0)) != np.inf
    """
    targ = np.inf if max else 0.0
    flip = -1.0 if max else 1.0
    a = float_class(0.0)
    b = float_class(-1.0 * flip) # (1.0)
    a_s, b_s = [a], [b]
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        while a!=b :
            if flip*np.exp(b) > flip*targ:
                a, b = b, b-(2*flip*abs(a-b)) 
            elif a==a_s[-2] and b==b_s[-2]:
                a_ = np.nextafter(a, 0.0)
                if np.exp(a_) == targ:
                    a = a_ 
                    b_ =  nextafter(b, 0.0)
                    if np.exp(b_) == targ:
                        b = b_
                elif np.exp(a) == targ:
                    b = a
                else:
                    a = b
            elif flip*np.exp(a) > flip*targ:
                qdiff = abs(a-b)/4.0 
                b += flip*qdiff # -=
                a -= flip*qdiff # +=
            else:
                a += flip*abs(a-b)/2 # -=, use abs
            a_s.append(a)
            b_s.append(b)
    return b

class Interval:
    """Defines an open, closed, or semi-open interval and can be used to test
    if a value is inside the interval using the `in` keyword
    
    >>> unit = Interval(0,1)
    >>> testvals = [-2, -1, 0, 0.5, 1, 2, 3]
    >>> [x in unit for x in testvals]
    [False, False, True, True, True, False, False]
    >>> unit.open = True
    >>> [x in unit for x in testvals]
    [False, False, False, True, False, False, False]
    >>> unit.closed_hi = True
    >>> [x in unit for x in testvals]
    [False, False, False, True, True, False, False]
    >>> unit.closed_lo = True
    >>> [x in unit for x in testvals]
    [False, False, True, True, True, False, False]
    >>> unit.min = -1
    >>> [x in unit for x in testvals]
    [False, True, True, True, True, False, False]
    >>> unit.max = 2
    >>> [x in unit for x in testvals]
    [False, True, True, True, True, True, False]
    >>> str_unit0 = Interval('[0, 1]')
    >>> str_unit1 = Interval('(0, 1]')
    >>> str_unit2 = Interval('[0, 1)')
    >>> str_unit3 = Interval('(0, 1)')
    >>> [x in str_unit0 for x in testvals]
    [False, False, True, True, True, False, False]
    >>> [x in str_unit1 for x in testvals]
    [False, False, False, True, True, False, False]
    >>> [x in str_unit2 for x in testvals]
    [False, False, True, True, False, False, False]
    >>> [x in str_unit3 for x in testvals]
    [False, False, False, True, False, False, False]
    """
    def __init__(
            self,
            lo: float|int|str = None,
            hi: float|int = None,
            closed: bool = None,
            closed_lo: bool = None,
            closed_hi: bool = None
        ):
        if isinstance(lo, str):
            paramstring = lo
            for param in [hi, closed, closed_lo, closed_hi]:
                if param is not None:
                    raise ValueError(
                        "Initialising an interval, if the first" +
                        "parameter is a string, no other params" +
                        "should be passed"
                    )
            if m := re.match(r'([(\[]) *([-+]?[.0-9einf]+), *([-+]?[.0-9einf]+) *([)\]])', paramstring):
                # regex does not specify exact float format, using float 
                # conversion to catch malformed strings. However, I'm
                # not `try...except`-ing these str-to-float conversions, 
                # because the ValueError raised if these are invalid is 
                # exactly the exception I would want to raise here anyway. 
                lo = float(m[2])
                hi = float(m[3])
                closed_lo = m[1] == '['
                closed_hi = m[4] == ']'
            else:
                raise ValueError(f'Invalid interval string {paramstring}')
        # ic(repr(lo), 'banana')
        self._min = lo if lo is not None else -np.inf
        self._max = hi if hi is not None else np.inf
        # ic(lo, type(lo), hi, type(hi))
        self._verify_minmax()
        if closed is not None:
            self.closed_lo = self.closed_hi = self.closed
        else:
            self._closed_lo = True if closed_lo is None else closed_lo
            self._closed_hi = True if closed_hi is None else closed_hi
        self._init_bounds()

    def __eq__(self, other: 'Interval')->bool:
        return (
            self.min == other.min and 
            self.max == other.max and
            self.closed_lo == other.closed_lo and
            self.closed_hi == other.closed_hi
        )

    # def __repr__(self):
    #     return f'{self.min} <{"=" if self._closed_lo else ""} x <{"=" if self._closed_hi else ""} {self.max}'

    # def __str__(self):
    #     return self.__repr__()

    def _verify_minmax(self):
        if self.min > self.max:
            raise ValueError(
                f'min ({self.min}) should not be greater than max ({self.max})'
            )
        if (self.min == self.max) and self.open:
            raise ValueError(
                f'min and max ({self.max}) should not be equal in an open interval'
            )

    def _init_bounds(self):
        self._over_min = (lambda v: self.min <= v) if self.closed_lo else (lambda v: self.min < v)
        self._under_max = (lambda v: self.max >= v) if self.closed_hi else (lambda v: self.max > v)

    @property
    def min(self):
        return self._min
    
    @min.setter
    def min(self, lo):
        self._min = lo
        self._verify_minmax()
        self._init_bounds()

    @property
    def max(self):
        return self._max
    
    @max.setter
    def max(self, max):
        self._max = max
        self._verify_minmax()
        self._init_bounds()

    @property
    def closed_lo(self):
        return self._closed_lo
    
    @closed_lo.setter
    def closed_lo(self, closed_lo):
        self._closed_lo = closed_lo
        if closed_lo:
            self._verify_minmax()
        self._init_bounds()

    @property
    def closed_hi(self):
        return self._closed_hi
    
    @closed_hi.setter
    def closed_hi(self, closed_hi):
        self._closed_hi = closed_hi
        if not closed_hi:
            self._verify_minmax()
        self._init_bounds()

    @property
    def closed(self):
        return self._closed_lo and self.closed_hi
    
    @closed.setter
    def closed(self, closed):
        self._closed_lo = self._closed_hi = closed
        if not closed:
            self._verify_minmax()
        self._init_bounds()

    @property
    def open(self):
        return not (self._closed_lo or self.closed_hi)
    
    @open.setter
    def open(self, open):
        self._closed_lo = self._closed_hi = not open
        if open:
            self._verify_minmax()
        self._init_bounds()
        
    def __contains__(self, val):
        return self._over_min(val) and self._under_max(val)
    
    def __rsub__(self, val):
        if val in self:
            return 0.0
        elif not self._under_max(val):
            return val - self.max
        else:
            return val - self.min
        
    def __str__(self):
        return f"{'[' if self.closed_lo else '('}{self.min}, {self.max}{']' if self.closed_hi else ')'}"


TANH_ZONE = Interval(tanh_arg_extremum(np.float32, max=False), tanh_arg_extremum(np.float32))

class Guardrail(Protocol):
    def __call__(self, raw: Float, cooked: Float) -> Float:
        ...

def no_guardrail(self, raw: Float, cooked: Float):
    return cooked

class TanhGuardrail:
    def __init__(self,
            lo: float|int = None,
            hi: float|int = None,
            closed: bool = None,
            closed_lo: bool = True,
            closed_hi: bool = True,
            base_penalty: float = 1.0,
            manager: 'GuardrailManager' = None,
            dtype: type = np.float32
        ) -> None:
        self.interval = Interval(
            lo if lo is not None else -1.0, 
            hi if hi is not None else 1.0, 
            closed=closed, closed_lo=closed_lo, closed_hi=closed_hi
        )
        self.raw_interval = Interval(
            tanh_arg_extremum(dtype, max=False), 
            tanh_arg_extremum(dtype)
        )
        self.gm: GuardrailManager = manager if manager else GuardrailManager()
        self.base_penalty = base_penalty

    def __repr__(self):
        s = f'TanhGuardrail({self.raw_interval}'.replace('x', 'raw')
        s += f', {self.interval})'.replace('x', 'cooked')
        return s

    def __call__(self, raw, cooked) -> float:
        """A guardrail receives both the raw output of the Neural Net (logit)
        and the processed ('cooked') value that will actually be used to set a
        parameter of some behaviour, which may be subject to a maximum or 
        minimum. This guardrail in particular deals with parameters that use 
        `tanh` to transform the NN output. The guardrail automatically 
        penalises any raw value `x` that falls outside the range which I am
        calling the `tanh zone` - the closed interval such that:

        $$ \\left [  \\min\\limits_{x} float(tanh(x))=-1, \\max\\limits_{x} float(tanh(x))=1 \\right ] $$

        -where 'float' is a floor function rounding real values to the 
        possible values of python floats, so that the neural network doesn't
        waste time wandering around the space of outputs that don't change 
        agent behaviour at all.

        It also penalises any value of the 'cooked' value outside of its 
        defined bounds. However, for any penalised behaviour, the penalty is 
        equal to the difference between the actual raw value and its estimate
        for the minimum and maximum raw values that generate 'cooked' values 
        within the permitted bounds. This estimate is initialised `tanh zone`; 
        but the min can be adjusted up and the max down, if a value-pair is 
        received such that the 'cooked' value is out of bounds, but the raw is
        not: in which case the provided 'raw' value is taken as the new raw
        minimum, if it is greater than the existing raw minimum, and it was 
        the 'cooked minimum' that was violated: or it is taken as the new raw
        maximum, if it less than the existing raw maximum, and it is the cooked 
        maximum that was violated.

        Penalties are pased to the agent's `GuardrailManager`, which creates 
        `Guardrails`, references them in a dictionary, and gathers penalties 
        due to guardrail violations. The penalty is the absolute difference
        between the violated boundary and the raw value, plus a base penalty,
        which is 1.0 by default.

        >>> tanh_only = TanhGuardrail(dtype=np.float64)
        >>> tanh_min = tanh_arg_extremum(np.float64, max=False)
        >>> tanh_max = tanh_arg_extremum(np.float64, max=True)
        >>> eta = 0.0000000001
        >>> print(tanh_only.interval)
        [-1.0, 1.0]
        >>> i(tanh_only(tanh_min, -1.0))
        -1.0
        >>> i(tanh_only.gm.reward)
        0.0
        >>> i(tanh_only(tanh_max, 1.0))
        1.0
        >>> i(tanh_only.gm.reward)
        0.0
        >>> i(tanh_only(tanh_min-eta, -1.0))
        -1.0
        >>> i(abs(tanh_only.gm.reward - (-(np.log(1.0+eta)+1.0))) < eta*1e-4)
        True
        >>> i(tanh_only.gm.reward) # cleared
        0.0
        >>> i(tanh_only(tanh_max+eta, 1.0))
        1.0
        >>> i(abs(tanh_only.gm.reward - (-1.0-eta)) < eta*1e-4)
        True
        >>> i(tanh_only.gm.reward) # cleared again
        0.0
        >>> trunc = TanhGuardrail(2.0, 100.0, dtype=np.float64)
        >>> def fit_to_int_range_factory(min, max):
        ...     def fit_to_int_range(x):
        ...         x = np.tanh(x)
        ...         x += 1
        ...         x *= (max-min)/2.0
        ...         x += min
        ...         return np.int32(x)
        ...     return fit_to_int_range
        >>> f = fit_to_int_range_factory(0, 102)
        >>> i(trunc(tanh_min-eta, -1.0))
        2.0
        >>> i(abs(trunc.gm.reward - (-1.0-eta)) < eta*1e-4)
        True
        >>> trunc(-7.0, f(-7.0))
        2.0
        >>> i(trunc.gm.reward)
        -1.0
        >>> trunc(tanh_min, 0.0)
        2.0
        >>> i(abs(trunc.gm.reward - (-(np.log(-tanh_min-6.0)+1.0))) < eta*1e-4)
        True
        >>> trunc(-2.5, f(-2.5))
        2.0
        >>> i(trunc.gm.reward)
        -1.0
        >>> trunc(-7.0, f(-7.0))
        2.0
        >>> i(abs(trunc.gm.reward - -(np.log(5.5)+1.0)) < eta*1e-4)
        True
        >>> trunc(-2.0, f(-2.0))
        2.0
        >>> i(trunc.gm.reward)
        -1.0
        >>> trunc(-2.5, f(-2.5))
        2.0
        >>> i(abs(trunc.gm.reward - -(np.log(1.5)+1.0)) < eta*1e-4)
        True
        >>> i(trunc(-7.0, f(-7.0)))
        2.0
        >>> i(abs(trunc.gm.reward - -(np.log(6.0)+1.0)) < eta*1e-4)
        True
        >>> i(trunc(-1.9, f(-1.9)))
        2
        >>> i(trunc.gm.reward)
        0.0
        >>> i(trunc(-1.0, 12))
        12
        >>> i(trunc.gm.reward)
        0.0
        >>> trunc(2.5, 101.0)
        100.0
        >>> i(trunc.gm.reward)
        -1.0
        >>> i(trunc(19, 102.0))
        100.0
        >>> i(abs(trunc.gm.reward - -(np.log(17.5)+1.0)) < eta*1e-4)
        True
        """
        if raw not in self.raw_interval:
            aberrance = raw - self.raw_interval
            reward = np.log(abs(aberrance) + 1.0) + self.base_penalty
            self.gm._reward -= reward
            return max(self.interval.min, cooked) if aberrance < 0 else min(cooked, self.interval.max)
        if cooked not in self.interval:
            aberrance = cooked - self.interval
            if aberrance < 0:
                self.raw_interval.min = max(self.raw_interval.min, raw)
                reward = np.log(abs(raw - self.raw_interval) + 1.0) + self.base_penalty
                self.gm._reward -= reward #np.log(reward)
                return self.interval.min
            else:
                self.raw_interval.max = min(self.raw_interval.max, raw)
                reward = np.log(abs(raw - self.raw_interval) + 1.0) + self.base_penalty
                # reward = abs(raw - self.raw_interval) + self.base_penalty
                self.gm._reward -= reward #np.log(reward)
                return self.interval.max
        return cooked

class ExponentialGuardrail:

    def __init__(self,
        coefficient: float = 1.0,
        constant: float = 0.0,
        boundary: float|None = None,
        boundary_penalty_func: None|Callable = None,
        base_penalty: float = 1.0,
        manager: 'GuardrailManager' = None,
        dtype: type = np.float32
    ):
        if coefficient == 0:
            raise ValueError(
                'Coefficient cannot be zero: the range of permitted values' +
                ' would be empty'
            )
        self.coeff = coefficient
        self.raw_interval = Interval(
            exp_arg_extremum(dtype, max=False),
            exp_arg_extremum(dtype, max=True)
        ) 
        # np.inf is multiplied by coefficient so that self.boundary is -inf if coefficient
        # is negative; if a finite boundary value is passed, it is not multiplied by 
        # coefficient
        self.boundary = boundary if boundary is not None else np.inf * coefficient 
        self.interval = Interval(
            constant, boundary
        ) if coefficient > 0 else Interval(
            boundary, constant
        )
        self.bpf = boundary_penalty_func if boundary_penalty_func else lambda x: x
        self.dtype = dtype
        self.gm: GuardrailManager = manager if manager else GuardrailManager()
        self.base_penalty = base_penalty

    def __call__(self):
        pass 

    def __repr__(self):
        pass 

class GuardrailManager(defaultdict):
    gr_kinds = {
        'tanh': TanhGuardrail,
        'exp': ExponentialGuardrail
    }

    def make(
            self,
            name, 
            kind: str = 'tanh',
            base_penalty: float = None,
            **kwargs
        ):
        if '_no_make' in kwargs or kind=='null':
            return
        if kind not in self.gr_kinds:
            raise ValueError(f'Invalid guardrail type, : {kind}')
        GR = self.gr_kinds[kind]
        gr = GR(
            manager=self,
            base_penalty = base_penalty if base_penalty is not None else self.base_penalty,
            **kwargs
        )
        self[name] = gr
        return gr

#     def make(
#             self,
#             name, 
#             min: float|int = None,
#             max: float|int = None,
#             closed: bool = None,
#             closed_lo: bool = True,
#             closed_hi: bool = True,
#             base_penalty: float = None,
#             **kwargs
#         ):
#         if '_no_make' in kwargs:
#             return
#         gr = TanhGuardrail(
#             min=min,
#             max=max,
#             closed=closed,
#             closed_lo=closed_lo,
#             closed_hi=closed_hi,
#             manager=self,
#             base_penalty = base_penalty if base_penalty is not None else self.base_penalty
#         )
#         self[name] = gr
#         return gr

    @property
    def reward(self):
        rew, self._reward = self._reward, 0.0
        return rew

    def __init__(self, *args, base_penalty=1.0, **kwargs) -> None:
        self._reward = 0.0
        self.base_penalty = base_penalty
        return super().__init__(lambda: no_guardrail, *args, **kwargs)
    
    def process_logits(self, logits, keys:str) -> np.ndarray:
        if len(logits)==len(keys):
            return {k: self[k](logit) for k, logit in zip(keys, logits)}




def main():
    import doctest
    doctest.testmod()


if __name__ == '__main__':
    main()

