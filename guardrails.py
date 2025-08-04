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

class FuncArgExtremum(Protocol):
    def __call__(float_class: type[Float], max_:bool=True) -> Float:
        ...

class PenaltyFunc(Protocol):
    def __call__(aberrance: Float) -> Float:
        ...

def tanh_arg_extremum(float_class: type[Float], max_: bool=True) -> Float:
    """If `max_==False`, finds the highest value of x in floating point number
    type `float_type` (float, no.float16, np.float32, etc), for which tanh(x) 
    rounds to -1, or if `max_==True`, finds the lowest value of x for which 
    tanh(x) rounds to 1.

    >>> for float_type in (np.float16, np.float32, np.float64, float):
    ...     tanh_min = tanh_arg_extremum(float_type, max_=False)
    ...     tanh_max = tanh_arg_extremum(float_type, max_=True)
    ...     assert np.tanh(tanh_min) == -1.0
    ...     assert np.tanh(tanh_max) == 1.0
    ...     assert np.tanh(nextafter(tanh_min, 0)) != -1.0
    ...     assert np.tanh(nextafter(tanh_max, 0)) != 1.0
    """
    lo, hi, prev, current = np.array([0.0, 1000.0, 0.0, 1000.0], dtype=float_class) - (0 if max_ else 1000)
    target = 1.0 if max_ else -1.0
    while prev != current:
        if (np.tanh(current)==target) if max_ else (np.tanh(current)!=target):
            current, prev, hi = np.mean([current, lo]), current, current
        else:
            current, prev, lo = np.mean([current, hi]), current, current
    while np.tanh(current) != target:
        current = nextafter(current, current+target)
    while np.tanh(nextafter(current, current-target)) == target:
        current = nextafter(current, current-target)
    return current

def exp_arg_extremum(float_class: type[Float], max_:bool=True) -> Float:
    """If `max_==False`, finds the highest value of x, given the local standard 
    for floats, for which tanh(x) rounds to -1, or if `max_==True`, finds the 
    lowest value of x for which tanh(x) rounds to 1.

    >>> for float_type in (np.float16, np.float32, np.float64, float):
    ...     exp_min = exp_arg_extremum(float_type, max_=False)
    ...     exp_max = exp_arg_extremum(float_type, max_=True)
    ...     assert np.exp(exp_min) == 0.0, (np.exp(exp_min), exp_min, float_type)
    ...     with warnings.catch_warnings():
    ...         warnings.filterwarnings('ignore')
    ...         assert np.exp(exp_max) == np.inf, np.exp(exp_max)
    ...     assert np.exp(nextafter(exp_min, 0)) != 0.0
    ...     assert np.exp(nextafter(exp_max, 0)) != np.inf
    """
    targ = np.inf if max_ else 0.0
    flip = -1.0 if max_ else 1.0
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

def linear_arg_extremum(float_class: type[Float], max_:bool=True) -> Float:
    return float_class('inf') * (max_-0.5)

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
    >>> unit4_0 = Interval(0, 1)
    >>> unit4_0
    Interval('[0, 1]')
    >>> unit4_1 = Interval(0, 1, True)
    >>> unit4_1
    Interval('[0, 1]')
    >>> unit4_2 = Interval(0, 1, True, True)
    >>> unit4_2
    Interval('[0, 1]')
    >>> unit5 = Interval(0, 1, False, True)
    >>> unit5
    Interval('(0, 1]')
    >>> unit6 = Interval(0, 1, True, False)
    >>> unit6
    Interval('[0, 1)')
    >>> unit7_0 = Interval(0, 1, False)
    >>> unit7_0
    Interval('(0, 1)')
    >>> unit7_1 = Interval(0, 1, False)
    >>> unit7_1
    Interval('(0, 1)')


    # >>> [x in unit4_0 for x in testvals]
    # [False, False, True, True, True, False, False]
    # >>> [x in unit4_1 for x in testvals]
    # [False, False, True, True, True, False, False]
    # >>> [x in unit4_2 for x in testvals]
    # [False, False, True, True, True, False, False]
    # >>> [x in unit5 for x in testvals]
    # [False, False, False, True, True, False, False]
    # >>> [x in unit6 for x in testvals]
    # [False, False, True, True, False, False, False]
    # >>> [x in unit7_0 for x in testvals]
    # [False, False, False, True, False, False, False]
    # >>> [x in unit7_1 for x in testvals]
    # [False, False, False, True, False, False, False]
    """
    def __init__(
            self,
            lo: float|int|str = None,
            hi: float|int = None,
            closed: bool = None,
            closed_hi: bool = None
        ):
        if isinstance(lo, str):
            paramstring = lo
            for param in [hi, closed, closed_hi]:
                if param is not None:
                    raise ValueError(
                        "Initialising an interval, if the first" +
                        "parameter is a string, no other params" +
                        "should be passed"
                    )
            if m := re.match(r'([(\[]) *([-+]?[.0-9einf\-]+), *([-+]?[.0-9einf\-]+) *([)\]])', paramstring):
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
        else:
            closed_lo = True if closed is None else closed
            closed_hi = closed_lo if closed_hi is None else closed_hi
        self._min = lo if lo is not None else -np.inf
        self._max = hi if hi is not None else np.inf
        self._verify_minmax()
        self._closed_lo = closed_lo
        self._closed_hi = closed_hi
        self._init_bounds()

    def __eq__(self, other: 'Interval')->bool:
        return (
            self.min == other.min and 
            self.max == other.max and
            self.closed_lo == other.closed_lo and
            self.closed_hi == other.closed_hi
        )

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

    @property
    def min_clipped(self):
        return self._min if self.closed_lo else np.nextafter(self._min, self._max)
    
    @min.setter
    def min(self, lo):
        self._min = lo
        self._verify_minmax()
        self._init_bounds()

    @property
    def max(self):
        return self._max

    @property
    def max_clipped(self):
        return self._max if self.closed_hi else np.nextafter(self._max, self._min)
    
    @max.setter
    def max(self, max_):
        self._max = max_
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
        return self.closed_lo and self.closed_hi
    
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

    def __repr__(self):
        return f"Interval('{str(self)}')"


TANH_ZONE = Interval(tanh_arg_extremum(np.float32, max_=False), tanh_arg_extremum(np.float32))

class Guardrail(Protocol):
    def __call__(self, raw: Float, cooked: Float) -> Float:
        ...

def no_guardrail(self, raw: Float, cooked: Float):
    return cooked

class FuncGuardrail:
    FUNC_NAME = "Func" # XXX probably not needed - remove?
    SYMMETRIC = True
    def func_arg_extremum(self, float_class: type[Float], max_:bool=True) -> Float:
        return linear_arg_extremum(float_class, max_)

    # np.log(abs(raw - self.raw_interval) + 1.0) + self.base_penalty
    def _penalty_func_lo(self, abs_aberrance: Float) -> Float:
        return np.log(1.0+abs_aberrance)

    def _penalty_func_hi(self, abs_aberrance: Float) -> Float:
        return self._penalty_func_lo(abs_aberrance)

    def penalise(self, aberrance: Float):
        penalty_func = self._penalty_func_lo if aberrance < 0.0 else self._penalty_func_hi
        penalty = penalty_func(abs(aberrance)) + self.base_penalty
        self.gm._reward -= penalty #np.log(reward)

    def __init__(self,
            lo: float|int = None,
            hi: float|int = None,
            closed: bool = None,
            closed_hi: bool = None,
            base_penalty: float = 1.0,
            manager: 'GuardrailManager' = None,
            dtype: type = np.float32,
            reversed_: bool = None,
            **kwargs
        ) -> None:
        closed = True if closed is None else closed
        closed_hi = closed if closed_hi is None else closed_hi
        self.interval = Interval(
            lo if lo is not None else -1.0, 
            hi if hi is not None else 1.0, 
            closed=closed, closed_hi=closed_hi
        )
        raw_lo = self.func_arg_extremum(dtype, max_=False)
        raw_hi = self.func_arg_extremum(dtype)
        # `True and None` is None, which would break the conditionals below, so, `reversed_ is True`
        # is used below to handle the case where reversed_ is None; this converts `None` to `False`,
        # so the conjunction is never `None`, but always `True` or `False` 
        flip_me = not self.SYMMETRIC and reversed_ is True 
        if not [closed, closed_hi][flip_me]: # this takes advantage of the int-castability of booleans
            raw_lo = np.nextafter(raw_lo, raw_hi) # because if SYMMETRIC is False and reversed_ is True
        if not [closed_hi, closed][flip_me]: # that changes which of closed and closed_hi is relevant to
            raw_hi = np.nextafter(raw_hi, raw_lo) # the truncation of which raw value
        self.raw_interval = Interval(raw_lo, raw_hi)
        self.gm: GuardrailManager = manager if manager else GuardrailManager()
        self.base_penalty = base_penalty
        # the reversed_ argument only makes sense for non-symmetric underlying functions:
        # therefore, if it is passed and SYMMETRIC is True, raise the same error as if
        # there were no such argument
        if reversed_ is not None and self.SYMMETRIC:
            raise TypeError(
                f"{self.__class__.__name__}.__init__() got an unexpected keyword argument 'reversed_'"
            )
        self.reversed = False if reversed_ is None else reversed_ 

    def __repr__(self):
        return f'{self.__class__.__name__}({self.raw_interval}, {self.interval})'

    def __call__(self, raw, cooked) -> float:
        """A guardrail receives both the raw output of the Neural Net (logit)
        and the processed ('cooked') value that will actually be used to set a
        parameter of some behaviour, which may be subject to a maximum or 
        minimum. This guardrail is a base template for others, but child 
        Guardrail classes exist for parameters that use `tanh` and `exp`
        to transform the NN output. For example, the TanhGuardrail automatically 
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
        """
        if raw not in self.raw_interval:
            aberrance = raw - self.raw_interval
            self.penalise(aberrance)
            return max(self.interval.min_clipped, cooked) if (aberrance < 0) != self.reversed else min(cooked, self.interval.max_clipped)
        if cooked not in self.interval:
            aberrance = raw - self.raw_interval
            self.penalise(aberrance)
            rev_if_rev = lambda x: x
            if not self.SYMMETRIC and self.reversed:
                rev_if_rev = lambda x: not x
            if rev_if_rev(cooked - self.interval < 0):
                self.raw_interval.min = max(self.raw_interval.min, raw)
            else:
                self.raw_interval.max = min(self.raw_interval.max, raw)
            if cooked - self.interval < 0 or cooked == -np.inf:
                return self.interval.min
            else:
                return self.interval.max
        return cooked

class TanhGuardrail(FuncGuardrail):
    FUNC_NAME = "Tanh"

    def func_arg_extremum(self, float_class: type[Float], max_:bool=True) -> Float:
        return tanh_arg_extremum(float_class, max_)


class ExponentialGuardrail(FuncGuardrail):
    FUNC_NAME = "Exponential"
    SYMMETRIC = False

    def func_arg_extremum(self, float_class: type[Float], max_:bool=True) -> Float:
        return exp_arg_extremum(float_class, max_)

    def __init__(self,
            lo: float|int = 0.0,
            hi: float|int = np.inf,
            closed: bool = None,
            closed_hi: bool = None,
            base_penalty: float = 1.0,
            manager: 'GuardrailManager' = None,
            dtype: type[Float] = np.float32,
            reversed_: bool = False,
            **kwargs
        ) -> None:
        """The parameters passed to initialise an exponential guardrail behave slightly
        differently; an exponentially transformed NN output may be translated along the 
        number line by the addition of a constant, or scaled by a coefficient (which
        may be positive or negative), but whatever the values of the constant or the
        coefficient, the transformed value will *in theory* asymptote towards the 
        constant as the raw value becomes arbitrarily negative, in practice becoming 
        equal to the constant when the difference becomes less than the minimum 
        difference expressible for the relevant Float dtype; and the transformed value
        in theory tends to an arbitrarily high absolute value as the raw value becomes
        arbitrarily large, but in practice will eventually overflow the Float dtype
        and become either np.inf or (if the coefficient is negative) -np.inf. This 
        implies certain validations:

        1)  `lo` can be `-inf`, and 'hi` can be `inf`, but one of them _must_ be finite
        2)  If `lo` is `-inf`, `reversed_` must be `True`, and if `hi` is `inf`, `reversed`
            must be `False`
        3)  If `lo` or `hi` is +/- `inf`, the corresponding `closed_lo|hi` parameter must
            be `False`, so that infinite `cooked` values should be rejected, returning
            the minimum or maximum values of the relevant Float type

        The (1) and (2) are enforced with ValueErrors; violations of (3) will cause the
        relevant value to be coerced to correctness, and a warning will be raised

        The `reversed_` param is also needed because the range of valid `raw` values is
        not symmetrical; the highest value at which `np.exp(raw)` becomes rounded to 0.0
        is not equal to zero minus the lowest value at which `np.exp(raw)` overflows to 
        `inf`.
        """
        if np.isinf(lo) and np.isinf(hi):
            raise ValueError(
                'lo and hi cannot both be infinite for an ExponentialGuardrail'
            )
        elif np.isinf(hi):
            if reversed_:
                raise ValueError(
                    'hi cannot be infinite for an ExponentialGuardrail if reversed_ is True'
                )
            if closed_hi or closed_hi is None:
                warn_ = closed is not None and not closed
                # This covers both the case where closed_hi is True, and where it is None:
                # however, it is not necessary to issue a warning if closed_hi is None and
                # closed is False, as in these cases, closed_hi would be given 
                # the correct value of False anyway 
                closed_hi = False
                if warn_:
                    warnings.warn(
                        'If an ExponentialGuardrail has an infinite upper bound, (`hi=np.inf`), ' +
                        '`closed_hi` must also `False`, so that `cooked` values may not ' + 
                        'overflow the Float dtype in use'
                    )
        elif np.isinf(lo):
            if not reversed_:
                raise ValueError(
                    'lo cannot be infinite for an ExponentialGuardrail if reversed_ is False'
                )
            if closed or closed is None:
                # `closed` is used to set the closure of the `hi` and `lo` bounds if no value
                # is provided for `closed_hi` (which may happen if these arguments are passed
                # positionally, and only three arguments are passed), and is only used to set
                # the `closed_lo` param if a value is provided for `closed_hi`. The default for
                # `closed` is `True`, so if `closed is `True` or `None`, it must be coerced to 
                # `False`; however, in these cases, if `closed_hi` is None, `closed_hi`
                # was intended to be `True`, and should not be coerced to be `False` simply 
                # simply because `closed` is being coerced 
                if closed_hi is None:
                    closed_hi = True
                closed = False
                warnings.warn(
                    'If a reversed ExponentialGuardrail has an infinite lower bound, ' +
                    '(`lo=-np.inf`), `closed_lo` must also be `False`, so that `cooked` values' +
                    ' may not overflow the Float dtype in use'
                )
        super().__init__(
            lo = lo,
            hi = hi,
            closed = closed,
            closed_hi = closed_hi,
            base_penalty = base_penalty,
            manager = manager,
            dtype = dtype,
            reversed_ = reversed_
        )

    # def __init__(self,
    #     coefficient: float = 1.0,
    #     constant: float = 0.0,
    #     boundary: float|None = None,
    #     boundary_penalty_func: None|Callable = None,
    #     base_penalty: float = 1.0,
    #     manager: 'GuardrailManager' = None,
    #     dtype: type = np.float32
    # ):
    #     if coefficient == 0:
    #         raise ValueError(
    #             'Coefficient cannot be zero: the range of permitted values' +
    #             ' would be empty'
    #         )
    #     self.coeff = coefficient
    #     self.raw_interval = Interval(
    #         exp_arg_extremum(dtype, max=False),
    #         exp_arg_extremum(dtype, max=True)
    #     ) 
    #     # np.inf is multiplied by coefficient so that self.boundary is -inf if coefficient
    #     # is negative; if a finite boundary value is passed, it is not multiplied by 
    #     # coefficient
    #     self.boundary = boundary if boundary is not None else np.inf * coefficient 
    #     self.interval = Interval(
    #         constant, boundary
    #     ) if coefficient > 0 else Interval(
    #         boundary, constant
    #     )
    #     self.bpf = boundary_penalty_func if boundary_penalty_func else lambda x: x
    #     self.dtype = dtype
    #     self.gm: GuardrailManager = manager if manager else GuardrailManager()
    #     self.base_penalty = base_penalty

    # def __call__(self):
    #     pass 

    # def __repr__(self):
    #     pass 

class GuardrailManager(defaultdict):
    gr_kinds = {
        'tanh': TanhGuardrail,
        'exp': ExponentialGuardrail
    }

    def make(
            self,
            name, 
            func: str = 'tanh',
            base_penalty: float = None,
            **kwargs
        ):
        if '_no_make' in kwargs or func=='null':
            return
        if func not in self.gr_kinds:
            raise ValueError(f'Invalid guardrail type, : {func}')
        self[name] = self.gr_kinds[func](
            manager=self,
            base_penalty = base_penalty if base_penalty is not None else self.base_penalty,
            **kwargs
        )
        return self[name]

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

