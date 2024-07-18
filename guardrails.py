import numpy as np
from typing import Any
from math import nextafter


def tanh_arg_extremum(max=True):
    """If `max==False`, finds the highest value of x, given the local standard 
    for floats, for which tanh(x) rounds to -1, or if `max==True`, finds the 
    lowest value of x for which tanh(x) rounds to 1.

    >>> tanh_min = tanh_arg_extremum(max=False)
    >>> tanh_max = tanh_arg_extremum(max=True)
    >>> np.tanh(tanh_min)
    -1.0
    >>> np.tanh(tanh_max)
    1.0
    >>> np.tanh(nextafter(tanh_min, 0)) == -1.0
    False
    >>> np.tanh(nextafter(tanh_max, 0)) == 1.0
    False
    """
    lo, hi, prev, current = np.array([0.0, 1000.0, 0.0, 1000.0]) - (0 if max else 1000)
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
    """
    def __init__(
            self,
            min: float|int = None,
            max: float|int = None,
            closed: bool = None,
            closed_lo: bool = True,
            closed_hi: bool = True
        ):
        self._min = min if min is not None else -np.inf
        self._max = max if max is not None else np.inf
        self._verify_minmax()
        if closed is not None:
            self.closed_lo = self.closed_hi = self.closed
        else:
            self._closed_lo = closed_lo
            self._closed_hi = closed_hi
        self._init_bounds()

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
        self._over_min = (lambda v: self._min <= v) if self._closed_lo else (lambda v: self._min < v)
        self._under_max = (lambda v: self._max >= v) if self._closed_hi else (lambda v: self._max > v)

    @property
    def min(self):
        return self._min
    
    @min.setter
    def min(self, min):
        self._min = min
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
            return val - self._max
        else:
            return val - self._min
        
    def __str__(self):
        return f"{'[' if self.closed_lo else '('}{self.min}, {self.max}{']' if self.closed_hi else ')'}"


TANH_ZONE = Interval(tanh_arg_extremum(max=False), tanh_arg_extremum())

class NoGuardrail:
    def __call__(self, raw, cooked):
        return cooked

class TanhGuardrail:
    def __init__(self,
            min: float|int = None,
            max: float|int = None,
            closed: bool = None,
            closed_lo: bool = True,
            closed_hi: bool = True,
            base_penalty: float = 1.0,
            manager: 'GuardrailManager' = None
        ) -> None:
        self.interval = Interval(
            min if min is not None else -1.0, 
            max if max is not None else 1.0, 
            closed=closed, closed_lo=closed_lo, closed_hi=closed_hi
        )
        self.raw_interval = Interval(
            tanh_arg_extremum(max=False), 
            tanh_arg_extremum()
        )
        self.gm: GuardrailManager = manager if manager else GuardrailManager()
        self.base_penalty = base_penalty

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

        >>> tanh_only = TanhGuardrail()
        >>> tanh_min = tanh_arg_extremum(max=False)
        >>> tanh_max = tanh_arg_extremum(max=True)
        >>> eta = 0.0000000001
        >>> print(tanh_only.interval)
        [-1.0, 1.0]
        >>> tanh_only(tanh_min, -1.0)
        -1.0
        >>> tanh_only.gm.reward
        0.0
        >>> tanh_only(tanh_max, 1.0)
        1.0
        >>> tanh_only.gm.reward
        0.0
        >>> tanh_only(tanh_min-eta, -1.0)
        -1.0
        >>> abs(tanh_only.gm.reward - (-1.0-eta)) < eta*1e-4
        True
        >>> tanh_only.gm.reward # cleared
        0.0
        >>> tanh_only(tanh_max+eta, 1.0)
        1.0
        >>> abs(tanh_only.gm.reward - (-1.0-eta)) < eta*1e-4
        True
        >>> tanh_only.gm.reward # cleared again
        0.0
        >>> trunc = TanhGuardrail(2.0, 100.0)
        >>> def fit_to_int_range_factory(min, max):
        ...     def fit_to_int_range(x):
        ...         x = np.tanh(x)
        ...         x += 1
        ...         x *= (max-min)/2.0
        ...         x += min
        ...         return np.int32(x)
        ...     return fit_to_int_range
        >>> f = fit_to_int_range_factory(0, 102)
        >>> trunc(tanh_min-eta, -1.0)
        2.0
        >>> abs(trunc.gm.reward - (-1.0-eta)) < eta*1e-4
        True
        >>> trunc(-7.0, f(-7.0))
        2.0
        >>> trunc.gm.reward
        -1.0
        >>> trunc(tanh_min, 0.0)
        2.0
        >>> abs(trunc.gm.reward - (tanh_min+6.0)) < eta*1e-4
        True
        >>> trunc(-2.5, f(-2.5))
        2.0
        >>> trunc.gm.reward
        -1.0
        >>> trunc(-7.0, f(-7.0))
        2.0
        >>> trunc.gm.reward
        -5.5
        >>> trunc(-2.0, f(-2.0))
        2.0
        >>> trunc.gm.reward
        -1.0
        >>> trunc(-2.5, f(-2.5))
        2.0
        >>> trunc.gm.reward
        -1.5
        >>> trunc(-7.0, f(-7.0))
        2.0
        >>> trunc.gm.reward
        -6.0
        >>> trunc(-1.9, f(-1.9))
        2
        >>> trunc.gm.reward
        0.0
        >>> trunc(-1.0, 12)
        12
        >>> trunc.gm.reward
        0.0
        >>> trunc(2.5, 101.0)
        100.0
        >>> trunc.gm.reward
        -1.0
        >>> trunc(19, 102.0)
        100.0
        >>> trunc.gm.reward
        -17.5
        """
        if raw not in TANH_ZONE:
            aberrance = raw - self.raw_interval
            reward = abs(aberrance) + self.base_penalty
            self.gm._reward -= np.log(reward)
            return max(self.interval.min, cooked) if aberrance < 0 else min(cooked, self.interval.max)
        if cooked not in self.interval:
            aberrance = cooked - self.interval
            if aberrance < 0:
                self.raw_interval.min = max(self.raw_interval.min, raw)
                reward = abs(raw - self.raw_interval) + self.base_penalty
                self.gm._reward -= np.log(reward)
                return self.interval.min
            else:
                self.raw_interval.max = min(self.raw_interval.max, raw)
                reward = abs(raw - self.raw_interval) + self.base_penalty
                self.gm._reward -= np.log(reward)
                return self.interval.max
        return cooked
            
class GuardrailManager(dict):
    def make(
            self,
            name, 
            min: float|int = None,
            max: float|int = None,
            closed: bool = None,
            closed_lo: bool = True,
            closed_hi: bool = True,
            base_penalty: float = None
        ):
        gr = TanhGuardrail(
            min=min,
            max=max,
            closed=closed,
            closed_lo=closed_lo,
            closed_hi=closed_hi,
            manager=self,
            base_penalty = base_penalty if base_penalty is not None else self.base_penalty
        )
        self[name] = gr
        return gr

    @property
    def reward(self):
        rew, self._reward = self._reward, 0.0
        return rew

    def __init__(self, *args, base_penalty=1.0, **kwargs) -> None:
        self._reward = 0.0
        self.base_penalty = base_penalty
        self['_'] = NoGuardrail()
        return super().__init__(*args, **kwargs)
    
    def process_logits(self, logits, keys:str) -> np.ndarray:
        if len(logits)==len(keys):
            return {k: self[k](logit) for k, logit in zip(keys, logits)}




def main():
    import doctest
    doctest.testmod()


if __name__ == '__main__':
    main()

