from abc import ABC, abstractmethod
from collections import deque
from typing import List, Callable, Union

#import matplotlib.pyplot as plt
#import numpy as np
import random, itertools, math

from utils import list_transpose, linear_interpolated_binned_means


class World(ABC):
    """Abstract Base Class defining the 'world' that agents observe and reason
    about. An epist.py `World` can be any collection of state which can be
    updated in time-steps according to deterministic or probabilistic rules and
    observed by agents.
    """

    @abstractmethod
    def observe(self, **kwargs) -> Union[dict, tuple, str, int, float]:
        """The function `Agent`s call to make observations of the `World`. This
        may be a single simple function, or may route queries to multiple
        different observation functions with different logic. Note that
        observations need not be, and ideally (for most philosophical questions
        you might want an epist.py model to address) should not be, direct
        access to the underlying state of the `World` which the update rules
        have access to.

        kwargs should be used to define the control agents have over the
        fine/coarse grainedness of observations, but observations should be
        limited in how many bytes a single observation can return.

        Returned values should be in lists or tuples - A singleton tuple may be
        returned, though.
        """
        ...

    @abstractmethod
    def increment_time(self):
        """Calculates the state of the `World` at the next time-step and updates
        accordingly.
        """
        ...

    @property
    def max_observation_size(self) -> int:
        """Agents are limited as to the maximum size of a single observation.
        This limit is represented as an int, but how this is interpreted
        depends on the subclass: bytes, length of tuple, length of str, etc"""
        return self._max_observation_size

    @max_observation_size.setter
    def max_observation_size(self, size: int):
        """Setter for max_observation_size."""
        # if size%12 != 0:
        #     raise ValueError()
        self._max_observation_size = size


class VectorWorld(World):
    """The simplest World subclass: the world is a linked list of numbers,
    which at each time-step adds a new number calculated from the existing
    members of the list to the head of the list, and if the length has reached
    the maximum defined in `__init__`, drops the tail item."""
    def __init__(
                self,
                length: int,
                initial_conditions: list,
                fn_list: List[Callable],
                max_observation_size: int,
                cell_type=float,
                transposal_probability: float = 0.0,
                jitter_stdev: float = 0.0
            ):
        """Sets up the world with initial parameter values.

        :param int length: The maximum length of the vector (actually a deque: a
            double-ended linked list) that makes up the world.
        :param list initial_conditions: The first len(initial_conditions) values
            in `_world` at time step 0
        :param function fn_list: The list of functions which calculates the next
            new value in `_world` from the existing values. The 'laws of
            physics' of `_world` are the function-composition of the functions
            in `fn_list`
        :param int max_observation_size: The maximum size of a single
            observation by an `Agent`. An observation is a list of values of the
            same type as `cell_type`, and `max_observation_size` is interpreted
            as the maximum length of this list.
        :param type cell_type: By default, `_world` is a deque of floats:
            however, it is possible to specify a different type for the elements
            in `world`.
        :param float transposal_probability: The probability p for any item in
            the range of an observation that it will be switched with one of its
            neighbours. More exactly, each item has a probability p^d of being
            switched with a neighbour at least distance d away. Note that this
            has the consequence that the effect of noise is reduced the more
            coarse-grained the observation, as observations wider than
            `max_observation_size` return the linearly interpolated means of
            bins of cells, and the larger the bin the lower the probability that
            any given transposal will switch cells in different bins.
        :param float jitter_stdev: If `cell_type==float` and `jitter_stdev>0.0`,
            gaussian noise with `sigma=jitter_stdev` will be applied to the
            cells in the range of an observation. Note that this means that
            the more coarse-grained an observation, the less it will be affected
            by noise, as observations wider than `max_observation_size` return
            the linearly interpolated means of bins of cells, and the larger the
            bin the more the jitter on each cell in the bin will be cancelled
            out by the jitter on other cells in the same bin.
        :raises TypeError: if `initial_conditions` contains values of a
            different type than `cell_type`.
        :raises ValueError: if `initial_conditions` is longer than the `length`
            of the world, if `max_observation_size` is less than 1 (`Agents`
            should be able to make observations), `jitter_stdev` is negative
            (standard deviations must be positive), or `transposal_probability`
            is not in the interval [0.0, 1.0) (must be a valid probability,
            cannot be 1.0 as this would result in infinite transposal distances)
        """
        self.length = length
        if len(initial_conditions) > length:
            raise ValueError(
                'The initial conditions of your World vector must fit inside ' +
                'your maximum World vector length. Your initial considitions ' +
                f'contained {len(initial_conditions)} elements, but your ' +
                f'maximum World vector length is {length}.'
            )
        for i, cell in enumerate(initial_conditions):
            if type(cell) != cell_type:
                raise TypeError(
                    'You specified that your World vector must consist of ' +
                    f'elements of type {cell_type}, but element {i} of ' +
                    f'initial_conditions, {repr(cell)}, was of type ' +
                    f'{type(cell)}.'
                )
        if max_observation_size < 1:
            raise ValueError(
                'max_observation_size must be greater than zero: you gave a ' +
                f'max_observation_size of {max_observation_size}.'
            )
        if jitter_stdev < 0.0:
            raise ValueError(
                'jitter_stdev cannot be negative: you gave a jitter_stdev of ' +
                f'{jitter_stdev}.'
            )
        if transposal_probability < 0.0 || transposal_probability > 1.0:
            raise ValueError(
                'transposal_probability is a probability, and so cannot be ' +
                f'{"negative" if transposal_probability < 0.0 else "greater than one"}' +
                f': you gave a transposal_probability of {transposal_probability}.'
            )
        if transposal_probability == 1.0:
            raise ValueError(
                'Each item in the range of an observation has a probability of'+
                ' transposal_probability^d of being switched in the '+
                'observation with a neighbour at least distance d away. '+
                'Therefore, transposal_probability cannot be 1.0, as all ' +
                'values sampled for an observation would be transposed with ' +
                'the cell +inf or -inf cells away with probability 1.0.'
            )
        self._world = deque(
            initial_conditions + [cell_type(0)]*(length-len(initial_conditions)),
            maxlen=length
        )
        self.fn_list = fn_list
        self.cell_type = cell_type
        self.max_observation_size = max_observation_size
        self.transposal_probability = transposal_probability
        self.jitter_stdev = jitter_stdev

    @property
    def fn_list(self):
        """Getter for fn_list"""
        return self._fn_list

    @fn_list.setter
    def fn_list(self, fn_list):
        """Setter for fn_list"""
        self._fn_list = fn_list

    def next_cell(self):
        """Calculates the value for the new cell at the next time-step, by
        applying the function composition `fn o ... o fn` of all functions in
        `fn_list` to `_world`
        """
        val=self._world
        for fn in self.fn_list:
            val = fn(val)
        return self.cell_type(val)

    def increment_time(self):
        """Appends the value of `next_cell()` to the start of `_world`, and
        deletes the last value from the end."""
        self._world.appendLeft(self.next_cell())

    @property
    def world(self):
        """Getter for `world`, the list that defines the `VectorWorld`."""
        return self._world

    def _make_transpose_map(self, length):
        """Helper function for `observe`. The interact through which `Agents`
        observe the `World` is noisy and imprecise

        The transpose_map defines the set of value-switches that must be
        performed while computing the observation thus, if `transpose_map[i]`
        equals k, the i^th item in the observation sample (i+start^th item
        in the world) will be transposed with the i+k^th item in the
        observation sample (i+k+start^th item in world)

        :param int length: The length of the transpose_map, which should be
            equal to the length of the observation in `observe`
        :return list[int] transpose_map: list giving the direction and magnitude
            of transposals

        >>> # can we initialise a world with out it fucking up?
        >>> world = VectorWorld(
        ...     length=20,
        ...     initial_conditions = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        ...     fn_list = [lambda x: 2*x],
        ...     max_observation_size = 20,
        ...     transposal_probability = 0.2,
        ...     jitter_stdev = 0.1
        ... )
        >>> # Is there a bias to the left or right
        >>> ozymandias_sum_of_sums = 0
        >>> tm_size, n = 20, 100000
        >>> for i in range(n): # doctest: +SKIP
        ...     ozymandias_sum_of_sums += sum(world._make_transpose_map(tm_size))
        ...
        >>> print(f"Mean = {abs(ozymandias_sum_of_sums/(tm_size*n)):.2f}")
        Mean = 0.00
        >>> # Test that the distribution of 1's, 2's & 3's matches expectations
        >>> # given `transposal_probability`
        >>> tm_size = 20
        >>> for n, t in [(100000, 0.5), (200000, 0.2), (2000000, 0.1)]: # doctest: +SKIP
        ...     # Initialise a World. All bollocks but the transposal_probability.
        ...     world = VectorWorld(
        ...         length=20,
        ...         initial_conditions = [0.0, 1.0, 2.0, 3.0, 4.0],
        ...         fn_list = [lambda x: 2*x],
        ...         max_observation_size = 20,
        ...         transposal_probability = t,
        ...         jitter_stdev = 0.1
        ...     )
        ...     # counters for the number of |1|'s, |2|'s, & |3|'s in the
        ...     # transpose_maps
        ...     ones = 0
        ...     twos = 0
        ...     threes = 0
        ...     for i in range(n):
        ...         tm = world._make_transpose_map(tm_size)
        ...         for tr in tm:
        ...             ones += (abs(tr)>0)
        ...             twos += (abs(tr)>1)
        ...             threes += (abs(tr)>2)
        ...     print(f"{(ones/(tm_size*n))/(t**1):.2f}")
        ...     print(f"{(twos/(tm_size*n))/(t**2):.2f}")
        ...     print(f"{(threes/(tm_size*n))/(t**3):.1f}")
        1.00
        1.00
        1.0
        1.00
        1.00
        1.0
        1.00
        1.00
        1.0
        """
        # prepopulate transpose_map with zeroes
        transpose_map = [0]*(length)
        for i in range(length):
            # For each value of `transpose_map`, calculate the magnitude of the
            # transposal, |d|. If t is `self.transposal_probability` and r is a
            # random float sampled over the closed interval [0, 1], then:
            # |d| = argmax x: r < t^x
            rnd = random.random()
            while rnd < self.transposal_probability**(transpose_map[i]+1):
                # x is initialised to 0, this loop check if t^(x+1) would be >r:
                # if so, increment x: when the loop stops running, |d| = x
                transpose_map[i] += 1
            # Flip a coin to see if the transposal is to the left or right
            transpose_map[i] *= random.randint(0,1) * 2 - 1
        return transpose_map

    def _shave_transpose_map(self, transpose_map: List[int], offset: int) -> None:
        """This is intended to deal with an edge case that will only happen
        extremely rarely, unless self.transposal_probability is set very high.
        If a value of transpose_map requires that a sample value be transposed
        with a value that is outwith the sample bounts, that's totally fine:
        that's why the initial sample copied from `_world` is taken with a bit
        of padding, if needed. If, on the other hand, the transpose distance
        goes outside of the bounds of `_world` too, we need some sensible
        behaviour for that case. Thus, the transposal will 'bounce' off the
        boundary: thus, if the transposal target is `self._world[-2]`, it will
        be changed to `self._world[2]`; if it is
        `self._world[len(self._world)+2]`, it will be changed to
        `self._world[len(self._world)-2]`. This itself is not the
        rare-for-sensible-values-of-self.transposal_probability edge case: the
        edge case is where the transposal length is so large, it needs to bounce
        a second time off the other edge of the world. One of you code gremlins
        is bound to try it, I just know it. For this reason, I'm using this
        function to convert all values in transpose_map to their eventual
        transposal distances, after all bouncing has happened. The values are
        changed in place.

        :param list[int] transpose_map: The transpose map.
        :param int offset: The start position of the transpose_map in `_world`

        >>> world = VectorWorld(
        ...     length=10,
        ...     initial_conditions = [
        ...         0.0, 1.0, 2.0, 3.0, 42.0, 5.0, 6.0, 7.0, 8.0, 9.0
        ...     ],
        ...     fn_list = [lambda x: 2*x],
        ...     max_observation_size = 10,
        ...     transposal_probability = 0.2,
        ...     jitter_stdev = 0.1
        ... )
        >>> for sig in [1,-1]:
        ...     for x in range(32):
        ...         tm = [-1,0,sig*x,0,2,0]
        ...         world._shave_transpose_map(tm, 2)
        ...         print(tm)
        ...     if sig==1:
        ...         print("Second verse, like the first but reversed")
        [-1, 0, 0, 0, 2, 0]
        [-1, 0, 1, 0, 2, 0]
        [-1, 0, 2, 0, 2, 0]
        [-1, 0, 3, 0, 2, 0]
        [-1, 0, 4, 0, 2, 0]
        [-1, 0, 5, 0, 2, 0]
        [-1, 0, 4, 0, 2, 0]
        [-1, 0, 3, 0, 2, 0]
        [-1, 0, 2, 0, 2, 0]
        [-1, 0, 1, 0, 2, 0]
        [-1, 0, 0, 0, 2, 0]
        [-1, 0, -1, 0, 2, 0]
        [-1, 0, -2, 0, 2, 0]
        [-1, 0, -3, 0, 2, 0]
        [-1, 0, -4, 0, 2, 0]
        [-1, 0, -3, 0, 2, 0]
        [-1, 0, -2, 0, 2, 0]
        [-1, 0, -1, 0, 2, 0]
        [-1, 0, 0, 0, 2, 0]
        [-1, 0, 1, 0, 2, 0]
        [-1, 0, 2, 0, 2, 0]
        [-1, 0, 3, 0, 2, 0]
        [-1, 0, 4, 0, 2, 0]
        [-1, 0, 5, 0, 2, 0]
        [-1, 0, 4, 0, 2, 0]
        [-1, 0, 3, 0, 2, 0]
        [-1, 0, 2, 0, 2, 0]
        [-1, 0, 1, 0, 2, 0]
        [-1, 0, 0, 0, 2, 0]
        [-1, 0, -1, 0, 2, 0]
        [-1, 0, -2, 0, 2, 0]
        [-1, 0, -3, 0, 2, 0]
        Second verse, like the first but reversed
        [-1, 0, 0, 0, 2, 0]
        [-1, 0, -1, 0, 2, 0]
        [-1, 0, -2, 0, 2, 0]
        [-1, 0, -3, 0, 2, 0]
        [-1, 0, -4, 0, 2, 0]
        [-1, 0, -3, 0, 2, 0]
        [-1, 0, -2, 0, 2, 0]
        [-1, 0, -1, 0, 2, 0]
        [-1, 0, 0, 0, 2, 0]
        [-1, 0, 1, 0, 2, 0]
        [-1, 0, 2, 0, 2, 0]
        [-1, 0, 3, 0, 2, 0]
        [-1, 0, 4, 0, 2, 0]
        [-1, 0, 5, 0, 2, 0]
        [-1, 0, 4, 0, 2, 0]
        [-1, 0, 3, 0, 2, 0]
        [-1, 0, 2, 0, 2, 0]
        [-1, 0, 1, 0, 2, 0]
        [-1, 0, 0, 0, 2, 0]
        [-1, 0, -1, 0, 2, 0]
        [-1, 0, -2, 0, 2, 0]
        [-1, 0, -3, 0, 2, 0]
        [-1, 0, -4, 0, 2, 0]
        [-1, 0, -3, 0, 2, 0]
        [-1, 0, -2, 0, 2, 0]
        [-1, 0, -1, 0, 2, 0]
        [-1, 0, 0, 0, 2, 0]
        [-1, 0, 1, 0, 2, 0]
        [-1, 0, 2, 0, 2, 0]
        [-1, 0, 3, 0, 2, 0]
        [-1, 0, 4, 0, 2, 0]
        [-1, 0, 5, 0, 2, 0]
        """
        for i, tr in enumerate(transpose_map):
            # Start with the notional transposal target index
            target = offset+i+tr
            # for all positive values of `target`, `target mod 2*len(_world)` is
            # exactly equivalent to `target`: for negative values,
            # `-(abs(target) mod 2*len(_world))`
            target = math.copysign(abs(target)%(2*(len(self._world)-1)), target)
            while target<0 or target>=len(self._world):
                #print(f"a: {target}")
                if target<0:
                    target = abs(target)
                    #print(f"b: {target}")
                if target>=len(self._world):
                    target = 2*(len(self._world)-1)-(target)
                    #print(f"c: {target}")
            transpose_map[i] = int(target-(i+offset))

    def _calculate_padding(self, transpose_map: List[int]) -> List[int]:
        """ Helper function for `observe`.

        `padding` defines the extent to which the transposals require sampling
        across a broader range of values in `world` than `world[start:end]`
        the zeroth value should be non-positive and the first should be non-
        negative: thus the sample drawn will actually be
        `self.world[start+padding[0]:end+padding[1]]`

        :param list[int] transpose_map: A list of length equal to the
            observation range given by the `Agent`, containing random integer
            values defining the transposals that will be applied to the
            observation
        :return list[int] padding: A list of length two, consisting of a
            non-positive number inticating how far the transposals overspill the
            observation range to the left, and a non-negative number indicating
            how far to the right

        >>> world = VectorWorld(
        ...     length=20,
        ...     initial_conditions = [0.0, 1.0, 2.0, 3.0, 4.0 ],
        ...     fn_list = [lambda x: 2*x],
        ...     max_observation_size = 20,
        ...     transposal_probability = 0.2,
        ...     jitter_stdev = 0.1
        ... )
        >>> world._calculate_padding([0,0,1,-1,0,-2,-2])
        [0, 0]
        >>> world._calculate_padding([2,0,1,-1,0,-2,-2])
        [0, 0]
        >>> world._calculate_padding([-1,-3,1,-1,0,-2,-2])
        [-2, 0]
        >>> world._calculate_padding([0,0,1,-1,0,2,-2])
        [0, 1]
        >>> world._calculate_padding([0,0,1,-1,0,4,-2])
        [0, 3]
        >>> world._calculate_padding([0,-2,1,-1,0,2,-2])
        [-1, 1]
        """
        padding = [0, 0]
        for i, tr in enumerate(transpose_map):
            padding[0] = min(tr+i, padding[0])
            padding[1] = max(tr+i-(len(transpose_map)-1), padding[1])
        return padding

    def _get_initial_observation_sample(self, start, end, padding) -> list:
        """Extracts the range of values from which the observation sample is to
        be derived, copying values from `world` without applying any noise to
        the data: transposal and jitter will be subsequently applied to the
        sample, by other methods

        :param int start: The start of the observation range supplied by the
            `Agent` making the observation (inclusive).
        :param int end: The end of the observation range supplied by the
            `Agent` making the observation (exclusive).
        :param list[int] padding: A list of length two, consisting of a
            non-positive number inticating how far the transposals overspill the
            observation range to the left, and a non-negative number indicating
            how far to the right

        >>> world = VectorWorld(
        ...     length=10,
        ...     initial_conditions = [
        ...         0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0
        ...     ],
        ...     fn_list = [lambda x: 2*x],
        ...     max_observation_size = 10,
        ...     transposal_probability = 0.2,
        ...     jitter_stdev = 0.1
        ... )
        >>> for x in range(7):
        ...     for y in range(7):
        ...         print(world._get_initial_observation_sample(3, 7, [-x, y]))
        ...
        [3.0, 4.0, 5.0, 6.0]
        [3.0, 4.0, 5.0, 6.0, 7.0]
        [3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        [2.0, 3.0, 4.0, 5.0, 6.0]
        [2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        """
        # Pull a sample from `world` covering the range to be sampled
        # from, consisting of the range asked for by the Agent, with padding on
        # either side if the transposal map requires it. However, in case the
        # padding required by the padding parameter overspills the beginning or
        # end of the world, the sample will only go as far as the bounds of the
        # `world`: cast this to `list`
        return list(
            itertools.islice(
                self._world,
                max(start+padding[0],0),
                min(end+padding[1], len(self._world))
            )
        )

    def _apply_transpose_map(self, sample: list, transpose_map: List[int], offset: int) -> None:
        """Iterates through transpose_map, wherever a non-zero value `tr` is
        found, switching the corresponding value in `sample` (at index `i`) with
        the value at index `i+tr` (left when `tr<0`, right when `tr>0`).

        :param list sample: list copied from the section of `_world` to be
            sampled from to make the observation. This includes the range
            specified by the `Agent`'s call on `world.observe`, but also any
            padding around that required for any transposals out of the
            observation range.
        :param list[int] transpose_map: list describing what transposals need to
            happen where. Each item in `transpose_map` corresponds to an item in
            `sample`, but because of padding, it may be shifted a bit to the
            right. The `offset` param is a non-positive number that can be
            subtracted from an index in `transpose_map` to get the corresponding
            index in `sample`. Where `tr` in `transpose_map` at index `i` is
            non-zero, `sample[i+offset]` will be switched with
            `sample[i+offset+tr]`
        :param int offset: A non-negative number that can be added from any
            index in `transpose_map` to get the corresponding index in `sample`.

        >>> world = VectorWorld(
        ...     length=10,
        ...     initial_conditions = [
        ...         0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0
        ...     ],
        ...     fn_list = [lambda x: 2*x],
        ...     max_observation_size = 10,
        ...     transposal_probability = 0.2,
        ...     jitter_stdev = 0.1
        ... )
        >>> sample = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        >>> tm = [-1,0,0,0,0,1]
        >>> world._apply_transpose_map(sample, tm, 1)
        >>> print(sample)
        [2.0, 1.0, 3.0, 4.0, 5.0, 6.0, 8.0, 7.0]
        >>> sample = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        >>> tm = [-1,0,-2,0,0,1]
        >>> world._apply_transpose_map(sample, tm, 1)
        >>> print(sample)
        [2.0, 4.0, 3.0, 1.0, 5.0, 6.0, 8.0, 7.0]
        """
        for i, tr in enumerate(transpose_map):
            if tr:
                list_transpose(sample, i+offset, i+offset+tr)

    def _unpad_sample(self, observation_sample: list, padding: List[int]) -> None:
        """If the sample was initially padded to allow transposals of values
        with targets outside of the requested sample range, this removes that
        padding. The list is altered in place

        :param list observation_sample: The sample that the `Agent`'s
            observation is to be drawn from, maybe with some padding at the ends
        :param list[int] padding: A list of two ints, the first non-positive,
            the second non-negative, describing how much padding was added to
            the start and end of the sample respectively

        >>> world = VectorWorld(
        ...     length=10,
        ...     initial_conditions = [
        ...         0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0
        ...     ],
        ...     fn_list = [lambda x: 2*x],
        ...     max_observation_size = 10,
        ...     transposal_probability = 0.2,
        ...     jitter_stdev = 0.1
        ... )
        >>> sample = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        >>> world._unpad_sample(sample, [0, 0])
        >>> print(sample)
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        >>> sample = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        >>> world._unpad_sample(sample, [-1, 0])
        >>> print(sample)
        [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        >>> sample = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        >>> world._unpad_sample(sample, [-1, 1])
        >>> print(sample)
        [2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        >>> sample = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        >>> world._unpad_sample(sample, [-2, 1])
        >>> print(sample)
        [3.0, 4.0, 5.0, 6.0, 7.0]
        >>> sample = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        >>> world._unpad_sample(sample, [-2, 2])
        >>> print(sample)
        [3.0, 4.0, 5.0, 6.0]
        >>> def test_pad_unpad(l_padding, r_padding):
        ...     w_vec = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        ...     w = VectorWorld(
        ...         length=10,
        ...         initial_conditions = w_vec,
        ...         fn_list = [lambda x: 2*x],
        ...         max_observation_size = 10,
        ...         transposal_probability = 0.2,
        ...         jitter_stdev = 0.1
        ...     )
        ...     s,e = 3,7
        ...     raw = list(w_vec[s:e])
        ...     transpose_map = [l_padding,0,0,r_padding]
        ...     w._shave_transpose_map(transpose_map, s)
        ...     padding = w._calculate_padding(transpose_map)
        ...     sample = w._get_initial_observation_sample(s, e, padding)
        ...     w._unpad_sample(sample, padding)
        ...     return raw==sample
        ...
        >>> failures = 0
        >>> for l in range(40):
        ...     for r in range(40):
        ...         failures += not test_pad_unpad(-l, r)
        ...
        >>> print(failures)
        0
        """
        if padding[0]:
            observation_sample[:-padding[0]] = []
        if padding[1]:
            observation_sample[len(observation_sample)-padding[1]:] = []

    def _apply_jitter(self, observation_sample):
        """Apply gaussian noise to each value in the sample, with
        stdev = self.jitter_stdev

        :param list observation_sample: The sample from which the `Agent`'s
            observation is to be derived. At this point, it should have had
            transposal applied and any padding removed

        >>> from scipy import stats
        >>> import numpy as np
        >>> def test_jitter(mu, sd):
        ...     world = VectorWorld(
        ...         length=10000,
        ...         initial_conditions = [0.0],
        ...         fn_list = [lambda x: 2*x],
        ...         max_observation_size = 10000,
        ...         transposal_probability = 0.2,
        ...         jitter_stdev = sd
        ...     )
        ...     sample = [mu]*1000000
        ...     s = np.array(world._apply_jitter(sample))
        ...     k2, p = stats.normaltest(s)
        ...     if p < 0.05 or abs((sum(s)/len(s))-mu)>=0.02 or abs(np.std(s)-sd)>=0.02:
        ...         print(
        ...             f"mu={mu}, sd={sd}, p={p}{' X' if p < 0.05 else ''}, " +
        ...             f"mu'={sum(s)/len(s)}" +
        ...             f"{' X' if abs((sum(s)/len(s))-mu)>=0.02 else ''}, " +
        ...             f"sd={np.std(s)}" +
        ...             f"{' X' if abs(np.std(s)-sd)>=0.02 else ''}"
        ...         )
        ...     return p >= 0.05 and abs((sum(s)/len(s))-mu)<0.02 and abs(np.std(s)-sd)<0.02
        ...
        >>> failures = 0
        >>> for mu in (-10.0, -1.0, 0.0, 1.0, 10.0):
        ...     for sd in (0.1, 0.2, 0.5, 1.0, 2.0):
        ...         failures += not test_jitter(mu, sd)
        ...
        >>> print(failures)
        0
        """
        return [random.gauss(samp, self.jitter_stdev) for samp in observation_sample]

    def _reduce_sample(self, observation_sample):
        """Takes a sample generated from `_world` and, if it is longer than the
        `max_observation_size`, divides it into `max_observation_size` many
        bins, takes the mean of each bin, and returns the list of means (in the
        same order as the original sample) as the observation that will be
        returned to the `Agent`. If the sample size does not evenly divide
        `max_observation_size`, linear interpolation will be used. If the
        `observation_sample` length is in fact less than or equal to
        `max_observation_size`, `observation_sample` will be returned unaltered.

        :param list observation_sample: The observation sample generated from
            world. At this point, the sample should have had any noise required
            added already.
        :return list: The unaltered `observation_sample` if it fits within
            `max_observation_size`, or else the sample reduced to the means of
            linearly interpolated bins

        >>> world = VectorWorld(
        ...     length=10,
        ...     initial_conditions = [
        ...         0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0
        ...     ],
        ...     fn_list = [lambda x: 2*x],
        ...     max_observation_size = 3,
        ...     transposal_probability = 0.2,
        ...     jitter_stdev = 0.1
        ... )
        >>> world._reduce_sample([1,2,3,4])
        [1.25, 2.5, 3.75]
        >>> world._reduce_sample([1,2,3])
        [1, 2, 3]
        """
        if len(observation_sample) <= self.max_observation_size:
            return observation_sample
        return linear_interpolated_binned_means(
            observation_sample,
            self.max_observation_size
        )


    def observe(self, start: int, end: int):
        """Make an observation covering a certain range of the world-vector,
        subject to a certain probability of elements in the observation being
        transposed, and a certain amout of gaussian jitter on each value.
        If the length of the observed range exceeds the `max_observation_size`,
        The noisified range will by default be divided into equal-sized
        subranges and averaged, using linear interpolation if
        `max_observation_size%observation_range.len != 0`

        :param int start: Index of start of observation range, inclusive.
        :param int end: Index of end of observation range, exclusive.
        """
        # Check that `start` and `end` are in the right range, raise ValueErrors
        # otherwise
        if start >= end:
            raise ValueError("End must be greater than start")
        if start < 0:
            raise ValueError("Start cannot be negative")
        if end > maxlen:
            raise ValueError(
                "Out of range: the maximum length of this world is" +
                f" {self.maxlen}"
            )
        # TODO: performance testing: is it better to copy out the initial_sample,
        # then transpose, or do both at once in a list comprehension?
        padding = []
        if self.transposal_probability:
            # The transpose_map defines the set of value-switches that must be
            # performed while computing the observation thus, if `transpose_map[i]`
            # equals k, the i^th item in the observation sample (i+start^th item
            # in the world) will be transposed with the i+k^th item in the
            # observation sample (i+k+start^th item in world)
            transpose_map = self._make_transpose_map(end-start)
            # Adjusts transposals in case they have to bounce off the edge of the
            # world. This is better explained in the `_shave_transpose_map`
            # docstring
            self._shave_transpose_map(transpose_map, start)
            # `padding` defines the extent to which the transposals require sampling
            # across a broader range of values in `world` than `world[start:end]`
            # the zeroth value should be non-positive and the first should be non-
            # negative: thus the sample drawn will actually be
            # `world[start+padding[0]:end+padding[1]]`
            padding = self._calculate_padding(transpose_map)
        # Copy data across to a fresh list: this should include padding at the
        # beginning or end or both, if there are transposals in the
        # transpose_map that reach out of the sample range
        observation_sample = self._get_initial_observation_sample(
            start, end, padding if padding else [0, 0]
        )
        if self.transposal_probability:
            # Apply transposals. In all the following, the sample is altered in
            # place.
            self._apply_transpose_map(observation_sample, transpose_map, -padding[0])
            # If padding was applied, remove it
            self._unpad_sample(observation_sample, padding)
        # Apply gaussian noise to each value in the sample, with
        # stdev = self.jitter_stdev
        if self.cell_type==float and self.jitter_stdev:
            observation_sample = self._apply_jitter(observation_sample)
        # Congratulations, you have a sample of the `_world` fucked up enough to
        # allow `Agents` to see it.
        # In case the sample is longer than `max_observation_size`, reduce it
        # to the linearly-interpolated means of `max_observation_size` many
        # bins, and return it; otherwise return it without further alteration.
        return _reduce_sample(observation_sample)

    # Could functools simplify this?
    @staticmethod
    def get_twin_inverted_decaying_sum_world():
        """Returns a sample VectorWorld"""
        def decaying_sum(ls, exponent, start_neg, is_alternating):
            return sum([(x/(exponent**i))*(1-((2*((i+start_neg)%2))*is_alternating)) for x, i in zip(list(ls), range(len(list(ls))))])

        fn_list = [
            lambda ls: (decaying_sum(ls, 2, True, True), decaying_sum(ls, 3, False, True)),
            lambda x: 1/x[0] + 1/x[1]
        ]
        world = VectorWorld(
            length=1000,
            initial_conditions = [1.0],
            fn_list = fn_list,
            max_observation_size = 24,
            transposal_probability = 0.2,
            jitter_stdev = 0.1
        )
        return world

def main():
    import doctest
    print(
    """Note that VectorWorld contains some doctests designed to check the
    distribution of certain random events, by repeatedly sampling them and
    taking the mean. This means that, unless these have the SKIP flag on:

    a: it takes hella long
    b: there is a small but non-zero change that the test will fail, even
    if the code works correctly
    c: These errors will look like (e.g.) Expected: 1.00 Got 0.99 or 1.01
    d: the number of samples is a trade off between 'a' and 'b'""")
    doctest.testmod()


if __name__ == '__main__':
    main()
