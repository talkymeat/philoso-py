from abc import ABC, abstractmethod
from math import log, floor, pi, cos, sin
from random import choice, random, uniform, gauss
from typing import Callable, Any, TypeVar
from functools import reduce

from icecream import ic
import tensorflow as tf

from logtools import MiniLog
from size_depth import SizeDepth 

T = TypeVar('T')

def _complement(ls, x):
    if x in ls:
        i = ls.index(x)
        return ls[:i] + ls[i+1:]
    else:
        return ls

class Mutator(ABC):
    """The abstract base class for a collection of classes for mutation
    operators for constants in Genetic Programming. Mutation works a bit
    differently depending on the type of the constant being mutated, so Mutator
    has subclasses for a standard set of types - `int`, `float`, `complex`, and
    `bool`, which can be created through the factory class MutatorFactory,
    which can be extended without modification to create custom Mutator classes
    too.

    The first round of testing will be done in a Jupyter Notebook, as it
    requires graphing distributions and kinda eyeballing them.
    """

    # XXX TODO remove mutation_sd from this
    def __init__(self, mutation_rate: float, mutation_sd=1.0, **kwargs):
        """Partially implemented initialiser in the abstract base class: all
        Mutators have a `mutation_rate` which must be initialised as a
        probability

        Parameters
        ----------
            mutation_rate : float
                A probability in the closed interval [0.0, 1.0]. How this is
                used in computing mutations depends on the Mutator subclass
            mutation_sd : float
                Standard deviation of mutations, used in some Mutators (those
                that mutate continuous valued types, specifically)
            kwargs:
                Not used in the four standard Mutators, but included for forward
                compatibility

        Raises
        ------
            ValueError:
                If mutation_rate is not a valid probability, in the closed
                interval [0.0, 1.0]
        """
        if 0.0 <= mutation_rate <= 1.0:
            self.mutation_rate = mutation_rate
        else:
            raise ValueError(
                "mutation_rate is a probability, and must not be less than " +
                "0.0 or more than 1.0"
            )

    @abstractmethod
    def __call__(self, val: T) -> T:
        """Takes the original value of a constant and applies the mutation
        operator to it. A magic method that makes a mutator Callable. Abstract.

        Parameters
        ----------
            val:
                The value of the constant to be mutated
        """
        pass

class NonMutator(Mutator):
    def __init__(self, **kwargs):
        """Placeholder mutator that doesn't mutate anything. Basically an identity 
        function.
        """
        pass

    def __call__(self, val: T) -> T:
        return val

class IntMutator(Mutator):
    """Mutates `ints` and integer-like classes. Inherits __init__ from parent
    class, as mutation_rate is the only parameter it needs.

    Parameters
    ----------
        mutation_rate : float
            A probability in the closed interval [0.0, 1.0]. See __call__ for
            how this is used.
    """

    def __call__(self, val: int) -> int:
        """Mutates the constant. The constant is changed by at least 1 with a
        probability of `self.mutation_rate`, by at least 2 with a probability of
        `self.mutation_rate^2`, at least 3 with a probability of
        `self.mutation_rate^3`, and at least n with a probability of
        `self.mutation_rate^n`.

        Parameters
        ----------
            val : int
                The value of the constant to be mutated

        Returns
        -------
            int:
                The mutated value (may be unchanged, with probability of
                `1-mutation_rate`)
        """
        # random value in the semi-open interval (0.0, 1.0].
        # Note the default behaviour of tf.random.uniform is to use the interval
        # [0.0, 1.0) - thus the multiplication by -1 and addition of 1. 2float32 
        # randval = random()
        randval = (tf.random.uniform(val.shape) * -1.0) + 1.0
        # If randval is more than the mutation rate, just return val. The 'else'
        # condition below also would return exactly `val` in this case, but this
        # case is handled separately to save computation
        if tf.math.reduce_all(randval > self.mutation_rate):
            return val
        # If randval <= self.mutation_rate, the mutation is computed as follows:
        # The magnitude of the mutation is computed as the log of randval
        # (base = self.mutation_rate), rounded down with the floor function.
        # The constant is changed by at least 1 with a probability of 
        # `self.mutation_rate`, by at least 2 with a probability of 
        # `self.mutation_rate^2`, at least 3 with a probability of 
        # `self.mutation_rate^3`, and at least n with a probability of 
        # `self.mutation_rate^n`. This  is multiplied by `choice((-1,1))` which 
        # determines the direction of the mutation, increasing or decreasing 
        # with equal probability. The magnitude of the change is calculated HERE:
        randval = tf.math.floordiv(
            tf.math.log(randval), 
            tf.math.log(self.mutation_rate)
        )
        # Since the floor function takes it to integer values only, it is now
        # safe to case randval to the same dtype as val
        randval = tf.cast(randval, val.dtype)
        # gets a tensor of randomly distributed 1's and -1's:
        # `categorical` gets a 2-d tensor of 0's and 1's. For some reason, it only
        #     outputs 2-d and demands logits
        # `size` gets the number of values in val, so the output of `categorical`
        #     can be of shape (1, size). That ensures the correct number of 1|-1 
        #     values...
        # `reshape` then squishes the result into the same shape as val
        rotations = tf.reshape(
            (tf.random.categorical(
                tf.math.log([[0.5, 0.5]]), 
                tf.size(val),
                dtype=val.dtype
            ) * 2) - 1, 
            val.shape
        )
        # val, randval, and rotations are now all same shape and dtype
        # multiply randval by rotations to randomly flip the signs
        randval = randval * rotations
        # Then add to val to get the mutated values - mostly 0's if mutation_rate
        # is low
        val += randval
        return val

class UintMutator(IntMutator):
    pass

class FloatMutator(Mutator):
    def __init__(self, mutation_rate: float, mutation_sd: float==1.0, **kwargs):
        """Initialises mutation rate and the standard deviation of (gaussian)
        mutations. Details of how these are used in __call__.

        Parameters
        ----------
            mutation_rate : float
                The frequency with which non-zero mutation occurs. A probability
                in the closed interval [0.0, 1.0]. A certainty by default.
            mutation_sd: float
                The standard deviation of the normally distributed mutation
                delta. A standard normal distibution by default.
        """
        self.mutation_sd = mutation_sd
        super().__init__(mutation_rate)

    def __call__(self, val: float) -> float:
        """If no value is given for mutation_rate, it is assumed that values
        always mutate, albeit less than `mutation_sd` ~85% of the time. For
        values less than 1.0 (1.0-mutation_rate)*100% of the time, no mutation
        will occur, and the rest of the time, a gaussian random value with a
        mean of 0 and sd of mutation_sd will be added to the input value to give
        the mutated value.

        Parameters
        ----------
            val : float
                The value of the constant to be mutated

        Returns
        -------
            float:
                The mutated value (may be unchanged, with probability of
                `1.0-mutation_rate`)
        """
        if int(self.mutation_rate) or random() <= self.mutation_rate:
            return gauss(val, self.mutation_sd)
        else:
            return val

class ComplexMutator(Mutator):
    def __init__(self, mutation_rate: float, mutation_sd: float==1.0, **kwargs):
        """Initialises mutation rate and the standard deviation of (gaussian)
        mutations. Details of how these are used in __call__.

        Parameters
        ----------
            mutation_rate : float
                The frequency with which non-zero mutation occurs. A probability
                in the closed interval [0.0, 1.0]. A certainty by default.
            mutation_sd: float
                The standard deviation of the normally distributed modulus of
                the  complex mutation delta. A standard normal distibution by
                default.
        """
        self.mutation_sd = mutation_sd
        super().__init__(mutation_rate)

    def __call__(self, val: complex) -> complex:
        """If no value is given for mutation_rate, it is assumed that values
        always mutate, albeit less than `mutation_sd` ~85% of the time. For
        values less than 1.0 (1.0-mutation_rate)*100% of the time, no mutation
        will occur, and the rest of the time, a random complex delta is added,
        with uniformly distributed argument and normally distributed modulus.

        Parameters
        ----------
            val : complex
                The value of the constant to be mutated

        Returns
        -------
            complex:
                The mutated value (may be unchanged, with probability of
                `1.0-mutation_rate`)

        """
        if int(self.mutation_rate) or random() <= self.mutation_rate:
            modulus_delta = gauss(0, self.mutation_sd)
            argument_delta = uniform(0.0, pi)
            delta = modulus_delta * complex(cos(argument_delta), sin(argument_delta))
            return val+delta
        else:
            return val

class BoolMutator(Mutator):
    """Mutates `bools`. Inherits __init__ from parent class, as mutation_rate is
    the only parameter it needs.

    Parameters
    ----------
        mutation_rate : float
            A probability in the closed interval [0.0, 1.0]. See __call__ for
            how this is used.
    """

    def __call__(self, val: bool) -> bool:
            """Flips the `bool`, with a probability of `self.mutation rate`

            Parameters
            ----------
                val : bool
                    The value of the constant to be mutated

            Returns
            -------
                bool:
                    The mutated value (may be unchanged, with probability of
                    `1.0-mutation_rate`)
            """
            return val != (random() <= self.mutation_rate)


# XXX Make same-attr mutators singletons!!!!!!
class MutatorFactory:
    """Factory class for mutation operators for constants in Genetic
    Programming. Mutation works a bit differently depending on the type of the
    constant being mutated, so Mutator has subclasses for a standard set of
    types - `int`, `float`, `complex`, and `bool`, and Mutator itself acts as a
    factory class for its subclasses, with a `dict`, `types` specifying which
    subclass should be instantiated given the type of the constant.

    If you want to implement GP with a different set of types, you will need to
    add entries to the `types`. If you wish to use a type `Spider`, you
    should create a subclass of `Mutator`, `SpiderMutator`, and add it as
    follows:

    `Mutator.type_dict[Spider] = SpiderMutator`

    Likewise, if you want to replace the mutator for a standard type with
    something else with custom behaviour:

    `Mutator.type_dict[int] = MyIntMutator`

    Some classes might be able to be used with existing Mutators: for example
    IntMutator would work fine with BigInteger (I presume. Haven't tested it
    yet).

    The first round of testing will be done in a Jupyter Notebook, as it
    requires graphing distributions and kinda eyeballing them.
    """
    class MutatorTest:
        def __init__(self, mutator: Mutator, test: Callable):
            self.mutator = mutator
            self.test = test
        
        def __call__(self, mutable: Any, nexts: list['MutatorFactory.MutatorTest']=None):
            return self.mutator if self.test(mutable) else nexts[0](mutable, nexts[1:]) if nexts else None
        
        def __str__(self):
            return f"{self.mutator.__class__.__name__}Test"

    type_tests = [
        MutatorTest(UintMutator, lambda x: x.dtype.is_integer and x.dtype.is_unsigned),
        MutatorTest(IntMutator, lambda x: x.dtype.is_integer),
        MutatorTest(FloatMutator, lambda x: x.dtype.is_floating),
        MutatorTest(ComplexMutator, lambda x: x.dtype.is_complex),
        MutatorTest(BoolMutator, lambda x: x.dtype.is_bool)
    ]

    @classmethod
    def reorder_tests(cls, new_order: list[int], verbose=True):
        if len(new_order) == len(cls.type_tests) and sorted(new_order) == list(range(len(cls.type_tests))):
            new_t_t = [None] * len(cls.type_tests)
            for pos, test in zip(new_order, cls.type_tests):
                new_t_t[pos] = test
            cls.type_tests = new_t_t
            if verbose:
                print("New MutatorFactory.type_tests ordering is:")
                cls.print_tests()
        else:
            raise ValueError(
                f"{new_order} is not a valid reordering of {len(cls.type_tests)} elements:" +
                " MutatorFactory.reorder_tests takes a list of unique ints from 0 " +
                "(inclusive) to the length of MutatorFactory.type_tests (exclusive), " +
                f"which is currently {len(cls.type_tests)}"
            )
        
    @classmethod
    def print_tests(cls):
        for test in cls.type_tests:
            print(test)

    @classmethod
    def add_mutator(cls, mutator: Mutator, test: Callable, pos: int=-1):
        if -1 <= pos and pos <= len(cls.type_tests):
            mt = cls.MutatorTest(mutator, test)
            pos = len(cls.type_tests) if pos == -1 else pos
            cls.type_tests = cls.type_tests[0:pos] + [mt] + cls.type_tests[pos:len(cls.type_tests)]
        else:
            raise IndexError(
                f"Only values between -1 and len(MutatorFactory.type_tests) inclusive" +
                " should be passed to MutatorFactory.add_mutator as 'pos'. " +
                "len(MutatorFactory.type_tests) is currently " +
                f"{len(MutatorFactory.type_tests)}. You passed {pos}."
            )


    def __new__(cls, const: tf.Tensor, mutation_rate: float, mutation_sd: float=None, **kwargs):
        """Initialising a Mutator always results in one of the concrete
        subclasses. The argument `const` should be the type of the constant
        that needs mutating, and the Mutator returned will be the subclass
        registered in the `types` dictionary for that type.

        Parameters
        ----------
            const : Any
                The type of the constant to be mutated
            mutation_rate : float
                A probability in the closed interval [0.0, 1.0]. How this is
                used in computing mutations depends on the Mutator subclass
            mutation_sd : float
                Standard deviation of mutations, used in some Mutators (those
                that mutate continuous valued types, specifically)
            kwargs:
                Not used in the four standard Mutators, but included for forward
                compatibility

        Raises
        ------
            AttributeError:
                If no Mutator subclass is defined for the provided `const`.

        >>> print(type(MutatorFactory(tf.constant(1, dtype=tf.int8), 0.25)).__name__)
        IntMutator
        >>> print(type(MutatorFactory(tf.constant(1, dtype=tf.int16), 0.25)).__name__)
        IntMutator
        >>> print(type(MutatorFactory(tf.constant(1, dtype=tf.int32), 0.25)).__name__)
        IntMutator
        >>> print(type(MutatorFactory(tf.constant(1, dtype=tf.int64), 0.25)).__name__)
        IntMutator
        >>> print(type(MutatorFactory(tf.constant(1, dtype=tf.uint8), 0.25)).__name__)
        UintMutator
        >>> print(type(MutatorFactory(tf.constant(1, dtype=tf.uint16), 0.25)).__name__)
        UintMutator
        >>> print(type(MutatorFactory(tf.constant(1, dtype=tf.uint32), 0.25)).__name__)
        UintMutator
        >>> print(type(MutatorFactory(tf.constant(1, dtype=tf.uint64), 0.25)).__name__)
        UintMutator
        >>> print(type(MutatorFactory(tf.constant(1.0, dtype=tf.float32), 0.25)).__name__)
        FloatMutator
        >>> print(type(MutatorFactory(tf.constant(1.0, dtype=tf.float64), 0.25)).__name__)
        FloatMutator
        >>> print(type(MutatorFactory(tf.constant(True, dtype=tf.bool), 0.25)).__name__)
        BoolMutator
        >>> print(type(MutatorFactory(tf.constant(1+0j, dtype=tf.complex64), 0.25, 1.25)).__name__)
        ComplexMutator
        >>> print(type(MutatorFactory(tf.constant(1+0j, dtype=tf.complex128), 0.25, 1.25)).__name__)
        ComplexMutator
        >>> print(type(MutatorFactory(tf.constant('one'), 0.25)).__name__)
        Traceback (most recent call last):
            ....
        AttributeError: You do not have a Mutator defined for constants of type string; or you have, but you haven't added it to MutatorFactory. If you wish to mutate objects of class 'spider', you should create a subclass of Mutator, 'SpiderMutator', and a test for spiders, 'spider_test and add it to MutatorFactory with the line `MutatorFactory.add_mutator(Spidermutator, spider_test[, pos]),where `pos`
        """
        mutator_class = cls.type_tests[0](const, cls.type_tests[1:])
        if mutator_class:
            return mutator_class(mutation_rate, mutation_sd, **kwargs)
        else:
            raise AttributeError(
                "You do not have a Mutator defined for constants of type " +
                f"{const.dtype.name if tf.is_tensor(const) else type(const).__name__}; " +
                "or you have, but you haven't added " +
                "it to MutatorFactory. If you wish to mutate objects of" +
                " class 'spider', you should create a subclass of Mutator, " +
                "'SpiderMutator', and a test for spiders, 'spider_test and add " +
                "it to MutatorFactory with the line " +
                "`MutatorFactory.add_mutator(Spidermutator, spider_test[, pos])," +
                "where `pos`"
            )

class CrossoverMutator(Mutator):


    def __init__(self, mutation_rate: float=0.0, max_depth: int=0, max_size: int=0, **kwargs):
        """Initialises mutation rate, and the maximum size and depth of mutated z
        trees. Details of how these are used in __call__.

        Parameters
        ----------
            mutation_rate : float
                The frequency with which non-zero mutation occurs. A probability
                in the closed interval [0.0, 1.0]. An impossibility by default.
            max_depth: int
                The maximum depth of the tree resulting from a mutation
            max_size: int
                The maximum size of the tree resulting from a mutation
        """
        self.max_depth = max_depth
        self.max_size = max_size
        super().__init__(mutation_rate)

    def __call__(self, val, sd: SizeDepth, ml: MiniLog=None):
        """Randomly decides whether or not to cross over, and if so looks for a 
        subtree that can be crossed in - one which has the same label, and and 
        won't make the resulting tree too big or too deep

        # >>> from tree_factories import RandomPolynomialFactory
        # >>> from gp import GPTreebank
        # >>> import pandas as pd
        # >>> import operators as ops
        # >>> ms, md = 300, 70
        # >>> gp = GPTreebank(
        # ...     mutation_rate = 0.2, 
        # ...     mutation_sd=0.02, 
        # ...     crossover_rate=0.5, 
        # ...     max_depth=md,
        # ...     max_size=ms, 
        # ...     operators=[ops.SUM, ops.PROD, ops.SQ, ops.POW, ops.CUBE]
        # ... )
        # >>> rpf = RandomPolynomialFactory(gp, 5, -10.0, 10.0)
        # >>> trees = [rpf('x', 'y') for _ in range(5)]
        # >>> df = pd.DataFrame({'x': [1.0, 1.0], 'y': [1.0, 1.0]})
        # >>> bigtrees, deeptrees = 0, 0
        # >>> bigness = []
        # >>> deepness = []
        # >>> for _ in range(200):
        # ...     tmax = None
        # ...     valmax = -np.inf
        # ...     for t in trees:
        # ...         val = t(**df)
        # ...         if isinstance(val, pd.Series):
        # ...             val = val.sum()
        # ...         if val is None:
        # ...             print(val)
        # ...             print(t)
        # ...         elif valmax < val:
        # ...             tmax = t
        # ...             valmax = val
        # ...     if tmax is None:
        # ...         print('tmax is None')
        # ...     newtrees = [tmax.copy(gp_copy=True) for _ in range(5)]
        # ...     for tt in trees:
        # ...         tt.delete()
        # ...     trees = newtrees
        # ...     beeeg = bool([tr for tr in trees if tr.size() > ms])
        # ...     if beeeg:
        # ...         bigness.append([(tr.size(), '>', ms) for tr in trees if tr.size() > ms])
        # ...     bigtrees += beeeg
        # ...     deeep = bool([tr for tr in trees if tr.depth() > md])
        # ...     if deeep:
        # ...         deepness.append([(tr.depth(), '>', md) for tr in trees if tr.depth() > md])
        # ...     deeptrees += deeep
        # >>> bd = bigtrees*deeptrees
        # >>> print(f"({bigtrees}){'n' if bd else 'u'}({deeptrees}){'' if bd else '=d'}") # SMILE! (0)u(0)=d
        # (0)u(0)=d
        """
        # Randomly decide whether or not to cross over
        if random() <= self.mutation_rate:
            size = val.size()
            pruned_depth = val.at_depth()
            # To keep the resulting tree below the maximum size and depth, work out:
            # 1) How far from root the substitution site is from root. If the tree 
            #    as a whole is less than max depth, then keeping it below max depth 
            #    means ensuring the new subtree of depth not greater than 
            #    `self.max_depth`, minus the distance from val to root: and...
            # 2) The size of the whole tree, minus the subtree at val which is being
            #    crossed over. The new subtree cannot add more nodes than 
            #    `self.max_size`, minus the nodes of the original tree that are *not* 
            #    being substituted out. Note these are only computed if the maximums
            #    are set - no need to do unnecessary computation
            pruned_size = sd.size - size
            # Make a collection of all same-label nodes in the treebank EXCEPT the 
            # substitution site, val
            complement = _complement(val.label.nodes, val)
            # pick a random subtree from that collection
            if not complement:
                return val
            subtree =  choice(complement)
            sts, std = subtree.size(), subtree.depth()
            # Now, it must be checked that it's within size and/or depth bounds.
            # SizeDepth is set up to be Callable, such that calling `sd` with
            # values for the new output tree size and depth results in the size and
            # depth values of sd being updated *if* they are within size & depth 
            # limits, and returns boolean, True if the update is successful, False
            # otherwise. Therefore, if the update succeeds, the loop condition is
            # broken
            while not sd(pruned_size+sts, max(sd.depth, pruned_depth+std)):
                # If we have to try again, remove the no-good subtree from the pool
                complement = _complement(complement, subtree)
                # If we drain the pool entirely, then the substitution is impossible,
                # and the original subtree will be returned
                if not complement:
                    return val
                subtree = choice(complement)
                sts, std = subtree.size(), subtree.depth()
            if len(subtree)==0:
                print('wut?')
            return subtree
        if len(val)==0:
            print('huh?')
        return val
    
    def get_subtree(self, complement, old_st):
        complement = _complement(complement, old_st)
        # If we drain the pool entirely, then the substitution is impossible,
        # and the original subtree will be returned
        if not complement:
            return old_st
        subtree =  choice(complement)
        sts, std = subtree.size(), subtree.depth()
        return subtree, complement, sts, std

def main():
    import doctest
    doctest.testmod()

if __name__ == '__main__':
    main()
