from abc import ABC, abstractmethod
from math import log, floor, pi, cos, sin
from typing import Any
# from random import choice, random, uniform, gauss

from icecream import ic
from matplotlib.pylab import Generator
import numpy as np

from logtools import MiniLog
from size_depth import SizeDepth 

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
    rng = np.random.Generator(np.random.PCG64(69))

    def mutate_here(self, tree):
        return self.rng.random() <= self.mutation_rate

    def __init__(self, treebank, **kwargs):
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
        self.rng = treebank.np_random if treebank.np_random else self.__class__.rng
        self.mutation_rate = treebank.mutation_rate if treebank.mutation_rate else 0.1
        if not (0.0 <= self.mutation_rate <= 1.0):
            raise ValueError(
                "mutation_rate is a probability, and must not be less than " +
                "0.0 or more than 1.0"
            )

    @abstractmethod
    def __call__(self, val, tree, *args, **kwargs) -> ('Tree', Any):
        """Takes the original value of a constant and applies the mutation
        operator to it. A magic method that makes a mutator Callable. Abstract.

        Parameters
        ----------
            val:
                The value of the constant to be mutated
        """
        pass

    def __matmul__(self, other: 'Mutator'):
        return CompositeMutator(self, other)

class NullMutator(Mutator):
    def __call__(self, val, tree, *args, **kwargs):
        return val, tree
    
class CompositeMutator(Mutator):
    def __init__(self, *mutators):
        self.mutators = []
        for mutator in mutators:
            self.mutators += (
                mutator.mutators 
                if isinstance(mutator, self.__class__) 
                else [mutator]
            )
        self.mutators = tuple(self.mutators)
    
    def __call__(self, val, tree, *args, **kwargs):
        for mutator in self.mutators:
            val, tree = mutator(val, tree, *args, **kwargs)
        return val, tree

class IntMutator(Mutator):
    """Mutates `ints` and integer-like classes. Inherits __init__ from parent
    class, as mutation_rate is the only parameter it needs.

    Parameters
    ----------
        mutation_rate : float
            A probability in the closed interval [0.0, 1.0]. See __call__ for
            how this is used.
    """

    def __call__(self, val, tree, *args, **kwargs):
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
        if not self.mutate_here(tree):
            return val, tree
        # random value in the closed interval [0.0, 1.0]
        randval = self.rng.random()
        # if randval is exactly 0.0 `log(randval, self.mutation_rate)` will be
        # infinite (or nan): very unlikely, but should be handled. If this
        # happens, just try again.
        while not randval:
            randval = self.rng.random()
        # If randval is more than the mutation rate, just return val. The 'else'
        # condition belwo also would return exactly `val` in this case, but this
        # case is handled separately to save computation
        # if randval > self.mutation_rate:
        #     return val
        # If randval <= self.mutation_rate, the mutation is computed as follows:
        # The magnitude of the mutation is computed as
        # floor(log(randval, self.mutation_rate)). The constant is changed by at
        # least 1 with a probability of `self.mutation_rate`, by at least 2 with
        # a probability of `self.mutation_rate^2`, at least 3 with a
        # probability of `self.mutation_rate^3`, and at least n with a
        # probability of `self.mutation_rate^n`. This  is multiplied by
        # `choice((-1,1))` which determines the direction of the mutation,
        # increasing or decreasing with equal probability.
        else:
            return val + (1 + floor(log(randval, self.mutation_rate))) * self.rng.choice((-1,1)), tree

class FloatMutator(Mutator):
    def __init__(self, treebank, **kwargs):
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
        self.mutation_sd = treebank.mutation_sd if treebank.mutation_sd else 1.0
        super().__init__(treebank)

    def __call__(self, val, tree, *args, **kwargs):
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
        if self.mutate_here(tree):
            return self.rng.normal(val, self.mutation_sd), tree
        else:
            return val, tree

class ComplexMutator(Mutator):
    def __init__(self, treebank, **kwargs):
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
        self.mutation_sd = treebank.mutation_sd if treebank.mutation_sd else 1.0
        super().__init__(treebank)

    def __call__(self, val, tree, *args, **kwargs):
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
        if self.mutate_here(tree):
            modulus_delta = self.rng.normal(0, self.mutation_sd)
            argument_delta = self.rng.uniform(0.0, pi)
            delta = modulus_delta * complex(cos(argument_delta), sin(argument_delta))
            return val+delta, tree
        else:
            return val, tree

class BoolMutator(Mutator):
    """Mutates `bools`. Inherits __init__ from parent class, as mutation_rate is
    the only parameter it needs.

    Parameters
    ----------
        mutation_rate : float
            A probability in the closed interval [0.0, 1.0]. See __call__ for
            how this is used.
    """

    def __call__(self, val, tree, *args, **kwargs):
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
        return (val != self.mutate_here(tree)), tree
    
class CrossoverMutator(Mutator):
    def __init__(self, 
           treebank, 
            **kwargs
        ):
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
        self.max_depth = treebank.max_depth
        self.max_size = treebank.max_size
        super().__init__(treebank)

    def __call__(self, val, tree, _max_size=None, _max_depth=None, ml: MiniLog=None):
        """Randomly decides whether or not to cross over, and if so looks for a 
        subtree that can be crossed in - one which has the same label, and and 
        won't make the resulting tree too big or too deep

        >>> from tree_factories import RandomPolynomialFactory
        >>> from gp import GPTreebank
        >>> from test_materials import DummyTreeFactory
        >>> import pandas as pd
        >>> import operators as ops
        >>> rng = np.random.Generator(np.random.PCG64())
        >>> ms, md = 300, 70
        >>> gp = GPTreebank(
        ...     mutation_rate = 0.2, 
        ...     mutation_sd=0.02, 
        ...     crossover_rate=0.5, 
        ...     max_depth=md,
        ...     max_size=ms, 
        ...     seed=rng,
        ...     operators=[ops.SUM, ops.PROD, ops.SQ, ops.POW, ops.CUBE], 
        ...     tree_factory=DummyTreeFactory()
        ... )
        >>> rpf = RandomPolynomialFactory(params = np.array([5, -10.0, 10.0], dtype=float), treebank=gp, seed=rng)
        >>> trees = [rpf('x', 'y') for _ in range(5)]
        >>> df = pd.DataFrame({'x': [1.0, 1.0], 'y': [1.0, 1.0]})
        >>> bigtrees, deeptrees = 0, 0
        >>> bigness = []
        >>> deepness = []
        >>> # Simple GP that selects for big-valued outputs, but subject to max values for tree size & depth
        >>> for _ in range(2000):
        ...     tmax = None
        ...     valmax = -np.inf
        ...     for t in trees:
        ...         val = t(**df)
        ...         if isinstance(val, np.ndarray):
        ...             val = val.sum()
        ...         if val is None:
        ...             print(val)
        ...             print(t)
        ...         elif valmax < val:
        ...             tmax = t
        ...             valmax = val
        ...     if tmax is None:
        ...         ic.enable()
        ...         print(ic('tmax is None'))
        ...     newtrees = []
        ...     for ___ in range(5):
        ...         newtrees.append(tmax.copy(gp_copy=True))
        ...     # newtrees = [tmax.copy(gp_copy=True) for ___ in range(5)]
        ...     for tt in trees:
        ...         tt.delete()
        ...     trees = newtrees
        ...     beeeg = bool([tr for tr in trees if tr.size() > ms])
        ...     if beeeg:
        ...         bigness.append([(tr.size(), '>', ms) for tr in trees if tr.size() > ms])
        ...     bigtrees += beeeg
        ...     deeep = bool([tr for tr in trees if tr.depth() > md])
        ...     if deeep:
        ...         deepness.append([(tr.depth(), '>', md) for tr in trees if tr.depth() > md])
        ...     deeptrees += deeep
        >>> bd = bigtrees+deeptrees
        >>> print(f"({bigtrees}){'n' if bd else 'u'}({deeptrees}){'=q' if bd else '=d'}") # SMILE! (0)u(0)=d
        (0)u(0)=d
        """
        # Randomly decide whether or not to cross over
        # mutate_here is an overrideable method of the base class
        if _max_size is None:
            _max_size = self.max_size
        if _max_depth is None:
            _max_depth = self.max_depth
        if self.mutate_here(tree):
            # The size of mutated trees must be kept within bounds of size and depth:
            # this gp operator is never called on the root node, so the maxima for a
            # a given tree node is calculated by the parent, allowing for the amount
            # of size and depth already accounted for by the rest of the tree. However,
            # the maxima passed from the treebank on initialisation are a reasonable
            # fallback.
            #     --
            # Make an array of all same-label nodes in the treebank EXCEPT the 
            # substitution site, val
            complement = (tree.label.nodes - tree).array()
            # If there aren't any, never mind, cancel the mutation and use the original
            # subtree
            if not len(complement):
                ic('bleh')
                return val, tree
            # shuffle the array and pick the first subtree, using the index i,
            # initialised to 0, to pick it
            i=0
            self.rng.shuffle(complement)
            subtree = complement[i]
            # Now, it must be checked that it's within size and/or depth bounds.
            # SizeDepth is set up to be Callable, such that calling `sd` with
            # values for the new output tree size and depth results in the size and
            # depth values of sd being updated *if* they are within size & depth 
            # limits, and returns boolean, True if the update is successful, False
            # otherwise. Therefore, if the update succeeds, the loop condition is
            # broken
            sts, std = subtree.size(), subtree.depth()
            while subtree.metadata.get('__no_xo__', False) or (sts > _max_size) or (std > _max_depth):
                # If it doesn't work, increment i and try again
                i+=1
                # If we run out of subtrees, then the substitution is impossible,
                # and the original subtree will be returned
                if i>=len(complement):
                    ic('fahhhk')
                    return val, tree
                subtree = complement[i]
                if subtree.metadata.get('__no_xo__', False):
                    continue
                # print('Size complement:', len(complement))
                sts, std = subtree.size(), subtree.depth()
                # print(f'trouble copying, n={n}')
                # n+=1
            if len(subtree)==0:
                print('wut?')

            return val, subtree
        if len(tree)==0:
            print('huh?')
        return val, tree
    
    def get_subtree(self, complement, old_st):
        complement = _complement(complement, old_st)
        # If we drain the pool entirely, then the substitution is impossible,
        # and the original subtree will be returned
        if not complement:
            return old_st
        subtree =  self.rng.choice(complement)
        sts, std = subtree.size(), subtree.depth()
        return subtree, complement, sts, std
    
class TaggingMutator(Mutator):
    TAG = '__mut8__'

    def __call__(self, val, tree, ml: MiniLog=None):
        if self.mutate_here(tree):
            tag_candidates = tree.findall((lambda x: self.tag_here_maybe(x)), trees_only=True)
            if tag_candidates:
                tag_candidates[self.rng.integers(len(tag_candidates))].tmp[self.TAG]=True
        return val, tree

    def mutate_here(self, tree):
        return not bool(tree.parent)
    
    def tag_here_maybe(self, x):
        return hasattr(x.gp_operator, 'TAG') and isinstance(x.gp_operator, TagTriggeredMutator) and x.gp_operator.TAG==self.TAG
    
class TaggingXOMutator(TaggingMutator):
    TAG = '__x0__'
    
    def tag_here_maybe(self, x):
        return x.parent and hasattr(x.parent.xo_operator, 'TAG') and (x.parent.xo_operator.TAG==self.TAG)

class TagTriggeredMutator(Mutator):
    TAG = '__mut8__'
    
    def mutate_here(self, tree):
        mh = tree.tmp.get(self.TAG, False)
        tree.tmp[self.TAG] = False
        return mh

class TagTriggeredIntMutator(TagTriggeredMutator, IntMutator):
    pass

class TagTriggeredFloatMutator(TagTriggeredMutator, FloatMutator):
    pass

class TagTriggeredBoolMutator(TagTriggeredMutator, BoolMutator):
    pass

class TagTriggeredComplexMutator(TagTriggeredMutator, ComplexMutator):
    pass

class TagTriggeredCrossoverMutator(TagTriggeredMutator, CrossoverMutator):
    TAG = '__x0__'

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

    types = {
        np.int64: IntMutator,
        np.float64: FloatMutator,
        np.complex128: ComplexMutator,
        np.bool_: BoolMutator,
    }

    def __new__(cls, const_type, treebank, **kwargs):
        """Initialising a Mutator always results in on of the concrete
        subclasses. The argument `const_type` should be the type of the constant
        that needs mutating, and the Mutator returned will be the subclass
        registered in the `types` dictionary for that type.

        Parameters
        ----------
            const_type : type
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
                If no Mutator subclass is defined for the provided `const_type`.

        >>> from test_materials import GP2
        >>> print(type(MutatorFactory(np.int64, GP2)).__name__)
        IntMutator
        >>> print(type(MutatorFactory(np.float64, GP2)).__name__)
        FloatMutator
        >>> print(type(MutatorFactory(np.bool_, GP2)).__name__)
        BoolMutator
        >>> print(type(MutatorFactory(np.complex128, GP2)).__name__)
        ComplexMutator
        >>> print(type(MutatorFactory(str, GP2)).__name__)
        Traceback (most recent call last):
            ....
        AttributeError: You do not have a Mutator defined for constants of type str; or you have, but you haven't added it to MutatorFactory.types. If you wish to mutate objects of class 'spider', you should create a subclass of Mutator, 'SpiderMutator', and add it to the type_dict with the line `Mutator.type_dict[spider] = SpiderMutator`
        """
        try:
            return cls.types[const_type](treebank, **kwargs)
        except KeyError:
            raise AttributeError(
                "You do not have a Mutator defined for constants of type " +
                f"{const_type.__name__}; or you have, but you haven't added " +
                "it to MutatorFactory.types. If you wish to mutate objects of" +
                " class 'spider', you should create a subclass of Mutator, " +
                "'SpiderMutator', and add it to the type_dict with the line " +
                "`Mutator.type_dict[spider] = SpiderMutator`"
            )

class NullMutatorFactory(MutatorFactory):
    """A useful MutatorFactory for cases where you don't want any leaf
    mutations, but you do want crossover.

    >>> from test_materials import T6, T7, T8, GP2
    >>> mm = single_xo_factory(GP2)
    >>> raw_arr = np.array(T6())
    >>> mm()
    >>> (raw_arr == np.array(T6())).all()
    True
    >>> sum_arr = np.zeros(4, dtype=np.complex128)
    >>> n = 1024*16
    >>> eta = 0.1
    >>> for _ in range(n):
    ...     t6_copy = T6.copy(gp_copy=True)
    ...     mut_arr = np.array(t6_copy())
    ...     t6_copy.delete()
    ...     assert (raw_arr == mut_arr).sum() == 3
    ...     sum_arr = sum_arr + mut_arr
    >>> mean_arr = sum_arr/n
    >>> targ_arr = np.array([6.0, 11.0, 30.0+15.0j, 0.75])
    >>> assert np.abs(np.real(targ_arr[0])-np.real(mean_arr[0])) < eta, np.real(targ_arr[0])-np.real(mean_arr[0])
    >>> assert np.abs(np.real(targ_arr[1])-np.real(mean_arr[1])) < eta, np.real(targ_arr[1])-np.real(mean_arr[1])
    >>> assert np.abs(np.real(targ_arr[2])-np.real(mean_arr[2])) < eta, np.real(targ_arr[2])-np.real(mean_arr[2])
    >>> assert np.abs(np.imag(targ_arr[2])-np.imag(mean_arr[2])) < eta, np.imag(targ_arr[2])-np.imag(mean_arr[2])
    >>> assert np.abs(np.real(targ_arr[3])-np.real(mean_arr[3])) < eta, np.real(targ_arr[3])-np.real(mean_arr[3])
    """
    types = {
        np.int64: NullMutator,
        np.float64: NullMutator,
        np.complex128: NullMutator,
        np.bool_: NullMutator
    }

class SinglePointLeafMutatorFactory(MutatorFactory):
    """If a tree is set with these leaf mutators, and 
    SinglePointLeafMutator on nonterminals, it will be 
    copied with exactly one leaf mutated, provided there is
    at least one mutable leaf

    >>> from test_materials import T6, GP2
    >>> mm = single_leaf_mutator_factory(GP2)
    >>> raw_arr = np.array(T6())
    >>> mm()
    >>> (raw_arr == np.array(T6())).all()
    True
    >>> for _ in range(512):
    ...     assert (raw_arr == np.array(T6.copy(gp_copy=True)())).sum() == 3
    """
    types = {
        np.int64: TagTriggeredIntMutator,
        np.float64: TagTriggeredFloatMutator,
        np.complex128: TagTriggeredComplexMutator,
        np.bool_: TagTriggeredBoolMutator
    }

class MutatorMutator:
    """Swaps out the mutation operaotrs for a tree when called

    >>> from test_materials import T6, GP2
    >>> mm = single_leaf_mutator_factory(GP2)
    >>> raw_arr = np.array(T6())
    >>> mm = single_leaf_mutator_factory(GP2)
    >>> mm()
    >>> print(type(T6.gp_operator).__name__)
    TaggingMutator
    >>> print(type(T6.xo_operator).__name__)
    NullMutator
    >>> for c in T6:
    ...     print(type(c.gp_operator).__name__)
    TagTriggeredIntMutator
    TagTriggeredFloatMutator
    TagTriggeredComplexMutator
    TagTriggeredBoolMutator
    """
    def __init__(self, 
            mutator_factory_class: type[MutatorFactory],
            treebank: 'GPTreebank',
            xo_class: type[CrossoverMutator] = NullMutator,
            nt_mutator_class: type[Mutator] = NullMutator,
            root_xo_class: type[Mutator]|None = None,
            root_mutator_class: type[Mutator]|None = None
        ) -> None:
        self.tb = treebank
        self.mutator_factory_class = mutator_factory_class
        self.x_over_class = xo_class
        self.root_xo_mutator_class = root_xo_class if root_xo_class else xo_class
        self.nt_self_op_class = nt_mutator_class
        self.root_mutator_class = root_mutator_class if root_mutator_class else nt_mutator_class

    def __call__(self):
        for label in self.tb.get_all_root_nodes().values():
            for tree in label:
                self._decorate_tree(tree)

    def _decorate_tree(self,
            tree: 'Tree'
        ) -> None:
        if tree.mutable:
            if hasattr(tree, 'leaf'):
                tree.gp_operator = self.mutator_factory_class(
                    tree.leaf_type, self.tb
                )
            else:
                mut8 = self.nt_self_op_class if tree.parent else self.root_mutator_class
                tree.gp_operator = mut8(self.tb)
                xo = self.x_over_class if tree.parent else self.root_xo_mutator_class
                tree.xo_operator = xo(self.tb)
        if hasattr(tree, 'children'):
            for c in tree:
                self._decorate_tree(c)

def random_mutator_factory(treebank: 'Treebank'):
    return MutatorMutator(
        MutatorFactory, treebank,
        xo_class=CrossoverMutator
    )

def single_leaf_mutator_factory(treebank: 'Treebank'):
    return  MutatorMutator(
        SinglePointLeafMutatorFactory, treebank,
        root_mutator_class=TaggingMutator
    )

def single_xo_factory(treebank: 'Treebank'):
    return MutatorMutator(
        NullMutatorFactory, treebank,
        xo_class=TagTriggeredCrossoverMutator, 
        root_mutator_class=TaggingXOMutator
    )

def main():
    import doctest
    doctest.testmod()

if __name__ == '__main__':
    main()