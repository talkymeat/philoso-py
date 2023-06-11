from abc import ABC, abstractmethod
from math import log, floor, pi, cos, sin
from random import choice, random, uniform, gauss

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

    def __init__(self, mutation_rate, mutation_sd=1.0, **kwargs):
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
    def __call__(self, val):
        """Takes the original value of a constant and applies the mutation
        operator to it. A magic method that makes a mutator Callable. Abstract.

        Parameters
        ----------
            val:
                The value of the constant to be mutated
        """
        pass

class IntMutator(Mutator):
    """Mutates `ints` and integer-like classes. Inherits __init__ from parent
    class, as mutation_rate is the only parameter it needs.

    Parameters
    ----------
        mutation_rate : float
            A probability in the closed interval [0.0, 1.0]. See __call__ for
            how this is used.
    """

    def __call__(self, val):
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
        # random value in the closed interval [0.0, 1.0]
        randval = random()
        # if randval is exactly 0.0 `log(randval, self.mutation_rate)` will be
        # infinite (or nan): very unlikely, but should be handled. If this
        # happens, just try again.
        while not randval:
            randval = random()
        # If randval is more than the mutation rate, just return val. The 'else'
        # condition belwo also would return exactly `val` in this case, but this
        # case is handled separately to save computation
        if randval > self.mutation_rate:
            return val
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
            return val + (floor(log(randval, self.mutation_rate)) * choice((-1,1)))

class FloatMutator(Mutator):
    def __init__(self, mutation_rate, mutation_sd=1.0, **kwargs):
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

    def __call__(self, val):
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
    def __init__(self, mutation_rate, mutation_sd=1.0, **kwargs):
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

    def __call__(self, val):
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

    def __call__(self, val):
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
        int: IntMutator,
        float: FloatMutator,
        complex: ComplexMutator,
        bool: BoolMutator
    }

    def __new__(cls, const_type, mutation_rate, mutation_sd=None, **kwargs):
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

        >>> print(type(MutatorFactory(int, 0.25)).__name__)
        IntMutator
        >>> print(type(MutatorFactory(float, 0.25, 1.25)).__name__)
        FloatMutator
        >>> print(type(MutatorFactory(bool, 0.25)).__name__)
        BoolMutator
        >>> print(type(MutatorFactory(complex, 0.25, 1.25)).__name__)
        ComplexMutator
        >>> print(type(MutatorFactory(str, 0.25)).__name__)
        Traceback (most recent call last):
            ....
        AttributeError: You do not have a Mutator defined for constants of type str; or you have, but you haven't added it to MutatorFactory.types. If you wish to mutate objects of class 'spider', you should create a subclass of Mutator, 'SpiderMutator', and add it to the type_dict with the line `Mutator.type_dict[spider] = SpiderMutator`
        """
        try:
            if mutation_sd == None:
                return cls.types[const_type](mutation_rate, **kwargs)
            else:
                return cls.types[const_type](mutation_rate, mutation_sd, **kwargs)
        except KeyError:
            raise AttributeError(
                "You do not have a Mutator defined for constants of type " +
                f"{const_type.__name__}; or you have, but you haven't added " +
                "it to MutatorFactory.types. If you wish to mutate objects of" +
                " class 'spider', you should create a subclass of Mutator, " +
                "'SpiderMutator', and add it to the type_dict with the line " +
                "`Mutator.type_dict[spider] = SpiderMutator`"
            )

class CrossoverMutator(Mutator):
    def __call__(self, val):
        if random() <= self.mutation_rate:
            try:
                return choice(_complement(val.label.nodes, val))
            except IndexError:
                print(_complement(val.label.nodes, val))
                raise IndexError
        else:
            return val

def main():
    import doctest
    doctest.testmod()

if __name__ == '__main__':
    main()
