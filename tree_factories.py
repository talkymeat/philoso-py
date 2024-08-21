from typing import Type, Union, Iterable, Mapping, Callable, TypeAlias, TypeVar, Sequence
from abc import ABC, abstractmethod
import operators as ops
from trees import Terminal, NonTerminal, Tree, SubstitutionSite
from gp_trees import GPTerminal, GPNonTerminal
import pandas as pd
import numpy as np
from itertools import chain, combinations
from scipy.special import comb
from functools import reduce
from dataclasses import dataclass
from type_ify import TypeNativiser
from observatories import GenFunc
from treebanks import Treebank
from rl_bases import Actionable
from gymnasium.spaces import Box
from tree_funcs import get_operators
from operators import Operator

from icecream import ic

T = TypeVar('T')
Var:   TypeAlias = tuple[type, str]
Const: TypeAlias = tuple[type[T], Callable[[], T]]

class TreeFactory(ABC):
    @property
    @abstractmethod
    def treebank(self)->"GPTreebank":
        pass

    @treebank.setter
    @abstractmethod
    def treebank(self, treebank: "GPTreebank"):
        pass

    @abstractmethod
    def __call__(self, *vars: Iterable[str], treebank: Treebank=None) -> Tree:
        pass

    @property
    @abstractmethod
    def op_set(self)->set[Operator]:
        pass

    @property
    def seed(self) -> int:
        return self.np_random.bit_generator.seed_seq.entropy
    
    @seed.setter
    def seed(self, seed: int|np.random.Generator|None):
        if isinstance(seed, np.random.Generator):
            self.np_random = seed
        else:
            self.np_random = np.random.Generator(np.random.PCG64(seed))
    
    @property
    def tf_params(self):
        return {}
    
    @property
    def prefix(self):
        pass


class TestTreeFactory(TreeFactory):
    def __init__(self, start_tree: Tree, **kwargs):
        self.start_tree = start_tree
        self.treebank = start_tree.treebank

    @property
    def treebank(self)->"GPTreebank":
        return self._treebank

    @treebank.setter
    def treebank(self, treebank: "GPTreebank"):
        self._treebank = treebank

    def __call__(self, *vars: Iterable[str], treebank: Treebank=None) -> Tree:
        return self.start_tree.copy_out(treebank if treebank else self.treebank)
    
    @property
    def op_set(self):
        return get_operators(self.start_tree)
    
    @property
    def prefix(self):
        return "test"
    
class CompositeTreeFactory(TreeFactory):
    def __init__(self, 
            tree_factories: list[TreeFactory], 
            weights: Iterable[float|int], 
            *args,
            seed: int|np.random.Generator|None = None, 
            treebank: Treebank|None = None, 
            **kwargs):
        if weights and len(weights) != len(tree_factories):
            raise ValueError(
                "tree_factories and weights should be the same length"
            )
        for tf in tree_factories:
            tf.seed = seed
        self.tree_factories = tree_factories
        self.weights = weights
        if treebank:
            self.treebank = treebank

    @property
    def treebank(self)->"GPTreebank":
        return self._treebank

    @treebank.setter
    def treebank(self, treebank: "GPTreebank"):
        self._treebank = treebank
        for tf in self.tree_factories:
            tf.treebank = treebank

    def __call__(self, *vars: Iterable[str], treebank: Treebank=None) -> Tree:
        return self.np_random.choice(
            self.tree_factories, 
            p=self.weights
        )[0](
            vars, 
            treebank = treebank if treebank else self.treebank
        )
    
    @property
    def seed(self):
        return [tf.seed for tf in self.tree_factories]
    
    @seed.setter
    def seed(self, seed: int|list[int]|None):
        if isinstance(seed, list):
            if len(seed) != len(self.tree_factories):
                raise ValueError(
                    "List of seeds should be the same length as " +
                    "list of TreeFactories"
                )
            for tf, s in zip(self.tree_factories, seed):
                tf.seed = s
        else:
            for tf in self.tree_factories:
                tf.seed = seed    
    
    @property
    def op_set(self)->set[Operator]:
        ops = set()
        for tf in self.tree_factories:
            ops |= set(tf.op_set)
        return ops
    
    @property
    def tf_params(self):
        param_dict = {f"{tf.prefix}_weight": weight for tf, weight in zip(self.tree_factories, self.weights)}
        for tf in self.tree_factories:
            param_dict = {**param_dict, **tf.tf_params}
        return param_dict
    
    @property
    def prefix(self):
        return ""

class RandomPolynomialFactory(TreeFactory):
    """A TreeFactory which makes random trees representing polynomials of a specified
    order (`order`) and set of variables (`vars`), with random coefficients uniformly 
    distributed in a specified range (`const_min` to `const_max`)
    """
    _act_param_space = Box(
        low=np.array([1.0, -np.inf, -np.inf]),
        high=np.array([25., np.inf, np.inf]),
        dtype=np.float32
    )
    _act_param_names = ['order', 'const_min', 'const_max']

    def __init__(
            self,
            params: np.ndarray|None=None,
            treebank: Treebank = None,
            seed: int|list[int]|None = None, # XXX get rid of this default
            max_size: int = None,
            max_depth: int = None
        ):
        self.seed = seed
        self.order_map = {}
        if max_size is not None and max_size < 1:
            raise ValueError(f'max_size must be 1 or greater: {max_size} is not a valid value')
        if max_depth is not None and max_depth < 1:
            raise ValueError(f'max_depth must be 1 or greater: {max_depth} is not a valid value')
        self.max_size = max_size
        self.max_depth = max_depth
        if params is not None:
            self.order = int(params[0])
            self.const_min = params[1]
            self.const_max = params[2]
        else:
            self.order = 3
            self.const_min = -0.05
            self.const_max = 0.05
        if self.const_min > self.const_max:
            self.const_min, self.const_max = self.const_max, self.const_min
        if treebank:
            self.treebank = treebank
        else:
            self.treebank = None
        self.operators = {
            "SUM": ops.SUM,
            "PROD": ops.PROD,
            "SQ": ops.SQ,
            "CUBE": ops.CUBE,
            "POW": ops.POW
        }
        if treebank:
            self.treebank.operators = self.operators

    def effective_order(self, num_vars: int):
        return self.order_map.get(num_vars, self.order)
    
    @property
    def tf_params(self):
        return {
            "rpf_const_min": self.const_min,
            "rpf_const_max": self.const_max,
            "rpf_order": self.order
        }

    @property
    def op_set(self)->set[Operator]:
        return set(self.operators.values())

    def _poly_terms(self, vars_: Iterable[str], order: int) -> tuple[tuple[str|int]]:
        """Generates a tuple of tuples, in which each tuple represents a term of 
        the polynomial. Each tuple representing a term is itself comprised of 
        two tuples, the first listing the variables of the term, and the second
        representing the powers they are to be raised to

        >>> from gp import GPTreebank
        >>> gp = GPTreebank(max_size=20, max_depth=10, operators=[
        ...     ops.SUM, ops.PROD, ops.SQ, ops.CUBE, ops.POW
        ... ])
        >>> rpf = RandomPolynomialFactory(params = np.array([3., -10.0, 10.0], dtype=np.float32), treebank=gp)
        >>> rpf._poly_terms(['x', 'y'], 3)
        (((), ()), (('x',), (1,)), (('y',), (1,)), (('x',), (2,)), (('y',), (2,)), (('x', 'y'), (1, 1)), (('x',), (3,)), (('y',), (3,)), (('x', 'y'), (2, 1)), (('x', 'y'), (1, 2)))
        >>> rpf._poly_terms(['x'], 3)
        (((), ()), (('x',), (1,)), (('x',), (2,)), (('x',), (3,)))
        >>> rpf._poly_terms(['x', 'y', 'z'], 2)
        (((), ()), (('x',), (1,)), (('y',), (1,)), (('z',), (1,)), (('x',), (2,)), (('y',), (2,)), (('z',), (2,)), (('x', 'y'), (1, 1)), (('x', 'z'), (1, 1)), (('y', 'z'), (1, 1)))
        """
        # Creates tuples representing every unique combination of variables,
        # excluding those with more variables than the order of the polynomial
        combos = tuple(
            chain.from_iterable(combinations(vars_, r) for r in range(order+1))
        )
        terms = []
        # this pairs copies of each combination of variables with powers that 
        # they can be raised to in polynomials of the specified order. Thus, this
        # outer loop iterate over the terms of order `n` from zero to `order`...
        for n in range(order+1):
            # and the inner loop iterates over the set of combinations of variables
            # in `combos` that do not contain more than `n variables`...
            for c in filter(lambda combo: len(combo) <= n, combos):
                # If the combination of variables is not empty, we add to the 
                # list of terms copies of the combination paired with each 
                # possible tuple positive ints such that the length of the tuple
                # equals the length of the tuple of variables, and the ints sum 
                # to `n`. Each pairing of a combination of varaibles with a tuple
                # of ints represents the variables and the powers they are raised
                # to of one term of the polynomial
                if len(c):
                    # '_all_sums' provides the tuples of ints summing to `n`
                    terms += [(c, powers) for powers in self._all_sums(n, len(c))]
                elif not n:
                    # if there are no varaibles and n==0, this represents the 
                    # constant term of the polynomial, represented by a pair of 
                    # empty tuples
                    terms.append(((), ()))
        return tuple(terms)

    def _all_sums(self, tot: int, k: int) -> tuple[tuple[int]]: # do a video on this
        """This function returns all tuples of positive ints of length `k` which 
        sum to `tot`.

        Parameters
        ==========
            tot (int): The total the values in each tuple must add to
            k (int): The number of terms in the tuple

        Returns
        =======
            tuple[tuple]

        >>> from gp import GPTreebank
        >>> rpf = RandomPolynomialFactory(params = np.array([3., -10.0, 10.0], dtype=np.float32), treebank=GPTreebank())
        >>> for k in range(1,6):
        ...     for tot in range(1,6):
        ...         print(rpf._all_sums(tot, k))
        ((1,),)
        ((2,),)
        ((3,),)
        ((4,),)
        ((5,),)
        ()
        ((1, 1),)
        ((2, 1), (1, 2))
        ((3, 1), (2, 2), (1, 3))
        ((4, 1), (3, 2), (2, 3), (1, 4))
        ()
        ()
        ((1, 1, 1),)
        ((2, 1, 1), (1, 2, 1), (1, 1, 2))
        ((3, 1, 1), (2, 2, 1), (2, 1, 2), (1, 3, 1), (1, 2, 2), (1, 1, 3))
        ()
        ()
        ()
        ((1, 1, 1, 1),)
        ((2, 1, 1, 1), (1, 2, 1, 1), (1, 1, 2, 1), (1, 1, 1, 2))
        ()
        ()
        ()
        ()
        ((1, 1, 1, 1, 1),)
        """
        if tot==k:
            # If `tot == k`, that's a tuple of `k` 1's
            return ((1,) * k,)
        elif k==1:
            # if `k == 1` that's a singleton tuple `(tot,)`
            # This function is recursive, and these first two conditions are 
            # the terminating conditions
            return ((tot,),)
        else:
            # In this case, `k` is neither `tot` nor 1, and so there are 
            # multiple possible sums 
            sums = () # stores the tuples to be returned
            head = tot - (k-1) # the value of the first value in a tuple. We 
            # first compute the maximum value that an item can have, given the 
            # values of `k` and `order`. Iterate through values of `head`, 
            # starting with the maximum and decrementing each loop
            while head>0: # for all positive values of `head`
                # if `head` is the first element of the tuple, use a recursive
                # call to find all he possible tails
                tails = self._all_sums(tot-head, k-1)
                # Combine all the possible tails with `head` and add them to 
                # the tuple of tuples, `sums`
                sums += tuple((head,) + tail for tail in tails)
                # decrement head and restart the loop
                head -= 1
            return sums

    def _binarise_tree(self, 
            op_name: str, 
            tree_list: Iterable[Tree], 
            initial: Tree=None, 
            nt: NonTerminal=None
        ) -> Tree:
        """Takes a list of trees and combines them by repeatedly applying an 
        operator pairwise. Thus, combining trees `A`, `B`, `C`, `D`, and `E` 
        would give a tree `(((A + B) + C) + D) + E)`

        Parameters
        ==========  
            op_name (str): Name of operator
            tree_list (list[Tree]): List of trees to be combined
            initial (Tree or None): An aditional tree may be included to be 
                operated with the first tree in the list

        Returns
        =======
            Tree

        Raises
        ======
            ValueError: If operator is not in the class's operator dict
        
        >>> from gp import GPTreebank
        >>> gp = GPTreebank(max_size=20, max_depth=10, operators=[
        ...     ops.SUM, ops.PROD, ops.SQ, ops.CUBE, ops.POW
        ... ])
        >>> rpf = RandomPolynomialFactory(params = np.array([3., -10.0, 10.0], dtype=np.float32), treebank=gp)
        >>> initial = gp.tree('([float]0.0)')
        >>> tl = [gp.tree('([float]1.0)'), gp.tree('([float]2.0)'), gp.tree('([float]3.0)'), gp.tree('([float]4.0)')]
        >>> t0 = rpf._binarise_tree('SUM', tl)
        >>> t0
        tree("([float]<SUM>([float]<SUM>([float]<SUM>([float]1.0)([float]2.0))([float]3.0))([float]4.0))")
        >>> t1 = rpf._binarise_tree('SUM', tl, initial)
        >>> t1
        tree("([float]<SUM>([float]<SUM>([float]<SUM>([float]<SUM>([float]0.0)([float]1.0))([float]2.0))([float]3.0))([float]4.0))")
        >>> t2 = rpf._binarise_tree('PROD', tl)
        >>> t2
        tree("([float]<PROD>([float]<PROD>([float]<PROD>([float]1.0)([float]2.0))([float]3.0))([float]4.0))")
        >>> t3 = rpf._binarise_tree('PROD', tl, initial)
        >>> t3
        tree("([float]<PROD>([float]<PROD>([float]<PROD>([float]<PROD>([float]0.0)([float]1.0))([float]2.0))([float]3.0))([float]4.0))")
        """
        # First, make sure we have a node-type `N` for NonTerminals
        N = nt if nt else self.N
        # If the provided operator is in the `tree_factory`'s operator set, we
        # can proceed 
        if op_name in self.operators:
            # We use `functools.reduce` to turn the list of subtrees into a 
            # single binary tree. This takes a function arg (itself having 2 
            # args, in this case all of type `Tree`, return type also `Tree`) to
            # pairwise combine all the elements of a list (here, a list of 
            # `Trees`). The function we use takes two `Trees` and combines them
            # under a new parent with the provided operator; the resulting 
            # subtree is then combined with the next `Tree` in the list in the
            # same way, and so on. `functools.reduce` has an optional arg,
            # `initial`, which if passed is combined by the pairwise function
            # with the first list element, which is then pased along with the
            # second element to the pairwise function, and so on---if `initial`
            # is not passed, reduce starts with the first two list items instead. 
            # Since we want 'initial' also to be optional for `_binarise_tree`, 
            # we place it in a list if passed, to be unrolled as *args in
            # `reduce`, or use an empty list as *args otherwise  
            args = [initial] if initial else []
            return reduce(
                lambda t1, t2: N(
                    self.treebank, 
                    float, 
                    t1, 
                    t2, 
                    operator=self.operators[op_name]
                    ), 
                tree_list,
                *args
            )
        else:
            raise ValueError(f'op_name "{op_name}" must be in the tree factory\'s operators dict, {self.operators}')

    @property
    def treebank(self)->"GPTreebank":
        return self._treebank

    @treebank.setter
    def treebank(self, treebank: "GPTreebank"):
        self._treebank = treebank
        self.T = treebank.T
        self.N = treebank.N
        self.max_size = treebank.max_size
        self.max_depth = treebank.max_depth

    def __call__(
            self, 
            *vars: str, 
            treebank: Treebank=None, 
            coefficients: Mapping[float, tuple[tuple[str], tuple[int]]]=None
        ) -> Tree:
        """Generates `gp_tree` representations of polynomials of a given order 
        (specified at `__init__`) and set of variables `*vars` with either random
        coefficients, or coefficients specified by the optional keyword arg 
        `coefficients`
        
        Parameters
        ==========
            *vars (str): one or more positional arguments, representing the
                names of the variables of the polynomial
            treebank (Treebank): a treebank for the output tree nodes may 
                optionally be provided, if the treebank provided as `__init__` 
                is not to be used
            coefficients (dict[tuple[tuple[str], tuple[int]], float]): a  
                dictionary providing values for some or all of the coefficients 
                of the polynomial may optionally given. The values should be 
                floats, and the keys should tuples comprising two tuples of 
                equal length. The first should contain between zero and 
                `len(vars)` strings, each being a unique member of `vars`, with
                elements in the same order they appear in `vars`; the second 
                should be integers that sum to no more than `order`, 
                corresponding to the powers the variables in the first tuple are
                raised to in some coefficient of the polynomial. Invalid keys 
                are ignored without error, and if no key is provided for a given 
                term, the coefficient for that term will be randomly generated 
                using a uniform distribution between `self.const_min` and 
                `self.const_max`. If no value is passed, all coefficients will
                be randomly generated.

        >>> from gp import GPTreebank
        >>> gp = GPTreebank(max_size=10, max_depth=5)
        >>> rpf = RandomPolynomialFactory(params = np.array([3., -10.0, 10.0], dtype=np.float32), treebank=gp)
        >>> p1 = rpf('x', coefficients={((), ()): 1, (('x',), (1,)): 1, (('x',), (2,)): 1, (('x',), (3,)): 1})
        >>> list(p1(**pd.DataFrame({'x': [1, 2, 3, 4]})))
        [4.0, 15.0, 40.0, 85.0]
        """
        # The overall proceedure here is to create subtrees for the terms of the
        # polynomial, then use  `_binarise_tree` to make a binary tree that adds
        # them up. `term_subtrees` is the container for the subtrees.
        term_subtrees = []
        # If no treebank is provided in the args, use self.treebank
        treebank = treebank if treebank else self.treebank
        # The `coefficients` dict arg is intended to allow some of the 
        # coefficients of the polynomial to be specified; for any coeffs not 
        # specified in the dict, a uniformly distributed random coefficient is 
        # generated. If no dict is provided, use an empty dict:
        if not coefficients:
            coefficients = {}
        # `_poly_terms` generates the terms of the polynomial, as a tuple of 
        # tuples, where the member tuples each represent a term.
        for term in self._poly_terms(vars, self.effective_order(len(vars))):
            # The coefficient for a term is either the coefficient specified in 
            # `coefficients`, or, if the corresponding key is not found, a
            # uniformly distributed random value between `const_min` and 
            # `const_max` will be used 
            coeff = coefficients.get(term, self.np_random.uniform(self.const_min, self.const_max))
            # the procedure for making the terms is to make subtrees for each 
            # variable, which are then combined using `_binarise_tree` and the 
            # 'PROD' operator. The factors can be ...
            var_pows = []
            for var, pow in zip(term[0], term[1]):
                # ... the variable by itself, (that is, raised to the 1st power)
                # ...
                if pow == 1:
                    var_pows.append(treebank.T(self.treebank, float, f'${var}'))
                # ... or, if it's raised to the 2nd or 3rd power, the variable
                # under a node with with the 'SQ' or 'CUBE' operator 
                # respectively ... 
                elif pow == 2:
                    var_pows.append(treebank.N(self.treebank, float, treebank.T(self.treebank, float, f'${var}'), operator=self.operators['SQ']))
                elif pow == 3:
                    var_pows.append(treebank.N(self.treebank, float, treebank.T(self.treebank, float, f'${var}'), operator=self.operators['CUBE']))
                # ... or, for a higher power, under a node with the 'POW' 
                # operator, with the second argument being the specified power
                else:
                    var_pows.append(treebank.N(self.treebank, float, treebank.T(self.treebank, float, f'${var}'), treebank.T(self.treebank, int, pow), operator=self.operators['POW']))
            # Once we have the variables raised to their respective powers, they 
            # are joined together in to make subtree for the whole terms with
            # the 'PROD' operator, along with the coefficient. If the 
            # coefficient equals 1 and the set of variables is non-empty, the
            # coefficient is implicit; otherwise it is provided as the `initial` 
            # arg for `_binarise_tree`. If the coefficient equals zero, then the 
            # term is redundant and need not be included.
            if var_pows and coeff == 1:
                term_subtrees.append(self._binarise_tree('PROD', var_pows, nt=treebank.N))
            elif coeff:
                term_subtrees.append(self._binarise_tree('PROD', var_pows, initial=treebank.T(self.treebank, float, float(coeff)), nt=treebank.N))
        # Finally, the terms are combined in a binary tree with 'SUM'
        out_tree = self._binarise_tree('SUM', term_subtrees, nt=treebank.N)
        if (self.max_size is None) and (self.max_depth is None):
            return out_tree
        elif (out_tree.size() <= self.max_size) and (out_tree.depth() <= self.max_depth):
            return out_tree
        elif self.effective_order(len(vars))==0:
            return out_tree
        out_tree.delete()
        self.order_map[len(vars)] = self.effective_order(len(vars)) - 1
        return self(*vars, treebank=treebank, coefficients=coefficients)
    
    @property
    def prefix(self):
        return 'poly'

class TreeFactoryFactory(Actionable):
    """Don't look at me."""
    def __init__(self, tf_type=type[TreeFactory], seed=int|None):
        self.tf_type = tf_type
        self.seed = seed
        self._act_param_space = tf_type._act_param_space
        self._act_param_names = tf_type._act_param_names

    def act(self, params: np.ndarray|dict|list, *args, **kwargs):
        super().act(params, *args, **kwargs)
        return self.tf_type(params, seed=self.seed, treebank=kwargs['treebank'])

# XXX consider `full` and `grow` (Poli et al. p.12) methods for tree seeding

class RandomTreeFactory(TreeFactory):
    tn = TypeNativiser()

    @dataclass
    class NTTemplate:
        root: type
        operator: ops.Operator
        children: Sequence[type]

        def __post_init__(self):
            """Checks that the TreeTemplate is typesafe with the Operator.

            >>> from gp import GPTreebank
            >>> gptb = GPTreebank()
            >>> RandomTreeFactory.NTTemplate(root=tuple, operator=ops.ID, children=(float, int, str))
            (float, int, str) -> tuple
            >>> RandomTreeFactory.NTTemplate(root=str, operator=ops.SUM, children=(int, int))
            Traceback (most recent call last):
                ...
            TypeError: root: str and children: (int, int) is not a valid template for Operator <SUM>
            >>> RandomTreeFactory.NTTemplate(root=int, operator=ops.SUM, children=(str, int))
            Traceback (most recent call last):
                ...
            TypeError: root: int and children: (str, int) is not a valid template for Operator <SUM>
            """
            if not self.operator.is_valid(self.root, *self.children):
                raise TypeError(
                    f"root: {self.root.__name__} and children: (" +
                    f"{', '.join([t.__name__ for t in self.children])}) is not a valid" + 
                    f" template for Operator {self.operator}")
            
        def __str__(self):
            """Describes the subtree represented by the TreeTemplate in function
            notation, showing the parameter types and return type.

            >>> from gp import GPTreebank
            >>> gptb = GPTreebank()
            >>> tt = RandomTreeFactory.NTTemplate(root=float, operator=ops.SUM, children=(float, int, float))
            >>> print(tt)
            SUM(float, int, float) -> float
            """
            param_str = ", ".join([c.__name__ for c in self.children])
            return f"{self.operator.name}({param_str}) -> {self.root.__name__!s}"
        
        def __repr__(self):
            return str(self)
        
        def __call__(self, treebank):
            """Generates a depth-1 subtree"""
            return treebank.N(
                treebank, 
                self.root, 
                *[
                    SubstitutionSite(treebank, child, i) 
                    for i, child 
                    in enumerate(self.children)
                ],
                operator=self.operator
            )
        
    @dataclass
    class ConstTemplate:
        root: type[T]
        genfunc: Callable[[], T]

        def __call__(self, treebank):
            """
            
            >>> from gp import GPTreebank
            >>> gf1 = lambda: 2
            >>> gf2 = lambda: 2.0
            >>> gf3 = lambda: 'bollocks'
            >>> gptb = GPTreebank()
            >>> ct1 = RandomTreeFactory.ConstTemplate(int, gf1)
            >>> ct1(gptb)
            tree("([int]2)")
            """
            x = self.genfunc()
            x = x.item() if hasattr(x, 'item') else x
            if isinstance(x, self.root):
                return treebank.T(treebank, self.root, x)
            raise TypeError(
                f"genfunc for ConstTemplate {self.root.__name__} returned " +
                f"{x}, which is not a {self.root.__name__}"
            )
        
        def __str__(self):
            return f"Const k: {self.root.__name__}"
        
        def __repr__(self):
            return str(self)
        
    @dataclass
    class VarTemplate:
        root: type[T]
        name: str

        def __call__(self, treebank):
            return treebank.T(treebank, self.root, f'${self.name}')
        
        def __str__(self):
            return f"Var ${self.name}: {self.root.__name__}"
        
        def __repr__(self):
            return str(self)


    def make_template(self, t: Sequence|Mapping|NTTemplate|VarTemplate|ConstTemplate):
        """`make_templete` makes templates for single depth-1 subtrees. It
        can handle `Mappings` and `Sequences` (if valid), which it converts 
        to Templates, and Templates, which it returns unchanged.
        
        The following type signatures of inputs generate the following
        outputs: 

        Output `NTTemplate`;
        ====================
        Sequence:   (type, Operator or Operator name, Sequence of types)
        Mapping:    {
                        'root': type, 
                        'operator': Operator or Operator name, 
                        'children': Sequence of types
                    }
        Template:   NTTemplate 

        Output `VarTemplate`;
        ====================
        Sequence:   (type, str)
        Mapping:    {
                        'root': type, 
                        'name': str 
                    }
        Template:   VarTemplate 

        Output `ConstTemplate`;
        ====================
        Sequence:   (type, generator function to generate initial random value)
        Mapping:    {
                        'root': type, 
                        'genfunc': generator function
                    }
        Template:   ConstTemplate 

        Any other input is rejected.

        >>> from test_materials import GP2
        >>> rtf = RandomTreeFactory([], [], ops.OperatorFactory(), treebank=GP2)
        >>> f = lambda: 2
        >>> ts = [
        ...     (float, 'SUM', [float, float]), 
        ...     {'root' :float, 'operator': rtf.op_fac('SUM'), 'children': [float, float]},
        ...     rtf.NTTemplate(float, rtf.op_fac('SUM'), [float, float]),
        ...     (bool, 'P'),
        ...     {'root': bool, 'name': 'P'},
        ...     rtf.VarTemplate(bool, 'P'),
        ...     (int, f),
        ...     {'root': int, 'genfunc': f},
        ...     rtf.ConstTemplate(root=int, genfunc=f)
        ... ]
        >>> for tt in ts:
        ...     print(rtf.make_template(tt))
        SUM(float, float) -> float
        SUM(float, float) -> float
        SUM(float, float) -> float
        Var $P: bool
        Var $P: bool
        Var $P: bool
        Const k: int
        Const k: int
        Const k: int
        """
        # If the input `t` is a mapping, put then in a tuple and call it `t`:
        # pass that on to the next set of conditions. This is done because the 
        # `Sequence` case includes all the type-checking of inputs
        if isinstance(t, Mapping):
            # 'root' should always be in t, if t is a Mapping: otherwise, reject
            if 'root' in t:
                # these are the Mapping entries for a Non terminal
                if len(t)==3 and 'operator' in t and 'children' in t:
                    t = [t['root'], t['operator'], t['children']]
                # Var and Const only need two
                elif len(t)==2:
                    # Var case
                    if 'name' in t:
                        t = [t['root'], t['name']]
                    # Const case
                    elif 'genfunc' in t:
                        t = [t['root'], t['genfunc']]
        if isinstance(t, Sequence):
            t = list(t)
            # all sequences must be len 2 or 3, with item 0 being a type, else reject
            if len(t) in (2, 3) and isinstance(t[0], type):
                # The Non-terminal case
                if len(t) == 3 and isinstance(t[1], (ops.Operator, str)) and isinstance(t[2], Sequence):
                    # if operator name string is given replace with Operator
                    t[1] = self.op_fac(t[1]) if isinstance(t[1], str) else t[1]
                    # Children must all be types
                    if sum([isinstance(c, type) for c in t[2]]) == len(t[2]):
                        return self.NTTemplate(*t)
                if len(t)==2:
                    # if len 2, item 1 is a string which is a valid python identifier, return VarTemplate
                    if isinstance(t[1], str) and t[1].isidentifier():
                        return self.VarTemplate(*t)
                    # if len 2, and item 1 is a function, return ConstTemplate
                    elif callable(t[1]):
                        return self.ConstTemplate(*t)
        # If t is a template, return it. User may pass args for templates, 
        # kwargs, or already-initialised templates as the templates list
        # in RPF.__init__, so this should handle all cases
        elif isinstance(t, (self.NTTemplate, self.VarTemplate, self.ConstTemplate)):
            return t
        # any input not handled above is invalid
        raise ValueError(f'Invalid tree template {t}')


    def __init__(self,
            templates: Sequence[Sequence|Mapping|NTTemplate|VarTemplate|ConstTemplate], 
            root_types: Sequence[type], 
            operator_factory: ops.OperatorFactory, 
            treebank: Treebank,
            *args,
            # weights=None, XXX later
            **kwargs): 
        self.op_fac = operator_factory
        self.treebank = treebank
        templates = [self.make_template(t) for t in templates]
        templates = self._validate_templates(templates, root_types)
        self.clear()
        self._sort_templates(templates, root_types)
        self._all_starts = None
        self._leaf_queues = {}
        self._op_set = set()

    def clear(self):
        self._roots, self._inners, self._consts, self._vars = {}, {}, {}, {}

    def _sort_templates(self, 
            templates, 
            root_types, 
            # weights=None
        ):
        """Takes the list of templates and places them in dicts, with 
        types (root-labels) for keys - with distinct dicts for start-sites,
        other non-terminals, consts, and vars.
        
        >>> from test_materials import RTF, TS
        >>> RTF._sort_templates(TS, [float])
        >>> print([k.__name__ for k in RTF._roots.keys()][0])
        float
        >>> print(', '.join(sorted([k.__name__ for k in RTF._inners.keys()])))
        bool, int
        >>> print([k.__name__ for k in RTF._vars.keys()][0])
        float
        >>> print(', '.join(sorted([k.__name__ for k in RTF._consts.keys()])))
        bool, float, int
        >>> print('\\n'.join(sorted([str(k) for k in RTF._roots[float]])))
        POW(float, int) -> float
        SUM(float, float) -> float
        SUM(int, int) -> float
        TERN_FLOAT(bool, float, float) -> float
        >>> print('\\n'.join(sorted([str(k) for k in RTF._inners[int]])))
        INT_SUM(int, int) -> int
        >>> print('\\n'.join(sorted([str(k) for k in RTF._inners[bool]])))
        EQ(float, float) -> bool
        EQ(int, int) -> bool
        >>> print('\\n'.join(sorted([str(k) for k in RTF._consts[float]])))
        Const k: float
        >>> print('\\n'.join(sorted([str(k) for k in RTF._consts[int]])))
        Const k: int
        >>> print('\\n'.join(sorted([str(k) for k in RTF._consts[bool]])))
        Const k: bool
        Const k: bool
        >>> print('\\n'.join(sorted([str(k) for k in RTF._vars[float]])))
        Var $x: float
        """
        for template in templates:
            if isinstance(template, self.NTTemplate):
                d = self._roots if template.root in root_types else self._inners
            elif isinstance(template, self.ConstTemplate):
                d = self._consts
            elif isinstance(template, self.VarTemplate):
                d = self._vars
            else:
                raise ValueError(f"{template} is an invalid template")
            d[template.root] = d.get(template.root, []) + [template]

    def rand_start(self, label=None):
        """Returns a tree node generated by a randomly selected template
        from the set of templates with a valid start-label at its root
        
        >>> from test_materials import RTF, TS
        >>> RTF.clear()
        >>> RTF._sort_templates(TS, [float, int])
        >>> tstrs = [
        ...     "([float]<POW>([float])([int]))",
        ...     "([float]<TERN_FLOAT>([bool])([float])([float]))",
        ...     "([float]<SUM>([int])([int]))",
        ...     "([float]<SUM>([float])([float]))",
        ...     "([int]<INT_SUM>([int])([int]))"
        ... ]
        >>> tstrs_cts = {tstr: 0 for tstr in tstrs}
        >>> n = 100000
        >>> for _ in range(n):
        ...     tstrs_cts[str(RTF.rand_start())] += 1
        >>> for tstr in tstrs:
        ...     ct = tstrs_cts[tstr]
        ...     assert abs(ct-(n/len(tstrs))) < 1000, f'{tstr}: {ct}'
        >>> tstrs_cts = {tstr: 0 for tstr in tstrs}
        >>> n = 100000
        >>> for __ in range(n):
        ...     tstrs_cts[str(RTF.rand_start(float))] += 1
        >>> for tstr in tstrs[:-1]:
        ...     ct = tstrs_cts[tstr]
        ...     assert abs(ct-(n/(len(tstrs)-1))) < 1000, f'{tstr}: {ct}'
        """
        if label:
            if label in self._roots:
                return self.np_random.choice(self._roots[label])(self.treebank)
            raise KeyError(f'No start NTTemplate found with root {label}')
        if not self._all_starts:
            self._all_starts = reduce(lambda a, b: a+b, self._roots.values(), [])
        return self.np_random.choice(self._all_starts)(self.treebank)
    
    def rand_inner(self, label: type):
        """Returns a tree node generated by a randomly selected template
        from the set of non-terminal templates
        
        >>> from test_materials import RTF, TS
        >>> RTF.clear()
        >>> RTF._sort_templates(TS, [float, int])
        >>> tstrs_f = [
        ...     "([float]<POW>([float])([int]))",
        ...     "([float]<TERN_FLOAT>([bool])([float])([float]))",
        ...     "([float]<SUM>([int])([int]))",
        ...     "([float]<SUM>([float])([float]))",
        ... ]
        >>> tstr_i = "([int]<INT_SUM>([int])([int]))"
        >>> tstrs_b = [
        ...     "([bool]<EQ>([float])([float]))",
        ...     "([bool]<EQ>([int])([int]))"
        ... ]
        >>> n = 100000
        >>> for tstrs, ty in ((tstrs_f, float), (tstrs_b, bool)):
        ...     tstrs_cts = {tstr: 0 for tstr in tstrs}
        ...     for __ in range(n):
        ...         tstrs_cts[str(RTF.rand_inner(ty))] += 1
        ...     for tstr in tstrs:
        ...         ct = tstrs_cts[tstr]
        ...         assert abs(ct-(n/len(tstrs))) < 1000, f'{tstr}: {ct}'
        >>> for _ in range(10):
        ...     t = str(RTF.rand_inner(int))
        ...     assert t == "([int]<INT_SUM>([int])([int]))"
        """
        if label in self._roots or label in self._inners:
            return self.np_random.choice(
                self._roots.get(label, []) + 
                self._inners.get(label, [])
            )(self.treebank)
        raise KeyError(f'No NTTemplate found with root {label}')
    
    def rand_leaf(self, label: type):
        """Returns a tree node generated by a randomly selected template
        from the set of leaf-node templates 
        
        >>> from test_materials import RTF, TS
        >>> RTF.clear()
        >>> RTF._sort_templates(TS, [float, int])
        >>> total = 0
        >>> n = 20000
        >>> for _ in range(n):
        ...     total += RTF.rand_leaf(float)(x=10.0)
        >>> assert abs(total-(n*6)) < 2000, total
        >>> for _ in range(10):
        ...     assert RTF.rand_leaf(int)() == 7
        >>> total = 0
        >>> for _ in range(n):
        ...     total += RTF.rand_leaf(bool)(x=10.0)
        >>> assert abs(total-(n*0.5)) < 250, total
        """
        if label in self._consts or label in self._vars:
            return self.np_random.choice(
                self._consts.get(label, []) + 
                self._vars.get(label, [])
            )(self.treebank)
        raise KeyError(f'No leaf template found with root {label}')
    
    def rand_var_else_const(self, label: type, reset=False):
        """Returns a tree node generated by a randomly selected template
        from the set of leaf-node templates 
        
        >>> from test_materials import RTF, _opf
        >>> RTF.clear()
        >>> ts = [
        ...     RTF.NTTemplate(float, _opf('SUM'), [float, float]),
        ...     RTF.ConstTemplate(float, lambda: 2.0),
        ...     RTF.ConstTemplate(float, lambda: 6.0),
        ...     RTF.VarTemplate(float, 'x'),
        ...     RTF.VarTemplate(float, 'y')
        ... ]
        >>> RTF._sort_templates(ts, [float])
        >>> totals = [0]*5
        >>> n = 20000
        >>> for _ in range(n):
        ...     for i in range(5):
        ...         totals[i] += RTF.rand_var_else_const(float, reset=not i)(x=10.0, y=20.0)
        >>> assert abs(totals[0]-(n*15)) < 3000, totals[0]
        >>> assert abs(totals[1]-(n*15)) < 3000, totals[1]
        >>> assert abs(totals[2]-(n*4)) < 2000, totals[2]
        >>> assert abs(totals[3]-(n*4)) < 2000, totals[3]
        >>> assert abs(totals[4]-(n*9.5)) < 3000, totals[4]
        """
        if reset:
            self._leaf_queues = {}
        if label not in self._leaf_queues:
            vars = self._vars.get(label, []).copy()
            consts = self._consts.get(label, []).copy()
            self.np_random.shuffle(vars) 
            self.np_random.shuffle(consts)
            self._leaf_queues[label] = list(
                vars + consts
            )
        if self._leaf_queues[label]:
            return self._leaf_queues[label].pop(0)(self.treebank)
        else:
            return self.rand_leaf(label)
        

    def _validate_templates(self,
            templates: list[NTTemplate|VarTemplate|ConstTemplate],
            start_types: set[type]):
        """

        >>> from test_materials import GP2
        >>> rtf = RandomTreeFactory([], [], ops.OperatorFactory(), treebank=GP2)
        >>> f2ff = rtf.NTTemplate(float, rtf.op_fac('SUM'), [float, float])
        >>> i2ii = rtf.NTTemplate(int, rtf.op_fac('INT_SUM'), [int, int])
        >>> fx = rtf.VarTemplate(float, 'x')
        >>> iy = rtf.VarTemplate(int, 'y')
        >>> sorted([str(vt) for vt in rtf._validate_templates([f2ff, fx], [float])])
        ['SUM(float, float) -> float', 'Var $x: float']
        >>> sorted([str(vt) for vt in rtf._validate_templates([i2ii, iy], [int])])
        ['INT_SUM(int, int) -> int', 'Var $y: int']
        >>> sorted([str(vt) for vt in rtf._validate_templates([i2ii, iy, f2ff, fx], [int, float])])
        ['INT_SUM(int, int) -> int', 'SUM(float, float) -> float', 'Var $x: float', 'Var $y: int']
        >>> rtf._validate_templates([f2ff, i2ii, iy], [int, float])
        Traceback (most recent call last):
        ...
        AttributeError: Some substitution site labels (float) appear nowhere in the roots of Terminals, and so cannot be completed with a Terminal, even if the maximum tree size, or maximum branch depth has been reached.
        >>> rtf._validate_templates([f2ff, i2ii, iy], [float])
        Traceback (most recent call last):
        ...
        AttributeError: Some substitution site labels (float) appear nowhere in the roots of Terminals, and so cannot be completed with a Terminal, even if the maximum tree size, or maximum branch depth has been reached.
        >>> rtf._validate_templates([f2ff, fx, iy], [float])
        Traceback (most recent call last):
        ...
        AttributeError: The following problems were observed with the template set:
        Some node labels (int) correspond to no labels on substitution sites/start sites, so the nodes with these labels cannot be placed anywhere.
        Some terminal node labels (int) appear nowhere in the substitution sites of NonTerminals, and so either cannot be placed, or can only be placed at a start site, making a depth-1 tree.
        >>> rtf._validate_templates([i2ii, iy], [float])
        Traceback (most recent call last):
        ...
        AttributeError: The following problems were observed with the template set:
        You have some start labels (float) which are  not the roots of any NonTerminal, and so either can only form a depth-1 tree, or none at all.
        Some labels on substitution sites/start sites (float) correspond to no node labels, no no node can be placed there.
        >>> rtf._validate_templates([i2ii, fx], [float])
        Traceback (most recent call last):
        ...
        AttributeError: The following problems were observed with the template set:
        You have some start labels (float) which are  not the roots of any NonTerminal, and so either can only form a depth-1 tree, or none at all.
        Some terminal node labels (float) appear nowhere in the substitution sites of NonTerminals, and so either cannot be placed, or can only be placed at a start site, making a depth-1 tree.
        Some substitution site labels (int) appear nowhere in the roots of Terminals, and so cannot be completed with a Terminal, even if the maximum tree size, or maximum branch depth has been reached.
        """
        start_types = set(start_types)
        child_types = set()
        nt_root_types = set()
        t_root_types = set()
        nts = []
        ts = []
        for tt in templates:
            if isinstance(tt, (self.VarTemplate, self.ConstTemplate)):
                t_root_types.add(tt.root)
                nts.append(tt)
            if isinstance(tt, self.NTTemplate):
                nt_root_types.add(tt.root)
                child_types.update(tt.children)
                ts.append(tt)
        problems = []
        if not start_types.issubset(nt_root_types):
            xs_starts = start_types - (start_types & nt_root_types)
            problems.append(
                "You have some start labels (" +
                f"{', '.join([x.__name__ for x in xs_starts])}) which are  " +
                "not the roots of any NonTerminal, and so either can only " +
                "form a depth-1 tree, or none at all."
            )
        if (start_types | child_types) != (t_root_types | nt_root_types):
            intersect = (start_types | child_types) & (t_root_types | nt_root_types)
            spare_subsites = (start_types | child_types) - intersect
            spare_nodes = (t_root_types | nt_root_types) - intersect
            if spare_subsites:
                problems.append(
                    "Some labels on substitution sites/start sites (" +
                    f"{', '.join([x.__name__ for x in spare_subsites])}) " +
                    "correspond to no node labels, no no node can be placed there."
                )
            if spare_nodes:
                problems.append(
                    "Some node labels (" +
                    f"{', '.join([x.__name__ for x in spare_nodes])}) " +
                    "correspond to no labels on substitution sites/start" +
                    " sites, so the nodes with these labels cannot be " +
                    "placed anywhere."
                )
        spare_trts = t_root_types - (child_types & t_root_types)
        if spare_trts:
            problems.append(
                "Some terminal node labels (" +
                f"{', '.join([x.__name__ for x in spare_trts])}" +
                ") appear nowhere in the substitution sites of NonTerminals, " +
                "and so either cannot be placed, or can only be placed at a " + 
                "start site, making a depth-1 tree."
            )
        spare_cts = child_types - (child_types & t_root_types)
        if spare_cts:
            problems.append(
                "Some substitution site labels (" +
                f"{', '.join([x.__name__ for x in spare_cts])}" +
                ") appear nowhere in the roots of Terminals, " +
                "and so cannot be completed with a Terminal, " +
                "even if the maximum tree size, or maximum branch" +
                " depth has been reached."
            )
        if len(problems)==1:
            raise AttributeError(problems[0])
        elif len(problems)>1:
            raise AttributeError(
                "The following problems were observed with the template set:\n" +
                ("\n".join(problems))
            )
        return templates
    
    @property
    def prefix(self):
        return 'rand'
    
    @property
    def op_set(self)->set[Operator]:
        """Returns all the operators in the factory's templates
        
        >>> from test_materials import RTF, TS
        >>> RTF.clear()
        >>> RTF._sort_templates(TS, [float, int])
        >>> print(', '.join(sorted([str(s) for s in list(RTF.op_set)])))
        <EQ>, <INT_SUM>, <POW>, <SUM>, <TERN_FLOAT>
        """
        if not self._op_set:
            self._op_set = set(
                [t.operator for ts in self._roots.values() for t in ts] +
                [t.operator for ts in self._inners.values() for t in ts]
            )
        return self._op_set

    @property
    def treebank(self):
        return self._treebank
    
    @treebank.setter
    def treebank(self, treebank: "GPTreebank"=None):
        self._treebank = treebank
        if treebank is not None:
            self.seed = treebank.np_random
            self.max_size = treebank.max_size
            self.max_depth = treebank.max_depth

    def __call__(self, 
            *vars, 
            treebank: Treebank=None, 
            label: type=None
        ) -> Tree:
        """

        >>> from test_materials import GP2, _opf
        >>> GP2.clear()
        >>> GP2.max_size = 7
        >>> GP2.max_depth = 3
        >>> rtf = RandomTreeFactory(
        ...     [
        ...         (float, 'SUM', [float, float]),
        ...         (float, 'TERN_FLOAT', [bool, float, float]),
        ...         (bool, 'EQ', [int, int]),
        ...         (int, lambda: 6),
        ...         (float, lambda: 66.0),
        ...         (bool, lambda: True),
        ...         (float, 'x')
        ...     ],
        ...     [float],
        ...     _opf, 
        ...     treebank=GP2
        ... )
        >>> hist = {}
        >>> n = 384000
        >>> for _ in range(n):
        ...     t = rtf()
        ...     assert t.size() <= GP2.max_size, f"{t} size: {t.size()}"
        ...     assert t.depth() == GP2.max_depth, f"{t} depth: {t.depth()}"
        ...     y = int(t(x=5.0))
        ...     hist[y] = hist.get(y, 0)+1
        >>> expected = {
        ...     20: 3000, 81: 12000, 142: 18000, 203: 12000, 264: 3000,
        ...     15: 6000, 76: 18000, 137: 18000, 198: 6000,
        ...     10: 32000, 71: 64000, 132: 32000,
        ...     5: 80000, 66: 80000
        ... }
        >>> set(hist)==set(expected)
        True
        >>> rmse = (sum([(hist[k]-expected[k])**2 for k in hist.keys()])/len(hist))**0.5
        >>> assert rmse<400
        """
        t = self.rand_start(label=label)
        subsites = t.get_all_substitution_sites()
        while subsites and t.size() < self.max_size:
            ss = subsites[self.np_random.integers(len(subsites))]
            lab = ss.label.class_id
            if (lab in self._roots or lab in self._inners) and (ss.at_depth()+1 < self.max_depth):
                new_node = self.rand_inner(lab)
            else:
                new_node = self.rand_var_else_const(lab)
            # if the new node would cause a size overshoot, break.
            # This will result in a small undershoot, but that's fine
            if len(new_node) + t.size() > self.max_size:
                # Don't leave an unused new_node lingering in the
                # treebank
                new_node.delete()
                break
            ss.perform_substitution(new_node)
            subsites = t.get_all_substitution_sites()
        while subsites:
            ss = subsites[self.np_random.integers(len(subsites))]
            ss.perform_substitution(self.rand_var_else_const(ss.label.class_id))
            subsites = t.get_all_substitution_sites()
        return t
    
class RandomAlgebraicTreeFactory(RandomTreeFactory):
    def __init__(self,
            *args, 
            treebank: Treebank=None,
            **kwargs): 
        templates = (
            (float, 'SUM', (float, float)),
            (float, 'PROD', (float, float)),
            (float, 'POW', (float, int)),
            (float, 'x'),
            (float, lambda: self.np_random.normal(0, 0.1)),
            (int, lambda: self.np_random.integers(10))
        )
        root_types = [float] 
        operator_factory = ops.OperatorFactory()
        super().__init__(
            templates, 
            root_types, 
            operator_factory, 
            treebank,
            *args,
            **kwargs
        )
        

def main():
    import doctest
    doctest.testmod()
        


if __name__ == '__main__':
    main()
