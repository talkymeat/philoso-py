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

T = TypeVar('T')
Var:   TypeAlias = tuple[type, str]
Const: TypeAlias = tuple[type[T], Callable[[], T]]

class TreeFactory(ABC):
    @abstractmethod
    def set_treebank(self, treebank: Treebank):
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

    def set_treebank(self, treebank: Treebank):
        self.treebank = treebank

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
            self.set_treebank(treebank)

    def set_treebank(self, treebank: Treebank):
        for tf in self.tree_factories:
            tf.set_treebank(treebank)

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
        param_dict = {f"{tf.prefix}_weight": weight for tf, weight in zip (self.tree_factories, self.weights)}
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
            seed: int|list[int]|None = None # XXX get rid of this default
        ):
        self.seed = seed
        if params is not None:
            self.order = int(params[0])
            self.const_min = params[1]
            self.const_max = params[2]
        else:
            self.order = 3
            self.const_min = -1.0
            self.const_max = 1.0
        if self.const_min > self.const_max:
            self.const_min, self.const_max = self.const_max, self.const_min
        if treebank:
            self.set_treebank(treebank)
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

        >>> from treebanks import TypeLabelledTreebank
        >>> rpf = RandomPolynomialFactory(params = np.array([3., -10.0, 10.0], dtype=np.float32), treebank=TypeLabelledTreebank())
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

        >>> from treebanks import TypeLabelledTreebank
        >>> rpf = RandomPolynomialFactory(params = np.array([3., -10.0, 10.0], dtype=np.float32), treebank=TypeLabelledTreebank())
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
        
        >>> from treebanks import TypeLabelledTreebank
        >>> tlt = TypeLabelledTreebank(operators=[
        ...     ops.SUM, ops.PROD, ops.SQ, ops.CUBE, ops.POW
        ... ])
        >>> rpf = RandomPolynomialFactory(params = np.array([3., -10.0, 10.0], dtype=np.float32), treebank=tlt)
        >>> initial = tlt.tree('([float]0.0)')
        >>> tl = [tlt.tree('([float]1.0)'), tlt.tree('([float]2.0)'), tlt.tree('([float]3.0)'), tlt.tree('([float]4.0)')]
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

    def set_treebank(self, treebank: Treebank):
        self.treebank = treebank
        self.T = treebank.T
        self.N = treebank.N

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
        >>> gp = GPTreebank()
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
        for term in self._poly_terms(vars, self.order):
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
        return self._binarise_tree('SUM', term_subtrees, nt=treebank.N)
    
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
        operator: ops.Operator
        root: type
        children: Sequence[type]
        treebank: Treebank

        def __post_init__(self):
            """Checks that the TreeTemplate is typesafe with the Operator.

            >>> from gp import GPTreebank
            >>> gptb = GPTreebank()
            >>> RandomTreeFactory.NTTemplate(root=tuple, operator=ops.ID, children=(float, int, str), treebank=gptb)
            (float, int, str) -> tuple
            >>> RandomTreeFactory.NTTemplate(root=str, operator=ops.SUM, children=(int, int), treebank=gptb)
            Traceback (most recent call last):
                ...
            TypeError: root: str and children: (int, int) is not a valid template for Operator <SUM>
            >>> RandomTreeFactory.NTTemplate(root=int, operator=ops.SUM, children=(str, int), treebank=gptb)
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
            >>> tt = RandomTreeFactory.NTTemplate(root=float, operator=ops.SUM, children = (float, int, float), treebank=gptb)
            >>> print(tt)
            SUM(float, int, float) -> float
            """
            param_str = ", ".join([c.__name__ for c in self.children])
            return f"{self.operator.name}({param_str}) -> {self.root.__name__!s}"
        
        def __repr__(self):
            return str(self)
        
        def __call__(self):
            """Generates a depth-1 subtree"""
            return GPNonTerminal(
                self.treebank, 
                self.root, 
                *[
                    SubstitutionSite(self.treebank, child, i) 
                    for i, child 
                    in enumerate(self.children)
                ]
            )
        
    @dataclass
    class ConstTemplate:
        root: type[T]
        genfunc: Callable[[], T]
        treebank: Treebank

        def __call__(self):
            """
            
            >>> from gp import GPTreebank
            >>> gf1 = lambda: 2
            >>> gf2 = lambda: 2.0
            >>> gf3 = lambda: 'bollocks'
            >>> gptb = GPTreebank()
            >>> ct1 = RandomTreeFactory.ConstTemplate(int, gf1, gptb)
            >>> ct1()
            tree("([int]2)")
            """
            x = self.genfunc()
            if isinstance(x, self.root):
                return GPTerminal(self.treebank, self.root, x)
            raise TypeError(
                f"genfunc for ConstTemplate {self.root.__name__} returned " +
                f"{x}, which is not a {self.root.__name__}"
            )
        
        def __str__(self):
            return f"Const k: {self.root}"
        
    @dataclass
    class VarTemplate:
        root: type[T]
        name: str
        treebank: Treebank

        def __call__(self):
            return GPTerminal(self.treebank, self.root, f'${self.name}')
        
        def __str__(self):
            return f"Var ${self.name}: {self.root}"


    def __init__(self,
            treebank, templates, root_label, max_size, *args,
            weights=None, **kwargs):
        if templates and len(templates) != len(weights):
            raise AttributeError("If you provide a RandomTreeFactory with " +
                "weights, the weight list must be the same length as the list" +
                " of tree templates.")
        self.templates = self._validate_templates(templates)
        self.weights = weights
        self.max_size = max_size

    def _validate_templates(self,
            templates: list[NTTemplate],
            root_types: set[type]):
        parent_types = set()
        for tt in self.templates:
            if not tt._is_valid():
                raise AttributeError(f"{tt!s} is not a valid subtree template")
            parent_types.add(tt.root)
            root_types.update(tt.children)
        if root_types == parent_types:
            return templates
        intersect = root_types & parent_types
        problems = []
        excess_roots = root_types - intersect
        if excess_roots:
            problems.append(
                "Union(S, C) of this template set includes some values (" +
                f"{', '.join([x for x in excess_roots])}) that are absent " +
                "P, which therefore cannot be completed"
            )
        excess_parents = parent_types - intersect
        if excess_parents:
            problems.append(
                "P of this template set includes some values (" +
                f"{', '.join([x for x in excess_parents])}) that are absent " +
                "Union(S, C), which therefore cannot be placed"
            )
        raise AttributeError("The union of the set S of start nodes for trees" +
            " and the set C of child nodes of subtree templates defines " +
            "the set of places tree templates can be inserted, and the " +
            "set P of parent nodes defines the set of subtrees that can " +
            "fill the places in Union(S, C): however, " +
            f"{': and '.join([x for x in problems])}.")
    
    @property
    def prefix(self):
        return 'rand'
    
    @property
    def op_set(self)->set[Operator]:
        return set()

    def __call__(self, *vars: Iterable[str]) -> Tree:
        ...

def main():
    import doctest
    doctest.testmod()
        


if __name__ == '__main__':
    main()
