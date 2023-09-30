from typing import Protocol, Type, Union, Iterable
import operators as ops
from trees import Terminal, NonTerminal, Tree
from gp_trees import GPTerminal, GPNonTerminal
import pandas as pd
from itertools import chain, combinations
from random import uniform, choices
from functools import reduce
from dataclasses import dataclass
from type_ify import TypeNativiser
from observatories import GenFunc
from treebanks import Treebank

class TreeFactory(Protocol):
    def set_treebank(self, treebank: Treebank):
        ...

    def __call__(self, *vars: Iterable[str], treebank: Treebank=None) -> Tree:
        ...


class TestTreeFactory:
    def __init__(self, start_tree: Tree):
        self.start_tree = start_tree
        self.treebank = start_tree.treebank

    def set_treebank(self, treebank: Treebank):
        self.treebank = treebank

    def __call__(self, *vars: Iterable[str], treebank: Treebank=None) -> Tree:
        return self.start_tree.copy_out(treebank if treebank else self.treebank)
    
class CompositeTreeFactory:
    def __init__(self, 
            treebank: Treebank=None, 
            tree_factories: list[TreeFactory], 
            weights: Iterable[float|int], 
            *args, **kwargs):
        if weights and len(weights) != len(tree_factories):
            raise ValueError(
                "tree_factories and weights should be the same length"
            )
        self.tree_factories = tree_factories
        self.weights = weights
        self.set_treebank(treebank)

    def set_treebank(self, treebank: Treebank):
        for tf in self.tree_factories:
            tf.treebank = treebank

    def __call__(self, *vars: Iterable[str], treebank: Treebank=None) -> Tree:
        return choices(
            self.tree_factories, self.weights)[0](vars, 
            treebank = treebank if treebank else self.treebank
        )

class RandomPolynomialFactory:
    """A TreeFactory which makes random trees representing polynomials of a specified
    order (`order`) and set of variables (`vars`), with random coefficients uniformly 
    distributed in a specified range (`const_min` to `const_max`)
    """
    def __init__(
            self,
            treebank: Treebank = Non,
            order: int = None,
            const_min: float = None,
            const_max: float = None):
        if treebank:
            self.set_treebank(treebank)
        self.const_min = const_min
        self.const_max = const_max
        self.operators = {
            "SUM": ops.SUM,
            "PROD": ops.PROD,
            "SQ": ops.SQ,
            "CUBE": ops.CUBE,
            "POW": ops.POW
        }
        self.treebank.operators = self.operators
        self.order = order
        

    def _poly_terms(self, vars_, order) -> tuple[tuple[str|int]]:
        """Generates a tuple of tuples, in which each tuple represents a term of 
        the polynomial. Each tuple representing a term is itself comprised of 
        two tuples, the first listing the variables of the term, and the second
        representing the powers they are to be raised to

        >>> from treebanks import TypeLabelledTreebank
        >>> rpf = RandomPolynomialFactory(TypeLabelledTreebank(), 3, -10.0, 10.0)
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
        >>> rpf = RandomPolynomialFactory(TypeLabelledTreebank(), 3, -10.0, 10.0)
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
            start: Tree=None, 
            nt: NonTerminal=None
        ) -> Tree:
        """Takes a list of trees and combines them by repeatedly applying an 
        operator pairwise. Thus, combining trees `A`, `B`, `C`, `D`, and `E` 
        would give a tree `(((A + B) + C) + D) + E)`

        Parameters
        ==========  
            op_name (str): Name of operator
            tree_list (list[Tree]): List of trees to be combined
            start (Tree or None): An aditional tree may be included to be 
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
        >>> rpf = RandomPolynomialFactory(tlt, 3, -10.0, 10.0)
        >>> start = tlt.tree('([float]0.0)')
        >>> tl = [tlt.tree('([float]1.0)'), tlt.tree('([float]2.0)'), tlt.tree('([float]3.0)'), tlt.tree('([float]4.0)')]
        >>> t0 = rpf._binarise_tree('SUM', tl)
        >>> t0
        tree("([float]<SUM>([float]<SUM>([float]<SUM>([float]1.0)([float]2.0))([float]3.0))([float]4.0))")
        >>> t1 = rpf._binarise_tree('SUM', tl, start)
        >>> t1
        tree("([float]<SUM>([float]<SUM>([float]<SUM>([float]<SUM>([float]0.0)([float]1.0))([float]2.0))([float]3.0))([float]4.0))")
        >>> t2 = rpf._binarise_tree('PROD', tl)
        >>> t2
        tree("([float]<PROD>([float]<PROD>([float]<PROD>([float]1.0)([float]2.0))([float]3.0))([float]4.0))")
        >>> t3 = rpf._binarise_tree('PROD', tl, start)
        >>> t3
        tree("([float]<PROD>([float]<PROD>([float]<PROD>([float]<PROD>([float]0.0)([float]1.0))([float]2.0))([float]3.0))([float]4.0))")
        """
###==#----#====#----##===#----#====#----###==#----#====#----##===#----#====#---+
        # if start:
        N = nt if nt else self.N
        if op_name in self.operators:
            args = [start] if start else []
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
            raise ValueError('op_name must be in the tree factory\'s operators dict')
        # else:
        #     return reduce(
        #         lambda t1, t2: N(
        #             self.treebank, 
        #             float, 
        #             t1, 
        #             t2, 
        #             operator=self.operators[op_name]
        #         ), 
        #         tree_list
        #     )

    def set_treebank(self, treebank: Treebank):
        self.treebank = treebank
        self.T = treebank.T
        self.N = treebank.N

    def __call__(self, *vars: Iterable[str], treebank: Treebank=None) -> Tree:
        term_subtrees = []
        treebank = treebank if treebank else self.treebank
        for term in self._poly_terms(vars, self.order):
            b = treebank.T(self.treebank, float, uniform(self.const_min, self.const_max))
            var_pows = []
            for var, pow in zip(term[0], term[1]):
                if pow == 1:
                    var_pows.append(treebank.T(self.treebank, float, f'${var}'))
                elif pow == 2:
                    var_pows.append(treebank.N(self.treebank, float, treebank.T(self.treebank, float, f'${var}'), operator=self.operators['SQ']))
                elif pow == 3:
                    var_pows.append(treebank.N(self.treebank, float, treebank.T(self.treebank, float, f'${var}'), operator=self.operators['CUBE']))
                else:
                    var_pows.append(treebank.N(self.treebank, float, treebank.T(self.treebank, float, f'${var}'), treebank.T(self.treebank, int, pow), operator=self.operators['POW']))
            term_subtrees.append(self._binarise_tree('PROD', var_pows, start=b))
        return self._binarise_tree('SUM', term_subtrees, nt=N)

# class RandomTreeFactory:
#     tn = TypeNativiser()

#     @dataclass
#     class TreeTemplate:
#         root: type
#         operator: ops.Operator
#         child_types: tuple[type]
#         leaf: GenFunc|str

#         def _is_valid(self):
#             return self.operator._type_seq_legal(*self.child_types) and (
#                 self.var_data
#                 if isinstance(RandomTreeFactory.tn.type_ify(self.var_data), self.root)
#                 else isinstance(self.operator.return_type, self.root)
#             )

#         def __str__(self):
#             """Describes the subtree represented by the TreeTemplate in function
#             notation, showing the parameter types and return type.

#             XXX FIXME

#             Exception raised:
#                 Traceback (most recent call last):
#                 File "/Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/doctest.py", line 1348, in __run
#                     exec(compile(example.source, filename, "single",
#                 File "<doctest __main__.RandomTreeFactory.TreeTemplate.__str__[0]>", line 1, in <module>
#                     tt = RandomTreeFactory.TreeTemplate(root=float, operator=ops.SUM, child_types = (float, int, float))
#                 TypeError: RandomTreeFactory.TreeTemplate.__init__() missing 1 required positional argument: 'var_data'

#             >>> tt = RandomTreeFactory.TreeTemplate(root=float, operator=ops.SUM, child_types = (float, int, float))
#             >>> print(tt)
#             SUM(a0: float, a1: int, a2: float) -> float
#             """
#             if self.var_data:
#                 return f"({self.root.__name__}: {self.var_data.name})"
#             param_str = ", ".join([f"a{i}: {c.__name__!s}" for i, c in enumerate(self.child_types)])
#             return f"{self.operator.name}({param_str}) -> {self.root.__name__!s}"

#     def __init__(self,
#             treebank, templates, root_label, max_size, *args,
#             weights=None, **kwargs):
#         if templates and len(templates) != len(weights):
#             raise AttributeError("If you provide a RandomTreeFactory with " +
#                 "weights, the weight list must be the same length as the list" +
#                 " of tree templates.")
#         self.templates = self._validate_templates(templates)
#         self.weights = weights
#         self.max_size = max_size

#     def _validate_templates(self,
#             templates: list[TreeTemplate],
#             root_types: set[type]):
#         parent_types = set()
#         for tt in self.templates:
#             if not tt._is_valid():
#                 raise AttributeError(f"{tt!s} is not a valid subtree template")
#             parent_types.add(tt.root)
#             root_types.update(tt.child_types)
#         if root_types == parent_types:
#             return templates
#         intersect = root_types & parent_types
#         problems = []
#         excess_roots = root_types - intersect
#         if excess_roots:
#             problems.append(
#                 "Union(S, C) of this template set includes some values (" +
#                 f"{', '.join([x for x in excess_roots])}) that are absent " +
#                 "P, which therefore cannot be completed"
#             )
#         excess_parents = parent_types - intersect
#         if excess_parents:
#             problems.append(
#                 "P of this template set includes some values (" +
#                 f"{', '.join([x for x in excess_parents])}) that are absent " +
#                 "Union(S, C), which therefore cannot be placed"
#             )
#         raise AttributeError("The union of the set S of start nodes for trees" +
#             " and the set C of child nodes of subtree templates defines " +
#             "the set of places tree templates can be inserted, and the " +
#             "set P of parent nodes defines the set of subtrees that can " +
#             "fill the places in Union(S, C): however, " +
#             f"{': and '.join([x for x in problems])}.")

#     def __call__(self, *vars: Iterable[str]) -> Tree:
#         ...

def main():
    import doctest
    doctest.testmod()

if __name__ == '__main__':
    main()
