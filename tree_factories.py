from typing import Protocol, Type, Union
import operators as ops
from trees import Terminal, NonTerminal, Tree
from gp_trees import GPTerminal, GPNonTerminal
import pandas as pd
from itertools import chain, combinations
from random import uniform, choices
from functools import reduce
from dataclasses import dataclass
from type_ify import TypeNativiser

class TreeFactory(Protocol):
    def __init__(self, treebank, *args, **kwargs):
        ...

    def __call__(self, *args, **kwargs) -> Tree:
        ...

class TestTreeFactory:
    def __init__(self, treebank, start_tree: Tree):
        self.start_tree = start_tree
        self.treebank = treebank

    def __call__(self, *args, **kwargs) -> Tree:
        return self.start_tree.copy_out(self.treebank)

class RandomPolynomialFactory:
    def __init__(
                self,
                treebank,
                order: int = None,
                const_min: float = None,
                const_max: float = None):
        self.treebank = treebank
        self.T = treebank.T
        self.N = treebank.N
        self.order = order
        self.const_min = const_min
        self.const_max = const_max
        self.operators = {
            "SUM": ops.SUM,
            "PROD": ops.PROD,
            "SQ": ops.SQ,
            "CUBE": ops.CUBE,
            "POW": ops.POW
        }

    def _all_sums(self, tot, k): # do a video on this
        if tot==k:
            return ((1,) * k,)
        elif k==1:
            return ((tot,),)
        else:
            sums = ()
            first = tot - (k-1)
            while first > 0:
                tails = self._all_sums(tot-first, k-1)
                t = tails[0]
                sums += tuple((first,) + tail for tail in tails)
                first -= 1
            return sums

    def _poly_terms(self, vars_, order):
        combos = tuple(chain.from_iterable(combinations(vars_, r) for r in range(order+1)))
        terms = []
        for n in range(order+1):
            for c in filter(lambda combo: len(combo) <= n, combos):
                if len(c):
                    terms += [(c, powers) for powers in self._all_sums(n, len(c))]
                elif not n:
                    terms.append(((), ()))
        return tuple(terms)

    def _binarise_tree(self, oper80r, tree_list, start=None):
        if start:
            return reduce(lambda t1, t2: self.N(self.treebank, float, t1, t2, operator=self.operators[oper80r]), tree_list, start)
        else:
            return reduce(lambda t1, t2: self.N(self.treebank, float, t1, t2, operator=self.operators[oper80r]), tree_list)


    def __call__(self, vars_: pd.DataFrame, *args, **kwargs) -> Tree:
        term_subtrees = []
        for term in self._poly_terms(vars_, self.order):
            b = self.T(self.treebank, float, uniform(self.const_min, self.const_max))
            var_pows = []
            for var, pow in zip(term[0], term[1]):
                if pow == 1:
                    var_pows.append(self.T(self.treebank, float, vars_[var]))
                elif pow == 2:
                    var_pows.append(self.N(self.treebank, float, self.T(self.treebank, float, vars_[var]), operator=self.operators['SQ']))
                elif pow == 3:
                    var_pows.append(self.N(self.treebank, float, self.T(self.treebank, float, vars_[var]), operator=self.operators['CUBE']))
                else:
                    var_pows.append(self.N(self.treebank, float, self.T(self.treebank, float, vars_[var]), self.T(self.treebank, int, pow), operator=self.operators['POW']))
            term_subtrees.append(self._binarise_tree('PROD', var_pows, start=b))
        return self._binarise_tree('SUM', term_subtrees)

class RandomTreeFactory:
    tn = TypeNativiser()

    @dataclass
    class TreeTemplate:
        root: type
        operator: ops.Operator
        child_types: tuple[type]
        var_data: Union[pd.Series, None]

        def _is_valid(self):
            return self.operator._type_seq_legal(*self.child_types) and (
                self.var_data
                if isinstance(RandomTreeFactory.tn.type_ify(self.var_data), self.root)
                else isinstance(self.operator.return_type, self.root)
            )

        def __str__(self):
            """Describes the subtree represented by the TreeTemplate in function
            notation, showing the parameter types and return type.

            XXX FIXME

            Exception raised:
                Traceback (most recent call last):
                File "/Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/doctest.py", line 1348, in __run
                    exec(compile(example.source, filename, "single",
                File "<doctest __main__.RandomTreeFactory.TreeTemplate.__str__[0]>", line 1, in <module>
                    tt = RandomTreeFactory.TreeTemplate(root=float, operator=ops.SUM, child_types = (float, int, float))
                TypeError: RandomTreeFactory.TreeTemplate.__init__() missing 1 required positional argument: 'var_data'

            >>> tt = RandomTreeFactory.TreeTemplate(root=float, operator=ops.SUM, child_types = (float, int, float))
            >>> print(tt)
            SUM(a0: float, a1: int, a2: float) -> float
            """
            if self.var_data:
                return f"({self.root.__name__}: {self.var_data.name})"
            param_str = ", ".join([f"a{i}: {c.__name__!s}" for i, c in enumerate(self.child_types)])
            return f"{self.operator.name}({param_str}) -> {self.root.__name__!s}"

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
            templates: list[TreeTemplate],
            root_types: set[type]):
        parent_types = set()
        for tt in self.templates:
            if not tt._is_valid():
                raise AttributeError(f"{tt!s} is not a valid subtree template")
            parent_types.add(tt.root)
            root_types.update(tt.child_types)
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

    def __call__(self, vars_: pd.DataFrame, *args, **kwargs) -> Tree:
        ...

def main():
    import doctest
    doctest.testmod()

if __name__ == '__main__':
    main()
