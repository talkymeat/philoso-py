from trees import Terminal, NonTerminal
from mutators import MutatorFactory, CrossoverMutator, Mutator, NullMutator
import pandas as pd
import numpy as np
from copy import copy
from tree_errors import OperatorError
from typing import TypeAlias
from size_depth import SizeDepth
from logtools import MiniLog
from typing import Any
from icecream import ic
import warnings
from copy import deepcopy
DEBUG = True

class _D:
    TYPES = {}

    # def f(self, val):
    #     return val # None 
    
    def __new__(cls, val):
        if isinstance(val, str):
            try:
                val=eval(val)
            except Exception:
                pass
        return cls.TYPES.get(type(val), lambda x: x)(val)
        # return val if _val is None else _val

class D64(_D):
    TYPES = {
        int: np.int64,
        float: np.float64,
        complex: np.complex128,
        str: np.bytes_,
        bool: np.bool_
    }

class D32(_D):
    TYPES = {
        int: np.int32,
        float: np.float32,
        complex: np.complex64,
        str: np.bytes_,
        bool: np.bool_
    }

class D16(_D): 
    """Note: there is no numpy dtype `complex32`; don't use D16 if you need 
    complex numbers
    """
    TYPES = {
        int: np.int16,
        float: np.float16,
        str: np.bytes_,
        bool: np.bool_
    }

D = D32

class GPNonTerminal(NonTerminal):
    """GPNonTerminals carry operators, and take valid return types of their operators
    as Label values. They can be called with arguments that are passed down to the 
    GPTerminals, which can be Constants or Variables. When the `kwargs`
    `dict`/`DataFrame` is passed down the tree, if a Variable has a name corresponding 
    to a key of `kwargs`, the corresponding value/`Series` will be passed back up the
    tree and operated on

    >>> from gp import GPTreebank
    >>> from test_materials import DummyTreeFactory
    >>> import operators as ops
    >>> op = [ops.SUM, ops.PROD, ops.SQ, ops.CUBE, ops.POW]
    >>> gp = GPTreebank(operators=op, tree_factory=DummyTreeFactory())
    >>> mewtwo = gp.tree("([float]<SUM>([float]<SQ>([int]$mu))([float]<SUM>([float]<PROD>([int]3)([int]$mu))([int]2)))")
    >>> mewtwo(mu=-2)
    0.0
    >>> mewtwo(mu=-1)
    0.0
    >>> mewtwo(mu=-3)
    2.0
    """
    cnt = 0
    def __init__(self, 
                treebank,
                label, 
                *children, 
                operator=None, 
                metadata: dict=None, 
                tmp: dict=None, 
                gp_operator: Mutator=None, 
                gp_child_op: Mutator=None):
        super().__init__(
            treebank, 
            label, 
            *children, 
            operator=operator, 
            metadata=metadata, 
            tmp=tmp
        )
        self.mutable = True
        self.gp_operator = gp_operator if gp_operator else NullMutator(treebank)
        self.xo_operator = gp_child_op if gp_child_op else CrossoverMutator(treebank)

    @property
    def is_valid(self):
        if not self._operator.is_valid(self.data_type, *[ch.label.data_type for ch in self]):
            raise OperatorError(
                f"the children and/or label of {self} are not a legal type-signature " +
                f"for its operator {self._operator.name}"
            )
        return True

    def copy_out(self, treebank = None, gp_copy=False, _at_depth=0, _max_size=None, **kwargs):
        """Generates a deep copy of a tree: the same structure, same Labels, and
        for the Terminals, same content: but each node a distinct object in
        memory from the corresponding node in the original. If a Treebank is
        passed to `treebank`, the new NonTerminal and its children will belong
        to `treebank`.

        If `treebank` is `None`, a new empty Treebank will be created, and the
        new NonTerminal will belong to that instead. This acts as a 'dummy
        treebank', in the case where trees are needed temporarily for some
        computation, without the overhead of storing them in an existing
        treebank.

        Parameters
        ----------
            treebank:
                The target treebank the tree is being copied into
            gp_copy (bool):
                If true, GP mutation and crossover operators will be applied
            sd (SizeDepth):
                A simple tracker which ensures GP crossover operators don't 
                make the output tree too big or too deep
            kwargs:
                Not used, but needed for compatibility with subclasses

        Returns
        -------
            NonTerminal: copy of original tree

        >>> from tree_factories import RandomPolynomialFactory
        >>> from gp import GPTreebank
        >>> from test_materials import DummyTreeFactory
        >>> import pandas as pd
        >>> import operators as ops
        >>> rng = np.random.Generator(np.random.PCG64())
        >>> gp = GPTreebank(
        ...     mutation_rate = 0.0, 
        ...     mutation_sd=0.0, 
        ...     crossover_rate=0.5, 
        ...     max_depth=70,
        ...     max_size=300, 
        ...     seed=rng,
        ...     operators=[ops.SUM, ops.PROD, ops.SQ, ops.POW, ops.CUBE], 
        ...     tree_factory=DummyTreeFactory()
        ... )
        >>> rpf = RandomPolynomialFactory(params = np.array([5, -10.0, 10.0], dtype=float), treebank=gp, seed=rng)
        >>> trees = [rpf('x', 'y') for _ in range(1)]
        >>> for t in trees:
        ...     t.meta_set_recursive(gen=0)
        >>> def test_msr(t):
        ...     assert 'gen' in t.metadata and t.metadata['gen']==0
        ...     if hasattr(t, 'children'):
        ...         for c in t:
        ...             test_msr(c)
        >>> for t in trees:
        ...     test_msr(t)
        >>> df = pd.DataFrame({'x': [1.0, 1.0], 'y': [1.0, 1.0]})
        >>> def incr_gen(t):
        ...     t.metadata['gen'] += 1
        >>> def _all_meta(t, g=None):
        ...     if hasattr(t, 'metadata'):
        ...         assert not t.metadata.get('__no_xo__', False)
        ...         if g is None:
        ...             g = t.metadata['gen']
        ...             print(g)
        ...         else:
        ...             assert g == t.metadata['gen']
        ...     if hasattr(t, 'children'):
        ...         for c in t:
        ...             _all_meta(c, g=g)
        >>> for _ in range(10):
        ...     new_trees = []
        ...     for t in trees:
        ...         _all_meta(t)
        ...         new_t = t.copy(gp_copy=True)
        ...         new_t.apply(incr_gen)
        ...         new_trees.append(new_t)
        ...     for t in trees:
        ...         t.delete()
        ...     for t in new_trees:
        ...         t.meta_set_recursive(__no_xo__=False)
        ...     trees = new_trees
        0
        1
        2
        3
        4
        5
        6
        7
        8
        9
        >>> for t in trees:
        ...     assert t.tree_map_reduce(max, map_any=lambda x: x.metadata['gen']) < 11
        """
        if gp_copy:
            return self.gp_copy_out(
                treebank=treebank, 
                gp_copy=gp_copy, 
                _at_depth=_at_depth, 
                _max_size=_max_size, 
                **kwargs
            )
        # If `treebank` is None...
        if not treebank:
            # create a dummy treebank, and then it won't be None. 
            treebank = self.treebank.__class__()
        copied_children = []
        for i, c in enumerate(self):
            c_copy = c.copy_out(
                treebank, 
                gp_copy=False, 
                **kwargs
            )
            copied_children.append(c_copy)
        return treebank.N(
            treebank,
            self.label if treebank == self.treebank else treebank.get_label(self.label.class_id),
            *copied_children,
            operator = self._operator,
            metadata = {**deepcopy(self.metadata), **{'__no_xo__': True}}
        )

    def gp_copy_out(self, treebank = None, gp_copy=False, _at_depth=0, _max_size=None, **kwargs):
        """Generates a deep copy of a tree: the same structure, same Labels, and
        for the Terminals, same content: but each node a distinct object in
        memory from the corresponding node in the original. If a Treebank is
        passed to `treebank`, the new NonTerminal and its children will belong
        to `treebank`.

        If `treebank` is `None`, a new empty Treebank will be created, and the
        new NonTerminal will belong to that instead. This acts as a 'dummy
        treebank', in the case where trees are needed temporarily for some
        computation, without the overhead of storing them in an existing
        treebank.

        Parameters
        ----------
            treebank:
                The target treebank the tree is being copied into
            gp_copy (bool):
                If true, GP mutation and crossover operators will be applied
            sd (SizeDepth):
                A simple tracker which ensures GP crossover operators don't 
                make the output tree too big or too deep
            kwargs:
                Not used, but needed for compatibility with subclasses

        Returns
        -------
            NonTerminal: copy of original tree

        >>> from tree_factories import RandomPolynomialFactory
        >>> from gp import GPTreebank
        >>> from test_materials import DummyTreeFactory
        >>> import pandas as pd
        >>> import operators as ops
        >>> rng = np.random.Generator(np.random.PCG64())
        >>> gp = GPTreebank(
        ...     mutation_rate = 0.0, 
        ...     mutation_sd=0.0, 
        ...     crossover_rate=0.5, 
        ...     max_depth=70,
        ...     max_size=300, 
        ...     seed=rng,
        ...     operators=[ops.SUM, ops.PROD, ops.SQ, ops.POW, ops.CUBE], 
        ...     tree_factory=DummyTreeFactory()
        ... )
        >>> rpf = RandomPolynomialFactory(params = np.array([5, -10.0, 10.0], dtype=float), treebank=gp, seed=rng)
        >>> trees = [rpf('x', 'y') for _ in range(1)]
        >>> for t in trees:
        ...     t.meta_set_recursive(gen=0)
        >>> def test_msr(t):
        ...     assert 'gen' in t.metadata and t.metadata['gen']==0
        ...     if hasattr(t, 'children'):
        ...         for c in t:
        ...             test_msr(c)
        >>> for t in trees:
        ...     test_msr(t)
        >>> df = pd.DataFrame({'x': [1.0, 1.0], 'y': [1.0, 1.0]})
        >>> def incr_gen(t):
        ...     t.metadata['gen'] += 1
        >>> def _all_meta(t, g=None):
        ...     if hasattr(t, 'metadata'):
        ...         assert not t.metadata.get('__no_xo__', False)
        ...         if g is None:
        ...             g = t.metadata['gen']
        ...             print(g)
        ...         else:
        ...             assert g == t.metadata['gen']
        ...     if hasattr(t, 'children'):
        ...         for c in t:
        ...             _all_meta(c, g=g)
        >>> for _ in range(10):
        ...     new_trees = []
        ...     for t in trees:
        ...         _all_meta(t)
        ...         new_t = t.copy(gp_copy=True)
        ...         new_t.apply(incr_gen)
        ...         new_trees.append(new_t)
        ...     for t in trees:
        ...         t.delete()
        ...     for t in new_trees:
        ...         t.meta_set_recursive(__no_xo__=False)
        ...     trees = new_trees
        0
        1
        2
        3
        4
        5
        6
        7
        8
        9
        >>> for t in trees:
        ...     assert t.tree_map_reduce(max, map_any=lambda x: x.metadata['gen']) < 11
        """
        # If `treebank` is None...
        self.__class__.cnt += 1
        if not treebank:
            # create a dummy treebank, and then it won't be None. 
            treebank = self.treebank.__class__()
        if _max_size is None:
            _max_size = treebank.max_size
        _max_depth = treebank.max_depth
        _, to_copy = self.gp_operator(None, self)
        child_sizes = [c.size() for c in self]
        size = self.size()
        gp_copied_children = []
        for i, c in enumerate(self):
            extra_size = _max_size - size
            allowed_child_size = extra_size + child_sizes[i]
            allowed_child_depth = _max_depth - (_at_depth+1)
            c_copy = self.xo_operator(
                None, c, 
                _max_size  = allowed_child_size,
                _max_depth = allowed_child_depth,
                **kwargs
            )[1]
            c_copy = c_copy.copy_out(
                treebank, 
                gp_copy=gp_copy, 
                _max_size  = allowed_child_size,
                _at_depth = _at_depth + 1,
                **kwargs
            )
            gp_copied_children.append(c_copy)
            c_size = c_copy.size()
            size += c_size - child_sizes[i]
            child_sizes[i] = c_size
            # ic(max([c.depth() for c in gp_copied_children]))
        return treebank.N(
            treebank,
            to_copy.label if treebank == to_copy.treebank else treebank.get_label(to_copy.label.class_id),
            *gp_copied_children,
            operator = to_copy._operator,
            metadata = {**deepcopy(to_copy.metadata), **{'__no_xo__': True}}
        )
    
    def __call__(self, **kwargs):
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="overflow encountered in scalar power")
                return super().__call__(**kwargs)
        # What if we get a numerical exception?
        except OverflowError:
            if DEBUG:
                ic.enable()
                ic('Numerical Overflow')
                ic(self)
                child_outputs = pd.DataFrame()
                for i, child in enumerate(self):
                    child_outputs[f'C_{i}'] = child(**kwargs)
                for j, row in child_outputs.iterrows():
                    tmp = (j, row) ### ??? XXX
                    try:
                        self._operator([val for val in row])
                    except:
                        ic(f'GUILTY: {self._operator} on {row} at {j}')
                ic.disable()
            self.root.tmp['penalty'] = self.root.tmp.get('penalty', 1.0) * 2.0
            return None
        except TypeError as e:
            if "'NoneType'" in str(e) and 'unsupported operand type' in str(e):
                ic.enable()
                ic('aw fuck, NoneType')
                ic.disable()
                return None
            raise e
        except AttributeError as e:
            if DEBUG:
                ic.enable()
                ic('Attribute Error')
                ic('This subtree:')
                ic(self)
                ic('in this tree:')
                ic(self.root)
                ic('did a fuckus wuckus')
                ic(e)
                ic.disable()
                raise e
        except ZeroDivisionError:
            if DEBUG:
                ic.enable()
                ic('Zero Division')
                ic(self)
                ic.disable()
            self.root.tmp['penalty'] = self.root.tmp.get('penalty', 1.0) * 2.0**0.1
            return None
            
        
class GPTerminal(Terminal):
    def __new__(cls, treebank, label, leaf, operator=None, metadata=None, tmp=None, gp_operator=None):
        if isinstance(leaf, str):
            if leaf.startswith('$'):
                return cls.__new__(
                    Variable, treebank, label, leaf,
                    operator=operator, metadata=metadata, tmp=tmp
                )
            try:
                leaf = eval(leaf) # the eval'd leaf doesn't get passed to Constant
            except Exception:
                pass
        return cls.__new__(
            Constant,
            treebank,
            label, leaf,
            operator=operator,
            metadata=metadata,
            tmp=tmp
        )


class Variable(GPTerminal):
    def __new__(cls, treebank, label, leaf, operator=None, metadata=None, tmp=None, gp_operator=None):
        return Terminal.__new__(cls)
    
    def __init__(self, treebank, label, leaf, operator=None, metadata=None, tmp=None, gp_operator=None):
        leaf = leaf.strip('$')
        self.mutable = False
        self.gp_operator = gp_operator if gp_operator else NullMutator(treebank)
        ## The line below used to have ```operator=None, metadata=None``` - it didn't seem to break anything, tho :-S
        super().__init__(treebank, label, leaf, operator=operator, metadata=metadata, tmp=tmp)

    def __str__(self):
        """Readable string representation of a Terminal. This consists of a pair
        of parentheses containing a representation of the node's label (label
        name in square brackets), followed by the leaf value, e.g.:

        ([float]$x)
        """
        return f"({self.label if self.label else ''}${self.leaf})"

    def __eq__(self, other):
        """Magic method to operator-overload `==` and `!=`

        Returns
        -------
            bool: True if class, label and leaf are the same, else False.
        """
        return (self.__class__ == other.__class__) and (self.leaf == other.leaf) and (self.label == other.label)

    def __call__(self, **kwargs):
        """When a GP Tree is called, it is given kwargs corresponding to the 
        variables of the expression"""
        #if self.label
        return D(kwargs[self.leaf])

    def to_LaTeX(self, top = True):
        """Converts trees to LaTeX expressions using the `qtree` package.
        Remember to include `\\usepackage{qtree}` in the document header. For
        Terminals, the format is `[.$label $leaf]`. The label expression is
        provided by a similar function in Label.

        Parameters
        ----------:
            top (bool): qtrees expressions must be prefixed with `\\Tree`, but
                this only applies to the whole tree: you don't put it in front
                of every node. However, this function uses recursive calls to
                make the LaTeX of child nodes, so making sure only the top node
                has this prefix takes a bit of extra logic. If the node has no
                parent, the `\\Tree` prefix is automaticaly applied, otherwise
                by default it isn't. However, if LaTeX is wanted for a partial
                tree, `top` may be set to `True`

        >>>
        """
        # prepends \Tree if needed
        LaTeX = r"\Tree " if not (hasattr(self, 'parent') and self.parent) or top else ""
        # LaTeX of the Label is . followed by the label name
        LaTeX += f"[{self.label.to_LaTeX()} {self.leaf} ] "
        return LaTeX.strip() if top else LaTeX
    
    @property
    def is_valid(self):
        return True

    def copy_out(self, treebank = None, gp_copy=False, **kwargs):
        """Generates a deep copy of a Terminal: same Labels, and same content:
        but a distinct object in memory from the original. If `treebank` is a
        Treebank, The new Terminal will be copied into `treebank`. If `treebank`
        is `None`, a dummy treebank will be created, and the Terminal will be
        copied into that.

        Parameters
        ----------
            treebank:
                Treebank: The target treebank the tree is being
                    copied into
            kwargs:
                Not used, but needed for compatibility with subclasses

        Returns
        -------
            Constant: copy of original tree
        """
        # If `treebank` is not provided...
        if not treebank:
            # ...make a dummy treebank for the copied Terminal to live in
            treebank = self.treebank.__class__()
            # XXX How to make sure TB has right OPS?
        # return the copy Terminal, with `treebank=treebank`
        _, to_copy = self.gp_operator(None, self) if gp_copy else (None, self)
        return Variable(
            treebank,
            to_copy.label if treebank == to_copy.treebank else treebank.get_label(to_copy.label.class_id),  ##-OK both, raw val
            '$' + copy(to_copy.leaf),
            operator = to_copy._operator,
            metadata = {**deepcopy(to_copy.metadata), **{'__no_xo__': True}},
            gp_operator=to_copy.gp_operator
        )

class Constant(GPTerminal):
    """A Terminal in which the leaf is always a pd.Series of length 1, with a
    GP operator which mutates the value of the constant.

    >>>
    >>>
    """

    def __new__(cls, treebank, label, leaf, operator=None, metadata=None, tmp=None, gp_operator=None):
        return Terminal.__new__(cls)

    def __init__(self, treebank, label, leaf, operator=None, metadata=None, tmp=None, gp_operator=None):
        leaf = D(leaf)
        self.leaf_type = None
        if not isinstance(leaf, (pd.Series, np.ndarray)):
            self.leaf_type = type(leaf)
            if self.leaf_type == str:
                ic.enable()
                ic("WUT", leaf)
                ic.disable()
            # leaf = pd.Series([leaf]) # XXX CHG
        self.leaf_type = self.leaf_type if self.leaf_type else treebank.tn.type_ify(leaf)
        self.gp_operator = gp_operator if gp_operator else MutatorFactory(
            self.leaf_type,
            treebank
        )
        self.mutable = True
        super().__init__(treebank, label, leaf, operator=operator, metadata=metadata, tmp=tmp)

    def _leaf_str(self):
        try:
            leaf_len = len(self.leaf)
        except TypeError:
            return str(self.leaf)
        return ("" if not leaf_len else
            self.leaf[0] if leaf_len == 1 else
            str(list(self.leaf)) if leaf_len < 7 else
            f"[{self.leaf[0]}, {self.leaf[1]}, {self.leaf[2]} ... " +
            f"{self.leaf[leaf_len-3]}, {self.leaf[leaf_len-2]}, {self.leaf[leaf_len-1]}]"
        )

    def __str__(self):
        """Readable string representation of a Terminal. This consists of a pair
        of parentheses containing a representation of the node's label (label
        name in square brackets), followed by the leaf value, e.g.:

        ([float]x).
        """
        return f"({self.label if self.label else ''}{self._leaf_str()})" 
    
    def __eq__(self, other):
        """Magic method to operator-overload `==` and `!=`

        Returns
        -------
            bool: True if class, label and leaf are the same, else False.
        """
        return (self.__class__ == other.__class__) and (self.leaf == other.leaf) and (self.label == other.label)

    def to_LaTeX(self, top = True):
        """Converts trees to LaTeX expressions using the `qtree` package.
        Remember to include `\\usepackage{qtree}` in the document header. For
        Terminals, the format is `[.$label $leaf]`. The label expression is
        provided by a similar function in Label.

        Parameters
        ----------:
            top (bool): qtrees expressions must be prefixed with `\\Tree`, but
                this only applies to the whole tree: you don't put it in front
                of every node. However, this function uses recursive calls to
                make the LaTeX of child nodes, so making sure only the top node
                has this prefix takes a bit of extra logic. If the node has no
                parent, the `\\Tree` prefix is automaticaly applied, otherwise
                by default it isn't. However, if LaTeX is wanted for a partial
                tree, `top` may be set to `True`

        >>>
        """
        # prepends \Tree if needed
        LaTeX = r"\Tree " if not (hasattr(self, 'parent') and self.parent) or top else ""
        # LaTeX of the Label is . followed by the label name
        LaTeX += f"[{self.label.to_LaTeX()} {self._leaf_str()} ] "
        return LaTeX.strip() if top else LaTeX

    def __getitem__(self, position):
        if not position in (0, (0,)):
            return super().__getitem__(position)
        return self.leaf # XXX CHG

    def copy_out(self, treebank = None, gp_copy=False, **kwargs):
        """Generates a deep copy of a Terminal: same Labels, and same content:
        but a distinct object in memory from the original. If `treebank` is a
        Treebank, The new Terminal will be copied into `treebank`. If `treebank`
        is `None`, a dummy treebank will be created, and the Terminal will be
        copied into that.

        Parameters
        ----------
            treebank:
                Treebank: The target treebank the tree is being
                    copied into
            gp_copy:
                bool: Iff true, the constant's Mutator will be applied
            kwargs:
                Not used, but needed for compatibility with subclasses

        Returns
        -------
            Constant: copy of original tree
        """
        # If `treebank` is not provided...
        if not treebank:
            # ...make a dummy treebank for the copied Terminal to live in
            treebank = self.treebank.__class__()
        # return the copy Terminal, with `treebank=treebank`
        copy_leaf, to_copy = self.gp_operator(copy(self.leaf), self) if gp_copy else (copy(self.leaf), self)
        return Constant(
            treebank,
            to_copy.label if treebank == to_copy.treebank else treebank.get_label(to_copy.label.class_id),  ##-OK both, raw val
            copy_leaf,
            operator = to_copy._operator,
            metadata = {**deepcopy(to_copy.metadata), **{'__no_xo__': True}},
            gp_operator=to_copy.gp_operator
        )

    @property
    def is_valid(self):
        if isinstance(self[0], self.label.class_id): ##-OK TL RAW
            return True
        elif issubclass(self.treebank.tn.type_ify(self[0]), self.label.class_id):
            return True
            #return term[0].apply(lambda _: type(_) == term.label.class_id).all() ##-OK TL RAW
        else:
            raise OperatorError(
                f"{self} has a mismatch of label-type " +
                f"({self.label.classname}) and leaf-type " + ##-OK TL NAME
                f"({type(self[0]).__name__})"
            )


def main():
    import doctest
    doctest.testmod()

if __name__ == '__main__':
    main()