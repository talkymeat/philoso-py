from trees import Terminal, NonTerminal
from mutators import MutatorFactory, CrossoverMutator
import pandas as pd
import numpy as np
from copy import copy
from treebanks import TypeLabelledTreebank
from tree_errors import OperatorError
from typing import TypeAlias
from size_depth import SizeDepth
from logtools import MiniLog
from icecream import ic
import warnings


DEBUG = True

class D:
    TYPES = {
        int: np.int64,
        float: np.float64,
        complex: np.complex128,
        str: np.string_
    }


    def f(self, *arg):
        return False

    def __new__(cls, val):
        return D.TYPES.get(type(val), cls.f)(val) or val


class GPNonTerminal(NonTerminal):
    """GPNonTerminals carry operators, and take valid return types of their operators
    as Label values. They can be called with arguments that are passed down to the 
    GPTerminals, which can be Constants or Variables. When the `kwargs`
    `dict`/`DataFrame` is passed down the tree, if a Variable has a name corresponding 
    to a key of `kwargs`, the corresponding value/`Series` will be passed back up the
    tree and operated on

    >>> from gp import GPTreebank
    >>> import operators as ops
    >>> op = [ops.SUM, ops.PROD, ops.SQ, ops.CUBE, ops.POW]
    >>> gp = GPTreebank(operators=op)
    >>> mewtwo = gp.tree("([float]<SUM>([float]<SQ>([int]$mu))([float]<SUM>([float]<PROD>([int]3)([int]$mu))([int]2)))")
    >>> mewtwo(mu=-2)
    0.0
    >>> mewtwo(mu=-1)
    0.0
    >>> mewtwo(mu=-3)
    2.0
    """
    def __init__(self, treebank, label, *children, operator=None, metadata=None):
        super().__init__(treebank, label, *children, operator=operator, metadata=metadata)
        self.gp_operator = CrossoverMutator(treebank.crossover_rate, treebank.max_depth, treebank.max_size)

    @property
    def is_valid(self):
        if not issubclass(self.label.class_id, self._operator.return_type): ##-OK TL RAW
            raise OperatorError(
                f"{self} has a mismatch of label-type " +
                f"({self.label.classname}) and operator " +  ##-OK TL NAME
                f"return-type ({self._operator.return_type.__name__})"
            )
        if not self._operator._type_seq_legal(*[ch.label.class_id for ch in self]): ##-OK TL RAW
            raise OperatorError(
                f"the children of {self} are not a legal argument-sequence " +
                f"for its operator {self._operator.name}"
            )
        return True

    def copy_out(self, treebank = None, gp_copy=False, _sd: SizeDepth=None, **kwargs):
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
                If true, GP mutation and corssover operators will be applied
            sd (SizeDepth):
                A simple tracker which ensures GP crossover operators don't 
                make the output tree too big or too deep
            kwargs:
                Not used, but needed for compatibility with subclasses

        Returns
        -------
            NonTerminal: copy of original tree
        """
        # If `treebank` is None...
        if not treebank:
            # create a dummy treebank, and then it won't be None. 
            treebank = TypeLabelledTreebank()
        sd = _sd if _sd and gp_copy else SizeDepth(
            size=self.size(),
            depth=self.depth(),
            max_size=treebank.max_size,
            max_depth=treebank.max_depth
        )
        return treebank.N(
            treebank,
            self.label if treebank == self.treebank else treebank.get_label(self.label.class_id),
            *[(
                self.gp_operator(
                    c, sd=sd
                ).copy_out(
                    treebank, gp_copy=gp_copy, _sd=sd, **kwargs
                )
                if gp_copy
                else c.copy_out(treebank, gp_copy=gp_copy, **kwargs)
            ) for c in self],
            operator = self._operator
        ) 
    
    def __call__(self, **kwargs):
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="overflow encountered in scalar power")
                return super().__call__(**kwargs)
        # What if we get a numerical exception?
        except OverflowError:
            if DEBUG:
                ic('Numerical Overflow')
                ic(self)
                child_outputs = pd.DataFrame()
                for i, child in enumerate(self):
                    child_outputs[f'C_{i}'] = child(**kwargs)
                for j, row in child_outputs.iterrows():
                    tmp = (j, row)
                    try:
                        self._operator([val for val in row])
                    except:
                        ic(f'GUILTY: {self._operator} on {row} at {j}')
            self.root.metadata['penalty'] = self.root.metadata.get('penalty', 1.0) * 2.0
            return None
        except TypeError as e:
            if "'NoneType'" in str(e) and 'unsupported operand type' in str(e):
                ic('aw fuck, NoneType')
                return None
            raise e
        except AttributeError as e:
            if DEBUG:
                ic('Attribute Error')
                ic('This subtree:')
                ic(self)
                ic('in this tree:')
                ic(self.root)
                ic('did a fuckus wuckus')
                ic(e)
        except ZeroDivisionError:
            if DEBUG:
                ic('Zero Division')
                ic(self)
            self.root.metadata['penalty'] = self.root.metadata.get('penalty', 1.0) * 2.0**0.1
            return None
            
        
class GPTerminal(Terminal):
    def __new__(cls, treebank, label, leaf, operator=None, metadata=None):
        if isinstance(leaf, str):
            if leaf.startswith('$'):
                return cls.__new__(
                    Variable, treebank, label, leaf,
                    operator=operator, metadata=metadata
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
            metadata=metadata
        )


class Variable(GPTerminal):
    def __new__(cls, treebank, label, leaf, operator=None, metadata=None):
        return Terminal.__new__(cls)
    
    def __init__(self, treebank, label, leaf, operator=None, metadata=None):
        leaf = leaf.strip('$')
        super().__init__(treebank, label, leaf, operator=None, metadata=None)

    def __str__(self):
        """Readable string representation of a Terminal. This consists of a pair
        of parentheses containing a representation of the node's label (label
        name in square brackets), followed by the leaf value, e.g.:

        ([float]x).
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

    def copy_out(self, treebank = None, **kwargs):
        """Generates a deep copy of a Terminal: same Labels, and same content:
        but a distinct object in memory from the original. If `treebank` is a
        Treebank, The new Terminal will be copied into `treebank`. If `treebank`
        is `None`, a dummy treebank will be created, and the Terminal will be
        copied into that.

        Parameters
        ----------
            treebank:
                TypeLabelledTreebank: The target treebank the tree is being
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
        return Variable(
            treebank,
            self.label if treebank == self.treebank else treebank.get_label(self.label.class_id),  ##-OK both, raw val
            '$' + copy(self.leaf),
            operator = self._operator
        )

class Constant(GPTerminal):
    """A Terminal in which the leaf is always a pd.Series of length 1, with a
    GP operator which mutates the value of the constant.

    >>>
    >>>
    """
    def __new__(cls, treebank, label, leaf, operator=None, metadata=None):
        return Terminal.__new__(cls)

    def __init__(self, treebank, label, leaf, operator=None, metadata=None):
        leaf = D(leaf)
        if isinstance(leaf, str):
            try:
                leaf = eval(leaf)
            except Exception:
                pass
        leaf_type = None
        if not isinstance(leaf, pd.Series):
            leaf_type = type(leaf)
            if leaf_type == str:
                ic("WUT", leaf)
            # leaf = pd.Series([leaf]) # XXX CHG
        self.gp_operator = MutatorFactory(
            leaf_type if leaf_type else treebank.tn.type_ify(leaf),
            treebank.mutation_rate,
            treebank.mutation_sd
        )
        super().__init__(treebank, label, leaf, operator=operator, metadata=metadata)

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
        return self.__class__ == other.__class__ and self.leaf == other.leaf and self.label == other.label

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
            return super.__getitem__(position)
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
                TypeLabelledTreebank: The target treebank the tree is being
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
            treebank = TypeLabelledTreebank()
            # XXX SHould that have been `treebank = self.treebank.__class__()`?
        # return the copy Terminal, with `treebank=treebank`
        return Constant(
            treebank,
            self.label if treebank == self.treebank else treebank.get_label(self.label.class_id),  ##-OK both, raw val
            self.gp_operator(copy(self.leaf)) if gp_copy else copy(self.leaf),
            operator = self._operator
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