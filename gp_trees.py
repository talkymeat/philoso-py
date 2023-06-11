from trees import Terminal, NonTerminal
from mutators import MutatorFactory, CrossoverMutator
import pandas as pd
from copy import copy

class GPNonTerminal(NonTerminal):
    def __init__(self, treebank, label, *children, operator=None, metadata=None):
        super().__init__(treebank, label, *children, operator=operator, metadata=metadata)
        self.gp_operator = CrossoverMutator(treebank.crossover_rate)

    def copy_out(self, treebank = None, gp_copy=False, **kwargs):
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
            kwargs:
                Not used, but needed for compatibility with subclasses

        Returns
        -------
            NonTerminal: copy of original tree
        """
        # If `treebank` is None...
        if not treebank:
            # create a dummy treebank, and then it won't be None
            treebank = TypeLabelledTreebank()
        #### ALSO this needs to be able to handle operators
        # Then create the copied NonTerminal, and recursively copy its children,
        # also to `treebank`: this means, if the function is called with
        # `treebank == None`, the whole tree-fragment will be copied to the same
        # dummy treebank.
        return treebank.N(
            treebank,
            self.label if treebank == self.treebank else treebank.get_label(self.label.class_id),
            *[(
                self.gp_operator(c).copy_out(treebank, gp_copy=gp_copy, **kwargs)
                if gp_copy
                else c.copy_out(treebank, gp_copy=gp_copy, **kwargs)
            ) for c in self],
            operator = self._operator
        )

class GPTerminal(Terminal):
    def __new__(cls, treebank, label, leaf, operator=None, metadata=None):
        if isinstance(leaf, str):
            try:
                leaf = eval(leaf)
            except Exception:
                pass
        if isinstance(leaf, pd.Series) and len(leaf) > 1:
            return cls.__new__(
                Variable, treebank, label, leaf,
                operator=operator, metadata=metadata
            )
        else:
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

    def __str__(self):
        """Readable string representation of a Terminal. This consists of a pair
        of parentheses containing a representation of the node's label (label
        name in square brackets), followed by the leaf value, e.g.:

        ([float]x).
        """
        return f"({self.label if self.label else ''}{self.leaf.name})"

    def __eq__(self, other):
        """Magic method to operator-overload `==` and `!=`

        Returns
        -------
            bool: True if class, label and leaf are the same, else False.
        """
        return self.__class__ == other.__class__ and (self.leaf == other.leaf).all() and self.label == other.label


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
        LaTeX += f"[{self.label.to_LaTeX()} {self.leaf.name} ] "
        return LaTeX.strip() if top else LaTeX

class Constant(GPTerminal):
    """A Terminal in which the leaf is always a pd.Series of length 1, with a
    GP operator which mutates the value of the constant.

    >>>
    >>>
    """
    def __new__(cls, treebank, label, leaf, operator=None, metadata=None):
        return Terminal.__new__(cls)

    def __init__(self, treebank, label, leaf, operator=None, metadata=None):
        if isinstance(leaf, str):
            try:
                leaf = eval(leaf)
            except Exception:
                pass
        leaf_type = None
        if not isinstance(leaf, pd.Series):
            leaf_type = type(leaf)
            leaf = pd.Series([leaf])
        self.gp_operator = MutatorFactory(
            leaf_type if leaf_type else treebank.tn.type_ify(leaf),
            treebank.mutation_rate,
            treebank.mutation_sd
        )
        super().__init__(treebank, label, leaf, operator=operator, metadata=metadata)

    def _leaf_str(self):
        leaf_len = len(self.leaf)
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
        return self.__class__ == other.__class__ and self.leaf.name == other.leaf.name and self.label == other.label

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
        return self.leaf[0]

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
        # return the copy Terminal, with `treebank=treebank`
        return Constant(
            treebank,
            self.label if treebank == self.treebank else treebank.get_label(self.label.class_id),  ##-OK both, raw val
            copy(self.leaf).apply(self.gp_operator) if gp_copy else copy(self.leaf),
            operator = self._operator
        )
