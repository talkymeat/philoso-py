#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
I am at last re-implementing my PhD code in Python, and also in a
less shit way than before, and with some enhancements. The plan is
to:
    * implement regular DOP & its derivatives
    * implement NN-like tuning between substitution and children,
        designed to learn the most effective DOP probability model
    * implement DDOP-style incremental running
    * implement other DDOP variants
    * combine DDOP with NeuroDOP for bootstrappable node-labelling
    * Motor DDOP

Right now, I'm just working on getting a suitable set of classes
for making labelled trees

Created on Thu Nov 30 16:59:45 2017

@author: Dave Cochran
"""

from copy import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Tuple, Union
from tree_errors import TreeIndexError
from tree_iter import TreeIter, DepthFirstBottomUp as DFBU
import treebanks as tbs
import operators as ops
# TODO: don't leave the operator import here: do that in GP, and in tests


class Tree(ABC):
    """Abstract Base Class defining behaviour for all tree nodes.

    All trees are labelled, but some have children (lists of tree nodes below
    them in the tree structure), and others have leaves - concrete content like
    words in syntactic parse trees or constants and variables in Genetic
    Programming systems: so the abstract base class contains a constructor that
    sets the label, but which the subclasses can extend to add leaves or
    children.

    Attributes
    ----------
        _label : Label
            Node label determining which subtrees can be substituted where. Set
            and got by the `label` @property
        parent : NonTerminal or None
            The node dirctly above the present in the tree. Each child has a
            reference to its parent, and each parent has references to all its
            children.
        treebank : Treebank
            Every tree belongs to a Treebank, which defines the set of Labels
            that `Tree`, its parents, and its children can take.

    Raises
    ------
        TypeError
            If the `label` passed to `__init__` is not a valid Label.
    """

    def __init__(self, treebank, label, *args, metadata=None, **kwargs):
        # Set parent to None: parent, if there is one, will set child's `parent`
        # param
        self.parent = None
        # If no metadata is provided, set an empty dict just in case
        self.metadata = metadata if metadata else {}
        # DOP uses strings as labels, GP uses types, so those are OK
        if isinstance(label, (str, type)):
            self.label = treebank.get_label(label)
        # Or if a Label is passed as label, that's fine too
        elif issubclass(type(label), Label):
            self.label = label;
        # Otherwise it can suck it
        else:
            raise TypeError(f"Invalid label of type {type(label)}")

    @property
    def treebank(self):
        return self.label.treebank

    @property
    def label(self):
        """Label: Attribute storing a node label"""
        if hasattr(self, '_label'):
            return self._label

    @label.setter
    def label(self, label):
        """Setter for label. Ensures that the tree node is registered with its
        label. Deletes label from node if called with `label=None`
        """
        if isinstance(label, Label):
            self._label = label
            self.label.add_node(self)
        elif label is None:
            del(self._label)
        else:
            raise TypeError(
                f"You can set a label to be a {Label}, or None, but not " +
                f"{type(label)}."
            )

    @abstractmethod
    def __len__(self) -> int:
        """Number of children"""
        pass

    def __iter__(self):
        return TreeIter(self)

    def copy(self, **kwargs):
        """Generates a deep copy of a tree: the same structure, same Labels, and
        for the Terminals, same content: but each node a distinct object in
        memory from the corresponding node in the original. The new NonTerminal
        and its children belong to the same treebank as the original.

        Parameters
        ----------
            kwargs: not used, but needed for compatibility with subclasses

        Returns
        -------
            NonTerminal: copy of original tree

        >>> from functools import reduce
        >>> from test_trees import test_trees, test_fragments
        >>> t = test_trees() + test_fragments()
        >>> tcopy = [copy(x) for x in t]
        >>> idents = [id(cp) == id(og) for cp, og in zip(tcopy, t)]
        >>> equals = [cp == og for cp, og in zip(tcopy, t)]
        >>> print(reduce(lambda a, b: a and b, equals))
        True
        >>> print(reduce(lambda a, b: a or b, idents))
        False
        """
        return self.copy_out(self.treebank, **kwargs)

    @abstractmethod
    def depth(self) -> int:
        """Length of chain from self to it's most distant descendant-node"""
        pass

    @abstractmethod
    def width(self) -> int:
        """Number of leaf nodes below current"""
        pass

    @abstractmethod
    def size(self) -> int:
        """Number of nodes in tree overall"""
        pass

    @abstractmethod
    def to_LaTeX(self) -> str:
        """Outputs LaTeX representation of tree. Handy for putting in papers"""
        pass

    @abstractmethod
    def __call__(self):
        """All nodes are callable: Terminals return their leaf, always:
        NonTerminals call an Operator that belongs to the Label
        """
        pass

    @abstractmethod
    def __eq__(self, other):
        """True if two trees are identical"""
        pass

    @abstractmethod
    def __str__(self):
        """Readable printout"""
        pass

    @abstractmethod
    def __getitem__(self, position):
        "So children can be indexed positionally"
        pass

    @abstractmethod
    def copy_out(self, treebank, **kwargs):
        """Copy self with all children. Copy exists in `treebank`."""
        pass

    @abstractmethod
    def delete(self):
        """Removes a tree and all its descendants from their treebank."""
        pass

    def __repr__(self) -> str:
        """Returns a string such that tree == eval(tree.__repr__())"""
        return f'tree("{str(self)}")'


class NonTerminal(Tree):
    """Generic class for non-terminal tree nodes: that is, nodes which do not
    carry terminal content like words in syntactic parsing or constants and
    variables in genetic programming, but which can take other tree nodes as
    children

    TODO: add a metadata attribute

    Parameters
    ----------
    _label : Label
        Inherited from Tree. Node label determining which subtrees can be
        substituted where. Set and got by the `label` @property
    parent : NonTerminal or None
        Inherited from Tree. The node dirctly above the present in the tree.
        Each child has a reference to its parent, and each parent has references
        to all its children.
    treebank : Treebank
        Every tree belongs to a Treebank, which defines the set of Labels that
        `Tree`, its parents, and its children can take.
    _children : list of Trees or None
        The nodes directly below the present in the tree.
    _operator : Operator
        An Operator is a function with a wrapper that enforces type-checking.
        `_operator` is the function called on the NonTerminals children when
        the NonTerminal is called using __call__.
    """

    """bool: If True, the `+` operator does leftmost nonterminal substitution,
    which throws an error if the liftmost nonterminal leaf node label of the
    left operand does not match the root label of the right operand; if False it
    substitutes the right operand into the leftmost matching nonterminal
    leafnode, and raises an error only if there is no match.
    """

    def __init__(self, treebank, label, *children, operator=None, metadata=None, **kwargs):
        self._children = []
        self._operator = None
        self.children = list(children)
        if operator:
            self._operator = operator
        elif treebank.default_op:
            self._operator = treebank.default_op
        else:
            raise AttributeError(
                "NonTerminals must be supplied with an Operator if Treebank has"
                + " no default Operator"
            )
        super().__init__(treebank, label)

    @property
    def children(self):
        """Getter for the children attribute"""
        return self._children

    @children.setter
    def children(self, children):
        """Checks that the elements of the `children` param are all Trees,
        and if they are, sets their `parent` attributes to `self`, and set the
        list to be the `_children` attribute of self. The attribute will only be
        set if all list members are valid children: if an invalid child is found
        an error is raised and any list members that had their `parent`
        attribute set to `self` will have it set back to `None`.

        TODO: check that children belong to same treebank, or copy children in
            to Treebank if arg `copy_in = True` (default to False)

        Raises
        ------
        TypeError
            If something other than a tree is passed as a child
        """
        for i, c in enumerate(children): # Make sure all children are Trees
            if isinstance(c, self.__class__) or isinstance(c, Terminal):
                c.parent = self # If the child is a tree, self can be its parent
            else:
                # If not...
                for orphan in children[:i]:
                    # This is the saddest line of code I have ever written
                    orphan.parent = None
                raise TypeError(f"You tried to set a {type(c)!s}, {c!s} as " +
                                f"a child of the Tree {self!s}. You should " +
                                "have used a Tree")
        # If all children typecheck OK, it's ok to go ahead and set them as
        # the children of the current node
        self._children = children

    def __len__(self) -> int:
        """All nodes have a length: Terminals have length 1, while NonTerminals
        have a length equal to the number of children

        Returns
        -------
        int:  Number of children

        >>> from test_trees import test_trees, test_fragments
        >>> t = test_trees()
        >>> print([len(tx) for tx in t])
        [1, 2, 2, 2, 2, 2, 2, 2]
        >>> print([len(tbs.tree(f"([S]{'([N]blah)'*x})")) for x in range(11)])
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        >>> tf = test_fragments()
        >>> print([len(tfx) for tfx in tf])
        [2, 2, 0]
        """
        return len(self._children) if hasattr(self, '_children') else 0

    def __str__(self):
        """Readable string representation of a tree. This consists of a pair of
        parentheses containing a representation of the tree's label (label name
        in square brackets), followed by a representation of the node's
        children, which is just each child's representation, concatenated, eg:

        ([VP]([V]'bites')([N]'man')).

        >>> # No doctest needed here, as doctest for test_trees also tests this
        """
        return f"({self.label if self.label else ''}{self._operator if self._operator else ''}{''.join([str(child) for child in self]) if self else ''})"

    def depth(self) -> int:
        """All nodes have a depth: the depth of a leaf node is 1, and the depth
        of any other node is the depth of its deepest child + 1

        Returns
        -------
        int:
            Tree depth

        >>> from test_trees import test_trees, test_fragments
        >>> t = test_trees()
        >>> print([tx.depth() for tx in t])
        [1, 3, 3, 3, 4, 3, 3, 2]
        >>> tf = test_fragments()
        >>> print([tfx.depth() for tfx in tf])
        [3, 3, 0]
        >>> blah_depths = []
        >>> blah = "([S]'blah')"
        >>> print(tbs.tree(blah).depth())
        1
        >>> for i in range(8):
        ...     blah = blah.replace("'blah'", blah)
        ...     blah_depths.append(tbs.tree(blah).depth())
        ...
        >>> print(blah_depths)
        [2, 4, 8, 16, 32, 64, 128, 256]
        >>> tb = tbs.Treebank()
        >>> fragment = tbs.tree("([S]([PN]'I')([VP]))", treebank = tb)
        >>> fragment.depth()
        2
        """
        # Recursive function with two stopping cases: if called on a
        # NonTerminal leafnode, or in a Terminal. The `else` case below handles
        # the former, and for the latter, Terminal has its own version of the
        # function
        return (1 + max([c.depth() for c in self])) if self.children else 0

    def width(self) -> int:
        """All nodes have a width: the width of a leaf node is 1, and the width
        of any other node is the sum of its children's widths. This is
        equivalent to the total number of leaf nodes in the tree

        Returns
        -------
        int:
            Tree width

        >>> from test_trees import test_trees, test_fragments
        >>> t = test_trees()
        >>> print([tx.width() for tx in t])
        [1, 3, 3, 3, 5, 3, 3, 2]
        >>> tf = test_fragments()
        >>> print([tfx.width() for tfx in tf])
        [3, 4, 1]
        >>> blah_widths = []
        >>> blah = "([N]'blah')"
        >>> print(tbs.tree(blah).width())
        1
        >>> for i in range(1,4):
        ...     blah_top = "([S]'blah')"
        ...     for j in range(8):
        ...         blah_top = blah_top.replace("'blah'", blah*i)
        ...         blah_widths.append(tbs.tree(blah_top).width())
        ...
        >>> print(blah_widths)
        [1, 1, 1, 1, 1, 1, 1, 1, 2, 4, 8, 16, 32, 64, 128, 256, 3, 9, 27, 81, 243, 729, 2187, 6561]
        >>> tb = tbs.Treebank()
        >>> fragment = tbs.tree("([S]([PN]'I')([VP]))", treebank = tb)
        >>> fragment.width()
        2
        """
        return max(sum([c.width() for c in self]), 1)

    def size(self) -> int:
        """All nodes have a size: the size of a leaf node is 1, and the size
        of any other node is the sum of its children's sizes, plus 1. This is
        the total number of nodes in the tree.

        Returns
        -------
        int:
            total number of nodes in the tree

        >>> from test_trees import test_trees, test_fragments
        >>> t = test_trees()
        >>> print([tx.size() for tx in t])
        [1, 5, 5, 5, 9, 5, 5, 3]
        >>> tf = test_fragments()
        >>> print([tfx.size() for tfx in tf])
        [5, 7, 1]
        >>> blah_sizes = []
        >>> blah = "([N]'blah')"
        >>> print(tbs.tree(blah).size())
        1
        >>> for i in range(1,4):
        ...     blah_top = "([S]'blah')"
        ...     for j in range(8):
        ...         blah_top = blah_top.replace("'blah'", blah*i)
        ...         blah_sizes.append(tbs.tree(blah_top).size())
        ...
        >>> print(blah_sizes)
        [2, 3, 4, 5, 6, 7, 8, 9, 3, 7, 15, 31, 63, 127, 255, 511, 4, 13, 40, 121, 364, 1093, 3280, 9841]
        """
        return 1 + sum([c.size() for c in self])

    def __eq__(self, other):
        """Checks that two trees are the same: for this to return true, the
        trees must be of the same type, with the same root label, with the same
        number of children, and each child in self must equal the child at the
        same index in other. `Terminal`s must also have equal leaves. Magic
        method for operator overloading on `==` and `!=`.

        TODO: make sure Label has an appropriate __eq__ method.

        Parameters
        ----------
        other : Tree
            The right operand

        Returns
        -------
        bool
            True if trees are equal, False otherwise

        >>> from test_trees import test_trees, test_fragments
        >>> t = test_trees()
        >>> t[0] == t[0]
        True
        >>> t[1] == t[1]
        True
        >>> t[3] == t[1]
        True
        >>> t[3] == t[2]
        True
        >>> t[1] == t[3]
        True
        >>> t[1] == t[2]
        True
        >>> t[2] == t[3]
        True
        >>> t[2] == t[1]
        True
        >>> t[1] == t[4]
        False
        >>> t[1] == t[5]
        False
        >>> t[1] == t[6]
        False
        >>> tf = test_fragments()
        >>> for tfx in tf:
        ...     print([int(tfx == tfy) for tfy in tf])
        ...
        [1, 0, 0]
        [0, 1, 0]
        [0, 0, 1]
        """
        # First check all the local property comparisons
        if self.__class__ == other.__class__ and self.label == other.label and len(self) == len(other):
            # If all of those come up true, check the children
            for self_c, other_c in zip(self, other):
                # If any children are not equal, self and other are not equal
                if self_c != other_c:
                    return False
            # If all of that fails to return false, self and other are equal
            return True
        # If the local comparisons do not all come out true, they are not equal
        else:
            return False

    def to_LaTeX(self, top = True):
        """Converts trees to LaTeX expressions using the `qtree` package.
        Remember to include `\\usepackage{qtree}` in the document header. For
        NonTerminals, the format is `[.$label $child* ]`. The label expression
        is provided by a similar function in Label. The $child expressions are
        recursively in the same format again (if they are also NonTerminals), or
        `[.$label $leaf]` for Terminals, which have a separate `to_LaTeX`
        method.

        Parameters
        ----------
        top : bool
            qtrees expressions must be prefixed with `\\Tree`, but this only
            applies to the whole tree: you don't put it in front of every node.
            However, this function uses recursive calls to make the LaTeX of
            child nodes, so making sure only the top node has this prefix takes
            a bit of extra logic. If the node has no parent, the `\\Tree` prefix
            is automaticaly applied, otherwise by default it isn't. However, if
            LaTeX is wanted for a partial tree, `top` may be set to `True`

        Returns
        -------
        str
            LaTeX expression representing the tree.

        >>> from test_trees import test_trees, test_fragments
        >>> t = test_trees()
        >>> for x in t[1:]:
        ...     print(x.to_LaTeX())
        ...
        \Tree [.S [.N sentence ] [.VP [.V parses ] [.N parser ] ] ]
        \Tree [.S [.N sentence ] [.VP [.V parses ] [.N parser ] ] ]
        \Tree [.S [.N sentence ] [.VP [.V parses ] [.N parser ] ] ]
        \Tree [.S [.NP [.Det the ] [.N sentence ] ] [.VP [.V parses ] [.NP [.Det the ] [.N parser ] ] ] ]
        \Tree [.S [.N word ] [.VP [.V parses ] [.N parser ] ] ]
        \Tree [.S [.N sentence ] [.VP [.V parses ] [.V parser ] ] ]
        \Tree [.S [.X x ] [.Y y ] ]
        >>> tf = test_fragments()
        >>> for tfx in tf:
        ...     print(tfx.to_LaTeX())
        ...
        \Tree [.S [.NP [.Det the ] [.N cat ] ] [.VP ] ]
        \Tree [.S [.NP [.Det the ] [.N cat ] ] [.VP [.V ate ] [.NP ] ] ]
        \Tree [.NP ]
        """
        # prepends \Tree if needed
        latex = r"\Tree " if not (hasattr(self, 'parent') and self.parent) or top else ""
        latex += f"[{self.label.to_LaTeX(op_name = self._operator.name)} {''.join([c.to_LaTeX(top = False) for c in self])}] "
        return latex.strip() if top else latex

    def __getitem__(self, position):
        """All trees are indexable, with the i^th index retrieving the i^th
        child, except with Terminals, where the zeroth and only item is the leaf
        value. Multiple indices can be provided, comma separated, to index a
        grandchild, great-grandchild, etc.

        Parameters
        ----------
        position : int or tuple of ints
            If `int`, the index of a child in `self`'s list of children. If
            `tuple`, a sequence of indices to navigate downwards from `self` to
            a specified descendant of `self`

        Returns
        -------
        Tree or str, int, float, etc, depending on the valid types for leaf
        content
            A descendant of `self` located by index. If the penultimate term in
            the list of tuples indicates a `Terminal`, and the last index is 0,
            this will return the leaf content of that `Terminal`

        Raises
        ------
        TreeIndexError:
            Raises a custom IndexError if the path through the tree represented
            by `position` does not exist. This can happen if:
                * A NonTerminal with children is indexed, but the index is out
                of range
                * A NonTerminal without children is indexed at all
                * A Terminal is indexed with an int other than 0
                * A Terminal is indexed with a tuple of length greater than 1,
                even if the first item in the tuple is 0

        >>> from test_trees import test_fragments
        >>> t1 = tbs.tree("([S]([N]sentence)([VP]([V]parses)([NP]([Det]a)([N]parser))))")
        >>> print(t1[0])
        ([N]'sentence')
        >>> print(t1[0, 0])
        sentence
        >>> print(t1[1, 0])
        ([V]'parses')
        >>> print(t1[1, 1])
        ([NP]([Det]'a')([N]'parser'))
        >>> print(t1[1, 1, 0])
        ([Det]'a')
        >>> print(t1[1, 1, 0, 0])
        a
        >>> print(t1[1, 2])
        Traceback (most recent call last):
            ....
        tree_errors.TreeIndexError: Invalid index. The Tree:
        ([S]([N]'sentence')([VP]([V]'parses')([NP]([Det]'a')([N]'parser'))))
        was indexed with (1, 2), but has no subtree at (1, 2). Index out of range.
        >>> tf = test_fragments()
        >>> print(tf[0][1])
        ([VP])
        >>> print(tf[0][1, 0, 0])
        Traceback (most recent call last):
            ....
        tree_errors.TreeIndexError: Invalid index. The Tree:
        ([S]([NP]([Det]'the')([N]'cat'))([VP]))
        was indexed with (1, 0, 0), but has no subtree at (1, 0). NonTerminal leaf-nodes cannot be indexed
        >>> print(tf[0][0, 1])
        ([N]'cat')
        >>> print(tf[1][1, 0])
        ([V]'ate')
        >>> print(tf[1][1])
        ([VP]([V]'ate')([NP]))
        >>> print(tf[1][1, 1])
        ([NP])
        >>> t2 = tbs.tree("([S]([PN]'we')([VP]([V]'ate')([NP]([Det]'a')([NN]([Adj])([N]'pie')))))")
        >>> print(t2[1][1,1,2])
        Traceback (most recent call last):
            ....
        tree_errors.TreeIndexError: Invalid index. The Tree:
        ([VP]([V]'ate')([NP]([Det]'a')([NN]([Adj])([N]'pie'))))
        was indexed with (1, 1, 2), but has no subtree at (1, 1, 2). Index out of range.
        >>> print(t2[1][1,1,2,99])
        Traceback (most recent call last):
            ....
        tree_errors.TreeIndexError: Invalid index. The Tree:
        ([VP]([V]'ate')([NP]([Det]'a')([NN]([Adj])([N]'pie'))))
        was indexed with (1, 1, 2, 99), but has no subtree at (1, 1, 2). Index out of range.
        >>> print(t2[1][1,1,0,0])
        Traceback (most recent call last):
            ....
        tree_errors.TreeIndexError: Invalid index. The Tree:
        ([VP]([V]'ate')([NP]([Det]'a')([NN]([Adj])([N]'pie'))))
        was indexed with (1, 1, 0, 0), but has no subtree at (1, 1, 0, 0). NonTerminal leaf-nodes cannot be indexed
        >>> print(t2[1][1,1,1,0,0,1,2,3])
        Traceback (most recent call last):
            ....
        tree_errors.TreeIndexError: Invalid index. The Tree:
        ([VP]([V]'ate')([NP]([Det]'a')([NN]([Adj])([N]'pie'))))
        was indexed with (1, 1, 1, 0, 0, 1, 2, 3), but has no subtree at (1, 1, 1, 0, 0). Terminals do not have child nodes.
        >>> print(t2[1][1,1,1,1])
        Traceback (most recent call last):
            ....
        tree_errors.TreeIndexError: Invalid index. The Tree:
        ([VP]([V]'ate')([NP]([Det]'a')([NN]([Adj])([N]'pie'))))
        was indexed with (1, 1, 1, 1), but has no subtree at (1, 1, 1, 1). Terminals can only be indexed with 0, which returns the terminal content.
        >>> print(t2[1][1,1,1,1,2,5])
        Traceback (most recent call last):
            ....
        tree_errors.TreeIndexError: Invalid index. The Tree:
        ([VP]([V]'ate')([NP]([Det]'a')([NN]([Adj])([N]'pie'))))
        was indexed with (1, 1, 1, 1, 2, 5), but has no subtree at (1, 1, 1, 1). Terminals can only be indexed with 0, which returns the terminal content.
        """
        # If an int is given as index i, make it a singleton tuple
        if isinstance(position, int):
            position = (position,)
        # If the node has no children, or the index at the head of position is
        # out of range for the list self._children, the index points to a node
        # that doesn't exist, and an IndexError should be raised
        if position[0] >= len(self) or position[0] < -len(self):
            # Indexing has failed: gather information to make an informative
            # error message.
            # if `position` is an `int`, convert it to a singleton `tuple`
            if isinstance(position, int):
                position = (position, )
            # define an extra message to explain exactly how it went wrong...
            if self:
                # ... if self has children, but the index is out of range
                err = "Index out of range."
            else:
                # ... or if self is a NonTerminal leaf-node, and hence
                # unindexible
                err = "NonTerminal leaf-nodes cannot be indexed"
            # These are combined into a custom exception. The final argument
            # represents the number of digits in `position` up to the first that
            # does not refer to any node of the tree.
            raise TreeIndexError(err, str(self), position, 1)
        # This try block does the actual work of accessing children and
        # recursing down the tree if needed, but it needs the error-handling in
        # case the recursive call runs into an IndexError. The error message
        # will have, in the call at which the error was first raised, the string
        # representation of the node at which the error was raised, and values
        # of position and wrong based on the index used int *that* call.
        # However, the error message shown in the end should show the node the
        # original call was made on and the index-tuples called on that node -
        # the complete 'position' and the tuple up to the error site, 'wrong'
        try:
            # If a tuple of length 1, use the int in the tuple to index self's
            # list of children
            if len(position) == 1:
                return self._children[position[0]]
            # For all other lengths, use the head of the list to get the correct
            # child, then a recursive call using the tail as the index
            # Note that this:
            #     return self[position[0]][position[1:]]
            # would also work, but would make the call stack almost twice as
            # deep, so I decided against it
            else:
                return self._children[position[0]][position[1:]]
        except TreeIndexError as e:
            # If the recursive call of __getitem__ in the try-block above raises
            # a TreeIndexError, the line below updates it to describe the
            # problem in relation to `self`, rather than the child of `self`
            # that raised the exception. In this way, the exception can be
            # passed back up the stack of recursive calls until it reaches the
            # original call, and display the error giving information about what
            # went wrong in relation to *that* call.
            e.update(str(self), position[0])
            raise e from None

    def findall(self, test):
        return list(filter(test, DFBU(self)))

    def contains_true(self, test):
        tour = DFBU(self)
        try:
            while not test(next(tour)):
                pass
            return True
        except StopIteration:
            return False

    def __setitem__(self, key, value):
        """The children of a Tree can be set using `int` indices, with the i^th
        index setting the i^th child, except with Terminals, where the zeroth
        and only index sets the leaf value. Multiple indices can be provided,
        comma separated, to set a grandchild, great-grandchild, etc. However,
        the path from `self` to site at which `value` is to be inserted must
        go through existing tree-nodes only, otherwise an IndexError is raised.
        If the i^th child slot is not empty, the existing child will be
        replaced and deleted from the `Label`.

        Parameters
        ----------
        key : int
            The index at which the new child is to be placed
        value : Tree or None
            The new child of `self` at index `key`

        Raises
        ------
        TreeIndexError:
            If key is not an int or tuple of ints, or is out of range
        AttributeError:
            If value is not a Tree

        >>> tb = tbs.Treebank()
        >>> t0 = tbs.tree("([S]([NP]([Det]'the')([N]'dog'))([VP]))", treebank = tb)
        >>> t1 = tbs.tree("([VP]([V]'eats')([NP]))", treebank = tb)
        >>> t2 = tbs.tree("([NP]([Det]'the')([N]'pie'))", treebank = tb)
        >>> t0[1] = t1
        >>> t0
        tree("([S]([NP]([Det]'the')([N]'dog'))([VP]([V]'eats')([NP])))")
        >>> t0[1].parent
        tree("([S]([NP]([Det]'the')([N]'dog'))([VP]([V]'eats')([NP])))")
        >>> t0[1, 1] = t2
        >>> t0
        tree("([S]([NP]([Det]'the')([N]'dog'))([VP]([V]'eats')([NP]([Det]'the')([N]'pie'))))")
        >>> t0[1, 1].parent
        tree("([VP]([V]'eats')([NP]([Det]'the')([N]'pie')))")
        >>> # TESTING TERROR HANDLING:
        >>> t0[2] = tbs.tree("([PP]([Prep]'in')([NP]))", treebank = tb)
        Traceback (most recent call last):
            ....
        tree_errors.TreeIndexError: Invalid index. The Tree:
        ([S]([NP]([Det]'the')([N]'dog'))([VP]([V]'eats')([NP]([Det]'the')([N]'pie'))))
        was indexed with (2,), but has no subtree at (2,). Index out of range.
        >>> t0[1,2] = tbs.tree("([PP]([Prep]'in')([NP]))", treebank = tb)
        Traceback (most recent call last):
            ....
        tree_errors.TreeIndexError: Invalid index. The Tree:
        ([S]([NP]([Det]'the')([N]'dog'))([VP]([V]'eats')([NP]([Det]'the')([N]'pie'))))
        was indexed with (1, 2), but has no subtree at (1, 2). Index out of range.
        >>> t0[2,0] = tbs.tree("([PP]([Prep]'in')([NP]))", treebank = tb)
        Traceback (most recent call last):
            ....
        tree_errors.TreeIndexError: Invalid index. The Tree:
        ([S]([NP]([Det]'the')([N]'dog'))([VP]([V]'eats')([NP]([Det]'the')([N]'pie'))))
        was indexed with (2, 0), but has no subtree at (2,). Index out of range.
        >>> t0[1,2,0] = tbs.tree("([PP]([Prep]'in')([NP]))", treebank = tb)
        Traceback (most recent call last):
            ....
        tree_errors.TreeIndexError: Invalid index. The Tree:
        ([S]([NP]([Det]'the')([N]'dog'))([VP]([V]'eats')([NP]([Det]'the')([N]'pie'))))
        was indexed with (1, 2, 0), but has no subtree at (1, 2). Index out of range.
        >>> t0[0,0,0] = tbs.tree("([PP]([Prep]'in')([NP]))", treebank = tb)
        Traceback (most recent call last):
            ....
        tree_errors.TreeIndexError: Invalid index. The Tree:
        ([S]([NP]([Det]'the')([N]'dog'))([VP]([V]'eats')([NP]([Det]'the')([N]'pie'))))
        was indexed with (0, 0, 0), but has no subtree at (0, 0, 0). Cannot assign a child to a Terminal.
        """
        # First, make sure
        # TODO: fix this antipattern: type comparison with type() - check rest of code for this
        if type(key) != int and (type(key) != tuple or (type(key) == tuple and type(key[0]) != int)):
            raise TreeIndexError("indices for Trees can only be ints or tuples of" +
                f" ints: you used a {type(key).__name__}")
        elif not isinstance(value, Tree) and not value is None:
            raise AttributeError("only Trees or None may be set as children " +
                f"of Trees: you used a {type(value).__name__}")
        elif type(key) == int or len(key) == 1:
            k = key if type(key) == int else key[0]
            try:
                # make a variable pointing to the current k'th child of self, or
                # None if the position is empty. This will allow it to be
                # deleted from the treebank after `value` has replaced it in the
                # tree. The variable is used, rather than deleting right away,
                # just in case addressing the child-list in the line below
                # raises an exception. This way, it is unlikely that the
                # treebank will be put in an anomalous state due to an exception
                dead_tree = self[k]
                self.children[k] = value
                self.children[k].parent = self
                if not dead_tree is None:
                    dead_tree.parent = None
                    dead_tree.delete(False)
            except IndexError:
                raise TreeIndexError("Index out of range.", str(self), (k,), 1)
        elif sum(type(k) != int for k in key):
            raise TreeIndexError(
                f"A tuple index must be a tuple of ints only."
            )
        else:
            try:
                # TODO update comments
                # Uses the head of the key to identify the correct child, then
                # recursively call __setitem__ on the child using the tail.
                # Note that this:
                #     self[key[0]][key[1:]] = value
                # would also work, but would require almost twice many recursive
                # calls, so I decided against it
                new_parent = self[key[:-1]]
                if type(new_parent) == NonTerminal:
                    new_parent.children[key[-1]] = value
                    new_parent.children[key[-1]].parent = new_parent
                else:
                    raise TreeIndexError(
                        "Cannot assign a child to a Terminal.", str(self),
                        key[:-1], len(key)
                    )
            except TreeIndexError as e:
                e.update(str(self), child_idx = key[-1])
                raise e from None
            except IndexError:
                raise TreeIndexError("Index out of range.", str(self), key, len(key))


    def copy_out(self, treebank = None, **kwargs):
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
            treebank = tbs.Treebank()
        #### ALSO this needs to be able to handle operators
        # Then create the copied NonTerminal, and recursively copy its children,
        # also to `treebank`: this means, if the function is called with
        # `treebank == None`, the whole tree-fragment will be copied to the same
        # dummy treebank.
        return treebank.N(
            treebank,
            self.label if treebank == self.treebank else treebank.get_label(self.label.class_id), ##-OK both, raw val
            *[c.copy_out(treebank, **kwargs) for c in self],
            operator = self._operator
        )

    def get_leftmost_substition_site(self):
        """
        Returns a SubstitutionSite object pointing to the leftmost non-terminal
        leaf-node: a non-terminal leaf node being a node that is capable of
        having children, but has none.

        Note that this has no counterpart in Terminal: it only makes sense to
        look for NonTerminal leaf nodes at or under NonTerminals

        Returns
        -------
        SubstitutionSite:
            An object containing the parent node of the leftmost nonterminal
            leafnode, the index in the parent at which the substitution site
            occurs, and the label at the substitution site

        >>> t0 = tbs.tree("([S]([NP]([Det])([N]sentence))([VP]([V]parses)([NP]([Det]the)([N]parser))))")
        >>> t1 = tbs.tree("([S]([NP]([Det])([N]sentence))([VP]([V]parses)([NP]([Det]the)([N]))))")
        >>> t2 = tbs.tree("([S]([NP]([Det]the)([N]))([VP]([V]parses)([NP])))")
        >>> t3 = tbs.tree("([S]([NP]([Det])([N]))([VP]([V])([NP]([Det])([N]))))")
        >>> t4 = tbs.tree("([S]([NP])([VP]))")
        >>> t5 = tbs.tree("([S]([NP]([Det]the)([N]sentence))([VP]([V]parses)([NP]([Det]the)([N]parser))))")
        >>> for t in (t0, t1, t2, t3, t4, t5):
        ...     ss = t.get_leftmost_substition_site()
        ...     print(ss.site if ss else ss)
        ...
        ([Det])
        ([Det])
        ([N])
        ([Det])
        ([NP])
        None
        """
        # It is assumed that this is not called directly on zero-depth
        # NonTerminals, but more usually on the root node of a tree
        for i, c in enumerate(self):
            # Don't bother checking any child that is a Terminal, we are only
            # interested in childless NonTerminals
            if not hasattr(c, 'leaf'):
                if not hasattr(c, 'children') or not len(c.children):
                    # If the child is childless, it is a nonterminal leafnode,
                    # which we return immediately as the first one we find is
                    # the leftmost. The object returned, however, is not the
                    # node itself, but a dataclass object containing `self` (the
                    # parent of the substitution site), i, the index of the
                    # substitution site in self, and the label at the
                    # substitution site, as this determines what can be
                    # substituted there. SubstitutionSite also has a method
                    # which takes a subtree as an argument, and performs the
                    # substitution if the label matches
                    return SubstitutionSite(self, i, c.label)
                # otherwise, recursively call this function
                ss = c.get_leftmost_substition_site()
                # If a ss is found, return it. This function doesn't reach a
                # `return` line if no nonterminal leaf is found, so will return
                # `None`, meaning lnl will be None if no ss is found my the
                # recursive call, so the conditional won't be triggered
                if ss:
                    return ss

    def get_all_substitition_sites(self):
        """
        Returns the all non-terminal leaf-node: a non-terminal leaf node
        being a node that is capable of having children, but has none.

        Note that this has no counterpart in Terminal: it only makes sense to
        look for NonTerminal leaf nodes at or under NonTerminals

        Returns
        -------
        list of SubstitutionSites:
            A list of dataclass objects containing the parent nodes of each
            nonterminal leafnode, its index in its parent, and its label.

        >>> t0 = tbs.tree("([S]([NP]([Det])([N]sentence))([VP]([V]parses)([NP]([Det]the)([N]parser))))")
        >>> t1 = tbs.tree("([S]([NP]([Det])([N]sentence))([VP]([V]parses)([NP]([Det]the)([N]))))")
        >>> t2 = tbs.tree("([S]([NP]([Det]the)([N]))([VP]([V]parses)([NP])))")
        >>> t3 = tbs.tree("([S]([NP]([Det])([N]))([VP]([V])([NP]([Det])([N]))))")
        >>> t4 = tbs.tree("([S]([NP])([VP]))")
        >>> t5 = tbs.tree("([S]([NP]([Det]the)([N]sentence))([VP]([V]parses)([NP]([Det]the)([N]parser))))")
        >>> for t in (t0, t1, t2, t3, t4, t5):
        ...     print([s.site for s in t.get_all_substitition_sites()])
        ...
        [tree("([Det])")]
        [tree("([Det])"), tree("([N])")]
        [tree("([N])"), tree("([NP])")]
        [tree("([Det])"), tree("([N])"), tree("([V])"), tree("([Det])"), tree("([N])")]
        [tree("([NP])"), tree("([VP])")]
        []
        """
        # Initialise an empty list: try to put substitution sites in it, if
        # there are any, if not return it empty
        sites = []
        # It is assumed that this is not called directly on zero-depth
        # NonTerminals, but more usually on the root node of a tree.
        # Loop over the enumeration of self, as we will need the index of any
        # substitution site found
        for i, c in enumerate(self):
            # Ignore Terminals
            if not hasattr(c, 'leaf'):
                # If a child is childess, it's a substitution site and can be
                # added to the list
                if not hasattr(c, 'children') or not len(c.children):
                    # The object returned, however, is not the
                    # node itself, but a dataclass object containing `self` (the
                    # parent of the substitution site), i, the index of the
                    # substitution site in self, and the label at the
                    # substitution site, as this determines what can be
                    # substituted there. SubstitutionSite also has a method
                    # which takes a subtree as an argument, and performs the
                    # substitution if the label matches
                    sites += [SubstitutionSite(self, i, c.label)]
                # Otherwise, recursively call on the child, and see if there are
                # any substitution sites further down the tree
                sites += c.get_all_substitition_sites()
        return sites

    def __iadd__(self, other):
        """Magic method used for operator overloading on `+=`. Either an alias
        for strict_addition (substitution on the leftmost nonterminal leaf node
        only: throws error if Labels unmatched) if the `STRICT_ADDITION` class
        attribute is set to `True`; otherwise an alias for permissive_addition
        (substitution on leftmost Label-matched nonterminal leaf node: only
        throws an error if no matching nonterminal leaf node exists). `self`
        is modified in place, and the added children will belong to the same
        treebank as self.

        Returns
        -------
        Tree
            self

        Raises
        ------
        TypeError
            If the right operand is not a Tree

        >>> dop3 = tbs.Treebank(ops.CONCAT)
        >>> t0 = tbs.tree("([S]([PN]'I')([VP]))", treebank = dop3)
        >>> t1 = tbs.tree("([VP]([V]'pet')([N]'cats'))", treebank = dop3)
        >>> t0+=t1
        >>> dop3.print_all_labels()
        ======================
        S:
        ([S]([PN]'I')([VP]([V]'pet')([N]'cats')))
        ======================
        PN:
        ([PN]'I')
        ======================
        VP:
        ([VP]([V]'pet')([N]'cats'))
        ======================
        V:
        ([V]'pet')
        ======================
        N:
        ([N]'cats')
        ======================
        """
        if not isinstance(other, Tree) :
            raise TypeError(
                f"Cannot add a {type(other)} to a NonTerminal; you can " +
                f"only add another Tree."
            )
        return self.add_strict(other) if type(self.treebank).STRICT_ADDITION else self.add_permissive(other)

    def __add__(self, other):
        """Magic method used for operator overloading on `+`. Either an alias
        for strict_addition (substitution on the leftmost nonterminal leaf node
        only: throws error if Labels unmatched) if the `STRICT_ADDITION` class
        attribute is set to `True`; otherwise an alias for permissive_addition
        (substitution on leftmost Label-matched nonterminal leaf node: only
        throws an error if no matching nonterminal leaf node exists). `self`
        is copied into a dummy treebank, and the added children will belong to
        the same dummy treebank as `self.copy()`.

        Returns
        -------
        Tree :
            self.copy() with other.copy() substituted at leftmost (if STRICT) or
            leftmost matching (if PERMISSIVE) nonterminal leaf node.

        Raises
        ------
        TypeError
            If the right operand is not a Tree

        >>> dop = tbs.Treebank(ops.CONCAT)
        >>> t0 = tbs.tree("([S]([PN]'I')([VP]))", treebank = dop)
        >>> t1 = tbs.tree("([VP]([V]'pet')([N]'cats'))", treebank = dop)
        >>> dop.print_all_labels()
        ======================
        S:
        ([S]([PN]'I')([VP]))
        ======================
        PN:
        ([PN]'I')
        ======================
        VP:
        ([VP])
        ([VP]([V]'pet')([N]'cats'))
        ======================
        V:
        ([V]'pet')
        ======================
        N:
        ([N]'cats')
        ======================
        >>> t0 + t1
        tree("([S]([PN]'I')([VP]([V]'pet')([N]'cats')))")
        >>> dop.print_all_labels()
        ======================
        S:
        ([S]([PN]'I')([VP]))
        ([S]([PN]'I')([VP]([V]'pet')([N]'cats')))
        ======================
        PN:
        ([PN]'I')
        ([PN]'I')
        ======================
        VP:
        ([VP])
        ([VP]([V]'pet')([N]'cats'))
        ([VP]([V]'pet')([N]'cats'))
        ======================
        V:
        ([V]'pet')
        ([V]'pet')
        ======================
        N:
        ([N]'cats')
        ([N]'cats')
        ======================
        """

        # copy self to dummy treebank
        cp = self.copy()
        # use __iadd__ to add `other`
        cp += other.copy()
        return cp

    def add_strict(self, other):
        """Adds Trees t1 and t2 by substituting t2 for the leftmost non-terminal
        leaf-node of t1, iff they have the same label. Changes t1 in place, but
        copies children from t2; returns changed t1.

        Parameters
        ----------
        other : Tree
            `Terminal` or `Nonterminal` to be substituted.

        Returns
        -------
        self : `self`, changed in place.

        Raises
        ------
        ValueError
            If there is no nonterminal leafnode in `self`, or the leftmost
            nonterminal leafnode has a label that does not match the root label
            of `other`.

        >>> tb = tbs.Treebank()
        >>> t0 = tbs.tree("([S]([NP]([Det])([N]sentence))([VP]([V]parses)([NP]([Det]the)([N]parser))))", treebank = tb)
        >>> t1 = tbs.tree("([S]([NP]([Det])([N]sentence))([VP]([V]parses)([NP]([Det]the)([N]))))", treebank = tb)
        >>> t2 = tbs.tree("([S]([NP]([Det]the)([N]))([VP]([V]parses)([NP])))", treebank = tb)
        >>> t3 = tbs.tree("([S]([NP]([Det])([N]))([VP]([V])([NP]([Det])([N]))))", treebank = tb)
        >>> t4 = tbs.tree("([S]([NP])([VP]))", treebank = tb)
        >>> t5 = tbs.tree("([S]([NP]([Det]the)([N]sentence))([VP]([V]parses)([NP]([Det]the)([N]parser))))", treebank = tb)
        >>> det = tbs.tree("([Det]a)", treebank = tb)
        >>> n = tbs.tree("([N]pudding)", treebank = tb)
        >>> np = tbs.tree("([NP]([Det]the)([NN]([Adj]big)([N]banana)))", treebank = tb)
        >>> v = tbs.tree("([V]badoinks)", treebank = tb)
        >>> vp = tbs.tree("([VP]([VP]([V]is)([NP]([Det]your)([N]god)))([Adv]now))", treebank = tb)
        >>> print(copy(t0+det))
        ([S]([NP]([Det]'a')([N]'sentence'))([VP]([V]'parses')([NP]([Det]'the')([N]'parser'))))
        >>> print(copy(t1+det))
        ([S]([NP]([Det]'a')([N]'sentence'))([VP]([V]'parses')([NP]([Det]'the')([N]))))
        >>> print(copy(t1+n))
        Traceback (most recent call last):
            ....
        ValueError: Subtree ([N]'pudding') with root label [N] cannot be substituted at index 0 of ([NP]([Det])([N]'sentence')) due to a mismatch with substitution site label [Det].
        >>> print(t1+det+n)
        ([S]([NP]([Det]'a')([N]'sentence'))([VP]([V]'parses')([NP]([Det]'the')([N]'pudding'))))
        >>> print(t1+n+det)
        Traceback (most recent call last):
            ....
        ValueError: Subtree ([N]'pudding') with root label [N] cannot be substituted at index 0 of ([NP]([Det])([N]'sentence')) due to a mismatch with substitution site label [Det].
        >>> print(t2+n)
        ([S]([NP]([Det]'the')([N]'pudding'))([VP]([V]'parses')([NP])))
        >>> print(t2+n+np)
        ([S]([NP]([Det]'the')([N]'pudding'))([VP]([V]'parses')([NP]([Det]'the')([NN]([Adj]'big')([N]'banana')))))
        >>> print(t3+det+n+v+det+n)
        ([S]([NP]([Det]'a')([N]'pudding'))([VP]([V]'badoinks')([NP]([Det]'a')([N]'pudding'))))
        >>> print(t4+np+vp)
        ([S]([NP]([Det]'the')([NN]([Adj]'big')([N]'banana')))([VP]([VP]([V]'is')([NP]([Det]'your')([N]'god')))([Adv]'now')))
        >>> print((t4+np+vp)())
        (('the', ('big', 'banana')), (('is', ('your', 'god')), 'now'))
        >>> print(t5+n)
        Traceback (most recent call last):
            ....
        ValueError: Cannot add Trees t1 and t2; t1 has no non-terminal leaf-nodes
        """
        # get leftmost nonterminal leaf node, if it exists. Note, the object
        # returned by get_leftmost_substition_site, however, is not the node
        # itself, but a dataclass object containing `self` (the parent of the
        # substitution site), i, the index of the substitution site in self, and
        # the label at the substitution site, as this determines what can be
        # substituted there.
        ss = self.get_leftmost_substition_site()
        if ss:
            # call method at SubstitutionSite which takes a subtree as an
            # argument, and performs the substitution if the label matches, or
            # raises an exception if it doesn't
            ss.perform_substitution(other)
            return self
        else:
            # If no substitution site found, raise error
            raise ValueError(
                f"Cannot add Trees t1 and t2; t1 has no non-terminal leaf-nodes"
            )

    def add_permissive(self, other):
        """Adds Trees t1 and t2 by substituting t2 for the leftmost non-terminal
        leaf-node of t1, iff they have the same label. Changes t1 in place, but
        copies children from t2; returns changed t1.

        Parameters
        ----------
        other : Tree
            `Terminal` or `Nonterminal` to be substituted.

        Returns
        -------
        NonTerminal: `self`, changed in place.

        Raises
        ------
        ValueError
            If there is no nonterminal leafnode in `self` with a label that
            matches the root label of `other`.

        >>> tbs.Treebank.STRICT_ADDITION = False
        >>> tbs.Treebank.STRICT_ADDITION
        False
        >>> tb = tbs.Treebank()
        >>> t0 = tbs.tree("([S]([NP]([Det])([N]sentence))([VP]([V]parses)([NP]([Det]the)([N]parser))))", treebank = tb)
        >>> t1 = tbs.tree("([S]([NP]([Det])([N]sentence))([VP]([V]parses)([NP]([Det]the)([N]))))", treebank = tb)
        >>> t2 = tbs.tree("([S]([NP]([Det]the)([N]))([VP]([V]parses)([NP])))", treebank = tb)
        >>> t3 = tbs.tree("([S]([NP]([Det])([N]))([VP]([V])([NP]([Det])([N]))))", treebank = tb)
        >>> t4 = tbs.tree("([S]([NP])([VP]))", treebank = tb)
        >>> t5 = tbs.tree("([S]([NP]([Det]the)([N]sentence))([VP]([V]parses)([NP]([Det]the)([N]parser))))", treebank = tb)
        >>> det = tbs.tree("([Det]a)", treebank = tb)
        >>> n = tbs.tree("([N]pudding)", treebank = tb)
        >>> np = tbs.tree("([NP]([Det]the)([NN]([Adj]big)([N]banana)))", treebank = tb)
        >>> v = tbs.tree("([V]badoinks)", treebank = tb)
        >>> vp = tbs.tree("([VP]([VP]([V]is)([NP]([Det]your)([N]god)))([Adv]now))", treebank = tb)
        >>> print(copy(t0+det))
        ([S]([NP]([Det]'a')([N]'sentence'))([VP]([V]'parses')([NP]([Det]'the')([N]'parser'))))
        >>> print(copy(t1+det))
        ([S]([NP]([Det]'a')([N]'sentence'))([VP]([V]'parses')([NP]([Det]'the')([N]))))
        >>> print(copy(t1)+copy(n))
        ([S]([NP]([Det])([N]'sentence'))([VP]([V]'parses')([NP]([Det]'the')([N]'pudding'))))
        >>> print(copy(t1)+copy(det)+copy(n))
        ([S]([NP]([Det]'a')([N]'sentence'))([VP]([V]'parses')([NP]([Det]'the')([N]'pudding'))))
        >>> print(copy(t1)+copy(n)+copy(det))
        ([S]([NP]([Det]'a')([N]'sentence'))([VP]([V]'parses')([NP]([Det]'the')([N]'pudding'))))
        >>> print(copy(t2)+copy(n))
        ([S]([NP]([Det]'the')([N]'pudding'))([VP]([V]'parses')([NP])))
        >>> print(copy(t2)+copy(n)+copy(np))
        ([S]([NP]([Det]'the')([N]'pudding'))([VP]([V]'parses')([NP]([Det]'the')([NN]([Adj]'big')([N]'banana')))))
        >>> print(copy(t3)+copy(det)+copy(n)+copy(v)+copy(det)+copy(n))
        ([S]([NP]([Det]'a')([N]'pudding'))([VP]([V]'badoinks')([NP]([Det]'a')([N]'pudding'))))
        >>> print(copy(t4)+copy(np)+copy(vp))
        ([S]([NP]([Det]'the')([NN]([Adj]'big')([N]'banana')))([VP]([VP]([V]'is')([NP]([Det]'your')([N]'god')))([Adv]'now')))
        >>> print((copy(t4)+copy(np)+copy(vp))())
        (('the', ('big', 'banana')), (('is', ('your', 'god')), 'now'))
        >>> print(copy(t5)+copy(n))
        Traceback (most recent call last):
            ....
        ValueError: Cannot add Trees t1 ([S]([NP]([Det]'the')([N]'sentence'))([VP]([V]'parses')([NP]([Det]'the')([N]'parser')))) and t2 ([N]'pudding'); Cannot add Trees t1 & t2; t1 has no nonterminal leafnodes
        >>> print(copy(t0)+tbs.tree("([IDEK]asdfg)"))
        Traceback (most recent call last):
            ....
        ValueError: Cannot add Trees t1 ([S]([NP]([Det])([N]'sentence'))([VP]([V]'parses')([NP]([Det]'the')([N]'parser')))) and t2 ([IDEK]'asdfg'); no nonterminal leafnode of t1 matches the root label of t2
        >>> tbs.Treebank.STRICT_ADDITION = True # Put these back where I found them, or other tests get broken
        """
        # Get list of all substitution sites. Note, the objects
        # returned by get_all_substitition_sites, however, are not the nodes
        # themselve, but dataclass objects containing the parent of the
        # substitution site, i, the index of the substitution site in the
        # parent, and the label at the substitution site, as this determines
        # what can be substituted there.
        subsites = self.get_all_substitition_sites()
        # Search the list for the first site matching the label of the subtree
        for ss in subsites:
            # If/when found ...
            if ss.label.class_id == other.label.class_id: ##-OK both, raw val
                # call method at SubstitutionSite which takes a subtree as an
                # argument, and performs the substitution if the label matches.
                # Go on to the next is it doesn't and...
                ss.perform_substitution(other.copy_out(self.treebank))
                return self
        else:
            # ...if no suitable node is found, raise error:
            raise ValueError(
                f"Cannot add Trees t1 {self} and t2 {other}; " +
                (
                    "no nonterminal leafnode of t1 matches the root label of t2"
                    if subsites else
                    "Cannot add Trees t1 & t2; t1 has no nonterminal leafnodes"
                )
            )

    def __call__(self):
        """Magic method that makes NonTerminals Callable: but really it's just
        calling label (Labels are also callable), which is just calling an
        Operator (also also callable).
        """
        return self._operator(*[c() for c in self])

    def index_of(self, child: Tree, descendants=False, strict=True) -> Union[int, Tuple[int]]:
        """Takes a child node and returns its position in the `NonTerminal`'s
        children, or, if `descendants` is True, it's descendants. If a `Tree`
        is passed to `child` that is not a child/descendant of `self`, returns
        the empty `tuple` if not `strict`, otherwise raises a `ValueError`.

        Parameters
        ----------
        child : Tree
            The `Tree` we are looking for.
        descendants : bool
            If `True`, search all the descendants of `self`, otherwise, just
            search its children. False by default.
        strict : bool
            If `True`, raises a ValueError if the child is not found: otherwise
            returns `()`.

        Returns
        -------
        int or tuple of ints
            Index. If `descendants` is `False` and `strict` is `True`, returns
            the index of `child` in `self` as an `int`: otherwise, it returns a
            tuple, giving the path from `self` to `child` - which is empty if
            there is no path and `strict` is false.

        Raises
        ------
        ValueError
            If `child` is not found and `strict` is `True`.

        >>> tb = tbs.Treebank()
        >>> tree0 = tbs.tree("([S]([NP]([PN]'she'))([VP]([V]'saw')([S]([NP]([NP]([Det]'the')([N]'dog'))([PP]([Prep]'with')([NP]([Det]'the')([N]'telescope'))))([VP]([V]'observing')([NP]([PN]'neptune'))))))", treebank = tb)
        >>> tree1 = tbs.tree("([NP]([Det]'the')([N]'dog'))", treebank = tb)
        >>> tree2 = tree0[1,1,0,0]
        >>> for i, t in enumerate(tree0):
        ...     print(i == tree0.index_of(t))
        True
        True
        >>> tree0.index_of(tree2, True, False)
        (1, 1, 0, 0)
        >>> tree0.index_of(tree1, True, False)
        ()
        >>> tree0.index_of(tree2, False, False)
        ()
        >>> tree0.index_of(tree1, False, False)
        ()
        >>> tree0.index_of(tree2, True, True)
        (1, 1, 0, 0)
        >>> tree0.index_of(tree1, True, True)
        Traceback (most recent call last):
            ....
        ValueError: Descendant not found
        >>> tree0.index_of(tree2, False, True)
        Traceback (most recent call last):
            ....
        ValueError: Child not found
        >>> tree0.index_of(tree1, False, True)
        Traceback (most recent call last):
            ....
        ValueError: Child not found
        """
        # if we are looking through the entire tree...
        # (note that in this condition all our indices are tuples, indicating a
        # path through the tree.
        if descendants:
            # ...then first, consider the case where the child we're looking for
            # is the direct child of self. In this case we can avoid code
            # duplication by calling index_of again, but this time taking the
            # !descendants branch. strict is false in the index_of call below,
            # even if the originating call is strict, because if we don't find
            # child in the children of self, we don't want to raise an
            # exception until we know it's not in the grandchildren, great-
            # grandchildren, etc. Because strict is false, we know that if child
            # is found by this call, it will return a tuple, which is what we
            # want
            idx = self.index_of(child, strict=False)
            # idx is either a singleton tuple (child is a child of self) or
            # empty. If singleton...
            if len(idx):
                # ...return it: otherwise...
                return idx
            else:
                # ...we need to look further down the tree. Loop with enumerate
                # because we need the index, in case the child is found. Since
                # we're using 'child' for the Tree we're looking for, we'll use
                # 'kid' for the actual children of self
                for i, kid in enumerate(self):
                    # Terminals don't have children, so the search is only
                    # continued if kid is a NonTerminal...
                    if isinstance(kid, NonTerminal):
                        # ...again, a recursive call is wanted here, but this
                        # time with `descendants` set to true: strict is still
                        # set to false in the recursive call: if the originating
                        # call is strict, this loop will complete without
                        # returning, in which case the following code will
                        # either return empty or raise ValueError.
                        idx = kid.index_of(child, True, False)
                        # If idx is not empty, child was found in the
                        # descendants of 'kid' at the tuple index at kid[*idx]:
                        # e.g., if (2,0,1) is returned, child == kid[2,0,1], so
                        # ...
                        if len(idx):
                            # given that kid == self[i], child == self[i, *idx],
                            # e.g., if kid = self[1] and child == kid[2,0,1],
                            # then child = self[1,2,0,1]
                            return (i,) + idx
                # If the function call is still running here, no child was found
                # and the proper behaviour is determined by whether `strict`.
                if not strict:
                    return ()
                else:
                    # The error specifies descendant since the function searched
                    # the descendants, not just thie children of self
                    raise ValueError("Descendant not found")
        # If not `descendants`, things are simpler. However, note that this is
        # the only branch of the function code that searches directly in the
        # children of a NonTerminal for a child: the descendant-searching
        # branch uses a call to this branch to check for immediate children.
        else:
            # Note that we are looking specifically for a child that is
            # *identical* to `child`, not just equal: however, the list.index()
            # function uses equality. Therefore, we convert `child`...
            ch_id = id(child)
            # ...and the list of children to ids
            k_ids = [id(k) for k in self._children]
            # if child's id is found in the list of the kids ids...
            if ch_id in k_ids:
                # ...then get the index it was found at...
                idx = k_ids.index(ch_id)
                # ...so return it, either as an int or a singleton tuple.
                return idx if strict else (idx,)
            # Otherwise, the child isn't found, and the appropriate response is
            # determined by `strict`.
            elif not strict:
                return ()
            else:
                raise ValueError("Child not found")

    def delete(self, _top = True):
        """Completely removes the NonTerminal from the treebank, along with all
        its descendants. Just to make sure the GC gets what needs to be got, as
        well as deleting all references to the tree in the Labels, it also
        typically (_top = False) removes all references held by the nodes
        of the tree to each other. However, the default behaviour of the
        function

        Parameters
        ----------
        _top : bool
            If `True`, will cause the parent node of `self` to fill `self's`
            position in the list of children with an non-terminal leaf node (a
            `NonTerminal` with no children). This is the default value, but is
            set to false when `delete` recursively calls itself on the chidren
            of `self`

        >>> tb = tbs.Treebank()
        >>> tb.print_all_labels()
        ======================
        >>> tree0 = tbs.tree("([S]([NP]([PN]'she'))([VP]([V]'saw')([S]([NP]([NP]([Det]'the')([N]'dog'))([PP]([Prep]'with')([NP]([Det]'the')([N]'telescope'))))([VP]([V]'observing')([NP]([PN]'neptune'))))))", treebank = tb)
        >>> tb.print_all_labels()
        ======================
        S:
        ([S]([NP]([NP]([Det]'the')([N]'dog'))([PP]([Prep]'with')([NP]([Det]'the')([N]'telescope'))))([VP]([V]'observing')([NP]([PN]'neptune'))))
        ([S]([NP]([PN]'she'))([VP]([V]'saw')([S]([NP]([NP]([Det]'the')([N]'dog'))([PP]([Prep]'with')([NP]([Det]'the')([N]'telescope'))))([VP]([V]'observing')([NP]([PN]'neptune'))))))
        ======================
        NP:
        ([NP]([PN]'she'))
        ([NP]([Det]'the')([N]'dog'))
        ([NP]([Det]'the')([N]'telescope'))
        ([NP]([NP]([Det]'the')([N]'dog'))([PP]([Prep]'with')([NP]([Det]'the')([N]'telescope'))))
        ([NP]([PN]'neptune'))
        ======================
        PN:
        ([PN]'she')
        ([PN]'neptune')
        ======================
        VP:
        ([VP]([V]'observing')([NP]([PN]'neptune')))
        ([VP]([V]'saw')([S]([NP]([NP]([Det]'the')([N]'dog'))([PP]([Prep]'with')([NP]([Det]'the')([N]'telescope'))))([VP]([V]'observing')([NP]([PN]'neptune')))))
        ======================
        V:
        ([V]'saw')
        ([V]'observing')
        ======================
        Det:
        ([Det]'the')
        ([Det]'the')
        ======================
        N:
        ([N]'dog')
        ([N]'telescope')
        ======================
        PP:
        ([PP]([Prep]'with')([NP]([Det]'the')([N]'telescope')))
        ======================
        Prep:
        ([Prep]'with')
        ======================
        >>> tree0[0].delete()
        >>> tree0
        tree("([S]([NP])([VP]([V]'saw')([S]([NP]([NP]([Det]'the')([N]'dog'))([PP]([Prep]'with')([NP]([Det]'the')([N]'telescope'))))([VP]([V]'observing')([NP]([PN]'neptune'))))))")
        >>> tb.print_all_labels()
        ======================
        S:
        ([S]([NP]([NP]([Det]'the')([N]'dog'))([PP]([Prep]'with')([NP]([Det]'the')([N]'telescope'))))([VP]([V]'observing')([NP]([PN]'neptune'))))
        ([S]([NP])([VP]([V]'saw')([S]([NP]([NP]([Det]'the')([N]'dog'))([PP]([Prep]'with')([NP]([Det]'the')([N]'telescope'))))([VP]([V]'observing')([NP]([PN]'neptune'))))))
        ======================
        NP:
        ([NP]([Det]'the')([N]'dog'))
        ([NP]([Det]'the')([N]'telescope'))
        ([NP]([NP]([Det]'the')([N]'dog'))([PP]([Prep]'with')([NP]([Det]'the')([N]'telescope'))))
        ([NP]([PN]'neptune'))
        ([NP])
        ======================
        PN:
        ([PN]'neptune')
        ======================
        VP:
        ([VP]([V]'observing')([NP]([PN]'neptune')))
        ([VP]([V]'saw')([S]([NP]([NP]([Det]'the')([N]'dog'))([PP]([Prep]'with')([NP]([Det]'the')([N]'telescope'))))([VP]([V]'observing')([NP]([PN]'neptune')))))
        ======================
        V:
        ([V]'saw')
        ([V]'observing')
        ======================
        Det:
        ([Det]'the')
        ([Det]'the')
        ======================
        N:
        ([N]'dog')
        ([N]'telescope')
        ======================
        PP:
        ([PP]([Prep]'with')([NP]([Det]'the')([N]'telescope')))
        ======================
        Prep:
        ([Prep]'with')
        ======================
        >>> tree0[1,1,1].delete()
        >>> tree0
        tree("([S]([NP])([VP]([V]'saw')([S]([NP]([NP]([Det]'the')([N]'dog'))([PP]([Prep]'with')([NP]([Det]'the')([N]'telescope'))))([VP]))))")
        >>> tb.print_all_labels()
        ======================
        S:
        ([S]([NP]([NP]([Det]'the')([N]'dog'))([PP]([Prep]'with')([NP]([Det]'the')([N]'telescope'))))([VP]))
        ([S]([NP])([VP]([V]'saw')([S]([NP]([NP]([Det]'the')([N]'dog'))([PP]([Prep]'with')([NP]([Det]'the')([N]'telescope'))))([VP]))))
        ======================
        NP:
        ([NP]([Det]'the')([N]'dog'))
        ([NP]([Det]'the')([N]'telescope'))
        ([NP]([NP]([Det]'the')([N]'dog'))([PP]([Prep]'with')([NP]([Det]'the')([N]'telescope'))))
        ([NP])
        ======================
        PN:
        ======================
        VP:
        ([VP]([V]'saw')([S]([NP]([NP]([Det]'the')([N]'dog'))([PP]([Prep]'with')([NP]([Det]'the')([N]'telescope'))))([VP])))
        ([VP])
        ======================
        V:
        ([V]'saw')
        ======================
        Det:
        ([Det]'the')
        ([Det]'the')
        ======================
        N:
        ([N]'dog')
        ([N]'telescope')
        ======================
        PP:
        ([PP]([Prep]'with')([NP]([Det]'the')([N]'telescope')))
        ======================
        Prep:
        ([Prep]'with')
        ======================
        >>> tree0[1,1].delete()
        >>> tree0
        tree("([S]([NP])([VP]([V]'saw')([S])))")
        >>> tb.print_all_labels()
        ======================
        S:
        ([S]([NP])([VP]([V]'saw')([S])))
        ([S])
        ======================
        NP:
        ([NP])
        ======================
        PN:
        ======================
        VP:
        ([VP]([V]'saw')([S]))
        ======================
        V:
        ([V]'saw')
        ======================
        Det:
        ======================
        N:
        ======================
        PP:
        ======================
        Prep:
        ======================
        >>> tree0.delete()
        >>> tb.print_all_labels()
        ======================
        S:
        ======================
        NP:
        ======================
        PN:
        ======================
        VP:
        ======================
        V:
        ======================
        Det:
        ======================
        N:
        ======================
        PP:
        ======================
        Prep:
        ======================
        """
        for child in self:
            if isinstance(child, Tree):
                child.delete(False)
        if self.parent:
            if _top:
                self.parent.children[self.parent.index_of(self)] = NonTerminal(
                    self.treebank, self.label, operator = self._operator
                )
            else:
                self.parent.children[self.parent.index_of(self)] = None
            self.parent = None
        # Note that Label.remove_node() also removes the Label from the node,
        # so no need to do that here
        if self.label:
            self.label.remove_node(self)


class Terminal(Tree):
    """A general class of terminal node, agnostic as to whether we are using
    Trees to represent strings, functions, music, visual images, motor scores,
    etc.

    TODO: add a metadata attribute

    Attributes:
        _label (Label): Inherited from Tree. Node label determining
            which subtrees can be substituted where. Set and got by the `label`
            @property
        parent (NonTerminal or None): Inherited from Tree. The node
            dirctly above the present in the tree. Each child has a reference to
            its parent, and each parent has references to all its children.
        treebank (Treebank): Every Tree belongs to a Treebank, which defines the
            set of Labels that `Tree`, its parents, and its children can take.
        leaf: The terminal content of this branch of the tree, like a word in
            syntactic parsing, or a constant or variable in Genetic Programming
    """
    def __init__(self, treebank, label, leaf, operator=None, metadata=None):
        self._operator = operator
        if isinstance(leaf, str):
            try:
                self.leaf = eval(leaf)
            except Exception:
                self.leaf = leaf
        else:
            self.leaf = leaf
        super().__init__(treebank, label)

    def __str__(self):
        """Readable string representation of a Terminal. This consists of a pair
        of parentheses containing a representation of the node's label (label
        name in square brackets), followed by the leaf value, e.g.:

        ([N]'telescope').
        """
        return f"({self.label if self.label else ''}{repr(self.leaf)})"

    def __eq__(self, other):
        """Magic method to operator-overload `==` and `!=`

        Returns
        -------
            bool: True if class, label and leaf are the same, else False.

        >>> # The tests on NonTerminal.__eq__ also test this
        >>> tb = tbs.Treebank()
        >>> tbs.tree('([N]bridge)', tb) == tbs.tree('([N]bridge)', tb)
        True
        >>> tbs.tree('([N]bridge)', tb) == tbs.tree('([V]bridge)', tb)
        False
        >>> tbs.tree('([N]bridge)', tb) == tbs.tree('([N]man)', tb)
        False
        >>> tbs.tree('([N]bridge)') == tbs.tree('([N]bridge)')
        False
        >>> tbs.tree('([N]bridge)') == tbs.tree('([V]bridge)')
        False
        >>> tbs.tree('([N]bridge)') == tbs.tree('([N]man)')
        False
        """
        return self.__class__ == other.__class__ and self.leaf == other.leaf and self.label == other.label

    def __len__(self) -> int:
        """All nodes have a length: Terminals have length 1, while NonTerminals
        have a length equal to the number of children

        Returns
        -------
        int:  1
        """
        return 1

    def depth(self) -> int:
        """All terminals have depth 1"""
        return 1

    def width(self) -> int:
        """All terminals have width 1"""
        return 1

    def size(self) -> int:
        """All terminals have size 1"""
        return 1

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

        >>> from test_trees import test_trees
        >>> t = test_trees()
        >>> for x in t:
        ...     print(x.to_LaTeX())
        ...
        \Tree [.N poo ]
        \Tree [.S [.N sentence ] [.VP [.V parses ] [.N parser ] ] ]
        \Tree [.S [.N sentence ] [.VP [.V parses ] [.N parser ] ] ]
        \Tree [.S [.N sentence ] [.VP [.V parses ] [.N parser ] ] ]
        \Tree [.S [.NP [.Det the ] [.N sentence ] ] [.VP [.V parses ] [.NP [.Det the ] [.N parser ] ] ] ]
        \Tree [.S [.N word ] [.VP [.V parses ] [.N parser ] ] ]
        \Tree [.S [.N sentence ] [.VP [.V parses ] [.V parser ] ] ]
        \Tree [.S [.X x ] [.Y y ] ]
        """
        # prepends \Tree if needed
        LaTeX = r"\Tree " if not (hasattr(self, 'parent') and self.parent) or top else ""
        # LaTeX of the Label is . followed by the label name
        LaTeX += f"[{self.label.to_LaTeX()} {self.leaf} ] "
        return LaTeX.strip() if top else LaTeX

    def __call__(self):
        """All nodes are callable, but on Terminals it just returns the leaf.

        Returns
        -------
            The leaf value.

        >>> from test_trees import test_trees
        >>> t = test_trees()[0]
        >>> print(t())
        poo
        """
        return self._operator(self.leaf) if self._operator else self.leaf

    def __getitem__(self, position):
        """Returns leaf if index is 0, throws error otherwise.

        Returns
        -------
            leaf

        Raises:
            TreeIndexError: If index is not 0

        >>> t = tbs.tree("([N]'sentence')")
        >>> print(t[0])
        sentence
        >>> print(t[0, 0])
        Traceback (most recent call last):
            ....
        tree_errors.TreeIndexError: Invalid index. The Tree:
        ([N]'sentence')
        was indexed with (0, 0), but has no subtree at (0, 0). Terminals do not have child nodes.
        >>> print(t[1])
        Traceback (most recent call last):
            ....
        tree_errors.TreeIndexError: Invalid index. The Tree:
        ([N]'sentence')
        was indexed with (1,), but has no subtree at (1,). Terminals can only be indexed with 0, which returns the terminal content.
        >>> print(t[1, 0])
        Traceback (most recent call last):
            ....
        tree_errors.TreeIndexError: Invalid index. The Tree:
        ([N]'sentence')
        was indexed with (1, 0), but has no subtree at (1,). Terminals can only be indexed with 0, which returns the terminal content.
        """
        # If the index at this point is anything other than 0 or (0,), it points
        # to a node that isn't there, and a TreeIndexError should be raised.
        # This shows the string representation of the tree where the error was
        # raised (`str(self)`); the erroneous index (`position`); the index i
        # such that position[:i] is the tree index up to and including the first
        # erroneous element; but no more (`cutoff`), and a message indicating
        # what went wrong (`err`)
        if not position in (0, (0,)):
            if type(position) == int:
                position = (position, )
            if position[0] == 0:
                cutoff = 2
                err = "Terminals do not have child nodes."
            else:
                cutoff = 1
                err = "Terminals can only be indexed with 0, which returns the terminal content."
            raise TreeIndexError(err, str(self), position, cutoff)
        return self.leaf


    def copy_out(self, treebank = None, **kwargs):
        """Generates a deep copy of a Terminal: same Labels, and  same content:
        but a distinct object in memory from the original. If `treebank` is a
        Treebank, The new Terminal will be copied into `treebank`. If `treebank`
        is `None`, a dummy treebank will be created, and the Terminal will be
        copied into that.

        Parameters
        ----------
            treebank:
                The target treebank the tree is being copied into
            kwargs:
                Not used, but needed for compatibility with subclasses

        Returns
        -------
            Terminal: copy of original tree
        """
        # If `treebank` is not provided...
        if not treebank:
            # ...make a dummy treebank for the copied Terminal to live in
            treebank = tbs.Treebank()
        # return the copy Terminal, with `treebank=treebank`
        return self.treebank.T(
            treebank,
            self.label if treebank == self.treebank else treebank.get_label(self.label.class_id),  ##-OK both, raw val
            copy(self.leaf),
            operator = self._operator
        )

    def delete(self, _top = True):
        if self.parent:
            if _top:
                self.parent.children[self.parent.index_of(self)] = NonTerminal(
                    self.treebank, self.label,
                    operator = self.treebank.default_op if self.treebank.default_op else ops.UNIT_ID
                )
            else:
                self.parent.children[self.parent.index_of(self)] = None
            self.parent = None
        self.label.remove_node(self)


class Label:
    """Node label used for all subclasses of Tree. Each category of
    tree-node has exactly one Label created, and all nodes of that category
    have the same Label object set to their `label` property.

    Attributes:
        treebank (Treebank): Every tree belongs to a Treebank, which defines the
            set of Labels that `Tree`, its parents, and its children can take.
        is_default (Label): A treebank may have a _default_label, to be supplied
            if a request for a label is made without specifying which Label is
            needed. If `self` is the _default_label, then `is_default == True`
        nodes (list of Trees): List of all the tree nodes labelled with
            this particular `Label`.
        class_id (str): Unique name for the `Label`.

    Raises:
        AttributeError: If creation of a `Label` is attempted with a non-unique
            class_id.
    """

    def __init__(self, treebank, class_id = None): ###OK
        # create a list to store all Trees with this Label
        self.nodes = []
        self._class_id = class_id ###OK
        treebank.add_label(self)
        self.treebank = treebank
        self.is_default = False

    @property
    def class_id(self): ##-OK
        """(str) The ID of the Label, and of the corresponding category of
        trees. Identical to class_id, but some subclasses may have non-string
        labels, such that the raw class_id and a nice version for printing might
        be different. Use class_id for identifying the label.

        Parameters
        ----------
        kwargs
            Does nothing, but needed for compatability
        """
        return self._class_id ##-OK

    @property
    def classname(self): ##-OK
        """Identical to class_id, but some subclasses may have non-string
        labels, such that the raw class_id and a nice version for printing might
        be different. Use classname for printing.
        """
        return self._class_id ##-OK

    @class_id.setter ##-OK
    def class_id(self, class_id): ##-OK
        """Setter for class_id. Checks that the name is a `str`, as this is the
        only valid type for class_ids, and checks that the name is unique.

        Parameters
        ----------
        class_id
            (str) unique ID of the class

        Raises:
            TypeError: if `class_id` is not a `str`
        """
        if name is None:
            name = ""
        if type(class_id) is not str: # Type check
            raise TypeError("Names of Labels can only be strings")
        # OK, fine I guess.
        self._class_id = class_id ##-OK

    def add_node(self, node: Tree):
        """When a label is added to a node, the node is also added to the label:
        a Label is fundamentally a substitutability class. Each label object has
        a list `self.nodes` in which all nodes in the class are stored.

        To guarantee consistency, `add_node` checks that `node` is indeed a
        subclass of `Tree`, and that it does not belong to any other
        `Label`, and that it has a reference to the present `Label`.

        Parameters
        ----------
        node : Tree
            The node to be added.

        Raise
        -----
        TypeError
            If the node is not actually a Tree

        >>> tb = tbs.Treebank()
        >>> tb.default_label = tbs.tr.Label(tb, '*')
        >>> adv = tbs.tr.Label(tb, 'Adv')
        >>> print(adv)
        [Adv]
        >>> sad_tree = tbs.tree("('alone')", tb)
        >>> print(sad_tree)
        ('alone')
        >>> adv.add_node(sad_tree)
        >>> adv.print_terminal_subtrees()
        ([Adv]'alone')
        """
        # Check the type first. The `if __name__ == '__main__'` is because,
        # when the doctests run, `Tree` in the type-check below will otherwise
        # be `__main__.Tree`, but the actual trees are created by `Treebank`,
        # which imports `Tree` and its subclasses from `trees.py`, so the
        # resulting trees are, e.g. `trees.NonTerminal`, subclass of
        # `trees.Tree`, but *not* of `__main__.Tree`. Pain in the arse, but
        # watchugonnado?
        # if isinstance(node, Tree if __name__ != '__main__' else tbs.Tree):
        if isinstance(node, self.treebank.N) or isinstance(node, self.treebank.T):
            # If it already has `self` as its label, all that is needed is to
            # add it to self.nodes. This is the expected behaviour, as the
            # setter for label in Tree is what calls add_node, and is
            # itself called in Tree.__init__, which its inheritors
            # typically extend rather than override; but just in case...
            if node.label is not self:
                # If the node.label just isn't set, all that is needed to set
                # it: but just in case...
                if node.label is not None:
                    # If it already has a different Label, then make sure it
                    # is removed from the other Label's node list
                    node.label.remove_node(node)
                # So now we're sure we can set the label param on node
                # TODO: There is no `self.nodes.append(node)` on this branch,
                # because this is done in the label setter in Tree.
                # I tried having the `self.nodes.append(node)` below under
                # a separate conditional checking that `node` isn't already in
                # `self.nodes`, which does remove the dependency but makes
                # everything super slow. Is there a better way of removing this
                # dependency
                node.label = self
            # And now we're sure it's OK to add the node to the node list,
            # unless node's label.setter already did this
            else:
                self.nodes.append(node)
        else:
            # But if the type check fails...
            raise TypeError(
                f"When you add a node to a Label, it must be a {self.treebank.N}" +
                f" or {self.treebank.T}. You added {node}, which is a " +
                f"{type(node)}, and that makes me sad." # womp womp
            )

    def remove_node(self, node: Tree):
        """A bit of tear-down is needed if a Label is removed from a node: the
        Label is a substitutability class, and maintains a record of all the
        trees in the class, so the node must also be removed from the `Label`'s
        list.

        Parameters
        ----------
        node : Tree
            The node to be removed

        >>> tb = tbs.Treebank()
        >>> tb.default_label = tbs.tr.Label(tb, '*')
        >>> t_1 = tbs.tree("([S]([NP]([Det]'the')([N]'sentence'))([VP]([V]'parses')([NP]([Det]'the')([N]'parser'))))", tb)
        >>> t_2 = tbs.tree("([S]([N]'word')([VP]([V]'parses')([N]'parser')))", tb)
        >>> t_3 = tbs.tree("([S]([N]'sentence')([VP]([V]'parses')([V]'parser')))", tb)
        >>> t_1[0,0].label.remove_node(t_1[0,0])
        >>> print(t_1)
        ([S]([NP]('the')([N]'sentence'))([VP]([V]'parses')([NP]([Det]'the')([N]'parser'))))
        >>> tb.default_label.add_node(t_1[0,0])
        >>> print(t_1)
        ([S]([NP]('the')([N]'sentence'))([VP]([V]'parses')([NP]([Det]'the')([N]'parser'))))
        """
        # Using this loop to remove the node from nodes, because __eq__ is
        # overriden in Tree, and it is important that the node that is removed
        # is the exact node passed to the method, not just the first one to have
        # the same value.
        for i, t in enumerate(self.nodes):
            if t is node:
                del self.nodes[i]
                break
        node.label = None

    def __str__(self):
        """String representation of Label for use with __str__ methods in Tree
        classes. Returns "" for unlabelled nodes.

        Returns
        -------
            str: classname in square brackets, or empty string for
                _default_label

        >>> # No doctests here - the tests for __str__ in the `Tree` classes
        >>> # implicitly also test this
        """
        return "" if self.classname == "" or self.is_default else f"[{self.classname}]"  ##-OK - cn is correct here, see if I can just let TL inherit

    def __contains__(self, tree):
        return bool(sum([id(tree)==id(t) for t in self.nodes]))

    def __repr__(self):
        return(f"Label('{self.classname}')")  ##-OK

    def to_LaTeX(self, op_name = ""):
        """String representation of Label for use with to_LaTeX methods in Tree
        classes. Returns "" for unlabelled nodes.

        Returns
        -------
            str: classname preceded by `.`, or empty string for _default_label

        >>> # No doctests here - the tests for to_LaTeX in the `Tree` classes
        >>> # implicitly also test this
        """
        if self.is_default and not op_name:
            return ""
        elif not self.is_default and op_name:
            return f".{self.classname}:{op_name}"  ##-OK
        elif op_name:
            f".{op_name}"
        else:
            return  f".{self.classname}"  ##-OK

    def print_terminal_subtrees(self, is_LaTeX=False):
        """A helper function for testing and inspection, mostly. Prints out all
        trees with the current `Label`, either using `__str__` or `to_LaTeX`.

        Parameters
        ----------
        is_LaTeX : bool
            If true, print the trees in LaTeX `qtree` format: otherwise just use
            the standard str format

        >>> tb = tbs.Treebank()
        >>> tb.default_label = tbs.tr.Label(tb, '*')
        >>> x_tree = tbs.tree("(('list')(('of')(('all')(('the')(('fucks')('I')('give'))))))", tb)
        >>> print(x_tree)
        (('list')(('of')(('all')(('the')(('fucks')('I')('give'))))))
        >>> tb.default_label.print_terminal_subtrees()
        ('list')
        ('of')
        ('all')
        ('the')
        ('fucks')
        ('I')
        ('give')
        (('fucks')('I')('give'))
        (('the')(('fucks')('I')('give')))
        (('all')(('the')(('fucks')('I')('give'))))
        (('of')(('all')(('the')(('fucks')('I')('give')))))
        (('list')(('of')(('all')(('the')(('fucks')('I')('give'))))))
        >>> tb.default_label.print_terminal_subtrees(is_LaTeX = True)
        \Tree [ list ]
        \Tree [ of ]
        \Tree [ all ]
        \Tree [ the ]
        \Tree [ fucks ]
        \Tree [ I ]
        \Tree [ give ]
        \Tree [ [ fucks ] [ I ] [ give ] ]
        \Tree [ [ the ] [ [ fucks ] [ I ] [ give ] ] ]
        \Tree [ [ all ] [ [ the ] [ [ fucks ] [ I ] [ give ] ] ] ]
        \Tree [ [ of ] [ [ all ] [ [ the ] [ [ fucks ] [ I ] [ give ] ] ] ] ]
        \Tree [ [ list ] [ [ of ] [ [ all ] [ [ the ] [ [ fucks ] [ I ] [ give ] ] ] ] ] ]
        """
        if is_LaTeX: # Resolve the conditional first, so it is only done once
            for tree in self.nodes:
                # LaTeXise all the things!
                print(tree.to_LaTeX())
        else:
            for tree in self.nodes:
                # Or just print them!
                print(tree)

    @property
    def roots(self) -> List[Tree]:
        """(list[Tree]): all nodes in `self` that have no parent node

        >>> tb = tbs.Treebank()
        >>> tb.default_label = tbs.tr.Label(tb, 'X')
        >>> t1 = tbs.tree("([S]([NP]('the')([N]'sentence'))([VP]([V]'parses')([NP]([Det]'the')([N]'parser'))))", tb)
        >>> t2 = tbs.tree("([S]([N]'word')([VP]([V]'parses')([N]'parser')))", tb)
        >>> t3 = tbs.tree("([S]([N]'sentence')([VP]([V]'parses')([V]'parser')))", tb)
        >>> x_tree = tbs.tree("(('list')(('of')(('all')(('the')(('fucks')('I')('give'))))))", tb)
        >>> hippo = tbs.tree("([NP]([Det]'a')([AdjP]([Adj]'giant')([N]'hippopotamus')))", tb)
        >>> sad = tbs.tree("([Adv]'alone')", tb)
        >>> print(tb.class_ids['V'].roots)
        []
        >>> def print_roots(name):
        ...     for root in tb.class_ids[name].roots:
        ...         print(root)
        ...
        >>> print_roots('S')
        ([S]([NP]('the')([N]'sentence'))([VP]([V]'parses')([NP]([Det]'the')([N]'parser'))))
        ([S]([N]'word')([VP]([V]'parses')([N]'parser')))
        ([S]([N]'sentence')([VP]([V]'parses')([V]'parser')))
        >>> print_roots('X')
        (('list')(('of')(('all')(('the')(('fucks')('I')('give'))))))
        >>> print_roots('Adv')
        ([Adv]'alone')
        >>> print_roots('NP')
        ([NP]([Det]'a')([AdjP]([Adj]'giant')([N]'hippopotamus')))
        """
        return list(filter(lambda node: not bool(node.parent), self.nodes))

        def __eq__(self, other):
            return self.class_id == other.class_id  ##-OK


class TypeLabel(Label):
    """An extension to `Label` that uses Python types as label names instead of
    strings
    """

    @property
    def class_id(self):  ##-OK
        """(type) The unique ID of the Label, and of the corresponding category
        of trees.
        """
        return self._class_id ##-OK

    @property
    def classname(self):  ##-OK
        """(type or str) The name of the Label, and of the corresponding category of
        trees.
        """
        return self._class_id.__name__ ##-OK

    @class_id.setter ##-OK
    def class_id(self, class_id): ##-OK
        """Setter for classname. Checks that the name is a `str` that evaluates
        to a `type`, or a `type`: if neither of those obtain, the type of the
        parameter is used instead. Also checks that the name is unique.

        Parameters
        ----------
        name : Any
            The type that names the class the label represents.

        Raises
        ------
        TypeError:
            If `name` is not a `type`
        """
        if isinstance(name, str):
            try:
                name = eval(name)
            except Exception:
                raise AttributeError("Invalid label")
        if isinstance(name, type): # Type check
            name = type(name)
        # OK, fine I guess.
        self._class_id = name ##-OK

    def __str__(self):
        """String representation of TypeLabel for use with __str__ methods in
        Tree classes.

        Returns
        -------
            str: name of the type in square brackets

        >>> # No doctests here - the tests for __str__ in the `Tree` classes
        >>> # implicitly also test this
        """
        return f"[{self.classname}]" ##-OK

    def __repr__(self):
        return(f"TypeLabel({self.classname})") ##-OK

    def to_LaTeX(self, op_name = ""):
        """String representation of Label for use with to_LaTeX methods in Tree
        classes. Returns "" for unlabelled nodes.

        Returns
        -------
            str: classname preceded by `.`, or empty string for _default_label

        >>> # No doctests here - the tests for to_LaTeX in the `Tree` classes
        >>> # implicitly also test this
        """
        return ".{$" + op_name + (r'\rightarrow ' if op_name else '') + self.classname + "$}"  ##-OK

# TESTME
@dataclass
class SubstitutionSite:
    """Dataclass containing all the information about a nonterminal leaf node
    needed to do a node substitution at that location: its label (so we can
    check the substitution is valid), its parent, and its index in its parent
    (so the substitution can be performed).

    Attributes:
        parent (NonTerminal): parent node of the nonterminal leaf node, needed
            to perform substitution.
        index (int): index of nonterminal leaf node, needed to perform
            substitution.
        label (Label): Label of nonterminal leaf node, needed to ensure
            substitution if valid.
    """
    parent: NonTerminal
    index: int
    label: Label

    def perform_substitution(self, subtree: Tree):
        """Takes a subtree and, if it has the same node label, swaps it for the
        nonterminal leafnode at the substitution site

        Parameters
        ----------
            subtree (Tree): The subtree to be swapped into place.
        Raises:
            ValueError: if the Label at the substitution site does not match the
                root label of the subtree to be substituted
        """
        # If substitution is legit...
        if subtree.label == self.label:
            # ...then, do it. ...
            self.parent[self.index] = subtree
            subtree.parent = self.parent
        else:
            # ... otherwise, noep.
            raise ValueError(
                f"Subtree {subtree} with root label {subtree.label} " +
                f"cannot be substituted at index " +
                f"{self.index} of {self.parent} due to a mismatch with " +
                f"substitution site label {self.label}."
            )

    @property
    def site(self) -> NonTerminal:
        """(NonTerminal) The actual nonterminal leafnode."""
        return self.parent[self.index]


def main():
    import doctest
    doctest.testmod()
    # t = tbs.tree("([S]([X]x)([Y]y))")
    # print(t())


if __name__ == '__main__':
    main()
