import inspect

def is_tree(obj):
    for cls in inspect.getmro(type(obj)):
        try:
            if cls.__module__ == 'trees' and cls.__name__ == 'Tree':
                return True
        except:
            print('!!!')
            return False
    return False

class TreeIter:
    """Iterator that iterates over a Tree, yielding the immediate child-nodes
    only. This was needed because I wanted to include informative IndexError
    messages in the custom __getitem__ methods in Tree, but the default __iter__
    used catching IndexErrors to tell when the iteraton was done, which meant
    that the code used to generate these informative error messages was running
    a *lot*, mostly to no purpose, and sometimes causing Python to throw a
    recursion depth exception.
    """
    def __init__(self, tree):
        self._pos = 0
        self._tree = tree

    def __iter__(self):
        return self

    def __next__(self):
        if self._pos < len(self._tree):
            self._pos += 1
            return self._tree[self._pos - 1]
        else:
            raise StopIteration
        

class DepthFirstBottomUp(TreeIter):
    """Unlike the standard TreeIter, this iterates over all nodes of the Tree,
    not just the immediate child nodes. It does so starting at the bottom
    leftmost node, then moving up if the node above has no other chidren. If it
    does, the lower leftmost unvisited child is visited, and a parent node is
    only visited after all the nodes below it have been visited. Thus, for a
    tree:

         A
        /  \\
       /    \\
      B      C
     / \    / \\
    D   E  F   G
    |   |  |   |
    h   i  j   k

    ...it would visit h, D, i, E, B, j, F, k, G, C, A.

    >>> from trees import *
    >>> from treebanks import *
    >>> import operators as ops
    >>> op = [ops.SUM, ops.PROD, ops.SQ, ops.EQ, ops.NOT, ops.OR, ops.AND]
    >>> tlt = TypeLabelledTreebank(op)
    >>> gpt0 = tlt.tree("([bool]<AND>([bool]<EQ>([float]<SQ>([int]6))([float]<PROD>([int]9)([float]<SUM>([float]2.5)([float]1.5))))([bool]<OR>([bool]<NOT>([bool]<EQ>([int]0)([int]1)))([bool]False)))")
    >>> it0 = DepthFirstBottomUp(gpt0)
    >>> has_next = True
    >>> while has_next:
    ...     try:
    ...         nxt = next(it0)
    ...         print(nxt)
    ...     except StopIteration:
    ...         has_next = False
    6
    ([int]6)
    ([float]<SQ>([int]6))
    9
    ([int]9)
    2.5
    ([float]2.5)
    1.5
    ([float]1.5)
    ([float]<SUM>([float]2.5)([float]1.5))
    ([float]<PROD>([int]9)([float]<SUM>([float]2.5)([float]1.5)))
    ([bool]<EQ>([float]<SQ>([int]6))([float]<PROD>([int]9)([float]<SUM>([float]2.5)([float]1.5))))
    0
    ([int]0)
    1
    ([int]1)
    ([bool]<EQ>([int]0)([int]1))
    ([bool]<NOT>([bool]<EQ>([int]0)([int]1)))
    False
    ([bool]False)
    ([bool]<OR>([bool]<NOT>([bool]<EQ>([int]0)([int]1)))([bool]False))
    ([bool]<AND>([bool]<EQ>([float]<SQ>([int]6))([float]<PROD>([int]9)([float]<SUM>([float]2.5)([float]1.5))))([bool]<OR>([bool]<NOT>([bool]<EQ>([int]0)([int]1)))([bool]False)))
    >>> gpt1 = tlt.tree("([bool]<AND>([bool]<EQ>([float]<SQ>([int]6))([float]))([bool]<OR>([bool]<NOT>([bool]<EQ>([int]0)([int]1)))([bool]False)))")
    >>> it1 = DepthFirstBottomUp(gpt1)
    >>> has_next = True
    >>> while has_next:
    ...     try:
    ...         nxt = next(it1)
    ...         print(nxt)
    ...     except StopIteration:
    ...         has_next = False
    6
    ([int]6)
    ([float]<SQ>([int]6))
    ([float])
    ([bool]<EQ>([float]<SQ>([int]6))([float]))
    0
    ([int]0)
    1
    ([int]1)
    ([bool]<EQ>([int]0)([int]1))
    ([bool]<NOT>([bool]<EQ>([int]0)([int]1)))
    False
    ([bool]False)
    ([bool]<OR>([bool]<NOT>([bool]<EQ>([int]0)([int]1)))([bool]False))
    ([bool]<AND>([bool]<EQ>([float]<SQ>([int]6))([float]))([bool]<OR>([bool]<NOT>([bool]<EQ>([int]0)([int]1)))([bool]False)))
    >>> gpt2 = tlt.tree("([bool]<AND>([bool]<EQ>([float]<SQ>([int]666))([float]<PROD>([int]9)([float]<SUM>([float]2.5)([float]1.5))))([bool]<OR>([bool]<NOT>([bool]<EQ>([int]0)([int]1)))([bool]False)))")
    >>> it2 = DepthFirstBottomUp(gpt2, trees_only=True)
    >>> has_next = True
    >>> while has_next:
    ...     try:
    ...         nxt = next(it2)
    ...         print(nxt)
    ...     except StopIteration:
    ...         has_next = False
    ([int]666)
    ([float]<SQ>([int]666))
    ([int]9)
    ([float]2.5)
    ([float]1.5)
    ([float]<SUM>([float]2.5)([float]1.5))
    ([float]<PROD>([int]9)([float]<SUM>([float]2.5)([float]1.5)))
    ([bool]<EQ>([float]<SQ>([int]666))([float]<PROD>([int]9)([float]<SUM>([float]2.5)([float]1.5))))
    ([int]0)
    ([int]1)
    ([bool]<EQ>([int]0)([int]1))
    ([bool]<NOT>([bool]<EQ>([int]0)([int]1)))
    ([bool]False)
    ([bool]<OR>([bool]<NOT>([bool]<EQ>([int]0)([int]1)))([bool]False))
    ([bool]<AND>([bool]<EQ>([float]<SQ>([int]666))([float]<PROD>([int]9)([float]<SUM>([float]2.5)([float]1.5))))([bool]<OR>([bool]<NOT>([bool]<EQ>([int]0)([int]1)))([bool]False)))
    """
    def __init__(self, tree, trees_only=False):
        super().__init__(tree)
        self.trees_only = trees_only
        if is_tree(self._tree) and len(self._tree):
            # DFBU works recursively: a DFBU iterator for a tree works by
            # creating DFBUs for each of its children
            self.it = DepthFirstBottomUp(self._tree[self._pos], trees_only=self.trees_only)
        else:
            # Unless self._tree is a leafnode
            self.it = None

    def __next__(self):
        # If you have a DFBU for a node one level down, call next on it...
        if self.it:
            # ... and return the result, if it hasn't Stopped Iteration.
            try:
                return next(self.it)
            except StopIteration:
                # ... if it has, then move on to the next child node...
                if self._pos + 1 < len(self._tree):
                    # (incrementing the position attribute to indicate this)
                    self._pos += 1
                    # ... make a DFBU for that node ...
                    self.it = DepthFirstBottomUp(self._tree[self._pos], trees_only=self.trees_only)
                    # ... and call next on *that*
                    return next(self)
                # ... unless self has exhausted DFBU iterators on *all* its
                # children ...
                elif self._pos + 1 == len(self._tree):
                    # in which case, increment position one more time, so you
                    # know that *next* call of next will raise StopIteration
                    self._pos += 1
                    # and return self._tree itself. This is how a DFBU iterator
                    # work, basically: run other DFBUs on all the child nodes
                    # of self._tree, then return self._tree...
                    return self._tree
                else:
                    # ... and then you're done
                    raise StopIteration
        elif not self._pos and not self.trees_only:
            # Alternatively, self._tree may be a leaf, in which case DFBU just
            # returns self._tree *once*, then Stops Iteration. self._pos is used
            # to tell if next has been called - if 0, it hasn't, so increment it
            # and return the leaf
            self._pos += 1
            return self._tree
        else:
            # Otherwise...
            raise StopIteration




def main():
    import doctest
    doctest.testmod()

if __name__ == '__main__':
    main()
