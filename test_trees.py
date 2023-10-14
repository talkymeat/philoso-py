from trees import *
from treebanks import *
from gp_trees import *
from gp import *
from tree_factories import *

def test_trees(tb = None):
    """
    Creates a collection of sample trees for testing.

    >>> t = test_trees()
    >>> for tx in t:
    ...     print(tx)
    ...
    ([N]'poo')
    ([S]([N]'sentence')([VP]([V]'parses')([N]'parser')))
    ([S]([N]'sentence')([VP]([V]'parses')([N]'parser')))
    ([S]([N]'sentence')([VP]([V]'parses')([N]'parser')))
    ([S]([NP]([Det]'the')([N]'sentence'))([VP]([V]'parses')([NP]([Det]'the')([N]'parser'))))
    ([S]([N]'word')([VP]([V]'parses')([N]'parser')))
    ([S]([N]'sentence')([VP]([V]'parses')([V]'parser')))
    ([S]([X]'x')([Y]'y'))
    """
    tb = tb if tb else Treebank()
    t = [None]*8
    t[0] = Terminal(tb, tb.get_label("N"), "poo")
    t[1] = tree(
        "([S]([N]sentence)([VP]([V]parses)([N]parser)))",
        tb
    )
    t[2] = NonTerminal(
        tb, "S", Terminal(
            tb, "N", "sentence"
        ), NonTerminal(
            tb, "VP", Terminal(
                tb, "V", "parses"
            ), Terminal(
                tb, "N", "parser"
            )
        )
    )
    t[3] = NonTerminal(
        tb, "S", Terminal(
            tb, "N", "sentence"
        ), NonTerminal(
            tb, "VP", Terminal(
                tb, "V", "parses"
            ), Terminal(
                tb, "N", "parser"
            )
        )
    )
    t[4] = tree(
        "([S]([NP]([Det]the)([N]sentence))([VP]([V]parses)([NP]([Det]the)([N]parser))))",
        treebank = tb
    )
    t[5] = tree("([S]([N]word)([VP]([V]parses)([N]parser)))", treebank = tb)
    t[6] = tree("([S]([N]sentence)([VP]([V]parses)([V]parser)))", treebank = tb)
    t[7] = tree("([S]([X]x)([Y]y))", treebank = tb)
    return t

def test_fragments(tb = None):
    tb = tb if tb else Treebank()
    return [
        tree("([S]([NP]([Det]the)([N]cat))([VP]))", treebank = tb),
        tree("([S]([NP]([Det]the)([N]cat))([VP]([V]ate)([NP])))", treebank = tb),
        tree("([NP])", treebank = tb)
    ]

def test_gp_trees() -> list[GPNonTerminal]:
    """
    >>> gpts =test_gp_trees()
    >>> df = pd.DataFrame({'x': [1.0, 2.0, 3.0, 4.0, 5.0]})
    >>> for gpt in gpts:
    ...     out = gpt(**df)
    ...     try:
    ...         print(list(out))
    ...     except TypeError:
    ...         print(out)
    [9.0, 18.0, 31.0, 48.0, 69.0]
    [14.0, 28.0, 48.0, 74.0, 106.0]
    [11.0, 37.0, 79.0, 137.0, 211.0]
    [18.0, 39.0, 68.0, 105.0, 150.0]
    7.0
    7.0
    """
    gp = GPTreebank(operators=[ops.SUM, ops.PROD, ops.SQ, ops.POW, ops.CUBE])
    rpf = RandomPolynomialFactory(gp, 2, -10.0, 10.0)
    return [
        rpf('x', treebank=gp, coefficients={(('x',), (2,)): 2.0, (('x',), (1,)): 3.0, ((), ()): 4.0}),
        rpf('x', treebank=gp, coefficients={(('x',), (2,)): 3.0, (('x',), (1,)): 5.0, ((), ()): 6.0}),
        rpf('x', treebank=gp, coefficients={(('x',), (2,)): 8.0, (('x',), (1,)): 2.0, ((), ()): 1.0}),
        rpf('x', treebank=gp, coefficients={(('x',), (2,)): 4.0, (('x',), (1,)): 9.0, ((), ()): 5.0}),
        gp.tree("([float]<SUM>([float]5.0)([float]2.0))"),
        gp.tree("([float]7.0)")
    ]

def main():
    import doctest
    doctest.testmod()

if __name__ == '__main__':
    main()