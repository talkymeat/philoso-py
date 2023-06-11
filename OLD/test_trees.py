from trees import *

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
