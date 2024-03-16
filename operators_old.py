from typing import List, Dict, Type, Any, Callable, Container
import pandas as pd
import numpy as np
from functools import reduce
from type_ify import TypeNativiser, _DoesNothing
# from icecream import ic

import re


def _id(*args):
    return args

def _unit_id(*args):
    return args[0]

def _concat(*args):
    return " ".join(args)

def _sum(*args):
    return reduce(lambda a, b: a + b, (0,) + args)

def _prod(*args):
    return reduce(lambda a, b: a * b, (1,) + args)

def _pow(*args):
    return args[0] ** args[1]

def _eq(*args):
    return args[0] == args[1]

def _neq(*args):
    return args[0] != args[1]

def _gt(*args):
    return args[0] > args[1]

def _egt(*args):
    return args[0] >= args[1]

def _lt(*args):
    return args[0] < args[1]

def _elt(*args):
    return args[0] <= args[1]

def _not(*args):
    return not args[0]

def _tern(*args):
    return args[1] if args[0] else args[2]

def _sq(*args):
    return args[0]**2

def _cube(*args):
    return args[0]**3


class Operator:
    """Operators which perform computations on the values of the child nodes of
    trees, such that the value of a nonterminal is
    `parent_operator(*[children_values])` and the value of a terminal is its
    content. An Operator is a Callable - essentially just a function with
    enforced type checking.

    In a DOP implementation for NLP, this means that the terminal
    contents will be (typically) words, and only one operator will
    be used, the concatenation operator CONCAT, and the value of
    any complete parse tree will be the sentence it parses, and the
    value of any node will be the constituent underneath it.

    In genetic programming, multiple operators will be used, for
    arithmetic and boolean operations - possibly more besides.

    Class Attributes:
        type_dict: a dictionary mapping from types to single-character strings.
            A default dictionary is provided, but this can be added to or
            overwritten. This is used define the arguments that an Operator can
            take when called: each operator, when initialised, takes a regex
            attribute, in which the legality of any argument-sequence can be
            judged by mapping the types of the arguments to the corresponding
            characters and seeing if the resulting type-string matches the
            regex: if it matches, it's legit; if not, AttributeError.

    Object Attributes:
        func (function): the function that `__call__` is to be an alias for:
            the function that combines the values of the child nodes to produce
            the value of their parent node. Must be able to take inputs that
            match the type-specification given in *args, and given a suitable
            set of inputs, must return the type given by return_type.
        name (str): the name of the operator - used, for example, in LaTeX
            representations of GP trees
        return_type (type): The return type of `func`. The default value is
            `Any`, which is compatible with any output. If at any time `func`
            returns an output incompatible with `return_type`, a TypeError is
            raised.
        arg_regex (str): a regular expression which specifies which sequences of
            types are legal as arguments when calling the Operator. Iff the
            argument-sequence can be converted to a string with type_dict, and
            that string matches `arg_regex`, the sequence is legal. If no value
            is passed for `arg_regex`, the default is a regex that accepts any
            non-empty sequence of types given in type_dict. If the operator is
            called with an illegal argument sequence, an AttributeError will be
            raised.
        force_type (bool): By default, this is `False`. If `False`, any
            difference between the type returned by `func` and return_type will
            raise a TypeError. However, if `True`, if the type returned does not
            match, it will in *some* circumstances be converted to
            `return_type`, depending on the value of `force_type_lossy`, below.
            If `force_type` is `True`, but the output cannot be converted, a
            TypeError will still be raised.
        force_type_lossy (bool): Iff `force_type` is False, this is redundant.
            By default, this `force_type_lossy` is `False`, in which case, if
            `force_type` is `True` the output of `func` will be converted to
            `return_type` losslessly: thus `int` `1` can be converted to `float`
            `1.0`, and `float` `1.0` can be converted to `int` `1`, but `float`
            `1.5` cannot be converted to `int` `2`. If `force_type_lossy` is
            `True`, 1.5 can be converted to 2.

    """

    tn = TypeNativiser()

    type_dict = {
        str:          's',
        int:          'i',
        float:        'f',
        complex:      'c',
        bool:         'b',
        _DoesNothing: 'x'
    }

    rev_type_dict = {v: k for k, v in type_dict.items()}

    
    def drt_simple(self, arg_match: re.Match) -> type:
        if arg_match.groups():
            typeset = {Operator.rev_type_dict[char] for char in ''.join(arg_match.groups())}
            return list(typeset)[0] if len(typeset)==1 else Any
        return self.return_type

    def __init__(
            self, 
            func: Callable, 
            name: str, 
            return_type: type = Any, 
            arg_regex: str = "", 
            force_type: bool = False, 
            force_type_lossy: bool = False, 
            apply: bool = False, 
            return_dtype = None,
            drt_func: Callable[[re.Match], type] = None
        ):
        self.func = func
        self.name = name
        self.return_type = return_type
        self.force_type = force_type
        self.force_type_lossy = force_type_lossy
        self.apply = apply
        self.return_dtype = return_dtype
        self.dynamic_return_type = drt_func if drt_func else self.drt_simple


        # Lambda to create a string of all the type-characters in the
        # type-dictionary, plus 'x' for 'other'. If a value is not provided for
        # `arg_regex`, this is used to the default `arg_regex`, which accepts
        # any non-empty set of arguments. A lambda is created, rather than a
        # string, so as to reduce the computational overhead in the case where
        # the default is not needed.
        type_chars = lambda: ''.join(Operator.type_dict.values()) + 'x'
        self.arg_regex = arg_regex if arg_regex else r'[' + type_chars() + r']+'

    def __str__(self):
        return f"<{self.name}>" if self.name else ""

    # def _arg_lengths(self, *args):
    #     leng = 1
    #     for arg in args:
    #         if not len(arg) in [1, leng]:
    #             if leng == 1:
    #                 leng = len(arg)
    #             else:
    #                 raise ValueError("Argument lengths do not match")
    #     return [pd.Series([arg[0]]*leng) if len(arg) == 1 else arg for arg in args]

    def _preprocess(self, *args):
        leng = 1
        indicators = [-1]*len(args)
        no_series = True
        for i, arg in enumerate(args):
            if isinstance(arg, np.ndarray):
                args[i] = pd.Series(arg)
            if isinstance(arg, pd.Series):
                no_series = False
                indicators[i] = len(arg)
                if len(arg) in [1, leng]:
                    pass
                elif leng == 1:
                    leng = len(arg)
                else:
                    raise ValueError("Argument lengths do not match")
        if no_series:
            return args, False
        if self.apply:
            return [pd.Series([arg[0]]*leng) if (ind == 1) else pd.Series([arg]*leng) if (ind==-1) else arg for ind, arg in zip(indicators, args)], True
        return [arg[0] if ind == 1 else arg for ind, arg in zip(indicators, args)], False

    def __call__(self, *args):
        """A magic method which makes `Operator`s Callable. Checks that the
        arguments and output are legal: returns the output if everything is OK.

        Raises
        ------
            AttributeError: If arguments are not legal
            TypeError: If output is not legal

        >>> def a_plus_b(*args):
        ...     return args[0] + args[1]
        ...
        >>> op_ID = Operator(_id, "ID")
        >>> op_ID()
        Traceback (most recent call last):
            ....
        AttributeError: This operator cannot be called without arguments
        >>> op_ID(2, 3)
        (2, 3)
        >>> op_ID("regard", "that", "cat")
        ('regard', 'that', 'cat')
        >>> op_CONCAT = Operator(_concat, "CONCAT", str, r"ss+")
        >>> op_CONCAT(2, 3)
        Traceback (most recent call last):
            ....
        AttributeError: Incorrect arguments for <CONCAT>: (2, 3)
        >>> op_CONCAT("regard", "that", "cat")
        'regard that cat'
        >>> op_A_PLUS_B_WRONG = Operator(a_plus_b, "A_PLUS_B_WRONG", int, r'[if][if]')
        >>> op_A_PLUS_B_WRONG("2", "3")
        Traceback (most recent call last):
            ....
        AttributeError: Incorrect arguments for <A_PLUS_B_WRONG>: ('2', '3')
        >>> op_A_PLUS_B_WRONG(2, 3)
        5
        >>> op_A_PLUS_B_WRONG(2.0, 3.0)
        Traceback (most recent call last):
            ....
        TypeError: Output float when int was expected
        >>> op_A_PLUS_B = Operator(lambda a, b: float(a+b), "A_PLUS_B", float, r'[if][if]')
        >>> op_A_PLUS_B("2", "3")
        Traceback (most recent call last):
            ....
        AttributeError: Incorrect arguments for <A_PLUS_B>: ('2', '3')
        >>> op_A_PLUS_B(2, 3)
        5.0
        >>> op_A_PLUS_B(2.0, 3.0)
        5.0
        >>> op_SUM = Operator(_sum, "SUM", float, r"[if]*")
        >>> op_SUM_FL = Operator(_sum, "SUM_FL", float, r"[if]*", True)
        >>> op_SUM_INT = Operator(_sum, "SUM_INT", int, r"[if]*", True)
        >>> op_SUM_INT_LOSSY = Operator(_sum, "SUM_INT_LOSSY", int, r"[if]*", True, True)
        >>> op_SUM(2, 3)
        Traceback (most recent call last):
            ....
        TypeError: Output int when float was expected
        >>> op_SUM_FL(2, 3)
        5.0
        >>> op_SUM_INT(2.0, 3.0)
        5
        >>> op_SUM_INT(2.0, 2.5)
        Traceback (most recent call last):
            ....
        TypeError: Output float when int was expected
        >>> op_SUM_INT_LOSSY(2.0, 2.5)
        4
        >>> op_SUM(2.0, 3.0, 5.0)
        10.0
        >>> op_A_PLUS_B(2, 3, 5)
        Traceback (most recent call last):
            ....
        AttributeError: Incorrect arguments for <A_PLUS_B>: (2, 3, 5)
        """
        def custom_type_err(output):
            # Inner function to create an error message, in the case where the
            # operator outputs an illegal type. `Operator.__call__` is
            # structured largely as a decision-tree determining whether to call
            # the underlying function, call the function and cast to
            # `self.return_type`, or throw an error: since this error message is
            # needed on three branches of the tree, using an inner function
            # avoids code duplication
            return f"Output {type(output).__name__} when {self.return_type.__name__} was expected"
        # First, check that the arguments are legal ...
        if self._arg_seq_legal(*args):
            # ... if so, call the function ...
            args, applicable = self._preprocess(*args)
            if applicable:
                output = pd.concat(args, axis=1).apply(
                        lambda row: self.func(*tuple(row)), axis = 1
                    ) 
            else: 
                output = self.func(*args)
            # ... and return the result if it's also legal.
            if self.return_type is Any or issubclass(Operator.tn.type_ify(output), self.return_type):
                # print(output, 'xxds')
                return output
            elif isinstance(output, self.return_type):
                return output
            # ... there's a possible complication if the return_type is one that
            # doesn't have a built-in numpy dtype - for instance, `pd.Series` of
            # strings by default get the dtype `object`; in this case, the
            # conditional checks that all the elements in output are the correct
            # native type, casts `self.return_dtype` on it
            elif isinstance(output, pd.Series) and reduce(lambda p, q: p and q, output.apply(lambda x: isinstance(x, self.return_type))):
                return output.astype(self.return_dtype if self.return_dtype else self.return_type)
                # ... If it just isn't the right type...
            else:
                # ... then is it allowable to try casting it to the right type?
                # `self.force_type` is an attrib of the Operator which
                # automatically rules this out if false. Is it true?
                if self.force_type:
                    # If so, it may still be illegal to do the conversion,
                    # because of broader type conversion rules - like a `str`
                    # can only be cast to `float` if it has the form of a
                    # `float` or `int` literal. `try` this and catch the
                    # `ValueError` if it fails.
                    try:
                        # This needs to be split between the cases where the
                        # output is a native type, and where it's a pandas
                        # Series
                        if type(output) == pd.Series:
                            # if a return dtype was specified when the Operator
                            # was initialised ...
                            if self.return_dtype:
                                # we first use the return_type  attribute to
                                # convert the Series elements, then return_dtype
                                # to convert the Series itself. For instance,
                                # converting a Series of non-strings to a string
                                # series first needs to be cast to str, but then
                                # the dtype will be object, so you also need to
                                # cast to "string"
                                converted = output.astype(self.return_type).astype(self.return_dtype)
                            else:
                                # ...on the other hand, casting output to float,
                                # for example, changes both the type of the
                                # elements to float, and the dtype to float64
                                converted = output.astype(self.return_type)
                            # If the conversion is successful, there still may be
                            # one thing to check. `self.force_type_lossy` is an
                            # attribute which, if true, permits lossy conversion.
                            # If lossy conversion is permitted, or the conversion is
                            # not lossy ...
                            if self.force_type_lossy or reduce(lambda p, q: p and q, output == converted):
                                #  ... then return the converted value.
                                return converted
                        else:
                            # If the output is a native type, then the
                            # conversion is simpler
                            converted = self.return_type(output)
                            # If the conversion is not illegally lossy ...
                            if self.force_type_lossy or output == converted:
                                # ... then return the converted value.
                                return converted
                        # If the conversion is lossy, and this isn't allowed,
                        # raise a TypeError ...
                        raise TypeError(custom_type_err(output))
                    # ... same if the output is not convertible ...
                    except ValueError:
                        raise TypeError(custom_type_err(output))
                # ... or if conversion of outputs is illegal.
                else:
                    raise TypeError(custom_type_err(output))
        # Raise AttributeErrors in the case of illegal arguments
        else:
            if len(args) == 0:
                raise AttributeError(
                    "This operator cannot be called without arguments"
                )
            else:
                raise AttributeError(
                    f"Incorrect arguments for {self}: {args}"
                )

    @classmethod
    def _arg_type_tuple(cls, *args):
        return tuple(cls.tn.type_ify(arg) for arg in args)

    @classmethod
    def _type_list_str(cls, *types):
        return ''.join([Operator.type_dict.get(t, 'x') for t in types])

    def _type_str_legal(self, type_str):
        return re.fullmatch(self.arg_regex, type_str)

    def _type_seq_legal(self, *types):
        return self._type_str_legal(Operator._type_list_str(*types))

    def _arg_str(self, *args):
        """Converts *args to a string of characters, where each character is
        the value in `Operator.type_dict` corresponding to the type of the
        argument, unless the type is not in `Operator.type_dict`, in which case
        'x' is used.

        >>> op = Operator(lambda x: x, "ID")
        >>> print(op._arg_str("a", 1, 1.0, 1j+0, True))
        sifcb
        >>> print(op._arg_str(["a", 1, 1.0, 1j+0, True]))
        x
        >>> print(repr(op._arg_str()))
        ''
        """
        return Operator._type_list_str(*Operator._arg_type_tuple(*args))
        #return ''.join([self.type_dict.get(type(arg), 'x') for arg in args])

    # Could these checks be memoised?
    def _arg_seq_legal(self, *args):
        """Checks that a sequence of arguments is legal, based on Operator's
        `arg_regex`: Iff the string representation of the sequence of types in
        *args is a regex match to `self.arg_regex`, the sequence is legal. If
        `self.arg_regex` is empty (`""` or `None`), all non-empty

        >>> op0 = Operator(_id, 'ID0')
        >>> op1 = Operator(_id, 'ID1', arg_regex = r'sifcb')
        >>> op2 = Operator(_id, 'ID2', arg_regex = r'f+')
        >>> op3 = Operator(_id, 'ID3', arg_regex = r'[if][if]')
        >>> op4 = Operator(_id, 'ID4', arg_regex = r'sf*')
        >>> ops = (op0, op1, op2, op3, op4)
        >>> outlst = []
        >>> for op in ops:
        ...     outtxt = ""
        ...     outtxt += 'T' if op._arg_seq_legal() else 'F'
        ...     outtxt += 'T' if op._arg_seq_legal('blep', 1, 1.0, 1j+1, True) else 'F'
        ...     outtxt += 'T' if op._arg_seq_legal(1.0) else 'F'
        ...     outtxt += 'T' if op._arg_seq_legal(2.0, 1.0) else 'F'
        ...     outtxt += 'T' if op._arg_seq_legal(2.0, 1.0, 7.0, 0.0) else 'F'
        ...     outtxt += 'T' if op._arg_seq_legal(1) else 'F'
        ...     outtxt += 'T' if op._arg_seq_legal(1, 2) else 'F'
        ...     outtxt += 'T' if op._arg_seq_legal(1.0, 2) else 'F'
        ...     outtxt += 'T' if op._arg_seq_legal(2, 3.0) else 'F'
        ...     outtxt += 'T' if op._arg_seq_legal(1, 2, 3) else 'F'
        ...     outtxt += 'T' if op._arg_seq_legal('z') else 'F'
        ...     outtxt += 'T' if op._arg_seq_legal('z', 'x') else 'F'
        ...     outtxt += 'T' if op._arg_seq_legal('z', 81) else 'F'
        ...     outtxt += 'T' if op._arg_seq_legal('zx', 8.0, 1.0) else 'F'
        ...     outtxt += 'T' if op._arg_seq_legal('zx', 81.0, 0.1, 3.2) else 'F'
        ...     outtxt += 'T' if op._arg_seq_legal(['zx', 81.0, 0.1, 3.2]) else 'F'
        ...     outlst.append(outtxt)
        ...
        >>> print(outlst[0])
        FTTTTTTTTTTTTTTT
        >>> print(outlst[1])
        FTFFFFFFFFFFFFFF
        >>> print(outlst[2])
        FFTTTFFFFFFFFFFF
        >>> print(outlst[3])
        FFFTFFTTTFFFFFFF
        >>> print(outlst[4])
        FFFFFFFFFFTFFTTF
        """
        return self._type_seq_legal(*Operator._arg_type_tuple(*args))
        #return re.fullmatch(self.arg_regex, self._arg_str(*args))


ID = Operator(_id, "ID", tuple, return_dtype='O', apply=True)
"""
Operator which returns a tuple of the node's children. The basic-ass default
operator. Mistakes ability to remember funny lines from telly for being funny.
Listens to local radio. Has one significant ambition in life, which is to win
the lottery.

Arguments:
    Any arguments are allowed, as long as there are more than zero arguments

Returns:
    tuple
"""

UNIT_ID = Operator(_unit_id, "UNIT_ID", arg_regex = r"(.)")
"""
Only takes a single argument, which it returns unaltered. The default for
Terminals. Even blander and more lacking in personality than ID. Occassionally
tries to be funny, by repeating jokes they heard previously, but invariably,
no matter how funny the original joke, always entirely fails to elicit even a
smile. Works in local radio. Has no significant ambition in life.

Arguments:
    Any argument is allowed, as long as there is only one.

Returns:
    The argument.
"""

CONCAT = Operator(_concat, "CONCAT", str, r"s+", apply=True, return_dtype="string")
"""
Concatenates the string representations of child node with a space
as a separator. Useful for Data Oriented Parsing, as it returns the
original sentence that the tree parses. `ID`'s hipster cousin who moved
to the Netherlands. Sighs loudly at tipexed-on bike lanes when back in
the UK. Doesn't go to coffeeshops cause they're for tourists. Would love
a good excuse to justify the cost of a bakfiets.

Arguments:
    Any non-zero number of strings, e.g.: "pet", "the", "cat"

Returns:
    (str) The concatenated string, e.g.: "pet the cat"
"""

SUM = Operator(_sum, "SUM", float, r"[if]*", force_type=True)
"""
Adds floats/ints, and converts the sum to float. Returns 0.0 if called with no
arguments. As a teenager, dreamed of being a Fields Medalist, but due to
over-focusing on grades rather than learning, as an undergraduate avoided maths
courses that looked too hard, ended up an accountant instead.

Arguments:
    Any number of floats or ints. Can be called with no arguments.

Returns:
    (float) The sum.
"""

PROD = Operator(_prod, "PROD", float, r"[if]*", force_type=True)
"""
Multiplies floats/ints, and converts the sum to float. Returns 1.0 if called
with no arguments. Was a classmate of SUM's in undergrad, and was a bit more
intellectually ambitious, but got kinda obsessed with the Lotka-Volterra
equations in 2nd year, and ended up doing honours in Population Biology.
Now works for the New South Wales government, controlling rabbit populations.

Arguments:
    Any number of floats or ints. Can be called with no arguments.

Returns:
    (float) The product.
"""

SQ = Operator(_sq, "SQ", float, r"[if]", force_type=True)
"""
Takes a single float or int, squares it, and returns the result as a float.
Another former classmate of SUM and PROD, but stopped talking to PROD after
PROD tried selling them illicit ritalin. Now works for the British Board of Film
Censors.

Arguments:
    A single float or int.

Returns:
    (float) The square.
"""

CUBE = Operator(_cube, "CUBE", float, r"[if]", force_type=True)
"""
Takes a single float or int, cubes it, and returns the result as a float.
Came through the same pre-honours maths program as SUM, PROD, and SQ, but
through a series of career changes ended up a designer for IKEA. Talks a lot
about the use of the vertical dimension in interior design, regardless of the
level of interest shown by interlocutors. Doesn't get invited out any more.

Arguments:
    A single float or int.

Returns:
    (float) The square.
"""

POW = Operator(_pow, "POW", float, r"[if][if]", force_type=True)
"""
Takes two numbers `x` and `n`, which can be either floats or ints, raises `x`
to the `n`th power, and returns the result as a float. The only member of SUM,
PROD, SQ, and CUBE's circle who stuck with Maths through honours, but went into
quantitative finance, and from there to politics. A Tory, obviously. Wants to be
Prime Minister. Has exactly one political idea, which is wanting to be Prime
Minister.

Arguments:
    x (float or int): the number to be raised to a power.
    n (float or int): the exponent.

Returns:
    (float) `x` raised to the `n`th power
"""

EQ = Operator(_eq, "EQ", bool, r"..")
"""
Compares two objects for equality, returning `True` if they are equal. Natural
enemy of POW. Likes public transport, strong tea, and socialism.

Arguments:
    Any two objects.

Returns:
    (bool)
"""

NEQ = Operator(_neq, "NEQ", bool, r"..")
"""
Compares two objects for inequality, returning `True` if they are not equal.
Works in social care. Spends wekends raising funds for food banks.

Arguments:
    Any two objects.

Returns:
    (bool)
"""

GT = Operator(_gt, "GT", bool, r"[bif][bif]")
"""
'Greater than' comparison Operator. Rich, claims to be 'self-made', but actually
started in business with a loan of £400,000 from the Bank of Mum and Dad, and
a trust-fund to live off of for the three years before the business turned a
profit. Still, regularly proclaims that the only reason poor people haven't
succeeded as in the same fashion is 'laziness and nothing more'. Pally with POW.
Secretly thinks POW's a twat. POW also secretly thinks GT is a twat.

Arguments:
    Any two objects.

Returns:
    (bool)
"""

EGT = Operator(_egt, "EGT", bool, r"[bif][bif]")
"""
'Greater than or equal to' comparison Operator. Claims to have a strong moral
commitment to equality and social justice, which primarily manifests as an
encyclopaedic knowledge of the terms favoured by different marginalised groups
prefer to refer to themselves by, a feeling of moral superiority to anyone
lacking such knowledge, and no actual activism in support of those groups. Gets
really uncomfortable when EQ starts talking about socialism. Thinks Kier
Starmer's great.

Arguments:
    Any two objects.

Returns:
    (bool)
"""

LT = Operator(_lt, "LT", bool, r"[bif][bif]")
"""
'Less than' comparison Operator. Perpetually crushed by low self esteem. A
writer. Actually very good. Surrounded by people who suck.

Arguments:
    Any two objects.

Returns:
    (bool)
"""

ELT = Operator(_elt, "ELT", bool, r"[bif][bif]")
"""
'Less than or equal to' comparison Operator. Works in media. Always the first to
shut down bullying or bigotry when directed at anyone else, but never fails to
make excuses for it and internalise it when directed at self. Will snap, one
day.

Arguments:
    Any two objects.

Returns:
    (bool)
"""

NOT = Operator(_not, "NOT", bool, r"b", apply=True)
"""
Boolean negation operator. An open minded free thinker who does their own
research and doesn't let the authorities and the *lame*stream media tell them
what to think. Which is to say, they Google until they find a source that
supports a position they like the sound of, then stop. Thinks of themself as
a devil's advocate and champion of free and open discussion. Actually just an
arsehole.

Arguments:
    A `bool`, P.

Returns:
    (bool) ~P
"""

OR = Operator(_sum, "OR", bool, r"b*", True, True)
"""
Boolean inclusive disjunction operator. Knows everyone's pronouns.

Arguments:
    P (bool)
    Q (bool)

Returns:
    (bool) P v Q
"""

AND = Operator(_prod, "AND", bool, r"b*", True, True)
"""
Boolean conjunction operator. Works as a back-end developer for Tinder.

Arguments:
    P (bool)
    Q (bool)

Returns:
    (bool) P & Q
"""


TERN = Operator(_tern, "TERN", Any, r"b(.)(.)", apply=True)
"""
A ternary `if P then X else Y` operator. Humanities undergraduate. Perpetually
in a state of agonising over which of two crushes to ask out. Invariably, by the
time they decide, it's too late, their crush is now with someone else.

Arguments:
    P (bool)
    X: can be anything
    Y: can be anything

Returns:
    X if P, else Y
"""

def _pand(arg):
    return arg if type(arg) == pd.Series or type(arg) == pd.DataFrame else pd.Series([arg])


class OperatorFactory:
    def __init__(self):
        self.op_dic = {
            "ID": ID,
            "UNIT_ID": UNIT_ID,
            "CONCAT": CONCAT,
            "SUM": SUM,
            "PROD": PROD,
            "SQ": SQ,
            "CUBE": CUBE,
            "POW": POW,
            "EQ": EQ,
            "NEQ": NEQ,
            "GT": GT,
            "EGT": EGT,
            "LT": LT,
            "ELT": ELT,
            "NOT": NOT,
            "OR": OR,
            "AND": AND,
            "TERN": TERN
        }

    def add_op(
            self, func, name, return_type = Any, arg_regex = "",
            force_type = False, force_type_lossy = False, apply = False,
            return_dtype = None):
        self.op_dic[name] = Operator(
            func, name, return_type = return_type, arg_regex = arg_regex,
            force_type = force_type, force_type_lossy = force_type_lossy,
            apply = apply, return_dtype = return_dtype)

    def __call__(self, names):
        if isinstance(names, str):
            names = [names]
        operator_dictionary = {}
        for name in names:
            try:
                operator_dictionary[name] = self.op_dic[name]
            except KeyError:
                raise AttributeError(f"OperatorFactory lacks Operator {name}")
        return operator_dictionary



def main():
    """More doctests live here.

    >>> df = pd.DataFrame(
    ...     {
    ...         "s1": ["one", "two", "three", "four"],
    ...         "s2": ["something's", "got", "to", "give"],
    ...         "i1": [1,2,3,4],
    ...         "i2": [42, 69, 420, 666],
    ...         "i3": [42, 2, 420, 4],
    ...         "f1": [1.0, 2.0, 3.0, 4.0],
    ...         "f2": [3.142, 2.718, 1.414, 1.618],
    ...         "c1": [1+4j, 2+3j, 3+2j, 4+1j],
    ...         "c2": [0.5+0.866j, -0.5+0.866j, -1+0j, -0.5-0.866j],
    ...         "b1": [False, False, True, True],
    ...         "b2": [True, False, True, False]
    ...     }
    ... )
    >>> df1 = pd.DataFrame(
    ...     {
    ...         "w1": ["something's"],
    ...         "w2": ["got"],
    ...         "w3": ["to"],
    ...         "w4": ["give"],
    ...         "i1": [5],
    ...         "i2": [404],
    ...         "f1": [1.999],
    ...         "f2": [9.0],
    ...         "f3": [3.0],
    ...         "c1": [1+1j],
    ...         "c2": [0.1+0.9j],
    ...         "b1": [False],
    ...         "b2": [True]
    ...     }
    ... )
    >>> df["s1"] = df["s1"].astype("string")
    >>> df["s2"] = df["s2"].astype("string")
    >>> df1["w1"] = df1["w1"].astype("string")
    >>> df1["w2"] = df1["w2"].astype("string")
    >>> df1["w3"] = df1["w3"].astype("string")
    >>> df1["w4"] = df1["w4"].astype("string")
    >>> ID(df["s1"], df1["w1"], df1["w2"], df1["w3"], df1["w4"])
    0      (one, something's, got, to, give)
    1      (two, something's, got, to, give)
    2    (three, something's, got, to, give)
    3     (four, something's, got, to, give)
    Name: ID(s1,something's,got,to,give), dtype: object
    >>> CONCAT('pet', 'that', 'cat')
    'pet that cat'
    >>> CONCAT(df["s1"], df1["w1"], df1["w2"], df1["w3"], df1["w4"])
    0      one something's got to give
    1      two something's got to give
    2    three something's got to give
    3     four something's got to give
    Name: CONCAT(s1,something's,got,to,give), dtype: string
    >>> CONCAT(df["s1"], "something's", "got", "to", "give")
    0      one something's got to give
    1      two something's got to give
    2    three something's got to give
    3     four something's got to give
    Name: CONCAT(s1,something's,got,to,give), dtype: string
    >>> SUM(4, 5)
    9.0
    >>> SUM(df["i1"], df['i2'])
    0     43.0
    1     71.0
    2    423.0
    3    670.0
    Name: SUM(i1,i2), dtype: float64
    >>> SUM(df['i2'], df['f2'], df1['i2'], df1['f2'])
    0     458.142
    1     484.718
    2     834.414
    3    1080.618
    Name: SUM(i2,f2,404,9.0), dtype: float64
    >>> PROD(1, 2, 3, 4)
    24.0
    >>> PROD(df['i1'], df['f1'])
    0     1.0
    1     4.0
    2     9.0
    3    16.0
    Name: PROD(i1,f1), dtype: float64
    >>> PROD(df['i1'], df['i1'], df1['f2'])
    0      9.0
    1     36.0
    2     81.0
    3    144.0
    Name: PROD(i1,i1,9.0), dtype: float64
    >>> SQ(12)
    144.0
    >>> SQ(df['i1'])
    0     1.0
    1     4.0
    2     9.0
    3    16.0
    Name: SQ(i1), dtype: float64
    >>> SQ(df['f1'])
    0     1.0
    1     4.0
    2     9.0
    3    16.0
    Name: SQ(f1), dtype: float64
    >>> CUBE(3)
    27.0
    >>> CUBE(df['i1'])
    0     1.0
    1     8.0
    2    27.0
    3    64.0
    Name: CUBE(i1), dtype: float64
    >>> CUBE(df['f1'])
    0     1.0
    1     8.0
    2    27.0
    3    64.0
    Name: CUBE(f1), dtype: float64
    >>> POW(2, 8)
    256.0
    >>> POW(df['i1'], df['i1'])
    0      1.0
    1      4.0
    2     27.0
    3    256.0
    Name: POW(i1,i1), dtype: float64
    >>> POW(df['i1'], df['f1'])
    0      1.0
    1      4.0
    2     27.0
    3    256.0
    Name: POW(i1,f1), dtype: float64
    >>> POW(df['f1'], df['i1'])
    0      1.0
    1      4.0
    2     27.0
    3    256.0
    Name: POW(f1,i1), dtype: float64
    >>> POW(df['f1'], df['f1'])
    0      1.0
    1      4.0
    2     27.0
    3    256.0
    Name: POW(f1,f1), dtype: float64
    >>> POW(df1['i1'], df['f1'])
    0      5.0
    1     25.0
    2    125.0
    3    625.0
    Name: POW(5,f1), dtype: float64
    >>> POW(df['f1'], df1['f2'])
    0         1.0
    1       512.0
    2     19683.0
    3    262144.0
    Name: POW(f1,9.0), dtype: float64
    >>> EQ(1, 2)
    False
    >>> EQ(1, 1.0)
    True
    >>> EQ(df["i1"], df["f1"])
    0    True
    1    True
    2    True
    3    True
    Name: EQ(i1,f1), dtype: bool
    >>> EQ(df["i3"], df["f1"])
    0    False
    1     True
    2    False
    3     True
    Name: EQ(i3,f1), dtype: bool
    >>> EQ(df["i1"], df1["f3"])
    0    False
    1    False
    2     True
    3    False
    Name: EQ(i1,3.0), dtype: bool
    >>> NEQ(1, 2)
    True
    >>> NEQ(1, 1.0)
    False
    >>> NEQ(df["i1"], df["f1"])
    0    False
    1    False
    2    False
    3    False
    Name: NEQ(i1,f1), dtype: bool
    >>> NEQ(df["i3"], df["f1"])
    0     True
    1    False
    2     True
    3    False
    Name: NEQ(i3,f1), dtype: bool
    >>> NEQ(df["i1"], df1["f3"])
    0     True
    1     True
    2    False
    3     True
    Name: NEQ(i1,3.0), dtype: bool
    >>> GT(df['i1'], df1['f3'])
    0    False
    1    False
    2    False
    3     True
    Name: GT(i1,3.0), dtype: bool
    >>> EGT(df['i1'], df1['f3'])
    0    False
    1    False
    2     True
    3     True
    Name: EGT(i1,3.0), dtype: bool
    >>> LT(df['i1'], df1['f3'])
    0     True
    1     True
    2    False
    3    False
    Name: LT(i1,3.0), dtype: bool
    >>> ELT(df['i1'], df1['f3'])
    0     True
    1     True
    2     True
    3    False
    Name: ELT(i1,3.0), dtype: bool
    >>> NOT(df['b1'])
    0     True
    1     True
    2    False
    3    False
    Name: NOT(b1), dtype: bool
    >>> OR(df['b2'], df['b1'])
    0     True
    1    False
    2     True
    3     True
    Name: OR(b2,b1), dtype: bool
    >>> AND(df['b2'], df['b1'])
    0    False
    1    False
    2     True
    3    False
    Name: AND(b2,b1), dtype: bool
    >>> EQ(df['i3'], TERN(df['b2'], df['i2'], df['i1']))
    0    True
    1    True
    2    True
    3    True
    Name: EQ(i3,TERN(b2,i2,i1)), dtype: bool
    """
    import doctest
    doctest.testmod()

if __name__ == '__main__':
    main()


# ReturnValidator = Callable[[tuple, Any], bool]
# Validator = Callable[[tuple[type|np.dtype],type|np.dtype], bool]
    # def __init__(self, 
    #     func: Callable, 
    #     name: str,
    #     validator: Validator = None, 
    #     return_validator: ReturnValidator|type|np.dtype=None,
    #     force_type: ForceType=ForceType.NO
    # ):
# func, name, validator, return_validator, force_type 