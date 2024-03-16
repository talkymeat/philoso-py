from typing import List, Dict, Type, Any, Callable, Container, Protocol, runtime_checkable
import pandas as pd
import numpy as np
from functools import reduce
from type_ify import TypeNativiser, _DoesNothing
from pandas.api.types import is_bool_dtype, is_integer_dtype, is_float_dtype, is_complex_dtype, is_string_dtype, is_bool, is_integer, is_float, is_complex
from enum import Enum
from utils import disjoin_tests, conjoin_tests
from icecream import ic

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

def _sq(*args):
    return args[0]**2

def _cube(*args):
    return args[0]**3

TYPE_TOOLS = {
    float:   {'test': disjoin_tests(is_float, is_float_dtype), 'dtype': np.float64  },
    int:     {'test': disjoin_tests(is_integer, is_integer_dtype), 'dtype': np.int32    },
    bool:    {'test': disjoin_tests(is_bool, is_bool_dtype), 'dtype': np.bool_    },
    complex: {'test': disjoin_tests(is_complex, is_complex_dtype), 'dtype': np.complex64},
    str:     {'test': disjoin_tests(lambda s: isinstance(s, str), is_string_dtype), 'dtype': np.str_     }
}

@runtime_checkable
class ReturnValidator(Protocol):
    def __call__(return_val, *args) -> bool:
        ...

@runtime_checkable
class SimpleValidator(Protocol):
    def __call__(arg) -> bool:
        ...

@runtime_checkable
class Validator(Protocol): 
    def __call__(return_val: type|np.dtype, *args: type|np.dtype) -> bool:
        ...

class ForceType(Enum):
    NO     = 0
    STRICT = 1
    LOSSY  = 2


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
    VALIDATE_RETURNS = True
    _FORCE_TYPE_ERR = (
        'You cannot set an Operator with a dynamic return type' +
        ' to use ForceType.STRICT or ForceType.LOSSY: pass a ' +
        'numpy.dtype or a native python type to `return_validator`' +
        ' to get a static return type, or use the default ' +
        'ForceType.NO for `force_type`.'
    ) 

    def __init__(self, 
        func: Callable, 
        name: str,
        validator: Validator = None, 
        return_validator: ReturnValidator|type|np.dtype=None,
        force_type: ForceType=ForceType.NO
    ):
        self._func      = func
        self.name       = name
        self._validator = (lambda a, *b: True) if validator is None else validator
        self.force_type = lambda x: x # does nothing
        type_forcer = None
        numpty = True
        if isinstance(return_validator, type):
            if return_validator in TYPE_TOOLS:
                self._return_validator = lambda return_value, *args: TYPE_TOOLS[return_validator]['test'](return_value)
                if force_type != ForceType.NO:
                    type_forcer = lambda outval: outval.astype(TYPE_TOOLS[return_validator]['dtype'])
            else:
                self._return_validator = lambda return_value, *args: isinstance(return_value, return_validator)
                if force_type != ForceType.NO:
                    type_forcer = lambda outval: return_validator(outval)
                    numpty = False
        elif isinstance(return_validator, np.dtype):
            self._return_validator = lambda return_value, *args: return_validator==return_value.dtype
            if force_type != ForceType.NO:
                type_forcer = lambda outval: outval.astype(return_validator)
        elif isinstance(return_validator, ReturnValidator):
            self._return_validator = return_validator
            if force_type != ForceType.NO:
                raise ValueError(Operator._FORCE_TYPE_ERR)
        elif return_validator is None:
            self._return_validator = lambda return_value, *args: True
            if force_type != ForceType.NO:
                raise ValueError(Operator._FORCE_TYPE_ERR)
        else:
            raise ValueError(
                'If you set an Operator to have a return_validator, you must either ' +
                'use a callable that takes the args passed to Operator.__call__ and ' +
                'the resulting returned value, and returns boolean, OR a type, OR a ' +
                f'numpy.dtype. You passed a {type(return_validator)}'
            )
        if type_forcer:
            if force_type==ForceType.STRICT:
                if numpty:
                    type_forcer = self._make_strict_np(type_forcer)
                else:
                    type_forcer = self._make_strict_py(type_forcer)
            self.force_type = type_forcer

    def _make_strict_py(self, fn: Callable):
        def strict_func_py(op_output):
            type_forced_output = fn(op_output)
            if op_output == type_forced_output:
                return type_forced_output
            raise TypeError("Cannot convert type without information loss")
        return strict_func_py
        
    def _make_strict_np(self, fn: Callable):
        def strict_func_np(op_output):
            type_forced_output = fn(op_output)
            if (op_output == type_forced_output).all():
                return type_forced_output
            raise TypeError("Cannot convert type without information loss")
        return strict_func_np

    def __call__(self, *args) -> Any:
        """A magic method which makes `Operator`s Callable. Checks that the
        arguments and output are legal: returns the output if everything is OK.

        Raises
        ------
            AttributeError: If arguments are not legal
            TypeError: If output is not legal

        >>> def a_plus_b(*args):
        ...     return args[0] + args[1]
        >>> ID(2, 3)
        (2, 3)
        >>> ID("regard", "that", "cat")
        ('regard', 'that', 'cat')
        >>> CONCAT(2, 3)
        '2 3'
        >>> CONCAT("regard", "that", "cat")
        'regard that cat'
        >>> op_A_PLUS_B_WRONG = Operator(a_plus_b, "A_PLUS_B_WRONG", validator=conjoin_tests(same_args_diff_rtn_validator_factory(int, int, float), num_args_eq_validator(2)), return_validator=int)
        >>> op_A_PLUS_B_WRONG.is_valid(int, str, str)
        False
        >>> op_A_PLUS_B_WRONG(2, 3)
        5
        >>> op_A_PLUS_B_WRONG(2.0, 3.0)
        Traceback (most recent call last):
            ....
        TypeError: Operator A_PLUS_B_WRONG cannot return output 5.0 of type <class 'numpy.float64'>.
        >>> op_A_PLUS_B = Operator(lambda a, b: a+b, "A_PLUS_B", validator=conjoin_tests(same_args_diff_rtn_validator_factory(int, int, float), num_args_eq_validator(2)), return_validator=float, force_type=ForceType.STRICT)
        >>> op_A_PLUS_B_WRONG.is_valid(float, str, str)
        False
        >>> op_A_PLUS_B(2, 3)
        5.0
        >>> op_A_PLUS_B(2.0, 3.0)
        5.0
        >>> op_SUM = Operator(_sum, "SUM_FL", validator=same_args_diff_rtn_validator_factory(Any, int, float), return_validator=float, force_type=ForceType.NO)
        >>> op_SUM_FL = Operator(_sum, "SUM_FL", validator=same_args_diff_rtn_validator_factory(Any, int, float), return_validator=float, force_type=ForceType.STRICT)
        >>> op_SUM_INT = Operator(_sum, "SUM_INT", validator=same_args_diff_rtn_validator_factory(int, int, float), return_validator=int, force_type=ForceType.STRICT)
        >>> op_SUM_INT_LOSSY = Operator(_sum, "SUM_INT_LOSSY", validator=same_args_diff_rtn_validator_factory(int, int, float), return_validator=int, force_type=ForceType.LOSSY)
        >>> op_SUM(2, 3)
        Traceback (most recent call last):
            ....
        TypeError: Operator SUM_FL cannot return output 5 of type <class 'numpy.int64'>.
        >>> op_SUM_FL(2, 3)
        5.0
        >>> op_SUM_INT(2.0, 3.0)
        5
        >>> op_SUM_INT(2.0, 2.5)
        Traceback (most recent call last):
            ....
        TypeError: Cannot convert type without information loss
        >>> op_SUM_INT_LOSSY(2.0, 2.5)
        4
        >>> op_SUM(2.0, 3.0, 5.0)
        10.0
        >>> op_A_PLUS_B.is_valid(int, int, int)
        False
        """
        output = self.force_type(
            self._func(
                *self._preprocess(*args)
            )
        )
        if self._return_validator(output, *args):
            return output
        t = output.dtype if isinstance(output, np.ndarray) else type(output)
        raise TypeError(f'Operator {self.name} cannot return output {output} of type {t}.')


    def is_valid(self, return_type: type|np.dtype, *arg_types: type|np.dtype):
        return self._validator(return_type, *arg_types)

    def __str__(self):
        return f"<{self.name}>" if self.name else ""

    def _preprocess(self, *args):
        return tuple(
            [arg if isinstance(arg, np.ndarray) else np.array(arg) for arg in args]
        )
    
    def drt_simple(self, arg_match: re.Match) -> type:
        if arg_match.groups():
            typeset = {Operator.rev_type_dict[char] for char in ''.join(arg_match.groups())}
            return list(typeset)[0] if len(typeset)==1 else Any
        return self.return_type


class DOPerator(Operator):
    def _preprocess(self, *args):
        return tuple(
            [arg if isinstance(arg, str) else str(arg) for arg in args]
        )
 

class NOPErator(Operator):
    def _preprocess(self, *args):
        return args

def single_validator_factory(t: type|np.dtype|SimpleValidator, simple=False):
    if isinstance(t, type):
        if t in TYPE_TOOLS and not simple:
            return TYPE_TOOLS[t]['test']
        else:
            return lambda x: issubclass(x, t)
    elif isinstance(t, np.dtype):
        return lambda x: x==t
    elif isinstance(t, SimpleValidator):
        return t
    else: 
        raise TypeError(
            'To make a single-type validator with `simple_validator_factory`,' +
            'pass a type, a numpy dtype, or a function'
        )

def monotype_validator_factory(t: type|np.dtype|SimpleValidator, simple=False):
    try:
        test = single_validator_factory(t, simple=simple)
    except TypeError:
        raise TypeError(
            'To make a single-type validator with `monotype_validator_factory`,' +
            'pass a type, a numpy dtype, or a function'
        )
    def validator(return_type: np.dtype|type, *arg_types: np.dtype|type):
        return reduce(
            lambda a, b: test(a) and test(b), 
            arg_types+(return_type,), 
            True
        )
    return validator
    
def same_args_diff_rtn_validator_factory(
        ret_t:   type|np.dtype|SimpleValidator, 
        *ts: type|np.dtype|SimpleValidator, 
        simples=None,
        simple=False
    ):
    if simples is None:
        simples = []
    if len(simples) < len(ts):
        simples += [False]*(len(ts)-len(simples))
    elif len(simples) > len(ts):
        raise ValueError('"simples" cannot be longer than "ts"')
    try:
        arg_test = disjoin_tests(*[single_validator_factory(t, simple=s) for t, s in zip(ts, simples)])
        ret_test = single_validator_factory(ret_t, simple)
    except TypeError:
        raise TypeError(
            'To make a single-type validator with ' +
            '`same_args_diff_rtn_validator_factory`,' +
            'pass a type, a numpy dtype, or a function'
        )
    def validator(return_type: np.dtype|type, *arg_types: np.dtype|type):
        return reduce(
            lambda a, b: arg_test(a) and arg_test(b), 
            arg_types, 
            True
        ) and ret_test(return_type)
    return validator

def num_args_eq_validator(num):
    def validator(return_type: np.dtype|type, *arg_types: np.dtype|type):
        return len(arg_types)==num
    return validator

def num_args_gt_validator(num):
    def validator(return_type: np.dtype|type, *arg_types: np.dtype|type):
        return len(arg_types)>num
    return validator

def num_args_lt_validator(num):
    def validator(return_type: np.dtype|type, *arg_types: np.dtype|type):
        return len(arg_types)<num
    return validator

# func, name, validator, return_validator, force_type
ID = NOPErator(_id, "ID", validator=num_args_gt_validator(0))
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

UNIT_ID = NOPErator(_unit_id, "UNIT_ID")
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

CONCAT = DOPerator(
    _concat, 
    "CONCAT", 
    validator=monotype_validator_factory(str, simple=True),
    return_validator=str
)
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

SUM = Operator(
    _sum, 
    "SUM", 
    validator=same_args_diff_rtn_validator_factory(float, int, float), 
    return_validator=float,
    force_type=ForceType.STRICT
)
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

PROD = Operator(
    _prod, 
    "PROD",
    validator=same_args_diff_rtn_validator_factory(float, int, float), 
    return_validator=float,
    force_type=ForceType.STRICT
)
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

SQ = Operator(
    _sq, 
    "SQ", 
    validator=conjoin_tests(
        same_args_diff_rtn_validator_factory(float, int, float),
        num_args_eq_validator(1)
    ), 
    return_validator=float, 
    force_type=ForceType.STRICT
)
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

CUBE = Operator(
    _cube, 
    "CUBE", 
    validator=conjoin_tests(
        same_args_diff_rtn_validator_factory(float, int, float),
        num_args_eq_validator(1)
    ), 
    return_validator=float, 
    force_type=ForceType.STRICT
)
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

POW = Operator(
    _pow, 
    "POW",
    validator=conjoin_tests(
        same_args_diff_rtn_validator_factory(float, int, float),
        num_args_eq_validator(2)
    ), 
    return_validator=float, 
    force_type=ForceType.STRICT
)
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

EQ = Operator(
    _eq, 
    "EQ", 
    validator=num_args_eq_validator(2), 
    return_validator=bool
)
"""
Compares two objects for equality, returning `True` if they are equal. Natural
enemy of POW. Likes public transport, strong tea, and socialism.

Arguments:
    Any two objects.

Returns:
    (bool)
"""

NEQ = Operator(
    _neq, 
    "NEQ", 
    validator=num_args_eq_validator(2), 
    return_validator=bool
)
"""
Compares two objects for inequality, returning `True` if they are not equal.
Works in social care. Spends wekends raising funds for food banks.

Arguments:
    Any two objects.

Returns:
    (bool)
"""

GT = Operator(
    _gt, 
    "GT",
    validator=conjoin_tests(
        same_args_diff_rtn_validator_factory(float, int, float, bool),
        num_args_eq_validator(2)
    ), 
    return_validator=bool
)
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

EGT = Operator(
    _egt, 
    "EGT",
    validator=conjoin_tests(
        same_args_diff_rtn_validator_factory(float, int, float, bool),
        num_args_eq_validator(2)
    ), 
    return_validator=bool
)
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

LT = Operator(
    _lt, 
    "LT",
    validator=conjoin_tests(
        same_args_diff_rtn_validator_factory(float, int, float, bool),
        num_args_eq_validator(2)
    ), 
    return_validator=bool
)
"""
'Less than' comparison Operator. Perpetually crushed by low self esteem. A
writer. Actually very good. Surrounded by people who suck.

Arguments:
    Any two objects.

Returns:
    (bool)
"""

ELT = Operator(
    _elt, 
    "ELT",
    validator=conjoin_tests(
        same_args_diff_rtn_validator_factory(float, int, float, bool),
        num_args_eq_validator(2)
    ), 
    return_validator=bool
)
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

NOT = Operator(
    np.invert, 
    "NOT",
    validator=conjoin_tests(
        monotype_validator_factory(bool),
        num_args_eq_validator(1)
    ), 
    return_validator=bool
) #  apply=True 
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

OR = Operator(
    _sum, 
    "OR", 
    validator=monotype_validator_factory(bool),
    return_validator=bool, 
    force_type=ForceType.LOSSY    
)
"""
Boolean inclusive disjunction operator. Knows everyone's pronouns.

Arguments:
    P (bool)
    Q (bool)

Returns:
    (bool) P v Q
"""

AND = Operator(
    _prod, 
    "AND", 
    validator=monotype_validator_factory(bool),
    return_validator=bool, 
    force_type=ForceType.LOSSY  
)
"""
Boolean conjunction operator. Works as a back-end developer for Tinder.

Arguments:
    P (bool)
    Q (bool)

Returns:
    (bool) P & Q
"""

def ternary_operator_factory(r_type):
    def validate_ternary(*arg_types, return_type):
        return len(
            arg_types
        )==3 and issubclass(
            arg_types[0], bool
        ) and (
            r_type==return_type==arg_types[1]==arg_types[2]
        )
    return Operator(
        np.where, 
        f"TERN_{str(r_type).upper().replace('.', '_')}", 
        validator=validate_ternary,
        return_validator=r_type
    )

TERN_INT = ternary_operator_factory(int)
TERN_FLOAT = ternary_operator_factory(float)
TERN_COMPLEX = ternary_operator_factory(complex)
TERN_BOOL = ternary_operator_factory(bool)
TERN_STR = ternary_operator_factory(str)

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
            "TERN_INT": TERN_INT,
            "TERN_FLOAT": TERN_FLOAT,
            "TERN_COMPLEX": TERN_COMPLEX,
            "TERN_BOOL": TERN_BOOL,
            "TERN_STR": TERN_STR
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
    >>> for w in df['s1']:  
    ...     print(ID(w, df1["w1"].item(), df1["w2"].item(), df1["w3"].item(), df1["w4"].item()))
    ('one', "something's", 'got', 'to', 'give')
    ('two', "something's", 'got', 'to', 'give')
    ('three', "something's", 'got', 'to', 'give')
    ('four', "something's", 'got', 'to', 'give')
    >>> CONCAT('pet', 'that', 'cat')
    'pet that cat'
    >>> for w in df['s1']:  
    ...     print(CONCAT(w, "something's", "got", "to", "give"))
    one something's got to give
    two something's got to give
    three something's got to give
    four something's got to give
    >>> SUM(4, 5)
    9.0
    >>> SUM(df["i1"], df['i2'])
    array([ 43.,  71., 423., 670.])
    >>> SUM(df['i2'], df['f2'], df1['i2'], df1['f2'])
    array([ 458.142,  484.718,  834.414, 1080.618])
    >>> PROD(1, 2, 3, 4)
    24.0
    >>> PROD(df['i1'], df['f1'])
    array([ 1.,  4.,  9., 16.])
    >>> PROD(df['i1'], df['i1'], df1['f2'])
    array([  9.,  36.,  81., 144.])
    >>> SQ(12)
    144.0
    >>> SQ(df['i1'])
    array([ 1.,  4.,  9., 16.])
    >>> SQ(df['f1'])
    array([ 1.,  4.,  9., 16.])
    >>> CUBE(3)
    27.0
    >>> CUBE(df['i1'])
    array([ 1.,  8., 27., 64.])
    >>> CUBE(df['f1'])
    array([ 1.,  8., 27., 64.])
    >>> POW(2, 8)
    256.0
    >>> POW(df['i1'], df['i1'])
    array([  1.,   4.,  27., 256.])
    >>> POW(df['i1'], df['f1'])
    array([  1.,   4.,  27., 256.])
    >>> POW(df['f1'], df['i1'])
    array([  1.,   4.,  27., 256.])
    >>> POW(df['f1'], df['f1'])
    array([  1.,   4.,  27., 256.])
    >>> POW(df1['i1'], df['f1'])
    array([  5.,  25., 125., 625.])
    >>> POW(df['f1'], df1['f2'])
    array([1.00000e+00, 5.12000e+02, 1.96830e+04, 2.62144e+05])
    >>> EQ(1, 2)
    False
    >>> EQ(1, 1.0)
    True
    >>> EQ(df["i1"], df["f1"])
    array([ True,  True,  True,  True])
    >>> EQ(df["i3"], df["f1"])
    array([False,  True, False,  True])
    >>> EQ(df["i1"], df1["f3"])
    array([False, False,  True, False])
    >>> NEQ(1, 2)
    True
    >>> NEQ(1, 1.0)
    False
    >>> NEQ(df["i1"], df["f1"])
    array([False, False, False, False])
    >>> NEQ(df["i3"], df["f1"])
    array([ True, False,  True, False])
    >>> NEQ(df["i1"], df1["f3"])
    array([ True,  True, False,  True])
    >>> GT(df['i1'], df1['f3'])
    array([False, False, False,  True])
    >>> EGT(df['i1'], df1['f3'])
    array([False, False,  True,  True])
    >>> LT(df['i1'], df1['f3'])
    array([ True,  True, False, False])
    >>> ELT(df['i1'], df1['f3'])
    array([ True,  True,  True, False])
    >>> NOT(df['b1'])
    array([ True,  True, False, False])
    >>> OR(df['b2'], df['b1'])
    array([ True, False,  True,  True])
    >>> AND(df['b2'], df['b1'])
    array([False, False,  True, False])
    >>> EQ(df['i3'], TERN_INT(df['b2'], df['i2'], df['i1']))
    array([ True,  True,  True,  True])
    """
    import doctest
    doctest.testmod()

if __name__ == '__main__':
    main()
