import pandas as pd
from pandas.api.types import is_bool_dtype, is_integer_dtype, is_float_dtype, is_complex_dtype, is_string_dtype
import numpy as np
from icecream import ic
from typing import Callable, Sequence, Any

class _DoesNothing:
    pass

class TypeNativiser:
    def __init__(self, *type_tests: Sequence[tuple[Callable[Any, bool], type]], dic = None):
        self.dtype_dict = dic if dic else {}
        self.dtype_dict[pd.core.arrays.string_.StringDtype] = str
        self.type_tests = [
            (is_float_dtype,   float  ), 
            (is_integer_dtype, int    ), 
            (is_bool_dtype,    bool   ), 
            (is_complex_dtype, complex),
            (is_string_dtype,  str    )
        ] + list(type_tests)
        self.type_set = {tt[1] for tt in self.type_tests}


    def type_ify(self, arg, default = 'raise'):
        """Method to allow operators that take arguments of type `t` to also
        take `pandas.Series` of an equivalent dtype. If given an argument that
        is not a `pandas.Series`, it just returns the type of the argument. If
        the argument *is* a pandas.Series, it first checks `Operator.dtype_dict`
        for an equivalence, and returns the type recorded there as equivalent,
        if there is one. If not, it searches the values of `Operator.type_dict`,
        which are Python native types, for one for which `arg.dtype ==
        native_type` evaluates to `True`. If one is found, this equivalence is
        recorded in the `dtype_dict`, to save computation next time.

        >>> tn = TypeNativiser()
        >>> df = pd.DataFrame(
        ...     {
        ...         "strings": ["one", "two", "three", "four"],
        ...         "ints": [1,2,3,4],
        ...         "floats": [1.0, 2.0, 3.0, 4.0],
        ...         "complexes": [1+4j, 2+3j, 3+2j, 4+1j],
        ...         "bools": [True, False, True, False]
        ...     }
        ... )
        >>> df["strings"] = df["strings"].astype("string")
        >>> df["dates"] = pd.Series(pd.date_range("20130101", periods=4))
        >>> ng = [
        ...     np.float_(1),
        ...     np.single(1),
        ...     np.half(1),
        ...     np.csingle(1),
        ...     np.intp(1),
        ...     np.longdouble(1),
        ...     np.ushort(1),
        ...     np.clongfloat(1),
        ...     np.bool_(1),
        ...     np.ulonglong(1),
        ...     np.int_(1),
        ...     np.longlong(1),
        ...     np.double(1),
        ...     np.ubyte(1),
        ...     np.short(1),
        ...     np.uint(1),
        ...     np.uintc(1),
        ...     np.cfloat(1),
        ...     np.unicode_(1),
        ...     np.singlecomplex(1),
        ...     np.cdouble(1),
        ...     np.byte(1),
        ...     np.intc(1),
        ...     np.complex_(1),
        ...     np.string_(1),
        ...     np.longfloat(1),
        ...     np.longcomplex(1),
        ...     np.clongdouble(1),
        ...     np.uintp(1),
        ... ]
        >>> for t in ["str", 0, 0.0, 0.0+1j, True, None]:
        ...     print(tn.type_ify(t))
        ...
        <class 'str'>
        <class 'int'>
        <class 'float'>
        <class 'complex'>
        <class 'bool'>
        <class '__main__._DoesNothing'>
        >>> for h in df:
        ...     print(tn.type_ify(df[h]))
        ...
        <class 'str'>
        <class 'int'>
        <class 'float'>
        <class 'complex'>
        <class 'bool'>
        <class '__main__._DoesNothing'>
        >>> for g in ng:
        ...     print(tn.type_ify(g))
        ...
        <class 'float'>
        <class 'float'>
        <class 'float'>
        <class 'complex'>
        <class 'int'>
        <class 'float'>
        <class 'int'>
        <class 'complex'>
        <class 'bool'>
        <class 'int'>
        <class 'int'>
        <class 'int'>
        <class 'float'>
        <class 'int'>
        <class 'int'>
        <class 'int'>
        <class 'int'>
        <class 'complex'>
        <class 'str'>
        <class 'complex'>
        <class 'complex'>
        <class 'int'>
        <class 'int'>
        <class 'complex'>
        <class 'str'>
        <class 'float'>
        <class 'complex'>
        <class 'complex'>
        <class 'int'>
        """
        ty = type(arg)
        if ty in self.type_set:
            return ty
        else:
            for tt in self.type_tests:
                if tt[0](arg):
                    return tt[1]
        return _DoesNothing

def main():
    import doctest
    doctest.testmod()


if __name__ == '__main__':
    main()
