import pandas as pd
import numpy as np

class _DoesNothing:
    pass

class TypeNativiser:
    def __init__(self, *types, dic = None):
        self.dtype_dict = dic if dic else {}
        self.dtype_dict[pd.core.arrays.string_.StringDtype] = str
        self.type_list = [int, bool, float, complex] + list(types)


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
        <class 'NoneType'>
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
        <class 'numpy.float128'>
        <class 'int'>
        <class 'numpy.complex256'>
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
        <class 'bytes'>
        <class 'numpy.float128'>
        <class 'numpy.complex256'>
        <class 'numpy.complex256'>
        <class 'int'>
        """
        if isinstance(arg, pd.Series) or isinstance(arg, np.generic):
            dt = self.dtype_dict.get(
                arg.dtype,
                self.dtype_dict.get(type(arg.dtype), None)
            )
            if dt:
                return dt
            if isinstance(arg, pd.Series):
                for k in self.type_list:
                    if arg.dtype == k:
                        self.dtype_dict[arg.dtype] = k
                        return k
                self.dtype_dict[arg.dtype] = _DoesNothing
                return _DoesNothing
            ty = type(arg.item())
            self.dtype_dict[type(arg)] = ty
            return ty
        return type(arg)

def main():
    import doctest
    doctest.testmod()


if __name__ == '__main__':
    main()
