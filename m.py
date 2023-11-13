# from icecream import ic
from typing import TypeAlias, Hashable, Iterable, Protocol, Mapping, Sequence, Collection
from utils import collect
from functools import reduce
from abc import ABC


# just make a zip function that can be imported

def zip(*args: Iterable, strict=False, force=False, **kwargs) -> 'MTuple':
    if force or sum([isinstance(arg, MSequence) for arg in args]):
        minlen, maxlen= (
            lambda lens: [f(lens) for f in (min, max)]
        )(
            [len(a) for a in args]
        )
        if strict and minlen!=maxlen:
            raise ValueError(
                f"zip() arguments should be the same length in strict mode")
        i=0
        while i < minlen:
            yield MTuple(*[t[i] for t in args])
            i+=1 # sdfdsfssdfd
    else:
        return __builtins__.zip(*args, **kwargs)
    
class M(Collection, ABC):
    
    @classmethod
    def get_mtype(cls, obj):
        ms = {tuple: MTuple, list: MList, dict: MDict, str: MString, bytes: MBytes} 
        for kt, vt in ms.items():
            if isinstance(obj, kt):
                return vt
        else: 
            return None

    # def __new__(cls, *args, **kwargs):
    #     ic('start?')
    #     ic(args)
    #     ic(len(args))
    #     if ic(len(args)) != 1 or not ic(isinstance(args[0], Collection)):
    #         ic('if True')
    #         return tuple.__new__(MTuple, *args, **kwargs)
    #     ic('berp')
    #     T = ic(ic(cls.get_mtype(args[0])) or MTuple)
    #     out = ic(type(args[0])).__new__(T, ('poopy')) # ic(args[0]), 
    #     ic(type(out))
    #     ic(isinstance(out, ic(cls)))
    #     return ic(out)

class MSequence(Sequence, ABC):

    def __getitem__(self, *idcs):
        out = type(self)()
        idcs = idcs[0] if isinstance(idcs[0], tuple) else idcs
        for i in idcs:
            if isinstance(i, int):
                if len(idcs)==1:
                    return tuple.__getitem__(self, i)
                ii = slice(i, i+1)
            else:
                ii = i
            out += super(ABC, self).__getitem__(ii)    
        return out
    
    def __mul__(self, other: int|Iterable) -> 'MSequence':
        if isinstance(other, int):
            return self.__class__(*super().__mul__(other))
        elif isinstance(other, Iterable):
            return self.__class__(*[
                self.__class__((
                    *collect(a, tuple, empty_if_none=True), 
                    *collect(b, tuple, empty_if_none=True)
                )) for a in self for b in other
            ])
        else:
            raise ValueError(
                "unsupported operand type(s) for " +
                f"** or pow(): '{self.__class__}' and '{type(other)}'"
            )
    

    def __rmul__(self, other: int|Sequence) -> 'MSequence':
        if isinstance(other, int):
            return self * other
        elif isinstance(other, Sequence):
            return self.__class__(*other) * self
    
    def __add__(self, other: Iterable) -> 'MSequence':
        """Doesn't do anything special, but make sure adding tuples and MTuples
        always results in a MTuple

        XXX and likewise with MLists etc

        >>> egs = [MTuple(1,2) + MTuple(3,4), (1,2) + MTuple(3,4), MTuple(1,2) + (3,4), (1,2) + (3,4)]
        >>> egs
        [(1, 2, 3, 4), (1, 2, 3, 4), (1, 2, 3, 4), (1, 2, 3, 4)]
        >>> [type(t) for t in egs]
        [<class '__main__.MTuple'>, <class '__main__.MTuple'>, <class '__main__.MTuple'>, <class 'tuple'>]
        >>> [len(t) for t in egs]
        [4, 4, 4, 4]
        """
        return self.__class__(super().__add__(other))
    
    def __radd__(self, other: Iterable) -> 'MSequence':
        return self.__class__(other.__add__(self))
    
    def __pow__(self, exponent: int) -> 'MSequence':
        if isinstance(exponent, int):
            if not exponent:
                return self.__class__()
            outval=self
            for i in range(exponent-1):
                outval *= self
            return outval
        else:
            raise ValueError(
                "unsupported operand type(s) for " +
                f"** or pow(): 'tuple' and '{type(exponent)}'"
            )
        
    def __rpow__(self, base: int):
        if base > 1:
            if len(self)>1:
                chains = self.__class__(self.__class__(*(None,)*len(self)), self.__class__(*[(v,) for v in self]))
                for i in range(3, base+1):
                    chains += (self.__class__(*[(a,) for a in chains[-1]]),)
                return self.__class__(*reduce(lambda a,b:a*b, zip(*chains)))
            elif len(self):
                return self.__class__(self, ())
            else:
                return self
        elif base:
            return (self, )
        else:
            raise ValueError('You can only raise a positive int to the power of a tuple')
        
class MTuple(MSequence, tuple):
    def __init__(self, *args, **kwargs):
        """
        
        >>> mt = MTuple((1,2,3))
        >>> mtt = MTuple((4,5,6))
        >>> mt
        (1, 2, 3)
        >>> mtt
        (4, 5, 6)
        >>> mmtt = MTuple((1,2,3), (4,5,6), (7,8,9), ())
        >>> mmtt
        ((1, 2, 3), (4, 5, 6), (7, 8, 9), ())
        >>> len(mmtt)
        4
        >>> type(mt), type(mtt), type(mmtt)
        (<class '__main__.MTuple'>, <class '__main__.MTuple'>, <class '__main__.MTuple'>)
        >>> # sum([isinstance(t, MTuple) for t in mmtt]) # Get all M's working before recursing
        >>> mtmtt = mt*mtt
        >>> type(mtmtt)
        <class '__main__.MTuple'>
        >>> mtmtt
        ((1, 4), (1, 5), (1, 6), (2, 4), (2, 5), (2, 6), (3, 4), (3, 5), (3, 6))
        >>> type(mtmtt[0])
        <class '__main__.MTuple'>
        >>> mtmtt[0,1]
        ((1, 4), (1, 5))
        >>> mtmtt[0,2,5]
        ((1, 4), (1, 6), (2, 6))
        >>> mtmtt[0:3,5]
        ((1, 4), (1, 5), (1, 6), (2, 6))
        >>> 2**mt
        ((), (3,), (2,), (2, 3), (1,), (1, 3), (1, 2), (1, 2, 3))
        >>> sum([len(x**mt)==x**len(mt) for x in range(1,7)])
        6
        """
        arg = args if len(args) != 1 else args[0]
        tuple.__init__(arg)
           
    def __new__(cls, *args, **kwargs):
        r = args[0] if len(args) == 1 and isinstance(args[0], Sequence) else args
        return super().__new__(cls, r, **kwargs)
    
    def __mul__(self, other):
        return super().__mul__(other)
    
    def __len__(self):
        return tuple.__len__(self)

class MList(MSequence, list):
    pass
        
class MString(MSequence, str):
    pass

class MBytes(MSequence, bytes):
    pass


class Multiplicable(Protocol):
    def __mul__(self, other):
        ...

class Addable(Protocol):
    def __add__(self, other):
        ...

class Exponentiable(Protocol):
    def __pow__(self, other):
        ...

# class arithmetise:
#     NAMES = {'__add__', '__mul__', '__pow__'}

#     def __call__(val, method_name = None):
#         if method_name in arithmetise.NAMES:
#             return (
#                 val 
#                 if hasattr(val, method_name) 
#                 else MDict(**val) 
#                 if isinstance(val, dict) 
#                 else MTuple(*val) 
#                 if isinstance(val, tuple) 
#                 else MTuple(val)
#             )
#         else:
#             raise ValueError('blehh do this later')





class MDict(dict):
    def __add__(self, other: dict) -> 'MDict':
        return {**self, **other, **{k: self[k] + other[k] for k in self if k in other}}
    
    def __radd__(self, other: dict) -> 'MDict':
        return self+other
    
    def __mul__(self, other: Mapping|Multiplicable) -> 'MDict':
        if isinstance(other, Mapping):
            return {
                M(k)*M(k_): M(v)*M(v_) 
                for k, v in self.items() 
                for k_, v_ in other.items()
            }
        else:
            return {k: M(v)*M(other) for k, v in self.items()}


CTDict: TypeAlias = MDict[Hashable, int]

class MCounter(CTDict):
    def __getitem__(self, __key: Hashable) -> int:
        return super().__getitem__(__key) if __key in self else 0
                
        
if __name__ == '__main__':
    import doctest
    doctest.testmod()
    

