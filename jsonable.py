from abc import ABC, abstractmethod
from hd import HierarchicalDict as HD
from utils import unfold_lists
from typing import Callable
from icecream import ic



ic.disable()

class JSONable(ABC):
    """In order to make it easy to recreate a model from JSON, some 
    classes must be able to output JSON of their params, so as to be 
    replicable, and be able to re-instantiate an object from its JSON.
    Abstract Base Class defining this functionality
    """
    @classmethod
    @abstractmethod
    def from_json(cls, json_: HD, *args, **kwargs):
        pass

    @property
    @abstractmethod
    def json(self) -> dict:
        pass


class SimpleJSONable(JSONable):
    """Partial implementation of JSONable, with the bare bones of
    a universal `from_json` implementation, and the `json` property 
    left abstract.
    """
    addr: list[str] = []
    args: tuple[str] = ()
    arg_source_order: list[bool]=None
    stargs: str|None = None
    kwargs: tuple[str] = ()

    @classmethod
    def from_json(cls, json_, *args, **kwargs):
        json_ = HD(json_)
        addr = cls.make_addr(locals())
        json_args = [
            json_.get(addr+[arg], None) for arg in cls.args
        ]
        if cls.arg_source_order is None:
            cls.arg_source_order = [False] * len(json_args) + [True] * len(args)
        inst_args = unfold_lists(cls.arg_source_order, json_args, args)
        return cls(
            *inst_args,
            *(json_.get(addr+[cls.stargs], ()) if cls.stargs else ()),
            **{
                kwarg: json_[addr+[kwarg]]
                for kwarg 
                in cls.kwargs 
                if addr+[kwarg] in json_ and kwarg not in kwargs
            },
            **kwargs
        )

    @property
    @abstractmethod
    def json(self) -> dict:
        pass

    @classmethod
    def make_addr(cls, locals_: dict):
        return [locals_.get(k[1:], locals_.get('kwargs', k[1:]).get(k[1:], k[1:])) if isinstance(k, str) and k.startswith('$') else k for k in cls.addr]


class JSONableFunc(JSONable):
    """Decorator class, used to wrap a function f, such that
    `JSONable(f)(*args, **kwargs)` returns the same as f(*args, **kwargs)
    and JSONable(f).json returns a dict containing only the mapping 
    `{'name': f.__name__}`
    """

    def __new__(cls, f):
        """Create a new `JSONableFunc` if `f` is not a `JSONableFunc`;
        otherwise, just return `f`.
        """
        if isinstance(f, cls):
            return f
        return super().__new__(cls)

    def __init__(self, f: Callable):
        """If `f` is a JSONableFunc, `__new__` will ensure `self` here
        is actually just `f`, in which case, it already has an `f` attr,
        and it need not be set. Otherwise, set the `f` attr.
        """
        if self!=f:
            self.f = f
        # self.__doc__ = f.__doc__

    def __call__(self, *args, **kwargs):
        """Calls f"""
        return self.f(*args, **kwargs)
    
    @classmethod
    def from_json(cls, json_: dict, f: Callable, *args, **kwargs):
        """This is something of a dummy `from_json`, as it ignores the
        actual json and just returns the function `f`, wrapped as a 
        `JSONableFunc` if it isn't already. However, ModelFactory may in
        places have a selection of `Callables`, some of which are simply 
        functions, while others are custom classes with a `__call__` method,
        which require parameters to initialise; Having `JSONableFuncs` in
        this context rather than bare functions allows them to be easily 
        handled by code in the same way as the custom `Callables`

        Parameters
        ----------
        json_ : dict
            JSON data. Not used
        f : Callable
            The function to be wrapped by `JSONableFunc`

        Returns
        -------
            JSONableFunc

        >>> def fn():
        ...     return 'fn'
        >>> fkn = JSONableFunc.from_json({}, fn)
        >>> fknl = JSONableFunc.from_json({}, fkn)
        >>> assert not isinstance(fn, JSONableFunc)
        >>> assert isinstance(fkn, JSONableFunc)
        >>> assert isinstance(fknl, JSONableFunc)
        >>> assert not isinstance(fkn.f, JSONableFunc)
        >>> assert not isinstance(fknl.f, JSONableFunc)
        >>> assert fn == fkn.f
        >>> assert fkn.f == fknl.f
        >>> assert fkn != fkn.f
        >>> assert fknl != fknl.f
        >>> assert 'fn' == fn()
        >>> assert fn() == fkn()
        >>> assert fkn() == fknl()
        >>> assert 'fn' == fknl()
        """
        # return f if isinstance(f, cls) else cls(f)
        return cls(f)

    @property
    def json(self) -> dict:
        """Returns a dict giving the name of the function `f`"""
        return {'name': self.__name__}
    
    def __str__(self):
        return f"{self.__name__}(*args, **kwargs)"
    
    def __repr__(self):
        return self.__str__()
    
    @property
    def __name__(self)->str:
        """Same name as self.f"""
        return self.f.__name__



def main():
    import doctest
    doctest.testmod()

if __name__ == '__main__':
    main()