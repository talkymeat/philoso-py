from abc import ABC, abstractmethod
from hd import HierarchicalDict as HD
from typing import Callable
from icecream import ic

ic.enable()

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
    stargs: str|None = None
    kwargs: tuple[str] = ()

    @classmethod
    def from_json(cls, json_, *args, **kwargs):
        return cls(
            *[
                json_.get(cls.addr+[arg], None) 
                for arg 
                in cls.args
            ],
            *(json_.get(cls.addr+[cls.stargs], ()) if cls.stargs else ()),
            **{
                kwarg: json_[cls.addr+[kwarg]] 
                for kwarg 
                in cls.kwargs 
                if cls.addr+[kwarg] in json_
            }
        )

    @property
    @abstractmethod
    def json(self) -> dict:
        pass

nw = 0
ini = 0

class JSONableFunc(JSONable):
    """Decorator class, used to wrap a function f, such that
    `JSONable(f)(*args, **kwargs)` returns the same as f(*args, **kwargs)
    and JSONable(f).json returns a dict containing only the mapping 
    `{'name': f.__name__}`
    """
    nw = 0
    ini = 0

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