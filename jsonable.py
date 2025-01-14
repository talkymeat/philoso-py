from abc import ABC, abstractmethod
from hd import HierarchicalDict as HD
from typing import Callable

class JSONable(ABC):
    @classmethod
    @abstractmethod
    def from_json(cls, json_: HD, *args, **kwargs):
        pass

    @property
    @abstractmethod
    def json(self) -> dict:
        pass

class SimpleJSONable(JSONable):
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

class JSONableFunc(JSONable):
    def __init__(self, f: Callable):
        self.f = f

    def __call__(self, *args, **kwargs):
        return self.f(*args, **kwargs)
    
    def from_json(cls, json_, f):
        return cls(f)

    @property
    def json(self) -> dict:
        return {'name': self.f.__name__}
    
    def __str__(self):
        return f"{self.f.__name__}(*args, **kwargs)"
    
    def __repr__(self):
        return self.__str__()
    
    @property
    def __name__(self)->str:
        return self.f.__name__

