from icecream import ic
from collections.abc import Sequence, Mapping, Hashable
from typing import Any

class HierarchicalDict(dict):
    def __init__(self, map: dict=None):
        """Initialiser ensures that any value in `map` that is also a `dict` is
        also converted to a `HierarchicalDict`, and so on recursively

        >>> hd1 = HierarchicalDict({'a': 1, 'b': 2})
        >>> hd2 = HierarchicalDict({'a': 1, 'b': {'c': 3, 'd': 4}})
        >>> hd3 = HierarchicalDict({'a': 1, 'b': {'c': 3, 'd': {'e': 5, 'f': 6}}})
        >>> assert isinstance(hd1, HierarchicalDict)
        >>> assert isinstance(hd2, HierarchicalDict)
        >>> assert isinstance(hd2['b'], HierarchicalDict)
        >>> assert isinstance(hd3, HierarchicalDict)
        >>> assert isinstance(hd3['b'], HierarchicalDict)
        >>> assert isinstance(hd3['b']['d'], HierarchicalDict)
        >>> assert isinstance(hd3[['b', 'd']], HierarchicalDict)
        """
        if map is None:
            super().__init__({})
        else:
            super().__init__({k: (self.__class__(v) if isinstance(v, dict) else v) for k, v in map.items()})

    def __getitem__(self, key):
        """Accepts any valid key (that is, any `Hashable`) OR any `list` 
        of one or more `Hashable`s. Lists may not be used as keys in normal 
        `dict`s, nor indeed in `HierarchicalDict`s, as they have no `__hash__` 
        value, but here they are interpretted as sequences of keys, functioning
        as 'addresses' of values in a hierarchy of nested `dict`s.

        A `HierarchicalDict` that is lower in the hierarchy will inherit 
        key-value pairs recursively from its parent, unless overriden by a local
        key-value pair

        IMPORTANT: `hd['k1', 'k2']` uses a `tuple`, `('k1', 'k2')`, as a key, 
        whereas `hd[['k1', 'k2']]`: uses a `list`, `['k1', 'k2']`, as an address

        ```
        HierarchicalDict(
            {('k1', 'k2'): 2}
        )['k1', 'k2']           # returns 2
        HierarchicalDict(
            {('k1', 'k2'): 2}
        )[['k1', 'k2']]         # KeyError
        HierarchicalDict(
            {'k1': {'k2': 5}}
        )['k1', 'k2']           # KeyError
        HierarchicalDict(
            {'k1': {'k2': 5}}
        )[['k1', 'k2']]         # returns 5
        ```

        Parameters
        ----------
            key : Hashable or list[Hashable]

        Returns
        -------
            Any

        >>> hd = HierarchicalDict({'a': 1, 'z': 7, 'b': {'c': 3, 'd': {'e': 5, 'f': 6, 'z': 8}}})
        >>> hd['a']
        1
        >>> hd['z']
        7
        >>> hd[['a']]
        1
        >>> hd['b']
        {'c': 3, 'd': {'e': 5, 'f': 6, 'z': 8}}
        >>> hd[['b']]
        {'c': 3, 'd': {'e': 5, 'f': 6, 'z': 8}}
        >>> hd[['b', 'c']]
        3
        >>> hd[['b', 'z']]
        7
        >>> hd[['b', 'd']]
        {'e': 5, 'f': 6, 'z': 8}
        >>> hd[['b', 'd', 'e']]
        5
        >>> hd[['b', 'd', 'f']]
        6
        >>> hd[['b', 'd', 'z']]
        8
        >>> hd[['b', 'd', 'a']]
        1
        >>> hd[['b', 'd', 'c']]
        3
        >>> hd[['a', 'c']]
        Traceback (most recent call last):
        ...
        KeyError: "'a' (of ['a', 'c']) is mot a dict"
        >>> hd[['z', 'f']]
        Traceback (most recent call last):
        ...
        KeyError: "'z' (of ['z', 'f']) is mot a dict"
        >>> hd[['q', 'f']]
        Traceback (most recent call last):
        ...
        KeyError: 'q'
        """
        # try-except ensures that if the key cannot be found, the same
        # exceptions will be raise as with a normal `dict` (tho the 
        # `except` case also handles the case where the addressed kv pair
        # is in a child `dict`, but the target `dict` does not itself 
        # contain them, but rather inherits them from a dict that is 
        # above it in the hierarchy) 
        try:
            # if the `key` isn't a list, it is a 'bare' key, which 
            # behaves as normal for keys in dicts
            if not isinstance(key, list):
                return super().__getitem__(key)
            else:
                # a key-list containing a single key is equivalent to 
                # the bare key
                if len(key)==1:
                    return super().__getitem__(key[0])
                # an empty list contains no keys, and is not a key, so
                # raises an AttributeError
                elif not key:
                    raise AttributeError(
                        'An empty list is an invalid key for ' +
                        'HierarchicalDict: Provide a Hashable object ' +
                        'or a list of one or more Hashables.'
                    )
                # If the key-list has more than one key, make sure the
                # first key *exists* ...
                elif super().__contains__(key[0]):
                    it = super().__getitem__(key[0])
                    # ... and the corresponding value *is a `dict`* ...
                    if isinstance(it, dict):
                        # ... if so, recursively call __getitem__ on that
                        # `dict`, using the same list with its head popped.
                        return it.__getitem__(key[1:])
                    # if it isn't a `dict`, the next key in the list can't
                    # be looked up, so a KeyError is raised 
                    raise KeyError(
                        f'{repr(key[0])} (of {key}) is mot a dict'
                    )
                else:
                    raise KeyError(key[0])
        # if a KeyError is raised (in particular, raised by the recursive
        # __getitem__ call) maybe the key it was looking for at the *end*
        # of the keylist doesn't exist in the *downstream* `dict`, but *is*
        # in this `dict`, which it inherits from...
        except KeyError as e:
            if isinstance(key, list):
                # ... if so, return the inherited value
                try:
                    return super().__getitem__(key[-1])
                except KeyError:
                    raise e
            # Otherwise, raise the same error `dict` would raise
            raise e
        
    def __delitem__(self, key):
        """Deletes key-value pairs, given a key or list of keys. A `list`  
        cannot be a key in its own right, as it is not `Hashable`; but a `list` 
        of `Hashables` can be used to address a value in a nested hierarchy of
        `dict`s. An item in `self` can be deleted using the corresponding
        `Hashable` key, while items in `self`, and recursively in `self`'s
        children, can be deleted with a list of keys.

        Note that while the key-value pairs of a parent `dict` are recursively
        inherited by its children (unless overriden), an item can only be 
        deleted at its 'local' address.
        
        Parameters
        ----------
            key : Hashable or list[Hashable]

        Returns
        -------
            None

        >>> hd = HierarchicalDict({'a': 1, 'b': {'c': 3, 'd': {'e': 5, 'f': 6}}})  
        >>> del hd['a']
        >>> hd
        {'b': {'c': 3, 'd': {'e': 5, 'f': 6}}}
        >>> del hd[['b', 'c']]
        >>> hd
        {'b': {'d': {'e': 5, 'f': 6}}}
        >>> del hd[['b', 'd', 'e']]
        >>> hd
        {'b': {'d': {'f': 6}}}
        >>> del hd[['b', 'd', 'f']]
        >>> hd
        {'b': {'d': {}}}
        >>> del hd[['b', 'd']]
        >>> hd
        {'b': {}}
        """
        if isinstance(key, list):
            if len(key)==1:
                key = key[0]
            elif len(key) > 1:
                return self[key[0]].__delitem__(key[1:])
        return super().__delitem__(key)
    
    def __setitem__(self, key, value):
        """Sets key-value pairs, given a key or list of keys. A `list`  
        cannot be a key in its own right, as it is not `Hashable`; but a `list` 
        of `Hashables` can be used to address a value in a nested hierarchy of
        `dict`s. If a key-list is given, any non-final members that do not 
        exist will be created. If a non-final key in the list exists in the 
        specified `dict`, but the value it points to is not a `dict`, a dict 
        will be created, which will replace the existing value.

        Parameters
        ----------
            key : Hashable or list[Hashable]
            value : Any

        Returns
        -------
            None

        >>> hd = HierarchicalDict({})
        >>> hd['a'] = 1
        >>> hd
        {'a': 1}
        >>> hd[['a', 'b']] = 2
        >>> hd
        {'a': {'b': 2}}
        >>> hd[['x', 'y', 'z']] = 3
        >>> hd
        {'a': {'b': 2}, 'x': {'y': {'z': 3}}}
        >>> hd['q'] = {'f': {}}
        >>> hd
        {'a': {'b': 2}, 'x': {'y': {'z': 3}}, 'q': {'f': {}}}
        >>> assert isinstance(hd, HierarchicalDict)
        >>> assert isinstance(hd['a'], HierarchicalDict)
        >>> assert isinstance(hd['x'], HierarchicalDict)
        >>> assert isinstance(hd[['x', 'y']], HierarchicalDict)
        >>> assert isinstance(hd['q'], HierarchicalDict)
        >>> assert isinstance(hd[['q', 'f']], HierarchicalDict)
        """
        if isinstance(value, dict):
            # `dict` values should be converted to `HierarchicalDict`
            value = self.__class__(value)
        # If `key` is a list, it's an address: if not, it is a 'bare key', and
        # should be handled like any `dict` handles any key
        if isinstance(key, list):
            if len(key)==1:
                # if key-list contains only one key, the key should be treated
                # the same as a bare key
                key = key[0]
            # If a key-list contains more than one key...
            elif len(key) > 1:
                # ... and the first key does not exist or points to a non-dict
                # value ...
                if key[0] not in self or not isinstance(self[key[0]], Mapping):
                    # ... then set the first key to point to a new empty
                    # HierarchicalDict  
                    self.__setitem__(key[0], HierarchicalDict({}))
                # Then, the `HierarchicalDict` the first key points to (new or
                # old) should have the tail of the keylist set with the value,
                # recursively
                return self[key[0]].__setitem__(key[1:], value)
        # This handles the case where a bare key is given, or a singleton
        # keylist is converted to a bare key
        return super().__setitem__(key, value)

    def __contains__(self, key: object) -> bool:
        """Checks for the presence of a key in a HierarchicalDict, or, using
        a list of keys, recursively checks for a key in a child dict. A `list`  
        cannot be a key in its own right, as it is not `Hashable`; but a `list` 
        of `Hashables` can be used to address a value in a nested hierarchy of
        `dict`s. Since `dict`s in a `HierarchicalDict` hierarchy inherit 
        key-value pairs from their parents, if the final key in a key-list is
        not in the target `dict`, but is in some dict above it in the key-list, 
        `__contains__` will return `True`.

        Parameters
        ----------
            key : Hashable or list[Hashable]

        Returns
        -------
            bool

        >>> hd = HierarchicalDict({'a': {'b': 2}, 'x': {'y': {'z': 3}}, 'q': {'f': {}}, 'e': 0})
        >>> assert 'a' in hd
        >>> assert ['a'] in hd
        >>> assert ['a', 'b'] in hd
        >>> assert ['x', 'y', 'z'] in hd
        >>> assert ['x', 'y', 'e'] in hd
        >>> assert 'f' not in hd
        >>> assert 'g' not in hd
        >>> assert ['a', 'y', 'z'] not in hd
        """
        # If key is not a key-list (a 'bare' key), use the super-class __contains__
        if not isinstance(key, list):
            return super().__contains__(key)
        # If the key is a key-list of length 1, use the sole key in the list like a
        # bare key 
        if len(key)==1:
            return super().__contains__(key[0])
        # Empty lists can't be keys or key-lists
        elif not key:
            raise IndexError(
                'An empty list is an invalid index for ' +
                'HierarchicalDict: Provide a Hashable object ' +
                'or a list of one or more Hashables.'
            )
        # If key-list contains multiple keys, but the current `dict` contains
        # the final key, the target dict will inherit that key from the current
        # dict, so return `True`
        elif super().__contains__(key[-1]):
            return True
        # Otherwise, if the first key in the list maps to a nested `dict`, call
        # __contains__ on that dict using the rest of the list 
        elif super().__contains__(key[0]) and isinstance(self.__getitem__(key[0]), dict):
            return self.__getitem__(key[0]).__contains__(key[1:])
        # Otherwise otherwise, the key is not here
        else:
            return False
        
    def get(self, key, default=None):
        """Returns the value mapped by `key`, or if there is no such value
        returns the default.

        Accepts any valid key (that is, any `Hashable`) OR any `list` 
        of one or more `Hashable`s. Lists may not be used as keys in normal 
        `dict`s, nor indeed in `HierarchicalDict`s, as they have no `__hash__` 
        value, but here they are interpretted as sequences of keys, functioning
        as 'addresses' of values in a hierarchy of nested `dict`s.

        A `HierarchicalDict` that is lower in the hierarchy will inherit 
        key-value pairs recursively from its parent, unless overriden by a local
        key-value pair

        Parameters
        ----------
            key : Hashable or list[Hashable]
            default : Any

        Returns
        -------
            Any

        >>> hd = HierarchicalDict({'a': 1, 'z': 7, 'b': {'c': 3, 'd': {'e': 5, 'f': 6, 'z': 8}}})
        >>> hd.get('a', 'asdfgh')
        1
        >>> hd.get('z', 'asdfgh')
        7
        >>> hd.get(['a'], 'asdfgh')
        1
        >>> hd.get('b', 'asdfgh')
        {'c': 3, 'd': {'e': 5, 'f': 6, 'z': 8}}
        >>> hd.get(['b'], 'asdfgh')
        {'c': 3, 'd': {'e': 5, 'f': 6, 'z': 8}}
        >>> hd.get(['b', 'c'], 'asdfgh')
        3
        >>> hd.get(['b', 'z'], 'asdfgh')
        7
        >>> hd.get(['b', 'd'], 'asdfgh')
        {'e': 5, 'f': 6, 'z': 8}
        >>> hd.get(['b', 'd', 'e'], 'asdfgh')
        5
        >>> hd.get(['b', 'd', 'f'], 'asdfgh')
        6
        >>> hd.get(['b', 'd', 'z'], 'asdfgh')
        8
        >>> hd.get(['b', 'd', 'a'], 'asdfgh')
        1
        >>> hd.get(['b', 'd', 'c'], 'asdfgh')
        3
        >>> hd.get(['a', 'c'], 'asdfgh')
        'asdfgh'
        >>> hd.get(['z', 'f'], 'asdfgh')
        'asdfgh'
        >>> hd.get(['q', 'f'], 'asdfgh')
        'asdfgh'
        """
        if isinstance(key, list): 
            return self.__getitem__(key) if self.__contains__(key) else default
        return super().get(key, default)
    
    def compare_except(self, 
            other: dict, 
            *exceptions: list[Hashable]|Hashable, 
            symmetric=True,
            _addr: list[Hashable]=None
        ):
        """A modified equality/subset check, which can be made to ignore 
        certain keys, given in the `*exceptions` params

        Should be replaced with set-like symmetric (`^`, implemented in 
        `__xor__`) and asymmetric (`-`, implemented as `__sub__`): if the 
        relevantdifference only includes keys in `*exceptions`, the check is 
        passed.

        Parameters
        ----------
            other : dict
                The other to which to compare
            *exceptions : Hashable or list[Hashable]
                each of these star-args is either a `Hashable` (a valid `dict` 
                key) or a `list` of `Hashable`s (the address of a value in a
                nested hierarchy of `dict`s). If an `exception` appears as a 
                key in one or both `dict`s, it will be ignored in the 
                comparison. 
            symmetric : bool
                If `False`, the non-exception keys in `self` will be checked to
                see if they exist and map to an equal value in `other`. If 
                `True` the non-exception keys in `self` will be checked to see
                if they exist and map to an equal value in `other`, AND vice
                versa. If no `*exceptions` are passed, and `symmetric` is 
                `True`, `compare_except` checks for equality. If no 
                `*exceptions` are passed and `symmetric` is `False`, 
                `compare_except` checks if `self` is a subset of `other`.
            _addr : list[Hashable]
                Used in recursive calls: if a non-excepted key exists and has a 
                `dict` value in in both `self` and `other`, the child dicts must 
                also be compared with `compare_except`. However, the 
                `*exceptions` are addresses relative to the `dict` at which the
                initial call was made: therefore, to know whether a key in the
                child `dict` is excepted, the address of the child relative to
                the dict the original `compare_except` call was made on must be
                provided to the child call.

        Returns
        -------
            bool

        >>> hd1 = HierarchicalDict({'a': 1, 'z': 7, 'b': {'c': 3, 'd': {'e': 5, 'f': 6, 'z': 8}}})
        >>> hd2 = HierarchicalDict({'a': 2, 'z': 7, 'b': {'c': 3, 'd': {'e': 5, 'f': 6, 'z': 8}}})
        >>> hd3 = HierarchicalDict({'a': 1, 'z': -1, 'b': {'c': 3, 'd': {'e': 5, 'f': 6, 'z': 8}}})
        >>> hd4 = HierarchicalDict({'a': 1, 'z': 7, 'b': {'c': 3, 'd': {'e': 3, 'f': 6, 'z': 8}}})
        >>> hd5 = HierarchicalDict({'a': 1, 'z': 7, 'b': {'c': 3, 'd': {'e': 5, 'f': 6, 'z': -1}}})
        >>> hd6 = HierarchicalDict({'a': 1, 'b': {'c': 3, 'd': {'e': 5, 'f': 6, 'z': 8}}})
        >>> hd7 = HierarchicalDict({'a': 1, 'z': 7, 'b': {'c': 3, 'd': {'e': 5, 'z': 8}}})
        >>> assert hd1.compare_except(hd1)
        >>> assert not hd1.compare_except(hd2)
        >>> assert hd1.compare_except(hd2, ['a'])
        >>> assert hd1.compare_except(hd3, 'z')
        >>> assert not hd1.compare_except(hd3, 'a')
        >>> assert not hd1.compare_except(hd2, ['z'])
        >>> assert not hd2.compare_except(hd3, ['a'])
        >>> assert not hd2.compare_except(hd3, ['z'])
        >>> assert hd2.compare_except(hd3, ['a'], 'z')
        >>> assert hd2.compare_except(hd3, 'a', 'z')
        >>> assert not hd1.compare_except(hd4)
        >>> assert not hd1.compare_except(hd5)
        >>> assert hd1.compare_except(hd4, ['b', 'd', 'e'])
        >>> assert not hd1.compare_except(hd5, ['b', 'd', 'e'])
        >>> assert not hd1.compare_except(hd4, ['b', 'd', 'z'])
        >>> assert hd1.compare_except(hd5, ['b', 'd', 'z'])
        >>> assert not hd4.compare_except(hd5)
        >>> assert not hd4.compare_except(hd5, ['b', 'd', 'e'])
        >>> assert not hd4.compare_except(hd5, ['b', 'd', 'z'])
        >>> assert hd4.compare_except(hd5, ['b', 'd', 'e'], ['b', 'd', 'z'])
        >>> assert not hd1.compare_except(hd6)
        >>> assert not hd1.compare_except(hd7)
        >>> assert not hd1.compare_except(hd6, symmetric=False)
        >>> assert not hd1.compare_except(hd7, symmetric=False)
        >>> assert hd6.compare_except(hd1, symmetric=False)
        >>> assert hd7.compare_except(hd1, symmetric=False)
        >>> assert hd1.compare_except(hd6, 'z')
        >>> assert hd1.compare_except(hd7, ['b', 'd', 'f'])
        """
        # The default value for _addr is None, but what is needed is an empty
        # list; however, setting the default directly to be [] creates the 
        # possibility that it might be changed during runtime, which creates
        # bugs
        _addr = _addr if _addr is not None else []
        # Single keys or key-lists can be passed as exceptions, but for 
        # consistent behaviour it is simpler if all exceptions are key-lists,
        # so `bare` keys are converted into singleton key-lists 
        exceptions = [(x if isinstance(x, list) else [x]) for x in exceptions]
        # A HierarchicalDict can only be equal to another dict
        if not isinstance(other, dict):
            return False
        # This loop checks that all keys in self exist in other and map to the 
        # same value ...
        for k in self:
            # ... as long as it's not an exception, that is
            if (_addr + [k]) not in exceptions:
                # if a key in self is not in other, False
                if k not in other:
                    return False
                # Recursively check any values in self that are also
                # HierarchicalDicts, appending the key to _addr so that 
                # exceptions that are multi-key key-lists can be correctly
                # matched. Note, there is no need to `symmetric` to be True
                # here, even if it is True in the root call. This is because 
                if isinstance(self[k], self.__class__):
                    if not self[k].compare_except(other[k], *exceptions, _addr=_addr+[k], symmetric=False):
                        return False
                # Also False if the key exists in both but maps to different 
                # values
                elif self[k] != other[k]:
                    return False
        # If symmetric, do the same comparison with self and other switched 
        if symmetric:
            return self.__class__(
                other
            ).compare_except(
                self, 
                *exceptions, 
                _addr=_addr, 
                symmetric=False
            )
        # And if none of that stuff returns False ...
        return True
    
    def simplify(self):
        """Child `dicts` in a `HierarchicalDict` hierarchy can inherit key-value 
        pairs recursively from their parents. It is therefore possible for a
        `HierarchicalDict` hierarchy to have some redundant pairs, if a child
        `dict` explicitly contains a pair it already inherits, or two or more 
        children on separate branches share a pair, which could instead be held
        just once at their closest common ancestor node. `simplify` removes such
        redundancies.

        If there is more than one instance of the same key which can be 
        consolidated onto the same ancestor node, but there is a conflict over
        the value it maps to, the majority value will be chosen. If there is a
        draw, the sequentially first value will be chosen.

        The hierarchy is traversed depth-first and bottom up, which affects the 
        way key collisions are handled: if the initial hierarchy contains three
        keys `'k'`, two mapping to `42` and three mapping to `666`, but the 
        lowest common ancestor of the `666`s is at the same level or lower than
        the `42`s, the `666`s will be collapsed to a single occurence before the 
        decision is made as to whether `42` or `666` takes priority, meaning `42`
        may end up being favoured.

        This method takes no arguments, modifies the `dict` in place, and returns
        `None`.

        Returns
        -------
            None

        >>> HD = HierarchicalDict
        >>> HD().simplify()
        {}
        >>> HD({'a': {'c': 3}, 'b': {'c': 3}}).simplify()
        {'a': {}, 'b': {}, 'c': 3}
        >>> HD({'z': {'c': 4}, 'a': {'c': 3}, 'b': {'c': 3}}).simplify()
        {'z': {'c': 4}, 'a': {}, 'b': {}, 'c': 3}
        >>> HD({'y': {'c': 4}, 'z': {'c': 4}, 'a': {'c': 3}, 'b': {'c': 3}}).simplify()
        {'y': {}, 'z': {}, 'a': {'c': 3}, 'b': {'c': 3}, 'c': 4}
        >>> HD({'a': {'c': 3}, 'b': {'c': 3}, 'c': 3}).simplify()
        {'a': {}, 'b': {}, 'c': 3}
        >>> HD({'a': {'c': 3}, 'b': {'c': 3}, 'c': 2}).simplify()
        {'a': {'c': 3}, 'b': {'c': 3}, 'c': 2}
        >>> HD({'d': {'a': {'c': 3}, 'b': {'c': 3}}, 'c': 2}).simplify()
        {'d': {'a': {}, 'b': {}, 'c': 3}, 'c': 2}
        >>> HD({
        ...     'a': {'b': {'k': 42}, 'c': {'k': 42}}, 
        ...     'd': {'e': {'k': 42}, 'f': {'k': 42}}, 
        ...     'g': {'h': {'k': 42}, 'i': {'k': 42}}, 
        ...     'z': {'y': {'k': 666}, 'x': {'k': 666}, 'w': {'k': 666}, 'v': {'k': 666}}, 
        ...     'u': {'t': {'k': 666}, 's': {'k': 666}, 'r': {'k': 666}, 'q': {'k': 666}}
        ... }).simplify().prettify(tab=4, prnt=True)
        {
            'a': {
                'b': {},
                'c': {}
            },
            'd': {
                'e': {},
                'f': {}
            },
            'g': {
                'h': {},
                'i': {}
            },
            'z': {
                'y': {},
                'x': {},
                'w': {},
                'v': {},
                'k': 666
            },
            'u': {
                't': {},
                's': {},
                'r': {},
                'q': {},
                'k': 666
            },
            'k': 42
        }
        """
        # So that downstream redundancies are consolidated first, we begin with 
        # a recursive call on all dict values in self. This ensures depth-first
        # bottom-up traversal. 
        for v in self.values():
            if isinstance(v, self.__class__):
                v.simplify()
        # `all_items` is like the standard `dict.items`, but it covers all keys
        # and key-lists of the dict, with the exception of those that are valid
        # only due to inheritance. Here, we unroll all of these into a list
        # of tuples containing numerical indices, keys/key-lists, and values.
        # The inclusion of indices allows us to search all unique non-ordered
        # pairs, looking for key(-list) and value matches 
        all = [(i, k, v) for i, (k, v) in enumerate(self.all_items())]
        # Record all hits here
        hits = {}
        # But also, if hits are found, we need to gather they unique values:
        # however, these can't be recorded in a set, as they are not guaranteed
        # to be Hashable: therefore, a set is made of the vals cast to strings,
        # and a separate dict is needed to map those strings back to the 
        # objects. 
        val_strs = {}
        # Search the upper triangle of the pairwise matrix of keylist-value pairs
        for i, k1, v1 in all:
            for _, k2, v2 in all[i+1:]:
                # For any keylist-value pair, the last item of the keylist and 
                # the value form a key-value pair in the downstream dict; if 
                # these match, a hit has been found 
                if k1[-1]==k2[-1]:
                    # We need a dict to represent the hits for this key - there
                    # may be key collisions, so we need a record of what keys
                    # map to which values at which addresses. If there isn't 
                    # already a dict recording previously found hits for this 
                    # key, create an empty dict 
                    kx_hits = hits.get(k1[-1], {})
                    # for each side of the hit, we need the key-list address
                    # (converted to tuple, for hashability), and the value 
                    # (converted to str, for hashability, but also the 
                    # original) 
                    for k, vstr, v in ((tuple(k1), str(v1), v1), (tuple(k2), str(v2), v2)):
                        # in the dict of values with the current key, we use the
                        # stringified value as the key (we need to be keeping 
                        # track of what values are mapped to the key, and, if
                        # there is more than one, how many instances of each
                        # exist: this is why we needed a hashable representation 
                        # for this, thus the stringification), and a set 
                        # containing the addresses, converted to tuples (tuples
                        # used again for hashability; a set was chosen so as not
                        # to double-count any addresses that participate in more
                        # than one hit: which will happen if some key-value pair 
                        # appears more than twice). The line below gets the 
                        # existing set if one exists, or else an empty set, and 
                        # unions it with a singleton set containing the 
                        # key-tuple: this set is then the new value assigned to
                        # the value-string
                        kx_hits[vstr] = kx_hits.get(vstr, set()) | {k}
                        # We also record a value-string to actual value mapping,
                        # so the value proper can be placed in a mapping with
                        # the key in the lowest common ancestor node, if there
                        # is no value collision, or this value is chosen in 
                        # collision-resolution 
                        val_strs[vstr] = v
                    # ...and then the dict of value-strings to key-tuple-sets
                    # is mapped to the key in the hits-dict.
                    hits[k1[-1]] = kx_hits
        # Having found all the hits, now we simplify, and resolve collisions if 
        # needed
        # `hit` is the key, `vals_2_addrs` is a dict mapping string 
        # representations of values to sets of tuplified key-lists
        for hit, vals_2_addrs in hits.items():
            # we need placeholders for the winning value, which will be 
            # consolidated up the hierarchy, and the addresses where the value
            # currently exists, as the duplicate copies of the key-value pair
            # need to be deleted 
            val_best = None
            addrs_best = None
            # If there is a collision, the key that appears at the greatest
            # number of addresses is consolidated. `max_len` is used to find
            # which address-set contains most key-tuples. This is initialised
            # to 1 rather than zero so collisions a key appears in different
            # places, always with a different value, are simply ignored: in that
            # case, there's nothing to consolidate 
            max_len = 1
            # loop through value-strings and sets of key-tuple addresses
            for val_str, addrs in vals_2_addrs.items():
                # if the size of the current set is the largest, assign this 
                # size to `new_len` with the walrus-operator ...
                if max_len < (new_len := len(addrs)):
                    # Then update max_len with the new size, val_best with
                    # the value corresponding to the value-string, and 
                    # addrs_best with the current address-set   
                    max_len = new_len
                    val_best = val_strs[val_str]
                    addrs_best = addrs
            # The consolidation occurs if `addrs_best` is not None (Nones occur
            # if the key appears multiply, but always with different values) and
            # there isn't already a conflicting value of the key in the current 
            # dict (the second conjunct evaluates True if `hit` is not in `self`
            # or it is, but the value is the same as `val_best`)
            if addrs_best and self.get(hit, val_best)==val_best:
                # If so, we can go through all the places where the key maps to 
                # val_best, and delete them
                for addr in addrs_best:
                    try:
                        del self[list(addr)]
                    except KeyError:
                        # If the key is already gone, restart the simplification
                        return self.simplify()
                # Now, the consolidated key-value pair can be placed in self
                self[hit] = val_best
        # return self to allow method chaining
        return self

    def all_items(self, _addr: Hashable|list[Hashable]=None): 
        """Generator similar to dict.items(), but instead of yielding key-value
        pairs, it yield keylist-value pairs. Specifically, it iterates through 
        all the addresses in the dict where an actual value is located, but NOT
        those which are valid addresses because of key-value pair inheritance

        Parameters
        ----------
            _addr : Hashable or list[Hashable]
                In order to iterate through child dicts, the generator must be
                called recursively: but these recursive calls need to return the
                address of the downstream item relative to the top-level 
                generator: the recursive call therfore needs to be passed the 
                relative address of the downstream dict

        Yields
        ------
            tuple containing a list of Hashables (the key-list) and Any (the 
            value)

        >>> HD = HierarchicalDict
        >>> def print_all_items(hd):
        ...     for k_l, v in hd.all_items():
        ...         print(k_l, '::', v)
        >>> print_all_items(HD())
        >>> print_all_items(HD({'a': 1, 'b': 2, 'c': 3}))
        ['a'] :: 1
        ['b'] :: 2
        ['c'] :: 3
        >>> print_all_items(HD({
        ...     'x': {'a': 1, 'b': 2, 'c': 3},
        ...     'y': {'d': 4, 'e': 5, 'f': 6},
        ...     'z': {
        ...         'gub': 7, 
        ...         ('h', 8, True): 9, 
        ...         'i': {'j': 10, 'k': {'l': 11, 'm': 12}}
        ...     }
        ... }))
        ['x'] :: {'a': 1, 'b': 2, 'c': 3}
        ['x', 'a'] :: 1
        ['x', 'b'] :: 2
        ['x', 'c'] :: 3
        ['y'] :: {'d': 4, 'e': 5, 'f': 6}
        ['y', 'd'] :: 4
        ['y', 'e'] :: 5
        ['y', 'f'] :: 6
        ['z'] :: {'gub': 7, ('h', 8, True): 9, 'i': {'j': 10, 'k': {'l': 11, 'm': 12}}}
        ['z', 'gub'] :: 7
        ['z', ('h', 8, True)] :: 9
        ['z', 'i'] :: {'j': 10, 'k': {'l': 11, 'm': 12}}
        ['z', 'i', 'j'] :: 10
        ['z', 'i', 'k'] :: {'l': 11, 'm': 12}
        ['z', 'i', 'k', 'l'] :: 11
        ['z', 'i', 'k', 'm'] :: 12
        """
        # First, convert _addr into a consistent format, as a list of Hashables 
        _addr = (
            list(_addr) 
            if isinstance(_addr, Sequence) and not isinstance(_addr, str) 
            else [_addr,]
            if _addr 
            else []
        )
        # In a recursive call, the first item yielded is the dict itself, with
        # an emptry address. Whereas the root call skips straight to iterating
        # over self.items()
        if _addr:
            #print('xxx', addr)
            yield [], self
        # Iterate over the key-value pairs of the current dict, but use a 
        # recursive call if the value is a dict 
        for k, v in self.items():
            if isinstance(v, Mapping):
                it = self.__class__(v).all_items(_addr+[k,])
                # Yield each of the items in the child dict, with the child's
                # key prepended to the key-list
                k_v = next(it, None)
                while k_v is not None:
                    yield [k,]+([k_v[0],] if isinstance(k_v[0], str) else k_v[0]), k_v[1]
                    k_v = next(it, None)
            else: 
                # If the value is not a dict, yield the key-list and value
                yield [k,], v

    def prettify(self, indent: int=0, tab: int=2, prnt: bool=False) -> str|None:
        """Pretty printer for nested dicts.

        Parameters
        ----------
            indent : int (default = 0)
                prepends `indent` many tabs to each line of output
            tab : int (default = 2)
                number of spaces per tab
            prnt : bool (default = False)
                If `True`, the output is printed, and if `False`, it is returned

        Returns
        -------
            None or str

        >>> HD = HierarchicalDict
        >>> print(HD().prettify())
        {}
        >>> print(HD({'a': 1, 'b': 2, 'c': 3}).prettify())
        {
          'a': 1,
          'b': 2,
          'c': 3
        }
        >>> print(HD({'x': {'a': 1, 'b': 2, 'c': 3},
        ...     'y': {'d': 4, 'e': 5, 'f': 6},
        ...     'z': {'gub': 7, ('h', 8, True): 9, 'i': {'j': 10, 'k': {'l': 11, 'm': 12}}
        ... }}).prettify(tab=4))
        {
            'x': {
                'a': 1,
                'b': 2,
                'c': 3
            },
            'y': {
                'd': 4,
                'e': 5,
                'f': 6
            },
            'z': {
                'gub': 7,
                ('h', 8, True): 9,
                'i': {
                    'j': 10,
                    'k': {
                        'l': 11,
                        'm': 12
                    }
                }
            }
        }
        """
        # Empty dicts are just {}
        if len(self) == 0:
            return '{}'
        # Create string building blocks for tabs
        t = " "*tab
        # `s` is the output string, which will have content appended as the 
        # method runs. Opening curly bracket & newline
        s = "{\n" 
        # Increment indent for dict contents
        indent += 1
        # iterate through indices, keys, and values. Index needed so as not to
        # add a comma after the last item
        for i, (k, v) in enumerate(self.items()):
            # append tabs, key, semicolon, space
            s += (t*indent) + f"{repr(k)}: "
            # If value is a HierarchicalDict, make its pretty-string and append
            if isinstance(v, HierarchicalDict):
                s += v.prettify(indent, tab)
            else:
                # Otherwise, just use its repr
                s += repr(v)
            # add a comma after all non-last items
            if i < len(self)-1:
                s += ','
            # and a newline after each item
            s += "\n"
        # The dict content is done, so decrement the indent again
        indent -= 1
        # close the opening curly bracket, with tabs
        s += (t*indent) + "}"
        # return or print
        if prnt:
            print(s)
        else:
            return s
        
    def update(self, other: dict):
        """
        """
        for k, v in other.items():
            if isinstance(v, dict) and k in self and isinstance(self[k], dict):
                self[k].update(v)
            else:
                self[k] = v
        
    
def main():
    import doctest
    doctest.testmod()

if __name__ == '__main__':
    main()