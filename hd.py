from icecream import ic
from collections.abc import Sequence, Mapping

class HierarchicalDict(dict):
    def __init__(self, map: dict):
            super().__init__({k: (self.__class__(v) if isinstance(v, dict) else v) for k, v in map.items()})

    def __getitem__(self, key):
        try:
            if not isinstance(key, list):
                return super().__getitem__(key)
            else:
                if len(key)==1:
                    return super().__getitem__(key[0])
                elif not key:
                    raise AttributeError(
                        'An empty list is an invalid key for ' +
                        'HierarchicalDict: Provide a Hashable object ' +
                        'or a list of one or more Hashables.'
                    )
                elif super().__contains__(key[0]):
                    it = super().__getitem__(key[0])
                    if isinstance(it, dict):
                        return it.__getitem__(key[1:])
                    raise KeyError(
                        'When presenting a list as a HierarchicalDict' +
                        ' key, the list elements ' +
                        'must map to a sequence of nested dicts, with' +
                        ' the optional exception of the last'
                    )
        except KeyError as e:
            if isinstance(key, list):
                return super().__getitem__(key[-1])
            raise e
        
    def __delitem__(self, key):
        # print('xxxx', key)
        if isinstance(key, list):
            if len(key)==1:
                key = key[0]
            else:
                self[key[0]].__delitem__(key[1:])
                return
        return super().__delitem__(key)

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, list):
            key = [key]
        if len(key)==1:
            return super().__contains__(key[0])
        elif not key:
            raise IndexError(
                'An empty list is an invalid index for ' +
                'HierarchicalDict: Provide a Hashable object ' +
                'or a list of one or more Hashables.'
            )
        elif super().__contains__(key[-1]):
            return True
        elif super().__contains__(key[0]) and isinstance(self.__getitem__(key[0]), dict):
            return self.__getitem__(key[0]).__contains__(key[1:])
        else:
            raise KeyError(key)
        
    def get(self, key, default=None):
        if isinstance(key, list): 
            return self.__getitem__(key) if self.__contains__(key) else default
        return super().get(key, default)
    
    def eq_except(self, other: dict, *exceptions: list[str]|str, addr: list[str]=None, _rev=True):
        addr = addr if addr is not None else []
        exceptions = [(x if isinstance(x, list) else [x]) for x in exceptions]
        if not isinstance(other, dict):
            return False
        for k in self:
            if (addr + [k]) not in exceptions:
                if k not in other:
                    return False
                if isinstance(self[k], self.__class__):
                    if not self[k].eq_except(other[k], *exceptions):
                        ic(self[k])
                        return False
                elif self[k] != other[k]:
                    ic(addr)
                    ic(4, self[k], other[k], k, exceptions)
                    return False
        if _rev:
            return self.__class__(other).eq_except(self, *exceptions, addr=addr, _rev=False)
        return True
    
    def simplify(self):
        for v in self.values():
            if isinstance(v, self.__class__):
                v.simplify()
        all = [(i, k, v) for i, (k, v) in enumerate(self.all_items())]
        hits = {}
        val_strs = {}
        for i, k1, v1 in all:
            for _, k2, v2 in all[i+1:]:
                if k1[-1]==k2[-1]:
                    kx_hits = hits.get(k1[-1], {})
                    for k, vstr, v in ((k1, str(v1), v1), (k2, str(v2), v2)):
                        kx_hits[vstr] = kx_hits.get(vstr, set()) | {k}
                        val_strs[vstr] = v
                    hits[k1[-1]] = kx_hits
        for hit, vals_2_addrs in hits.items():
            val_best = None
            addrs_best = None
            max_len = 1
            for val_str, addrs in vals_2_addrs.items():
                if max_len < (new_len := len(addrs)):
                    max_len = new_len
                    val_best = val_strs[val_str]
                    addrs_best = addrs
            if addrs_best and self.get(hit, val_best)==val_best:
                for addr in addrs_best:
                    # print('-----')
                    try:
                        del self[list(addr)]
                    except KeyError:
                        return self.simplify()
                self[hit] = val_best
        return self

    def all_items(self, addr=None):
        addr = (
            tuple(addr) 
            if isinstance(addr, Sequence) and not isinstance(addr, str) 
            else (addr,) 
            if addr 
            else ()
        )
        if addr:
            yield (), self
        for k, v in self.items():
            if isinstance(v, Mapping):
                it = self.__class__(v).all_items(addr+(k,))
                k_v = next(it, None)
                while k_v is not None:
                    yield (k,)+((k_v[0],) if isinstance(k_v[0], str) else k_v[0]), k_v[1]
                    k_v = next(it, None)
            else: 
                yield (k,), v

    def pp(self, indent=0, tab=2):
        t = " "*tab
        s = "{\n" 
        indent += 1
        for k, v in self.items():
            s += (t*indent) + f"{k}: "
            if isinstance(v, Mapping):
                s += v.pp(indent, tab)
            else:
                s += str(v)
            s += "\n"
        indent -= 1
        s += (t*indent) + "}"
        if indent:
            return s
        else: 
            print(s)

def _tup(x):
    return tuple(x) if isinstance(x, Sequence) and not isinstance(x, str) else (x,)