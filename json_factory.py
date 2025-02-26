
import json
from functools import reduce
from copy import deepcopy
from hd import HierarchicalDict as HD
from collections.abc import Container
from icecream import ic
from string import ascii_lowercase
from math import log, ceil
# ic.disable()

class IDCoder:
    def __init__(self, n_chars: int, chars=ascii_lowercase):
        self.n = n_chars
        self.chars = chars
        self.i = 0

    def __iter__ (self):
        return self

    def __next__ (self):
        if(self.i >= len(self.chars)**self.n):
            raise StopIteration()
        id_ = "".join([(self.chars)[(self.i//(len(self.chars)**j))%(len(self.chars))] for j in range(self.n)])
        self.i += 1
        return id_[-1::-1]

def make_jsons(alts: dict, cartesians: Container[str], prefix: str|IDCoder):
    jsons = [HD(deepcopy(alts['base']))]
    for k, sub_dics in alts.items():
        if k=='base':
            continue
        elif k in cartesians:
            new_jsons = []
            for sub_dic in sub_dics:
                for json_ in jsons:
                    json_copy = HD(deepcopy(json_))
                    json_copy.update(sub_dic)
                    new_jsons.append(json_copy)
            jsons = new_jsons
        else:
            for json_ in jsons:
                json_.update(sub_dics[0])
    for j in jsons:
        pref = prefix if isinstance(prefix, str) else next(prefix)
        j['out_dir'] = j['out_dir'].replace('*', pref)
        j['output_prefix'] = j['output_prefix'].replace('*', pref)
        yield j, f'model_{pref}.json'
    # return jsons

def count_jsons(alts: dict, all_cartesians: Container[Container[str]]):
    sum_ = 0
    for rene in all_cartesians:
        prod = 1
        for val in rene:
            prod *= len(alts[val])
        sum_ += prod
    return ic(sum_)

def make_all_jsons(alts: dict, all_cartesians: Container[Container[str]], chars=ascii_lowercase):
    n_chars = len(chars)
    n_jsons = count_jsons(alts, all_cartesians)
    k_chars = ceil(log(n_jsons, n_chars))
    idc = IDCoder(k_chars)
    for cartesians in all_cartesians:
        for j, jfname in make_jsons(alts, cartesians, prefix=idc):
            yield j, jfname 

def save_all_jsons(alts: dict, all_cartesians: Container[Container[str]], chars=ascii_lowercase):
    for j, fn in make_all_jsons(alts, all_cartesians, chars=ascii_lowercase):
        json.dump(j, open(fn, 'w'), indent=4)

if __name__ == '__main__':
    alts = json.load(open('alternatives.json'))
    cartesians = [
        ['n_agents'], 
        ['policy_lr', 'value_lr'],
        ['day_len'], 
        ['volume'],
        ['publication_params', 'memory_dims'], 
        ['num_treebanks'], 
        ['network_class'], 
        ['short_term_mem_size'], 
        ['ppo_clip_val'], 
        ['def_fitness'], 
        ['weight_threshhold']
    ]
    save_all_jsons(alts, cartesians)
    # jsons = make_jsons(alts, ['publication_params'], 'x')
    # print(len(jsons))
    # for i, j in enumerate(jsons):
    #     if i > 0:
    #         print('-'*40)
    #     print(j.prettify())