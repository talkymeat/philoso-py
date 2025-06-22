from pathlib import Path
import re
import os
from datetime import datetime
from glob import glob

def count_days(path: str):
    return len(glob(str(Path(path) / 'a_0' / 'days')+os.sep+'*'))//4

def folders(path: str):
    print('beep')
    return [check_sub_path(dir) for dir in sorted(glob(path+os.sep+'*')) if os.path.isdir(dir)]

def report(path: str):
    return {dir.split(os.sep)[-1]: count_days(dir) for dir in folders(path)}

def show(report_: dict):
    maxlen = max([len(k) for k in report_.keys()])
    for k, v in report_.items():
        print(f'{k: <{maxlen}} | {"#"*v:-<{100}} {v}')

def check_sub_path(f: str):
    print(f)
    idchars = f.split(os.sep)[-1]
    print(idchars)
    subf = glob(f+os.sep+'*')
    print(subf, 'subf')
    if len(subf)==1 and subf[0].split(os.sep)[-1]==idchars:
        print('fix_to', f + os.sep + idchars)
        return subf
    print('ok', f)
    return f


def ag_folders(f: str):
    return [check_sub_path(f_)+os.sep+'gp_out' for f_ in folders(f) if f_.split(os.sep)[-1].startswith('a_')]

def list_flatten(ls: list[list]):
    new_ls = []
    for it in ls:
        if isinstance(it, list):
            new_ls += list_flatten(it)
        else: 
            new_ls.append(it)
    return new_ls

def max_t(fname: str) -> int:
    print('max_t', fname)
    ls = list_flatten([folders(f) for f in ag_folders(fname)])
    print(ls)
    ls = [f_.split(os.sep)[-1].split('_')[-1] for f_ in list_flatten([folders(f) for f in ls])]
    print(ls)
    ls = [t for t in ls if re.match(r't[0-9]+', t)]
    print(ls)
    vals = [1+int(t[1:]) for t in ls]
    print(vals, ';', max(vals) if vals else 0)
    return max(vals) if vals else 0

def detailed_report(root: str):
    return {f.split(os.sep)[-1]: max_t(f) for f in folders(root)}

def show_detailed(report_: dict):
    maxlen = max([len(k) for k in report_.keys()])
    print(datetime.now())
    for k, v in report_.items():
        try:
            print(f'{k: <{maxlen}} | {"#"*(v//100):-<{100}} <hundreds>\n{" "*maxlen} | {"="*(v%100):-<{100}} <units> total: {v}')
        except TypeError as e:
            print(k, v, type(v))
            raise e


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(prog='prog_check')
    parser.add_argument('root', default='')
    args = parser.parse_args()
    show_detailed(detailed_report(str(Path(args.root)/'philoso-py'/'output')))
