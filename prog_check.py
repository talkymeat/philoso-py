import os
from glob import glob
from pathlib import Path

def count_days(path: str):
    return len(glob(str(Path(path) / 'a_0' / 'days')+os.sep+'*'))//4

def folders(path: str):
    return [dir for dir in sorted(glob(path+os.sep+'*')) if os.path.isdir(dir)]

def report(path: str):
    return {dir.split(os.sep)[-1]: count_days(dir) for dir in folders(path)}

def show(report_: dict):
    maxlen = max([len(k) for k in report_.keys()])
    for k, v in report_.items():
        print(f'{k: <{maxlen}} | {"#"*v:-<{100}} {v}')

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(prog='prog_check')
    parser.add_argument('root', default='')
    args = parser.parse_args()
    show(report(str(Path(args.root)/'philoso-py'/'output')))
