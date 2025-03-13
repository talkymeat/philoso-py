#!/usr/bin/env python3
"""Script for generating experiments.txt"""
import os
from glob import glob

def main(start=None, end=None):
    # The home dir on the node's scratch disk
    USER = os.getenv('USER')
    # This may need changing to e.g. /disk/scratch_fast depending on the cluster
    SCRATCH_DISK = '/disk/scratch'  
    SCRATCH_HOME = f'{SCRATCH_DISK}/{USER}/philoso-py'

    list_jsons = sorted(glob('model_json/*'))

    with open("experiment.txt", "w") as expts_file:
        for json_f in list_jsons[slice(start, end)]:
            print(
                f'python philoso_py {json_f} -o {SCRATCH_HOME}',
                file=expts_file
            )

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(prog='make-experiments')
    parser.add_argument('range', nargs='?')
    args = parser.parse_args()
    s, e = None, None
    if args.range is not None:
        range_ = args.range.split('-')
        if len(range_) != 2:
            print('range is incorrectly formatted: should be `$start-$end`, `$start-`, or `-$end`')
        else:
            if range_[0] != '':
                s = int(range_[0])
            if range_[1] != '':
                e = int(range_[1])
    main(start=s, end=e)
