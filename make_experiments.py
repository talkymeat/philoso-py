#!/usr/bin/env python3
"""Script for generating experiments.txt"""
import os
from glob import glob

def main(start=None, end=None):
    # The home dir on the node's scratch disk
    USER = os.getenv('USER')
    # This may need changing to e.g. /disk/scratch_fast depending on the cluster
    # SCRATCH_DISK = '/disk/scratch'  
    SCRATCH_DISK = '/home' # womp womp scratch isn't working
    SCRATCH_HOME = f'{SCRATCH_DISK}/{USER}/philoso-py'

    list_jsons = sorted(glob('model_json/*'))

    with open("experiments.txt", "w") as expts_file:
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

# RUN WITH THIS COMMAND:
# sbatch --array=1-2%2 --time=0-06:00:00 --gres=gpu:1 --partition=PGR-Standard --mem 14000 --nodes=1 --output=/home/s0454279/philoso-py/output/logs/slurm-%A_%a.out --error=/home/s0454279/philoso-py/output/errors/slurm-%A_%a.out --cpus-per-task=1 run_models.sh experiments.txt