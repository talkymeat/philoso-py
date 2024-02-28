from treebanks import TypeLabelledTreebank
from gp_trees import GPTerminal, GPNonTerminal
from gp_fitness import *
import pandas as pd
import numpy as np
from icecream import ic
from copy import copy
from typing import Sequence # Union, List, Callable, Mapping, 
from observatories import *
from tree_factories import *
from utils import collect
from logtools import Stopwatch
# import pickle
from datetime import datetime
from time import time
from tree_iter import DepthFirstBottomUp as DFBU
import json
from string import ascii_lowercase as lcase
import os
from pathlib import Path

DEBUG = False

def _print(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)

def flatten(s: Sequence):
    t = type(s)
    new_s = t()
    for it in s:
        new_s += flatten(it) if isinstance(it, t) else t([it])
    return new_s

class GPTreebank(TypeLabelledTreebank):
    """This is the main class for running genetic programming"""
    def __init__(self,
            crossover_rate: float = 0.0,
            mutation_rate: float = 0.0,
            mutation_sd: float = 0.0,
            temp_coeff: float = 1.0,
            max_depth: int = 0,
            max_size: int = 0,
            seed_pop_node_max: int = None, # not used
            seed_pop_tree_max: int = None, # not used
            default_op = None,
            operators = None,
            seed: int|np.random.Generator|None = None,
            _dir: str = 'outputs'
        ):
        super().__init__(default_op = default_op, operators = operators)
        self._dir = Path(_dir)
        if not self._dir.is_dir():
            os.mkdir(self._dir)
        if isinstance(seed, np.random.Generator):
            self.np_random = seed
        else:
            self.np_random = np.random.Generator(np.random.PCG64(seed))
        self.crossover_rate = crossover_rate
        self.seed_pop_node_max = seed_pop_node_max
        self.seed_pop_tree_max = seed_pop_tree_max
        self.mutation_rate = mutation_rate
        self.mutation_sd = mutation_sd
        self.max_depth = max_depth
        self.max_size = max_size
        self.temp_coeff = temp_coeff
        self.T = GPTerminal
        self.N = GPNonTerminal
        self.sw = Stopwatch()

    def _generate_starting_sample(self, pop, genfunc=None, vars_=None, **kwargs):
        """Creates the initial population for a GP run"""
        self.clear()
        for _ in range(pop):
            genfunc(*vars_, **kwargs)

    def _run_step(
            self, 
            scoreboard: GPScoreboard, 
            expt_outvals: list[str],
            n: int,
            steps: int,
            elitism: int,
            pop: int,
            record: pd.DataFrame,
            grapher: Grapher = None
        )-> tuple[pd.DataFrame, dict]:
        """A single evolutionary step for GP"""
        # And allow for non-float roots
        # for name in ['dead_', 'garn_', 'score_', 'kbest_', 'mut8_', 'kill_', 'record_', '']:
        #     if f'{name}time' not in record:
        #         record[f'{name}time'] = np.zeros(steps)
        # record.at[n,'dead_time'] = self.sw()
        # record.at[n,'time'] += record.at[n,'dead_time']
        old_gen = self.get_all_root_nodes()[float].array()
        # print(type(old_gen), len(old_gen), old_gen.shape)
        # record.at[n,'garn_time'] = self.sw()
        # record.at[n,'time'] += record.at[n,'garn_time']
        record_means = ['penalty', 'hasnans', 'survive']
        best = scoreboard.score_trees(old_gen, except_for=expt_outvals)
        # record.at[n,'score_time'] = self.sw()
        # record.at[n,'time'] += record.at[n,'score_time']
        scoreboard.k_best(elitism)
        # record.at[n,'kbest_time'] = self.sw()
        # record.at[n,'time'] += record.at[n,'kbest_time']
        if n == steps-1:
            final_best = scoreboard.k_best(1, mark_4_del=False, tree_only=False)
            # print('-=-=-'*60)
            # print(type(final_best))
            # try: 
            #     print(final_best['tree'][0])
            # except KeyError as e:
            #     print('x'*80)
            #     print(final_best['tree'])
            #     raise e
            # for k, v in final_best.items():
            #     print(k)
            #     print('>>>>>>><<<<<<<')
            #     print(type(v))
            #     print(v)
            #     print('<<<<<<<>>>>>>>')
            # print(final_best)
            # print('FINAL BEST ' * 30)
        else:
            final_best = None
        scores = np.array(scoreboard['fitness'].fillna(0))
        sum_scores = np.sum(scores)
        i=0
        if sum_scores != 0:
            print('G', pop-elitism)
            if np.min(scores) < 0:
                scores -= np.min(scores)
                sum_scores = np.sum(scores)
            normed_scores = scores/sum_scores
            for t in self.np_random.choice(
                old_gen, 
                p=normed_scores, # scores/np.sum(scores), 
                size=pop-elitism
            ):
                if not i%10:
                    print(f'gggg-{i}')
                i += 1
                t.copy(gp_copy=True) # WUT XXX
        else:
            # print('+'*200)
            # print(scoreboard)
            # print('+'*200)
            print(scoreboard['fitness'].fillna(0).sum())
            print('H', pop-elitism)
            for t in self.np_random.choice(old_gen, size=pop-elitism):
                if not i%10:
                    print(f'hhhh-{i}')
                i += 1
                t.copy(gp_copy=True) # WUT XXX
        # record.at[n,'mut8_time'] = self.sw()
        # record.at[n,'time'] += record.at[n,'mut8_time']
        if final_best is not None:
            # print('@'*200)
            # print(final_best['tree'][0])
            # tour = DFBU(final_best['tree'][0])
            # for t in tour:
            #     print(t)
            #     if hasattr(t, 'metadata'):
            #         print(t.metadata)
            #     print('~~~~~'*40)
            final_best['tree'][0].metadata['to_delete'] = False
            # print('@'*200)
            # print(final_best['tree'][0].metadata)
            # print(elitism, 'hhhhh'*30)
        deathlist = filter(
            lambda t: t.metadata['to_delete'] == True, old_gen
        ) if elitism else [
            t for t in old_gen if id(t) != id(final_best['tree'][0])
        ] if final_best is not None else old_gen
        # if final_best is not None and final_best['tree'][0] in deathlist:
        #     print('WTF-'*30)
        for t in deathlist:
            t.delete()
        # if final_best is not None:
        #     print(-1, final_best['tree'][0])
        # record.at[n,'kill_time'] = self.sw()
        # record.at[n,'time'] += record.at[n,'kill_time']
        # TODO make functions for the following... (or even a class?)
        for k, v in best.items():
            if not k in record:
                record[k] = np.zeros(steps, dtype=float if k in record_means else type(v))
            if k in record_means:
                record.at[n, k] = scoreboard[k].mean()
            else:
                record.at[n, k] = v
        # record.at[n,'record_time'] = self.sw()
        # record.at[n,'time'] += record.at[n,'record_time']
        if grapher and n>0:
            self.update_grapher(grapher, record, n)
            if final_best is not None:
                grapher.save(self.make_filename('plots_1', 'png'))
        return record, final_best
    
    def update_grapher(self, grapher, record, n):
        if n==1:
            grapher.plot_data(**record, n=n)
        else:
            grapher.set_data(**record, n=n)
    
    def make_filename(self, name: str, ext: str, _dir=None) -> str:
        dt=datetime.now()
        fpath = Path(f"{name}_{dt.year}_{dt.month}_{dt.day}_{dt.hour}_{dt.minute}_{dt.second}.{ext}")
        if _dir:
            fpath = _dir / fpath
        elif _dir is None:
            fpath = self.data_dir / fpath
        return fpath

    def run(
            self,
            tree_factory: TreeFactory,
            observatory: Observatory,
            steps: int,
            pop: int, 
            dv: str=None, 
            fitness: GPScoreboard=None, 
            def_fitness: str = None,
            elitism: int = 0,
            temp_coeff: float = None,
            best_outvals: str|list[str] = None,
            expt_outvals: str|list[str] = None,
            ping_freq: int = 1,
            to_graph: Collection[str] = None
        ) -> tuple[dict[str, list], dict, GPNonTerminal]:
        """Sets up a genetic programming run and runs it for a specified number 
        of steps
        
        TODO: separate this into two steps - _setup and _run
        """
        # make base filename for saving data. This means the filename will record
        # when the run started, not when the file was saved. It also means pickle
        # saves of the run data will overwrite previous pickles.
        self.data_dir = Path(
            self._dir, 
            (''.join([self.np_random.choice(list(lcase)) for _ in range(8)]))
        )
        if not self.data_dir.is_dir():
            self.data_dir.mkdir()
        # If a scoreboard is provided in kwargs, use it
        if fitness is not None:
            scoreboard = fitness
        # Otherwise, configure one
        else:
            # Make a dict to use for kwargs. The temperature coefficient and
            # observatory are always going to be needed
            sb_kwargs = {
                "temp_coeff": temp_coeff if temp_coeff is not None else self.temp_coeff,
                "obs": observatory,
            }
            def_outputs = collect(best_outvals, list) if best_outvals else None
            for arg, val in (
                ('def_outputs', def_outputs), 
                ('dv', dv), 
                ('def_fitness', def_fitness)
            ):
                val = eval(arg)
                if val:
                    sb_kwargs[arg] = val 
            scoreboard = GPScoreboard(**sb_kwargs)
        if to_graph:
            for g in flatten(to_graph):
                if 'time' not in g and g not in scoreboard.def_outputs:
                    raise ValueError(
                        f"Grapher is instructed to graph {g}, which is not in the " +
                        "pipeline"
                    )
        grapher = Grapher(to_graph) if to_graph else None
        expt_outvals = collect(expt_outvals, list)
        record = pd.DataFrame()
        # First generate the initial population that will be evolved
        tree_factory.set_treebank(self)
        self._generate_starting_sample(pop, tree_factory, *observatory.ivs)
        self.sw()
        for n in range(steps):
            # Would be nice to replace this with a proper progress bar
            if not n%ping_freq:
                print(f'{n} generations')
            record, final_best = self._run_step(
                steps=steps,
                pop=pop, 
                scoreboard=scoreboard, 
                elitism=elitism,
                expt_outvals=expt_outvals,
                grapher=grapher,
                n=n, 
                record=record
            )
            # if final_best is not None:
            #     print(1, final_best['tree'][0])
            if n==0 or not (n+1)%ping_freq:
                record.to_parquet(self.make_filename('record', 'parquet'))
                scoreboard[[
                    col for col in scoreboard if col!='tree'
                ]].to_parquet(self.make_filename(f'scoreboard_{n}', 'parquet'))
            # if final_best is not None:
            #     print(2, final_best['tree'][0])
            #     pickle_jar = {
            #         'gp': self, 
            #         'tree_factory': tree_factory, 
            #         'observatory': observatory, 
            #         'steps': steps, 
            #         'pop': pop, 
            #         'dv': dv, 
            #         'fitness': fitness, 
            #         'def_fitness': def_fitness,
            #         'elitism': elitism, 
            #         'temp_coeff': temp_coeff, 
            #         'best_outvals': best_outvals, 
            #         'expt_outvals': expt_outvals, 
            #         'ping_freq': ping_freq, 
            #         'to_graph': to_graph, 
            #         'n': n+1, 
            #         'scoreboard': scoreboard, 
            #         'grapher': grapher, 
            #         'record': record, 
            #     }
            #     pickle.dump(pickle_jar, file=open(f'gp_run_{fn}.pickle', 'wb'))
        # print(3, final_best['tree'][0])
        record.to_parquet(self.make_filename(f'record', 'parquet'))
        # print(4, final_best['tree'][0])
        scoreboard[[
            col for col in scoreboard if col!='tree'
        ]].to_parquet(self.make_filename(f'record_{n}', 'parquet'))
        # print(5, final_best['tree'])
        # print(final_best['tree'][0])
        # print('--------')
        final_best = {k: list(v)[0] for k, v in final_best.items()}
        # print(6, final_best['tree'])
        # print(final_best['tree'].label)
        best_tree = final_best['tree'].copy_out()
        final_best['tree'] = f'{best_tree}'
        with open(self.make_filename('final', 'json'), 'w') as file:
            json.dump(final_best, file)
        return record, final_best, best_tree

# Things to catch and penalise: 
# RuntimeWarning: divide by zero encountered in scalar power
#  return args[0] ** args[1]
# RuntimeWarning: invalid value encountered in scalar multiply
#  return reduce(lambda a, b: a * b, (1,) + args) 
# RuntimeWarning: invalid value encountered in scalar add
#  return reduce(lambda a, b: a + b, (0,) + args)
# RuntimeWarning: overflow encountered in reduce
#  return umr_sum(a, axis, dtype, out, keepdims, initial, where) 

# def run_gp_from_pickle(fname: str):
#     from_pickle = pickle.load(open(fname, 'rb'))
#     gp = from_pickle['gp']

def main():
    import doctest
    doctest.testmod()

if __name__ == '__main__':
    main()


        
