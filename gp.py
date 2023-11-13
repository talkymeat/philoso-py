from treebanks import TypeLabelledTreebank
from gp_trees import GPTerminal, GPNonTerminal
from gp_fitness import *
import pandas as pd
import numpy as np
from icecream import ic
from copy import copy
from typing import Sequence # Union, List, Callable, Mapping, 
from random import choices, choice
from observatories import *
from tree_factories import *
from utils import collect
from logtools import Stopwatch
# import pickle
from datetime import datetime
from time import time
import json
from string import ascii_lowercase as lcase

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
    def __init__(self,
            crossover_rate: float = 0.0,
            mutation_rate: float = 0.0,
            mutation_sd: float = 0.0,
            temp_coeff: float = 1.0,
            max_depth: int = 0,
            max_size: int = 0,
            seed_pop_node_max: int = None,
            seed_pop_tree_max: int = None,
            default_op = None,
            operators = None):
        super().__init__(default_op = default_op, operators = operators)
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
        self.clear()
        for i in range(pop):
            genfunc(*vars_, **kwargs)

    def _run_step(
            self, 
            scoreboard: GPScoreboard, 
            expt_outvals: list[str],
            n: int,
            steps: int,
            elitism: int,
            pop: int,
            record: dict[str, np.ndarray],
            grapher: Grapher
        )-> tuple[pd.DataFrame, dict]:
        # And allow for non-float roots
        old_gen = self.get_all_root_nodes()[float]
        record_means = ['penalty', 'hasnans', 'survive']
        best = scoreboard.score_trees(old_gen, except_for=expt_outvals)
        scoreboard.k_best(elitism)
        if n == steps-1:
            final_best = scoreboard.k_best(1, mark_4_del=False)[0]
        else:
            final_best = None
        for t in choices(old_gen, scoreboard['fitness'].fillna(0), k=pop-elitism):
            t.copy(gp_copy=True) # WUT XXX
        deathlist = filter(
            lambda t: t.metadata['to_delete'] == True, old_gen
        ) if elitism else old_gen
        for t in deathlist:
            t.delete()
        # TODO make functions for the following... (or even a class?)
        for k, v in best.items():
            if not k in record:
                record[k] = np.zeros(steps, dtype=float if k in record_means else type(v))
            if k in record_means:
                record.at[n, k] = scoreboard[k].mean()
            else:
                record.at[n, k] = v
        if 'time' not in record:
            record['time'] = np.zeros(steps)
        record.at[n,'time'] = self.sw()
        if grapher and n>0:
            self.update_grapher(grapher, record, n)
            if final_best:
                grapher.save(f'plots_1_{self.make_filename()}.png')
        return record, final_best
    
    def update_grapher(self, grapher, record, n):
        if n==1:
            grapher.plot_data(**record, n=n)
        else:
            grapher.set_data(**record, n=n)
    
    def make_filename(self) -> str:
        dt=datetime.now()
        dt_str = f"{dt.year}_{dt.month}_{dt.day}_{dt.hour}_{dt.minute}_{dt.second}"
        rnd_str = ''.join([choice(lcase) for _ in range(8)])
        return f"{dt_str}_{rnd_str}"

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
            ping_freq: int = 25,
            to_graph: Collection[str] = None
        ) -> tuple[dict[str, list], GPNonTerminal]:
        # make base filename for saving data. This means the filename will record
        # when the run started, not when the file was saved. It also means pickle
        # saves of the run data will overwrite previous pickles.
        fn = self.make_filename()
        # If a scoreboard is provided in kwargs, use it
        if fitness:
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
                if g!='time' and g not in scoreboard.def_outputs:
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
            if n==0 or not (n+1)%ping_freq:
                record.to_parquet(f'record_{fn}.parquet')
                scoreboard[[
                    col for col in scoreboard if col!='tree'
                ]].to_parquet(f'scoreboard_{fn}_{n}.parquet')
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
        record.to_parquet(f'record_{fn}.parquet')
        scoreboard[[
            col for col in scoreboard if col!='tree'
        ]].to_parquet(f'record_{fn}_{n}.parquet')
        jsonised = final_best.copy()
        jsonised['tree'] = f'{jsonised["tree"]}'
        with open(f'final_{fn}.json', 'w') as file:
            json.dump(jsonised, file)
        return record, final_best

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


        
