from treebanks import TypeLabelledTreebank
from gp_trees import GPTerminal, GPNonTerminal
from mutators import MutatorMutator, random_mutator_factory
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
from tree_funcs import sum_all, unions_for_all, get_operators
from copy import copy


def ic_mut8rz(tree):
    ic(tree)
    if hasattr(tree, 'gp_operator'):
        ic(tree.gp_operator)
    if hasattr(tree, 'xo_operator'):
        ic(tree.xo_operator)
    if hasattr(tree, 'children'):
        for c in tree:
            ic_mut8rz(c)

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
            pop: int=1000000000,
            tree_factory: TreeFactory|None=None,
            observatory: Observatory|None=None,
            mutator_factories: list[Callable]|None=None,
            mutator_weights: list[float]|None=None,
            crossover_rate: float = 0.0,
            mutation_rate: float = 0.0,
            mutation_sd: float = 0.0,
            temp_coeff: float = 1.0,
            max_depth: int = 0,
            max_size: int = 0,
            episode_len: int = 0,
            # seed_pop_node_max: int = None, # not used
            # seed_pop_tree_max: int = None, # not used
            best_outvals: str|list[str]|None = None,
            expt_outvals: str|list[str]|None = None,
            default_op = None,
            operators = None,
            fitness: GPScoreboard|None=None, 
            seed: int|np.random.Generator|None = None,
            _dir: str = 'outputs',
            elitism: int = 0,
            dv: str|None=None, 
            def_fitness: str|None = None,
            ping_freq = 5
        ):
        super().__init__(default_op = default_op, operators = operators)
        if isinstance(seed, np.random.Generator):
            self.np_random = seed
        else:
            self.np_random = np.random.Generator(np.random.PCG64(seed))
        # self._dir = Path(_dir)
        # if not self._dir.is_dir():
        #     os.mkdir(self._dir)
        self.data_dir = _dir
        ###############################
        ## Sets up data recording - maybe should be separated / conditional
        ###############################
        
        ###############################
        self.crossover_rate = crossover_rate
        # self.seed_pop_node_max = seed_pop_node_max
        # self.seed_pop_tree_max = seed_pop_tree_max
        self.mutation_rate = mutation_rate
        self.mutation_sd = mutation_sd
        self.max_depth = max_depth
        self.max_size = max_size
        self.episode_len = episode_len
        self.temp_coeff = temp_coeff
        self.pop = pop
        if not self.pop:
            print(': my dude, the population is set to zero :' * 40)
        self.elitism = elitism
        self.ping_freq = ping_freq
        self.best = None
        self.gens_past = 0
        self.observatory = observatory
        self.T = GPTerminal #    << XXX move into super init call
        self.N = GPNonTerminal # << XXX also this
        self.sw = Stopwatch()
        self.record_means = ['penalty', 'hasnans', 'survive']
        # If a scoreboard is provided in kwargs, use it << but wait XXX, does the scoreboard come preloaded with an observatory?
        if fitness is not None:
            self.scoreboard = fitness
            # Otherwise, configure one
            # Make a dict to use for kwargs. The temperature coefficient and
            # observatory are always going to be needed
        else:
            sb_kwargs = {
                "temp_coeff": self.temp_coeff,
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
            self.scoreboard = GPScoreboard(**sb_kwargs)
        self.expt_outvals = collect(expt_outvals, list)
        self.record = pd.DataFrame()
        # Set up the tree factory to generate the initial population that will be evolved
        if tree_factory and tree_factory.treebank is not self:
            tree_factory.treebank = self
        self.tree_factory = tree_factory
        self.mutator_factories = (
            mutator_factories 
            if mutator_factories 
            else [random_mutator_factory]
        )
        self.mutator_weights = (
            mutator_weights 
            if mutator_weights 
            else [1.0/len(self._mut_fac)] * len(self._mut_fac)
        )

    @property
    def data_dir(self):
        return self._data_dir
    
    @data_dir.setter
    def data_dir(self, _dir):
        self._data_dir = Path(_dir)                            ##
        if not self._data_dir.is_dir():#
            self._data_dir.mkdir(parents=True, exist_ok=True)    ##


    @property
    def best(self):
        return self._best

    @best.setter
    def best(self, best_tree_with_data):
        if not best_tree_with_data:
            self._best = None
            return
        best_tree = best_tree_with_data['tree']
        metadata = {k: v for k, v in best_tree_with_data.items() if k != 'tree'}
        metadata = {
            **metadata,
            **{
                "crossover_rate": self.crossover_rate,
                "mutation_rate": self.mutation_rate,
                "mutation_sd": self.mutation_sd,
                "max_depth": self.max_depth,
                "max_size": self.max_size,
                "temp_coeff": self.temp_coeff,
                "pop": self.pop,
                "elitism": self.elitism
            },
            **self.observatory.obs_params,
            **self.tree_factory.tf_params,
            **self.scoreboard.sb_params
        }
        self._best = {'tree': best_tree, 'data': metadata}

    def does_nothing(self):
        """I just wanted a place to put a doctest for some properties
        below. Decorators screw with doctest.

        >>> from test_materials import MarkedDummyA, MarkedDummyB, MarkedDummyC
        >>> mfs = [MarkedDummyA, MarkedDummyB, MarkedDummyC]
        >>> gp = GPTreebank(pop=120, mutator_factories=mfs)
        a
        b
        c
        >>> gp.mutator_weights = [1,1,1]
        >>> gp._mutator_change_indices
        [0, 40, 80]
        >>> [str(mf) for mf in gp.mutator_factories]
        ['a', 'b', 'c']
        >>> gp.mutator_weights = [0,0,0]
        >>> [str(mf) for mf in gp.mutator_factories]
        ['a', 'b', 'c']
        >>> gp._mutator_change_indices
        [0, 40, 80]
        >>> gp.mutator_weights = [0,1,1]
        >>> [str(mf) for mf in gp.mutator_factories]
        ['b', 'c']
        >>> gp._mutator_change_indices
        [0, 60]
        >>> gp.mutator_weights = [2,0,1]
        >>> [str(mf) for mf in gp.mutator_factories]
        ['a', 'c']
        >>> gp._mutator_change_indices
        [0, 80]
        >>> gp.mutator_weights = [0,0.5362523,0]
        >>> [str(mf) for mf in gp.mutator_factories]
        ['b']
        >>> gp._mutator_change_indices
        [0]
        """
        pass

    @property
    def mutator_factories(self) -> list[MutatorMutator]:
        """`self._mut_fac` represents the complete set of available mutators:
        the property `self.mutator_factories` is the subset of these that is
        *not* assigned a weight of zero. Specifically, they are MutatorMutators
        that change out the mutators on every tree in `self`
        """
        return list(self._mut_fac[self._mfw > 0])
    
    @mutator_factories.setter
    def mutator_factories(self, factories: list|np.ndarray):
        """The mutator factories in `factories` are factory functions that generate 
        MutatorMutators (a callable class) set up to point to a Treebank - when 
        called, the switch around the mutators on all the trees in the Treebank."""
        self._mut_fac = np.array([f(self) for f in factories]) 
        # f(self) calls the factory function and makes the MutatorMutator


    @property
    def mutator_weights(self) -> np.ndarray:
        """`self._mfw` represents the complete set of available mutators weights:
        the property `self.mutator_weights` is the subset of these that is
        *not* assigned a weight of zero
        """
        return self._mfw[self._mfw > 0]
    
    @mutator_weights.setter # TEST ME
    def mutator_weights(self, weights) -> None:
        """Mutator weights need a bit of processing when set: 
        1) they must be normed to sum to one
        2) if any weight is less than zero, all weights must be shifted uniformly 
           so that zero is the minimum value
        3) In the edge case where all raw weights are zero, assign uniform weight
           to all mutators.
        """
        self._mfw = np.array(weights)
        if not self._mfw.any(): # handles edge case in (3)
            self._mfw = np.ones(self._mfw.shape, dtype=self._mfw.dtype)
        elif self._mfw.min() < 0: # handles edge case (2)
            self._mfw -= self._mfw.min()
        self._mfw = self._mfw/self._mfw.sum() # norms weights as per (1)
    
    @property
    def _mutator_change_indices(self):
        """During a GP run, if multiple mutation regimes are used (each 
        represented by a `MutatorFactory`), `mutator_weights` are used to
        determine at what point during a generation to switch mutators.
        The `_mutator_change_indices` determines the point in the seqence
        of generations within the run
        """
        if len(self.mutator_weights)==1:
            return [0]
        return [0] + list((
                self.mutator_weights*(
                    self.pop-self.elitism
                )
            ).astype(np.int64).cumsum()[:-1]
        )

    def run(
            self,
            ping_freq: int = None,
            to_graph: Collection[str] = None
        ) -> tuple[dict[str, list], dict, GPNonTerminal]:
        self.set_up_run(to_graph)
        return self.continue_run(ping_freq=ping_freq if ping_freq else self.ping_freq)
    
    def continue_run(self, ping_freq: int = None):
        best = self.run_episode(ping_freq=ping_freq if ping_freq else self.ping_freq)
        return self.process_run_output(best)
    
    def insert_trees(self, bonus_trees: Sequence[Tree]):
        #self.pop += len(bonus_trees) <<<== leave the treebank a bit overpopulated - it will correct on next gen. Test this XXX TODO
        # Ignore any that exceed max size 
        # ?? XXX TODO maybe penalise this?
        bonus_trees = [t for t in bonus_trees if t is not None]
        bonus_trees = [t for t in bonus_trees if t.size() <= self.max_size]
        bonus_trees = [t for t in bonus_trees if t.depth() <= self.max_depth]
        foreign_ops = unions_for_all(*[
            get_operators(t) for t in bonus_trees
        ])
        foreign_ops = {op.name: op for op in foreign_ops}
        self.operators = {**self.operators, **foreign_ops}
        for t in bonus_trees:
            if t.size()==1:
                raise ValueError(f"WTTF, a bonus tree {t} with size 1, in bonus set {bonus_trees}")
            t.copy_out(self)

    def set_up_run(
            self,
            to_graph: Collection[str] = None
        ) -> tuple[dict[str, list], dict, GPNonTerminal]:
        """Sets up a genetic programming run.
        """
        # make base filename for saving data. This means the filename will record
        # when the run started, not when the file was saved. It also means pickle
        # saves of the run data will overwrite previous pickles.
        # If a scoreboard is provided in kwargs, use it
        if to_graph:
            for g in flatten(to_graph):
                if 'time' not in g and g not in self.scoreboard.def_outputs:
                    raise ValueError(
                        f"Grapher is instructed to graph {g}, which is not in the " +
                        "pipeline"
                    )
        self.grapher = Grapher(to_graph) if to_graph else None
        self._generate_starting_sample(self.pop, self.tree_factory, *self.scoreboard.obs.ivs)
        # self.sw()

    def _generate_starting_sample(self, pop, genfunc=None, vars_=None, **kwargs):
        """Creates the initial population for a GP run"""
        self.clear()
        for _ in range(pop):
            genfunc(*vars_, **kwargs)

    def run_episode(self, ping_freq=1):
        for i in range(self.episode_len):
            # Would be nice to replace this with a proper progress bar
            if ping_freq and not i%ping_freq:
                print(f'{i} generations')
            best = self._run_step() 
            self.update_record(best, i)
            if self.grapher:
                self.update_plots(i)
            # XXX this may be useful for agentless GP, 
            # XXX maybe make a param to turn it back on
            # if ping_freq and (not i or not (i+1)%ping_freq) and i != self.episode_len-1:
            #     self.make_parquets(i)
        if self.grapher:
            self.grapher.save(self.make_filename('plots_1', 'png'))
        return self._last_step()

    def _run_step(self): 
        """A single evolutionary step for GP"""
        mc_idxs = copy(self._mutator_change_indices)
        mutmuts = copy(self.mutator_factories)
        # First, get an array of all the trees (root nodes only)
        # (right now it's float-rooted nodes only ,but may change this later)  
        old_gen = self.get_all_root_nodes()[float].array()
        for _t in old_gen:
            if _t.size() == 1:
                raise ValueError(f"{_t} in old_gen WTFFF")
        # Output a dict containing the best tree and its scores
        best = self.scoreboard.score_trees(old_gen, except_for=self.expt_outvals)
        # Mark the non-k-best trees for deletion
        self.scoreboard.k_best(self.elitism)
        # extracts fitness scores from scoreboard, replacing NANs with 0's
        scores = np.array(self.scoreboard['fitness'].fillna(0))
        # normalise these to sum to 1 and range between 0 and 1
        sum_scores = np.sum(scores)
        # As long as they don't sum to zero, this can be done
        the_masses = self.pop-self.elitism
        new_gen = []
        # If there are negative scores, shift the range up so the
        # minimum value is zero, and take the sum again
        if np.min(scores) < 0:
            scores -= np.min(scores)
            sum_scores = np.sum(scores)
        if sum_scores != 0:
            # calculate normed scores    
            normed_scores = scores/sum_scores
            # make copies of trees, to replace those marked for
            # deletion, using the normed scores as the probability 
            # for each tree of being copied in an individual copying
            # event
            # Temp try-except for debugging
            n = 0
            for t in self.np_random.choice(
                old_gen, 
                p=normed_scores, # scores/np.sum(scores), 
                size=the_masses
            ):
                # Note that making a copy automatically adds it to the
                # treebank, so there's no need to assign this 
                # print(f'Tree {n} of {the_masses}, copying {t}')
                if mc_idxs and n == mc_idxs[0]: # if the list of indices at which 
                    # mutators are to be changed is non-empty, and the first index
                    # listed is the current generation...
                    mutmuts.pop(0)() # pop the head of the list of MutatorMutators
                    # and call it, which will change the mutators on all trees 
                    mc_idxs.pop(0) # also pop the head of the list of indices, as it
                    # has been used
                n +=1
                new_gen.append(t.copy(gp_copy=True)) 
                if new_gen[-1].size()==1:
                    ic(t.metadata)
                    for tt in old_gen:
                        if tt.size():
                            ic(tt)
                    ic(new_gen[-1].metadata)
                    ic.disable()
                    raise ValueError(
                        f"Item {len(new_gen)-1} is a size-1 tree, {new_gen[-1]}, copied fron {t}: WTF?!"
                    )
        # However, if they sum to exactly zero, that is practically
        # certainly because all the trees are NANing out - in which case
        # 1. print out deets, because that is a fucky outcome
        # 2. pick trees to GP-copy at random
        else:
            for t in self.np_random.choice(old_gen, size=the_masses):
                new_gen.append(t.copy(gp_copy=True)) 
        ###|##|#######################################
        ## |  | If final_best (the output of the run) exists, make sure it doesn't get deleted
        ###V##V#######################################
        # if final_best is not None:
        #     final_best['tree'][0].metadata['to_delete'] = False
        # gather up all trees of the old gen marked for deletion (if we're doing elitism)
        # or just the entire old gen (if we're not), and ...
        # deathlist = filter(
        #     lambda t: t.metadata['to_delete'] == True, old_gen
        # ) if self.elitism else [
        #     t for t in old_gen if id(t) != id(final_best['tree'][0])
        # ] if final_best is not None else old_gen
        deathlist = filter(
            lambda t: t.tmp['to_delete'] == True, old_gen
        ) if self.elitism else old_gen
        for t in new_gen:
            t.meta_set_recursive(__no_xo__=False)
        # ... and delete them
        for t in deathlist:
            t.delete()
        # and increment the generation counter
        self.gens_past += 1
        return best

    def _last_step(self)-> tuple[pd.DataFrame, dict]: # XXX TODO what does this actually return?
        """A single evolutionary step for GP"""

        # First, get an array of all the trees (root nodes only)
        # (right now it's float-rooted nodes only ,but may change this later)
        old_gen = self.get_all_root_nodes()[float].array()
        # Calculate fitness of all trees
        self.scoreboard.score_trees(old_gen, except_for=self.expt_outvals)
        self.gens_past += 1
        # and return the best
        return self.scoreboard.k_best(1, mark_4_del=False, tree_only=False)

    def update_record(self, best, i):
        for k, v in best.items():
            # in case k is a variable that hasn't previously been recorded
            # assign it zeroes in any previous rows 
            if not k in self.record:
                self.record[k] = np.zeros(len(self.record), dtype=float if k in self.record_means else type(v))
            # for some variables, the mean of the scoreboard is considered
            # more informative than the individual value for the best tree:
            # self.record_means lists these variables 
            if k in self.record_means:
                self.record.at[i, k] = self.scoreboard[k].mean()
            else:
                self.record.at[i, k] = v
        # and some graphing (which should also be in its own method)
    
    def update_plots(self, n):
        if n>0:
            self.grapher.update(self.record, n)
    
    def make_filename(self, name: str, ext: str, _dir=None) -> str:
        dt=datetime.now()
        fpath = Path(f"{name}_{dt.year}_{dt.month}_{dt.day}_{dt.hour}_{dt.minute}_{dt.second}.{ext}")
        if _dir:
            fpath = _dir / fpath
        elif _dir is None:
            fpath = self.data_dir / fpath
        return fpath

    def make_parquets(self, i):
        pth = self.make_filename('record', 'parquet')
        self.record.to_parquet(pth)
        self.scoreboard[[
            col for col in self.scoreboard if col!='tree'
        ]].to_parquet(self.make_filename(f'scoreboard_{i}', 'parquet'))

    def process_run_output(self, final_best):
        self.make_parquets('final')
        self.best = {k: list(v)[0] for k, v in final_best.items()}
        #print(final_best['tree'])
        best_tree = final_best.loc[0, 'tree'].copy_out()
        final_best['tree'] = f'{best_tree}'
        # XXX fix this
        # with open(self.make_filename('final', 'json'), 'w') as file:
        #     json.dump({**{k: v for k, v in self.best.items() if k != 'tree'}, **{'tree': f"{self.best['tree']}"}}, file)
        return self.record.copy(), final_best, best_tree

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


        
