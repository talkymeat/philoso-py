from treebanks import TypeLabelledTreebank
from gp_trees import GPTerminal, GPNonTerminal
from gp_fitness import *
import pandas as pd
import numpy as np
from copy import copy
from typing import Union, List, Callable, Mapping
from random import choices, choice
from observatories import *
from tree_factories import *
from utils import collect

DEBUG = False

def _print(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)

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

    def generate_starting_sample(self, pop, genfunc=None, vars_=None, **kwargs):
        self.clear()
        for i in range(pop):
            genfunc(*vars_, **kwargs)


    def run_gp(
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
            ping_freq: int = 50
        ) -> tuple[dict[str, list], GPNonTerminal]:
        if fitness:
            scoreboard = fitness
        else:
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
        expt_outvals = collect(expt_outvals, list)
        record = {}
        # First generate the initial population that will be evolved
        tree_factory.set_treebank(self)
        self.generate_starting_sample(pop, tree_factory, *observatory.ivs)
        for n in range(steps):
            if not n%ping_freq:
                print(f'{n} generations')
            old_gen = self.get_all_root_nodes()[float]
            best = scoreboard.score_trees(old_gen, except_for=expt_outvals)
            scoreboard.k_best(elitism)
            if n == steps-1:
                final_best = scoreboard.k_best(-1, mark_4_del=False)[0]
            for t in choices(old_gen, scoreboard['fitness'], k=pop-elitism):
                t.copy(gp_copy=True) # WUT XXX
            deathlist = filter(
                lambda t: t.metadata['to_delete'] == True, old_gen
            ) if elitism else old_gen
            for t in deathlist:
                t.delete()

            # TODO make functions for the following... (or even a class?)
            for k, v in best.items():
                if not k in record:
                    record[k] = [] + [None]*(n-1)
                record[k].append(v)
            for _k, _v in record.items():
                shortfall = n - len(_v)
                if shortfall:
                    if shortfall > 1:
                        raise ValueError(
                            "Value lists in `record`, should never be of" +
                            " a length less than n-1 where `n` is the " +
                            f"number of generations: but somehow, {_k} " +
                            f"is of length {len(_k)}, {shortfall} less " +
                            f"than {n}. Weird."
                        )
                    else:
                        _v.append(None)
        return record, final_best


def main():
    import doctest
    doctest.testmod()

if __name__ == '__main__':
    main()





    # # XXX score_trees has been changed - new name & new sig XXX
    # def gp_update(
    #             self, 
    #             observatory: Observatory,
    #             pop=None, 
    #             fitness: GPScoreboard=None,  
    #             elitism=0, 
    #             best_tree: bool=True, 
    #             rmses: bool=True,
    #             best_rmse: bool=True,
    #             **kwargs
    #         ):
    #     """Once fitness scores are calculated, this generates the next
    #     generation of trees. If `elitism` > 0, the value of the `elitism` param
    #     gives the number of trees that will be spared without chance of mutation
    #     or crossover into the next generation. The rest of the next generation
    #     is produced by replicating trees from the previous, with a probability
    #     related to fitness, subject to mutation and crossover.

    #     >>> #Note that the doctest for this method has been placed in a function
    #     >>> #and SKIPped. The SKIP tag should be removed if the method is
    #     >>> #changed and needs to be re-tested
    #     >>> def test():
    #     ...     import operators as ops
    #     ...     op = [ops.PROD]
    #     ...     totals = np.array([0,0,0,0,0])
    #     ...     w = np.array([.5, .3, .1, .07, .03])
    #     ...     n = 20000
    #     ...     targ_totals = n * (np.array([1., 1., 0., 0., 0.]) + (3 * w))
    #     ...     for _ in range(n):
    #     ...         gp = GPTreebank(operators = op)
    #     ...         t = [
    #     ...             gp.tree("([float]<PROD>([float]1.)([float]0.))"),
    #     ...             gp.tree("([float]<PROD>([float]1.)([float]1.))"),
    #     ...             gp.tree("([float]<PROD>([float]1.)([float]2.))"),
    #     ...             gp.tree("([float]<PROD>([float]1.)([float]3.))"),
    #     ...             gp.tree("([float]<PROD>([float]1.)([float]4.))")
    #     ...         ]
    #     ...         scores = pd.Series(w)
    #     ...         gp.gp_update(scores=scores, elitism=2)
    #     ...         for t in gp.get_all_root_nodes()[float]:
    #     ...             totals[int(t()[0])] +=1
    #     ...     d = (totals-targ_totals)/totals
    #     ...     if not (d < 0.05).all():
    #     ...         print("alas:", arr)
    #     ...     return d
    #     >>> test() # doctest: +SKIP
    #     array([ True,  True,  True,  True,  True])
    #     """
    #     pass
        
