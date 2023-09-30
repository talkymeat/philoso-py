from treebanks import TypeLabelledTreebank
from gp_trees import GPTerminal, GPNonTerminal
import pandas as pd
import numpy as np
from copy import copy
from typing import Union, List, Callable, Mapping
from random import choices, choice
from observatories import *
from tree_factories import *

DEBUG = False

def _print(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)

class GP(TypeLabelledTreebank):
    

    def __init__(self,
            crossover_rate: float = 0.0,
            mutation_rate: float = 0.0,
            mutation_sd: float = 0.0,
            temperature_coeff: float = 1.0,
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
        self.temperature_coeff = temperature_coeff
        self.T = GPTerminal
        self.N = GPNonTerminal

    def create_seed_polynomial(
            self, vars_, order=None, const_min=None, const_max=None):
        pass

    def k_best(self, k, scoreboard, mark_4_del=True):
        """Selects the $k$ best trees in a collection based on a provided list
        of fitness scores. Optionally, marks the $|trees| - k$ non-best trees
        for deletion.

        Parameters
        ----------
            k (int):
                The number of highest-scoring trees to be returned
            trees (list[Tree]):
                The trees assigned scores by the fitness function
            scores (list[float]):
                The scores assigned by the fitness function, in order
            mark_4_del (bool):
                If `mark_4_del` is set to `True`, all trees are given a boolean
                `metadata` tag, `'to_delete'`, which is `False` for the $k$ best
                and `True` otherwise.

        Returns
        -------
            A list of the `k` trees wih the highest values of `scores`

        >>> import operators as ops
        >>> op = [ops.SUM, ops.PROD]
        >>> gp = GP(operators = op)
        >>> t = [
        ...     gp.tree("([float]<SUM>([int]0)([int]1))"),
        ...     gp.tree("([float]<SUM>([int]2)([int]3))"),
        ...     gp.tree("([float]<SUM>([int]4)([int]5))"),
        ...     gp.tree("([float]<SUM>([int]6)([int]7))"),
        ...     gp.tree("([float]<SUM>([int]8)([int]9))")
        ... ]
        >>> s = [6.66, 10.1, 0.02, 4.2, 9.9]
        >>> scoreboard = pd.DataFrame({'trees': t, 'fitness': s})
        >>> for k in range(6):
        ...     best = gp.k_best(k, scoreboard)
        ...     print(len(best))
        ...     print(sorted([tb().item() for tb in best]))
        ...     print([tt.metadata["to_delete"] for tt in t])
        0
        []
        [True, True, True, True, True]
        1
        [5.0]
        [True, False, True, True, True]
        2
        [5.0, 17.0]
        [True, False, True, True, False]
        3
        [1.0, 5.0, 17.0]
        [False, False, True, True, False]
        4
        [1.0, 5.0, 13.0, 17.0]
        [False, False, True, False, False]
        5
        [1.0, 5.0, 9.0, 13.0, 17.0]
        [False, False, False, False, False]
        >>> t.append(gp.tree("([float]<SUM>([int]10)([int]11))"))
        >>> s.append(12.34)
        >>> t.append(gp.tree("([float]<SUM>([int]12)([int]13))"))
        >>> s.append(-12.34)
        >>> best = gp.k_best(1, scoreboard, mark_4_del=False)
        >>> print([tt.metadata.get("to_delete", None) for tt in t])
        [False, False, False, False, False, None, None]
        >>> print(best[0]().item())
        5.0
        """
        best = scoreboard.nlargest(k, 'fitness')['trees']
        if mark_4_del:
            scoreboard['trees'].apply(lambda t: t.metadata.__setitem__('to_delete', True))
            best.apply(lambda t: t.metadata.__setitem__('to_delete', False))
        return list(best)
    
    def get_estimates(self, observatory: Observatory) -> pd.DataFrame:
        

    def score_trees(
                self, trees_, target, best_tree=False, 
                rmses=False, best_rmse=False
            ):
        """
        This calculates a number of measures of the accuracy of the estimates of
        the trees in the treebank, including the negative of the Mean Squared
        Error, which is used as a fitness function. It also calculates the MSEs
        and the best MSE, and some other measures which are optional. MSE, -MSE,
        and other per-tree measures are returned in a pandas dataframe, and the
        best MSE and other measures related to the best tree are returned in a
        dict.

        XXX: This should be refactored, so that fitness functions can be
        imported and passed into GP as a parameter (dependency injection)

        Parameters
        ----------
            target (DataFrame[float]):
                The target values which GP is trying to learn
            best_tree (bool):
                If `True`, include a string representation of the best-scoring
                tree in the returned dict, `best`
            rmses (bool):
                If `True`, include a column giving the Root Mean Squared rror
                in the returned DataFrame, `results`
            best_rmse (bool):
                If `True`, include the Root Mean Squared Error the best-scoring
                tree in the returned DataFrame, `results`

        Returns
        -------
            results (DataFrame):
                Contains the following columns:
                    'mses': Mean Squared Error for each tree in the treebank's
                        estimation of `target`
                    'fitness': temp/(temp+MSE) - where temp is a positive number
                    'rmse': (optional) Root Mean Squared Error
            best (dict):
                Contains the following keys/value pairs:
                    'best_mse': the lowest value in results['mses']
                    'best_rmse': the lowest value in results['rmses']
                    'best_tree': string representation of the tree that produced
                        the lowest MSE

        >>> import operators as ops
        >>> op = [ops.PROD]
        >>> gp = GP(operators = op, temperature_coeff=0.5)
        >>> df = pd.DataFrame({'x': [0., 2., 4., 6., 8.,]})
        >>> t = [
        ...     gp.tree("([float]<PROD>([float]$x)([int]1))"),
        ...     gp.tree("([float]<PROD>([float]$x)([int]3))"),
        ...     gp.tree("([float]<PROD>([float]$x)([int]5))"),
        ...     gp.tree("([float]<PROD>([float]$x)([int]7))"),
        ...     gp.tree("([float]<PROD>([float]$x)([int]9))"),
        ...     gp.tree("([float]<PROD>([float]6.)([int]6))")
        ... ]
        >>> t0 = pd.Series([0., 0., 0., 0., 0.])
        >>> t1 = pd.Series([0., 10., 20., 30., 40.])
        >>> res0, bes0 = gp.make_scoreboard(t, t0, best_tree=True, rmses=True, best_rmse=True)
        >>> res0['mses']
        0      24.0
        1     216.0
        2     600.0
        3    1176.0
        4    1944.0
        5    1296.0
        Name: mses, dtype: float64
        >>> res0['fitness'].round(6)
        0    0.948052
        1    0.669725
        2    0.421965
        3    0.271375
        4    0.183879
        5    0.252595
        Name: fitness, dtype: float64
        >>> res0['rmses'].round(4)
        0     4.8990
        1    14.6969
        2    24.4949
        3    34.2929
        4    44.0908
        5    36.0000
        Name: rmses, dtype: float64
        >>> bes0['best_mse']
        24.0
        >>> round(bes0['best_rmse'], 4)
        4.899
        >>> bes0['best_tree']
        '([float]<PROD>([float]x)([int]1))'
        >>> res1, bes1 = gp.make_scoreboard(t, t1, best_tree=True, rmses=True, best_rmse=True)
        >>> res1['mses']
        0    384.0
        1     96.0
        2      0.0
        3     96.0
        4    384.0
        5    456.0
        Name: mses, dtype: float64
        >>> res1['fitness'].round(6)
        0    0.235060
        1    0.551402
        2    1.000000
        3    0.551402
        4    0.235060
        5    0.205575
        Name: fitness, dtype: float64
        >>> res1['rmses'].round(4)
        0    19.5959
        1     9.7980
        2     0.0000
        3     9.7980
        4    19.5959
        5    21.3542
        Name: rmses, dtype: float64
        >>> bes1['best_mse']
        0.0
        >>> round(bes1['best_rmse'], 4)
        0.0
        >>> bes1['best_tree']
        '([float]<PROD>([float]x)([int]5))'
        """
        # Using a dataframe to store error measures and scores - C under the
        # hood and SIMD funtimes
        # Retrieve the current population
        scoreboard = pd.DataFrame({'trees': trees_})
        # if the variables are df columns of length > 1, estimates will also be
        # a df column--so using df.apply to generate a column in which each cell
        # contains a df of shape (len(target), 1) doesn't work: pandas assumes
        # you're doing something weird with multiple columns and
        # exception-shames you for it. So, instead make an empty columns and
        # fill it using a for loop
        results = pd.DataFrame({'estimates': [None]*len(trees_)})
        for i, t in enumerate(scoreboard['trees']):
            results['estimates'][i] = t()
        # Calculate MSE: this takes slightly different code for the case where
        # there is no variable anywhere in the tree, so the output is a df
        # containing a single number: if I subtract the target from a 1x1 df,
        # the result will be a df with on numerical value and a load of nulls:
        # if I subtract the target from a raw number `x`, the result will be a
        # df containing `x-target[i]` for all values of `i`
        scoreboard["mses"] = results['estimates'].apply(
            lambda estimate:
                np.square(estimate - target).mean() # MSE
                if len(estimate) > 1               # In case a tree has no vars,
                else np.square(estimate[0] - target).mean() #  in which case the
        )                                          # output will be one number
        # use the temp over temp+MSE as fitness.
        temp = scoreboard['mses'].mean() * self.temperature_coeff
        scoreboard["fitness"] = temp/(scoreboard["mses"]+temp) # return value
        best = {} # ... so will `best`
        best["best_mse"] = min(scoreboard["mses"]) # ... with the best MSE (lowest)
        if rmses: # optional return values: Root Mean Squard Error
            scoreboard['rmses'] = [mse**.5 for mse in scoreboard["mses"]]
        # Find the best tree
        best_tr = self.k_best(1, scoreboard, mark_4_del=False)
        if best_tree: # optionally, include the string representation of the
            best["best_tree"] = str(best_tr[0]) # best tree in best
        if best_rmse: # and, optionally, the best RMSE
            best["best_rmse"] = best["best_mse"]**.5
        return scoreboard, best
    
    # XXX score_trees has been chenged - new name & new sig XXX
    def gp_update(
                self, 
                observatory: Observatory,
                pop=None, 
                fitness: Callable[
                    [list[GPNonTerminal], pd.DataFrame, dict], tuple[pd.DataFrame, dict]
                ]=lambda trees, target, param_dict: GP.score_trees(trees, target, **param_dict),  
                elitism=0, 
                best_tree: bool=True, 
                rmses: bool=True,
                best_rmse: bool=True,
                **kwargs
            ):
        """Once fitness scores are calculated, this generates the next
        generation of trees. If `elitism` > 0, the value of the `elitism` param
        gives the number of trees that will be spared without chance of mutation
        or crossover into the next generation. The rest of the next generation
        is produced by replicating trees from the previous, with a probability
        related to fitness, subject to mutation and crossover.

        >>> #Note that the doctest for this method has been placed in a function
        >>> #and SKIPped. The SKIP tag should be removed if the method is
        >>> #changed and neds to be re-tested
        >>> def test():
        ...     import operators as ops
        ...     op = [ops.PROD]
        ...     totals = np.array([0,0,0,0,0])
        ...     w = np.array([.5, .3, .1, .07, .03])
        ...     n = 20000
        ...     targ_totals = n * (np.array([1., 1., 0., 0., 0.]) + (3 * w))
        ...     for _ in range(n):
        ...         gp = GP(operators = op)
        ...         t = [
        ...             gp.tree("([float]<PROD>([float]1.)([float]0.))"),
        ...             gp.tree("([float]<PROD>([float]1.)([float]1.))"),
        ...             gp.tree("([float]<PROD>([float]1.)([float]2.))"),
        ...             gp.tree("([float]<PROD>([float]1.)([float]3.))"),
        ...             gp.tree("([float]<PROD>([float]1.)([float]4.))")
        ...         ]
        ...         scores = pd.Series(w)
        ...         gp.gp_update(scores=scores, elitism=2)
        ...         for t in gp.get_all_root_nodes()[float]:
        ...             totals[int(t()[0])] +=1
        ...     d = (totals-targ_totals)/totals
        ...     if not (d < 0.05).all():
        ...         print("alas:", arr)
        ...     return d
        >>> test() # doctest: +SKIP
        array([ True,  True,  True,  True,  True])
        """
        old_gen = self.get_all_root_nodes()[float]
        if pop is None:
            pop = len(old_gen)
        scoreboard, best = fitness(
            old_gen, 
            observatory, 
            **kwargs
        )
        self.k_best(elitism, scoreboard['trees'], scoreboard['fitness'])
        new_gen = [
            t.copy(gp_copy=True) for t in choices(old_gen, scoreboard['fitness'], k=pop-elitism)
        ]
        for t in filter(lambda t: t.metadata['to_delete'] == True, old_gen):
            t.delete()
        return scoreboard, best
        

    def generate_starting_sample(self, pop, genfunc=None, vars_=None, **kwargs):
        self.clear()
        for i in range(pop):
            genfunc(*vars_, **kwargs)

    # The type hint for fitness ... maybe replace with pydantic?
    # def gp_step(self, target, pop, vars_=None, elitism=0,
    #         fitness: Callable[
    #             [list[GPNonTerminal]], tuple[pd.DataFrame, dict]
    #         ]=self.score_trees, 
    #         best_tree=False, rmses=False, best_rmse=False,
    #         tree_factories: Union[TreeFactory, List[TreeFactory]]=None,
    #         tree_factory_weights: List[Union[int, float]] = None):
    #     """...

    #     >>> import operators as op
    #     >>> from tree_factories import TestTreeFactory
    #     >>> import math
    #     >>> GP.DEBUG = True
    #     >>> ops = [op.SQ, op.CUBE]
    #     >>> gp = GP(crossover_rate=0.5, temperature_coeff=0.5, operators=ops)
    #     >>> gpx = GP(operators=ops)
    #     >>> tf0 = TestTreeFactory(gp, gpx.tree("([float]<SQ>([float]3.0))"))
    #     >>> tf1 = TestTreeFactory(gp, gpx.tree("([float]<CUBE>([float]2.0))"))
    #     >>> res = gp.gp_step(pd.DataFrame({'targ': [4.0]}), pop=20000, tree_factories=[tf0, tf1])
    #     generate initial sample, multiple factories
    #     >>> gp.get_all_root_nodes().keys()
    #     >>> pop = pd.DataFrame({'trees': gp.get_all_root_nodes()[float]})
    #     >>> pop['ests'] = pop['trees'].apply(lambda t: t())
    #     >>> pop['ests'].value_counts()
    #     >>> res = gp.gp_step(pd.DataFrame({'targ': [4.0]}), pop=20000, tree_factories=[tf0, tf1], elitism=20000, scores=pd.DataFrame({'fitness': [0.0]*20000}))
    #     evolution step
    #     >>> pop = pd.DataFrame({'trees': gp.get_all_root_nodes()[float]})
    #     >>> pop['ests'] = pop['trees'].apply(lambda t: t())
    #     >>> pop['ests'].value_counts()
    #     """
        # if scores is not None:
        #     _print('evolution step')
        #     self.gp_update(pop, fitness(scores), elitism=elitism)
        # else:
        #     raise AttributeError("if GP.gp_step() isn't given a non-empty " +
        #         "scores attribute, it must be given one or more TreeFactories")
        # return self.score_trees(
        #     target, best_tree=best_tree, rmses=rmses, best_rmse=best_rmse
        # )

    def run_gp(
            self,
            # vars_: pd.DataFrame,
            # target: pd.Series,
            tree_factory: TreeFactory,
            observatory: Observatory,
            steps: int,
            pop: int, 
            fitness: Callable[
                [list[GPNonTerminal], Observatory, dict], tuple[pd.DataFrame, dict]
            ]=lambda trees, target, param_dict: GP.score_trees(trees, target, **param_dict), 
            elitism: int = 0,
            best_tree: bool = False,
            rmses: bool = False,
            best_rmse: bool = False):
        scores = {}
        record = {}
        # First generate the initial population that will be evolved
        tree_factory.set_treebank(self)
        self.generate_starting_sample(pop, tree_factory, *observatory.ivs)
        for i in range(steps):
            scores, best = self.gp_update(
                observatory, 
                pop=pop,
                fitness=fitness,
                elitism=elitism,
                best_tree=best_tree, 
                rmses=rmses, 
                best_rmse=best_rmse)
            # TODO make functions for the following... (or even a class?)
            for k, v in scores.items():
                if not k in record:
                    record[k] = []
                if isinstance(v, list):
                    record[k].append(sorted(v))
                else:
                    record[k].append(v)
        record['best'] = best[0]
        return record


def main():
    import doctest
    doctest.testmod()

if __name__ == '__main__':
    main()
