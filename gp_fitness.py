import pandas as pd
from typing import Collection, Protocol, Iterable, Callable, Generic
from gp_trees import GPNonTerminal
from observatories import Observatory
from dataclasses import dataclass
from copy import deepcopy
from functools import wraps

def collect(a, t: type[Collection]):
    return a if isinstance(a, t) else t(a) if isinstance(a, Collection) and not isinstance(a, str) else t([a])


# use pydantic/pandera to specify Dataframe should have 'trees: Tree' (can I make a custon dtype?) and 'fitness: float'
class FitnessFunction(Protocol):
    def __call__(trees: Collection[GPNonTerminal], obs: Observatory, temp_coeff: float, **kwargs) -> pd.DataFrame:
        ...

@dataclass
class ScoreboardPipelineElement:
    arg_keys: list[str]|str
    out_key: str
    vec: bool = False
    fn: Callable = None

    def __post_init__(self):
        self.arg_keys = collect(self.arg_keys, list)
        if 'tree' in self.arg_keys:
            self.vec = False
        if len(self.arg_keys) != 1 and self.fn is None:
            raise ValueError(
                'If no function is passed to ScoreboardPipelineElement(), ' +
                'it will be interpretted as an instruction to rename a ' +
                'single column of the scoreboard. As such, `arg_keys` ' +
                'be a string or a list containing a single string. ' +
                f'Instead, {self.arg_keys} was passed.'
            )

    def __call__(self, sb: pd.DataFrame, **kwargs):
        if not self.fn:
            sb.rename(columns={self.arg_keys[0]: self.out_key}, inplace=True)
        elif not (self.arg_keys or self.out_key):
            self.fn(sb)
        elif self.vec:
            sb[self.out_key] = self.fn(**sb[self.arg_keys], **kwargs)
        else:
            sb[self.out_key] = sb.apply(lambda row: self.fn(**row, **kwargs), axis=1)

    def __str__(self):
        return f"ScoreboardPipelineElement(arg_keys={self.arg_keys}, out_key='{self.out_key}', vec={self.vec}, fn={self.fn.__name__ if self.fn else 'None'})"
    
    @property
    def requires(self) -> list[str]:
        return self.arg_keys if self.arg_keys else self._reqs if hasattr(self, '_reqs') else ['<UNK>']
    
    @property
    def adds(self) -> list[str]:
        return [self.out_key] if self.out_key else self._adds  if hasattr(self, '_adds') else ['<UNK>']
    
    @property
    def deletes(self) -> list[str]:
        if self.out_key and self.arg_keys:  
            return self.arg_keys if self.fn is None else []
        else:
            return self._dels if hasattr(self, '_dels') else ['<UNK>']
        
    @property
    def __name__(self):
        return 'rename' if self.fn is None else self.fn.__name__
        
    def _set_validators(self, reqs, adds, dels):
        self._reqs = reqs
        self._adds = adds
        self._dels = dels


def scoreboard_pipeline_element(arg_keys=['tree'], out_key='fitness', vec=False):
    # Outer decorator returns the inner decorator
    def spe_decorator(fn: Callable):
        return ScoreboardPipelineElement(arg_keys=arg_keys, out_key=out_key, vec=vec, fn=fn)
    # return the decorator that actually takes the function in as the input
    return spe_decorator

def flexi_pipeline_element(vec=True, out_key_maker: Callable = None):
    # Outer decorator returns the inner decorator
    def fpe_decorator(fn: Callable):
        # For reasons which will become clear in a few lines time,
        # it is necessary to ensure the function below, `spe_maker`, 
        # has an attribute `__wrapper__` that refers to the function
        # the call to `flexi_pipeline_element` decorates 
        @wraps(fn)
        def spe_maker(*arg_keys, out_key=None, **kwargs):
            out_key = out_key if out_key else out_key_maker(arg_keys) if out_key_maker else f"{fn.__name__}_{'_'.join(arg_keys)}"
            argstr = ', '.join(arg_keys)
            # Note that in the global namespace, the function name in the line
            # below does not refer to the function `fn` the decorator wraps, 
            # but the decorated function, which is underlyingly `spe_maker`.
            # To ensure the correct behaviour, `spe_maker` was decorated with
            # functools.wraps, which ensures that when the `fpe_decorator`
            # maps the name of the decorated function to an instance of
            # `spe_maker` in the global namespace, the function created by the
            # `exec` call below can use `__wrapped__` to access the decorated
            # function, not `spe_maker`, which is the desired behaviour  
            exec(
                f"def {out_key}({argstr}, **kwargs):\n" +
                f"\treturn {fn.__name__}.__wrapped__({argstr}, **kwargs)\n"
            )
            return ScoreboardPipelineElement(
                arg_keys=arg_keys, 
                out_key=out_key, 
                fn=locals()[out_key],
                vec=vec
            )
        return spe_maker
    return fpe_decorator


@flexi_pipeline_element(out_key_maker=lambda args: f"i{args[0]}")
def safe_inv(arg, **kwargs):
    return 1/(arg+1)

@scoreboard_pipeline_element(out_key='mse')
def mse(tree: GPNonTerminal, target: pd.Series=None, ivs=None, **kwargs):
    return ((target - tree(**ivs))**2).mean()

@scoreboard_pipeline_element(out_key='sae')
def sae(tree: GPNonTerminal, target: pd.Series=None, ivs=None, **kwargs):
    return (target - tree(**ivs)).abs().mean()

@scoreboard_pipeline_element(arg_keys="mse", out_key="rmse", vec=True)
def rmse(mse: float, **kwargs):
    return mse**0.5

def rename(old_name: str, new_name: str):
    return ScoreboardPipelineElement(arg_keys=old_name, out_key=new_name)

@scoreboard_pipeline_element(arg_keys="raw_fitness", out_key="fitness", vec=True)
def heat(raw_fitness: float, temp_coeff: float, **kwargs):
    if temp_coeff:
        temp = raw_fitness.mean() * temp_coeff
    else:
        temp = 0.0
    return raw_fitness + temp

def clear(*to_clear: str, except_for: str|list[str]=None):
    to_clear = list(to_clear)
    if except_for:
        if to_clear:
            raise ValueError(
                "The `except_for` keyword arg should only be used with clear" +
                " if no positional args are passed: if no positional args are" +
                "passed, all columns are clear, except for those in `except_for`"
            )
        except_for = collect(except_for, list)
    @scoreboard_pipeline_element(arg_keys=[], out_key="")
    def clr(sb: GPScoreboard, **kwargs):
        if not to_clear:
            # new variable name with underscore, because if I assign to 
            # `to_clear` here, Python will think this is a new variable in
            # the inner function namespace, and then get confused by the
            # line *above*, thinking it's referring to the as-yet-non-value
            # -bearing local *below*, and not the variable in the outer 
            # function namespace. <facepalm/>
            _to_clear = list(sb.columns)
            for exn in except_for:
                if exn in _to_clear:
                    _to_clear.remove(exn)
        sb.drop(columns=to_clear if to_clear else _to_clear, inplace=True)
        return sb
    clr._set_validators(
        reqs=to_clear, adds=[], dels = to_clear if to_clear else ['<ALL>'] + [f"-{exn}" for exn in except_for])
    return clr

class GPScoreboard(pd.DataFrame):
    _metadata = ['obs', 'pipeline', 'kwargs', 'provide', 'require']

    def __init__(
            self,
            *pipeline: ScoreboardPipelineElement,
            obs: Observatory = None, 
            temp_coeff: float = 0.0,
            provide: str|Collection[str]='tree', 
            require: str|Collection[str]='fitness',
            **kwargs
        ) -> None:
        super().__init__()
        self.obs = obs
        self.pipeline = pipeline
        self.provide = collect(provide, set)
        self.require = collect(require, set)
        self.kwargs = {"temp_coeff": temp_coeff, **kwargs}
        valid, error = self._validate_pipeline()
        if not valid:
            raise ValueError(error)

    def _validate_pipeline(self) -> tuple[bool, str]:
        def format_error(p, i, e):
            err = "Pipeline failed"
            if i != -1:
                err += f" at element {i}, {p.__name__}"
            err += f": {e}"
            return err
        cols = deepcopy(self.provide)
        for i, p in enumerate(self.pipeline):
            reqs = p.requires
            adns = p.adds
            dlns = p.deletes
            if "<UNK>" in reqs:
                return False, format_error(p, i, "Unknown requirements")
            for req in reqs:
                if req not in cols:
                    return False, format_error(p, i, f"Required input '{req}' not found")
            if "<UNK>" in dlns:
                return False, format_error(p, i, 
                    "Scoreboard in unknown state, specify deletions for this element"
                )
            if "<ALL>" in dlns:
                new_cols = set()
                if len(dlns) > 1:
                    for dln in dlns:
                        if dln.startswith('-'):
                            if dln[1:] in cols:
                                new_cols.add(dln[1:])
                        elif dln != '<ALL>':
                            return False, format_error(p, i, 
                                f"Element clears all columns, but has other deletions ({dln}) listed"
                            )
                cols = new_cols
            else:
                for dln in dlns:
                    if dln not in reqs:
                        return False, format_error(p, i, 
                            f"Element deletes column '{dln}', but does not specify '{dln}' as a requirement"
                        )
                    if dln in adns:
                        return False, format_error(p, i, 
                            f"Scoreboard in unknown state, element specifies '{dln}' as both an addition and deletion"
                        )
                    cols.remove(dln)
            if "<UNK>" in adns:
                return False, format_error(p, i, 
                    "Scoreboard in unknown state, specify additions for this element"
                )
            for adn in adns:
                cols.add(adn)
        for rqmt in self.require:
            if rqmt not in cols:
                return False, format_error(p, -1, 
                    f"Pipeline completes, but required column {rqmt} is not in scoreboard"
                )
        return True, None

    def __call__(
            self, 
            trees: Iterable[GPNonTerminal], 
            obs: Observatory = None, 
            temp_coeff: float = None,
            dv: str=None, 
            **kwargs
        ):
        self['tree'] = pd.Series(trees)
        if temp_coeff is not None:
            self.kwargs["temp_coeff"] = temp_coeff 
        self.obs = self.obs if obs is None else obs
        if len(self.obs.dvs)>1:
            if not (dv and dv in obs.dvs):
                raise ValueError(
                    f"DV {dv} not found in Observatory dvs"
                    if dv else
                    f"Observatory obs has multiple DVs, but no kwarg `dv` is given to specify which one"
                )
        else:
            dv = self.obs.dvs[0]
        kwargs['ivs'] = next(self.obs)
        kwargs['target'] = self.obs.target[dv]
        for pipe_ele in self.pipeline:
            pipe_ele(self, **{**self.kwargs, **kwargs})
        return self
    
    def k_best(self, k, mark_4_del=True):
        """Selects the $k$ best trees in a collection based on a provided list
        of fitness scores. Optionally, marks the $|trees| - k$ non-best trees
        for deletion.

        Parameters
        ----------
            k (int):
                The number of highest-scoring trees to be returned
            mark_4_del (bool):
                If `mark_4_del` is set to `True`, all trees are given a boolean
                `metadata` tag, `'to_delete'`, which is `False` for the $k$ best
                and `True` otherwise.

        Returns
        -------
            A list of the `k` trees wih the highest values of `scores`

        >>> import operators as ops
        >>> from gp import *
        >>> op = [ops.SUM, ops.PROD]
        >>> gp = GP(operators = op)
        >>> t = [
        ...     gp.tree("([float]<SUM>([int]0)([int]1))"),
        ...     gp.tree("([float]<SUM>([int]2)([int]3))"),
        ...     gp.tree("([float]<SUM>([int]4)([int]5))"),
        ...     gp.tree("([float]<SUM>([int]6)([int]7))"),
        ...     gp.tree("([float]<SUM>([int]8)([int]9))")
        ... ]
        >>> gps = GPScoreboard(require=[])
        >>> gps['tree'] = t
        >>> gps['fitness'] = [6.66, 10.1, 0.02, 4.2, 9.9]
        >>> for k in range(6):
        ...     best = gps.k_best(k)
        ...     print(len(best))
        ...     print(sorted([tree() for tree in best])) 
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
        >>> nr = {'tree': gp.tree("([float]<SUM>([int]10)([int]11))"), 'fitness': 12.34}
        >>> gps.loc[len(gps)] = nr
        >>> nr = {'tree': gp.tree("([float]<SUM>([int]12)([int]13))"), 'fitness': -12.34}
        >>> gps.loc[len(gps)] = nr
        >>> best = gps.k_best(1, mark_4_del=False)
        >>> print([tt.metadata.get("to_delete", None) for tt in gps['tree']])
        [False, False, False, False, False, None, None]
        >>> print(best[0]()) 
        21.0
        """
        best = self.nlargest(k, 'fitness')['tree']
        if mark_4_del:
            self['tree'].apply(lambda t: t.metadata.__setitem__('to_delete', True))
            best.apply(lambda t: t.metadata.__setitem__('to_delete', False))
        return list(best)

    def winner(self, *cols, except_for: str|list[str]=None):
        """
        This retrieves a dictionary of selected row elements from the row
        containing the fittest tree. If the column for the tree itself is 
        included, a string representation is given, not the tree.

        Parameters
        ----------
            cols (str):
                If positional arguments are passed, they will be interpretted
                as a list of the columns that are to be included in the dict. 
                If none are passed, by default all columns will be used.
            except_for (str|list[str]):
                If positional arguments are passed, but `except_for` is 
                non-empty, all columns will be used except for those in
                `except_for`

        Returns
        -------
            best_dic (dict):
                record of the best (fittest) tree

        >>> import operators as ops
        >>> from gp import GP
        >>> from observatories import StaticObservatory
        >>> op = [ops.PROD]
        >>> gp = GP(operators = op, temperature_coeff=0.5)
        >>> iv = [0., 2., 4., 6., 8.,]
        >>> dv0 = [0., 0., 0., 0., 0.]
        >>> dv1 = [0., 10., 20., 30., 40.]
        >>> df = pd.DataFrame({'x': iv, 'y0': dv0, 'y1': dv1})
        >>> t = [
        ...     gp.tree("([float]<PROD>([float]$x)([int]1))"),
        ...     gp.tree("([float]<PROD>([float]$x)([int]3))"),
        ...     gp.tree("([float]<PROD>([float]$x)([int]5))"),
        ...     gp.tree("([float]<PROD>([float]$x)([int]7))"),
        ...     gp.tree("([float]<PROD>([float]$x)([int]9))"),
        ...     gp.tree("([float]<PROD>([float]6.)([int]6))")
        ... ]
        >>> obs0 = StaticObservatory('x', 'y0', sources=df, obs_len=5)
        >>> obs1 = StaticObservatory('x', 'y1', sources=df, obs_len=5)
        >>> gps0 = GPScoreboard(sae, mse, rmse, safe_inv('mse'), safe_inv('rmse'), rename('irmse', 'raw_fitness'), heat, obs=obs0, temp_coeff=1.0)
        >>> gps0(t)[list(gps0.columns)[1:]]
            sae     mse       rmse      imse  raw_fitness   fitness
        0   4.0    24.0   4.898979  0.040000     0.169521  0.227852
        1  12.0   216.0  14.696938  0.004608     0.063707  0.122038
        2  20.0   600.0  24.494897  0.001664     0.039224  0.097555
        3  28.0  1176.0  34.292856  0.000850     0.028334  0.086666
        4  36.0  1944.0  44.090815  0.000514     0.022177  0.080509
        5  36.0  1296.0  36.000000  0.000771     0.027027  0.085359
        >>> gps1 = GPScoreboard(sae, mse, rmse, safe_inv('mse'), safe_inv('rmse'), rename('irmse', 'raw_fitness'), heat, obs=obs1, temp_coeff=1.0)
        >>> gps1(t)[list(gps1.columns)[1:]]
            sae    mse       rmse      imse  raw_fitness   fitness
        0  16.0  384.0  19.595918  0.002597     0.048553  0.269730
        1   8.0   96.0   9.797959  0.010309     0.092610  0.313787
        2   0.0    0.0   0.000000  1.000000     1.000000  1.221177
        3   8.0   96.0   9.797959  0.010309     0.092610  0.313787
        4  16.0  384.0  19.595918  0.002597     0.048553  0.269730
        5  17.6  456.0  21.354157  0.002188     0.044734  0.265911
        >>> gps0.winner('mse')
        {'mse': 24.0}
        >>> best0 = gps0.winner(except_for = 'mse')
        >>> round(best0['rmse'], 3)
        4.899
        >>> best0['tree']
        '([float]<PROD>([float]$x)([int]1))'
        >>> best0.keys()
        dict_keys(['tree', 'sae', 'rmse', 'imse', 'raw_fitness', 'fitness'])
        >>> best1 = gps1.winner()
        >>> best1['mse']
        0.0
        >>> round(best1['rmse'], 4)
        0.0
        >>> best1['tree']
        '([float]<PROD>([float]$x)([int]5))'
        >>> gps1 = GPScoreboard(sae, mse, rmse, safe_inv('mse'), safe_inv('rmse'), rename('irmse', 'raw_fitness'), heat, clear('tree'), obs=obs1, temp_coeff=1.0)
        >>> gps1(t)[list(gps1.columns)]
            sae    mse       rmse      imse  raw_fitness   fitness
        0  16.0  384.0  19.595918  0.002597     0.048553  0.269730
        1   8.0   96.0   9.797959  0.010309     0.092610  0.313787
        2   0.0    0.0   0.000000  1.000000     1.000000  1.221177
        3   8.0   96.0   9.797959  0.010309     0.092610  0.313787
        4  16.0  384.0  19.595918  0.002597     0.048553  0.269730
        5  17.6  456.0  21.354157  0.002188     0.044734  0.265911
        """
        if cols:
            if except_for:
                raise ValueError(
                    "Invalid keyword argument for GPScoreboard.winner: " +
                    "`except_for` should not be used unless no " +
                    "positional arguments are passed. If no positional " +
                    "arguments are passed, the method uses all columns " +
                    "in the scoreboard, except for any passed to " +
                    "`except_for`"
                )
            for col in cols:
                if col not in self:
                    raise ValueError(
                        "Invalid argument for GPScoreboard.winner: " +
                        f"'{col}' not in scoreboard."
                    )
        else:
            cols = list(self.columns)
            if except_for:
                ef = collect(except_for, list)
                for exn in ef:
                    if exn in cols:
                        cols.remove(exn)
        best = self.nlargest(1, 'fitness')[list(cols)]
        best_dic = {}
        for col in cols:
            if col=="tree":
                best_dic[col] = str(best[col].item())
            else:
                best_dic[col] = best[col].item()
        return best_dic
    

def main():
    import doctest
    doctest.testmod()
    # from test_trees import test_gp_trees
    # from observatories import StaticObservatory
    # tgpt = test_gp_trees()
    # sources = pd.DataFrame({
    #     'x': [1.0, 2.0, 3.0, 4.0, 5.0], 
    #     'y': [9.0, 18.0, 31.0, 48.0, 69.0]
    # })
    # obs = StaticObservatory('x', 'y', sources=sources, obs_len=5)
    # gps = GPScoreboard(
    #     sae, mse, rmse, safe_inv('mse'), safe_inv('rmse'), 
    #     rename('irmse', 'raw_fitness'), heat, obs=obs, temp_coeff=2.0)
    # gps(tgpt)
    # print(gps)
    
if __name__ == '__main__':
    main()


"""XXX besides ffs, think about selection algos:

tournament
elitism


XXX think abt mutators, too
modulate crossover prob  - leaf to func ratios
flip operator point mutations
headless chicken
FGGP p17: GP doesn't normally compose mutation operators - mutually exclusive

qs for running GP (FGGP.19):

what is the terminal set?
func set?
fit func
hyperparams
termination criteria

consider 0-arg funcs - measurement funcs, rand(), etc

p20 - ephemeral rand consts: don't mutate???

more ops at p21, table 3.1

FGGP says keep all types consistent, but doesn't type-labeling remove that prob
yes, FGGP notes

% is protected division = returns default for x/0
or, do error-catching

not a prob in py, but what about numeric overflow? actually, no, is prob for floats

think about ops with side effects - order of eval now matters!

max prog size

prog size fitness

composite ff's

90/10 crossover/point mutation, 50/50 also works
pop at least 500, some ppl use much more
    some ppl use smaller, with more point mutation than crossover
"""


#deleteme_1 XXX = {"""# if the variables are df columns of length > 1, estimates will also be
# a df column--so using df.apply to generate a column in which each cell
# contains a df of shape (len(target), 1) doesn't work: pandas assumes
# you're doing something weird with multiple columns and
# exception-shames you for it. So, instead make an empty columns and
# fill it using a for loop"""}
    
#XXX = {"""# for i, t in enumerate(scoreboard['tree']):
#     results['estimates'][i] = (((t(**ivs) - obs.target)**2).mean())**0.5
# Calculate MSE: this takes slightly different code for the case where
# there is no variable anywhere in the tree, so the output is a df
# containing a single number: if I subtract the target from a 1x1 df,
# the result will be a df with on numerical value and a load of nulls:
# if I subtract the target from a raw number `x`, the result will be a
# df containing `x-target[i]` for all values of `i`
# scoreboard["mses"] = results['estimates'].apply(
#     lambda estimate:
#         np.square(estimate - target).mean() # MSE
#         if len(estimate) > 1               # In case a tree has no vars,
#         else np.square(estimate[0] - target).mean() #  in which case the
# )                                          # output will be one number
# use the temp over temp+MSE as fitness."""}
    

# best = {} # ... so will `best`
# best["best_mse"] = min(scoreboard["mses"]) # ... with the best MSE (lowest)
# if rmses: # optional return values: Root Mean Squard Error
#     scoreboard['rmses'] = [mse**.5 for mse in scoreboard["mses"]]
# ## Find the best tree
# best_tr = self.k_best(1, scoreboard, mark_4_del=False)
# if best_tree: # optionally, include the string representation of the
#     best["best_tree"] = str(best_tr[0]) # best tree in best
# if best_rmse: # and, optionally, the best RMSE
#     best["best_rmse"] = best["best_mse"]**.5
# return scoreboard, best


# def imse_fitness(
#         trees: Collection[GPNonTerminal], 
#         obs: Observatory, 
#         temp_coeff: float = 0.0,
#         dv: str=None, 
#         rmse: bool=False,
#         use_rmse: bool=False,
#         **kwargs
#     ) -> pd.DataFrame:
#     """Fitness function based on the inverse of either Mean Squared Error or
#     Root Mean Squared Error. This assumes only a single DV, but one or more IVs.

#     Parameters
#     ==========
#         trees (Collection[GPNonTerminal]): The root treenodes of a GP treebank
#         obs (Observatory): An Observatory object that supplies values for the
#             IVs, and the target value of the DV.
#         temp_coeff (float): A coefficient which determines the 'temperature' of
#             the system. The `temp` of the system is equal to the mean of the 
#             error measure that determines fitness (MSE or RMSE) times the 
#             `temp_coeff`. Therefore, if `temp_coeff` is 0.0, `temp` is 0.0 too.
#             The idea here is that the inverse of error is a tree's 'raw 
#             fitness', but its 'effective fitness' is 'raw fitness' + temp. 
#             This is equivalent to dividing each round of selection into a part
#             where probability of selection is proportionate to raw fitness, and
#             a part where the probability of selection is equal for all trees,
#             where the relative sizes of the two parts is determined by `temp`.
#         dv (str): If obs returns multiple dvs, this fitness function can only 
#             attend to one, and `dv` specifies which. If `obs` only gives one
#             dv, this arg is not needed, and can be left as its default value,
#             `None`. 
#         rmse (bool): If `True`, calculate RMSE (default False). If 
#             `use_rmse` is `True`, RSME must be calculated, so `rmse` is 
#             redundant
#         use_rmse (bool): If `True`, use RMSE as the error measure from which
#             fitness is calculated. If `False` (default), use MSE.

#     >>> from test_trees import test_gp_trees
#     >>> from observatories import StaticObservatory
#     >>> tgpt = test_gp_trees()
#     >>> sources = pd.DataFrame({'x': [1.0, 2.0, 3.0, 4.0, 5.0], 'y': [9.0, 18.0, 31.0, 48.0, 69.0]})
#     >>> obs = StaticObservatory('x', 'y', sources=sources, obs_len=5)
#     >>> imse_fitness(tgpt, obs)
#     """
#     # Using a dataframe to store error measures and scores - C under the
#     # hood and SIMD funtimes
#     # Retrieve the current population
#     scoreboard = pd.DataFrame({'tree': trees})
#     dv = obs.dvs[0]
#     if len(obs.dvs)>1:
#         if not (dv and dv in obs.dvs):
#             raise ValueError(
#                 f"DV {dv} not found in Observatory dvs"
#                 if dv else
#                 f"Observatory obs has multiple DVs, but no kwarg `dv` is given to specify which one"
#             )
#     ivs = next(obs)
#     scoreboard['mse'] = scoreboard['tree'].map(
#         lambda t: ((obs.target[dv] - t(**ivs))**2).mean()
#     )
#     use_rmse = kwargs.get('use_rmse', False)
#     err_measure = 'rmse' if use_rmse else 'mse'
#     if kwargs.get('rmse', False) or use_rmse:
#         scoreboard['rmse'] = scoreboard.mse**0.5
#     scoreboard["raw_fitness" if temp_coeff else "fitness"] = (
#         1.0/(1.0+scoreboard[err_measure])
#     ) # fitness value, if temp coeff is 0
#     if temp_coeff:
#         temp = scoreboard["raw_fitness"].mean() * temp_coeff
#         scoreboard['fitness'] = scoreboard['fitness'] + temp
#     return scoreboard