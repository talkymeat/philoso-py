import pandas as pd
import numpy as np
from typing import Collection, Iterable, Callable
from gymnasium.spaces import Box
from gp_trees import GPNonTerminal
from observatories import Observatory
from rl_bases import Actionable
from dataclasses import dataclass
from copy import deepcopy
from functools import wraps
from utils import collect
from grapher import Grapher
import torch

from icecream import ic


# use pydantic/pandera to specify Dataframe should have 'trees: Tree' (can I make a custon dtype?) and 'fitness: float' XXX This isn't needed
# class FitnessFunction(Protocol):
#     def __call__(trees: Collection[GPNonTerminal], obs: Observatory, temp_coeff: float, **kwargs) -> pd.DataFrame:
#         ...

@dataclass
class ScoreboardPipelineElement:
    """The `GPScoreboard` class is an extension of `pandas.DataFrame` which 
    takes a column of GP Trees, and has an amount of apparatus for processing 
    the trees' output and giving calculating their fitness. The main system for
    doing this is a pipeline of operators which performs operations on the
    dataframe, usually by taking one or more columns of the dataframe and 
    performing an operation, the result of which is appended as a new column.
    The elements of this pipeline are ScoreboardPipelineElements, which is a
    Callable dataclass.

    To allow the Scoreboard to validate the pipeline before running, a set of 
    exist properties that track how the pipeline changes the set of columns in 
    the scoreboard. In most cases, this can be worked out from the attributes, 
    but in some cases, where no calues of `arg_keys` and `out_key` are given
    and `fn` operates on the whole dataframe, they can't; so additional 
    attributes may be added to provide that information. The properties 
    (documented below) are `requires`, `adds`, and `deletes`: the optional 
    attributes are `_reqs`, `_adds`, and `_dels`

    Attributes
    ==========
        arg_keys (string or tuple of strings): Keys which identify dataframe
            columns which be the input to the function, `fn`. If a `str` is
            passed, `__post_init__` will wrap the string in a singleton list.
            If no function is given, this and `out_key` will be used to rename
            a column, with `arg_keys[0]` being the column to be renamed and
            `out_key` being the new name: if a function is given but no values 
            for `arg_keys` and`out_key`, the function is assumed to apply to 
            the whole dataframe
        out_key (string): In most cases, this is the name given to the new 
            column output by `fn` - see ``arg_keys` for details of exceptions
        vec (bool): If true, it is assumed that `fn` vectorises: that is, it
            can operate on a pd.Series without needing to loop or use `df.map` 
            or `df.apply` to apply the function to each entry. If False, it 
            will be applied using `df.apply`
        fn (function): a function which can either be applied to one or more
            `pd.Series`, or a whole GPScoreboard. It takes positional arguments
            corresponding to `arg_keys`, and may have other kwargs to provide
            values that are stored in a parameter dictionary in GPScoreboard
        spem_kwargs (dict[strings, Any]): a dict containing any values specified
            at time of creation which need to be present for `fn` to be called
    """
    arg_keys: tuple[str]|str
    out_key: str
    vec: bool = False
    fn: Callable = None
    spem_kwargs: dict = None

    def __post_init__(self):
        """A housekeeping method that tidies the data a bit after 
        initialisation: it converts string `argkeys` values to singleton lists,
        sets `vec` to true if the function operates on the 'trees' column (which
        cannot vectorise), and raises errors for some non-permitted combinations
        of attributes
        """
        self.arg_keys = collect(self.arg_keys, tuple, empty_if_none=True)
        self.spem_kwargs = self.spem_kwargs if self.spem_kwargs else {}
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
        """The GPScoreboard pipeline calls the ScoreboardPipelineElements in 
        sequence: __call__ is the method than enables this. This behaves 
        slightly differently in different cases:

        1)  The Standard Case: `arg_keys` identifies scoreboard columns which 
            are passed to `fn`, the output of which is assigned to a new column
            named by `out_key`. `arg_keys`, `out_key` and `fn` must all be 
            non-null and non-empty
        2)  Renaming Mask: `arg_keys` is a singleton list, the member of which 
            is the label of the column to be renamed, and `out_key` is the key 
            that is output. `arg_keys`, `out_key` and `fn` must all be non-null 
            and non-empty, bu `fn` must be `None`.
        3)  Other: `fn` is called on the whole dataframe, `fn` must be non-null,
            but `arg_keys` and `out_key` must be empty.
        """
        # This method behaves differently depending on the existence of three
        # optional attributes, `arg_keys`, `out_keys` (together 'the keys) and
        # `fn` 
        if not self.fn:
            # If there is no `fn`, either we have an invalid element, if 
            # one or both keys are missing...
            if len(self.arg_keys) != 1 or not self.out_key:
                raise ValueError(
                    "If no function is given, `arg_keys` `out_key` must each " +
                    "be a single non-empty string"
                )
            # ...or both exist, and we are in the 'Renaming Mask' case
            sb.rename(columns={self.arg_keys[0]: self.out_key}, inplace=True)
        #  The renaming conditions are those where we know `fn` exists...
        elif not (self.arg_keys or self.out_key):
            # If neither 'key' exists, it's the 'Other' case...
            self.fn(sb, **self.spem_kwargs)
        elif not (self.arg_keys or self.out_key):
            # If only one, it's invalid
            raise ValueError(
                'If `fn` exists, either both `out_key` and `arg_keys` must ' +
                'exist, or neither'    
            )
        # At this stage, all three optional attributes are known to exist.
        # `vec` Should be set to true if `fn` uses vectorisation to operate
        # directly on whole columns 
        elif self.vec:
            sb[self.out_key] = self.fn(**sb[list(self.arg_keys)], **kwargs, **self.spem_kwargs)
        # If it can't, `vec` should be false, and `apply` will be used to go row
        # by row 
        else:
            sb[self.out_key] = sb.apply(lambda row: self.fn(**row, **kwargs, **self.spem_kwargs), axis=1)

    def __str__(self):
        """Returns a string representation of the ScoreboardPipelineElement"""
        return f"ScoreboardPipelineElement(arg_keys={self.arg_keys}, out_key='{self.out_key}', vec={self.vec}, fn={self.fn.__name__ if self.fn else 'None'})"
    

    @property
    def requires(self) -> list[str]:
        """The columns which must be in the scoreboard for the 
        ScoreboardPipelineElement to run. The element must have values for 
        `arg_keys` or `_reqs`: otherwise ['<UNK>'] is returned, the indicating  
        that the element's needs are unknown, and it may unpredicatably fail if
        run: this will cause validation to fail. Note that `_reqs` (but not
        `arg_keys`) can be empty.
        """
        return self.arg_keys if self.arg_keys else self._reqs if hasattr(self, '_reqs') else ['<UNK>']
    
    @property
    def adds(self) -> list[str]:
        """The columns which the ScoreboardPipelineElement adds the scoreboard 
        when run. The element must have values for `out_key` or `_adds`: 
        otherwise ['<UNK>'] is returned, the indicating that the element leaves
        the scoreboard in an unknown state, and it may unpredicatably fail if
        run: this will cause validation to fail. Note that `_reqs` (but not
        `arg_keys`) can be empty.
        """
        return [self.out_key] if self.out_key else self._adds  if hasattr(self, '_adds') else ['<UNK>']
    
    @property
    def deletes(self) -> list[str]:
        """The columns which the ScoreboardPipelineElement removes from the 
        scoreboard when run. If the element has values for `arg_keys` and 
        `out_key`, either we are in the `Renaming Mask` condition (if there is 
        no `fn`), and `out_key` will be removed from the scoreboard (the column 
        remains, but has a new name), or `fn` exists, and we are in the Standard
        Case, which does not remove columns. Otherwise, we are in the Other 
        Case, in which case either `_dels` has been set, and can be returned, or
        ['<UNK>'] is returned, indicating that the element leaves the scoreboard 
        in an unknown state, and it may unpredicatably fail if run: this will 
        cause validation to fail. 
        """
        if self.out_key and self.arg_keys:  
            return self.arg_keys if self.fn is None else []
        else:
            return self._dels if hasattr(self, '_dels') else ['<UNK>']
        
    @property
    def __name__(self):
        """The name of the element is given by the function `fn`, unless `fn` is
        None, in which case it is 'rename'
        """
        return 'rename' if self.fn is None else self.fn.__name__
        
    def _set_validators(self, 
            reqs: str|list[str], 
            adds: str|list[str], 
            dels: str|list[str]
        ):
        """If no calues are given for the 'key' attributes, the element must
        be hard-coded with values for it to pass during validation. Rules are 
        specified below for how to handle cases where the changes to the set of
        columns in the scoreboard are non-deterministic. The goal is always to 
        ensure that the pipeline provides each element with the columns it cannot
        function without. If this cannot be guaranteed, validation should fail.
        
        Parameters
        ==========
            reqs (string or list of strings):
                Hard-coded value for property `requires`. Note that `reqs` 
                should only be the columns the element cannot reliably function
                without: columns which may optionally be used, if present
                should not be included
            adds (string or list of strings):
                Hard-coded value for property `adds`. Note that this should only
                be used for columns that the element will definitely always add;
                if a function sometimes adds a column, but not always, it should
                not be included
            dels (string or list of strings):
                Hard-coded value for property `requires`. Any column which might
                be deleted, even if it is not certain that it will be, should be 
                included
        """
        # This is a setter for three attributes at once, because if one of then 
        # needs set, all of them do. This does mean using looping over both 
        # attribute names (`attrname`) and the args to be set with that attribute 
        # (`arg`), which I've put together in tuples - then using `setattr` like 
        # a filthy pervert, rather than using a normal assignment the way god 
        # intended. I'm not sorry. Fuck you.
        for attrname, arg in (('_reqs', reqs), ('_adds', adds), ('_dels', dels)):
            # Make sure the attr is set as [] if arg is none or empty
            setattr(self, attrname, collect(arg, list, empty_if_none=True))


def scoreboard_pipeline_element(arg_keys=('tree'), out_key='fitness', vec=False):
    """A double-decker decorator, which converts the decorated function to a
    ScoreboardPipelineElement, which is callable. The decorator must be provided
    with arguments, which are all the attributes of the 
    ScoreboardPipelineElement *except* `fn`, which will be the decorated 
    function. It's a 'double decker' because it returns another decorator, which
    is what actually decorates the decorated function.

    Functions decorated with this decorator should be listed by name in the 
    pipeline list in GPScoreboard, and should not be called there.

    Parameters
    ==========
        arg_keys (string or list of strings)
        out_key (string)
        vec (bool)
    """
    # Outer decorator is called with the arguments and returns the inner 
    # decorator
    def spe_decorator(fn: Callable):
        # which returns the ScoreboardPipelineElement, which takes the
        # place of the decorated function in the namespace 
        return ScoreboardPipelineElement(arg_keys=arg_keys, out_key=out_key, vec=vec, fn=fn)
    # return the decorator that actually takes the function in as the input
    return spe_decorator

def flexi_pipeline_element(vec=True, out_key_maker: Callable = None, **fpe_kwargs):
    """Kind of a triple-decker. That is, it's a double-decker decorator again,
    but with the output being a function which, given some more arguments,
    returns a ScoreboardPipelineElement, with the following values:
    
    `vec`:      given as an argument to the decorator 
    `fn`:       is the function generated by the function generated by the inner
                decorator, and is a wrapper for the decorated function. This 
                differs only in name and names of arguments from the decorated
                function, but this is needed to ensure the 
                `ScoreboardPipelineElement` uses with the correct columns
    `out_key`   is generated by a function, `out_key_maker`, which is passed as  
                an argument to the decorator, *unless* the function the inner 
                decorator returns is given an overriding value as `out_key`, or 
                neither are provided, in which case a value is generated by 
                joining the name of the decorated function and the `arg_keys`
                all with underscores (e.g. `add_x_y`)
    `arg_keys`  are passed as `*arg_keys` to the function the inner decorator
                outputs. The function takes the `arg_keys` and uses `exec` and
                an f-string to make a function which wraps the decorated 
                function. `arg_keys` must be the same length as the number of
                positional arguments the decorated function takes

    The purpose here is to allow a base function which may be used with 
    different columns to be used to generate multiple 
    `ScoreboardPipelineElement`s, differing based on which columns they handle.
    
    To use functions decorated with `@flexi_pipeline_element` in the pipeline 
    list, a call to the function should be placed in the list, not the bare 
    name, like `@scoreboard_pipeline_element` functions. This should take the
    same args as `spe_maker` (below), *not* those of the decorated function.

    Parameters
    ==========
        vec (bool): The 'vec' attribute of the ScoreboardPipelineElement
        out_key_maker (function): Function to generate the value of the 
            `out_key` attribute of the ScoreboardPipelineElement

    Parameters of the Decorator-Created Function
    ============================================
        *arg_keys (string): The 'arg_keys' attribute of the 
            ScoreboardPipelineElement, which are mapped in sequence to the 
            positional arguents of the wrapped function
        `out_key` (string): `out_key` attribute of the 
            ScoreboardPipelineElement, overriding `out_key_maker`
    """
    # Outer decorator returns the inner decorator
    def fpe_decorator(fn: Callable):
        # For reasons which will become clear in a few lines time,
        # it is necessary to ensure the function below, `spe_maker`, 
        # has an attribute `__wrapper__` that refers to the function
        # the call to `flexi_pipeline_element` decorates 
        @wraps(fn)
        def spe_maker(*arg_keys, out_key=None, **spem_kwargs):
            # Sets the value for `out_key` - using the spe_maker override if
            # available, else a generated value from `out_key_maker`, else
            # a default made from the function and argument names
            out_key = out_key if out_key else out_key_maker(arg_keys) if out_key_maker else f"{fn.__name__}_{'_'.join(arg_keys)}"
            # To keep the f-string that creates the wrapper function simple,
            # the string representing the argumenrs of the function is 
            # generated here
            argstr = ', '.join(arg_keys)
            # kwargstr = (
            #     ', ' if spem_kwargs else ''
            # ) + (
            #     ', '.join([f'{k} = {k}' for k in spem_kwargs.keys()])
            # )
            # The code for the wrapper function is generated using an f-string,
            # and run using exec. This adds a function to the local namespace,
            # with a name equal to the value of `out_key`
            # Note that in the global namespace, the function name in the line
            # below does not refer to the function `fn` the decorator wraps, 
            # but the decorated function, which is underlyingly `spe_maker`.
            # To ensure the correct behaviour, `spe_maker` was decorated with
            # functools.wraps, which ensures that when the `fpe_decorator`
            # maps the name of the decorated function to an instance of
            # `spe_maker` in the global namespace, the function created by the
            # `exec` call below can use `__wrapped__` to access the decorated
            # function, not `spe_maker`, which is the desired behaviour  
            # if 'n_minus' in out_key:
            #     with open('log.txt', 'a') as log:
            #         log.write(
            #             f"def {out_key}({argstr}, **kwargs):\n" +
            #             # ''.join([f"\t{key} = {repr(arg)}\n" for key in spem_kwargs.keys()]) +
            #             f"\treturn {fn.__name__}.__wrapped__({argstr}{kwargstr}, **kwargs)\n" +
            #             str(locals()['spem_kwargs']) + '\n'
            #         )
            exec(
                # "with open('log2.txt', 'a') as log2:\n"+
                # "\tlog2.write(str(locals()['spem_kwargs']) + '\\n')\n" +
                f"def {out_key}({argstr}, **kwargs):\n" +
                # ''.join([f"\t{key} = {repr(arg)}\n" for key in spem_kwargs.keys()]) +
                # f"\treturn {fn.__name__}.__wrapped__({argstr}{kwargstr}, **kwargs)\n",
                f"\treturn {fn.__name__}.__wrapped__({argstr}, **kwargs)\n",
            )
            # Note `locals()` is the dict of all local variables, so
            # `locals()[out_key]` retrieves the function generated by `exec`
            return ScoreboardPipelineElement(
                arg_keys=arg_keys, 
                out_key=out_key, 
                fn=locals()[out_key],
                vec=vec,
                spem_kwargs=spem_kwargs
            )
        return spe_maker
    return fpe_decorator

# @flexi_pipeline_element()
# def has_var(tree: GPNonTerminal, var='x'):

@flexi_pipeline_element(out_key_maker=lambda args: f"n_minus_{args[0]}") #, n=0)
def n_minus(arg, **kwargs):
    """Calling `n_minus(arg, n=k)` generates a ScoreboardPipelineElement which 
    subtracts scoreboard column `arg` from `k`.
    """
    return kwargs['n']-arg

@flexi_pipeline_element(out_key_maker=lambda args: 'ws_' + ('_'.join(args)))
def weighted_sum(*args, **kwargs):
    return sum([w * x for w, x in zip(kwargs['weights'], args)])


@flexi_pipeline_element(out_key_maker=lambda args: f"i{args[0]}")
def safe_inv(arg, **kwargs):
    """Multiplicative inversion operator, protected from ZeroDivisionErrors
    by adding 1 to the denominator. Generates the name of the new column by
    prepending `i` to the name of the old. Pass the name of the column you 
    want to invert to get the ScoreboardPipelineElement you need. e.g.: 
    `safe_inv('rmse')` gives an element `irmse` that inverts `rmse` 
    """
    return 1/(arg+1)

def get_errs(tree: GPNonTerminal, target: pd.Series=None, ivs=None) -> pd.Series:
    ests = tree(**ivs)
    errs = target - ests
    nans = np.isnan(ests).sum()
    if nans:
        tree.tmp['hasnans'] = True
        if nans <= np.round(len(ests) * 0.04):
            tree.tmp['penalty'] = tree.tmp.get('penalty', 1.0) * 2.0**nans
        else:
            tree.tmp['survive'] = False
    return errs

@scoreboard_pipeline_element(out_key='mse')
def mse(tree: GPNonTerminal, target: pd.Series=None, ivs=None, **kwargs):
    """Calculates the Mean Squared Error of tree outputs. Adds `mse`, 
    requires `tree`
    """
    return (get_errs(tree, target=target, ivs=ivs)**2).mean()

@scoreboard_pipeline_element(out_key='sae')
def sae(tree: GPNonTerminal, target: pd.Series=None, ivs=None, **kwargs):
    """Calculates the Summed Average Error of tree outputs. Adds `sae`, 
    requires `tree`
    """
    return (get_errs(tree, target=target, ivs=ivs)).abs().mean()

@scoreboard_pipeline_element(arg_keys="mse", out_key="rmse", vec=True)
def rmse(mse: float, **kwargs):
    """Root Mean Squared Error: adds 'rmse', requires 'mse'"""
    return mse**0.5

def rename(old_name: str, new_name: str):
    """Creates a ScoreboardPipelineElement which renames a column. Put a 
    call to this in the pipeline list, with params `old_name` and 
    `new_name` for the old and new names.
    """
    return ScoreboardPipelineElement(arg_keys=old_name, out_key=new_name)

@scoreboard_pipeline_element(arg_keys="raw_fitness", out_key="fitness", vec=True)
def heat(raw_fitness: float, temp_coeff: float, **kwargs):
    """Calculates the temperateure of the GP system as the product of the average
    fitness and `temp_coeff`, which is added to the raw fitness column. This boosts
    the chances of lower-fitness trees to reproduce, maintaining a bit of diversity
    in the treebank
    """
    if temp_coeff:
        temp = raw_fitness.mean() * temp_coeff
    else:
        # No need to compute the mean is temp_coeff is 0
        temp = 0.0
    try:
        return raw_fitness + temp
    except Exception as e:
        print('UW'*49+"U")
        print(temp_coeff)
        print(raw_fitness)
        print(temp)
        print('OW'*49+"O")
        raise e

def from_tmp(varname: str, default=None):
    """ Outputs metadata attached to trees into a scoreboard column. 
    Common examples:

    Examples
    --------
        Penalty: Trees receive penalties during processing for things like  
            OverflowErrorsand ZeroDivisionErrors. Penalties are applied by 
            dividing the tree's fitness, so the default 'no penalty' value 
            is zero
        Kill: A tree may also be flagged as generating problems, and to be
            removed from the population. For example, this may be done if
            the tree .generates excessive NaN outputs
        Hasnans: Indicates that the tree generated NaNs
    """
    @scoreboard_pipeline_element(out_key=varname)
    def get_tmp(tree: GPNonTerminal, **kwargs):
        return tree.tmp.get(varname, default)
    return get_tmp

penalty = from_tmp('penalty', default=1)
survive = from_tmp('survive', default=True)
hasnans = from_tmp('hasnans', default=False)

# @scoreboard_pipeline_element(out_key='penalty')
# def penalty(tree: GPNonTerminal, **kwargs):
#     """Trees receive penalties during processing for things like OverflowErrors 
#     and ZeroDivisionErrors. Penalties are applied by dividing the tree's fitness,
#     so the default 'no penalty' value is zero
#     """
#     return tree.metadata.get('penalty', 1.0)

# @scoreboard_pipeline_element(out_key='kill')
# def kill(tree: GPNonTerminal, **kwargs):
#     return tree.metadata.get('kill', False)

@flexi_pipeline_element()
def divide(num: float, denom: float, **kwargs):
    """Divides one column by another, and replaces any `inf` values (from dividing 
    by zero) with NaNs
    
    I wrote this one to apply penalties to trees, but do whatever with it.
    """
    return (num/denom).replace([np.inf, -np.inf], np.nan)

@flexi_pipeline_element()
def multiply(a: float, b: float, **kwargs):
    """multiplies one column by another.
    
    I wrote this one to apply penalties to trees, but do whatever with it.
    """
    return a*b

@flexi_pipeline_element()
def nan_zero(a: float, **kwargs):
    return a.fillna(0)

@scoreboard_pipeline_element(out_key='size')
def size(tree: GPNonTerminal, **kwargs):
    try:
        return tree.size()
    except Exception as e:
        print("."*500)
        print(tree)
        print("."*500)
        raise e

@scoreboard_pipeline_element(out_key='depth')
def depth(tree: GPNonTerminal, **kwargs):
    return tree.depth()

def clear(*to_clear: str, except_for: str|list[str]=None):
    """A function called to generate a ScoreboardPipelineElement which deletes 
    columns from the scoreboard. 

    Parameters
    ==========
        *to_clear (string): The names of all the columns to be removed. Clears all
            if empty, unless a value is given for `except_for`.
        `except_for` (string or list of strings): The names of all columns to be 
            spared: all other columns will be deleted. Raises a ValueError if 
            provided alongside positional arguments
    """
    to_clear = list(to_clear)
    if except_for:
        if to_clear:
            raise ValueError(
                "The `except_for` keyword arg should only be used with clear" +
                " if no positional args are passed: if no positional args are" +
                "passed, all columns are clear, except for those in `except_for`"
            )
        except_for = collect(except_for, list)
    else:
        except_for = []
    @scoreboard_pipeline_element(arg_keys=[], out_key="")
    def clr(sb: GPScoreboard, **kwargs):
        if not to_clear:
            # If `to_clear_` not provided, use the complete list of columns.
            # The new variable name gets an underscore, because if I assign to 
            # `to_clear` here, Python will think this is a new variable in
            # the inner function namespace, and then get confused by the
            # line *above*, thinking it's referring to the as-yet-non-value
            # -bearing local *below*, and not the variable in the outer 
            # function namespace. <facepalm/>
            _to_clear = list(sb.columns)
            # Remove any exceptions listed in `except_for`
            for exn in except_for:
                if exn in _to_clear:
                    _to_clear.remove(exn)
        # Drop all columns named in `_to_clear` or `to_clear`
        sb.drop(columns=to_clear if to_clear else _to_clear, inplace=True)
        return sb
    # Element cannot infer changes to scoreboard, so specify with `_reqs`, `_adds`
    # and `_dels`. This adds no columns, removes the cols named in `to_clear` (if 
    # given), and requires them. If `to_clear` is not given, specify '<ALL>' in 
    # `_dels`, along with any exceptions, prepended with `-`
    clr._set_validators(
        reqs=to_clear, 
        adds=[], 
        dels = to_clear if to_clear else ['<ALL>'] + [f"-{exn}" for exn in except_for])
    return clr

class SimpleGPScoreboardFactory:
    def __init__(self, best_outvals, dv, def_fitness: str='irmse'):
        self.def_fitness = def_fitness
        self.best_outvals = best_outvals
        self.dv = dv

    def __call__(self, observatory: Observatory, temp_coeff, weights=None):
        self.observatory = observatory
        _weights = weights if weights is not None else [1, 0, 0]
        temp_coeff = temp_coeff.item() if isinstance(temp_coeff, torch.Tensor) else temp_coeff
        sb_kwargs = {
            "temp_coeff": temp_coeff,
            "obs": self.observatory,
            "weights": _weights
        }
        pipeline          = [clear(except_for='tree'), size, depth]
        match self.def_fitness:
            case 'imse':
                pipeline += [mse]
            case 'irmse':
                pipeline += [mse, rmse]
            case 'isae':
                pipeline += [sae]
        pipeline         += [safe_inv(self.def_fitness[1:])]
        if weights is not None:
            pipeline     += [
                                safe_inv('size'), 
                                safe_inv('depth'), 
                                weighted_sum(self.def_fitness, 'isize', 'idepth')
                            ]
            def_2_fitness = f"ws_{self.def_fitness}_isize_idepth"
        else: 
            def_2_fitness = self.def_fitness
        if temp_coeff:
            pipeline     += [rename(def_2_fitness, 'raw_fitness'), heat, rename("fitness", 'pre_fitness_1')]
        else:
            pipeline     += [rename(def_2_fitness, 'pre_fitness_1')]
        pipeline         += [
                                hasnans, 
                                penalty, 
                                divide('pre_fitness_1', 'penalty', out_key='pre_fitness_2'),
                                survive, 
                                multiply('pre_fitness_2', 'survive', out_key='pre_fitness_3'),
                                nan_zero('pre_fitness_3', out_key='fitness'),  
                            ]
        def_outputs = collect(self.best_outvals, list) if self.best_outvals else None
        dv = self.dv # XXX JAAAAAANK
        for arg, val in (
            ('def_outputs', def_outputs), 
            ('dv', self.dv)
        ):
            val = eval(arg)
            if val:
                sb_kwargs[arg] = val 
        return GPScoreboard(*pipeline, **sb_kwargs)

class GPScoreboard(pd.DataFrame):
    """Extension to pandas.Dataframe to function as a scoreboard for GP,
    with a pipeline for fitness and record-keeping calculations

    >>> df = pd.DataFrame({'x': [0,1,2,3,4,5,6,7,8,9,10]})
    >>> n = n_minus('x', n=10)
    >>> ws = weighted_sum('x', 'n_minus_x', weights=[1,2])
    >>> n(df)
    >>> ws(df)
    >>> print(df)
         x  n_minus_x  ws_x_n_minus_x
    0    0         10              20
    1    1          9              19
    2    2          8              18
    3    3          7              17
    4    4          6              16
    5    5          5              15
    6    6          4              14
    7    7          3              13
    8    8          2              12
    9    9          1              11
    10  10          0              10
    """

    _metadata = [
        'obs', '_pipeline', 'pipeline', 'kwargs', 'provide', 'require', 'def_outputs'
    ] # Reserve these variable names so pandas doesn't think I'm trying to create
    # a column

    # XXX TODO Pipeline visualiser
    # TODO Scoreboard configurator

    def __init__(
            self,
            *pipeline: ScoreboardPipelineElement,
            def_fitness: str=None,
            dv: str=None,
            obs: Observatory = None, 
            temp_coeff: float = 0.0,
            provide: str|Collection[str]='tree', 
            require: str|Collection[str]='fitness',
            def_outputs: str|Collection[str]=None,
            # to_graph: Collection[str] = None,
            **kwargs
        ) -> None:
        super().__init__()
        self.obs = None
        dv = self.dv_obs_check(dv=dv, obs=obs)
        if def_fitness:
            if pipeline:
                raise ValueError(
                    "Don't pass a value to def_fitness if you are" +
                    " also passing pipeline elements as positional" +
                    " arguments"
                )
            if def_fitness.lower() not in ['isae', 'imse', 'irmse']:
                raise ValueError(
                    "def_fitness is used to chose between the three" +
                    "standardly provided fitness functions:\n" +
                    "    ISAE:   Inverse Sum of Average Errors\n" +
                    "    IMSE:   Inverse Mean Squared Error\n" +
                    "    IRMSE:  Inverse Root Mean Squared Error\n" +
                    "If you wish to use something else, you can " +
                    "create your own ScoreboardPipelineElement for" +
                    " it."
                )
        def_fitness = def_fitness if def_fitness else 'irmse'
        self.provide = collect(provide, set)
        self.require = collect(require, set)
        self.kwargs = {"temp_coeff": temp_coeff, **kwargs}
        self.def_outputs = collect(def_outputs, set)
        # This runs the pipeline property setter, which validates the pipeline
        # and updates the attribute `def_outputs` to to list the set of columns
        # in the final dataframe
        self.pipeline = pipeline if pipeline else tuple([
            clear(except_for='tree'),
            mse, rmse, safe_inv('rmse'), size, depth, 
        ] + (
            [rename(def_fitness, 'raw_fitness'), heat, rename("fitness", 'pre_fitness_1')]
            if temp_coeff else
            [rename(def_fitness, 'pre_fitness_1')]
        ) + (
            [
                hasnans,
                penalty, divide('pre_fitness_1', 'penalty', out_key='pre_fitness_2'),
                survive, multiply('pre_fitness_2', 'survive', out_key='pre_fitness_3'),
                nan_zero('pre_fitness_3', out_key='fitness'),  
            ]
        ))
        # We can then use this to validate the list of variables to be graphed
        # by Grapher


    @property
    def pipeline(self):
        return self._pipeline
    
    @pipeline.setter
    def pipeline(self, pipeline: tuple[ScoreboardPipelineElement]):
        self._pipeline = pipeline
        self._post_process_pipeline()

    # XXX TODO XXX This is good enough for my first tests, but needs to generalise
    @property
    def sb_params(self):
        return {
            'temp_coeff': self.kwargs['temp_coeff'],
            'wt_fitness': self.kwargs['weights'][0],
            'wt_size': self.kwargs['weights'][1],
            'wt_depth': self.kwargs['weights'][2]
        } 
    

    def _post_process_pipeline(self):
        final_cols, error = self._validate_pipeline()
        if error:
            raise ValueError(str(error) + ((
                ". Also, _validate_pipeline returned both success and error " +
                "values, which makes no sense. the success value was " +
                f"'{final_cols}'"
            ) if final_cols else ""))
        elif self.def_outputs:
            for d_o in self.def_outputs:
                if d_o not in final_cols:
                    print('dshfs'*20)
                    print(d_o)
                    print(final_cols)
                    print("as"+ ('A'*60) + 'sa')
                    raise ValueError(
                        "A value listed as a default scoreboard output, " +
                        f"{d_o}, is not in the final outputs of the tree" +
                        f"scoring pipeline. Valid values are: {final_cols}"
                    )
        else:
            self.def_outputs = final_cols

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
            if reqs is None:
                print(p, "has reqs=None")
            adns = p.adds
            if adns is None:
                print(p, "has adns=None")
            dlns = p.deletes
            if dlns is None:
                print(p, "has dlns=None")
            if "<UNK>" in reqs:
                return None, [format_error(
                    p, i, "Unknown requirements"
                )]
            for req in reqs:
                if req not in cols:
                    return None, [format_error(
                        p, i, f"Required input '{req}' not found"
                    )]
            if "<UNK>" in dlns:
                return None, [format_error(
                    p, i, 
                    "Scoreboard in unknown state, specify " +
                    "deletions for this element"
                )]
            if "<ALL>" in dlns:
                new_cols = set()
                if len(dlns) > 1:
                    for dln in dlns:
                        if dln.startswith('-'):
                            if dln[1:] in cols:
                                new_cols.add(dln[1:])
                        elif dln != '<ALL>':
                            return None, [format_error(
                                p, i, 
                                "Element clears all columns," +
                                "but has other deletions " +
                                f"({dln}) listed"
                            )]
                cols = new_cols
            else:
                for dln in dlns:
                    if dln not in reqs:
                        return None, [format_error(
                            p, i, 
                            f"Element deletes column '{dln}'," +
                            f" but does not specify '{dln}' " +
                            "as a requirement"
                        )]
                    if dln in adns:
                        return None, [format_error(
                            p, i, 
                            "Scoreboard in unknown state, " +
                            f"element specifies '{dln}' as " +
                            "both an addition and deletion"
                        )]
                    cols.remove(dln)
            if "<UNK>" in adns:
                return None, [format_error(p, i, 
                    "Scoreboard in unknown state, specify " +
                    "additions for this element"
                )]
            for adn in adns:
                cols.add(adn)
        for rqmt in self.require:
            if rqmt not in cols:
                return None, [format_error(
                    p, -1, 
                    "Pipeline completes, but required " +
                    f"column {rqmt} is not in scoreboard"
                )]
        return list(cols), None
    
    # TODO XXX make this a setter, self.obs; make a getter
    def dv_obs_check(self, dv: str=None, obs: Observatory=None):
        self.obs = self.obs if obs is None else obs
        if not self.obs:
            if dv:
                raise ValueError(
                    "Don't set `dv` if you haven't set `obs`: `dv`" +
                    " is only used if obs is set and has multiple DVs"
                )
        elif len(self.obs.dvs)>1:
            if not (dv and dv in obs.dvs):
                raise ValueError(
                    f"DV {dv} not found in Observatory dvs"
                    if dv else
                    f"Observatory obs has multiple DVs, but no kwarg `dv` is given to specify which one"
                )
            return dv
        else:
            return self.obs.dvs[0]


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
        dv = self.dv_obs_check(dv=dv, obs=obs)
        kwargs['ivs'] = next(self.obs)
        kwargs['target'] = self.obs.target[dv]
        try:
            for pipe_ele in self.pipeline:
                pipe_ele(self, **{**self.kwargs, **kwargs})
        except Exception as e:
            print("',"*250)
            print(self)
            print("',"*250)
            raise e
        return self
    
    def k_best(self, k: int, mark_4_del: bool=True, tree_only: bool=True):
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
        >>> from test_materials import DummyTreeFactory
        >>> op = [ops.SUM, ops.PROD]
        >>> gp = GPTreebank(operators = op, tree_factory=DummyTreeFactory())
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
        ...     print([tt.tmp["to_delete"] for tt in t])
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
        >>> print([tt.tmp.get("to_delete", None) for tt in gps['tree']])
        [False, False, False, False, False, None, None]
        >>> print(best[0]()) 
        21.0
        """
        best = self.nlargest(k, 'fitness')
        if tree_only:
            best = best['tree']
        # else:
        #     ic(best)
        if mark_4_del:
            self['tree'].apply(lambda t: t.tmp.__setitem__('to_delete', True))
            best.apply(lambda t: t.tmp.__setitem__('to_delete', False))
        return list(best) if tree_only else best.reset_index()

    def winner(self, *cols, except_for: str|list[str]=None, **kwargs)-> dict: 
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
        >>> from gp import GPTreebank
        >>> from observatories import StaticObservatory
        >>> from test_materials import DummyTreeFactory
        >>> op = [ops.PROD, ops.EQ, ops.TERN_FLOAT, ops.GT]
        >>> gp = GPTreebank(operators = op, temp_coeff=0.5, tree_factory=DummyTreeFactory)
        >>> iv = [0., 2., 4., 6., 8.,]
        >>> dv0 = [0., 0., 0., 0., 0.]
        >>> dv1 = [0., 10., 20., 30., 40.]
        >>> iv2 = np.linspace(0, 2, 129)
        >>> dv2 =  np.linspace(0.015625, 2.015625, 129)
        >>> df = pd.DataFrame({'x': iv, 'y0': dv0, 'y1': dv1})
        >>> df2 = pd.DataFrame({'x': iv2, 'y': dv2})
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
        >>> obs2 = StaticObservatory('x', 'y', sources=df2, obs_len=129)
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
        >>> gps2 = GPScoreboard(obs=obs2, temp_coeff=0.015624999999999944) # temp_coeff picked so pre_fitness_1 would be 1.0
        >>> for t_ in t:
        ...     t_.delete()
        >>> wrong = [
        ...     gp.tree("([float]<PROD>([float]$x)([float]np.nan))"),
        ...     gp.tree("([float]<PROD>([float]$x)([float]<TERN_FLOAT>([bool]<EQ>([float]$x)([float]1.))([float]1.0)([float]np.nan)))"),
        ...     gp.tree("([float]<PROD>([float]$x)([float]<TERN_FLOAT>([bool]<EQ>([float]$x)([float]1.))([float]np.nan)([float]1.0)))"),
        ...     gp.tree("([float]<PROD>([float]$x)([float]<TERN_FLOAT>([bool]<GT>([float]$x)([float]1.92))([float]np.nan)([float]1.0)))"),
        ...     gp.tree("([float]<PROD>([float]$x)([float]<TERN_FLOAT>([bool]<GT>([float]$x)([float]1.93))([float]np.nan)([float]1.0)))"),
        ...     gp.tree("([float]<PROD>([float]$x)([float]<TERN_FLOAT>([bool]<GT>([float]$x)([float]1.95))([float]np.nan)([float]1.0)))"),
        ...     gp.tree("([float]<PROD>([float]$x)([float]<TERN_FLOAT>([bool]<GT>([float]$x)([float]1.96))([float]np.nan)([float]1.0)))"),
        ...     gp.tree("([float]<PROD>([float]$x)([float]<TERN_FLOAT>([bool]<GT>([float]$x)([float]1.98))([float]np.nan)([float]1.0)))"),
        ...     gp.tree("([float]<PROD>([float]$x)([float]<TERN_FLOAT>([bool]<GT>([float]$x)([float]1.99))([float]np.nan)([float]1.0)))"),
        ...     gp.tree("([float]<PROD>([float]$x)([float]1.0))")
        ... ]
        >>> cols = list(gps2(wrong).columns)
        >>> cols[:7]
        ['tree', 'mse', 'rmse', 'raw_fitness', 'size', 'depth', 'pre_fitness_1']
        >>> cols[7:]
        ['hasnans', 'penalty', 'pre_fitness_2', 'survive', 'pre_fitness_3', 'fitness']
        >>> gps2[cols[1:7]]
                mse      rmse  raw_fitness  size  depth  pre_fitness_1
        0       NaN       NaN          NaN     3      2            NaN
        1  0.000244  0.015625     0.984615     8      4            1.0
        2  0.000244  0.015625     0.984615     8      4            1.0
        3  0.000244  0.015625     0.984615     8      4            1.0
        4  0.000244  0.015625     0.984615     8      4            1.0
        5  0.000244  0.015625     0.984615     8      4            1.0
        6  0.000244  0.015625     0.984615     8      4            1.0
        7  0.000244  0.015625     0.984615     8      4            1.0
        8  0.000244  0.015625     0.984615     8      4            1.0
        9  0.000244  0.015625     0.984615     3      2            1.0
        >>> gps2[cols[7:]] 
           hasnans  penalty  pre_fitness_2  survive  pre_fitness_3  fitness
        0     True      1.0            NaN    False            NaN  0.00000
        1     True      1.0        1.00000    False        0.00000  0.00000
        2     True      2.0        0.50000     True        0.50000  0.50000
        3     True      1.0        1.00000    False        0.00000  0.00000
        4     True     32.0        0.03125     True        0.03125  0.03125
        5     True     16.0        0.06250     True        0.06250  0.06250
        6     True      8.0        0.12500     True        0.12500  0.12500
        7     True      4.0        0.25000     True        0.25000  0.25000
        8     True      2.0        0.50000     True        0.50000  0.50000
        9    False      1.0        1.00000     True        1.00000  1.00000
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
        elif except_for:
            cols = list(self.columns)
            ef = collect(except_for, list)
            for exn in ef:
                if exn in cols:
                    cols.remove(exn)
        else:
            cols = deepcopy(self.def_outputs)
        best = self.nlargest(1, 'fitness')[list(cols)]
        best_dic = {}
        for col in cols:
            if col=="tree":
                best_dic[col] = str(best[col].item())
            else:
                best_dic[col] = best[col].item()
        return best_dic
    
    def score_trees(
            self,
            trees: Iterable[GPNonTerminal],  
            *cols: str,
            **kwargs
        )-> dict:
        return self(trees, **kwargs).winner(*cols, **kwargs)

# class SimpleScoreboardFactory(Actionable):
#     """Generates scoreboards for RL-generated GP runs

#     >>> import operators as ops
#     >>> from gp import GPTreebank
#     >>> from observatories import StaticObservatory
#     >>> from test_materials import DummyTreeFactory
#     >>> op = [ops.PROD, ops.EQ, ops.TERN_FLOAT, ops.GT]
#     >>> gp = GPTreebank(operators = op, temp_coeff=0.5, tree_factory=DummyTreeFactory())
#     >>> iv = [0., 2., 4., 6., 8.,]
#     >>> dv0 = [0., 0., 0., 0., 0.]
#     >>> dv1 = [0., 10., 20., 30., 40.]
#     >>> df = pd.DataFrame({'x': iv, 'y0': dv0, 'y1': dv1})
#     >>> t = [
#     ...     gp.tree("([float]<PROD>([float]$x)([int]1))"),
#     ...     gp.tree("([float]<PROD>([float]$x)([int]3))"),
#     ...     gp.tree("([float]<PROD>([float]$x)([int]5))"),
#     ...     gp.tree("([float]<PROD>([float]$x)([int]7))"),
#     ...     gp.tree("([float]<PROD>([float]$x)([int]9))"),
#     ...     gp.tree("([float]<PROD>([float]6.)([int]6))")
#     ... ]
#     >>> obs0 = StaticObservatory('x', 'y0', sources=df, obs_len=5)
#     >>> obs1 = StaticObservatory('x', 'y1', sources=df, obs_len=5)
#     >>> ssf = SimpleScoreboardFactory(use_irmse=True)
#     >>> params = np.array([0.5, 1., 2., 3., 0.04, 0.05], dtype=np.float32) 
#     >>> sb0 = ssf.act(params, obs0, max_size=20, max_depth=8, best_outvals='fitness', dv='y')
#     >>> cols = list(sb0(t).columns) #[list(sb0.columns)[1:]]
#     >>> (cols[:7])
#     ['tree', 'size', 'depth', 'mse', 'imse', 'rmse', 'irmse']
#     >>> cols[7:13]
#     ['sae', 'isae', 'n_minus_size', 'n_minus_depth', 'raw_fitness', 'pre_fitness_1']
#     >>> cols[13:]
#     ['hasnans', 'penalty', 'pre_fitness_2', 'survive', 'pre_fitness_3', 'fitness']
#     >>> sb0[cols[1:7]]
#        size  depth     mse      imse       rmse     irmse
#     0     3      2    24.0  0.040000   4.898979  0.169521
#     1     3      2   216.0  0.004608  14.696938  0.063707
#     2     3      2   600.0  0.001664  24.494897  0.039224
#     3     3      2  1176.0  0.000850  34.292856  0.028334
#     4     3      2  1944.0  0.000514  44.090815  0.022177
#     5     3      2  1296.0  0.000771  36.000000  0.027027
#     >>> sb0[cols[7:13]]
#         sae      isae  n_minus_size  n_minus_depth  raw_fitness  pre_fitness_1
#     0   4.0  0.200000            17              6     1.959042       2.614677
#     1  12.0  0.076923            17              6     1.342791       1.998426
#     2  20.0  0.047619            17              6     1.202968       1.858603
#     3  28.0  0.034483            17              6     1.140967       1.796602
#     4  36.0  0.027027            17              6     1.105950       1.761585
#     5  36.0  0.027027            17              6     1.115906       1.771541
#     >>> sb0[cols[13:]]
#        hasnans  penalty  pre_fitness_2  survive  pre_fitness_3   fitness
#     0    False        1       2.614677     True       2.614677  2.614677
#     1    False        1       1.998426     True       1.998426  1.998426
#     2    False        1       1.858603     True       1.858603  1.858603
#     3    False        1       1.796602     True       1.796602  1.796602
#     4    False        1       1.761585     True       1.761585  1.761585
#     5    False        1       1.771541     True       1.771541  1.771541
#     """
#     def __init__(self, 
#             use_irmse: bool=False,
#             use_imse: bool=True,
#             use_isae: bool=True,
#             use_size: bool=True,
#             use_depth: bool=True,
#             # other_measures: dict[str, ScoreboardPipelineElement]=None, XXX later
#             # use_others: dict[str, bool]=None
#         ):
#         self.use_imse = use_imse
#         self.use_irmse = use_irmse
#         self.use_isae = use_isae
#         self.use_size = use_size
#         self.use_depth = use_depth
#         self.fitness_factors = []
#         num_measures = sum(
#             [
#                 self.use_irmse, self.use_imse, self.use_isae, 
#                 self.use_size, self.use_depth
#             ] # + list(use_others.values())
#         )
#         self.pipeline_setup = [clear(except_for='tree'), size, depth]
#         if self.use_imse or self.use_irmse:
#             self.pipeline_setup += [mse] 
#             if self.use_imse:
#                 self.pipeline_setup += [safe_inv('mse')]
#                 self.fitness_factors += ['imse']
#             if self.use_irmse:
#                 self.pipeline_setup += [rmse, safe_inv('rmse')]
#                 self.fitness_factors += ['irmse']
#         if self.use_isae:
#             self.pipeline_setup += [sae, safe_inv('sae')]
#             self.fitness_factors += ['isae']
#         self.pipeline_post = [
#             hasnans,
#             penalty, 
#             divide('pre_fitness_1', 'penalty', out_key='pre_fitness_2'),
#             survive, 
#             multiply('pre_fitness_2', 'survive', out_key='pre_fitness_3'),
#             nan_zero('pre_fitness_3', out_key='fitness'),  
#         ]
#         if num_measures == 0:
#             raise ValueError('Scoreboard uses no fitness measures at all!')
#         elif num_measures == 1:
#             num_measures = 0
#         self._act_param_space = Box(              # First term is temp_coeff, the rest are 
#             low=np.array([0.0]*(num_measures+1), dtype=np.float32),           # weights on fitness measures: however,
#             high=np.array([np.inf] + ([1.0]*num_measures), dtype=np.float32), # if only one measure is used no weights
#             dtype=np.float32)                     # are needed

#     def act(self, 
#             params: np.ndarray, 
#             observatory: Observatory, 
#             max_size: int=0, 
#             max_depth: int=0,
#             best_outvals: str|list[str]=None,
#             dv: str=None
#         ):
#         sb_kwargs = {
#             "temp_coeff": params[0],
#             "obs": observatory,
#         }
#         def_outputs = collect(best_outvals, list) if best_outvals else None
#         for arg, val in (
#             ('def_outputs', def_outputs), 
#             ('dv', dv), 
#         ):
#             val = eval(arg)
#             if val:
#                 sb_kwargs[arg] = val 
#         fitness_factors = []
#         pipeline = []
#         if self.use_size:
#             if max_size:
#                 pipeline += [n_minus('size', n=max_size)]
#                 fitness_factors += ['n_minus_size']
#             else:
#                 pipeline += [safe_inv('size')]
#                 fitness_factors += ['isize']
#         if self.use_depth:
#             if max_depth:
#                 pipeline += [n_minus('depth', n=max_depth)]
#                 fitness_factors += ['n_minus_depth']
#             else:
#                 pipeline += [safe_inv('depth')]
#                 fitness_factors += ['idepth']
#         if len(fitness_factors) > 1:
#             pipeline += [weighted_sum(
#                 *(self.fitness_factors+fitness_factors), 
#                 weights=params[1:]
#             )]
#             def_fitness = 'ws_' + ('_'.join(self.fitness_factors + fitness_factors))
#         else:
#             def_fitness = fitness_factors[0]
#         pipeline += [
#             rename(def_fitness, 'raw_fitness'), 
#             heat, 
#             rename("fitness", 'pre_fitness_1')
#         ]
#         return GPScoreboard(
#             *self.pipeline_setup+pipeline+self.pipeline_post, **sb_kwargs
#         )
    

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
