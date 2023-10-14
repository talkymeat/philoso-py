import pandas as pd
from typing import Collection, Protocol, Iterable, Callable, Generic
from gp_trees import GPNonTerminal
from observatories import Observatory
from dataclasses import dataclass
from copy import deepcopy
from functools import wraps
from utils import collect

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
        arg_keys (string or list of strings): Keys which identify dataframe
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
    """
    arg_keys: list[str]|str
    out_key: str
    vec: bool = False
    fn: Callable = None

    def __post_init__(self):
        """A housekeeping method that tidies the data a bit after 
        initialisation: it converts string `argkeys` values to singleton lists,
        sets `vec` to true if the function operates on the 'trees' column (which
        cannot vectorise), and raises errors for some non-permitted combinations
        of attributes
        """
        self.arg_keys = collect(self.arg_keys, list, empty_if_none=True)
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
            self.fn(sb)
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
            sb[self.out_key] = self.fn(**sb[self.arg_keys], **kwargs)
        # If it can't, `vec` should be false, and `apply` will be used to go row
        # by row 
        else:
            sb[self.out_key] = sb.apply(lambda row: self.fn(**row, **kwargs), axis=1)

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


def scoreboard_pipeline_element(arg_keys=['tree'], out_key='fitness', vec=False):
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

def flexi_pipeline_element(vec=True, out_key_maker: Callable = None):
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
        def spe_maker(*arg_keys, out_key=None, **kwargs):
            # Sets the value for `out_key` - using the spe_maker override if
            # available, else a generated value from `out_key_maker`, else
            # a default made from the function and argument names
            out_key = out_key if out_key else out_key_maker(arg_keys) if out_key_maker else f"{fn.__name__}_{'_'.join(arg_keys)}"
            # To keep the f-string that creates the wrapper function simple,
            # the string representing the argumenrs of the function is 
            # generated here
            argstr = ', '.join(arg_keys)
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
            exec(
                f"def {out_key}({argstr}, **kwargs):\n" +
                f"\treturn {fn.__name__}.__wrapped__({argstr}, **kwargs)\n"
            )
            # Note `locals()` is the dict of all local variables, so
            # `locals()[out_key]` retrieves the function generated by `exec`
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
    """Multiplicative inversion operator, protected from ZeroDivisionErrors
    by adding 1 to the denominator. Generates the name of the new column by
    prepending `i` to the name of the old. Pass the name of the column you 
    want to invert to get the ScoreboardPipelineElement you need. e.g.: 
    `safe_inv('rmse')` gives an element `irmse` that inverts `rmse` 
    """
    return 1/(arg+1)

@scoreboard_pipeline_element(out_key='mse')
def mse(tree: GPNonTerminal, target: pd.Series=None, ivs=None, **kwargs):
    """Calculates the Mean Squared Error of tree outputs. Adds `mse`, 
    requires `tree`
    """
    return ((target - tree(**ivs))**2).mean()

@scoreboard_pipeline_element(out_key='sae')
def sae(tree: GPNonTerminal, target: pd.Series=None, ivs=None, **kwargs):
    """Calculates the Summed Average Error of tree outputs. Adds `sae`, 
    requires `tree`
    """
    return (target - tree(**ivs)).abs().mean()

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
    return raw_fitness + temp

@scoreboard_pipeline_element(out_key='penalty')
def penalty(tree: GPNonTerminal, **kwargs):
    """Trees receive penalties during processing for things like OverflowErrors 
    and ZeroDivisionErrors. Penalties are applied by dividing the tree's fitness,
    so the default 'no penalty' value is zero
    """
    return tree.metadata.get('penalty', 1.0)

# @flexi_pipeline_element(out_key_maker=lambda args: "fitness")

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

class GPScoreboard(pd.DataFrame):
    _metadata = [
        'obs', 'pipeline', 'kwargs', 'provide', 'require', 'def_outputs'
    ]

    def __init__(
            self,
            *pipeline: ScoreboardPipelineElement,
            def_fitness: str=None,
            dv: str=None,
            obs: Observatory = None, 
            temp_coeff: float = 0.0,
            provide: str|Collection[str]='tree', 
            require: str|Collection[str]='fitness',
            def_outputs: str|Collection[str]= None,
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
        self.pipeline = pipeline if pipeline else ([
            clear(except_for='tree'),
            sae, mse, rmse, 
            safe_inv('sae'), safe_inv('mse'), safe_inv('rmse')
        ] + (
            [rename(def_fitness, 'raw_fitness'), heat]
            if temp_coeff else
            [rename(def_fitness, 'fitness')]
        ))
        self.provide = collect(provide, set)
        self.require = collect(require, set)
        self.kwargs = {"temp_coeff": temp_coeff, **kwargs}
        final_cols, error = self._validate_pipeline()
        if error:
            raise ValueError(error + ((
                ". Also, _validate_pipeline returned both success and error " +
                "values, which makes no sense. the success value was " +
                f"'{final_cols}'"
            ) if final_cols else ""))
        elif def_outputs:
            def_outputs = collect(def_outputs, list)
            for d_o in def_outputs:
                if d_o not in final_cols:
                    raise ValueError(
                        "A value listed as a default scoreboard output, " +
                        f"{d_o}, is not in the final outputs of the tree" +
                        "scoring pipeline"
                    )
            self.def_outputs = def_outputs
        else:
            self.def_outputs = final_cols

    def _validate_pipeline(self) -> tuple[bool, str]:
        def format_error(p, i, e):
            err = "Pipeline failed"
            if i != -1:
                err += f" at element {i}, {p.__name__}"
            err += f": {e[0]}"
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
        >>> gp = GPTreebank(operators = op)
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

    def winner(self, *cols, except_for: str|list[str]=None, **kwargs):
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
        >>> op = [ops.PROD]
        >>> gp = GPTreebank(operators = op, temperature_coeff=0.5)
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
        ):
        return self(trees, **kwargs).winner(*cols, **kwargs)



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
#         trees (Collection[GPNonTerminal]): The root treenodes of a GPTreebank treebank
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