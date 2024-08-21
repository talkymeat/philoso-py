
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Sequence, Callable
from collections import OrderedDict
from functools import cached_property

from gp import GPTreebank
from observatories import ObservatoryFactory
from tree_factories import TreeFactory, CompositeTreeFactory
from model_time import ModelTime
from repository import Archive, Publication
from gp_fitness import SimpleGPScoreboardFactory
from utils import scale_to_sum, InsufficientPostgraduateFundingError, _i
from guardrails import GuardrailManager
from cuboid import Cuboid

import torch
import numpy as np
from gymnasium.spaces.utils import flatten, unflatten, flatten_space
from gymnasium.spaces import Dict, Discrete, Box, MultiBinary, MultiDiscrete, Space
from torch.distributions import Bernoulli, Categorical, Normal

from icecream import ic

def min_dim(t: torch.Tensor) -> int:
    return len([d for d in t.shape if d > 1])

def scale_unit_to_range(val, min_, max_):
    return min_ + (val * (max_ - min_))

def space_2_distro(space: Space):
    if isinstance(space, Discrete):
        # return catag
        return lambda logits: Categorical(logits=logits)
    elif isinstance(space, MultiDiscrete):
        if np.all(space.nvec==space.nvec[0]):
            return lambda logits: Categorical(
                logits=torch.reshape(
                    logits, 
                    (*logits.shape[:-1], len(space.nvec), space.nvec[0]) if min_dim(logits)>1 else (len(space.nvec), space.nvec[0])
                )
            )
        else:
            raise InsufficientPostgraduateFundingError(
                "Handling MultiDiscretes of unequal order is a good " +
                "and useful piece of functionality, but it isn't " +
                "needed for to make the models for Xan's MSc work. If " +
                "wish to see this functionality implemented, kindly " +
                "give Xan Cochran funding to do a PhD. They'll get " +
                "right on it."
            )
    elif isinstance(space, MultiBinary):
        return lambda logits: Bernoulli(logits=logits)
    elif isinstance(space, Box):
        return lambda logits: Normal(loc=logits, scale=0.2)
    else:
        raise ValueError(
            'space_2_distro only takes fundamental Spaces, ' +
            f'not {type(space).__name__}'
        )


class Action(ABC):
    def __init__(self, 
            controller,
            *args
        ) -> None:
        self.ac = controller

    def __call__(self, act: np.ndarray|OrderedDict|tuple):
        self.do(*self.process_action(act))
    
    @property
    @abstractmethod
    def action_space(self) -> Space:
        pass

    @abstractmethod
    def process_action(self, 
        in_vals: OrderedDict[str, np.ndarray|int|float|bool]
    ):
        pass

    @abstractmethod
    def do(self, *args, **kwargs):
        pass
        # subspace sizes prepended with 0
        sspw0 = np.cumsum([0] + [
            flatten_space(ss).shape[0] for ss in self.action_space.values()
        ])
        # slice objects to divide the logits according to their destined subspace
        return [slice(sspw0[i], sspw0[i+1]) for i in range(len(sspw0)-1)]
    
    @cached_property
    def distributions(self):
        return {k: space_2_distro(ss) for k, ss in self.action_space.items()}
        pass
        # if len(self.logit_slicers)==1:
        #     return logits
        # if len(logits.shape)==1:
        #     return [logits[s] for s in self.logit_slicers]
        # else:
        #     dims = []
        #     for i, dim in enumerate(logits.shape):
        #         if dim==0:
        #             raise ValueError(f'Oh no, an empty tensor, {logits}')
        #         if dim>1:
        #             dims.append(i)
        #     if len(dims)==1:
        #         slicers = [
        #             [
        #                 (s if i==dims[0] else slice(None)) 
        #                 for i 
        #                 in range(len(logits.shape))
        #             ] 
        #             for s 
        #             in self.logit_slicers
        #         ]
        #         return [logits[s] for s in slicers]
        #     else:
        #         slicers = [
        #             ([slice(None)]*(len(logits.shape)-1))+[s]
        #             for s 
        #             in self.logit_slicers
        #         ]
        #         return [logits[s] for s in slicers]
                
                # ValueError(
                #     f'slice_logits cannot slice arrays with more than one '+
                #     f' significant (greater than 1) dimension: {logits}'
                # )
    
    @property
    def logit_dims(self):
        dims = [None] * len(self.action_space)
        for i, sp in enumerate(self.action_space.values()):
            if isinstance(sp, MultiDiscrete):
                if np.all(sp.nvec==sp.nvec[0]):
                    dims[i] = (len(sp.nvec), sp.nvec[0])
                else:
                    raise InsufficientPostgraduateFundingError(
                        "Handling MultiDiscretes of unequal order would be a good " +
                        "and useful piece of functionality, but it isn't " +
                        "needed for to make the models for Xan's MSc work. If " +
                        "wish to see this functionality implemented, kindly " +
                        "give Xan Cochran funding to do a PhD. They'll get " +
                        "right on it."
                    )
        return dims
        return OrderedDict({
            k: distro(
                torch.reshape(sllogits, shape)
                if shape
                else sllogits
            )
            for k, distro, sllogits, shape
            in zip(
                self.action_space.keys(),
                self.distributions, 
                self.slice_logits(logits),
                self.logit_dims
            )
        })
    
    def logits_2_distros(self, logits: dict):
        return OrderedDict({
            sub_name: self.distributions[sub_name](sub_logits)
            for sub_name, sub_logits
            in logits.items()
        })
        

# GP should put hyperparam and obs param data in self.best, so that's available when it's called by storemem, etc
class GPNew(Action):
    # pick a treebank slot
    # pick hyperparam values
    # add selection if Usemem has been run
    # run
    # Reward improvement ??

    def __init__(self, 
            controller,
            obs_factory: ObservatoryFactory,
            tree_factory_classes: list[type[TreeFactory]],
            rng: np.random.Generator,
            out_dir: str|Path,
            time: ModelTime,
            dv: str, 
            def_fitness: str,
            max_volume: int,
            mutators: list[Callable],
            theta: float = 0.05,
            ping_freq=5,

        ) -> None:
        super().__init__(controller)
        self.gptb_list  = self.ac.gptb_list
        self.gptb_cts   = self.ac.gptb_cts
        self.gp_vars_out = self.ac.gp_vars_out
        self.guardrails: GuardrailManager = self.ac.guardrail_manager
        self.sb_factory: SimpleGPScoreboardFactory = self.ac.sb_factory
        # note that the defined fitness value always has a raw weight of 1
        self.num_sb_weights: int = self.sb_factory.num_sb_weights - 1
        if CompositeTreeFactory not in tree_factory_classes:
            self.tf_options = tree_factory_classes
        else:
            raise ValueError("CompositeTreeFactory cannot be included in tree_factory options")
        self.num_tf_opts = (
            len(self.tf_options) if len(self.tf_options) > 1 else 0
        ) 
        self.obs_fac = obs_factory
        self.num_wobf_params = len(self.obs_fac.wobf_param_ranges)
        self.rng = rng
        self.out_dir = out_dir if isinstance(out_dir, Path) else Path(out_dir)
        self.t = time
        self.dv = dv 
        self.mutators = mutators
        self.num_mut_wts = (
            len(self.mutators) if len(self.mutators) > 1 else 0
        )
        self.def_fitness = def_fitness
        # XXX TODO make these size factors user-settable params (not less than 2,3,2 tho)
        self.min_pop = 2
        self.min_max_size = 3
        self.min_ep_len = 2
        self.size_factor_cuboid = Cuboid(
            self.min_pop,
            self.min_max_size,
            self.min_ep_len,
            max_volume
        )
        self.theta = theta
        self.ping_freq = ping_freq
        self.set_guardrails()
        # self.gp_best_vec_out = gp_best_vec_out

    @property
    def action_space(self) -> Space:
        """The action space for GP_New contains:

        SPACES
        ------

        gp_register (Discrete): 
            A single index to choose the GP slot to be used

            Length: 
                Same as the number of treebank slots
            Probability distribution:
                Categorical

        --

        The following are all -inf to inf Boxes, squashed with `tanh`, and 
        scaled if needed so these can be handled as a single long Box and 
        sliced locally:

        size_factors (pop, max_size, steps: Box):
            Define a `max_step_size` param: (pop, max_size, steps) will be
            normed so that `pop * max_size * steps` is clamped to be less than
            `max_step_size`, but also to be as close to it as possible with
            whole number values given that, they should also be as close as 
            possible to proportionate to the box values

            Range:
                All [-inf, inf] >-tanh-> [0.0, 1.0]
            Shape:
                3

        sb_weights (Box):
            Used to determine the weight given to the inverses of the main error
            measure (rmse, mse, or sae), size, and depth in determining the 
            fitness of trees. Normed to sum to 1, but below a threshhold *theta*,
            values will be reduced to zero. No value is needed for the error
            measure, though - it gets a value of 1, so when normed it has a 
            range of [0.333..., 1.0]

            Range:
                All [-inf, inf] >-tanh-> [0.0, 1.0] 
            Shape:
                2

        mutator_weights (Box):
            Used to determine how many of the trees in each generation will be
            replicated with each of the available mutation protocols

            Range:
                All [-inf, inf] >-tanh-> [0.0, inf] 
            Shape:
                len(self.mutators) or 0

        misc (crossover_rate, mutation_rate, mutation_sd, max_depth, 
                elitism, temp_coeff: Box):
            These are single vars that are determined by their own [0.0, 1.0]
            Box value, though some have different scaling factors:

            crossover_rate:
                Tanh, unscaled
            mutation_rate,
                Tanh, unscaled
            mutation_sd,
                Tanh, unscaled
            max_depth,
                Tanh, Scaled to [ceil(log2(size)), size/2], ceilinged
            elitism
                Tanh x `pop`, rounded with floor
            temp_coeff:
                Tanh x 2

            Range:
                All [-inf, inf] >-tanh-> [0.0, 1.0]
            Shape:
                6

        tf_weight (Box):
            Omit if there's only one tree factory. Use a threshold (*theta*) so 
            if the normalised value of any one of the weights is less than 
            *theta*, weight is set to 0 and that TF is not used.

            Range:
                All [0.0, 1.0]
            Shape:
                1D, same length as the number of TFs

        --

        tf_params (Dict) (TO BE IMPLEMENTED XXX TODO XXX)
            Inherited from TFs. May be omitted for now. However. I think the 
            max depth and size should also be passed to TFs, as these maxima
            should be respected. This may clamp the `order` of the random 
            polynomial factory
        """
        gp_register_sp = Discrete(len(self.gptb_list)) 
        num_wobf_params = len(self.obs_fac.wobf_param_ranges)
        box_len = 9 + self.num_sb_weights + num_wobf_params + (
            len(self.tf_options) if len(self.tf_options) > 1 else 0
        ) + (
            len(self.mutators) if len(self.mutators) > 1 else 0
        )
        long_box = Box(low=-np.inf, high=np.inf, shape=(box_len,))
        return Dict({
            'gp_register': gp_register_sp,
            'long_box': long_box
        })

    def set_guardrails(self):
        self.guardrails.make('pop', min=self.min_pop, max=np.inf)
        self.guardrails.make('max_size', min=self.min_max_size, max=np.inf)
        self.guardrails.make('episode_len', min=self.min_ep_len, max=np.inf)
        for n in range(self.num_sb_weights):
            self.guardrails.make(f'sb_weight_{n}', min=-np.inf, max=np.inf)
        self.guardrails.make('crossover_rate', min=0, max=1)
        self.guardrails.make('mutation_rate', min=0, max=1)
        self.guardrails.make('mutation_sd', min=-np.inf, max=np.inf)
        self.guardrails.make('max_depth', min=1, max=np.inf)
        self.guardrails.make('elitism', min=0, max=1)
        self.guardrails.make('temp_coeff', min=0, max=np.inf)
        for i in range(len(self.mutators)):
            self.guardrails.make(f'mutator_wt_{i}', min=0, max=1)
        self.obs_guardrail_names = []
        for params in self.obs_fac.wobf_guardrail_params:
            self.obs_guardrail_names.append(params['name'])
            if '_no_make' not in params:
                self.guardrails.make(**params)

    def process_action(self, 
        in_vals: OrderedDict[str, np.ndarray|int|float|bool]
    ):
        gp_register = int(in_vals['gp_register'][0])
        raws = in_vals['long_box'][0]
        arr = (torch.tanh(raws) + 1)/2
        pop, max_size, episode_len = self.size_factor_cuboid(*arr[:3])
        pop = self.guardrails['pop'](raws[0], pop)
        max_size = self.guardrails['max_size'](raws[1], max_size)
        episode_len = self.guardrails['episode_len'](raws[2], episode_len)
        # XXX TODO make this able to handle more weights
        sb_weights = scale_to_sum(np.append(1.0, arr[3:3+self.num_sb_weights]))
        for i, raw in zip(range(self.num_sb_weights), raws[3:3+self.num_sb_weights]):
            sb_weights[i] = self.guardrails[f'sb_weight_{i}'](raw, sb_weights[i])
        crossover_rate = self.guardrails['crossover_rate'](raws[3+self.num_sb_weights], arr[3+self.num_sb_weights]) # no scaling
        mutation_rate = self.guardrails['mutation_rate'](raws[4+self.num_sb_weights], arr[4+self.num_sb_weights])  # no scaling
        mutation_sd = self.guardrails['mutation_rate'](raws[5+self.num_sb_weights], arr[5+self.num_sb_weights])    # no scaling
        min_max_depth = np.ceil(np.log2(max_size))
        max_max_depth = max_size/2
        max_depth = int(scale_unit_to_range(arr[6+self.num_sb_weights], min_max_depth, max_max_depth))
        max_depth = self.guardrails['max_depth'](raws[6+self.num_sb_weights], max_depth)
        elitism =  int(self.guardrails['elitism'](raws[7+self.num_sb_weights], arr[7+self.num_sb_weights])*pop)
        temp_coeff = self.guardrails['temp_coeff'](raws[8+self.num_sb_weights], arr[8+self.num_sb_weights])*2
        obs_params = arr[
            9+self.num_sb_weights:9+self.num_sb_weights+len(self.obs_fac.wobf_param_ranges)
        ]
        obs_params_raw = raws[
            9+self.num_sb_weights:9+self.num_sb_weights+len(self.obs_fac.wobf_param_ranges)
        ]
        obs_args = [scale_unit_to_range(val, *bounds) for val, bounds in zip(obs_params, self.obs_fac.wobf_param_ranges)]
        obs_args = tuple([
            self.guardrails[gr_name](raw_param, obs_arg) 
            for raw_param, obs_arg, gr_name 
            in zip(obs_params_raw, obs_args, self.obs_guardrail_names)
        ])
        tf_weights = arr[
            9+self.num_sb_weights
            +len(self.obs_fac.wobf_param_ranges)
            :9+self.num_sb_weights
            +len(self.obs_fac.wobf_param_ranges)
            +len(self.tf_options)
        ] if len(self.tf_options) > 1 else None
        tf_choices = None
        if tf_weights:
            tf_weights = scale_to_sum(tf_weights)
            mask = tf_weights > self.theta
            if mask.sum() < len(tf_weights): # XXX test me
                tf_weights = scale_to_sum(tf_weights[mask])
                tf_choices = self.tf_options[mask]
            else:
                tf_choices = self.tf_options
        mut8or_weights = None
        if len(self.mutators) > 1:
            mut8or_weights = arr[
                9+self.num_sb_weights
                +self.num_wobf_params
                +self.num_tf_opts:
            ]
            mut8or_raws = raws[
                9+self.num_sb_weights
                +self.num_wobf_params
                +self.num_tf_opts:
            ]
            for i, raw in zip(range(len(mut8or_weights)), mut8or_raws):
                mut8or_weights[i] = self.guardrails[f'mutator_wt_{i}'](raw, mut8or_weights[i])
        
        # temp_coeff,
        return (
            gp_register, 
            tf_choices, 
            tf_weights, 
            #tf_params, XXX to implement
            pop,
            crossover_rate,
            mutation_rate,
            mutation_sd,
            temp_coeff,
            max_depth,
            max_size,
            elitism,
            episode_len,
            sb_weights,
            obs_args,
            mut8or_weights
        )
    
    def do(self, 
            gp_register: int, 
            tf_choices: list[int], 
            tf_weights, 
            #tf_params, 
            pop,
            crossover_rate,
            mutation_rate,
            mutation_sd,
            temp_coeff,
            max_depth,
            max_size,
            #operators, # operators, < derive from TFs
            elitism,
            episode_len,
            sb_weights: np.ndarray[float],
            obs_args,
            mutator_weights,
            *args, **kwargs
        ):
        if isinstance(gp_register, torch.Tensor):
            gp_register = gp_register.item()
        observatory = self.obs_fac(*[_i(arg) for arg in obs_args])
        tf_params = [None] # XXX implement this XXX
        tree_factory = self.get_tfs(tf_choices, tf_weights, tf_params, self.rng)
        scoreboard = self.sb_factory(observatory, _i(temp_coeff), sb_weights)
        if gp_register >= 0 and gp_register < len(self.gptb_list):
            self.gptb_cts[gp_register] += 1
            self.gptb_list[gp_register] = GPTreebank(
                pop =         int(_i(pop))          ,
                tree_factory=        tree_factory   ,
                observatory=         observatory    ,
                crossover_rate =  _i(crossover_rate),
                mutation_rate =   _i(mutation_rate) ,
                mutation_sd =     _i(mutation_sd)   ,
                temp_coeff =      _i(temp_coeff)    ,
                max_depth =   int(_i(max_depth))    ,
                max_size =    int(_i(max_size))     ,
                episode_len = int(_i(episode_len))  ,
                operators =          tree_factory.op_set,
                seed =               self.rng,
                _dir =              (
                                        self.out_dir     / 
                                        'gp_out'         / 
                                        f"{gp_register}" / 
                                        f"{self.gptb_cts[
                                            gp_register
                                        ]}"              / 
                                        'g0'             / 
                                        f"t{self.t}"
                                    ),
                elitism =         _i(elitism),
                fitness =            scoreboard,
                ping_freq =          self.ping_freq,
                mutator_factories=   self.mutators,
                mutator_weights=     mutator_weights.tolist()
                # dv = self.dv, 
                # def_fitness = self.def_fitness,
            )
            print(f"""Agent {self.ac.name} started a new GP at register {gp_register}:
            {tf_choices=}, {tf_weights=},
            {int(_i(pop))=}, {int(_i(max_size))=}, {int(_i(episode_len))=}, {int(_i(max_depth))=},
            {_i(crossover_rate)=}, {_i(mutation_rate)=}, {_i(mutation_sd)=},
            {_i(temp_coeff)=}, {_i(elitism)=},            
            {sb_weights=}, {mutator_weights=}
            {obs_args=}""")
            self.gptb_list[gp_register].set_up_run()
            self.gptb_list[gp_register].insert_trees(self.ac.mems_to_use)
            self.gptb_list[gp_register].continue_run()
        else:
            raise ValueError(f"GPTreebank index[{gp_register} out of range]")

    def get_tf(self, idx:int, seed: np.random.Generator, params):
        if idx >= 0 and idx < len(self.tf_options):
            return self.tf_options[idx](seed=seed, params=params)
        raise ValueError(f'TreeFactory index {idx} out of range')
    
    def get_tfs(self, idxs:list[int], weights:list[float], param_list: list[dict[str, Any]], seed: np.random.Generator):
        if weights and(len(weights) != len(idxs) or len(idxs) != len(param_list)):
            raise ValueError(
                f"idxs (len was {len(idxs)}), param_list (len was {len(param_list)}), " +
                f"and weights (len was {len(weights)}), must be the same length"
            )
        if len(self.tf_options) > 1:
            if len(idxs) > 1:
                tfs = [self.get_tf(i, seed, params) for i, params in zip(idxs, param_list)]
                return CompositeTreeFactory(tfs, weights, seed=seed)
            else:
                return self.get_tf(idxs[0], seed, param_list[0])
        else:
            return self.get_tf(0, self.rng, param_list[0])

class GPContinue(Action):
    # pick a treebank slot
    # pick hyperparam values (smaller set can be changed here)
    # add selection if Usemem has been run
    # run
    # Reward improvement ??
    # nothing happens if slot empty

    @property
    def action_space(self) -> Space:
        """The action space for GP_New contains:

        SPACES
        ------

        gp_register (Discrete): 
            A single index to choose the GP slot to be used

            Length: 
                Same as the number of treebank slots
            Probability distribution:
                Categorical

        misc (crossover_rate, mutation_rate, mutation_sd, max_depth, 
                elitism, temp_coeff, *mutator_weights: Box):
            These are single vars that are determined by their own [0.0, 1.0]
            Box value, though some have different scaling factors:

            crossover_rate:
                Unscaled
            mutation_rate,
                Unscaled
            mutation_sd,
                Unscaled
            elitism
                Scaled to [0, pop], rounded
            temp_coeff:
                Doubled
            mutator_weight_0 ... mutator_weight_{n}
                Unscaled

            Range:
                All [0.0, 1.0]
            Shape:
                5 + len(mutators)

        XXX TODO: it might be nice to allow this to also adjust scoreboard 
        weights XXX
        """
        gp_register_sp = Discrete(len(self.gptb_list)) 
        misc_box       = Box(low=0.0, high=1.0, shape=(5+self.num_mutators,))
        return Dict({
            'gp_register': gp_register_sp,
            'misc_box': misc_box
        })

    def process_action(self, 
        in_vals: OrderedDict[str, np.ndarray|int|float|bool]
    ):
        # Some refactoring would be good: this would make a good
        # parent class to GPNew
        raws = in_vals['misc_box'][0]
        arr = (torch.tanh(raws) + 1)/2
        if self.gptb_list[in_vals['gp_register'].item()]:
            gp_register = in_vals['gp_register'].item()
            crossover_rate = self.guardrails['crossover_rate'](raws[0], arr[0]) # no scaling
            mutation_rate = self.guardrails['mutation_rate'](raws[1], arr[1])  # no scaling
            mutation_sd = self.guardrails['mutation_sd'](raws[2], arr[2])  # no scaling
            elitism =  int(
                _i(self.guardrails['elitism'](raws[3], arr[3]))*self.gptb_list[gp_register].pop
            ) if self.gptb_list[gp_register] else -1
            temp_coeff = _i(self.guardrails['temp_coeff'](raws[4], arr[4]))*2
            mut8or_weights = None
            if self.num_mutators > 1:
                mut8or_weights = arr[5:]
                mut8or_raws = raws[5:]
                for i, raw in zip(range(len(mut8or_weights)), mut8or_raws):
                    mut8or_weights[i] = self.guardrails[f'mutator_wt_{i}'](raw, mut8or_weights[i])
        else:
            gp_register = -1
            crossover_rate = mutation_rate = mutation_sd = temp_coeff = 0.0
            elitism = 0
            if self.num_mutators > 1:
                mut8or_weights = [0.0]*self.num_mutators
        return (
            gp_register, 
            crossover_rate,
            mutation_rate,
            mutation_sd,
            temp_coeff,
            elitism,
            mut8or_weights
        )

    def __init__(self, 
            controller,
            out_dir: str|Path,
            time: ModelTime,
            num_mutators: int
        ) -> None:
        super().__init__(controller)
        self.gptb_list    = self.ac.gptb_list
        self.gptb_cts     = self.ac.gptb_cts
        self.guardrails: GuardrailManager = self.ac.guardrail_manager
        self.out_dir      = out_dir if isinstance(out_dir, Path) else Path(out_dir)
        self.t            = time
        self.num_mutators = num_mutators

    def do(self, 
            gp_register: int, 
            crossover_rate,
            mutation_rate,
            mutation_sd,
            temp_coeff,
            elitism,
            mut8or_weights,
            *args, **kwargs
        ):
        if isinstance(gp_register, torch.Tensor):
            gp_register = gp_register.item()
        if gp_register == -1:
            print(f'Agent {self.ac.name} Tried to continue GP in an empty slot')
            pass # do nothing if agent tries to use an empty slot
        elif gp_register >= 0 and gp_register < len(self.gptb_list):
            gp: GPTreebank = self.gptb_list[gp_register]
            if gp:
                gp.insert_trees(self.ac.mems_to_use)
                gp.data_dir = self.out_dir / 'gp_out' / f"{gp_register}" / f"{self.gptb_cts[gp_register]}" / f"g{gp.gens_past}" / f"t{self.t}"
                gp.crossover_rate  = _i(crossover_rate)
                gp.mutation_rate   = _i(mutation_rate)
                gp.mutation_sd     = _i(mutation_sd)
                gp.temp_coeff      = _i(temp_coeff)
                gp.elitism         = _i(elitism) 
                gp.mutator_weights = mut8or_weights.tolist()
                print(f"""Agent {self.ac.name} continued GP in slot {gp_register}:
                {_i(crossover_rate)=} {_i(mutation_rate)=} {_i(mutation_sd)=}
                {_i(temp_coeff)=} {_i(elitism)=} {mut8or_weights=}
                """)
                gp.continue_run()
            # Note: no else block. If the NN asks to continue a non-existent GP, nothing happens
        else:
            # print('asdsdfdfdsgfsdgs'*40)
            # print(args[0])
            # print(args[0]['gp_register'].item())
            # print(self.gptb_list[args[0]['gp_register'].item()])
            # print('=+-+'*30)
            raise ValueError(f"GPTreebank index[{gp_register}] out of range]")

class UseMem(Action):
    """Retrieves trees from memory for use in GP, up to a maximum

    """
    # store archive slot numbers in place GP* Actions will look

    def __init__(self, 
            controller
        ) -> None:
        super().__init__(controller)
        self.memory: Archive = self.ac.memory

    @property
    def action_space(self) -> Space:
        return Dict({'mb': MultiBinary(
            [len(self.memory.tables), len(self.memory.tables[0])]
        )})

    def process_action(self, 
        in_vals: OrderedDict[str, np.ndarray|int|float|bool]
    ):
        # XXX this comes in the wrong shape and I don't know why
        # XXX investigate later TODO
        # XXX I think it's fixed now?
        in_vals['mb'] = unflatten(self.action_space['mb'], in_vals['mb'])
        locations = []
        for i, x in enumerate(in_vals['mb']):
            for j, y in enumerate(x):
                if y:
                    locations.append((i,j))
        return (locations,)

    def do(self, 
            locations: Sequence[tuple[int, int]],
            *args, **kwargs
        ):
        self.ac.mems_to_use = [self.memory[table, row, 'tree'] for table, row in locations]
        print(
            f"Agent {self.ac.name} will use the following from " +
            f"memory with GP: \n" +
            '\n'.join([f'{t}' for t in self.ac.mems_to_use])
        )


class StoreMem(Action):
    # Treebank number of mem to be stored
    # address of mem register for

    def __init__(self, 
            controller,
        ) -> None:
        super().__init__(controller)
        self.memory: Archive = self.ac.memory
        self.gptb_list = self.ac.gptb_list
        self.vars = self.ac.gp_vars_out

    @property
    def action_space(self) -> Space:
        gp_registers   = MultiBinary(len(self.gptb_list))
        mem_table_idxs = MultiDiscrete(
            [len(self.memory.tables)] * len(self.gptb_list)
        )
        mem_row_idxs   = MultiDiscrete(
            [len(self.memory.tables[0])] * len(self.gptb_list)
        )
        return Dict({
            'gp_registers':   gp_registers,
            'mem_table_idxs': mem_table_idxs,
            'mem_row_idxs':   mem_row_idxs
        })

    def process_action(self, 
        in_vals: OrderedDict[str, np.ndarray|int|float|bool]
    ):
        num_gps = len(self.gptb_list)
        # gp_registers_bool = arr[:num_gps]
        # mem_table_idxs    = arr[num_gps:2*num_gps]
        # mem_row_idxs      = arr[2*num_gps:]
        gp_registers      = []
        memory_locations  = []

        for gp_idx, gp_bool, table_idx, row_idx in zip(
                range(num_gps), 
                in_vals['gp_registers'][0], # XXX At some point I should
                in_vals['mem_table_idxs'], # figure out why only gp_registers
                in_vals['mem_row_idxs'] # is sneakily 2-D
            ):
            if gp_bool:
                gp_registers.append(gp_idx)
                memory_locations.append((table_idx, row_idx))
        
        return gp_registers, memory_locations

    def do(self,
            gp_registers: Sequence[int],
            memory_locations: Sequence[tuple[int, int]],
            *args, **kwargs
        ):
        for gp_reg, mem_loc in zip(gp_registers, memory_locations):
            if self.gptb_list[gp_reg]:
                tree_data = ic(self.gptb_list[gp_reg].best)
                if tree_data['tree'] is not None:
                    data = {k: v for k, v in tree_data['data'].items() if k in self.vars}
                    self.memory.insert_tree(tree_data['tree'], journal=_i(mem_loc[0]), pos=_i(mem_loc[1]), **data)
                    print(f'>>>> Agent {self.ac.name} remembered a tree with fitness {data['fitness']}')
                else:
                    print(f'>>>> Agent {self.ac.name} tried to remember a tree from a GP with no stored best')
            else:
                print(f'>>>> Agent {self.ac.name} tried to remember a tree from an empty GP slot')
            # no else, do nothing if it tries to pull from an empty slot

class Publish(Action):
    # location of mem to be published
    # publication journal number

    def __init__(self, 
            controller,
        ) -> None:
        super().__init__(controller)
        self.repo: Publication = self.ac.repository
        self.gptb_list = self.ac.gptb_list
        self.vars = self.ac.gp_vars_out

    @property
    def action_space(self) -> Space:
        return Dict({
            'gp_register': Discrete(len(self.gptb_list)),
            'journal_num': Discrete(len(self.repo.tables))
        })

    def process_action(self, 
        in_vals: OrderedDict[str, np.ndarray|int|float|bool]
    ):
        return in_vals['gp_register'], in_vals['journal_num']

    def do(self,
            gp_register: int,
            journal_num: int,
            *args, **kwargs
        ):
        # Do nothing if an empty register is selected. nyaaaa
        # XXX in future, penalise this
        # XXX aaaaalso - what about going from memory to journal?
        if self.gptb_list[gp_register]:
            tree_data = self.gptb_list[gp_register].best
            if tree_data['tree'] is not None:
                data = {k: v for k, v in tree_data['data'].items() if k in self.vars}
                self.repo.insert_tree(tree_data['tree'], self.ac.name, journal=journal_num, data=data)
                print(f'>>>> Agent {self.ac.name} published a tree with fitness {data['fitness']} to journal {journal_num}')
            else:
                print(f'>>>> Agent {self.ac.name} tried to publish a tree from a GP with no stored best')
        else:
            print(f'>>>> Agent {self.ac.name} tried to publish a tree from an empty GP slot')

class Read(Action):
    # publication numbers and addresses,
    # memory registers to store what's read

    def __init__(self, 
            controller,
            max_readings: int,
            vars: list[str]
        ) -> None:
        super().__init__(controller)
        self.repo: Publication = self.ac.repository
        self.memory: Archive = self.ac.memory
        self.max_readings = max_readings
        self.vars = vars

    @property
    def action_space(self) -> Space:
        mask   = MultiBinary(self.max_readings)
        mem_table_idxs = MultiDiscrete(
            [len(self.memory.tables)] * self.max_readings
        )
        mem_row_idxs   = MultiDiscrete(
            [len(self.memory.tables[0])] * self.max_readings
        )
        repo_table_idxs = MultiDiscrete(
            [len(self.repo.tables)] * self.max_readings
        )
        repo_row_idxs   = MultiDiscrete(
            [len(self.repo.tables[0])] * self.max_readings
        )
        return Dict({
            'mask': mask,
            'mem_table_idxs': mem_table_idxs,
            'mem_row_idxs': mem_row_idxs,
            'repo_table_idxs': repo_table_idxs,
            'repo_row_idxs': repo_row_idxs
        })

    def process_action(self, in_vals: OrderedDict[str, np.ndarray|int|float|bool]|np.ndarray[int|float]):
        mask            = in_vals['mask'][0].int()
        try:
            mem_table_idxs  = in_vals['mem_table_idxs'][mask]
        except Exception as e:
            print('F'*300)
            print(in_vals['mem_table_idxs'])
            print(mask)
            raise e
        mem_row_idxs    = in_vals['mem_row_idxs'][mask]
        repo_table_idxs = in_vals['repo_table_idxs'][mask]
        repo_row_idxs   = in_vals['repo_row_idxs'][mask]
        memory_locs = [(tbl, row) for tbl, row in zip(mem_table_idxs, mem_row_idxs)]
        journal_locs = [(tbl, row) for tbl, row in zip(repo_table_idxs, repo_row_idxs)]
        return memory_locs, journal_locs

    def do(self,
            memory_locs:  Sequence[tuple[int, int]],
            journal_locs: Sequence[tuple[int, int]],
            *args, **kwargs
        ):
        tree_data = [self.repo[*loc] for loc in journal_locs]
        for td, mem_loc in zip(tree_data, memory_locs):
            if td['tree'] is not None:
                self.memory.insert_tree(td['tree'], journal=_i(mem_loc[0]), pos=_i(mem_loc[1]), **td[self.vars])
                print(f'Agent {self.ac.name} read tree {td['tree']}')

class PunchSelfInFace(Action):
    def __init__(self, controller) -> None:
        super().__init__(controller)
    
    @property
    def action_space(self) -> Space:
        return Dict({'twice': Discrete(2)})

    def process_action(self, in_vals: OrderedDict):
        twice = in_vals['twice'].bool()
        return twice

    def do(self, twice, *args, **kwargs):
        self.ac.tmp['punches'] = -1000 * (1 + twice)

##################################################
# an extra action that would be good (later):    #
# class Verify(Action):                          #
#     # runs further testing on trees in memory  #
#     # the amount of testing done is stored in  #
#     # mem data dict                            #
#     pass                                       #
##################################################

##################################################
# Also:                                          #
# class Simplify(Action):                        #
#     # uses an existing tree in memory as the   #
#     # source of a FunctionObservatory, with    #
#     # * zero noise                             #
#     # * heavy penalties for error outside a    #
#     #   given tolerance                        #
#     # * if tolerances are met, high value for  #
#     ¢   reducing tree size                     #
#     pass                                       #
##################################################

##################################################
# Stop reading and publishing duplicates         #
#                                                #
# Also, on reflection, better that `Publish`     #
# take from GP.best rather than mem: prevents    #
# plagiarism                                     #
##################################################

