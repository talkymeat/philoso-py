
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Sequence
from collections import OrderedDict

from gp import GPTreebank
from observatories import ObservatoryFactory
from tree_factories import TreeFactory, CompositeTreeFactory
from operators import Operator
from model_time import ModelTime
from repository import Archive, Publication
from gp_fitness import SimpleGPScoreboardFactory
from utils import scale_to_sum, InsufficientPostgraduateFundingError


# might not be needed here, maybe perhaps
import torch
import numpy as np
from gymnasium.spaces.utils import flatten, unflatten, flatten_space
from gymnasium.spaces import Dict, Tuple, Discrete, Box, MultiBinary, MultiDiscrete, Space
from torch.distributions import Bernoulli, Categorical, Normal

from icecream import ic

def _i(item):
    if isinstance(item, torch.Tensor) and np.prod(item.shape)==1:
        return item.item()
    else:
        return item


def scale_unit_to_range(val, min_, max_):
    return min_ + (val * (max_ - min_))

# def catag(logits):
#     print(f'~c{"="*50}8')
#     print(logits)
#     print(f'8{"="*50}o~')
#     return Categorical(logits=logits)

def space_2_distro(space: Space):
    if isinstance(space, Discrete):
        # return catag
        return lambda logits: Categorical(logits=logits)
    elif isinstance(space, MultiDiscrete):
        if np.all(space.nvec==space.nvec[0]):
            return lambda logits: Categorical(logits=torch.reshape(logits, (len(space.nvec), space.nvec[0])))
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

    @property
    def logit_slicers(self) -> list[slice]:
        # subspace sizes prepended with 0
        sspw0 = np.cumsum([0] + [
            flatten_space(ss).shape[0] for ss in self.action_space.values()
        ])
        # slice objects to divide the logits according to their destined subspace
        return [slice(sspw0[i], sspw0[i+1]) for i in range(len(sspw0)-1)]
    
    @property
    def distributions(self):
        return [space_2_distro(ss) for ss in self.action_space.values()]
    
    def slice_logits(self, logits):
        if len(logits.shape)==1:
            return [logits[s] for s in self.logit_slicers]
        else:
            dims = []
            for i, dim in enumerate(logits.shape):
                if dim==0:
                    raise ValueError(f'Oh no, an empty tensor, {logits}')
                if dim>1:
                    dims.append(i)
            if len(dims)==1:
                slicers = [
                        [
                            (s if i==dims[0] else slice(None)) 
                            for i 
                            in range(len(logits.shape))
                        ] 
                        for s 
                        in self.logit_slicers
                    ]
                return [logits[s] for s in slicers]
            else:
                ValueError(
                    f'slice_logits cannot slice arrays with more than one '+
                    f' significant (greater than 1) dimension: {logits}'
                )
    
    @property
    def logit_dims(self):
        dims = [None] * len(self.action_space)
        for i, sp in enumerate(self.action_space.values()):
            if isinstance(sp, MultiDiscrete):
                if np.all(sp.nvec==sp.nvec[0]):
                    dims[i] = (len(sp.nvec), sp.nvec[0])
                else:
                    raise InsufficientPostgraduateFundingError(
                        "Handling MultiDiscretes of unequal order is a good " +
                        "and useful piece of functionality, but it isn't " +
                        "needed for to make the models for Xan's MSc work. If " +
                        "wish to see this functionality implemented, kindly " +
                        "give Xan Cochran funding to do a PhD. They'll get " +
                        "right on it."
                    )
        return dims
    
    def logits_2_sample(self, logits):
        return flatten(
            self.action_space,
            OrderedDict({
                k: distro(
                    torch.reshape(sllogits, shape)
                    if shape
                    else sllogits
                ).sample().numpy()
                for k, distro, sllogits, shape
                in zip(
                    self.action_space.keys(),
                    self.distributions, 
                    self.slice_logits(logits),
                    self.logit_dims
                )
            })
        )
    
    def logits_2_distros(self, logits):
        # for k_, distro_, sllogits_, shape_ in zip(
        #         self.action_space.keys(),
        #         self.distributions, 
        #         self.slice_logits(logits),
        #         self.logit_dims
        #     ):
            # if shape_:
            #     print("XXXXXXXX "*10)
            #     print(k_) 
            #     print(distro_)
            #     print(shape_)
            #     print(logits)
            #     print(logits.shape)
            #     print(sllogits_)
            #     print(sllogits_.shape)
            #     print('_-_-Z-Z---============-----------')
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
            # operators: list[Operator],
            rng: np.random.Generator,
            # best_outvals: str|list[str],
            # expt_outvals: str|list[str],
            out_dir: str|Path,
            time: ModelTime,
            dv: str, 
            def_fitness: str,
            max_volume: int,
            # gp_best_vec_out: list[str],
            theta: float = 0.05
        ) -> None:
        super().__init__(controller)
        self.gptb_list  = self.ac.gptb_list
        self.gptb_cts   = self.ac.gptb_cts
        self.gp_vars_out = self.ac.gp_vars_out
        self.sb_factory: SimpleGPScoreboardFactory = self.ac.sb_factory
        if CompositeTreeFactory not in tree_factory_classes:
            self.tf_options = tree_factory_classes
        else:
            raise ValueError("CompositeTreeFactory cannot be included in tree_factory options")
        self.obs_fac = obs_factory
        self.rng = rng
        # self.best_outvals = best_outvals
        # self.expt_outvals = expt_outvals
        # self.operator_set = operators
        self.out_dir = out_dir if isinstance(out_dir, Path) else Path(out_dir)
        self.t = time
        self.dv = dv 
        self.def_fitness = def_fitness
        self.max_volume = max_volume
        self.theta = theta
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
#----#----#----#----#----#----#----#----#----#----#----#----#----#----#----#----
        gp_register_sp = Discrete(len(self.gptb_list)) 
        num_wobf_params = len(self.obs_fac.wobf_param_ranges)
        box_len = 11 + num_wobf_params + (
            len(self.tf_options) if len(self.tf_options) > 1 else 0
        )
        long_box       = Box(low=-np.inf, high=np.inf, shape=(box_len,))
        return Dict({
            'gp_register': gp_register_sp,
            'long_box': long_box
        })

    def process_action(self, 
        in_vals: OrderedDict[str, np.ndarray|int|float|bool]
    ):
        
        gp_register = int(in_vals['gp_register'][0])
        arr = (torch.tanh(in_vals['long_box'])[0] + 1)/2
        try:
            pop, max_size, episode_len = self.calculate_size_factors(arr[:3])
        except Exception as e:
            print("SFJHKJGKJGHSGKHJ"*8)
            print(arr)
            print("DJKSDKJJDGDJKGHKSHGKJSD"*8)
            raise e
        sb_weights = scale_to_sum(np.append(1.0, arr[3:5])) 
        crossover_rate = arr[5] # no scaling
        mutation_rate = arr[6]  # no scaling
        mutation_sd = arr[7]    # no scaling
        min_max_depth = np.ceil(np.log2(max_size))
        max_max_depth = max_size/2
        try:
            max_depth = int(scale_unit_to_range(arr[8], min_max_depth, max_max_depth))
        except Exception as e:
            print('-'*80)
            print(in_vals['long_box'])
            print(torch.tanh(in_vals['long_box']))
            print(arr[:3])
            print(max_size)
            print(np.log2(max_size))
            print(arr[8], '>>', min_max_depth, '<<', max_max_depth)
            raise e
        elitism =  int(arr[9]*pop)
        temp_coeff = arr[10]*2
        obs_params = arr[11:11+len(self.obs_fac.wobf_param_ranges)]
        obs_args = tuple([scale_unit_to_range(val, *bounds) for val, bounds in zip(obs_params, self.obs_fac.wobf_param_ranges)])
        tf_weights = arr[11+len(self.obs_fac.wobf_param_ranges):] if len(self.tf_options) > 1 else None
        
        tf_choices = None
        if tf_weights:
            tf_weights = scale_to_sum(tf_weights)
            mask = tf_weights > self.theta
            if mask.sum() < len(tf_weights): # XXX test me
                tf_weights = scale_to_sum(tf_weights[mask])
                tf_choices = self.tf_options[mask]
            else:
                tf_choices = self.tf_options
         
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
            obs_args
        )
    
    def calculate_size_factors(self, sz_f: np.ndarray):
        """The formula to scale an array such that it's product is a specified
        value (in this case `self.max_volume`, which approximately  defines the
        number of nodes*timesteps in an action) is:

        scaling = [inverse product of the array * max_vol] to the root of the  
            length of the array
        new_arr = array * scaling

        Note since max tree size, tree population, and number of steps are all
        integers, this may fall short, as casting np.float to int rounds using
        `floor` 
        """
        # print("HJFGDJ " *60)
        # print(sz_f)
        # print("JDSJDD " *40)
        return ((self.max_volume/sz_f.prod())**(1/len(sz_f)) * sz_f).int()

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
            *args, **kwargs
        ):
        if isinstance(gp_register, torch.Tensor):
            gp_register = gp_register.item()
        observatory = self.obs_fac(*obs_args)
        tf_params = [None] # XXX implement this XXX
        tree_factory = self.get_tfs(tf_choices, tf_weights, tf_params, self.rng)
        scoreboard = self.sb_factory(observatory, temp_coeff, sb_weights)
        if gp_register >= 0 and gp_register < len(self.gptb_list):
            self.gptb_cts[gp_register] += 1
            self.gptb_list[gp_register] = GPTreebank(
                pop = _i(pop),
                tree_factory=tree_factory,
                observatory=observatory,
                crossover_rate = _i(crossover_rate),
                mutation_rate = _i(mutation_rate),
                mutation_sd = _i(mutation_sd),
                temp_coeff = _i(temp_coeff),
                max_depth = _i(max_depth),
                max_size = _i(max_size),
                episode_len=_i(episode_len),
                # commented out params that aren't needed when pre-made scoreboard is passed
                # best_outvals = self.best_outvals,
                # expt_outvals = self.expt_outvals,
                operators = tree_factory.op_set,
                seed = self.rng,
                _dir = self.out_dir / 'gp_out' / f"{gp_register}" / f"{self.gptb_cts[gp_register]}" / 'g0' / f"t{self.t}",
                elitism = _i(elitism),
                fitness = scoreboard
                # dv = self.dv, 
                # def_fitness = self.def_fitness,
            )
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
                elitism, temp_coeff: Box):
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

            Range:
                All [0.0, 1.0]
            Shape:
                5

        XXX TODO: it might be nice to allow this to also adjust scoreboard 
        weights XXX
        """
#----#----#----#----#----#----#----#----#----#----#----#----#----#----#----#----
        gp_register_sp = Discrete(len(self.gptb_list)) 
        misc_box       = Box(low=0.0, high=1.0, shape=(5,))
        return Dict({
            'gp_register': gp_register_sp,
            'misc_box': misc_box
        })

    def process_action(self, 
        in_vals: OrderedDict[str, np.ndarray|int|float|bool]
    ):
        if self.gptb_list[in_vals['gp_register'].item()]:
            gp_register = in_vals['gp_register'].item()
            crossover_rate = in_vals['misc_box'][0,0] # no scaling
            mutation_rate = in_vals['misc_box'][0,1]  # no scaling
            mutation_sd = in_vals['misc_box'][0,2]    # no scaling
            elitism =  int(
                in_vals['misc_box'][0,3].item()*self.gptb_list[gp_register].pop
            ) if self.gptb_list[gp_register] else -1
            temp_coeff = in_vals['misc_box'][0,4].item()*2
        else:
            gp_register = -1
            crossover_rate = mutation_rate = mutation_sd = temp_coeff = 0.0
            elitism = 0
        return (
            gp_register, 
            crossover_rate,
            mutation_rate,
            mutation_sd,
            temp_coeff,
            elitism,
            #in_vals # DELETE THIS XXX XXX XXX XXX XXX
        )

    def __init__(self, 
            controller,
            out_dir: str|Path,
            time: ModelTime,
        ) -> None:
        super().__init__(controller)
        self.gptb_list = self.ac.gptb_list
        self.gptb_cts  = self.ac.gptb_cts
        self.out_dir   = out_dir if isinstance(out_dir, Path) else Path(out_dir)
        self.t         = time

    # def __call__(self, tch: torch.tensor):
    #     return super().__call__(tch)

    def do(self, 
            gp_register: int, 
            crossover_rate,
            mutation_rate,
            mutation_sd,
            temp_coeff,
            elitism,
            # valid: bool,
            *args, **kwargs
        ):
        if isinstance(gp_register, torch.Tensor):
            gp_register = gp_register.item()
        if gp_register == -1:
            pass # do nothing if agent tries to use an empty slot
        elif gp_register >= 0 and gp_register < len(self.gptb_list):
            gp: GPTreebank = self.gptb_list[gp_register]
            if gp:
                gp.insert_trees(self.ac.mems_to_use)
                gp.data_dir = self.out_dir / 'gp_out' / f"{gp_register}" / f"{self.gptb_cts[gp_register]}" / f"g{gp.gens_past}" / f"t{self.t}"
                gp.crossover_rate = _i(crossover_rate)
                gp.mutation_rate  = _i(mutation_rate)
                gp.mutation_sd    = _i(mutation_sd)
                gp.temp_coeff     = _i(temp_coeff)
                gp.elitism        = _i(elitism) 
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

class StoreMem(Action):
    # Treebank number of mem to be stored
    # address of mem register for

    def __init__(self, 
            controller,
        ) -> None:
        super().__init__(controller)
        self.memory: Archive = self.ac.memory
        self.gptb_list = self.ac.gptb_list

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
                tree_data = self.gptb_list[gp_reg].best
                if tree_data['tree'] is not None:
                    self.memory.insert_tree(tree_data['tree'], journal=mem_loc[0], pos=mem_loc[1], data=tree_data['data'])
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
        if self.gptb_list[gp_register]:
            tree_data = self.gptb_list[gp_register].best
            if tree_data['tree'] is not None:
                self.repo.insert_tree(tree_data['tree'], journal=journal_num, data=tree_data['data'])

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
        try:
            mask            = in_vals['mask'][0].int()
            mem_table_idxs  = in_vals['mem_table_idxs'][mask]
        except Exception as e:
            print("dsjfhsdkjfd"*6)
            print(mask)
            print('djfdjsdjksjsk'*11)
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
            # print('XXX'*40)
            # print(td)
            # print('XXX'*40)
            if td['tree'] is not None:
                self.memory.insert_tree(td['tree'], journal=mem_loc[0], pos=mem_loc[1], data=td[self.vars])


##################################################
# an extra action that would be good (later):    #
# class Verify(Action):                          #
#     # runs further testing on trees in memory  #
#     # the amount of testing done is stored in  #
#     # mem data dict                            #
#     pass                                       #
##################################################
