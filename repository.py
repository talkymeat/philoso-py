from treebanks import TypeLabelledTreebank
from gp_trees import GPTerminal, GPNonTerminal
from gp_fitness import *
import pandas as pd
import numpy as np
import torch
# from copy import copy
from typing import Sequence, Collection, Mapping # List, Callable, 
from observatories import *
from tree_factories import *
from tree_funcs import * 
from tree_errors import UserIDCollision
from model_time import ModelTime
from typing import Protocol
from collections import deque
from m import MDict
from functools import reduce
from icecream import ic
from utils import simplify

class PublicationRewardFunc(Protocol):
    """Typically, reward should be positive if the submission is successfully
    published to the repository (index is non-negative), and negative if it is
    rejected (no value for index given)
    """
    def __call__(self, index=-1, **measures) -> float:
        ...

def rank_reward_func_factory(n: int) -> PublicationRewardFunc:
    def rank_reward_func(index=-1, **measures) -> float:
        if index == -1:
            return -1.0
        return (n-index)/n
    return rank_reward_func

def null_reward_func(self, index=-1, **measures) -> float:
    return 0.0

class Archive(TypeLabelledTreebank): #M #P
    ESSENTIAL_COLS = ['t', 'exists', 'tree']

    def __init__(self, 
            cols: Sequence[str],  #M #P
            rows: int,  #M #P
            model_time: ModelTime,  #M? #P
            *args,
            types: Sequence[np.dtype]|Mapping[str, np.dtype]|np.dtype|None=None,  #M #P
            tables: int=1,  #M #P
            value: str="fitness",  #P
            max_size: int = 1000,
            max_depth: int = 1000
        ):
        if types is None:
            types = np.float64
        self.types = types
        self._initialise_df(rows, cols, types)
        self._init_multi_dfs(tables)
        self._validate_cols(cols, value)
        self.value = value
        self.rows = rows
        # observations are a 3d np.ndarray, tables x rows x the length of 
        # the user-provided cols, plus 1 for the tree ages
        self.observation_dims = (tables, rows, len(cols)+1)
        self._t = model_time
        self.max_size = max_size
        self.max_depth = max_depth
        self.mutation_rate = 0.0
        self.mutation_sd = 0.0
        self.np_random = None
        super().__init__()
        self.N = GPNonTerminal
        self.T = GPTerminal

    @property
    def _common_json(self) -> dict:
        noncore = self.noncore
        return {
            'cols': list(noncore.columns),
            'rows': len(noncore),  
            'types': self.noncore_types,
            'tables': len(self.tables), 
            'value': self.value, 
        }

    @property
    def json(self) -> dict:
        return {
            **self._common_json,
            **{
                'max_size': self.max_size,
                'max_depth': self.max_depth
            }
        }
    
    @property
    def noncore(self):
        return self.tables[0].drop(self.ESSENTIAL_COLS, axis=1)
    
    @property
    def noncore_types(self):
        return simplify([str(dtype) for dtype in self.noncore.dtypes])

    def observe(self):
        return np.array([self.observe_journal(journal) for journal in self.tables])

    def observe_journal(self, journal: pd.DataFrame) -> np.ndarray:
        obs = np.array(journal.loc[:, (journal.columns != 'tree') & (journal.columns != 'exists')], dtype=np.float64)
        obs[:, 0] = obs[:, 0] - self._t()
        return obs

    def _initialise_df(self, rows, cols, types, *args):
        def_type = types if isinstance(type, np.dtype) else np.float64  #M #P
        types = types if isinstance(types, (Mapping, Sequence)) else {}
        df_core = pd.DataFrame({
            't': np.zeros(rows, dtype=np.int32),
            'exists': np.zeros(rows, dtype=bool),
            'tree': [None]*rows
        })
        if isinstance(types, Mapping):
            df_data_cols = pd.DataFrame({head: np.zeros(rows, dtype=types.get(head, def_type)) for head in cols})
        elif isinstance(types, Sequence):
            if len(types)==len(cols):
                df_data_cols = pd.DataFrame({head: np.zeros(rows, dtype=type) for head, type in zip(cols, types)})
            else:
                raise ValueError(
                    f"Length of types [{len(types)}] should match length of cols[{len(cols)}]" +
                    " if types is a Sequence"
                )
        else:
            raise ValueError(
                "types should be a Sequence, a Mapping, a numpy.dtype, or None: it should" +
                f" not be a {type(types)}"
            )
        self.tables = [df_core.join(df_data_cols)]

    def _init_multi_dfs(self, tables):
        if len(self.tables) > 1:
            raise ValueError(
                "_init_multi_df should only be called on a n Archive if it currently has only"
                + " one DataFrame"
            )
        self.tables += [self.tables[0].copy() for _ in range(tables-1)]

    def _validate_cols(self, cols, value):
        self.cols = set(cols)
        if len(cols) != len(self.cols):
            raise ValueError(f'Duplicate column names in cols {cols}')
        if value not in cols:
            print(cols)
            raise ValueError(f"No column '{value}' exists")

    def _preprocess_entry(self, tree, **data) -> bool:
        if set(data.keys()) != self.cols:
            surplus = set(data.keys()) - self.cols
            missing = self.cols - set(data.keys())
            surplus_msg = f"contains the keys {surplus}, which are not in Repository.cols" if surplus else ""
            missing_msg = f"lacks the keys {missing}, which are required in Repository.cols" if missing else ""
            and_msg = "; and " if surplus and missing else ""
            raise ValueError(f"metadata values passed to Repository.insert_tree {surplus_msg}{and_msg}{missing_msg}.")
        ic("fjkghfjgkhdfgkjhdjfhjgfd", tree)
        data['tree'] = tree.copy_out(treebank=self)
        ic('grnglhlglglglgl', data['tree'])
        data['exists'] = True
        data['t'] = self._t()
        return data

    def _get_journal(self, journal):
        if journal < 0:
            if len(self.tables) > 1:
                raise ValueError('No value for `table` was passed to Repositories.insert_tree, when the repository has multiple tables')
            journal = 0
        elif journal >= 0 and len(self.tables) == 1:
            raise ValueError('A value of `table` was passed to Repositories.insert_tree, when the repository has only one table')
        return self.tables[journal]

    def insert_tree(self, tree, pos, journal=-1, **data) -> float:
        # Do not add the tree if it is already in the repo
        if tree.size()==1:
            raise ValueError(f"Tree inserted in memory, {tree} is size 1, data is {data}")
        for table in self.tables:
            if (table['tree'] == ic(tree)).any():
                return
        data = self._preprocess_entry(tree, **ic(data))
        _table = self._get_journal(journal)
        if pos < 0 or pos >= self.rows:
            raise ValueError(f"Value of `pos` [{pos}] is out of range")
        _table.loc[pos] = ic(data) 

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self.tables[idx]
        i = idx[0].item() if isinstance(idx[0], torch.Tensor) else idx[0]
        if len(idx) == 1 or isinstance(idx, str):
            if isinstance(idx, int):   
                return self.tables[i]
            if isinstance(idx, str):
                return np.array([
                    t[idx] for t in self.tables
                ])
        elif len(idx) == 2:
            j = idx[1].item() if isinstance(idx[1], torch.Tensor) else idx[1]
            return self.tables[i].iloc[j]
        elif len(idx) == 3:
            j = idx[1].item() if isinstance(idx[1], torch.Tensor) else idx[1]
            return self.tables[i].loc[j, idx[2]]
        else:
            raise IndexError('Archives take indices of one orr two ints, or ' 
                             + 'two ints and a column name, or just column ' 
                             + f'name: you gave {idx}, of type {type(idx)}.')
    

        
class Publication(Archive):
    REWARD_FUNCS = {
        'ranked': rank_reward_func_factory
    } # other possibilities: sd, time-lagged sd, composite XXX
    ESSENTIAL_COLS = ['credit', 't', 'exists', 'tree']

    def __init__(self, 
            cols: Sequence[str],  #M #P
            rows: int,  #M #P
            model_time: ModelTime,  #M? #P
            agent_names: dict[str, int], #P
            types: Sequence[np.dtype]|Mapping[str, np.dtype]|np.dtype|None=None,  #M #P
            tables: int=1,  #M #P
            value: str="fitness",  #P
            reward: PublicationRewardFunc|str|None=None,  #P
            decay: float=0.95
        ):
        super().__init__(
            cols+([] if 'credit' in cols else ['credit']), 
            rows, 
            model_time, 
            value=value, 
            types=types, 
            tables=tables
        )
        # observations are a 3d np.ndarray: as with Archive, the first 2 dims are
        # tables & rows x, but now the cols are:
        # *   the length of the user-provided cols
        # *   plus the number of agents using the Publication (crediting agents with one-hot encoding)
        # *   plus 1 for the tree ages
        self.observation_dims = (tables, rows, len(cols)+1) # this doesn't count the credit col
        #self._validate_users_cols(cols, users)
        self.agent_names = agent_names
        for journal in self.tables:
            journal['credit'] = len(self.agent_names)
        self._reward = null_reward_func if reward is None else reward  #P
        if decay > 1 or decay < 0:
            raise ValueError(f'Decay must be between 0 and 1: {decay} is invalid')
        self.decay = decay
        if isinstance(self._reward, Callable):
            self.reward_name = self._reward.__name__
        elif isinstance(self._reward, str):
            self.reward_name = self._reward
            match self._reward:
                case 'ranked':
                    self._reward = self.__class__.REWARD_FUNCS['ranked'](rows)
                case _:
                    if self._reward in self.__class__.REWARD_FUNCS:
                        self._reward = self.__class__.REWARD_FUNCS[reward]
                    else:
                        raise ValueError(f"Reward function '{self._reward}' not recognised")
        else:
            raise ValueError(f"Reward function '{self._reward}' not a string or a Callable")
        self._agents = pd.DataFrame({'agent': [], 'reward': []}) # P
        self.value = value

    @classmethod
    def from_json(cls, json, time=None, agent_names=None):
        if not time or not isinstance(time, ModelTime):
            raise AttributeError('A ModelTime object must be passed as `time`')
        if not agent_names or not isinstance(agent_names, dict) or sum([(not (isinstance(name, str) and isinstance(i, int))) for name, i in agent_names.items()]):
            raise AttributeError('A dict mapping strings to ints must be passed as `agent_names`')
        cols = json.get(
            ['publication_params', 'cols'], 
            (
                json.get(['publication_params', 'gp_vars_core'], []) 
                + json.get(['publication_params', 'gp_vars_more'], [])
            )
        )
        if not cols:
            raise ValueError(
                "publication_params must have at least one of " +
                "'cols', 'gp_vars_core', and 'gp_vars_more'"
            )
        args = [
            cols,
            json[['publication_params', 'rows']],
            time,
            agent_names
        ]
        kwargs = {
            pram: json[['publication_params', pram]] 
            for pram 
            in [
                'tables', 
                'reward', 
                'value', 
                'decay'
            ] 
            if [
                'publication_params', 
                pram
            ] in json
        }
        if ['publication_params', 'types'] in json:
            types = json[['publication_params', 'types']]
            if isinstance(types, dict):
                types = {k: np.dtype(v) for k, v in types.items()}
            elif isinstance(types, list):
                types = [np.dtype(v) for v in types]
            elif isinstance(types, str):
                types = np.dtype(types)
            else:
                raise TypeError(
                    '`types` must be a dict of strings, a list of strings, a string, or None'
                )
            kwargs['types'] = types
        return cls(*args, **kwargs)

    @property
    def json(self) -> dict:
        return {
            **self._common_json,
            **{
                'agent_indices': self.agent_names, 
                'reward': self.reward_name,
                'decay': self.decay
            }
        }

    def add_users(self, users):
        """Do not use with a running model. that is a feature not yet implemented"""
        for user in users:
            if user.name in self.agent_names:
                self._add_user(user)
        for journal in self.tables:
            journal['credit'] = len(self._agents)


    def _add_user(self, user): # P
        if user.name in self._agents:
            if self._agents[user.name] is not user:
                raise UserIDCollision(id=user.name)
            return
        self._agents.loc[user.name] = {'agent': user, 'reward': 0.0}     
        
    def _assign_tree_to_agent(self, agent_name, **vals): # P
        vals['credit'] = self.agent_names[agent_name]
        return vals

    def insert_tree(self, tree, agent_name, journal=-1, **data) -> float:
        if tree.size()==1:
            raise ValueError(f"Tree inserted in journal, {tree} is size 1, data is {data}")
        for table in self.tables:
            if (table['tree'] == tree).any():
                return
            for i, row in table.iterrows():
                if str(row['tree']) == str(tree):
                    raise ValueError(f"""
                        wtf {str(row['tree'])} == {str(tree)} BUT {(row['tree'] == tree)=})
                        {row['tree'][0,0]=}, {tree[0,0]}: {row['tree'][0,0] - tree[0,0]=}
                        {row['tree'][1,0]=}, {tree[1,0]}: {row['tree'][1,0] - tree[1,0]=}
                    """)
        # XXX JANK
        data = data['data'] if 'data' in data else data
        data = self._assign_tree_to_agent(agent_name, **data) 
        data = self._preprocess_entry(tree, **data)
        tree = data['tree']
        _journal = self._get_journal(journal)
        
        if agent_name not in self._agents.index: 
            raise ValueError(f'User {agent_name} does not have access to this repository')  
        
        empty = _journal[~_journal['exists']] # why was the below commented out? XXX TODO
        # if len(empty):
        #     _journal.iloc[empty.index[0]] = data
        if data[self.value] < _journal[self.value].min():
            # WOMP WOMP
            print(f"&&&& {agent_name}'s tree was rejected" )
            self._agents.loc[agent_name, 'reward'] += self._reward(**data, reject=True) 
            return # "you suck"
        else:
            # SUCCESS
            # calculate rewards to others
            rewards = tree.tree_map_reduce(sum_all, agent_name, map_any=calculate_credit) 
            # calculate self reward
            index = (_journal[self.value] > data[self.value]).sum()
            own_reward = self._reward(index, **data)
            if rewards:
                rewards = {k: reward * own_reward/sum(rewards.values()) for k, reward in rewards.items()}
            # sign tree
            tree.apply(init_reward_tree, own_reward)
            tree.apply(sign_tree, agent_name)  # just needs name
            # punish agent whose tree was kicked out
            last = len(_journal)
            _journal.loc[last] = data
            _journal.sort_values(self.value, inplace=True, ignore_index=True, ascending=False)
            dead_tree = _journal.loc[last, 'tree']
            if dead_tree:
                dt_age = self._t() - _journal.loc[last, 't']
                dt_reward = -(dead_tree.metadata['init_reward']) * self.decay ** dt_age
                dt_author = dead_tree.metadata['credit'][0]
                # remove old
                dead_tree.delete()
                rewards += MDict({dt_author: dt_reward})
            _journal.drop(last, inplace=True)
            # Push punishments & rewards to buffer
            rewards += MDict({agent_name: own_reward}) 
        for id, rew in rewards.items():
            self._agents.loc[id, 'reward'] += rew
            # XXX handle case where repo is not yet full

    def rewards(self):
        rews = MDict({row.name: row['reward'] for row in self._agents.iloc})
        self._agents['reward'] = np.zeros(len(self._agents), dtype=np.float64)
        return rews


def main():
    import doctest
    doctest.testmod()

if __name__ == '__main__':
    main()