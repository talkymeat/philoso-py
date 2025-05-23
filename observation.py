from abc import ABC, abstractmethod
from typing import Sequence, Callable

import pandas as pd
import numpy as np
import torch
from gymnasium.spaces.utils import flatten, unflatten
from gymnasium.spaces import Dict, Tuple, Discrete, Box, MultiBinary, MultiDiscrete, Space

from gp import GPTreebank
from gp_fitness import GPScoreboard
from repository import Archive, Publication
from icecream import ic
from utils import taper

def clean_df(df: pd.DataFrame, scale=1_000_000):
    return df.map(taper, scale=scale).fillna(0)

def nan_check(func):
    def _nan_check(*args, **kwargs):
        arr_out = func(*args, **kwargs)
        if np.isnan(arr_out).all():
            print('fuck! '*100)
            print(arr_out)
            for sub in arr_out:
                print(sub)
            raise ValueError("This shouldn't contain a NaN but it does")
        return arr_out
    return _nan_check

class Observation(ABC):
    def __init__(self, 
            controller,
            dtype: np.dtype|str = np.float64,
            *args
        ) -> None:
        self.ac = controller
        self.dtype = np.dtype(dtype) if isinstance(dtype, str) else dtype

    def __call__(self) -> torch.Tensor:
        return self.process_observation(*self.observe())
        # return torch.tensor(flatten(self.observation_space, self.process_observation(*self.observe())))
    
    @property
    @abstractmethod
    def observation_space(self) -> Space:
        pass

    @abstractmethod
    def process_observation(self, *args) -> np.ndarray:
        pass

    @abstractmethod
    def observe(*args, **kwargs) -> tuple:
        pass


class GPObservation(Observation):
    def __init__(self, 
            controller, 
            sb_statfuncs: Sequence[Callable],
            record_len: int,
            gp_best_outvals: Sequence[str],
            dtype: np.dtype|str = np.float64,
            *args
        ) -> None:
        super().__init__(controller, dtype, *args)
        self.gptb_list: list[GPTreebank] = self.ac.gptb_list
        self.gp_vars_out = self.ac.gp_vars_out
        self.gp_vars_core = self.ac.gp_vars_core
        self.sb_statfuncs = sb_statfuncs
        self.record_len = record_len
        self.gp_best_outvals = gp_best_outvals


    # def __call__(self) -> np.ndarray:
    #     return self.process_action(self.observe())
    
    @property
    def observation_space(self) -> Space:
        """What is needed here?

        Best (len(best[data])) << DONE
        Scoreboard?
            -not directly, as it's variable-len
            -use summary statistics for each var
            -correlation matrices? (advanced idea for later)
            -use all possible vars, zero-pad any absences 
            particular GPTB, but leave it in
        Record?
            -again, use all possible vars
            -up to max depth
            -zero-pad missing values
        """
#----#----#----#----#----#----#----#----#----#----#----#----#----#----#----#----
        gp_best_vals = Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(
                len(self.gptb_list),
                len(self.gp_vars_out)
            ),
            dtype=self.dtype # 'float 64'
        )
        gp_scoreboards = Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(
                len(self.gptb_list),
                len(self.gp_vars_core), 
                len(self.sb_statfuncs)
            ),
            dtype=self.dtype # 'float 64'
        )
        gp_records = Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(
                len(self.gptb_list),
                len(self.gp_best_outvals), 
                self.record_len
            ),
            dtype=self.dtype # 'float 64'
        )
        return Tuple([gp_best_vals, gp_scoreboards, gp_records])

    def process_observation(self, 
                            bests, 
                            scoreboards: list[GPScoreboard], 
                            records: list[pd.DataFrame]) -> np.ndarray:
        
        return (
            self.process_bests(bests), 
            self.process_scoreboards(scoreboards),
            self.process_records(records)
        )
        
    @nan_check
    def process_bests(self, bests):
        return np.nan_to_num(
            np.array([
                (
                    np.array([best['data'].get(var, 0.0) for var in self.gp_vars_out], dtype=self.dtype) 
                    if best is not None
                    else np.zeros(len(self.gp_vars_out), dtype=self.dtype)
                ) for best in bests
            ]), 
            nan=0.0
        ) #.clip(np.finfo(self.dtype).min/4.0, np.finfo(self.dtype).max/4.0)

    @nan_check
    def process_scoreboards(self, scoreboards):
        return np.nan_to_num(np.array([
            (
                [
                    (
                        [f(sb[var]) for f in self.sb_statfuncs] 
                        # [f(sb[var][np.isfinite(sb[var])]) for f in self.sb_statfuncs] 
                        if var in sb 
                        else np.zeros(
                            len(self.sb_statfuncs), 
                            dtype=self.dtype
                        )
                    ) 
                    for var 
                    in self.gp_vars_core
                ]
                if sb is not None
                else np.zeros((
                    len(self.gp_vars_core), 
                    len(self.sb_statfuncs)
                ))
            ) 
            for sb 
            in scoreboards
        ], dtype=self.dtype)) #.clip(np.finfo(self.dtype).min/4.0, np.finfo(self.dtype).max/4.0)
    
    @nan_check
    def process_records(self, records: list[pd.DataFrame]):
        return np.nan_to_num(np.array([
            (
                [
                    (
                        np.append(
                            rec[var][:self.record_len], 
                            np.zeros(max(0, self.record_len-len(rec)))
                        )
                        if var in rec
                        else np.zeros(self.record_len)
                    )
                    for var 
                    in self.gp_best_outvals
                ]
                if rec is not None
                else np.zeros(((len(self.gp_best_outvals)), self.record_len))
            )
            for rec 
            in records
        ], dtype=self.dtype)) #.clip(np.finfo(self.dtype).min/4.0, np.finfo(self.dtype).max/4.0)

    def observe(self, *args, **kwargs) -> tuple:
        gp_best_vals = [(gp.best if gp else None) for gp in self.gptb_list]
        gp_scoreboards = [(gp.scoreboard if gp else None) for gp in self.gptb_list]
        gp_records = [(gp.record if gp else None) for gp in self.gptb_list]
        return gp_best_vals, gp_scoreboards, gp_records

class Remembering(Observation):
    def __init__(self, 
            controller, 
            dtype: np.dtype|str = np.float64,
            *args
        ) -> None:
        super().__init__(controller, *args)
        self.repo = self.ac.memory
        self.gp_vars_core = self.ac.gp_vars_core
    
    @property
    def observation_space(self) -> Space:
        """What is needed here?

        MultiContinuous of tables x cols x rows, with zero-padding
        """
#----#----#----#----#----#----#----#----#----#----#----#----#----#----#----#----
        return Box(
            low=-np.inf, 
            high=np.inf,
            dtype=self.dtype, 
            shape=(
                len(self.repo.tables),
                self.repo.tables[0].shape[0],
                len(self.gp_vars_core) # XXX XXX XXX maybe use more vars?
            )
        )

    def process_observation(self, repo) -> np.ndarray:
        return self.process_repo_data(repo)
        
    @nan_check
    def process_repo_data(self, repo: list[pd.DataFrame]):
        try:
            return np.nan_to_num(
                np.array(
                    [clean_df(table[self.gp_vars_core]) for table in repo], 
                    dtype=self.dtype
                )
            )
        except KeyError:
            for table in repo:
                for key in self.gp_vars_core:
                    if key not in table:
                        table[key] = np.zeros(len(table), dtype=self.dtype)
            return self.process_observation(repo)

    def observe(self, *args, **kwargs) -> tuple:
        return (self.repo.tables,)

class LitReview(Remembering):
    def __init__(self, 
            controller, 
            dtype: np.dtype|str = np.float64,
            *args
        ) -> None:
        super().__init__(controller, *args)
        self.repo = self.ac.repository
        self.agent_names: dict[str, int] = self.ac.agent_names
        self.gptb_list: list[GPTreebank] = self.ac.gptb_list
    
    @property
    def observation_space(self) -> Space:
        """What is needed here?

        MultiContinuous of tables x cols x rows, with zero-padding
        Add record of ownership to col - MultiDiscrete, tables x rows, with zero-padding
        """
        return Tuple([
            super().observation_space,
            MultiDiscrete(
                np.ones(
                    (len(self.gptb_list), len(self.repo[0])), dtype=np.int32
                )*(len(self.agent_names)+1) # The extra +1 here is so empty 
            )    # rows can be marked with an out of range value in 'credit'
            #MultiBinary((len(self.gptb_list), len(self.agent_names), len(self.repo[0])))
        ])

    def process_observation(self, repo: list[pd.DataFrame]) -> np.ndarray:
        return (self.process_repo_data(repo), self.process_credits(repo))

    @nan_check
    def process_credits(self, the_literature: list[pd.DataFrame]):
        return np.array([
            table['credit'] for table in the_literature
        ]).astype(np.int32)

    def observe(self, *args, **kwargs) -> tuple:
        return (self.repo.tables, )