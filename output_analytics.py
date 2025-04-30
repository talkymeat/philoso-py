import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import torch
from gymnasium.spaces.utils import unflatten

from os.path import sep
from glob import glob
import re
from pathlib import Path
from functools import reduce
from collections.abc import Sequence

from hd import HierarchicalDict as HD
from world import SineWorld
from philoso_py import ModelFactory
from agent import Agent

from icecream import ic


def _path(p: str|Path) -> Path:
    if isinstance(p, str):
        return Path(p)
    return p

def get_data(root_dir: str) -> Sequence[pd.DataFrame]:
    return sorted(glob(f'{Path(root_dir)}/**', recursive=True))

def sort_folders(file_str_list: list[str], root_dir: str|Path) -> HD:
    folders = HD()
    for i, path_ in enumerate(file_str_list):
        path__ = path_[len(str(root_dir))+1:].split('/')
        k, v = path__[:-1] + [''], path__[-1]
        if '.' in v:
            folders[k] = folders.get(k, []) + [v]
    return folders

def get_sorted_data(root_dir: str|Path) -> HD:
    return sort_folders(get_data(root_dir), root_dir)

def get_dirs_only(root_dir: str|Path) -> list[str]:
    return [path for path in glob(f'{_path(root_dir)}{sep}**') if Path(path).is_dir()]

def agents_from_dirs(root_dir: str|Path) -> list[str]:
    root = _path(root_dir)
    agents = [
        dir_.split(sep)[-1]
        for dir_
        in get_dirs_only(root_dir) 
        if dir_ != f'{root / 'publications'}' 
    ]
    return sorted(agents)

def agent_day_data(root_dir: str|Path, agent: str):
    ic.enable()
    root = _path(root_dir)
    folders = get_sorted_data(root)
    for _agent in agents_from_dirs(root):
        if _agent != agent:
            continue
        _agent_data = []
        addr = [_agent, 'days', '']
        agent_days = folders[addr]
        for j in range(4):
            dfs = agent_days[j::4]
            short = len(dfs[0])
            fnames = [
                fname for fname in agent_days[j::4] if len(fname)==short
            ] + [
                fname for fname in agent_days[j::4] if len(fname)>short
            ]
            datapath = root / addr[0] / addr[1]
            datapaths = [
                (datapath / fname) for fname in fnames
            ]
            _agent_data.append([pd.read_csv(dp) if str(dp).endswith('.csv') else pd.read_parquet(dp) for dp in datapaths])
        return (
            _agent, 
            AgentMemData(_agent, _agent_data[1:], root, folders), 
            AgentDeedData(_agent, _agent_data[0], root, folders)
        )

def cols_from_tensor_col(col: pd.Series):
    width = 0
    idx = 0
    while not width:
        if isinstance(col[idx], str):
            width = tensor_str_2_np(val).shape[0]
    mat = np.array([detensorise(item) for item in col])
    return mat.T

def detensorise(val: str|float, width: int) -> np.ndarray:
    if isinstance(val, float) and np.isnan(val):
        return np.array([np.nan]*width)
    else:
        return tensor_str_2_np(val)

def tensor_str_2_np(val:str):
    return tensor_str_2_tensor(val).numpy()[0]

def tensor_str_2_tensor(val:str):
    return eval('torch.'+val)

def strip_tensor_str_kwargs(val:str):
    m = re.match(r'(tensor\(\[[0-9]+\]).*(\))', val)
    return m[1] + m[2]
    

class AgentMemData:
    def __init__(self, name, memories, root, folders) -> None:
        self.name = name
        self.memories = memories
        self.root = root
        self.folders = folders
        self._mem_summary_data = None
        self._make_obs_width()
        self.make_model()
        self.agent = [agt for agt in self.model.agents if agt.name==self.name][0]
        
    def make_model(self):
        json_file = [file for file in self.folders[''] if file.endswith('.json')][0]
        json_file = Path(self.root, json_file)
        self.model = ModelFactory().from_json(json_file)

    def _make_obs_width(self) -> None:
        for dfs in self.memories:
            for df in dfs:
                df['obs_width'] = (df['obs_stop']-df['obs_start']).abs()

    @property
    def memory_summary(self) -> pd.DataFrame:
        if self._mem_summary_data is not None:
            return self._mem_summary_data
        return self.compute_memory_summary()

    # @property
    # def all_data(self) -> list[pd.DataFrame]:
    #     return [self.deeds, *self.memories]
    
    @property
    def mem_cols(self) -> pd.Index:
        if hasattr(self, 'memories') and self.memories:
            return self.memories[0][0].columns

    def compute_memory_summary(self) -> pd.DataFrame:
        return pd.concat(
            [
                pd.DataFrame([
                    getattr(
                        df[self.mem_cols.drop(['t', 'tree', 'exists'])], 
                        stat
                    )() 
                    for df 
                    in mems
                ]).rename(columns=lambda n: f'{n}_{stat}_{i}') 
                for stat 
                in ('min', 'mean', 'std', 'max') 
                for i, mems 
                in enumerate(self.memories)
            ], 
            axis=1
        ).sort_index(axis=1)

    def plot_mem_col_one_stat(self, col, stat):
        self.memory_summary[[f'{col}_{stat}_{i}' for i in range(len(self.memories))]].plot()

    def plot_mem_col(self, col):
        for stat in ('min', 'mean', 'std', 'max'):
            self.plot_mem_col_one_stat(col, stat)

    def some_mem_graphs(self, start: int, num: int):
        go, end = start, start+num
        cols = self.mem_cols.drop(['t', 'tree', 'exists'])
        while go < len(cols)-3:
            for col in cols[go:min(end, len(cols))]:
                print(col)
                self.plot_mem_col(col)
            yield
            go, end = end, end+num

    def mem_t_vals_plot(self):
        for j, dfs in enumerate(self.memories):
            t_vals = pd.DataFrame()
            for i in range(6):
                t_vals[f'mem_{j}_register_{i}'] = [df.at[i, 't'] for df in dfs]
            t_vals.plot()

    @property
    def memT(self) -> list[pd.DataFrame]:
        data = [pd.concat(dfs, ignore_index=True) for dfs in zip(*self.memories)]
        return [df[df['exists']] for df in data]
            
    def get_best_mems(self, targ: str, max_=True):
        maxmin = max_-0.5
        return pd.DataFrame([
            df[df[targ]*maxmin==(df[targ]*maxmin).max()].squeeze() 
            for df 
            in self.memT
        ]).reset_index()
    
    def plot_best_mems(self, targ, max_=True):
        bests = self.get_best_mems(targ, max_=max_)
        for col in bests.drop(['t', 'tree', 'exists', 'index'], axis=1):
            pd.DataFrame({col: bests[col]}).plot(title=col)

    def regress_all(self, targ: str, clip: float=None):
        print(targ)
        data = pd.concat(self.memT)
        if clip is not None:
            mask = data[targ].abs() < clip
        else:
            mask = pd.Series(True, data[targ].index)
        data = data[mask]
        for col in data.columns.drop(['t', 'tree', 'exists', targ]):
            data.plot(x=col, y=targ, kind='scatter', title=f'{col} vs {targ}')

    def _single_df_act_counts(self, df: pd.DataFrame):
        counts = df["Action"].value_counts().to_frame().T
        counts.columns.name = None
        for act in self.act_sort_order:
            if act not in counts:
                counts[act] = 0
        return counts.reset_index().drop('index', axis=1)[self.act_sort_order]

    @property
    def act_counts(self):
        return pd.concat([self._single_df_act_counts(df) for df in self.deeds]).reset_index().drop('index', axis=1)

    def plot_act_counts(self):
        self.act_counts.plot.area(stacked=True, figsize=(12,10))
        self.plot(subplots=True, figsize=(12,10))
        


class AgentDeedData:
    ACTION_NAMES = {
        'tensor([0])': 'New GP', 
        'tensor([2])': 'New GP with Mems', 
        'tensor([1])': 'Continue GP', 
        'tensor([3])': 'Continue GP with Mems', 
        'tensor([4])': 'Store Mem', 
        'tensor([5])': 'Publish', 
        'tensor([6])': 'Read'
    }
    ACTS = {
        'tensor([0])': 'gp_new', 
        'tensor([2])': 'gp_new,use_mem', 
        'tensor([1])': 'gp_continue', 
        'tensor([3])': 'gp_continue,use_mem', 
        'tensor([4])': 'store_mem', 
        'tensor([5])': 'publish', 
        'tensor([6])': 'read'
    }
    MEANABLES = [
        'obs_start', 'obs_stop', 'obs_width', 'obs_num', 'temp_coeff',  'pop',
        'crossover_rate', 'mutation_rate', 'mutation_sd', 'max_depth',
        'max_size', 'episode_len', 'elitism', 'elitism_normed', 
    ]
    MUT8OR_W8S = ['mutator_weights_0', 'mutator_weights_1']
    STORE_MEMS = ['store_mem_gp_0_table', 'store_mem_gp_0_row', 'store_mem_gp_1_table', 'store_mem_gp_1_row']
    READ = [
        'memory_loc_0_table', 'memory_loc_1_table', 'memory_loc_2_table', 
        'memory_loc_0_row', 'memory_loc_1_row', 'memory_loc_2_row', 
        'journal_loc_0_table', 'journal_loc_1_table', 'journal_loc_2_table', 
        'journal_loc_0_row', 'journal_loc_1_row', 'journal_loc_2_row'
    ]
    PUBLISH = ['journal_num', 'gp_register']
    USE_MEM = ['locations']
    
    def __init__(self, name, deeds, root, folders) -> None:
        self.name = name
        self.deeds = deeds
        self.act_sort_order = list(self.ACTION_NAMES.values())
        for df in self.deeds:
            act_ = df["('act', 'choice')"].apply(strip_tensor_str_kwargs)
            df["Action"] = act_.map(self.ACTION_NAMES)
            df["('act', 'choice')"] = act_.map(self.ACTS)
        self.folders = folders
        self.root = root
        self.make_model()
        self.agent = [agt for agt in self.model.agents if agt.name==self.name][0]
        self.process_action_tensors()
        self._gp_registers = None
        self.track_gp_ages()
        self.expand_observations()

    def _single_df_act_counts(self, df: pd.DataFrame):
        counts = df["Action"].value_counts().to_frame().T
        counts.columns.name = None
        for act in self.act_sort_order:
            if act not in counts:
                counts[act] = 0
        return counts.reset_index().drop('index', axis=1)[self.act_sort_order]

    @property
    def act_counts(self):
        return pd.concat([self._single_df_act_counts(df) for df in self.deeds]).reset_index().drop('index', axis=1)

    def plot_act_counts(self):
        self.act_counts.plot.area(stacked=True, figsize=(12,10))
        self.act_counts.plot(subplots=True, figsize=(12,10))
        
    def make_model(self):
        json_file = [file for file in self.folders[''] if file.endswith('.json')][0]
        json_file = Path(self.root, json_file)
        self.model = ModelFactory().from_json(json_file)

    def process_action_tensors(self):
        post = PostProcessor(self.agent)
        for df in self.deeds:
            post(df)

    @property
    def means(self):
        return pd.DataFrame({
            f'{col}_{f}': [getattr(df[col], f)() for df in self.deeds] 
            for col 
            in (self.MEANABLES+self.MUT8OR_W8S) 
            for f 
            in ('mean', 'std', 'min', 'max')
        })
    
    def plot_action_means(self):
        for col in self.MEANABLES:
            plt.figure()
            idx = self.means[f'{col}_mean'].index
            mean = self.means[f'{col}_mean']
            plt.plot(idx, self.means[f'{col}_mean'])
            plt.fill_between(
                idx, 
                self.means[f'{col}_min'], 
                self.means[f'{col}_max'], 
                color="b", 
                alpha=0.2
            )
            plt.fill_between(
                idx, 
                mean - self.means[f'{col}_std'],
                mean + self.means[f'{col}_std'], 
                color="r", 
                alpha=0.3
            )
            plt.title(col)

    def plot_mutator_weights(self):
        self.means[[
            f'{mw}_mean' for mw in self.MUT8OR_W8S
        ]].plot.area(
            stacked=True, figsize=(14,5)
        )

    @property
    def gp_registers(self):
        if self._gp_registers is None:
            self._gp_registers = reduce(
                lambda x, y: x|y,
                [
                    set(df['gp_register'][df['gp_register'].notna()].unique()) 
                    for df 
                    in self.deeds
                ],
                set()
            )
        return self._gp_registers

    def track_gp_ages(self):
        def bias(row):
            denom = np.mean([row[x] for x in self.gp_age_cols.values() if x != self.gp_age_cols[reg_]]).item()
            if denom:
                return row[self.gp_age_cols[reg_]]/denom 
            return np.nan
        age_trackers = {
            reg: GPAgeTracker(reg) 
            for reg 
            in self.gp_registers
        }
        for df in self.deeds:
            for reg_ in self.gp_registers:
                df[self.gp_age_cols[reg_]] = [age_trackers[reg_](row) for row in df.iloc]
            for reg_ in self.gp_registers:
                df[self.age_bias_cols[reg_]] = df.apply(bias, axis=1)

    @property
    def gp_age_cols(self):
        if not self.gp_registers:
            return {}
        return {
            reg: f'GP {reg} Age' 
            for reg 
            in self.gp_registers
        }

    @property
    def age_bias_cols(self):
        if not self.gp_registers:
            return {}
        return {
            reg: f'Age Bias {reg}' 
            for reg 
            in self.gp_registers
        }

    @property
    def gp_ages_df(self):
        return pd.concat([
            df[
                ['Action', 'gp_register'] +
                list(self.gp_age_cols.values()) +
                list(self.age_bias_cols.values())
            ].copy() 
            for df 
            in self.deeds
        ]).reset_index()
    
    def plot_gp_ages(self):
        self.gp_ages_df[self.gp_age_cols.values()].plot()
        self.gp_ages_df[self.age_bias_cols.values()].plot()

    @property
    def age_publication_bias(self):
        def apb(df):
            publications = df[df['Action']=='Publish']
            return {
                'Mean Pub Age': (
                    publications.apply(
                        lambda row: row[f'Age Bias {row["gp_register"]}'], 
                        axis=1
                    ).mean() if len(publications) else np.nan
                ),
                'Mean Age': df[
                    self.gp_age_cols.values()
                ].mean().mean()
            }
        return pd.DataFrame([apb(df) for df in self.deeds])
    
    def plot_age_pub_bias(self):
        apb = self.age_publication_bias
        apb['Mean Age'].plot(color='r')
        apb['Mean Pub Age'].plot(color='g')

    def expand_observations(self):
        pass


class PostProcessor:
    """Make this a class, store the pop values of new GPs"""
    def __init__(self, agent: Agent):
        self.gp_pops = {}
        self.agent = agent
        
    def process_action(self, action: str, row: pd.Series):
        cols = [idx for idx in row.index if idx.startswith('(') and eval(idx)[:2]==('act', action)]
        col_tuples = [eval(col) for col in cols]
        in_vals = {
            col_tuple[2]: tensor_str_2_tensor(row[col])
            for col, col_tuple
            in zip(cols, col_tuples)
        }
        return self.agent.actions[col_tuples[0][1]].interpret(in_vals)

    def process_act_from_row(self, row: pd.Series):
        actions = row["('act', 'choice')"].split(',')
        return tuple([self.process_action(action, row) for action in actions])

    def __call__(self, df):
        df['elitism_normed'] = pd.Series([None]*len(df))
        for i, row in enumerate(df.iloc):
            for act in self.process_act_from_row(row):
                for k, v in act.items():
                    if k not in df:
                        df[k] = pd.Series([None]*len(df))
                    df.at[i, k] = v
            row = df.iloc[i]
            if 'New GP' in row['Action']:
                self.gp_pops[row['gp_register']] = row['pop']
                df.at[i, 'elitism_normed'] = row['elitism']/row['pop']
            elif 'Continue GP' in row['Action']:
                pop =self.gp_pops.get(row['gp_register'], -1)
                if pop > -1:
                    df.at[i, 'elitism_normed'] = row['elitism']
                    df.at[i, 'elitism'] = int(row['elitism']*pop)
        self.clear_old_cols(df)
        self.decrap(df)
    
    def clear_old_cols(self, df):
        for col in df.columns:
            if '(' in col:
                df.drop(col, inplace=True, axis=1)

    def decrap(self, df):
        for crap in ('Unnamed: 0', 'tf_choices', 'tf_weights', 'sb_weights'):
            if crap in df:
                df.drop(crap, inplace=True, axis=1)

class GPAgeTracker:
    def __init__(self, reg: int, prior:int=0):
        self.reg = reg
        self.current = prior

    def __call__(self, row: pd.Series):
        if row['gp_register']==self.reg:
            if row['Action'].startswith("New GP"):
                self.current = 1
            elif row['Action'].startswith("Continue GP"):
                if self.current > 0:
                    self.current += 1
        return self.current