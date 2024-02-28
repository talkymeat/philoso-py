from gp import GPTreebank
import pandas as pd
import numpy as np
import operators as ops
import random as rnd
from matplotlib import pyplot as plt
# from figtree import showtree
import seaborn as sns
from gp_trees import GPTerminal, GPNonTerminal
from tree_factories import RandomPolynomialFactory
from typing import Callable
from observatories import FunctionObservatory, StaticObservatory, Observatory
from icecream import ic

def make_var_name():
    alphabet = 'abcdefghijklmnopqrstuvwyz'
    chain = [0]
    while True:
        yield ''.join([alphabet[x] for x in chain])
        i = len(chain)-1
        done = False
        while not done:
            chain[i] += 1
            if chain[i] == len(alphabet):
                chain[i] = 0
                if i == 0:
                    chain = [0] + chain
                    done = True
                else:
                    i -= 1
            else:
                done = True

# def gp_sin_test(
#         n, generations=100, pop=100, iv_min=-5*np.pi, iv_max=5*np.pi, coeff_min=-0.1,
#         coeff_max=0.1, mutation_rate = 0.2, mutation_sd=0.02, crossover_rate=0.2,
#         elitism=5, order=6):
#     gp = GPTreebank(
#         mutation_rate = mutation_rate, mutation_sd=mutation_sd,
#         crossover_rate=crossover_rate,
#         operators=[ops.SUM, ops.PROD, ops.SQ, ops.POW, ops.CUBE]
#     )
#     iv_dict = {'x': [rnd.uniform (iv_min, iv_max) for j in range(n)]}
#     iv_data = pd.DataFrame(iv_dict)
#     target = np.sin(iv_data['x'])
#     res = gp.run_gp(iv_data, target, generations, pop,
#                 elitism=elitism, best_tree=True, rmses=True, best_rmse=True,
#                 tree_factories = RandomPolynomialFactory(
#                     params = np.array([order, coeff_min, coeff_min], dtype=np.float32),
#                     treebank = gp
#                 ))
#     print("Best trees by generation:")
#     for i in range(0, len(res['best_tree']), 10):
#         print(f"Gen {i}: {res['best_tree'][i],}")
#         print(f"RMSE = {res['best_rmse'][i]}")
#     print("="*30)
#     print("Best tree:")
#     print(res['best_tree'][-1])
#     showtree(res['best'])
#     print(f"RMSE = {res['best_rmse'][-1]}")
#     estimate = res['best']()
#     if len(estimate) == 1:
#         estimate = pd.Series([estimate[0]]*len(target))
#     fig, axs = plt.subplots(3, 2)
#     fig.set_size_inches(18.5, 27, forward=True)
#     axs[0, 0].plot(res['best_mse'])
#     axs[0, 0].set_title("Best MSE")
#     axs[0, 1].plot(res['best_rmse'])
#     axs[0, 1].set_title("Best RMSE")
#     sns.heatmap(pd.DataFrame(res['mses']).transpose(), ax=axs[1, 0])
#     axs[1, 0].set_title("MSE Heatmap")
#     sns.heatmap(pd.DataFrame(res['rmses']).transpose(), ax=axs[1, 1])
#     axs[1, 1].set_title("RMSE Heatmap")
#     axs[2, 0].scatter(iv_data['x'], estimate, c='b', label='target')
#     axs[2, 0].scatter(iv_data['x'], target, c='r', label = 'est')

#     return res

# def target_tree_func(target_tree: GPNonTerminal, ivs: pd.Series) -> pd.Series:
#     def 
#         target_tree(**ivs)
OPSET = [ops.SUM, ops.PROD, ops.SQ, ops.POW, ops.CUBE]
  
def uniform_iv_func(iv_min: float, iv_max: float):
    def uniform_iv(obs_len: int, **kwargs):
        return pd.Series([rnd.uniform (iv_min, iv_max) for i in range(obs_len)])
    return uniform_iv

def target_poly_func(var: str, *vars: str, order: int, const_min: float, const_max: float):
    return RandomPolynomialFactory(
        params = np.array([order, const_min, const_min], dtype=np.float32),
        treebank = GPTreebank(operators=OPSET)
    )(*(var,)+vars)

def sin(x=None):
    return np.sin(x)

def poly_obs_factory(
        n: int, order: int, 
        const_min: float, const_max: float, 
        iv_min: float, iv_max: float
    ) -> FunctionObservatory:
    return FunctionObservatory(
        'x', 'y', 
        {
            'x': uniform_iv_func(iv_min, iv_max), 
            'y': target_poly_func(
                'x', 
                order=order, 
                const_min=const_min, 
                const_max=const_max
            )
        }, 
        n
    )

def func_obs_factory(
        n: int, f: Callable, iv_min: float, iv_max: float
    ) -> FunctionObservatory:
    return FunctionObservatory(
        'x', 'y', 
        {
            'x': uniform_iv_func(iv_min, iv_max), 
            'y': f
        }, 
        n
    )

EXAMPLE_POLY_OBS_FAC = poly_obs_factory(
    100, 
    order=6, 
    const_min=-0.1, 
    const_max=0.1, 
    iv_min=-5*np.pi, 
    iv_max=5*np.pi
)

def gp_rand_poly_test(
        n, generations=100, pop=100, iv_min=-5*np.pi, iv_max=5*np.pi, coeff_min=-0.1,
        coeff_max=0.1, mutation_rate = 0.2, mutation_sd=0.02, crossover_rate=0.2,
        elitism=5, order=6, def_fitness=None, temp_coeff=1.0, max_size=300, 
        max_depth=70, to_graph=[
            ['penalty', 'hasnans', 'survive'], 
            ["mse", "rmse", 'raw_fitness'], 
            ['garn_time', 'score_time', 'kbest_time'], 
            ['mut8_time', 'kill_time', 'record_time'],
            ["size", "depth", 'time']
    ]):
    obs = poly_obs_factory(
        n, 
        order=order, 
        const_min=coeff_min, const_max=coeff_max, 
        iv_min=iv_min, iv_max=iv_max
    )
    return gp_func_test(
        obs, 
        generations=generations, 
        pop=pop, 
        iv_min=iv_min, 
        iv_max=iv_max, 
        coeff_min=coeff_min,
        coeff_max=coeff_max, 
        mutation_rate=mutation_rate, 
        mutation_sd=mutation_sd, 
        crossover_rate=crossover_rate,
        elitism=elitism, 
        order=order, 
        def_fitness=def_fitness, 
        temp_coeff=temp_coeff, 
        max_size=max_size, 
        max_depth=max_depth, 
        to_graph=to_graph
    )

def gp_sin_test(
        n, generations=100, pop=100, iv_min=-5*np.pi, iv_max=5*np.pi, coeff_min=-0.1,
        coeff_max=0.1, mutation_rate = 0.2, mutation_sd=0.02, crossover_rate=0.2,
        elitism=5, order=6, def_fitness=None, temp_coeff=1.0, max_size=300, 
        max_depth=70, to_graph=[
            ['penalty', 'hasnans', 'survive'], 
            ["mse", "rmse", 'raw_fitness'], 
            ["size", "depth", 'time']
    ]):
    obs = func_obs_factory(
        n, f=sin,
        iv_min=iv_min, iv_max=iv_max
    )
    return gp_func_test(
        obs, 
        generations=generations, 
        pop=pop, 
        iv_min=iv_min, 
        iv_max=iv_max, 
        coeff_min=coeff_min,
        coeff_max=coeff_max, 
        mutation_rate=mutation_rate, 
        mutation_sd=mutation_sd, 
        crossover_rate=crossover_rate,
        elitism=elitism, 
        order=order, 
        def_fitness=def_fitness, 
        temp_coeff=temp_coeff, 
        max_size=max_size, 
        max_depth=max_depth, 
        to_graph=to_graph
    )

def gp_func_test(
        obs, generations=100, pop=100, iv_min=-5*np.pi, iv_max=5*np.pi, coeff_min=-0.1,
        coeff_max=0.1, mutation_rate = 0.2, mutation_sd=0.02, crossover_rate=0.2,
        elitism=5, order=6, def_fitness=None, temp_coeff=1.0, max_size=300, 
        max_depth=70, to_graph=[
            ['penalty', 'hasnans', 'survive'], 
            ["mse", "rmse", 'raw_fitness'], 
            ["size", "depth", 'time']
    ]):
    gp = GPTreebank(
        mutation_rate = mutation_rate, mutation_sd=mutation_sd,
        crossover_rate=crossover_rate, max_size=max_size, max_depth=max_depth,
        operators=OPSET
    )
    res, data, final_tree = gp.run(
        RandomPolynomialFactory(
            params = np.array([order, coeff_min, coeff_max], dtype=np.float32),
            treebank = gp
        ),
        obs, 
        generations, pop,
        def_fitness=def_fitness, 
        elitism=elitism,
        temp_coeff=temp_coeff,
        to_graph=to_graph
    )
    print("Best trees by generation:")
    n = len(res['tree'])
    for i in range(min(10, n)):
        print(f"Gen {i}: {res['tree'][i],}")
        print(f"RMSE = {res['rmse'][i]}")
    for i in range(min(10, n), n, 10):
        print(f"Gen {i}: {res['tree'][i],}")
        print(f"RMSE = {res['rmse'][i]}")
    print("="*30)
    print("Best tree:")
    print(final_tree)
    # showtree(final_tree)
    print(res['rmse'])
    print(f"RMSE = {res['rmse'][n-1]}")
    # XXX FIXME XXX
    # obs.set_obs_len(10*5)
    # show_data = {}
    # show_data['x'] = np.linspace(iv_min, iv_max, 1000)
    # show_data['est_y'] = final_tree(x=show_data['x'])
    # obs.set_iv_data(show_data['x'])
    # show_data['y'] = obs.target
    # fig, axs = plt.subplots(3, 2)
    # fig.set_size_inches(18.5, 27, forward=True)
    # axs[0, 0].plot(res['mse'])
    # axs[0, 0].set_title("Best MSE")
    # axs[0, 1].plot(res['rmse'])
    # axs[0, 1].set_title("Best RMSE")
    # sns.heatmap(pd.DataFrame(res['mse']).transpose(), ax=axs[1, 0])
    # axs[1, 0].set_title("MSE Heatmap")
    # sns.heatmap(pd.DataFrame(res['rmse']).transpose(), ax=axs[1, 1])
    # axs[1, 1].set_title("RMSE Heatmap")
    # axs[2, 0].plot(show_data['x'], show_data['est_y'], c='b', label='estimate')
    # axs[2, 0].plot(show_data['x'], show_data['y'], c='r', label = 'target')
    # plt.savefig(f'plots_2_{gp.make_filename()}.png')
    return res, final_tree

# def gp_poly_test(order, num_vars, n, generations=100, pop=100, iv_min=-100,
#                 iv_max=100, coeff_min=-20, coeff_max=20, mutation_rate = 0.2,
#                 mutation_sd=1.0, elitism=5):
#     gp = GPTreebank(
#         mutation_rate = mutation_rate, mutation_sd=mutation_sd,
#         operators=[ops.SUM, ops.PROD, ops.SQ, ops.POW, ops.CUBE]
#         )
#     var_name_maker = make_var_name()
#     iv_dict = {}
#     for i in range(num_vars):
#         iv_dict[next(var_name_maker)] = [rnd.uniform (iv_min, iv_max) for j in range(n)]
#     iv_data = pd.DataFrame(iv_dict)
#     factory = RandomPolynomialFactory(
#         params = np.array([order, coeff_min, coeff_min], dtype=np.float32),
#         treebank = gp
#     )
#     target_poly = factory(iv_data)
#     print('Target:')
#     showtree(target_poly)
#     target = target_poly()
#     res = gp.run_gp(iv_data, target, generations, pop,
#                 elitism=elitism, best_tree=True, rmses=True, best_rmse=True,
#                 tree_factories = factory)
#     print("Best trees by generation:")
#     for i in range(0, len(res['best_tree']), 10):
#         print(f"Gen {i}: {res['best_tree'][i],}")
#         print(f"RMSE = {res['best_rmse'][i]}")
#     print("="*30)
#     print("Best tree:")
#     print(res['best_tree'][-1])
#     print(f"RMSE = {res['best_rmse'][-1]}")
#     fig, axs = plt.subplots(2, 2)
#     fig.set_size_inches(18.5, 18.5, forward=True)
#     axs[0, 0].plot(res['best_mse'])
#     axs[0, 0].set_title("Best MSE")
#     axs[0, 1].plot(res['best_rmse'])
#     axs[0, 1].set_title("Best RMSE")
#     sns.heatmap(pd.DataFrame(res['mses']).transpose(), ax=axs[1, 0])
#     axs[1, 0].set_title("MSE Heatmap")
#     sns.heatmap(pd.DataFrame(res['rmses']).transpose(), ax=axs[1, 1])
#     axs[1, 1].set_title("RMSE Heatmap")
#     return res

# def fac(x: int):
#     if x < 0:
#         raise ValueError('!x undefined for x<0')
#     elif x == 0:
#         return 1
#     return x * fac(x-1)

# def random_polynomial_func(
#         min_deg: int|list=3, max_deg: int=7, min_coeff=-10, max_coeff=10,
#         coeff_taper=False, decompose=False
#     ):
#     if isinstance(min_deg, list):
#         deg = len(min_deg)-1
#         coeffs = min_deg
#         exponents = list(range(len(coeffs)))
#     else:
#         if min_deg<1:
#             raise ValueError('Minimum degree must be at least 1')
#         if min_deg>max_deg:
#             raise ValueError(
#                 'Minimum degree should be less than or equal to maximum degree'
#             )
#         if min_coeff>max_coeff:
#             raise ValueError(
#                 'Minimum coefficient should be less than or equal to maximum' +
#                 ' coefficient'
#             )
#         deg = rnd.randint (min_deg, max_deg)
#         exponents = list(range(deg+1))
#         coeffs = [rnd.uniform (min_coeff/fac(p), max_coeff/fac(p)) for p in exponents]
#     pairs = []
#     component_funcs = list(map(lambda pair: lambda x: pair[0] * x**pair[1], zip(coeffs, exponents)))
#     polynomial_func = lambda x: sum([f(x) for f in component_funcs])
#     if decompose:
#         return lambda x: [f(x) for f in ([polynomial_func] + component_funcs)]
#     return lambda x: sum([b * x**p for b, p in zip(coeffs, exponents)])

if __name__ == '__main__':
    result = gp_rand_poly_test(500, generations=5, mutation_sd=0.3)
    print(result)
    # result = gp_sin_test(500, generations=2000, mutation_sd=0.3)