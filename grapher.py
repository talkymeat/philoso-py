import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from icecream import ic
from typing import Collection
import operators as ops
from trees import *
from gp import GPTreebank
from tree_factories import RandomPolynomialFactory
import warnings

# XXX TODO Docs & comments
class Grapher:
    def __init__(self, names, subplot_size=4):
        """Creates a plot or grid of plots that can be updated with 
        live updating data

        TODO: Allow plots with multiple lines
        """
        # turn on interactive mode
        plt.ion()
        self.axnames = {}
        self.linenames = {}
        h, v = 1, 1
        if isinstance(names, Collection) and not isinstance(names, str):
            h = len(names)
            for i, n in enumerate(names):
                if isinstance(n, Collection) and not isinstance(n, str):
                    v = max(len(n), v)
                else:
                    names[i] = [n]
        else:
            names = [[names]]
        self.fig, self.axs = plt.subplots(
            v, h, figsize=(subplot_size*h, subplot_size*v)
        )
        if h!=1 and v!=1:
            for ih, name_col in enumerate(names):
                for jv, name in enumerate(name_col):
                    self.axs[jv, ih].set_title(name)
                    self.axnames[name] = self.axs[jv, ih]
                    self.d = 2
        elif h!=1:
            for ih, name in enumerate(names):
                self.axs[ih].set_title(name[0])
                self.axnames[name[0]] = self.axs[ih]
                self.d = 1
        elif v!=1:
            for jv, name in enumerate(names[0]):
                self.axs[jv].set_title(name)
                self.axnames[name] = self.axs[jv]
                self.d = 1
        else:
            self.axs.set_title(names[0][0])
            self.axnames[names[0][0]] = self.axs
            self.d = 0

    def __getitem__(self, key: str):
        return self.axnames[key]
    
    
    def plot_data(self, **data):
        for k, v in data.items():
            if k in self.axnames:
                line = self.axnames[k].plot(v)
                self.linenames[k] = line
                plt.draw()
                plt.pause(0.01)
    
    def set_data(self, **data):
        for k, v in data.items():
            if k in self.axnames:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    y = np.array(v)
                    self.linenames[k][0].set_data(np.arange(len(v)), y)
                    self.axnames[k].set_ylim(y.min(), y.max())
                    self.axnames[k].set_xlim(0, len(y))
                    plt.draw()
                    plt.pause(0.01)
    
    def ioff(self):
        plt.ioff()

def grapher_example(ms = 300, md = 70, lims= True):
    gp = GPTreebank(
        mutation_rate = 0.2, 
        mutation_sd=0.02, 
        crossover_rate=0.5, 
        max_depth=md if lims else 0, 
        max_size=ms if lims else 0, 
        operators=[ops.SUM, ops.PROD, ops.SQ, ops.POW, ops.CUBE]
    )
    gr = None
    rpf = RandomPolynomialFactory(gp, 5, -10.0, 10.0)
    trees = [rpf('x', 'y') for _ in range(5)]
    df = pd.DataFrame({'x': [1.0, 1.0], 'y': [1.0, 1.0]})
    bigtrees, deeptrees = 0, 0
    valmaxes = []
    sizemaxes = []
    depthmaxes = []
    bigness = []
    deepness = []
    for _ in range(200):
        tmax = None
        valmax = -np.inf
        for t in trees:
            val = t(**df)
            if isinstance(val, pd.Series):
                val = val.sum()
            elif valmax < val:
                tmax = t
                valmax = val
        newtrees = [tmax.copy(gp_copy=True) for _ in range(5)]
        for tt in trees:
            tt.delete()
        trees = newtrees
        bigtrees += bool([tr for tr in trees if tr.size() > ms])
        if bool([tr for tr in trees if tr.size() > ms]):
            bigness.append([(tr.size(), '>', ms) for tr in trees if tr.size() > ms])
        if bool([tr for tr in trees if tr.depth() > md]):
            deepness.append([(tr.depth(), '>', md) for tr in trees if tr.depth() > md])
        deeptrees += bool([tr for tr in trees if tr.depth() > md])
        sizemaxes.append(max([_t.size() for _t in trees]))
        depthmaxes.append(max([_t.depth() for _t in trees]))
        valmaxes.append(valmax)
        if _:
            results = pd.DataFrame({'sizemaxes': sizemaxes, 'depthmaxes': depthmaxes, 'valmaxes': valmaxes})
            results.loc[results.valmaxes == np.inf, 'valmaxes'] = results.loc[
                results.valmaxes != np.inf, 'valmaxes'
            ].max()
            logs = np.emath.logn(results.max(), results)
            lognormresults = pd.DataFrame()
            for i in range(3):
                lognormresults[f"log_norm_{results.columns[i]}"] = logs[:, i]
            full_results = pd.concat([results, lognormresults], axis=1)
            if _==1:
                gr = Grapher(list(zip(results.columns, lognormresults.columns)))
                gr.plot_data(**full_results)
            if _>1:
                gr.set_data(**full_results)
    gr.ioff()

    print("="*80)
    print(f"{bigtrees}u{deeptrees}") # 0u0
    return sizemaxes, depthmaxes, valmaxes, bigness, deepness

def main():
    sizemaxes, depthmaxes, valmaxes, bigness, deepness = grapher_example()
    ic(max(sizemaxes))
    ic(max(depthmaxes))
    ic(bigness)
    ic(deepness)

if __name__ == '__main__':
    main()
