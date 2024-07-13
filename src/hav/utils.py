from itertools import chain
import numpy as np
import pandas as pd
from collections.abc import Iterable
import agentpy as ap
from scipy.stats import *

def flatten(data):
    return search_depth(data)

def search_depth(data):
    if not isinstance(data, Iterable): return [data]
    new_data=[]
    for sub_data in data:
        new_data+=search_depth(sub_data)
    return new_data

def log_results(results, filename="results.log"):
    with open(filename, "w") as f:
        for result in results:
            f.write(f"Level: {result.level}, Method: {result.method}, Result: {result.status}, Time: {result.time}, Details: {';'.join(result.details)}")

def run_ABM(ABM:ap.Model, I:dict[str:list[float]]={}):
    for k,v in I.items():
        v=np.reshape(v, np.array(ABM.p[k]).shape)
        ABM.p[k]=v
    ABM.sim_reset()
    return ABM.run(display=False)

# functions called in Bayesian method
def log_prior(params, I_Range):
    for i, param in enumerate(params):
        name = list(I_Range.keys())[i]
        lower, upper = I_Range[name]
        if not (lower <= param <= upper):
            return -np.inf
    return 0

def log_likelihood(params, ABM, B, I_Range):
    I = {list(I_Range.keys())[i]: [param] for i, param in enumerate(params)}
    sim_res = run_ABM(ABM, I)
    O_sim = sim_res.variables[list(sim_res.variables)[0]]
    log_like = 0
    for att in B.keys():
        log_like += np.sum(norm.logpdf(O_sim[att], np.mean(B[att]), np.std(B[att])))
    return log_like

def log_posterior(params, ABM, B, I_Range):
    lp = log_prior(params, I_Range)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(params, ABM, B, I_Range)