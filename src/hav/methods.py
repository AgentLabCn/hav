# basic dependecies
import numpy as np
import pandas as pd
from .utils import run_ABM, log_posterior, construct_causal_graph
import agentpy as ap

# other dependecies
from scipy.stats import ks_2samp # ks test, sensitivity analysis
from scipy.stats import moment  # statistics(MSE, variance, skewness, kurtosis)
import emcee # bayesian
from SALib.sample import saltelli # sensitivity analysis
from SALib.analyze import sobol # sensitivity analysis
import matplotlib.pyplot as plt # bayesian


def MSE(S:list, B:list): 
    mse = np.mean([(sorted(S)[i] - sorted(B)[i]) ** 2 for i in range(len(B))])
    return mse

def variance_test(S:list, B:list): 
    var_S = moment(S, 2)
    var_B = moment(B, 2)
    return var_S, var_B

def skewness_test(S:list, B:list): 
    var_S = moment(S, 3)
    var_B = moment(B, 3)
    return var_S, var_B

def kurtosis_test(S:list, B:list): 
    var_S = moment(S, 4)
    var_B = moment(B, 4)
    return var_S, var_B

def ks_test(S:list, B:list): 
    _, p = ks_2samp(S, B)
    return p

def causal_analysis(S: pd.DataFrame, B: pd.DataFrame): 
    gh_S, gh_B = construct_causal_graph(S.values), construct_causal_graph(B.values)
    overlap = np.sum(gh_S == gh_B) / gh_S.size
    return overlap

def bayesian(ABM: ap.Model, R: dict[str,list[float]], B: dict[str,list[float]], I_Range: dict[str,list[float]], n_walkers = 10, n_steps = 10): 
    ndim = len(I_Range)
    initial_pos = [np.random.uniform(low = I_Range[key][0], high = I_Range[key][1], size = n_walkers) for key in I_Range]
    initial_pos = np.array(initial_pos).T

    sampler = emcee.EnsembleSampler(n_walkers, ndim, log_posterior, args = (ABM, B, I_Range))
    sampler.run_mcmc(initial_pos, n_steps, progress = True)

    samples = sampler.get_chain(discard = int(n_steps/2), thin = 15, flat = True)
    
    # Draw a posterior distribution map
    fig, axes = plt.subplots(ndim, figsize = (10, 7), sharex = True)
    labels = list(I_Range.keys())
    for i in range(ndim): 
        ax = axes[i]
        ax.plot(samples[: , i], "k", alpha = 0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)

    axes[-1].set_xlabel("Step number")
    plt.show()

    return


def sentivity_analysis(ABM: ap.Model, B: dict[str,list[float]], I_Range: dict[str,list[float]], n_samples = 1000): 
     # Define the range of input parameters
    problem = {
        'num_vars': len(I_Range),
        'names': list(I_Range.keys()),
        'bounds': [I_Range[key] for key in I_Range]
    }

    # Generate sampling data
    param_values = saltelli.sample(problem, n_samples)
    
    # Run the model and collect output data
    model_outputs = []
    for params in param_values: 
        input_params = {key: params[i] for i, key in enumerate(I_Range.keys())}
        sim_res = run_ABM(ABM, input_params)
        R = sim_res.variables[list(sim_res.variables)[0]]
        model_outputs.append(R)
    
    # Convert the model output to a DataFrame
    model_outputs_df = pd.DataFrame(model_outputs)
    
    # sensitivity analysis
    sensitivity_indices = {}
    for output_var in model_outputs_df.columns: 
        Y = model_outputs_df[output_var].values
        Si = sobol.analyze(problem, Y)
        sensitivity_indices[output_var] = Si
    
    # Test similarity
    similarity_results = {}
    for output_var in B.keys(): 
        R_vals = model_outputs_df[output_var].values
        B_vals = B[output_var]
        stat, p_value = ks_2samp(R_vals, B_vals)
        similarity_results[output_var] = {
            'stat': stat,
            'p_value': p_value,
            'similar': p_value > 0.05 
        }
    
    return all(indice for _, indice in sensitivity_indices.items)
