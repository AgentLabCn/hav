# basic dependecies
import numpy as np
import pandas as pd
from .utils import run_ABM, flatten, log_posterior
import agentpy as ap

# other dependecies
from scipy.stats import ks_2samp # ks test, sensitivity analysis
from scipy.stats import moment  # statistics(MSE, variance, skewness, kurtosis)
import catboost # surrogate analysis
from pgmpy.estimators import PC # causal analysis
import emcee # bayesian
from SALib.sample import saltelli # sensitivity analysis
from SALib.analyze import sobol # sensitivity analysis
from statsmodels.tsa.api import VAR # causal analysis
import matplotlib.pyplot as plt # bayesian


def MSE(R: dict[str,list[float]], B: dict[str,list[float]], rtol: float = 0.05): 
    atts = R.keys() if R.keys() ==  B.keys() else []
    report = pd.DataFrame([[0 for _ in atts] for _ in range(2)], columns = atts, index = ['result','MSE'])
    for att in atts: 
        report[att]['MSE'] = np.mean((np.array(R[att]) - np.array(B[att]))**2)
        report[att]['result'] = int(report[att]['MSE'] <=  (rtol*np.mean(B[att]))**2)
    details = [f"*{att} {'passed' if report[att]['result']>0 else 'failed'} MSE test, MSE = {report[att]['MSE']}"\
              for att in atts]
    return all(report[att]['result']>0 for att in atts), details


def variance_test(R: dict[str,list[float]], B: dict[str,list[float]], rtol: float = 5e-2): 
    atts = R.keys() if R.keys() == B.keys() else []
    report = pd.DataFrame([[0 for _ in atts] for _ in range(3)], columns = atts, index = ['result','R var','B var'])
    for att in atts: 
        # var
        report[att]['R var'], report[att]['B var'] = moment(R[att],2), moment(B[att],2)
        report[att]['result'] = 1 if np.isclose(report[att]['R var'], report[att]['B var'], rtol = rtol) else 0

    details = [f"*{att} {'passed' if report[att]['result']>0 else 'failed'} variance test, Variance(R) = {report[att]['R var']}, Variance(B) = {report[att]['B var']}"\
             for att in atts]

    return all(report[att]['result']>0 for att in atts), details

def skewness_test(R: dict[str,list[float]], B: dict[str,list[float]], rtol: float = 5e-2): 
    atts = R.keys() if R.keys() == B.keys() else []
    report = pd.DataFrame([[0 for _ in atts] for _ in range(3)], columns = atts, index = ['result','R skew','B skew'])
    for att in atts: 
        # skew
        report[att]['R skew'], report[att]['B skew'] = moment(R[att],3), moment(B[att],3)
        report[att]['result'] = 1 if np.isclose(report[att]['R skew'], report[att]['B skew'], rtol = rtol) else 0

    details = [f"*{att} {'passed' if report[att]['result']>0 else 'failed'} skewness test, Skewness(R) = {report[att]['R skew']}, Skewness(B) = {report[att]['B skew']}"\
             for att in atts]

    return all(report[att]['result']>0 for att in atts), details

def kurtosis_test(R: dict[str,list[float]], B: dict[str,list[float]], rtol: float = 5e-2): 
    atts = R.keys() if R.keys() == B.keys() else []
    report = pd.DataFrame([[0 for _ in atts] for _ in range(3)], columns = atts, index = ['result','R kur','B kur'])
    for att in atts: 
        # kur
        report[att]['R kur'], report[att]['B kur'] = moment(R[att],4), moment(B[att],4)
        report[att]['result'] = 1 if np.isclose(report[att]['R kur'], report[att]['B kur'], rtol = rtol) else 0

    details = [f"*{att} {'passed' if report[att]['result']>0 else 'failed'} kurtosis test, Kurtosis(R) = {report[att]['R kur']}, Kurtosis(B) = {report[att]['B kur']}"\
             for att in atts]

    return all(report[att]['result']>0 for att in atts), details

def ks_test(R: dict[str,list[float]], B: dict[str,list[float]]): 
    atts = R.keys() if R.keys() == B.keys() else []
    report = pd.DataFrame([[0 for _ in atts]], columns = atts, index = ['p value'])
    for att in atts: 
        # ks test
        _, p = ks_2samp(R[att], B[att])
        report[att]['p value'] = p
    
    details = [f"*{att} {'passed' if report[att]['p value']>0.05 else 'failed'} KS test, P = {report[att]['p value']}"\
             for att in atts]

    return all(report[att]['p value']>0.05 for att in atts), details


def surrogate_analysis(ABM: ap.Model, I: dict[str,list[float]], B: dict[str,list[float]], I_Range: dict[str,list[float]],\
                        iterations: int = 100, level: str = 'agent'): 
    param_pool = {k: [np.random.uniform(v[0], v[1]) for _ in range(iterations)] for k, v in I_Range.items()}

    X_train = []
    y_train = []
    
    while len(y_train) < int(iterations/10): 
        params = {k: v[np.random.randint(len(v))] for k, v in param_pool.items()}
        I.update(params)
        
        abm_res = run_ABM(ABM, I)
        abm_vars = abm_res.variables[list(abm_res.variables)[int(level == 'agent')]]
        R = {k: abm_vars[k].values for k in B.keys()}
        res, _ = ks_test(R, B)

        # Label data points based on p-values (threshold for significance level)
        X_train.append(flatten(list(params.values())))
        y_train.append(1 if res else 0)
    
    # Train the initial surrogate model
    model = catboost.CatBoostClassifier(iterations = 200, learning_rate = 0.1, depth = 6)
    model.fit(X_train, y_train)
    
    # Iteratively refine the model and sample pool
    for _ in range(iterations): 
        param_candidates = {k: [np.random.uniform(v[0], v[1]) for _ in range(int(iterations/10))] for k, v in I_Range.items()}
        X_candidates = []
        for _ in range(10): 
            params = {k: np.random.choice(v) for k, v in param_candidates.items()}
            X_candidates.append(flatten(list(params.values())))
        
        # Predict the likelihood of positive calibration
        y_candidates = model.predict(X_candidates)
        
        # Select new positive samples to evaluate
        for i, y_pred in enumerate(y_candidates): 
            if y_pred == 1: 
                params = {k: X_candidates[i][j] for j, k in enumerate(I.keys())}
                I.update(params)

                abm_res = run_ABM(ABM, I)
                abm_vars = abm_res.variables[list(abm_res.variables)[int(level == 'agent')]]
                R = {k: abm_vars[k].values for k in B.keys()}
                res, _ = ks_test(R, B)
                
                X_train.append(list(params.values()))
                y_train.append(1 if res else 0)
        
        # Retrain the surrogate model with the new data
        model.fit(X_train, y_train)
    
    # The model and X_train can be used for further analysis or predictions
    return model, X_train, y_train

def causal_analysis(R: dict[str,list[float]], B: dict[str,list[float]]): 
    atts = R.keys() if R.keys() == B.keys() else []
    report = pd.DataFrame([[0 for _ in atts]], columns = atts, index = ['result'])
    for att in atts: 
        # Create data box
        data = pd.DataFrame({'R': R[att], 'B': B[att]})

        # Estimate VAR model
        constant_columns = data.columns[data.nunique() ==  1]
        data = data.drop(columns = constant_columns)
        model = VAR(data)
        model.fit(maxlags = 15, ic = 'aic')

        # Using PC algorithm for causal search
        pc = PC(data)
        causal_graph_R = pc.estimate(return_type = "dag", variant = "orig", significance_level = 0.05, max_cond_vars = 4)
        causal_graph_B = pc.estimate(return_type = "dag", variant = "orig", significance_level = 0.05, max_cond_vars = 4)

        report[att]['result'] = int(causal_graph_R.edges == causal_graph_B.edges)

        pass

    details = [f"*{att} {'passed' if report[att]['result']>0 else 'failed'} Causal Analysis" for att in atts]

    return all(report[att]['result']>0 for att in atts), details

def bayesian(ABM: ap.Model, R: dict[str,list[float]], B: dict[str,list[float]], I_Range: dict[str,list[float]], n_walkers = 10, n_steps = 10): 
    atts = R.keys() if R.keys() == B.keys() else []
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

    # Calculate KS inspection results
    res, _ = ks_test(R, B)

    return res, 'details'


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
    
    # Verify similarity
    similarity_results = {}
    for output_var in B.keys(): 
        R_vals = model_outputs_df[output_var].values
        B_vals = B[output_var]
        stat, p_value = ks_2samp(R_vals, B_vals)
        similarity_results[output_var] = {
            'stat': stat,
            'p_value': p_value,
            'similar': p_value > 0.05  # 如果p值大于0.05，则认为两组数据相似
        }
    
    return all(indice for _, indice in sensitivity_indices.items), 'details'
