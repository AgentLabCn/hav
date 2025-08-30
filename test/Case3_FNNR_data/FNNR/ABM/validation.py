import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import pearsonr
import multiprocessing as mp
from model import *
from land import *
from humans import *
from environment import *
import os

def run_model(Input):
    params = {
        "household_area": Input.get("household_area", "400"),
        "farm_area": Input.get("farm_area", "300"),
        "forest_area": Input.get("forest_area", "200"),
        "family_count": Input.get("family_count", 20),
        "scenario": Input.get("scenario", "Flat"),
        "flat_comp": Input.get("flat_comp", 270),
        "dry_comp": Input.get("dry_comp", 200),
        "rice_comp": Input.get("rice_comp", 400),
        "before_comp": Input.get("before_comp", 200),
        "after_comp": Input.get("after_comp", 350),
        "break_year": Input.get("break_year", 4),
        "pes_length": Input.get("pes_length", 8),
        "fertility_rate": Input.get("fertility_rate", "2.5"),
        "reversion_factor": Input.get("reversion_factor", "0.25"),
        "conversion_factor": Input.get("conversion_factor", "0.25"),
        "min_gtgp_area": Input.get("min_gtgp_area", "0.3"),
        "years": Input.get("years", 5)
    }

    def set_filelist(hh, fm, fr):
        file_list.clear()
        hh_file = f'hh_ascii{hh}.txt' if hh in ["100", "400", "800"] else 'hh_ascii400.txt'
        farm_file = f'farm_ascii{fm}.txt' if fm in ["0", "300", "600"] else 'farm_ascii300.txt'
        forest_file = f'forest_ascii{fr}.txt' if fr in ["0", "200", "400"] else 'forest_ascii200.txt'
        file_list.extend([hh_file, farm_file, forest_file])

    def set_core_settings():
        setting_list.clear()
        scenario_list.clear()
        pes_span.clear()
        fertility_scenario.clear()
        min_threshold_list.clear()
        no_pay_part_list.clear()
        min_non_gtgp_list.clear()

        setting_list.extend([params["family_count"], 'With Humans', 'Normal Run'])
        pes_span.append(params["pes_length"])
        fertility_scenario.append(params["fertility_rate"])
        min_threshold_list.append(params["reversion_factor"])
        no_pay_part_list.append(params["conversion_factor"])
        min_non_gtgp_list.append(params["min_gtgp_area"])

        scenario_list.append(params["scenario"])
        if params["scenario"] == "Flat":
            scenario_list.append(params["flat_comp"])
        elif params["scenario"] == "Land Type":
            scenario_list.extend([params["dry_comp"], params["rice_comp"]])
        elif params["scenario"] == "Time":
            scenario_list.extend([params["before_comp"], params["after_comp"], params["break_year"]])

    set_filelist(params["household_area"], params["farm_area"], params["forest_area"])
    set_core_settings()

    model = Movement()
    total_steps = 73 * params["years"]  

    monkey_num_lst = []
    human_num_lst = []
    forest_area_lst = []
    forest_types = (Broadleaf, Mixed, Deciduous)
    for _ in range(total_steps):
        model.step()
        monkey_num_lst.append(model.number_of_monkeys)
        human_num_lst.append(model.number_of_humans)

        monkey_positions = [agent.current_position for agent in model.schedule.agents 
                            if isinstance(agent, Family)]
        if not monkey_positions:
            preferred_veg_ratio = 0.0
        else:
            activity_area = set()
            for (x, y) in monkey_positions:
                for dx in range(-2, 3):
                    for dy in range(-2, 3):
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < model.grid.width and 0 <= ny < model.grid.height:
                            activity_area.add((nx, ny))
            
            total_in_activity = len(activity_area)
            preferred_in_activity = 0
            for (x, y) in activity_area:
                agents = model.grid.get_cell_list_contents((x, y))
                for agent in agents:
                    if isinstance(agent, forest_types):
                        preferred_in_activity += 1
                        break
            
            preferred_veg_ratio = preferred_in_activity / total_in_activity if total_in_activity > 0 else 0.0
        
        forest_area_lst.append(preferred_veg_ratio)

        pass

    def gtgp_participation(model):
        gtgp_hh_ids = set()
        for agent in model.schedule.agents:
            if isinstance(agent, Land) and agent.gtgp_enrolled:
                gtgp_hh_ids.add(agent.hh_id)
        return len(gtgp_hh_ids)

    return {
        "monkey_num_lst": monkey_num_lst,
        "human_num_lst": human_num_lst,
        "forest_area_lst": forest_area_lst,
        "final_monkey_num": monkey_num_lst[-1:][0],
        "final_human_num": human_num_lst[-1:][0],
        "gtgp_participation": gtgp_participation(model) / human_num_lst[-1:][0],
    }

def Regression(df, y_col, x_cols, hypothesis, expected_signs, core_x_cols=None):
    if core_x_cols is None:
        core_x_cols = x_cols
    
    X = df[x_cols]
    y = df[y_col]
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    
    print(f"\n=== Logic: {hypothesis} ===")
    print(model.summary().tables[1])
    
    for x in core_x_cols:
        coef = model.params[x]
        p_value = model.pvalues[x]
        expected = expected_signs[x]
        actual_sign = 1 if coef > 0 else (-1 if coef < 0 else 0)
        
        sign_ok = (actual_sign == expected)
        significant = (p_value < 0.05)
        
        print(f"Core variable {x}: Coefficient = {coef:.4f}, P-value = {p_value:.4f} → "
              f"Sign {'matches' if sign_ok else 'does not match'} expectation ({expected}), "
              f"{'significant' if significant else 'not significant'}")
    
    control_vars = [x for x in x_cols if x not in core_x_cols]
    if control_vars:
        print("\nControl variable information:")
        for x in control_vars:
            coef = model.params[x]
            p_value = model.pvalues[x]
            print(f"Control variable {x}: Coefficient = {coef:.4f}, P-value = {p_value:.4f}")


def vali_A(run_exp=False):
    if run_exp:
        flat_comp_samples = np.random.randint(0, 1000, size=30)
        tasks = [
            {
                "params": {"scenario": "Flat", "flat_comp": comp},
                "seed": 1 + i
            } 
            for i, comp in enumerate(flat_comp_samples)
        ]

        with mp.Pool(processes=mp.cpu_count()) as pool:
            results = pool.map(
                task_process,
                tasks
            )
        df = pd.DataFrame({
            "flat_comp": flat_comp_samples,
            "gtgp_participation": [res["gtgp_participation"] for res in results]
        })
        df.to_csv(os.getcwd() + "/samples_A.csv")
    else: df = pd.read_csv(os.getcwd() + "/samples_A.csv")

    Regression(
        df=df,
        y_col="gtgp_participation",
        x_cols=["flat_comp"],
        hypothesis="flat_comp has a positive correlation with gtgp_participation (expected sign +1)",
        expected_signs={"flat_comp": 1}
    )

def vali_M(run_exp=False):
    if run_exp:
        fertility_rate_samples = np.random.uniform(1, 10, size=30)
        tasks = [{"params": {"fertility_rate": fertility_rate}, "seed": 1 + i}
                 for i, fertility_rate in enumerate(fertility_rate_samples)]
        with mp.Pool(processes=mp.cpu_count()) as pool:
            results = pool.map(
                task_process,
                tasks
            )
        df = pd.DataFrame({
            "fertility_rate": [fertility_rate for fertility_rate in fertility_rate_samples],
            "final_monkey_num": [res["final_monkey_num"] for res in results]
        })
        df.to_csv(os.getcwd() + "/samples_M.csv")
    else: df = pd.read_csv(os.getcwd() + "/samples_M.csv")
        
    Regression(
        df=df,
        y_col="final_monkey_num",
        x_cols=["fertility_rate"],
        hypothesis="fertility_rate has a negative correlation with final_monkey_num (expected sign -1)",
        expected_signs={"fertility_rate": -1}
    )

def vali_O(run_exp=False):
    if run_exp:
        tasks = [{"params": {}, "seed": 1 + i} for i in range(10)]
        with mp.Pool(processes=mp.cpu_count()) as pool:
            results = pool.map(
                task_process,
                tasks
            )
        forest_area_lst = [res["forest_area_lst"] for res in results]
        monkey_num_lst = [res["monkey_num_lst"] for res in results]

        df = pd.DataFrame({
            "avg_forest": np.mean(forest_area_lst, axis=0),
            "avg_monkey": np.mean(monkey_num_lst, axis=0)
        })
        df.to_csv(os.getcwd() + "/samples_O.csv")
    else: df = pd.read_csv(os.getcwd() + "/samples_O.csv")
    
    corr_coef, p_value = pearsonr(df["avg_forest"], df["avg_monkey"])
    
    print("\n=== Pearson correlation analysis between vegetation area and monkey population ===")
    print(f"Average sequence length: {len(df['avg_forest'])} (time steps)")
    print(f"Correlation coefficient: {corr_coef:.4f}")
    print(f"P-value: {p_value:.4f}")
    print(f"Correlation conclusion: {'significant' if p_value < 0.05 else 'not significant'}")
    print(f"Correlation direction: {'positive correlation' if corr_coef > 0 else 'negative correlation' if corr_coef < 0 else 'no correlation'}")

def task_process(task):
    return run_model_wrapper(task["params"], task["seed"])

def run_model_wrapper(params, seed):
    np.random.seed(seed)
    return run_model(params)

if __name__ == "__main__":
    print("=== Executing vali_A ===")
    vali_A()
    
    print("\n=== Executing vali_M ===")
    vali_M()
    
    print("\n=== Executing vali_O ===")
    vali_O()

