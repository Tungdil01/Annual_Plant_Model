# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# ### The code is a modification of the Yenni et al. (2012) analysis:
# #### - has the option to keep the filter S1 >= 1 & S2 >= 1 or remove it
# #### - does not truncate the values
# #### - considers extinction N<1e-6 rather than N<1
# #### - includes Cushing et al. (2004) analytical results
#
# #### their original code: https://github.com/gmyenni/RareStabilizationSimulation

import os
import gc
import time
import warnings
import shap
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.proportion import proportion_confint
from scipy import stats
from tqdm import tqdm
from numba import jit
from itertools import combinations
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance


# # analyN_function.r

def difference_equation(r1, r2, a11, a12, a21, a22, solver='dynamics'):
    if solver == 'analyN': # Analytical approximation for equilibrium populations
        denominator1 = a11 - (a21 * a12) / a22
        denominator2 = a22 - (a21 * a12) / a11
        N1 = (r1 - 1 - (a12 / a22) * (r2 - 1)) / denominator1 if denominator1 != 0 else np.nan
        N2 = (r2 - 1 - (a21 / a11) * (r1 - 1)) / denominator2 if denominator2 != 0 else np.nan
        if np.isinf(N1) or np.isinf(N2) or np.isnan(N1) or np.isnan(N2):
            initialNsp1 = 0
            initialNsp2 = 0
            N = np.zeros((100, 2))
            N[0, :] = [initialNsp1, initialNsp2]   
            for i in range(1, 100):
                N[i, 0] = max((r1 - 1 - a12 * N[i-1, 1]) / a11, 0)
                N[i, 1] = max((r2 - 1 - a21 * N[i-1, 0]) / a22, 0)
            N1 = np.mean(N[:, 0])
            N2 = np.mean(N[:, 1])
        if N1 < 0 and N2 >= 0:
            N1, N2 = 0.0, (r2 - 1) / a22 if a22 != 0 else 0.0
        elif N2 < 0 and N1 >= 0:
            N1, N2 = (r1 - 1) / a11 if a11 != 0 else 0.0, 0.0
        elif N1 < 0 and N2 < 0:
            N1, N2 = 0.0, 0.0
        return N1, N2
    elif solver == 'dynamics': # Numerical simulation
        y1 = np.array([5.0], dtype=np.float64)
        y2 = np.array([5.0], dtype=np.float64)
        stop_run = False
        i = 0
        while not stop_run and i < 10000:
            denom1 = 1 + a11 * y1[i] + a12 * y2[i]
            denom2 = 1 + a22 * y2[i] + a21 * y1[i]
            per_cap1 = r1 / denom1
            per_cap2 = r2 / denom2
            new_y1 = y1[i] * per_cap1
            new_y2 = y2[i] * per_cap2
            y1 = np.append(y1, new_y1)
            y2 = np.append(y2, new_y2)
            if i >= 1:
                if (abs(y1[-1] - y1[-2]) < 1e-6 and abs(y2[-1] - y2[-2]) < 1e-6):
                    stop_run = True
            i += 1
        return y1[-1], y2[-1]


def compare_counts_test(filtered_data, print_on=False):
    print('\nPGR Statistics:\n')
    count_PGR1, count_PGR2 = [], []
    for _, row in filtered_data.iterrows():
        PGR1, PGR2 = getPCG(row['r1'], row['r2'], row['a11'], row['a12'], row['a21'], row['a22'], row['N1'], row['N2'])
        count_PGR1.append(PGR1)
        count_PGR2.append(PGR2)
    count_PGR1 = pd.Series(count_PGR1).dropna()
    count_PGR2 = pd.Series(count_PGR2).dropna()
    if count_PGR1.empty or count_PGR2.empty:
        print("No valid PGR data.")
        return None
    # Calculate statistics
    stats = {
        "mean_PGR1": count_PGR1.mean(),
        "mean_PGR2": count_PGR2.mean(),
        "std_PGR1": count_PGR1.std(ddof=1),
        "std_PGR2": count_PGR2.std(ddof=1),
        "median_PGR1": count_PGR1.median(),
        "median_PGR2": count_PGR2.median()
    }
    print(f"PGR1 Mean \u00B1 SD: {stats['mean_PGR1']:.2g} \u00B1 {stats['std_PGR1']:.2g}")
    print(f"PGR2 Mean \u00B1 SD: {stats['mean_PGR2']:.2g} \u00B1 {stats['std_PGR2']:.2g}")
    return stats


# # getNFD.r

# +
@jit
def SOS(r1, r2, a11, a12, a21, a22):
    S1 = r2 / (1 + (a12 / a22) * (r2 - 1))
    S2 = r1 / (1 + (a21 / a11) * (r1 - 1))
    return S1, S2

@jit
def getPCG(r1, r2, a11, a12, a21, a22, N1, N2): # Per capita growth rate calculation
    newN1 = r1 * N1 / (1 + a11 * N1 + a12 * N2) if N1 > 0 else np.nan
    newN2 = r2 * N2 / (1 + a22 * N2 + a21 * N1) if N2 > 0 else np.nan
    PGR1 = np.log(newN1) - np.log(N1) if N1 > 0 else np.nan
    PGR2 = np.log(newN2) - np.log(N2) if N2 > 0 else np.nan
    return PGR1, PGR2

@jit
def calculate_metrics(r1, r2, a11, a12, a21, a22, N1, N2, extinc_crit_1=True):
    S1, S2 = SOS(r1, r2, a11, a12, a21, a22) # Strength of Stabilization
    FE1, FE2 = r1 / r2, r2 / r1 # Fitness equivalence
    Asy = S1 - S2 # Asymmetry
    Rare = 0 if N1 == 0 and N2 == 0 else N1 / (N1 + N2)
    # Calculating covariance for SoS
    x = np.array([N1, N2])
    y_sos = np.array([S1, S2])
    cor_matrix_sos = np.cov(x, y_sos)
    cor_sos = cor_matrix_sos[0, 1] # Extracting the correlation between N and SoS
    Rank = 0 if N1 == 0 and N2 == 0 else (2 if N1 / (N1 + N2) <= 0.25 else 1)
    # Equilibrium points
    E1 = (r1 - 1) / a11
    E2 = (r2 - 1) / a22
    P = (r1 - 1) / a12
    Q = (r2 - 1) / a21
    # Calculate conditions for A, B, C, D
    A = P > E2 and E1 > Q
    B = E2 > P and Q > E1
    C = P > E2 and Q > E1
    D = E2 > P and E1 > Q
    # Call getPCG to calculate PGR1 and PGR2
    PGR1, PGR2 = getPCG(r1, r2, a11, a12, a21, a22, N1, N2)
    if extinc_crit_1:
        Coexist = 0 if N1 < 1 or N2 < 1 else 1
    else:
        Coexist = 0 if N1 < 1.0e-6 or N2 < 1.0e-6 else 1
    return {"FE1": FE1, "S1": S1, "FE2": FE2, "S2": S2, "Rank": Rank, "Coexist": Coexist, "Asy": Asy, "cor_sos": cor_sos, "Rare": Rare, "PGR1": PGR1, "PGR2": PGR2, "A": A, "B": B, "C": C, "D": D}


# -

# # annualplant_2spp_det_par.r

# +
def preprocess_data(pars):
    # Defines frequency-dependent parameters
    if pars == 'r_code': # Their R code
         r1_v = np.arange(10, 21, 1)
         r2_v = np.arange(10, 21, 1)
         a11_v = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1, 1.5, 2, 2.5, 3])
         a12_v = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1])
         a21_v = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1])
         a22_v = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1])
    elif pars == 'table1': # Reproduce their Table 1
        r1_v = np.arange(15, 21, 1)
        r2_v = np.arange(15, 21, 1)
        a11_v = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1, 1.5, 2, 2.5, 3])
        a12_v = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1])
        a21_v = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1])
        a22_v = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1])
    elif pars == 'paper': # They describe in the paper
         r1_v = np.arange(15, 21, 1)
         r2_v = np.arange(11, 21, 1)
         a11_v = np.array([0.7, 0.3, 0.5, 0.7, 0.9, 1, 1.5, 2, 2.5, 3])
         a12_v = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1])
         a21_v = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1])
         a22_v = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1])
    else: # minimal: Reduced set of parameters
        r1_v = np.array([15, 17, 18, 20])
        r2_v = np.array([15, 17, 18, 20])
        a11_v = np.array([0.1, 1, 3])
        a12_v = np.array([0.1, 0.5, 1])
        a21_v = np.array([0.1, 0.5, 1])
        a22_v = np.array([0.1, 0.5, 1])
    # Generate all combinations of parameters using NumPy's meshgrid
    mesh = np.array(np.meshgrid(r1_v, r2_v, a11_v, a12_v, a21_v, a22_v)).T.reshape(-1, 6)
    return mesh

def Sim(k, mesh_row, extinc_crit_1=False, solver='dynamics'):
    start_time = time.time()
    r1, r2, a11, a12, a21, a22 = mesh_row
    N1, N2 = difference_equation(r1, r2, a11, a12, a21, a22, solver=solver)
    metrics = calculate_metrics(r1, r2, a11, a12, a21, a22, N1, N2, extinc_crit_1)
    execution_time = time.time() - start_time
    return {**metrics, "N1": N1, "N2": N2, "r1": r1, "r2": r2, "a11": a11, "a12": a12, "a21": a21, "a22": a22}

def postprocess_results(results, outfile):
    column_order = ['r1', 'r2', 'a11', 'a12', 'a21', 'a22', 'N1', 'N2', 'FE1', 'S1', 'FE2', 'S2', 'Rank', 'Coexist', 'Asy', 'cor_sos', 'Rare', 'PGR1', 'PGR2', 'A', 'B', 'C', 'D']
    simul = pd.DataFrame(results, columns=column_order)
    simul.to_csv(outfile, index=False)


# -

# # cor_figure.r

def cor_figure(filter, truncate=False):
    dat_det = pd.read_csv("csv/annplant_2spp_det_rare.csv")
    if filter == 'inverted':
        dat_det = dat_det.query('Rank == 2 & S1 < 1 & S2 < 1').copy()
    elif filter == 'on':
        dat_det = dat_det.query('Rank == 2 & S1 >= 1 & S2 >= 1').copy()
    else: # 'off'
        dat_det = dat_det.query('Rank == 2').copy()
    dat_det.reset_index(drop=True, inplace=True)
    if truncate:
        dat_det = np.trunc(dat_det * 100) / 100.0
    dat_det.sort_values(by=['a22', 'a21', 'a12', 'a11', 'r2', 'r1'], inplace=True)
    dat_det.to_csv(f"csv/annplant_2spp_det_rare_filtered_{filter}.csv", index=False)


# # figures_det.r

# +
def perform_logistic_regression(dat, analysis_type, print_on=False):
    predictors_map = {
        'SoS': ['S1', 'FE1', 'cor_sos'],
    }    
    predictors = predictors_map[analysis_type]
    X = sm.add_constant(dat[predictors])
    y = dat['Coexist']
    model = sm.GLM(y, X, family=sm.families.Binomial())
    result = model.fit()
    print(result.summary())
    coef = result.params
    std_err = result.bse
    z_scores = result.tvalues
    p_values = result.pvalues
    intercept = coef[0]
    coef = coef[1:]
    if print_on: # same analysis in more detail
        print("\n\n--------------------------------------------------------\n\n")
        result_table = result.summary2().tables[1]
        # Apply maximum precision to coefficient-related statistics
        result_table['Coef.'] = result_table['Coef.']
        result_table['Std.Err.'] = result_table['Std.Err.']
        result_table['z'] = result_table['z'].apply(lambda x: np.format_float_scientific(x, precision=4))
        result_table['P>|z|'] = result_table['P>|z|'].apply(lambda x: np.format_float_scientific(x, precision=4))
        result_table = result_table.round(4)
        print(f"\n{analysis_type} Analysis (in more detail):")
        print(result_table)
    return intercept, coef, std_err, z_scores, p_values

def calculate_proportions(dat, correlation_type):
    proportions = {}
    for cor_type in [correlation_type]:
        proportions[f'positive_coexistence_{cor_type}'] = len(dat[(dat[cor_type] >= 0) & (dat['Coexist'] == 1)])
        proportions[f'positive_exclusion_{cor_type}'] = len(dat[(dat[cor_type] >= 0) & (dat['Coexist'] == 0)])
        proportions[f'negative_coexistence_{cor_type}'] = len(dat[(dat[cor_type] < 0) & (dat['Coexist'] == 1)])
        proportions[f'negative_exclusion_{cor_type}'] = len(dat[(dat[cor_type] < 0) & (dat['Coexist'] == 0)])
    return proportions

def report_coexistence_analysis(proportions, correlation_type):
    positive_key = f'positive_coexistence_{correlation_type}'
    negative_key = f'negative_coexistence_{correlation_type}'
    neg_confint = proportion_confint(count=proportions[negative_key], nobs=proportions[negative_key] + proportions[f'negative_exclusion_{correlation_type}'], alpha=0.05, method='wilson')
    pos_confint = proportion_confint(count=proportions[positive_key], nobs=proportions[positive_key] + proportions[f'positive_exclusion_{correlation_type}'], alpha=0.05, method='wilson')
    print(f"\nAnalysis on Negative \u03BD for {correlation_type.upper()}:")
    print(f"Proportion of coexistence with \u03BD \u2265 0: {proportions[positive_key] / (proportions[positive_key] + proportions[f'positive_exclusion_{correlation_type}']):.2g} (95% CI: {pos_confint})")
    print(f"Proportion of coexistence with \u03BD < 0: {proportions[negative_key] / (proportions[negative_key] + proportions[f'negative_exclusion_{correlation_type}']):.2g} (95% CI: {neg_confint})")
    
def analyze_coexistence_effect(data, print_on):
    original_dat = data.copy()
    models_results = {}
    for correlation_type in ['SoS']:
        analysis_type = f'{correlation_type}'
        correlation_column = 'cor_sos'
        if correlation_column not in data.columns:
            continue
        print(f"\n--- Analysis for {analysis_type} ---")
        intercept, coef, std_err, z_scores, p_values = perform_logistic_regression(data, analysis_type, print_on=print_on)
        models_results[analysis_type] = {
            'statsmodels': (intercept, coef, std_err, z_scores, p_values),
        }
        proportions = calculate_proportions(data, correlation_column)
        report_coexistence_analysis(proportions, correlation_column)
        table_data = {
            '\u03BD \u2265 0': [proportions[f'positive_coexistence_{correlation_column}'], proportions[f'positive_exclusion_{correlation_column}']],
            '\u03BD < 0': [proportions[f'negative_coexistence_{correlation_column}'], proportions[f'negative_exclusion_{correlation_column}']]
        }
        table_df = pd.DataFrame(table_data, index=['Coexistence', 'Exclusion'])
        print(f"Coexistence and Exclusion based on \u03BD for {analysis_type}:\n", table_df)
        # Proportion of coexistence calculations and confidence intervals
        pos_confint = proportion_confint(count=proportions[f'positive_coexistence_{correlation_column}'], nobs=proportions[f'positive_coexistence_{correlation_column}'] + proportions[f'positive_exclusion_{correlation_column}'], alpha=0.05, method='wilson')
        neg_confint = proportion_confint(count=proportions[f'negative_coexistence_{correlation_column}'], nobs=proportions[f'negative_coexistence_{correlation_column}'] + proportions[f'negative_exclusion_{correlation_column}'], alpha=0.05, method='wilson')
        # Decision making based on confidence intervals
        if neg_confint[1] >= pos_confint[0] and neg_confint[0] <= pos_confint[1]:  # Overlap
            print(f"The confidence intervals overlap for {analysis_type}, indicating they are statistically the same, not supporting the authors' results.")
        elif neg_confint[1] > pos_confint[0]:  # Negative larger than positive
            print(f"Higher coexistence observed with \u03BD < 0 for {analysis_type}, supporting the authors' results.")
        else:  # Negative smaller than positive
            print(f"Higher coexistence observed with \u03BD \u2265 0 for {analysis_type}, not supporting the authors' results.")
    return models_results


# -

def count_abcd(filtered_data):
    # Initialize counters
    count_A_0, count_A_1 = 0, 0
    count_B_0, count_B_1 = 0, 0
    count_C_0, count_C_1 = 0, 0
    count_D_0, count_D_1 = 0, 0
    # Loop over the filtered dataset and apply the conditions for A, B, C, D
    for index, row in filtered_data.iterrows():
        P = (row['r1'] - 1) / row['a12']
        E2 = (row['r2'] - 1) / row['a22']
        Q = (row['r2'] - 1) / row['a21']
        E1 = (row['r1'] - 1) / row['a11']
        if P != E2 and Q != E1:
            if P > E2 and E1 > Q:  # Case A
                if row['Coexist'] == 0:
                    count_A_0 += 1
                else:
                    count_A_1 += 1
            elif E2 > P and Q > E1:  # Case B
                if row['Coexist'] == 0:
                    count_B_0 += 1
                else:
                    count_B_1 += 1
            elif P > E2 and Q > E1:  # Case C
                if row['Coexist'] == 0:
                    count_C_0 += 1
                else:
                    count_C_1 += 1
            elif E2 > P and E1 > Q:  # Case D
                if row['Coexist'] == 0:
                    count_D_0 += 1
                else:
                    count_D_1 += 1
        elif P == E2 and Q == E1:
            if row['Coexist'] == 0:
                count_C_0 += 1
            else:
                count_C_1 += 1
        elif P == E2:  # If P equals E2, only look at the relationship between Q and E1
            if Q > E1:
                if row['Coexist'] == 0:
                    count_B_0 += 1
                else:
                    count_B_1 += 1
            elif E1 > Q:
                if row['Coexist'] == 0:
                    count_A_0 += 1
                else:
                    count_A_1 += 1
        elif Q == E1:  # If Q equals E1, only look at the relationship between P and E2
            if P > E2:
                if row['Coexist'] == 0:
                    count_A_0 += 1
                else:
                    count_A_1 += 1
            elif E2 > P:
                if row['Coexist'] == 0:
                    count_B_0 += 1
                else:
                    count_B_1 += 1
    # Calculate totals for A, B, C, D
    count_A_total = count_A_0 + count_A_1
    count_B_total = count_B_0 + count_B_1
    count_C_total = count_C_0 + count_C_1
    count_D_total = count_D_0 + count_D_1
    total_count = len(filtered_data)
    # Calculate proportions
    prop_A = count_A_total / total_count if total_count != 0 else 0
    prop_B = count_B_total / total_count if total_count != 0 else 0
    prop_C = count_C_total / total_count if total_count != 0 else 0
    prop_D = count_D_total / total_count if total_count != 0 else 0
    # Print results
    print(f"A\nCoexist==0: {count_A_0}\nCoexist==1: {count_A_1}\nTotal: {count_A_total}\nProportion: {prop_A:.2g}")
    print(f"\nB\nCoexist==0: {count_B_0}\nCoexist==1: {count_B_1}\nTotal: {count_B_total}\nProportion: {prop_B:.2g}")
    print(f"\nC\nCoexist==0: {count_C_0}\nCoexist==1: {count_C_1}\nTotal: {count_C_total}\nProportion: {prop_C:.2g}")
    print(f"\nD\nCoexist==0: {count_D_0}\nCoexist==1: {count_D_1}\nTotal: {count_D_total}\nProportion: {prop_D:.2g}")


def plot_phase_plane():
    # Parameters for each scenario
    scenarios = {
        "A: $E_1 > Q$ and $P > E_2$": {'r1': 18, 'r2': 16, 'a11': 0.5, 'a12': 1, 'a21': 1, 'a22': 1},
        "B: $Q > E_1$ and $E_2 > P$": {'r1': 20, 'r2': 15, 'a11': 2, 'a12': 1, 'a21': 1, 'a22': 0.5},
        "C: $Q > E_1$ and $P > E_2$":  {'r1': 20, 'r2': 15, 'a11': 3, 'a12': 0.5, 'a21': 1, 'a22': 0.5},
        "D: $E_1 > Q$ and $E_2 > P$":  {'r1': 16, 'r2': 18, 'a11': 0.3, 'a12': 1, 'a21': 1, 'a22': 0.5},
    }    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()    
    for i, (title, params) in enumerate(scenarios.items()):
        ax = axes[i]
        # Unpack parameters
        r1 = params['r1']
        r2 = params['r2']
        a11 = params['a11']
        a12 = params['a12']
        a21 = params['a21']
        a22 = params['a22']
        # Equilibrium points
        E1 = [(r1 - 1) / a11, 0]
        Q = [(r2 - 1) / a21, 0]
        E2 = [0, (r2 - 1) / a22]
        P = [0, (r1 - 1) / a12]
        E0 = [0, 0]
        # Calculate the intersection point of lines (E1, Q) and (E2, P)
        a1 = (P[1] - E1[1]) / (P[0] - E1[0]) if P[0] != E1[0] else float('inf')
        b1 = E1[1] - a1 * E1[0]
        a2 = (E2[1] - Q[1]) / (E2[0] - Q[0]) if E2[0] != Q[0] else float('inf')
        b2 = Q[1] - a2 * Q[0]
        if a1 != a2:  # Ensure lines are not parallel
            E3_x = (b2 - b1) / (a1 - a2) if a1 != float('inf') and a2 != float('inf') else 0
            E3_y = a1 * E3_x + b1 if a1 != float('inf') else a2 * E3_x + b2
            E3 = [E3_x, E3_y]
        else:
            E3 = None
        # Extend axis limits by 10%
        max_N1 = max(E1[0], Q[0])
        max_N2 = max(E2[1], P[1])
        N1 = np.linspace(0, max_N1, 30)
        N2 = np.linspace(0, max_N2, 30)
        N1, N2 = np.meshgrid(N1, N2)
        # Compute the discrete system
        N1_next = r1 * N1 / (1 + a11 * N1 + a12 * N2)
        N2_next = r2 * N2 / (1 + a22 * N2 + a21 * N1)
        # Plot vector field
        ax.quiver(N1, N2, N1_next - N1, N2_next - N2, angles='xy', scale_units='xy', scale=15, color='grey', alpha=1)
        # Plot equilibrium points
        ax.plot(E0[0], E0[1], 'ko', label='E0', markersize=8)
        ax.plot(E1[0], E1[1], 'bo', label='E1', markersize=8)
        ax.plot(Q[0], Q[1], 'ro', label='Q', markersize=8)
        ax.plot(E2[0], E2[1], 'ro', label='E2', markersize=8)
        ax.plot(P[0], P[1], 'bo', label='P', markersize=8)
        # Draw lines between points
        ax.plot([E1[0], P[0]], [E1[1], P[1]], 'b-', lw=2)  # Line between P and E1 (blue)
        ax.plot([Q[0], E2[0]], [Q[1], E2[1]], 'r-', lw=2)  # Line between Q and E2 (red)
        # Plot intersection point E3 if it exists within the plot limits and above the lines
        if E3 is not None and (0 <= E3[0] <= 1.1 * max_N1) and (0 <= E3[1] <= 1.1 * max_N2):
            ax.plot(E3[0], E3[1], 'go', label=r'$E_3$', markersize=8)
            # Annotate E3 near the point
            ax.annotate(f'$E_3$', xy=(E3[0], E3[1]), xytext=(E3[0] + 0.3, E3[1] + 0.3), fontsize=18, color='green')
        # Set labels and title
        ax.set_xlabel(r'$N_1$', fontsize=18)
        ax.set_ylabel(r'$N_2$', fontsize=18)
        # Move title to the left
        ax.set_title(title, fontsize=18, loc='left')
        # Set xticks and yticks with labels for E1, E2, P, Q
        ax.set_xticks([0, E1[0], Q[0]])
        ax.set_xticklabels([r'$E_0$', r'$E_1$', r'$Q$'])
        ax.set_yticks([0, E2[1], P[1]])
        ax.set_yticklabels([r'$E_0$', r'$E_2$', r'$P$'])
        ax.tick_params(axis='both', which='major', labelsize=18)
    # Adjust layout and save the figure
    plt.tight_layout()
    os.makedirs('img', exist_ok=True)
    plt.savefig('img/phase_plane.png')
    plt.show()


def getEqDensity(focal_species, N_focal, r1, r2, a11, a12, a21, a22):
    if focal_species == 0:
        return max(0,(r2-1 - a21*N_focal)/a22) if a22!=0 else 0
    else:
        return max(0,(r1-1 - a12*N_focal)/a11) if a11!=0 else 0


def getNFD(r1, r2, a11, a12, a21, a22, lowN, deltaN):
    # Species 1 at low density
    N1_low = lowN
    N2_low = getEqDensity(0, N1_low, r1, r2, a11, a12, a21, a22)
    total_low = N1_low + N2_low
    pgr1_low, pgr2_low = getPCG(r1, r2, a11, a12, a21, a22, N1_low, N2_low)
    freq1_low = N1_low / total_low if total_low > 0 else 0
    # Species 2 at low density
    N2_low2 = lowN
    N1_low2 = getEqDensity(1, N2_low2, r1, r2, a11, a12, a21, a22)
    total_low2 = N1_low2 + N2_low2
    pgr1_low2, pgr2_low2 = getPCG(r1, r2, a11, a12, a21, a22, N1_low2, N2_low2)
    freq2_low = N2_low2 / total_low2 if total_low2 > 0 else 0
    # Species 1 at high density
    N1_high = lowN + deltaN
    N2_high = getEqDensity(0, N1_high, r1, r2, a11, a12, a21, a22)
    total_high = N1_high + N2_high
    pgr1_high, pgr2_high = getPCG(r1, r2, a11, a12, a21, a22, N1_high, N2_high)
    freq1_high = N1_high / total_high if total_high > 0 else 0
    # Species 2 at high density
    N2_high2 = lowN + deltaN
    N1_high2 = getEqDensity(1, N2_high2, r1, r2, a11, a12, a21, a22)
    total_high2 = N1_high2 + N2_high2
    pgr1_high2, pgr2_high2 = getPCG(r1, r2, a11, a12, a21, a22, N1_high2, N2_high2)
    freq2_high = N2_high2 / total_high2 if total_high2 > 0 else 0
    return {
        'pgr1': [pgr1_low, pgr1_high],
        'freq1': [freq1_low, freq1_high],
        'pgr2': [pgr2_low2, pgr2_high2],
        'freq2': [freq2_low, freq2_high]
    }


def plot_pgr_figures(filter_option, save_fig=False, extinc_crit_1=False):
    # Configure global plotting parameters
    plt.rcParams.update({
        'axes.titlesize': 14,
        'axes.labelsize': 18,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 12,
        'font.size': 18,
        'lines.linewidth': 1.5
    })
    summary_path = f"csv/pgr_analysis_summary_{filter_option}.csv"
    cols = [
        'r1', 'r2', 'a11', 'a12', 'a21', 'a22',
        'cor_sos', 'nu_sign', 'coexist',
        'left_PGR1_dominant', 'right_PGR1_dominant', 'curve_cross'
    ]
    with open(summary_path, 'w') as f:
        f.write(','.join(cols) + '\n')
    lowN = 0.001
    deltaN = 100.0
    df = pd.read_csv(f"csv/annplant_2spp_det_rare_filtered_{filter_option}.csv")
    df['cor_sos'] = pd.to_numeric(df['cor_sos'], errors='coerce')
    df['Coexist'] = pd.to_numeric(df['Coexist'], errors='coerce').fillna(0).astype(int)
    conditions = [
        df['cor_sos'] < -0.001,
        (df['cor_sos'] >= -0.001) & (df['cor_sos'] <= 0.001),
        df['cor_sos'] > 0.001
    ]
    choices = ['negative', 'zero', 'positive']
    df['nu_sign'] = np.select(conditions, choices, default='invalid')
    for _, row in df.iterrows():
        if row['nu_sign'] == 'invalid':
            continue
        r1, r2 = row['r1'], row['r2']
        a11, a12, a21, a22 = row['a11'], row['a12'], row['a21'], row['a22']
        # Compute low- and high-density PGR and frequencies via getNFD
        nfd = getNFD(r1, r2, a11, a12, a21, a22, lowN, deltaN)
        freq1, pgr1 = nfd['freq1'], nfd['pgr1']
        freq2, pgr2 = nfd['freq2'], nfd['pgr2']
        # Determine dominance at edges
        left_PGR1_dominant = 1 if pgr1[0] > pgr2[0] else 0 if pgr1[0] < pgr2[0] else None
        right_PGR1_dominant = 1 if pgr1[1] > pgr2[1] else 0 if pgr1[1] < pgr2[1] else None
        curve_cross = 1 if left_PGR1_dominant != right_PGR1_dominant else 0
        # Append to summary CSV
        line = (
            f"{r1},{r2},{a11},{a12},{a21},{a22},"
            f"{row['cor_sos']},{row['nu_sign']},{row['Coexist']},"
            f"{left_PGR1_dominant},{right_PGR1_dominant},{curve_cross}\n"
        )
        with open(summary_path, 'a') as f:
            f.write(line)
        if save_fig:
            fig, ax = plt.subplots(figsize=(8, 6))
            # Plot species 1 curve: frequency vs log(PGR)
            ax.plot(freq1, pgr1, '-', label='N1')
            # Plot species 2 curve: frequency vs log(PGR)
            ax.plot(freq2, pgr2, '--', label='N2')
            ax.axhline(0, color='black', linestyle=':', linewidth=0.8)
            ax.set_xlabel('Frequency', fontsize=16)
            ax.set_ylabel('log(PGR)', fontsize=16)
            ax.set_xlim(-0.005, 1.01)
            ax.set_ylim(-1.01, 1.01)
            ax.set_title(
                f"\u03BD={row['cor_sos']:.2g}, Coexist={row['Coexist']}\n"
                f"r1={r1:.2g}, a11={a11:.2g}, a12={a12:.2g}\n"
                f"r2={r2:.2g}, a21={a21:.2g}, a22={a22:.2g}",
                wrap=True, fontsize=14
            )
            ax.legend()
            top = (
                "PGR_same" if left_PGR1_dominant is None
                else "PGR1_dominant" if left_PGR1_dominant
                else "PGR1_nondominant"
            )
            nu_folder = {
                'positive': "nu_larger_zero",
                'negative': "nu_smaller_zero",
                'zero':     "nu_zero"
            }.get(row['nu_sign'], "nu_invalid")
            if row['nu_sign'] in ['positive', 'negative', 'zero']:
                outcome = "coexist" if row['Coexist'] else "exclusion"
                save_dir = os.path.join("png", top, nu_folder, outcome)
            else:
                save_dir = os.path.join("png", top, nu_folder)
            os.makedirs(save_dir, exist_ok=True)
            fname = (f"r1_{r1}_r2_{r2}_a11_{a11}_a12_{a12}_a21_{a21}_a22_{a22}.png")
            fig.savefig(os.path.join(save_dir, fname), dpi=150, bbox_inches='tight')
            plt.close(fig)
            gc.collect()


def analyze_hypotheses_rf(filter_option, summary_path, seed=1234):
    df = pd.read_csv(summary_path)
    nu_map = {'negative': 0, 'zero': 1, 'positive': 2}
    df['nu_sign'] = df['nu_sign'].map(nu_map)
    required_features = ['nu_sign', 'curve_cross', 'left_PGR1_dominant'] # Hypotheses, , 'right_PGR1_dominant'
    X = df[required_features].copy()
    y = df['coexist'].values
    unique_classes = np.unique(y)
    if unique_classes.size != 2:
        raise ValueError(f"Coexist column must be binary (0/1). Found: {unique_classes}")
    model = RandomForestClassifier(n_estimators=1000, random_state=seed, n_jobs=-1, class_weight='balanced_subsample') # Fit Random Forest
    model.fit(X, y)
    perm_imp = permutation_importance(model, X, y, n_repeats=100, random_state=seed, n_jobs=-1) # Permutation importance
    # SHAP analysis
    shap_imp = pd.Series(np.nan, index=X.columns)
    shap_vals = None
    try:
        explainer = shap.Explainer(model, X, feature_perturbation="interventional")
        sv = explainer(X)
        vals = sv.values
        if vals.ndim == 3:
            shap_vals = vals[:, :, 1]
        elif vals.ndim == 2:
            shap_vals = vals
        else:
            raise ValueError(f"Unsupported SHAP values ndim: {vals.ndim}")
        if shap_vals.shape == (X.shape[1], X.shape[0]):
            shap_vals = shap_vals.T
        if shap_vals.shape != (X.shape[0], X.shape[1]):
            raise ValueError(f"SHAP/Feature dimension mismatch after transpose check: {shap_vals.shape} vs {X.shape}")
        abs_imp = np.abs(shap_vals).mean(axis=0)
        total = abs_imp.sum()
        if total > 0:
            shap_imp = pd.Series((abs_imp / total).round(3), index=X.columns)
    except Exception as e:
        print(f"SHAP Calculation Error: {e}")
    # Build and print the importance table
    imp_df = pd.DataFrame({
        'Feature': X.columns,
        'Permutation Importance': perm_imp.importances_mean.round(3),
        'SHAP Impact (%)': (shap_imp * 100).round(1)
    }).sort_values('SHAP Impact (%)', ascending=False)
    print("\n=== Feature Importance ===")
    print(imp_df.to_string(index=False, float_format="%.2g"))
    if shap_vals is not None:
        try:
            plt.figure(figsize=(10, 6))
            shap.summary_plot(
                shap_vals, X.values,
                feature_names=X.columns,
                plot_type='dot', show=False
            )
            plt.tight_layout()
            os.makedirs('shap', exist_ok=True)
            plt.savefig(f'shap/shap_{filter_option}.png', dpi=150, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"SHAP Visualization Error: {e}")
    else:
        print("Skipping SHAP visualization due to missing values")


def analyze_hypotheses(filter_option, summary_path):
    print("\n" + "="*140)
    print("Hypothesis Definitions:")
    print("\n(H1) Sign of \u03BD (cor_sos)")
    print("\n(H2) Curve crossing (left/right edge dominance differs)")
    print("\n(H3) PGR1 at left edge is higher")
    print("="*50 + "\n")
    df = pd.read_csv(summary_path)
    df['curve_cross'] = (df['left_PGR1_dominant'] != df['right_PGR1_dominant']).astype(int)
    print("\n--- Hypothesis Analysis Results ---")
    analyze_hypotheses_rf(filter_option, summary_path)


def analyze_coexistence_deterministic(filter_option):
    summary_path = f"csv/pgr_analysis_summary_{filter_option}.csv"
    if not os.path.exists(summary_path):
        print(f"\n-----------\nGenerating analysis data for {filter_option}...")
        plot_pgr_figures(filter_option, save_fig=False, extinc_crit_1=False)
    try:
        df = pd.read_csv(summary_path)
        if df.empty:
            raise ValueError("Empty summary file - regenerate manually")
        df = df[df['nu_sign'].isin(['negative', 'zero', 'positive'])]
        df['curve_cross'] = (df['left_PGR1_dominant'] != df['right_PGR1_dominant']).astype(int)
        analyze_hypotheses(filter_option, summary_path)
    except Exception as e:
        print(f"Analysis failed: {str(e)}")
        return
    nu_order = ['negative', 'zero', 'positive']
    cross_labels = ['No Cross', 'Cross']
    nu_symbols = {'negative': '\u03BD<0', 'zero': '\u03BD\u2248 0', 'positive': '\u03BD>0'}
    plt.rcParams.update({
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 10,
        'font.size': 12,
        'lines.linewidth': 1.5
    })
    # 1. Coexistence vs Curve Crossing
    s1 = df.groupby('curve_cross')['coexist'].agg(['sum', 'count'])
    s1['p_co'] = s1['sum'] / s1['count']
    s1['p_no'] = 1 - s1['p_co']
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(s1))
    ax.bar(x, s1['p_co'], color='blue', label='Coexist')
    ax.bar(x, s1['p_no'], bottom=s1['p_co'], color='red', label='Non-coexist')
    # Annotate each segment
    for i in x:
        total = s1.iloc[i]['count']
        p_co = s1.iloc[i]['p_co']
        p_no = s1.iloc[i]['p_no']
        sum_co = int(s1.iloc[i]['sum'])
        sum_no = int(total - sum_co)
        # Coexist segment
        ax.text(i, p_co/2, f"{p_co:.1%}\n({sum_co})",  # :.0g
                ha='center', va='center', color='white', fontsize=8)
        # Non-coexist segment
        ax.text(i, p_co + p_no/2, f"{p_no:.1%}\n({sum_no})",  # :.0g
                ha='center', va='center', color='white', fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(cross_labels)
    ax.set_ylabel("Percentage")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.legend(loc='upper right', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    os.makedirs('img', exist_ok=True)
    plt.savefig(f'img/hypothesis_1_{filter_option}.png', dpi=300)
    plt.close()
    # 2. Coexistence vs nu sign
    s2 = df.groupby('nu_sign')['coexist'].agg(['sum', 'count']).reindex(nu_order)
    s2['p_co'] = s2['sum'] / s2['count']
    s2['p_no'] = 1 - s2['p_co']
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(s2))
    ax.bar(x, s2['p_co'], color='blue', label='Coexist')
    ax.bar(x, s2['p_no'], bottom=s2['p_co'], color='red', label='Non-coexist')
    # Annotate each segment
    for i in x:
        total = s2.iloc[i]['count']
        p_co = s2.iloc[i]['p_co']
        p_no = s2.iloc[i]['p_no']
        sum_co = int(s2.iloc[i]['sum'])
        sum_no = int(total - sum_co)
        ax.text(i, p_co/2, f"{p_co:.1%}\n({sum_co})",  # :.0g
                ha='center', va='center', color='white', fontsize=8)
        ax.text(i, p_co + p_no/2, f"{p_no:.1%}\n({sum_no})",  # :.0g
                ha='center', va='center', color='white', fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([nu_symbols[nu] for nu in nu_order])
    ax.set_ylabel("Percentage")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.legend(loc='upper right', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig(f'img/hypothesis_2_{filter_option}.png', dpi=300)
    plt.close()
    # 3. Coexistence vs Curve Crossing by nu sign
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, nu in zip(axes, nu_order):
        sub = df[df['nu_sign'] == nu]
        s3 = sub.groupby('curve_cross')['coexist'].agg(['sum', 'count'])
        if s3.empty:
            continue
        s3['p_co'] = s3['sum'] / s3['count']
        s3['p_no'] = 1 - s3['p_co']
        x_sub = np.arange(len(s3))
        ax.bar(x_sub, s3['p_co'], color='blue')
        ax.bar(x_sub, s3['p_no'], bottom=s3['p_co'], color='red')
        # Annotate each segment
        for j in x_sub:
            total = s3.iloc[j]['count']
            p_co = s3.iloc[j]['p_co']
            p_no = s3.iloc[j]['p_no']
            sum_co = int(s3.iloc[j]['sum'])
            sum_no = int(total - sum_co)
            ax.text(j, p_co/2, f"{p_co:.1%}\n({sum_co})",  # :.0g
                    ha='center', va='center', color='white', fontsize=8)
            ax.text(j, p_co + p_no/2, f"{p_no:.1%}\n({sum_no})",  # :.0g
                    ha='center', va='center', color='white', fontsize=8)
        ax.set_xticks(x_sub)
        ax.set_xticklabels(cross_labels)
        ax.set_title(nu_symbols[nu])
        if ax == axes[0]:
            ax.set_ylabel("Percentage")
    handles = [plt.Rectangle((0,0),1,1, color=c, edgecolor='k') for c in ['blue', 'red']]
    fig.legend(handles, ['Coexist', 'Non-coexist'], loc='upper right', bbox_to_anchor=(0.99, 0.99))
    plt.tight_layout()
    plt.savefig(f'img/hypothesis_3_{filter_option}.png', dpi=300)
    plt.close()
    # 4. Heatmap: Coexistence count/total and percentage
    heat = df.groupby(['nu_sign', 'curve_cross']).agg(
        total=('coexist', 'count'),
        coexist=('coexist', 'sum')
    ).reset_index().set_index(['nu_sign', 'curve_cross'])
    # Prepare matrices
    total_mat = heat['total'].unstack().reindex(index=nu_order, columns=[0, 1]).fillna(0)
    co_mat = heat['coexist'].unstack().reindex(index=nu_order, columns=[0, 1]).fillna(0)
    pct_mat = co_mat / total_mat.replace(0, np.nan)
    # Create annotation text: coexist/total (pct%)
    annot = (co_mat.astype(int).astype(str) + "/" + total_mat.astype(int).astype(str) + 
             "\n(" + (pct_mat * 100).round(1).astype(str) + "%)").values
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        pct_mat,
        annot=annot,
        fmt='',
        cmap="YlGnBu",
        cbar_kws={'label': 'Coexistence %'},
        linewidths=0.5,
        linecolor='grey',
        ax=ax
    )
    ax.set_xlabel("Curve Crossing")
    ax.set_ylabel("\u03BD Sign")
    ax.set_xticklabels(cross_labels)
    ax.set_yticklabels([nu_symbols[nu] for nu in nu_order], rotation=0)
    plt.tight_layout()
    plt.savefig(f'img/coexistence_heatmap_{filter_option}.png', dpi=300)
    plt.close()
    # 5. Core Analysis Tables
    analysis_data = df.groupby(
        ['nu_sign', 'curve_cross', 'left_PGR1_dominant', 'coexist']
    ).size().unstack(fill_value=0)
    analysis_data['Total'] = analysis_data.sum(axis=1)
    # Ensure consistent ordering
    index = pd.MultiIndex.from_product(
        [nu_order, [0, 1], [0, 1]],
        names=['\u03BD Sign', 'Curve Cross', 'Left N1 Dominant']
    )
    analysis_data = analysis_data.reindex(index, fill_value=0)
    total_all = analysis_data['Total'].sum()
    formatted = analysis_data.copy()
    for col in [0, 1]:
        formatted[col] = (
            analysis_data[col] / total_all
        ).map("{:.1%}".format) + " (" + analysis_data[col].astype(int).astype(str) + ")"
    formatted.to_csv(f'csv/coexistence_output_{filter_option}.csv')
    print("\n=== Coexistence Analysis ===")
    print(formatted.to_string())


def setup_pipeline(filters, base_file, solver, truncate, save_fig, extinc_crit_1):
    os.makedirs('csv', exist_ok=True)
    warnings.filterwarnings("ignore")
    if not os.path.exists(base_file):
        print("Running simulation...")
        mesh = preprocess_data('table1')
        results = [Sim(k, row, extinc_crit_1=extinc_crit_1, solver=solver) 
                   for k, row in tqdm(enumerate(mesh), total=len(mesh))]
        postprocess_results(results, base_file)
    for filter_option in filters:
        filtered_filename = f"csv/annplant_2spp_det_rare_filtered_{filter_option}.csv"
        if not os.path.exists(filtered_filename):
            print(f"\nGenerating data for filter={filter_option}...")
            cor_figure(filter_option, truncate)
        summary_path = f"csv/pgr_analysis_summary_{filter_option}.csv"
        if not os.path.exists(summary_path):
            plot_pgr_figures(filter_option, save_fig, extinc_crit_1=False)
        try:
            filtered_data = pd.read_csv(filtered_filename)
            bool_cols = ['Coexist', 'A', 'B', 'C', 'D']
            for col in bool_cols:
                if col in filtered_data.columns:
                    filtered_data[col] = filtered_data[col].astype(bool)
            print("\nAnalysis:")
            analyze_coexistence_effect(filtered_data, False)
            plot_phase_plane()
            count_abcd(filtered_data)
            if 'C' in filtered_data.columns:
                compare_counts_test(filtered_data[filtered_data['C']], True)
        except Exception as e:
            print(f"Processing error: {str(e)}")
        analyze_coexistence_deterministic(filter_option)


def main():
    filters = ['on', 'off']
    base_file = "csv/annplant_2spp_det_rare.csv"
    solver = 'dynamics' # 'analyN'
    truncate = False
    save_fig = True
    extinc_crit_1 = False
    setup_pipeline(filters, base_file, solver, truncate, save_fig, extinc_crit_1)


if __name__ == "__main__":
    main()
