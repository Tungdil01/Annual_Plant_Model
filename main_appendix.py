# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# ### The code is a modification of the Yenni et al. (2012) analysis:
# #### - runs the analysis with and without the filter S1 >= 1 & S2 >= 1
# #### - includes Cushing et al. (2004) analytical results
#
# #### their original code: https://github.com/gmyenni/RareStabilizationSimulation

import os
import time
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.proportion import proportion_confint
from tqdm import tqdm
from numba import jit
from joblib import Parallel, delayed
from scipy.stats import qmc


# # analyN_function.r

def getEqDensity(r1, r2, a11, a12, a21, a22): # Coexistence equilibrium populations
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


def time_simul(r1, r2, a11, a22, a12, a21, y01=5.0, y02=5.0, eps=1e-3):
    y1 = np.array([y01], dtype=np.float64)
    y2 = np.array([y02], dtype=np.float64)
    stop_run = False
    i = 0
    while not stop_run and i < 1000:
        denom1 = 1 + a11 * y1[i] + a12 * y2[i]
        denom2 = 1 + a22 * y2[i] + a21 * y1[i]
        per_cap1 = r1 / denom1
        per_cap2 = r2 / denom2
        new_y1 = y1[i] * per_cap1
        new_y2 = y2[i] * per_cap2
        y1 = np.append(y1, new_y1)
        y2 = np.append(y2, new_y2)
        if i >= 1:
            if (abs(y1[-1] - y1[-2]) < eps and abs(y2[-1] - y2[-2]) < eps):
                stop_run = True
        i += 1
    return y1, y2


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
    # Calculate conditions for A, B, C
    A1 = P > E2 and E1 > Q
    A2 = E2 > P and Q > E1
    B = P > E2 and Q > E1
    C = E2 > P and E1 > Q
    # Call getPCG to calculate PGR1 and PGR2
    PGR1, PGR2 = getPCG(r1, r2, a11, a12, a21, a22, N1, N2)
    if extinc_crit_1:
        Coexist = 0 if N1 < 1 or N2 < 1 else 1
    else:
        Coexist = 0 if N1 < 1.0e-6 or N2 < 1.0e-6 else 1
    return {"FE1": FE1, "S1": S1, "FE2": FE2, "S2": S2, "Rank": Rank, "Coexist": Coexist, "Asy": Asy, "cor_sos": cor_sos, "Rare": Rare, "PGR1": PGR1, "PGR2": PGR2, "A1": A1, "A2": A2, "B": B, "C": C}


# -

# # annualplant_2spp_det_par.r

# def preprocess_data(pars):
#     # Defines frequency-dependent parameters
#     if pars == 'r_code': # Their R code
#          r1_v = np.arange(10, 21, 1)
#          r2_v = np.arange(10, 21, 1)
#          a11_v = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1, 1.5, 2, 2.5, 3])
#          a12_v = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1])
#          a21_v = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1])
#          a22_v = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1])
#     elif pars == 'table1': # Reproduce their Table 1
#         r1_v = np.arange(15, 21, 1)
#         r2_v = np.arange(15, 21, 1)
#         a11_v = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1, 1.5, 2, 2.5, 3])
#         a12_v = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1])
#         a21_v = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1])
#         a22_v = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1])
#     elif pars == 'paper': # They describe in the paper
#          r1_v = np.arange(15, 21, 1)
#          r2_v = np.arange(11, 21, 1)
#          a11_v = np.array([0.7, 0.3, 0.5, 0.7, 0.9, 1, 1.5, 2, 2.5, 3])
#          a12_v = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1])
#          a21_v = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1])
#          a22_v = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1])
#     else: # broad # we use in our paper in contrast to the narrow sets above from Yenni
#         r1_v = np.arange(1, 21, 1)
#         r2_v = np.arange(1, 21, 1)
#         a11_v = np.arange(0.1, 3, 0.1)
#         a12_v = np.arange(0.1, 3, 0.1)
#         a21_v = np.arange(0.1, 3, 0.1)
#         a22_v = np.arange(0.1, 3, 0.1)
#     # Generate all combinations of parameters using NumPy's meshgrid
#     mesh = np.array(np.meshgrid(r1_v, r2_v, a11_v, a12_v, a21_v, a22_v)).T.reshape(-1, 6)
#     return mesh
#
# def Sim(k, mesh_row, extinc_crit_1=False):
#     start_time = time.time()
#     r1, r2, a11, a12, a21, a22 = mesh_row
#     N1, N2 = getEqDensity(r1, r2, a11, a12, a21, a22)
#     metrics = calculate_metrics(r1, r2, a11, a12, a21, a22, N1, N2, extinc_crit_1)
#     execution_time = time.time() - start_time
#     return {**metrics, "N1": N1, "N2": N2, "r1": r1, "r2": r2, "a11": a11, "a12": a12, "a21": a21, "a22": a22}

# +
def preprocess_data(case):
    if case == 'yenni':
        r1_v = np.arange(15, 21, 1)
        r2_v = np.arange(15, 21, 1)
        a11_v = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1, 1.5, 2, 2.5, 3])
        a12_v = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1])
        a21_v = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1])
        a22_v = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1])
        mesh = np.array(np.meshgrid(r1_v, r2_v, a11_v, a12_v, a21_v, a22_v)).T.reshape(-1, 6)
    else:
        n_samples = 77760
        sampler = qmc.LatinHypercube(d=6, seed=1234)
        sample = sampler.random(n=n_samples)
        lb = np.array([1.0, 1.0, 0.1, 0.1, 0.1, 0.1])
        ub = np.array([21.0, 21.0, 3.0, 3.0, 3.0, 3.0])
        mesh = lb + (ub - lb) * sample
    return mesh

def Sim(k, mesh_row, extinc_crit_1=False, use_correct_equilibrium=False):
    r1, r2, a11, a12, a21, a22 = mesh_row
    if use_correct_equilibrium:
        E1 = (r1-1)/a11; E2 = (r2-1)/a22; P = (r1-1)/a12; Q = (r2-1)/a21
        A1 = (P > E2) and (E1 > Q); A2 = (E2 > P) and (Q > E1)
        B = (P > E2) and (Q > E1); C = (E2 > P) and (E1 > Q)
        if B:
            N1, N2 = getEqDensity(r1, r2, a11, a12, a21, a22)
        elif A1:
            N1 = (r1-1)/a11; N2 = 0.0
        elif A2:
            N1 = 0.0; N2 = (r2-1)/a22
        else:
            N1 = 0.0; N2 = 0.0
    else:
        N1, N2 = getEqDensity(r1, r2, a11, a12, a21, a22)
    metrics = calculate_metrics(r1, r2, a11, a12, a21, a22, N1, N2, extinc_crit_1)
    return {**metrics, "N1": N1, "N2": N2, "r1": r1, "r2": r2, "a11": a11, "a12": a12, "a21": a21, "a22": a22}

def postprocess_results(results, outfile):
    column_order = ['r1', 'r2', 'a11', 'a12', 'a21', 'a22', 'N1', 'N2', 'FE1', 'S1', 'FE2', 'S2', 'Rank', 'Coexist', 'Asy', 'cor_sos', 'Rare', 'PGR1', 'PGR2', 'A1', 'A2', 'B', 'C']
    simul = pd.DataFrame(results, columns=column_order)
    simul.to_csv(outfile, index=False)


# -

# # cor_figure.r

# def cor_figure(filter, truncate=False):
#     dat_det = pd.read_csv("csv/annplant_2spp_det_rare.csv")
#     for col in ['A1', 'A2', 'B', 'C']:
#         dat_det[col] = dat_det[col].astype(bool)
#     # keep only stable coexistence (B) or competitive exclusion (A1/A2), remove saddle (C) and borderline
#     dat_det = dat_det[(dat_det['A1']) | (dat_det['A2']) | (dat_det['B'])]  # filter_dynamical = on means we remove saddle and borderline, which is the more natural; while filter = off wrongly allows saddle and borderline, closer to Yenni
#     if filter == 'inverted':
#         dat_det = dat_det.query('Rank == 2 & S1 < 1 & S2 < 1').copy()
#     elif filter == 'on':
#         dat_det = dat_det.query('Rank == 2 & S1 >= 1 & S2 >= 1').copy()
#     else: # 'off'
#         dat_det = dat_det.query('Rank == 2').copy()
#     dat_det.reset_index(drop=True, inplace=True)
#     if truncate:
#         dat_det = np.trunc(dat_det * 100) / 100.0
#     dat_det.sort_values(by=['a22', 'a21', 'a12', 'a11', 'r2', 'r1'], inplace=True)
#     dat_det.to_csv(f"csv/annplant_2spp_det_rare_filtered_{filter}.csv", index=False)

def cor_figure(case):
    if case == 'yenni':
        dat_det = pd.read_csv("csv/annplant_2spp_det_rare_yenni.csv")
        for col in ['A1', 'A2', 'B', 'C']: dat_det[col] = dat_det[col].astype(bool)
        dat_det = dat_det.query('Rank == 2 & S1 >= 1 & S2 >= 1').copy()
    else:
        dat_det = pd.read_csv("csv/annplant_2spp_det_rare_broad.csv")
        for col in ['A1', 'A2', 'B', 'C']: dat_det[col] = dat_det[col].astype(bool)
        dat_det = dat_det[(dat_det['A1']) | (dat_det['A2']) | (dat_det['B'])].copy()
        dat_det = dat_det.query('Rank == 2').copy()
    dat_det.reset_index(drop=True, inplace=True)
    dat_det.sort_values(by=['a22', 'a21', 'a12', 'a11', 'r2', 'r1'], inplace=True)
    outfile = f"csv/annplant_2spp_det_rare_filtered_{case}.csv"
    dat_det.to_csv(outfile, index=False)


# # figures_det.r

# +
def perform_logistic_regression(dat, analysis_type):
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
    return intercept, coef, std_err, z_scores, p_values

def calculate_proportions(dat, correlation_column):
    proportions = {}
    proportions[f'positive_coexistence_{correlation_column}'] = len(dat[(dat[correlation_column] >= 0) & (dat['Coexist'] == 1)])
    proportions[f'positive_exclusion_{correlation_column}'] = len(dat[(dat[correlation_column] >= 0) & (dat['Coexist'] == 0)])
    proportions[f'negative_coexistence_{correlation_column}'] = len(dat[(dat[correlation_column] < 0) & (dat['Coexist'] == 1)])
    proportions[f'negative_exclusion_{correlation_column}'] = len(dat[(dat[correlation_column] < 0) & (dat['Coexist'] == 0)])
    return proportions

def report_coexistence_analysis(proportions, correlation_column):
    positive_key = f'positive_coexistence_{correlation_column}'
    negative_key = f'negative_coexistence_{correlation_column}'
    positive_excl_key = f'positive_exclusion_{correlation_column}'
    negative_excl_key = f'negative_exclusion_{correlation_column}'
    pos_total = proportions[positive_key] + proportions[positive_excl_key]
    neg_total = proportions[negative_key] + proportions[negative_excl_key]
    neg_confint = proportion_confint(count=proportions[negative_key], nobs=neg_total, alpha=0.05, method='wilson')
    pos_confint = proportion_confint(count=proportions[positive_key], nobs=pos_total, alpha=0.05, method='wilson')
    print(f"\nAnalysis on Negative \u03BD for {correlation_column.upper()}:")
    print(f"Proportion of coexistence with \u03BD \u2265 0: {proportions[positive_key] / pos_total:.3g} (95% CI: ({pos_confint[0]:.3g}, {pos_confint[1]:.3g}))")
    print(f"Proportion of coexistence with \u03BD < 0: {proportions[negative_key] / neg_total:.3g} (95% CI: ({neg_confint[0]:.3g}, {neg_confint[1]:.3g}))")

def analyze_coexistence_effect(data):
    models_results = {}
    correlation_column = 'cor_sos'
    analysis_type = 'SoS'
    if correlation_column not in data.columns:
        return models_results
    print(f"\n--- Analysis for {analysis_type} ---")
    intercept, coef, std_err, z_scores, p_values = perform_logistic_regression(data, analysis_type)
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
    # Confidence intervals
    pos_total = proportions[f'positive_coexistence_{correlation_column}'] + proportions[f'positive_exclusion_{correlation_column}']
    neg_total = proportions[f'negative_coexistence_{correlation_column}'] + proportions[f'negative_exclusion_{correlation_column}']
    pos_confint = proportion_confint(count=proportions[f'positive_coexistence_{correlation_column}'], nobs=pos_total, alpha=0.05, method='wilson')
    neg_confint = proportion_confint(count=proportions[f'negative_coexistence_{correlation_column}'], nobs=neg_total, alpha=0.05, method='wilson')
    # Decision logic
    if neg_confint[1] >= pos_confint[0] and neg_confint[0] <= pos_confint[1]:
        print(f"The confidence intervals overlap for {analysis_type}, indicating they are statistically the same, not supporting the authors' results.")
    elif neg_confint[1] > pos_confint[0]:
        print(f"Higher coexistence observed with \u03BD < 0 for {analysis_type}, supporting the authors' results.")
    else:
        print(f"Higher coexistence observed with \u03BD \u2265 0 for {analysis_type}, not supporting the authors' results.")
    return models_results


# -

def plot_phase_plane():
    # Parameters for each scenario
    scenarios = {
        "A1: $E_1 > Q$ and $P > E_2$": {'r1': 18, 'r2': 16, 'a11': 0.5, 'a12': 1, 'a21': 1, 'a22': 1},
        "A2: $Q > E_1$ and $E_2 > P$": {'r1': 20, 'r2': 15, 'a11': 2, 'a12': 1, 'a21': 1, 'a22': 0.5},
        "B: $Q > E_1$ and $P > E_2$":  {'r1': 20, 'r2': 15, 'a11': 3, 'a12': 0.5, 'a21': 1, 'a22': 0.5},
        "C: $E_1 > Q$ and $E_2 > P$":  {'r1': 16, 'r2': 18, 'a11': 0.3, 'a12': 1, 'a21': 1, 'a22': 0.5},
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
    plt.savefig('img/phase_plane.pdf', bbox_inches='tight', dpi=300)
    plt.show()


def load_and_compute_classification(txt_path, extinc_crit_1):
    col_names = ['ID','l1','l2','a11','a12','a21','a22','N1','N2','E1','S1','E2','S2','Rank','CoexistRank','Asy','cor','Rare']
    dat = pd.read_csv(txt_path, sep=',', quotechar='"', header=None, names=col_names)
    for c in dat.columns:
        dat[c] = pd.to_numeric(dat[c], errors='coerce')
    dat.dropna(subset=['l1','l2','a11','a12','a21','a22','N1','N2'], inplace=True)
    dat['r1'] = dat['l1']
    dat['r2'] = dat['l2']
    dat['cor_sos'] = dat['cor']
    if extinc_crit_1:
        dat['Coexist'] = ((dat['N1'] >= 1) & (dat['N2'] >= 1)).astype(int)
    else:
        dat['Coexist'] = ((dat['N1'] >= 1e-6) & (dat['N2'] >= 1e-6)).astype(int)
    E1 = (dat['r1'] - 1) / dat['a11']
    E2 = (dat['r2'] - 1) / dat['a22']
    P = (dat['r1'] - 1) / dat['a12']
    Q = (dat['r2'] - 1) / dat['a21']
    dat['A1'] = (P > E2) & (E1 > Q)
    dat['A2'] = (E2 > P) & (Q > E1)
    dat['B'] = (P > E2) & (Q > E1)
    dat['C'] = (E2 > P) & (E1 > Q)
    return dat


def print_classification_table(rows_data):
    df = pd.DataFrame(rows_data)
    print(df.to_string(index=False))


def report_classification_from_txt(txt_path, extinc_crit_1):
    dat = load_and_compute_classification(txt_path, extinc_crit_1)
    total = len(dat)
    exclusion_mask = dat['A1'] | dat['A2']
    coex_mask = dat['B']
    saddle_mask = dat['C']
    borderline_mask = (~dat['A1']) & (~dat['A2']) & (~dat['B']) & (~dat['C'])
    n_exclusion = exclusion_mask.sum()
    n_coex = coex_mask.sum()
    n_saddle = saddle_mask.sum()
    n_borderline = borderline_mask.sum()
    correct_excl = (exclusion_mask & (dat['Coexist'] == 0)).sum()
    wrong_excl = (exclusion_mask & (dat['Coexist'] == 1)).sum()
    correct_coex = (coex_mask & (dat['Coexist'] == 1)).sum()
    wrong_coex = (coex_mask & (dat['Coexist'] == 0)).sum()
    wrong_saddle_coex = (saddle_mask & (dat['Coexist'] == 1)).sum()
    wrong_saddle_excl = (saddle_mask & (dat['Coexist'] == 0)).sum()
    wrong_border_coex = (borderline_mask & (dat['Coexist'] == 1)).sum()
    wrong_border_excl = (borderline_mask & (dat['Coexist'] == 0)).sum()
    pct_correct_excl = correct_excl/n_exclusion*100 if n_exclusion>0 else 0.0
    pct_wrong_excl = wrong_excl/n_exclusion*100 if n_exclusion>0 else 0.0
    pct_correct_coex = correct_coex/n_coex*100 if n_coex>0 else 0.0
    pct_wrong_coex = wrong_coex/n_coex*100 if n_coex>0 else 0.0
    pct_saddle_coex = wrong_saddle_coex/n_saddle*100 if n_saddle>0 else 0.0
    pct_saddle_excl = wrong_saddle_excl/n_saddle*100 if n_saddle>0 else 0.0
    pct_border_coex = wrong_border_coex/n_borderline*100 if n_borderline>0 else 0.0
    pct_border_excl = wrong_border_excl/n_borderline*100 if n_borderline>0 else 0.0
    total_correct = correct_excl + correct_coex
    total_wrong = total - total_correct
    pct_total_correct = total_correct/total*100 if total>0 else 0.0
    pct_total_wrong = total_wrong/total*100 if total>0 else 0.0
    print(f"Total parameter sets: {total}")
    print(f"Stable coexistence (B): {n_coex} ({n_coex/total:.3g})")
    print(f"Competitive exclusion (total A1+A2): {n_exclusion} ({n_exclusion/total:.3g})")
    print(f"Saddle equilibrium (C): {n_saddle} ({n_saddle/total:.3g})")
    print(f"Borderline (equalities): {n_borderline} ({n_borderline/total:.3g})")
    rows = [
        {"Theoretical class": "Exclusion (A1+A2)", "Correct (percent)": f"{correct_excl} ({pct_correct_excl:.3g}%)", "Wrong (percent)": f"{wrong_excl} ({pct_wrong_excl:.3g}%)", "Total": n_exclusion},
        {"Theoretical class": "Coexistence B", "Correct (percent)": f"{correct_coex} ({pct_correct_coex:.3g}%)", "Wrong (percent)": f"{wrong_coex} ({pct_wrong_coex:.3g}%)", "Total": n_coex},
    ]
    if n_saddle > 0:
        rows.append({"Theoretical class": "Saddle C (flagged 'coexist')", "Correct (percent)": "0 (0%)", "Wrong (percent)": f"{wrong_saddle_coex} ({pct_saddle_coex:.3g}%)", "Total": n_saddle})
        rows.append({"Theoretical class": "Saddle C (flagged 'exclusion')", "Correct (percent)": "0 (0%)", "Wrong (percent)": f"{wrong_saddle_excl} ({pct_saddle_excl:.3g}%)", "Total": n_saddle})
    if n_borderline > 0:
        rows.append({"Theoretical class": "Borderline (flagged 'coexist')", "Correct (percent)": "0 (0%)", "Wrong (percent)": f"{wrong_border_coex} ({pct_border_coex:.3g}%)", "Total": n_borderline})
        rows.append({"Theoretical class": "Borderline (flagged 'exclusion')", "Correct (percent)": "0 (0%)", "Wrong (percent)": f"{wrong_border_excl} ({pct_border_excl:.3g}%)", "Total": n_borderline})
    rows.append({"Theoretical class": "TOTAL", "Correct (percent)": f"{total_correct} ({pct_total_correct:.3g}%)", "Wrong (percent)": f"{total_wrong} ({pct_total_wrong:.3g}%)", "Total": total})
    print("\nClassification table:")
    print_classification_table(rows)


# def setup_pipeline(filters, base_file, truncate, extinc_crit_1):
#     os.makedirs('csv', exist_ok=True)
#     warnings.filterwarnings("ignore")
#     mesh = preprocess_data('table1') # table1 is Yenni
#     results = [Sim(k, row, extinc_crit_1=extinc_crit_1)
#                 for k, row in tqdm(enumerate(mesh), total=len(mesh))]
#     postprocess_results(results, base_file)
#     for filter_option in filters:
#         filtered_filename = f"csv/annplant_2spp_det_rare_filtered_{filter_option}.csv"
#         print(f"\nGenerating data for filter={filter_option}...")
#         cor_figure(filter_option, truncate)
#         summary_path = f"csv/pgr_analysis_summary_{filter_option}.csv"
#         filtered_data = pd.read_csv(filtered_filename)
#         analyze_coexistence_effect(filtered_data)
#         plot_phase_plane()

def report_classification_from_df(dat, extinc_crit_1):
    if 'Coexist' not in dat.columns:
        if extinc_crit_1:
            dat['Coexist'] = ((dat['N1'] >= 1) & (dat['N2'] >= 1)).astype(int)
        else:
            dat['Coexist'] = ((dat['N1'] >= 1e-6) & (dat['N2'] >= 1e-6)).astype(int)
    total = len(dat)
    exclusion_mask = dat['A1'] | dat['A2']; coex_mask = dat['B']; saddle_mask = dat['C']
    borderline_mask = (~dat['A1']) & (~dat['A2']) & (~dat['B']) & (~dat['C'])
    n_exclusion = exclusion_mask.sum(); n_coex = coex_mask.sum(); n_saddle = saddle_mask.sum(); n_borderline = borderline_mask.sum()
    correct_excl = (exclusion_mask & (dat['Coexist'] == 0)).sum(); wrong_excl = (exclusion_mask & (dat['Coexist'] == 1)).sum()
    correct_coex = (coex_mask & (dat['Coexist'] == 1)).sum(); wrong_coex = (coex_mask & (dat['Coexist'] == 0)).sum()
    wrong_saddle_coex = (saddle_mask & (dat['Coexist'] == 1)).sum(); wrong_saddle_excl = (saddle_mask & (dat['Coexist'] == 0)).sum()
    wrong_border_coex = (borderline_mask & (dat['Coexist'] == 1)).sum(); wrong_border_excl = (borderline_mask & (dat['Coexist'] == 0)).sum()
    pct_correct_excl = correct_excl/n_exclusion*100 if n_exclusion>0 else 0.0; pct_wrong_excl = wrong_excl/n_exclusion*100 if n_exclusion>0 else 0.0
    pct_correct_coex = correct_coex/n_coex*100 if n_coex>0 else 0.0; pct_wrong_coex = wrong_coex/n_coex*100 if n_coex>0 else 0.0
    pct_saddle_coex = wrong_saddle_coex/n_saddle*100 if n_saddle>0 else 0.0; pct_saddle_excl = wrong_saddle_excl/n_saddle*100 if n_saddle>0 else 0.0
    pct_border_coex = wrong_border_coex/n_borderline*100 if n_borderline>0 else 0.0; pct_border_excl = wrong_border_excl/n_borderline*100 if n_borderline>0 else 0.0
    total_correct = correct_excl + correct_coex; total_wrong = total - total_correct
    pct_total_correct = total_correct/total*100 if total>0 else 0.0; pct_total_wrong = total_wrong/total*100 if total>0 else 0.0
    print(f"Total parameter sets: {total}")
    print(f"Stable coexistence (B): {n_coex} ({n_coex/total:.3g})")
    print(f"Competitive exclusion (total A1+A2): {n_exclusion} ({n_exclusion/total:.3g})")
    print(f"Saddle equilibrium (C): {n_saddle} ({n_saddle/total:.3g})")
    print(f"Borderline (equalities): {n_borderline} ({n_borderline/total:.3g})")
    rows = [
        {"Theoretical class": "Exclusion (A1+A2)", "Correct (percent)": f"{correct_excl} ({pct_correct_excl:.3g}%)", "Wrong (percent)": f"{wrong_excl} ({pct_wrong_excl:.3g}%)", "Total": n_exclusion},
        {"Theoretical class": "Coexistence B", "Correct (percent)": f"{correct_coex} ({pct_correct_coex:.3g}%)", "Wrong (percent)": f"{wrong_coex} ({pct_wrong_coex:.3g}%)", "Total": n_coex},
    ]
    if n_saddle > 0:
        rows.append({"Theoretical class": "Saddle C (flagged 'coexist')", "Correct (percent)": "0 (0%)", "Wrong (percent)": f"{wrong_saddle_coex} ({pct_saddle_coex:.3g}%)", "Total": n_saddle})
        rows.append({"Theoretical class": "Saddle C (flagged 'exclusion')", "Correct (percent)": "0 (0%)", "Wrong (percent)": f"{wrong_saddle_excl} ({pct_saddle_excl:.3g}%)", "Total": n_saddle})
    if n_borderline > 0:
        rows.append({"Theoretical class": "Borderline (flagged 'coexist')", "Correct (percent)": "0 (0%)", "Wrong (percent)": f"{wrong_border_coex} ({pct_border_coex:.3g}%)", "Total": n_borderline})
        rows.append({"Theoretical class": "Borderline (flagged 'exclusion')", "Correct (percent)": "0 (0%)", "Wrong (percent)": f"{wrong_border_excl} ({pct_border_excl:.3g}%)", "Total": n_borderline})
    rows.append({"Theoretical class": "TOTAL", "Correct (percent)": f"{total_correct} ({pct_total_correct:.3g}%)", "Wrong (percent)": f"{total_wrong} ({pct_total_wrong:.3g}%)", "Total": total})
    print("\nClassification table:")
    print_classification_table(rows)


def generate_comprehensive_table():
    print("Comprehensive impact breakdown of the three differences:\n")
    for param_label, param_key in [("Narrow ranges (Yenni)", "yenni"), ("Broad ranges", "broad")]:
        mesh = preprocess_data(param_key)
        n_total = len(mesh)
        A1_count = A2_count = B_count = C_count = border_count = 0
        yenni_correct_excl = yenni_wrong_excl = 0
        yenni_correct_coex = yenni_wrong_coex = 0
        broad_correct_excl = broad_wrong_excl = 0
        broad_correct_coex = broad_wrong_coex = 0
        yenni_miscl_sfilter = 0
        yenni_miscl_formula = 0
        yenni_miscl_both = 0
        for row in tqdm(mesh, total=n_total, desc=f"Processing {param_label}"):
            r1, r2, a11, a12, a21, a22 = row
            E1 = (r1-1)/a11; E2 = (r2-1)/a22; P = (r1-1)/a12; Q = (r2-1)/a21
            A1 = (P > E2) and (E1 > Q); A2 = (E2 > P) and (Q > E1)
            B = (P > E2) and (Q > E1); C = (E2 > P) and (E1 > Q)
            if A1: theo = 'A1'
            elif A2: theo = 'A2'
            elif B: theo = 'B'
            elif C: theo = 'C'
            else: theo = 'border'
            if A1 or A2:
                if A1: A1_count += 1
                else: A2_count += 1
                true_N1 = E1 if A1 else 0.0
                true_N2 = 0.0 if A1 else E2
                true_coexist = 0
            elif B:
                B_count += 1
                denom = a11*a22 - a12*a21
                true_N1 = ((r1-1)*a22 - (r2-1)*a12) / denom
                true_N2 = ((r2-1)*a11 - (r1-1)*a21) / denom
                true_coexist = 1 if (true_N1 >= 1.0e-6 and true_N2 >= 1.0e-6) else 0
            elif C:
                C_count += 1
                true_N1 = 0.0; true_N2 = 0.0
                true_coexist = 0
            else:
                border_count += 1
                true_N1 = 0.0; true_N2 = 0.0
                true_coexist = 0
            S1, S2 = SOS(r1, r2, a11, a12, a21, a22)
            Y_N1, Y_N2 = getEqDensity(r1, r2, a11, a12, a21, a22)
            if S1 >= 1 and S2 >= 1:
                Y_coexist = 1 if (Y_N1 >= 1 and Y_N2 >= 1) else 0
            else:
                Y_coexist = 0
            O_N1, O_N2 = true_N1, true_N2
            O_coexist = true_coexist
            if theo in ['A1','A2']:
                if Y_coexist == 0: yenni_correct_excl += 1
                else: yenni_wrong_excl += 1
            elif theo == 'B':
                if Y_coexist == 1: yenni_correct_coex += 1
                else: yenni_wrong_coex += 1
            if O_coexist == true_coexist:
                if theo in ['A1','A2']: broad_correct_excl += 1
                elif theo == 'B': broad_correct_coex += 1
            else:
                if theo in ['A1','A2']: broad_wrong_excl += 1
                elif theo == 'B': broad_wrong_coex += 1
            if Y_coexist != O_coexist:
                Y_coexist_nofilter = 1 if (Y_N1 >= 1 and Y_N2 >= 1) else 0
                if Y_coexist_nofilter == O_coexist:
                    yenni_miscl_sfilter += 1
                elif (S1 >= 1 and S2 >= 1) and (Y_coexist == O_coexist):
                    pass
                else:
                    if S1 >= 1 and S2 >= 1:
                        O_coexist_sfilter = true_coexist
                        if (O_coexist_sfilter == O_coexist) and (Y_coexist != O_coexist):
                            yenni_miscl_formula += 1
                        else:
                            yenni_miscl_both += 1
                    else:
                        yenni_miscl_formula += 1
        n_excl = A1_count + A2_count
        pct_yenni_excl_correct = yenni_correct_excl/n_excl*100 if n_excl>0 else 0.0
        pct_yenni_coex_correct = yenni_correct_coex/B_count*100 if B_count>0 else 0.0
        pct_yenni_total = (yenni_correct_excl+yenni_correct_coex)/(n_excl+B_count)*100 if (n_excl+B_count)>0 else 0.0
        pct_broad_excl_correct = broad_correct_excl/n_excl*100 if n_excl>0 else 0.0
        pct_broad_coex_correct = broad_correct_coex/B_count*100 if B_count>0 else 0.0
        pct_broad_total = (broad_correct_excl+broad_correct_coex)/(n_excl+B_count)*100 if (n_excl+B_count)>0 else 0.0
        print(f"{'='*70}")
        print(f"Parameter set: {param_label} (n={n_total})")
        print(f"  Theoretical classes: A1={A1_count}, A2={A2_count}, B={B_count}, C={C_count}, Border={border_count}")
        print(f"\n  Yenni et al. method (S filter on, incorrect formula):")
        print(f"    Exclusion correct: {yenni_correct_excl} ({pct_yenni_excl_correct:.3g}%)  wrong: {yenni_wrong_excl}")
        print(f"    Coexistence correct: {yenni_correct_coex} ({pct_yenni_coex_correct:.3g}%)  wrong: {yenni_wrong_coex}")
        print(f"    Total correct: {yenni_correct_excl+yenni_correct_coex} ({pct_yenni_total:.3g}%)")
        print(f"\n  Our method (correct formula, no S filter):")
        print(f"    Exclusion correct: {broad_correct_excl} ({pct_broad_excl_correct:.3g}%)  wrong: {broad_wrong_excl}")
        print(f"    Coexistence correct: {broad_correct_coex} ({pct_broad_coex_correct:.3g}%)  wrong: {broad_wrong_coex}")
        print(f"    Total correct: {broad_correct_excl+broad_correct_coex} ({pct_broad_total:.3g}%)")
        print(f"\n  Sources of Yenni's misclassifications:")
        print(f"    Due to S >= 1 filter alone: {yenni_miscl_sfilter}")
        print(f"    Due to incorrect equilibrium formula alone: {yenni_miscl_formula}")
        print(f"    Require both factors: {yenni_miscl_both}")
        print(f"{'='*70}\n")


def generate_filtered_analysis(case):
    cor_figure(case)
    filtered_file = f"csv/annplant_2spp_det_rare_filtered_{case}.csv"
    filtered_data = pd.read_csv(filtered_file)
    analyze_coexistence_effect(filtered_data)


def run_pipeline(case):
    if case == 'yenni':
        extinc_crit_1 = True
        use_correct_equilibrium = False
        param_set = 'yenni'
    else:
        extinc_crit_1 = False
        use_correct_equilibrium = True
        param_set = 'broad'
    os.makedirs('csv', exist_ok=True)
    warnings.filterwarnings("ignore")
    mesh = preprocess_data(param_set)
    n_jobs = -1 if param_set == 'broad' else 1
    if n_jobs == -1:
        results = Parallel(n_jobs=-1)(delayed(Sim)(k, row, extinc_crit_1, use_correct_equilibrium) for k, row in tqdm(enumerate(mesh), total=len(mesh)))
    else:
        results = [Sim(k, row, extinc_crit_1, use_correct_equilibrium) for k, row in tqdm(enumerate(mesh), total=len(mesh))]
    base_file = f"csv/annplant_2spp_det_rare_{case}.csv"
    postprocess_results(results, base_file)
    dat = pd.read_csv(base_file)
    E1 = (dat['r1']-1)/dat['a11']; E2 = (dat['r2']-1)/dat['a22']
    P = (dat['r1']-1)/dat['a12']; Q = (dat['r2']-1)/dat['a21']
    dat['A1'] = (P > E2) & (E1 > Q); dat['A2'] = (E2 > P) & (Q > E1)
    dat['B'] = (P > E2) & (Q > E1); dat['C'] = (E2 > P) & (E1 > Q)
    report_classification_from_df(dat, extinc_crit_1)
    generate_filtered_analysis(case)
    plot_phase_plane()


def count_legitimate_removed_by_sfilter():
    mesh = preprocess_data('yenni')
    results = []
    for row in mesh:
        r1, r2, a11, a12, a21, a22 = row
        E1 = (r1-1)/a11; E2 = (r2-1)/a22; P = (r1-1)/a12; Q = (r2-1)/a21
        A1 = (P > E2) and (E1 > Q)
        A2 = (E2 > P) and (Q > E1)
        B = (P > E2) and (Q > E1)
        C = (E2 > P) and (E1 > Q)
        theo = 'A1' if A1 else ('A2' if A2 else ('B' if B else ('C' if C else 'border')))
        S1, S2 = SOS(r1, r2, a11, a12, a21, a22)
        passes_filter = (S1 >= 1 and S2 >= 1)
        results.append((theo, passes_filter))
    df = pd.DataFrame(results, columns=['theo', 'passes_S_filter'])
    excl_mask = df['theo'].isin(['A1', 'A2'])
    coex_mask = df['theo'] == 'B'
    n_excl_total = excl_mask.sum()
    n_coex_total = coex_mask.sum()
    n_excl_passing = (excl_mask & df['passes_S_filter']).sum()
    n_coex_passing = (coex_mask & df['passes_S_filter']).sum()
    n_excl_removed = n_excl_total - n_excl_passing
    n_coex_removed = n_coex_total - n_coex_passing
    total_legitimate = n_excl_total + n_coex_total
    total_removed = n_excl_removed + n_coex_removed
    pct_excl_removed = n_excl_removed/n_excl_total*100 if n_excl_total>0 else 0.0
    pct_coex_removed = n_coex_removed/n_coex_total*100 if n_coex_total>0 else 0.0
    pct_total_removed = total_removed/total_legitimate*100 if total_legitimate>0 else 0.0
    print("\nLegitimate parameter sets (A1, A2, B only) removed by Yenni's S>=1 filter:")
    rows = [
        {"Theoretical class": "Exclusion (A1+A2)", "Removed by S<1": f"{n_excl_removed} ({pct_excl_removed:.3g}%)", "Total legitimate": n_excl_total},
        {"Theoretical class": "Coexistence B", "Removed by S<1": f"{n_coex_removed} ({pct_coex_removed:.3g}%)", "Total legitimate": n_coex_total},
        {"Theoretical class": "TOTAL", "Removed by S<1": f"{total_removed} ({pct_total_removed:.3g}%)", "Total legitimate": total_legitimate}
    ]
    print_classification_table(rows)


def main():
    report_classification_from_txt("csv/annplant_2spp_det_rare.txt", True)
    count_legitimate_removed_by_sfilter()
    case = 'compare'
    if case == 'compare':
        generate_comprehensive_table()
        run_pipeline('yenni')
        run_pipeline('broad')
    else:
        run_pipeline(case)


if __name__ == "__main__":
    main()
