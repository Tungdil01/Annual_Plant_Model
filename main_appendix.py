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

# ### The code reproduces the original Yenni et al. (2012) deterministic analysis using their provided file annplant_2spp_det_rare.txt:
# #### - compares Yenni's classification (S filter + equilibrium formula) to the correct mathematical conditions
# #### - quantifies misclassification sources and generates contingency tables
# #### Original code by Yenni et al.: https://github.com/gmyenni/RareStabilizationSimulation

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


def get_theoreticalretical_class(r1, r2, a11, a12, a21, a22, eps=1e-12):
    ratio1 = (r1 - 1) / (r2 - 1) if (r2 - 1) != 0 else float('inf')
    ratio2 = (r2 - 1) / (r1 - 1) if (r1 - 1) != 0 else float('inf')
    left1 = a12
    right1 = a22 * ratio1 if ratio1 != float('inf') else float('inf')
    left2 = a21
    right2 = a11 * ratio2 if ratio2 != float('inf') else float('inf')
    if abs(left1 - right1) < eps and abs(left2 - right2) < eps:
        name = 'Borderline'
        idx = 4
        A1 = A2 = B = C = False
    elif left1 < right1 - eps and left2 > right2 + eps:
        name = 'Exclusion N2 (A1)'
        idx = 0
        A1 = True
        A2 = B = C = False
    elif left1 > right1 + eps and left2 < right2 - eps:
        name = 'Exclusion N1 (A2)'
        idx = 1
        A2 = True
        A1 = B = C = False
    elif left1 < right1 - eps and left2 < right2 - eps:
        name = 'Coexistence (B)'
        idx = 2
        B = True
        A1 = A2 = C = False
    elif left1 > right1 + eps and left2 > right2 + eps:
        name = 'Saddle point (C)'
        idx = 3
        C = True
        A1 = A2 = B = False
    else:
        name = 'Borderline'
        idx = 4
        A1 = A2 = B = C = False
    return {'name': name, 'index': idx, 'A1': A1, 'A2': A2, 'B': B, 'C': C}


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


def calculate_metrics(r1, r2, a11, a12, a21, a22, N1, N2, extinc_crit_1=True):
    S1, S2 = SOS(r1, r2, a11, a12, a21, a22) # Strength of Stabilization
    FE1, FE2 = r1 / r2, r2 / r1 # Fitness equivalence
    Asy = S1 - S2 # Asymmetry
    Rare = 0 if N1 == 0 and N2 == 0 else N1 / (N1 + N2)
    x = np.array([N1, N2])
    y_sos = np.array([S1, S2])
    cor_matrix_sos = np.cov(x, y_sos)
    cor_sos = cor_matrix_sos[0, 1] # Extracting the correlation between N and SoS
    Rank = 0 if N1 == 0 and N2 == 0 else (2 if N1 / (N1 + N2) <= 0.25 else 1)
    cls = get_theoreticalretical_class(r1, r2, a11, a12, a21, a22) # Calculate Cushing conditions
    A1, A2, B, C = cls['A1'], cls['A2'], cls['B'], cls['C']
    PGR1, PGR2 = getPCG(r1, r2, a11, a12, a21, a22, N1, N2)
    if extinc_crit_1:
        Coexist = 0 if N1 < 1 or N2 < 1 else 1
    else:
        Coexist = 0 if N1 < 1.0e-6 or N2 < 1.0e-6 else 1
    return {"FE1": FE1, "S1": S1, "FE2": FE2, "S2": S2, "Rank": Rank, "Coexist": Coexist, "Asy": Asy, "cor_sos": cor_sos, "Rare": Rare, "PGR1": PGR1, "PGR2": PGR2, "A1": A1, "A2": A2, "B": B, "C": C}


# -

# # annualplant_2spp_det_par.r

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
    # Report logic
    if neg_confint[1] >= pos_confint[0] and neg_confint[0] <= pos_confint[1]:
        print(f"The confidence intervals overlap for {analysis_type}, indicating they are statistically the same, not supporting the authors' results.")
    elif neg_confint[1] > pos_confint[0]:
        print(f"Higher coexistence observed with \u03BD < 0 for {analysis_type}, supporting the authors' results.")
    else:
        print(f"Higher coexistence observed with \u03BD \u2265 0 for {analysis_type}, not supporting the authors' results.")
    return models_results


# -

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
    A1_list = []; A2_list = []; B_list = []; C_list = []
    for _, row in dat.iterrows():
        cls = get_theoreticalretical_class(row['r1'], row['r2'], row['a11'], row['a12'], row['a21'], row['a22'])
        A1_list.append(cls['A1']); A2_list.append(cls['A2']); B_list.append(cls['B']); C_list.append(cls['C'])
    dat['A1'] = A1_list; dat['A2'] = A2_list; dat['B'] = B_list; dat['C'] = C_list
    return dat


def print_classification_table(rows_data):
    df = pd.DataFrame(rows_data)
    print(df.to_string(index=False))


def report_classification_from_txt(txt_path, extinc_crit_1):
    dat = load_and_compute_classification(txt_path, extinc_crit_1)
    theoretical_class = []
    for _, row in dat.iterrows():
        if row['A1'] or row['A2']:
            theoretical_class.append('Exclusion (A)')
        elif row['B']:
            theoretical_class.append('Coexistence (B)')
        elif row['C']:
            theoretical_class.append('Saddle point (C)')
        else:
            theoretical_class.append('Borderline')
    yenni_class = []
    for _, row in dat.iterrows():
        if row['N1'] >= 1 and row['N2'] >= 1:
            yenni_class.append('Coexistence')
        else:
            yenni_class.append('Exclusion')
    df_conf = pd.crosstab(pd.Series(yenni_class, name='Yenni_txt'),
                          pd.Series(theoretical_class, name='Coexistence Condition'),
                          dropna=False)
    row_order = ['Coexistence', 'Exclusion']
    col_order = ['Coexistence (B)', 'Exclusion (A)', 'Saddle point (C)', 'Borderline']
    df_conf = df_conf.reindex(index=row_order, columns=col_order, fill_value=0)
    print("\nContingency table (Yenni txt classification vs true mathematical class):")
    print(df_conf.to_string())


def report_classification_from_df(dat, extinc_crit_1):
    if 'Coexist' not in dat.columns:
        if extinc_crit_1:
            dat['Coexist'] = ((dat['N1'] >= 1) & (dat['N2'] >= 1)).astype(int)
        else:
            dat['Coexist'] = ((dat['N1'] >= 1e-6) & (dat['N2'] >= 1e-6)).astype(int)
    theoretical_class = [] # Cushing conditions
    for _, row in dat.iterrows():
        if row['A1']:
            theoretical_class.append('Exclusion N2 (A1)')
        elif row['A2']:
            theoretical_class.append('Exclusion N1 (A2)')
        elif row['B']:
            theoretical_class.append('Coexistence (B)')
        elif row['C']:
            theoretical_class.append('Saddle point (C)')
        else:
            theoretical_class.append('Borderline')
    df_class = []
    for _, row in dat.iterrows():
        if row['N1'] >= 1 and row['N2'] >= 1:
            df_class.append('Coexistence (B)')
        elif row['N1'] >= 1 and row['N2'] < 1:
            df_class.append('Exclusion N2 (A1)')
        elif row['N1'] < 1 and row['N2'] >= 1:
            df_class.append('Exclusion N1 (A2)')
        else:  # both < 1
            if row['C']:
                df_class.append('Saddle point (C)')
            else:
                df_class.append('Borderline')
    df_conf = pd.crosstab(pd.Series(df_class, name='df'),
                          pd.Series(theoretical_class, name='Coexistence Condition'),
                          dropna=False)
    row_order = ['Exclusion N2 (A1)', 'Exclusion N1 (A2)', 'Coexistence (B)', 'Saddle point (C)', 'Borderline']
    col_order = ['Exclusion N2 (A1)', 'Exclusion N1 (A2)', 'Coexistence (B)', 'Saddle point (C)', 'Borderline']
    df_conf = df_conf.reindex(index=row_order, columns=col_order, fill_value=0)
    print("\nContingency table (DataFrame classification vs true mathematical class):")
    print(df_conf.to_string())


def generate_comprehensive_table(param_keys=None):
    if param_keys is None:
        param_keys = [("Narrow ranges (Yenni)", "yenni"), ("Broad ranges", "broad")]
    print("Comprehensive impact breakdown of the three differences:\n")
    for param_label, param_key in param_keys:
        mesh = preprocess_data(param_key)
        n_total = len(mesh)
        categories = ['Exclusion N2 (A1)', 'Exclusion N1 (A2)', 'Coexistence (B)', 'Saddle point (C)', 'Borderline']
        yenni_conf = np.zeros((5,5), dtype=int)
        our_conf = np.zeros((5,5), dtype=int)
        yenni_miscl_sfilter = 0
        yenni_miscl_formula = 0
        yenni_miscl_both = 0
        for row in tqdm(mesh, total=n_total, desc=f"Processing {param_label}"):
            r1, r2, a11, a12, a21, a22 = row
            cls = get_theoreticalretical_class(r1, r2, a11, a12, a21, a22)
            theoretical_idx = cls['index']
            Y_N1, Y_N2 = getEqDensity(r1, r2, a11, a12, a21, a22) # Yenni method
            S1, S2 = SOS(r1, r2, a11, a12, a21, a22)
            if S1 >= 1 and S2 >= 1:
                if Y_N1 >= 1 and Y_N2 >= 1:
                    yenni_idx = 2
                elif Y_N1 >= 1 and Y_N2 < 1:
                    yenni_idx = 0
                elif Y_N1 < 1 and Y_N2 >= 1:
                    yenni_idx = 1
                else:
                    if theoretical_idx == 3:
                        yenni_idx = 3
                    else:
                        yenni_idx = 4
            else:
                if Y_N1 >= 1 and Y_N2 >= 1:
                    yenni_idx = 2
                elif Y_N1 >= 1 and Y_N2 < 1:
                    yenni_idx = 0
                elif Y_N1 < 1 and Y_N2 >= 1:
                    yenni_idx = 1
                else:
                    if theoretical_idx == 3:
                        yenni_idx = 3
                    else:
                        yenni_idx = 4
            yenni_conf[yenni_idx, theoretical_idx] += 1
            if param_key == 'yenni':
                true_coexist_binary = 1 if theoretical_idx == 2 else 0
                if S1 >= 1 and S2 >= 1:
                    Y_coexist_binary = 1 if (Y_N1 >= 1 and Y_N2 >= 1) else 0
                else:
                    Y_coexist_binary = 0
                if Y_coexist_binary != true_coexist_binary:
                    Y_coexist_nofilter = 1 if (Y_N1 >= 1 and Y_N2 >= 1) else 0
                    if Y_coexist_nofilter == true_coexist_binary:
                        yenni_miscl_sfilter += 1
                    elif (S1 >= 1 and S2 >= 1) and (Y_coexist_binary == true_coexist_binary):
                        pass
                    else:
                        if S1 >= 1 and S2 >= 1:
                            O_coexist_sfilter = true_coexist_binary
                            if (O_coexist_sfilter == true_coexist_binary) and (Y_coexist_binary != true_coexist_binary):
                                yenni_miscl_formula += 1
                            else:
                                yenni_miscl_both += 1
                        else:
                            yenni_miscl_formula += 1
            if param_key == 'broad': # Our method
                coexist_thresh = 1e-6
            else:
                coexist_thresh = 1.0
            if theoretical_idx == 0:
                O_N1 = (r1-1)/a11; O_N2 = 0.0
            elif theoretical_idx == 1:
                O_N1 = 0.0; O_N2 = (r2-1)/a22
            elif theoretical_idx == 2:
                denom = a11*a22 - a12*a21
                O_N1 = ((r1-1)*a22 - (r2-1)*a12) / denom
                O_N2 = ((r2-1)*a11 - (r1-1)*a21) / denom
            else:
                O_N1 = 0.0; O_N2 = 0.0
            if O_N1 >= coexist_thresh and O_N2 >= coexist_thresh:
                our_idx = 2
            elif O_N1 >= coexist_thresh and O_N2 < coexist_thresh:
                our_idx = 0
            elif O_N1 < coexist_thresh and O_N2 >= coexist_thresh:
                our_idx = 1
            else:
                if theoretical_idx == 3:
                    our_idx = 3
                else:
                    our_idx = 4
            our_conf[our_idx, theoretical_idx] += 1
        theoretical_counts = yenni_conf.sum(axis=0)
        print(f"{'='*70}")
        print(f"Parameter set: {param_label} (n={n_total})")
        print(f"  theoretical classes: A1={theoretical_counts[0]}, A2={theoretical_counts[1]}, B={theoretical_counts[2]}, C={theoretical_counts[3]}, Border={theoretical_counts[4]}")
        print("\n  Yenni et al. method (S filter on, incorrect formula, coexistence defined as N>=1):")
        yenni_df = pd.DataFrame(yenni_conf, index=categories, columns=categories)
        print(yenni_df.to_string())
        print("\n  Our method (correct formula, no S filter, coexistence defined as N>1e-6):")
        our_df = pd.DataFrame(our_conf, index=categories, columns=categories)
        print(our_df.to_string())
        if param_key == 'yenni':
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


def count_legitimate_removed_by_sfilter():
    mesh = preprocess_data('yenni')
    results = []
    for row in mesh:
        r1, r2, a11, a12, a21, a22 = row
        cls = get_theoreticalretical_class(r1, r2, a11, a12, a21, a22)
        theoretical = cls['name']
        S1, S2 = SOS(r1, r2, a11, a12, a21, a22)
        passes_filter = (S1 >= 1 and S2 >= 1)
        results.append((theoretical, passes_filter))
    df = pd.DataFrame(results, columns=['theoretical', 'passes_S_filter'])
    excl_mask = df['theoretical'].isin(['Exclusion N2 (A1)', 'Exclusion N1 (A2)'])
    coex_mask = df['theoretical'] == 'Coexistence (B)'
    n_excl_total = excl_mask.sum()
    n_coex_total = coex_mask.sum()
    n_excl_passing = (excl_mask & df['passes_S_filter']).sum()
    n_coex_passing = (coex_mask & df['passes_S_filter']).sum()
    total_legitimate = n_excl_total + n_coex_total
    total_passing = n_excl_passing + n_coex_passing
    pct_excl_total = n_excl_total / total_legitimate * 100 if total_legitimate > 0 else 0.0
    pct_coex_total = n_coex_total / total_legitimate * 100 if total_legitimate > 0 else 0.0
    pct_excl_passing = n_excl_passing / total_passing * 100 if total_passing > 0 else 0.0
    pct_coex_passing = n_coex_passing / total_passing * 100 if total_passing > 0 else 0.0
    percent_coex_total = n_coex_total / total_legitimate * 100 if total_legitimate > 0 else 0.0
    percent_coex_passing = n_coex_passing / total_passing * 100 if total_passing > 0 else 0.0
    print("\nLegitimate parameter sets (A1, A2, B only) removed by Yenni's S>=1 filter:")
    print(f"{'':<30} {'Yenni (S>=1 filter)':<30} {'Without filter (all legitimate)':<30}")
    print(f"{'Coexistence (B)':<30} {n_coex_passing} ({pct_coex_passing:.2g}%){'':<15} {n_coex_total} ({pct_coex_total:.3g}%)")
    print(f"{'Exclusion (A1+A2)':<30} {n_excl_passing} ({pct_excl_passing:.2g}%){'':<15} {n_excl_total} ({pct_excl_total:.3g}%)")
    print(f"{'% cases of coexistence':<30} {percent_coex_passing:.3g}%{'':<26} {percent_coex_total:.3g}%")


def count_coexistence_stats():
    col_names = ['ID','l1','l2','a11','a12','a21','a22','N1','N2','E1','S1','E2','S2','Rank','CoexistRank','Asy','cor','Rare']
    yenni_df = pd.read_csv("csv/annplant_2spp_det_rare.txt", header=None, names=col_names)
    for c in ['N1','N2']:
        yenni_df[c] = pd.to_numeric(yenni_df[c], errors='coerce')
    yenni_coex = yenni_df[(yenni_df['N1'] >= 1) & (yenni_df['N2'] >= 1)].copy()
    yenni_rare = yenni_coex['N1']
    yenni_common = yenni_coex['N2']
    broad_df = pd.read_csv("csv/annplant_2spp_det_rare_broad.csv")
    if 'Coexist' in broad_df.columns:
        broad_coex = broad_df[broad_df['Coexist'] == 1].copy()
    else:
        for c in ['N1','N2']:
            broad_df[c] = pd.to_numeric(broad_df[c], errors='coerce')
        broad_coex = broad_df[(broad_df['N1'] > 1e-6) & (broad_df['N2'] > 1e-6)].copy()
    if 'Rank' in broad_coex.columns:
        broad_rare = broad_coex.apply(lambda row: row['N1'] if row['Rank'] == 2 else row['N2'], axis=1)
        broad_common = broad_coex.apply(lambda row: row['N2'] if row['Rank'] == 2 else row['N1'], axis=1)
    else:
        broad_rare = broad_coex['N1']
        broad_common = broad_coex['N2']
    print("\nCoexistence cases (Yenni: N>=1; Broad: N>1e-6):")
    print(f"Yenni et al. (2012): n={len(yenni_coex)}")
    if len(yenni_coex) > 0:
        print(f"  N_rare mean={yenni_rare.mean():.2f}, median={yenni_rare.median():.2f}")
        print(f"  N_common mean={yenni_common.mean():.2f}, median={yenni_common.median():.2f}")
    else:
        print("  No coexistence cases found.")
    print(f"\nBroad ranges (our study): n={len(broad_coex)}")
    if len(broad_coex) > 0:
        print(f"  N_rare mean={broad_rare.mean():.2f}, median={broad_rare.median():.2f}")
        print(f"  N_common mean={broad_common.mean():.2f}, median={broad_common.median():.2f}")
    else:
        print("  No coexistence cases found.")


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
    A1_list = []; A2_list = []; B_list = []; C_list = []
    for _, row in dat.iterrows():
        cls = get_theoreticalretical_class(row['r1'], row['r2'], row['a11'], row['a12'], row['a21'], row['a22'])
        A1_list.append(cls['A1']); A2_list.append(cls['A2']); B_list.append(cls['B']); C_list.append(cls['C'])
    dat['A1'] = A1_list; dat['A2'] = A2_list; dat['B'] = B_list; dat['C'] = C_list
    report_classification_from_df(dat, extinc_crit_1)
    generate_filtered_analysis(case)


def main():
    include_broad = True
    report_classification_from_txt("csv/annplant_2spp_det_rare.txt", True)
    count_legitimate_removed_by_sfilter()
    if include_broad:
        generate_comprehensive_table()
        run_pipeline('yenni')
        run_pipeline('broad')
        count_coexistence_stats()
    else:
        generate_comprehensive_table(param_keys=[("Narrow ranges (Yenni)", "yenni")])
        run_pipeline('yenni')


if __name__ == "__main__":
    main()
