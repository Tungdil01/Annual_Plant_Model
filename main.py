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

def Sim(k, mesh_row, extinc_crit_1=False):
    start_time = time.time()
    r1, r2, a11, a12, a21, a22 = mesh_row
    N1, N2 = getEqDensity(r1, r2, a11, a12, a21, a22)
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
    print(f"Proportion of coexistence with \u03BD \u2265 0: {proportions[positive_key] / pos_total:.2g} (95% CI: ({pos_confint[0]:.2g}, {pos_confint[1]:.2g}))")
    print(f"Proportion of coexistence with \u03BD < 0: {proportions[negative_key] / neg_total:.2g} (95% CI: ({neg_confint[0]:.2g}, {neg_confint[1]:.2g}))")

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


def setup_pipeline(filters, base_file, truncate, extinc_crit_1):
    os.makedirs('csv', exist_ok=True)
    warnings.filterwarnings("ignore")
    print("Running simulation...")
    mesh = preprocess_data('table1')
    results = [Sim(k, row, extinc_crit_1=extinc_crit_1)
                for k, row in tqdm(enumerate(mesh), total=len(mesh))]
    postprocess_results(results, base_file)
    for filter_option in filters:
        filtered_filename = f"csv/annplant_2spp_det_rare_filtered_{filter_option}.csv"
        print(f"\nGenerating data for filter={filter_option}...")
        cor_figure(filter_option, truncate)
        summary_path = f"csv/pgr_analysis_summary_{filter_option}.csv"
        filtered_data = pd.read_csv(filtered_filename)
        analyze_coexistence_effect(filtered_data)
        plot_phase_plane()


def main():
    filters = ['on', 'off']
    base_file = "csv/annplant_2spp_det_rare.csv"
    # truncate, extinc_crit_1 = True, True # Yenni et al. (2012) truncate the values and consider extinction N<1
    truncate, extinc_crit_1 = False, True # the numerical results are slightly different if not truncated
    # truncate, extinc_crit_1 = False, False # the numerical results are slightly different if not truncated and extinction N<0
    setup_pipeline(filters, base_file, truncate, extinc_crit_1)


if __name__ == "__main__":
    main()
