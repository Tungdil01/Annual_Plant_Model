# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# ### The code aims to replicate Table 1 of Yenni et al. (2012):
# #### - keeps the parameters' variations from the code
# #### - filters S1 >= 1 & S2 >= 1, without it I cannot reproduce
# #### - keeps the truncated the values
#
# #### their original code: https://github.com/gmyenni/RareStabilizationSimulation

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.proportion import proportion_confint
from scipy import stats
from scipy.stats import ttest_ind
from datetime import datetime
import warnings


# # analyN_function.r

def analyN(r1, r2, a1, a12, a21, a2):
    N1 = (r1 - 1 - (a12 / a2) * (r2 - 1)) / (a1 - a21 * a12 / a2)
    N2 = (r2 - 1 - (a21 / a1) * (r1 - 1)) / (a2 - a21 * a12 / a1)
    if np.isinf(N1) or np.isinf(N2) or np.isnan(N1) or np.isnan(N2):
        initialNsp1 = 0
        initialNsp2 = 0
        N = np.zeros((100, 2))
        N[0, :] = [initialNsp1, initialNsp2]
        for i in range(1, 100):
            N[i, 0] = max((r1 - 1 - a12 * N[i-1, 1]) / a1, 0)
            N[i, 1] = max((r2 - 1 - a21 * N[i-1, 0]) / a2, 0)
        N1 = np.mean(N[:, 0])
        N2 = np.mean(N[:, 1])
    if N1 < 0 and N2 >= 0:
        N1 = 0
        N2 = (r2 - 1) / a2
    elif N2 < 0 and N1 >= 0:
        N2 = 0
        N1 = (r1 - 1) / a1
    return N1, N2


# # getNFD.r

def calculate_metrics(l1, l2, a11, a12, a21, a22, N1, N2):
    CoexistRank = 0 if N1 < 1 else 1
#     The original code of Yenni et al. replaced l1 with l2 in the numerator:
    S1 = l2 / (1 + (a12 / a22) * (l2 - 1))
    S2 = l1 / (1 + (a21 / a11) * (l1 - 1))
    E1, E2 = l1 / l2, l2 / l1  # Fitness equivalence
    Asy = S1 - S2  # Asymmetry
    Rare = 0 if N1 == 0 and N2 == 0 else N1 / (N1 + N2)
    # Calculating covariance:
    x = np.array([N1, N2])
    y = np.array([S1, S2])
    cor_matrix = np.cov(x, y)
    cor = cor_matrix[0, 1]  # Extracting the covariance between N and S
    Rank = 0 if N1 == 0 and N2 == 0 else (2 if N1 / (N1 + N2) <= 0.25 else 1)
    return CoexistRank, E1, S1, E2, S2, Asy, cor, Rare, Rank


# # annualplant_2spp_det_par.r

# +
def preprocess_data():
    # Defines frequency-dependent parameters
    l1_v = np.arange(15, 21)
    l2_v = np.arange(15, 21)
    a11_v = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1, 1.5, 2, 2.5, 3])
    a12_v = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1])
    a21_v = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1])
    a22_v = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1])
    # Generate all combinations of parameters using NumPy's meshgrid
    mesh = np.array(np.meshgrid(l1_v, l2_v, a11_v, a12_v, a21_v, a22_v)).T.reshape(-1, 6)
    return mesh

def Sim(k, mesh_row):
    l1, l2, a11, a12, a21, a22 = mesh_row
    N1, N2 = analyN(l1, l2, a11, a12, a21, a22)
    CoexistRank, E1, E2, S1, S2, Asy, cov, Rare, Rank = calculate_metrics(l1, l2, a11, a12, a21, a22, N1, N2)
    return np.array([l1, l2, a11, a12, a21, a22, N1, N2, E1, E2, S1, S2, Rank, CoexistRank, Asy, cov, Rare])

def postprocess_results(results, outfile):
    column_order = ['l1', 'l2', 'a11', 'a12', 'a21', 'a22', 'N1', 'N2', 'E1', 'S1', 'E2', 'S2', 'Rank', 'CoexistRank', 'Asy', 'cor', 'Rare']
    simul = pd.DataFrame(results, columns=column_order)
    simul.to_csv(outfile, index=False)

if __name__ == "__main__":
    outfile = "csv/annplant_2spp_det_rare.csv"
    mesh = preprocess_data()
    results = np.empty((len(mesh), 17), dtype=float)
    # Run the simulation for each row in the parameter combination mesh
    for k in range(len(mesh)):
        results[k] = Sim(k, mesh[k])
    postprocess_results(results, outfile)


# -

# # cor_figure.r

def cor_figure():
    dat_det = pd.read_csv("csv/annplant_2spp_det_rare.csv")
    dat_det = dat_det.query('Rank == 2 & S1 >= 1 & S2 >= 1').copy() # Apply filter
    dat_det.reset_index(drop=True, inplace=True)
    dat_det = np.trunc(dat_det * 100) / 100.0  # Truncate to two decimals
    dat_det.sort_values(by=['a22', 'a21', 'a12', 'a11', 'l2', 'l1'], inplace=True)
    dat_det.to_csv("csv/annplant_2spp_det_rare_filtered.csv", index=False)


# # figures_det.r

def analyze_coexistence_effect(file_path):
    dat = pd.read_csv(file_path)
    # Logistic regression
    X = sm.add_constant(dat[['S1', 'E1', 'cor']])
    y = dat['CoexistRank']
    model = sm.GLM(y, X, family=sm.families.Binomial())
    result = model.fit()
    print(f"{result.summary()}")
    # Calculation of proportions and table preparation
    nu_positive_coexistence = len(dat[(dat['cor'] >= 0) & (dat['CoexistRank'] == 1)])
    nu_positive_exclusion = len(dat[(dat['cor'] >= 0) & (dat['CoexistRank'] == 0)])
    nu_negative_coexistence = len(dat[(dat['cor'] < 0) & (dat['CoexistRank'] == 1)])
    nu_negative_exclusion = len(dat[(dat['cor'] < 0) & (dat['CoexistRank'] == 0)])
    # Display the table
    table_data = {
        '\u03BD \u2265 0': [nu_positive_coexistence, nu_positive_exclusion],
        '\u03BD < 0': [nu_negative_coexistence, nu_negative_exclusion]
    }
    table_df = pd.DataFrame(table_data, index=['coexistence', 'exclusion'])
    print("\nCoexistence and Exclusion based on \u03BD:\n", table_df)
    negative_nu = dat[dat['cor'] < 0]
    non_negative_nu = dat[dat['cor'] >= 0]
    negative_nu_coexist = negative_nu[negative_nu['CoexistRank'] == 1]
    non_negative_nu_coexist = non_negative_nu[non_negative_nu['CoexistRank'] == 1]
    proportion_negative_nu = len(negative_nu_coexist) / len(negative_nu) if len(negative_nu) > 0 else 0
    proportion_non_negative_nu = len(non_negative_nu_coexist) / len(non_negative_nu) if len(non_negative_nu) > 0 else 0
    # Confidence intervals for proportions
    neg_nu_confint = proportion_confint(count=len(negative_nu_coexist), nobs=len(negative_nu), alpha=0.05, method='wilson')
    non_neg_nu_confint = proportion_confint(count=len(non_negative_nu_coexist), nobs=len(non_negative_nu), alpha=0.05, method='wilson')
    print("\nAnalysis on Negative \u03BD:")
    print(f"Proportion of coexistence with \u03BD < 0: {proportion_negative_nu:.4f} (95% CI: {neg_nu_confint})")
    print(f"Proportion of coexistence with \u03BD \u2265 0: {proportion_non_negative_nu:.4f} (95% CI: {non_neg_nu_confint})")
    # Comparing confidence intervals for decision making
    if neg_nu_confint[1] < non_neg_nu_confint[0]:
        print("Higher coexistence observed with \u03BD \u2265 0, not supporting the authors' claim that \n'coexistence is predicted more often when \u03BD is negative'.")
    elif neg_nu_confint[0] > non_neg_nu_confint[1]:
        print("Higher coexistence observed with \u03BD < 0, supporting the authors' claim that \n'coexistence is predicted more often when \u03BD is negative'.")
    else:
        print("Confidence intervals for proportions overlap, suggesting the effect of nu on coexistence is inconclusive, relative to the authors' claim that \n'coexistence is predicted more often when \u03BD is negative'.")


# +
def main():
    # Suppress warnings
    warnings.filterwarnings("ignore")
    # Generate simulation results
    output_file = "csv/annplant_2spp_det_rare_filtered.csv"
    data_grid = preprocess_data()
    simulations = np.array([Sim(k, row) for k, row in enumerate(data_grid)])
    postprocess_results(simulations, output_file)
    cor_figure() # apply filters
    print("Original Results by Yenni et al.")
    analyze_coexistence_effect("csv/annplant_2spp_det_rare.txt")
    print("\nReproduction of the Authors' Results")
    analyze_coexistence_effect(output_file)

if __name__ == "__main__":
    main()
