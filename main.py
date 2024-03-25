# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# ### The code aims to modify the analysis of Yenni et al. (2012):
# #### - corrected the SoS calculation
# #### - modified the parameters to paper's description: "r2 integers from 11 to 20"
# #### - removed the additional filter S1 >= 1 & S2 >= 1
# #### - did not truncate the values
# #### - included additional metrics: CA and CE
#
# #### their original code: https://github.com/gmyenni/RareStabilizationSimulation

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.proportion import proportion_confint
from scipy import stats
from scipy.stats import ttest_ind
from scipy.special import expit
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
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

# +
def calculate_metrics(l1, l2, a11, a12, a21, a22, N1, N2):
    CoexistRank = 0 if N1 < 1 else 1

#     The original code of Yenni et al. replaced l1 with l2 in the numerator:
#     S1 = l2 / (1 + (a12 / a22) * (l2 - 1))
#     S2 = l1 / (1 + (a21 / a11) * (l1 - 1))
#     # Corrected Strength of Stabilization:
    S1 = l1 / (1 + (a12 / a22) * (l2 - 1))
    S2 = l2 / (1 + (a21 / a11) * (l1 - 1))

    # Fitness equivalence
    E1, E2 = l1 / l2, l2 / l1

    # Asymmetry between the species
    Asy = S1 - S2

    # Calculation for the rare species
    Rare = 0 if N1 == 0 and N2 == 0 else N1 / (N1 + N2)

    # Competitive Ability and Efficiency calculations
    CA1, CA2 = competitive_ability(l1, l2, a11, a22, a12, a21)
    CE1, CE2, CE_status = competitive_efficiency(l1, l2, a11, a22, a12, a21)

    # Array for abundance
    x = np.array([N1, N2])
    
    # Covariances calculations
    y_sos = np.array([S1, S2])
    cor_sos = np.cov(x, y_sos)[0, 1]  # Covariance for SoS
    
    y_ca = np.array([CA1, CA2])
    cor_ca = np.cov(x, y_ca)[0, 1]  # Covariance for CA

    y_ce = np.array([CE1, CE2])
    cor_ce = np.cov(x, y_ce)[0, 1]  # Covariance for CE

    CE_status_map = {'global_competitive_exclusion': 1, 'local_coexistence': 2, 'global_coexistence': 3, 'local_competitive_exclusion': 4}
    CE_status_num = CE_status_map[CE_status]
    
    Rank = 0 if N1 == 0 and N2 == 0 else (2 if N1 / (N1 + N2) <= 0.25 else 1)
    
    metrics = [CoexistRank, E1, S1, E2, S2, Asy, cor_sos, Rare, Rank, CA1, CA2, CE1, CE2, CE_status_num, cor_ca, cor_ce]
    return np.array(metrics)


# -

# # Additional competitive metrics:
# ### - Competitive Ability CA Hart et al. (2018)
# ### - Competitive Efficiency CE Streipert and Wolkowicz (2023)

# +
def competitive_ability(r1, r2, a11, a22, a12, a21):
    CA1 = (r1 - 1) / np.sqrt(a12 * a11)
    CA2 = (r2 - 1) / np.sqrt(a21 * a22)
    return CA1, CA2

def competitive_efficiency(r1, r2, a11, a22, a12, a21):
    tolerance = 1e-9
    CE1 = ((r1 - 1) / a12) - ((r2 - 1) / a22)
    CE2 = ((r2 - 1) / a21) - ((r1 - 1) / a11)
    if abs(CE1) <= tolerance and abs(CE2) <= tolerance:
        return CE1, CE2, 'local_coexistence'
    elif CE1 * CE2 < 0 or (abs(CE1) <= tolerance and abs(CE2) >= tolerance) or (abs(CE2) <= tolerance and abs(CE1) >= tolerance):
        return CE1, CE2, 'global_competitive_exclusion'
    elif CE1 < 0 and CE2 < 0:
        return CE1, CE2, 'local_competitive_exclusion'
    elif CE1 > 0 and CE2 > 0:
        return CE1, CE2, 'global_coexistence'


# -

# # annualplant_2spp_det_par.r

# +
def preprocess_data():
    # Defines frequency-dependent parameters
#     l1_v = np.arange(15, 21)
#     l2_v = np.arange(15, 21)
    a11_v = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1, 1.5, 2, 2.5, 3])
    a12_v = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1])
    a21_v = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1])
    a22_v = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1])
    l1_v = np.arange(15, 21)
    l2_v = np.arange(11, 21)
#     a11_v = np.arange(0.7, 3,0.1)
#     a12_v = np.arange(0.1, 1,0.1)
#     a21_v = np.arange(0.1, 1,0.1)
#     a22_v = np.arange(0.1, 1,0.1)

    # Generate all combinations of parameters using NumPy's meshgrid
    mesh = np.array(np.meshgrid(l1_v, l2_v, a11_v, a12_v, a21_v, a22_v)).T.reshape(-1, 6)
    return mesh

def Sim(k, mesh_row):
    l1, l2, a11, a12, a21, a22 = mesh_row
    # Simulate and calculate additional metrics based on the parameters
    N1, N2 = analyN(l1, l2, a11, a12, a21, a22)
    metrics = calculate_metrics(l1, l2, a11, a12, a21, a22, N1, N2)
    CoexistRank, E1, S1, E2, S2, Asy, cor_sos, Rare, Rank, CA1, CA2, CE1, CE2, CE_status_num, cor_ca, cor_ce = metrics    
    return np.array([l1, l2, a11, a12, a21, a22, N1, N2, E1, E2, S1, S2, Rank, CoexistRank, Asy, cor_sos, Rare, CA1, CA2, CE1, CE2, CE_status_num, cor_ca, cor_ce])

def postprocess_results(results, outfile):
    column_order = ['l1', 'l2', 'a11', 'a12', 'a21', 'a22', 'N1', 'N2', 'E1', 'S1', 'E2', 'S2', 'Rank', 'CoexistRank', 'Asy', 'cor_sos', 'Rare', 'CA1', 'CA2', 'CE1', 'CE2', 'CE_status', 'cor_ca', 'cor_ce']
    simul = pd.DataFrame(results, columns=column_order)
    simul.to_csv(outfile, index=False)

if __name__ == "__main__":
    print(datetime.now())
    outfile = "csv/annplant_2spp_det_rare.csv"
    mesh = preprocess_data()
    column_order = ['l1', 'l2', 'a11', 'a12', 'a21', 'a22', 'N1', 'N2', 'S1', 'S2', 'E1', 'E2', 'CA1', 'CA2', 'CE1', 'CE2', 'CE_status', 'Rank', 'CoexistRank', 'Asy', 'Rare', 'cor_sos', 'cor_ca', 'cor_ce']
    results = np.empty((len(mesh), len(column_order)), dtype=float)
    # Run the simulation for each row in the parameter combination mesh
    for k in range(len(mesh)):
        results[k] = Sim(k, mesh[k])
    postprocess_results(results, outfile)


# -

# # cor_figure.r

def cor_figure():
    dat_det = pd.read_csv("csv/annplant_2spp_det_rare.csv")
    dat_det = dat_det.query('Rank == 2').copy() # Apply filter  & S1 >= 1 & S2 >= 1
    dat_det.reset_index(drop=True, inplace=True)
#     dat_det = np.trunc(dat_det * 100) / 100.0  # Truncate to two decimals
    dat_det.sort_values(by=['a22', 'a21', 'a12', 'a11', 'l2', 'l1'], inplace=True)
    dat_det.to_csv("csv/annplant_2spp_det_rare_filtered.csv", index=False)


# # Model Selection

# +
def fit_and_summarize_model(X, y):
    model = sm.GLM(y, X, family=sm.families.Binomial())
    result = model.fit()
    n = len(y)  # Sample size
    k = X.shape[1] - 1  # Number of parameters, excluding the constant
    aic = result.aic
    bic = result.bic
    aicc = aic + (2 * k * (k + 1)) / (n - k - 1)  # Corrected AIC for small sample sizes
    return aic, aicc, bic, result

def compute_weights(criterion_values):
    min_value = np.min(criterion_values)
    delta_values = criterion_values - min_value
    relative_likelihoods = np.exp(-0.5 * delta_values)
    sum_likelihoods = np.sum(relative_likelihoods)
    weights = relative_likelihoods / sum_likelihoods
    return weights

def apply_pca_to_pairs(dat, pairs):
    scaler = StandardScaler()  # Standardizing the features
    for pair in pairs:
        sub_X = dat[list(pair)].copy()
        sub_X_scaled = scaler.fit_transform(sub_X)  # Standardize the data before applying PCA
        pca = PCA(n_components=1)
        X_pca = pca.fit_transform(sub_X_scaled)
        explained_variance = pca.explained_variance_ratio_[0]  # Get the explained variance
        print(f"PCA applied for {pair}.")
        print(f"Explained Variance: {explained_variance}")  # Print the explained variance
        pca_col_name = f"{pair[0]}_{pair[1]}_PCA"
        dat[pca_col_name] = X_pca
        dat.drop(list(pair), axis=1, inplace=True)
    return dat


# -

# # figures_det.r

# +
def analyze_coexistence_effect(file_path):
    dat = pd.read_csv(file_path)
    
    # Applying PCA to specified pairs
    pairs_to_check = [('S1', 'S2'), ('E1', 'E2'), ('CA1', 'CA2'), ('CE1', 'CE2')]
    dat = apply_pca_to_pairs(dat, pairs_to_check)

    # Prepare data for each model
    X_sos = sm.add_constant(dat[['S1_S2_PCA', 'E1_E2_PCA', 'cor_sos']])
    X_ca = sm.add_constant(dat[['CA1_CA2_PCA', 'cor_ca']])
    X_ce = sm.add_constant(dat[['CE1_CE2_PCA', 'cor_ce']])

    models = {'SoS': X_sos, 'CA': X_ca, 'CE': X_ce}
    criterion_values = {'AIC': [], 'AICc': [], 'BIC': []}
    fitted_models = {}

    # Fit models and calculate criterion values
    for name, X in models.items():
        model = sm.GLM(dat['CoexistRank'], X, family=sm.families.Binomial())
        fitted_model = model.fit()
        n = len(dat['CoexistRank'])  # Sample size
        k = X.shape[1] - 1  # Number of parameters, excluding the constant
        aic = fitted_model.aic
        bic = fitted_model.bic
        aicc = aic + (2 * k * (k + 1)) / (n - k - 1)  # Corrected AIC for small sample sizes
        criterion_values['AIC'].append(aic)
        criterion_values['AICc'].append(aicc)
        criterion_values['BIC'].append(bic)
        fitted_models[name] = fitted_model

    # Display raw AIC, AICc, and BIC values in a table
    print("\nCriterion Values:")
    criterion_df = pd.DataFrame(criterion_values, index=models.keys())
    print(criterion_df)

    # Compute weights for AIC, AICc, and BIC
    aic_weights = compute_weights(np.array(criterion_values['AIC']))
    aicc_weights = compute_weights(np.array(criterion_values['AICc']))
    bic_weights = compute_weights(np.array(criterion_values['BIC']))

    # Display the weights in a separate table
    print("\nModel Weights:")
    weights_df = pd.DataFrame({
        'AIC Weight': aic_weights,
        'AICc Weight': aicc_weights,
        'BIC Weight': bic_weights
    }, index=models.keys())
    print(weights_df)

    for name, model in fitted_models.items():
        print(f"{name} Model Summary:\n{model.summary()}\n")

def perform_proportion_analysis(dat, cor_column, metric_name):
    negative_nu = dat[dat[cor_column] < 0]
    non_negative_nu = dat[dat[cor_column] >= 0]
    negative_nu_coexist = negative_nu[negative_nu['CoexistRank'] == 1]
    non_negative_nu_coexist = non_negative_nu[non_negative_nu['CoexistRank'] == 1]

    # Counting coexistence and exclusion
    nu_positive_coexistence = len(non_negative_nu_coexist)
    nu_positive_exclusion = len(non_negative_nu) - nu_positive_coexistence
    nu_negative_coexistence = len(negative_nu_coexist)
    nu_negative_exclusion = len(negative_nu) - nu_negative_coexistence

    # Display the table
    table_data = {
        f'\u03BD_{metric_name} \u2265 0': [nu_positive_coexistence, nu_positive_exclusion],
        f'\u03BD_{metric_name} < 0': [nu_negative_coexistence, nu_negative_exclusion]
    }
    table_df = pd.DataFrame(table_data, index=['coexistence', 'exclusion'])
    print("\nCoexistence and Exclusion based on \u03BD:\n", table_df)

    proportion_negative_nu = len(negative_nu_coexist) / len(negative_nu) if len(negative_nu) > 0 else 0
    proportion_non_negative_nu = len(non_negative_nu_coexist) / len(non_negative_nu) if len(non_negative_nu) > 0 else 0

    neg_nu_confint = proportion_confint(count=len(negative_nu_coexist), nobs=len(negative_nu), alpha=0.05, method='wilson')
    non_neg_nu_confint = proportion_confint(count=len(non_negative_nu_coexist), nobs=len(non_negative_nu), alpha=0.05, method='wilson')

    print(f"\nAnalysis on Negative \u03BD for {metric_name}:")
    print(f"Proportion of coexistence with \u03BD_{metric_name} < 0: {proportion_negative_nu:.4f} (95% CI: {neg_nu_confint})")
    print(f"Proportion of coexistence with \u03BD_{metric_name} \u2265 0: {proportion_non_negative_nu:.4f} (95% CI: {non_neg_nu_confint})")

    if neg_nu_confint[1] < non_neg_nu_confint[0]:
        print(f"Higher coexistence observed with \u03BD \u2265 0 for {metric_name}, not supporting the authors' claim that 'coexistence is predicted more often when \u03BD is negative'.")
    elif neg_nu_confint[0] > non_neg_nu_confint[1]:
        print(f"Higher coexistence observed with \u03BD < 0 for {metric_name}, supporting the authors' claim that 'coexistence is predicted more often when \u03BD is negative'.")
    else:
        print(f"Confidence intervals for proportions overlap for {metric_name}, suggesting the effect of \u03BD on coexistence is inconclusive.")


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

    dat = pd.read_csv(output_file)  # Load the simulation results
    analyze_coexistence_effect(output_file)

    perform_proportion_analysis(dat, 'cor_sos', "SoS")
    perform_proportion_analysis(dat, 'cor_ca', "CA")
    perform_proportion_analysis(dat, 'cor_ce', "CE")

if __name__ == "__main__":
    main()
