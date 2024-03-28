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
# #### - included one additional metric: CA
# #### - employed PCA to the variables
#
# #### their original code: https://github.com/gmyenni/RareStabilizationSimulation

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.proportion import proportion_confint
from scipy import stats
from scipy.stats import ttest_ind
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
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


def competitive_ability(r1, r2, a11, a22, a12, a21):
    CA1 = (r1 - 1) / np.sqrt(a12 * a11)
    CA2 = (r2 - 1) / np.sqrt(a21 * a22)
    return CA1, CA2


# # getNFD.r

def calculate_metrics(l1, l2, a11, a12, a21, a22, N1, N2):
    CoexistRank = 0 if N1 < 1 else 1
#     The code of Yenni et al. replaced l1 with l2 in the numerator:
#     S1 = l2 / (1 + (a12 / a22) * (l2 - 1))
#     S2 = l1 / (1 + (a21 / a11) * (l1 - 1))
#     # Corrected Strength of Stabilization:
    S1 = l1 / (1 + (a12 / a22) * (l2 - 1))
    S2 = l2 / (1 + (a21 / a11) * (l1 - 1))
    E1, E2 = l1 / l2, l2 / l1  # Fitness equivalence
    Asy = S1 - S2  # Asymmetry
    Rare = 0 if N1 == 0 and N2 == 0 else N1 / (N1 + N2)
    # Calculating covariance for SoS
    x = np.array([N1, N2])
    y_sos = np.array([S1, S2])
    cor_matrix_sos = np.cov(x, y_sos)
    cor_sos = cor_matrix_sos[0, 1]  # Extracting the correlation between N and SoS
    Rank = 0 if N1 == 0 and N2 == 0 else (2 if N1 / (N1 + N2) <= 0.25 else 1)
    # Competitive Ability
    CA1, CA2 = competitive_ability(l1, l2, a11, a22, a12, a21)
    # Calculating covariance for CA
    y_ca = np.array([CA1, CA2])
    cor_matrix_ca = np.cov(x, y_ca)
    cor_ca = cor_matrix_ca[0, 1]  # Extracting the correlation between N and CA
    return {"CoexistRank": CoexistRank, "E1": E1, "S1": S1, "E2": E2, "S2": S2, "Asy": Asy, "cor_sos": cor_sos, "cor_ca": cor_ca, "Rare": Rare, "Rank": Rank, "CA1": CA1, "CA2": CA2}


# # annualplant_2spp_det_par.r

# +
def preprocess_data():
    # Defines frequency-dependent parameters
    l1_v = np.arange(15, 21)
    l2_v = np.arange(11, 21)
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
    metrics = calculate_metrics(l1, l2, a11, a12, a21, a22, N1, N2)
    return {**metrics, "l1": l1, "l2": l2, "a11": a11, "a12": a12, "a21": a21, "a22": a22}

def postprocess_results(results, outfile):
    column_order = ['l1', 'l2', 'a11', 'a12', 'a21', 'a22', 'N1', 'N2', 'E1', 'S1', 'E2', 'S2', 'Rank', 'CoexistRank', 'Asy', 'cor_sos', 'cor_ca', 'Rare', 'CA1', 'CA2']
    simul = pd.DataFrame(results, columns=column_order)
    simul.to_csv(outfile, index=False)


# -

# # cor_figure.r

def cor_figure():
    dat_det = pd.read_csv("csv/annplant_2spp_det_rare.csv")
    dat_det = dat_det.query('Rank == 2').copy() #  & S1 >= 1 & S2 >= 1
    dat_det.reset_index(drop=True, inplace=True)
#     dat_det = np.trunc(dat_det * 100) / 100.0
    dat_det.sort_values(by=['a22', 'a21', 'a12', 'a11', 'l2', 'l1'], inplace=True)
    dat_det.to_csv("csv/annplant_2spp_det_rare_filtered.csv", index=False)


def apply_pca(simul):
    # Apply PCA to SoS
    features_sos = ['S1', 'S2']
    x_sos = StandardScaler().fit_transform(simul[features_sos].values)
    pca_sos = PCA(n_components=1)
    principalComponent_sos = pca_sos.fit_transform(x_sos)
    simul['SoS_PCA'] = principalComponent_sos[:, 0]
    variance_sos = pca_sos.explained_variance_ratio_.sum() * 100
    print(f"PCA captured {variance_sos:.2f}% of the variance for the SoS pair.")
    # Apply PCA to FE
    features_fe = ['E1', 'E2']
    x_fe = StandardScaler().fit_transform(simul[features_fe].values)
    pca_fe = PCA(n_components=1)
    principalComponent_fe = pca_fe.fit_transform(x_fe)
    simul['FE_PCA'] = principalComponent_fe[:, 0]
    variance_fe = pca_fe.explained_variance_ratio_.sum() * 100
    print(f"PCA captured {variance_fe:.2f}% of the variance for the FE pair.")
    # Apply PCA to CA
    features_ca = ['CA1', 'CA2']
    x_ca = StandardScaler().fit_transform(simul[features_ca].values)
    pca_ca = PCA(n_components=1)
    principalComponent_ca = pca_ca.fit_transform(x_ca)
    simul['CA_PCA'] = principalComponent_ca[:, 0]
    variance_ca = pca_ca.explained_variance_ratio_.sum() * 100
    print(f"PCA captured {variance_ca:.2f}% of the variance for the CA pair.")
    return simul


# # figures_det.r

# +
def perform_logistic_regression(dat, analysis_type):
    predictors_map = {
        'PCA_SoS': ['SoS_PCA', 'cor_sos'],
        'PCA_CA': ['CA_PCA', 'cor_ca'],
        'NonPCA_SoS': ['S1', 'E1', 'cor_sos'],
        'NonPCA_CA': ['CA1', 'cor_ca']
    }
    predictors = predictors_map[analysis_type]
    X = sm.add_constant(dat[predictors])
    y = dat['CoexistRank']
    model = sm.GLM(y, X, family=sm.families.Binomial())
    result = model.fit()
    print(f"\n{analysis_type} Analysis:\n{result.summary()}")
    return result

def calculate_proportions(dat, correlation_type):
    proportions = {}
    for cor_type in [correlation_type]:
        proportions[f'positive_coexistence_{cor_type}'] = len(dat[(dat[cor_type] >= 0) & (dat['CoexistRank'] == 1)])
        proportions[f'positive_exclusion_{cor_type}'] = len(dat[(dat[cor_type] >= 0) & (dat['CoexistRank'] == 0)])
        proportions[f'negative_coexistence_{cor_type}'] = len(dat[(dat[cor_type] < 0) & (dat['CoexistRank'] == 1)])
        proportions[f'negative_exclusion_{cor_type}'] = len(dat[(dat[cor_type] < 0) & (dat['CoexistRank'] == 0)])
    return proportions

def report_coexistence_analysis(proportions, correlation_type):
    positive_key = f'positive_coexistence_{correlation_type}'
    negative_key = f'negative_coexistence_{correlation_type}'
    neg_confint = proportion_confint(count=proportions[negative_key], nobs=proportions[negative_key] + proportions[f'negative_exclusion_{correlation_type}'], alpha=0.05, method='wilson')
    pos_confint = proportion_confint(count=proportions[positive_key], nobs=proportions[positive_key] + proportions[f'positive_exclusion_{correlation_type}'], alpha=0.05, method='wilson')
    print(f"\nAnalysis on Negative \u03BD for {correlation_type.upper()}:")
    print(f"Proportion of coexistence with \u03BD < 0: {proportions[negative_key] / (proportions[negative_key] + proportions[f'negative_exclusion_{correlation_type}']):.4f} (95% CI: {neg_confint})")
    print(f"Proportion of coexistence with \u03BD \u2265 0: {proportions[positive_key] / (proportions[positive_key] + proportions[f'positive_exclusion_{correlation_type}']):.4f} (95% CI: {pos_confint})")

def analyze_coexistence_effect(file_path):
    dat = pd.read_csv(file_path)
    original_dat = dat.copy()
    models_results = {}
    scenarios = ['NonPCA', 'PCA']
    for scenario in scenarios:
        dat = apply_pca(original_dat) if scenario == 'PCA' else original_dat.copy()
        for correlation_type in ['SoS', 'CA']:
            analysis_type = f'{scenario}_{correlation_type}'
            correlation_column = 'cor_sos' if correlation_type == 'SoS' else 'cor_ca'
            if correlation_column not in dat.columns:
                continue
            print(f"\n--- Analysis for {analysis_type} ---")
            result = perform_logistic_regression(dat, analysis_type)
            aic = result.aic
            bic = result.bic
            n = len(dat)
            k = len(result.params)
            aicc = aic + (2 * k * (k + 1)) / (n - k - 1)
            # Store AIC, AICC, and BIC in models_results for each analysis type
            models_results[analysis_type] = {'AIC': aic, 'AICC': aicc, 'BIC': bic}
            proportions = calculate_proportions(dat, correlation_column)
            report_coexistence_analysis(proportions, correlation_column)
            table_data = {
                '\u03BD \u2265 0': [proportions[f'positive_coexistence_{correlation_column}'], proportions[f'positive_exclusion_{correlation_column}']],
                '\u03BD < 0': [proportions[f'negative_coexistence_{correlation_column}'], proportions[f'negative_exclusion_{correlation_column}']]
            }
            table_df = pd.DataFrame(table_data, index=['Coexistence', 'Exclusion'])
            print(f"Coexistence and Exclusion based on \u03BD for {analysis_type}:\n", table_df)
            # Proportion of coexistence calculations and confidence intervals
            for coexistence_type in ['positive', 'negative']:
                coexist_col = f'{coexistence_type}_coexistence_{correlation_column}'
                total_col = f'{coexistence_type}_exclusion_{correlation_column}'
                proportion = proportions[coexist_col] / (proportions[coexist_col] + proportions[total_col]) if (proportions[coexist_col] + proportions[total_col]) > 0 else 0
                confint = proportion_confint(count=proportions[coexist_col], nobs=proportions[coexist_col] + proportions[total_col], alpha=0.05, method='wilson')
#                 print(f"\nProportion of {coexistence_type} coexistence for {analysis_type}: {proportion:.4f} (95% CI: {confint})")
            # Decision making based on confidence intervals
            if coexistence_type == 'negative':
                if confint[1] < 0.5:
                    print(f"Higher coexistence observed with \u03BD \u2265 0 for {analysis_type}, not supporting the authors' claim.")
                elif confint[0] > 0.5:
                    print(f"Higher coexistence observed with \u03BD < 0 for {analysis_type}, supporting the authors' claim.")
                else:
                    print(f"Confidence intervals for proportions overlap for {analysis_type}, suggesting the effect of \u03BD on coexistence is inconclusive.")
    return models_results


# -

def model_selection(models_results):
    # Define the scenarios
    scenarios = ['NonPCA_SoS', 'PCA_SoS', 'NonPCA_CA', 'PCA_CA']
    # Prepare a dictionary to store AIC, AICC, and BIC for SoS+FE and CA
    criteria = {scenario: {'AIC': None, 'AICC': None, 'BIC': None} for scenario in scenarios}
    # Populate the dictionary with values from models_results
    for scenario in scenarios:
        if scenario in models_results:
            criteria[scenario]['AIC'] = models_results[scenario]['AIC']
            criteria[scenario]['AICC'] = models_results[scenario]['AICC']
            criteria[scenario]['BIC'] = models_results[scenario]['BIC']
    # Convert the dictionary to a pandas DataFrame for easy tabular display
    criteria_df = pd.DataFrame(criteria).T
    criteria_df.index.name = 'Scenario'
    criteria_df.columns.name = 'Criteria'
    print("\nModel Selection Criteria Table:")
    print(criteria_df)
    best_aic = criteria_df['AIC'].idxmin()
    best_aicc = criteria_df['AICC'].idxmin()
    best_bic = criteria_df['BIC'].idxmin()
    print(f"\nOverall best model based on AIC: {best_aic} (AIC: {criteria_df.loc[best_aic, 'AIC']})")
    print(f"Overall best model based on AICc: {best_aicc} (AICc: {criteria_df.loc[best_aicc, 'AICC']})")
    print(f"Overall best model based on BIC: {best_bic} (BIC: {criteria_df.loc[best_bic, 'BIC']})\n")


# +
def main():
    warnings.filterwarnings("ignore")
    # Specify paths for the output files
    initial_output_file = "csv/annplant_2spp_det_rare.csv"
    filtered_output_file = "csv/annplant_2spp_det_rare_filtered.csv"
    # Generate the parameter mesh
    mesh = preprocess_data()
    # Run the simulation for each parameter set in the mesh
    results = [Sim(k, row) for k, row in enumerate(mesh)]
    # Convert the list of dictionaries into a DataFrame and save to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(initial_output_file, index=False)
    # Apply filters and generate the filtered data CSV
    cor_figure()
    # Initialize dictionary for model results
    models_results = {}
    # Analysis for all scenarios using the filtered dataset
    print("Analysis for All Scenarios:")
    results = analyze_coexistence_effect(filtered_output_file)
    models_results.update(results)
    # Model Selection based on AIC, AICc, and BIC
    model_selection(models_results)

if __name__ == "__main__":
    main()
