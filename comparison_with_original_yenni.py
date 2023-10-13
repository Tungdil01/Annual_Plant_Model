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

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats
from scipy.stats import ttest_ind
from datetime import datetime


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
            new_N1 = max((r1 - 1 - a12 * N[i-1, 1]) / a1, 0)
            new_N2 = max((r2 - 1 - a21 * N[i-1, 0]) / a2, 0)
            N[i, :] = [new_N1, new_N2]
        
        N1 = np.mean(N[:, 0])
        N2 = np.mean(N[:, 1])
    
    if N1 < 0:
        N1 = 0
        N2 = (r2 - 1) / a2
    
    if N2 < 0:
        N2 = 0
        N1 = (r1 - 1) / a1
    
    return N1, N2


# # annualplant_2spp_det_par.r

# +
# Print the current date and time
print(datetime.now())

# Define output file name
outfile = "csv/annplant_2spp_det_rare.csv"

# Define frequency-dependent parameters
l1_v = np.arange(15, 21)
l2_v = np.arange(15, 21)
a11_v = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1, 1.5, 2, 2.5, 3])
a12_v = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1])
a21_v = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1])
a22_v = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1])

# Generate all combinations of parameters using NumPy's meshgrid
mesh = np.array(np.meshgrid(l1_v, l2_v, a11_v, a12_v, a21_v, a22_v)).T.reshape(-1, 6)

# Initialize an empty NumPy array to hold the results
n_rows = mesh.shape[0]
results = np.empty((n_rows, 17), dtype=float)

# Simulation function
def Sim(k, mesh_row):
    l1, l2, a11, a12, a21, a22 = mesh_row
    
    N1, N2 = analyN(l1, l2, a11, a12, a21, a22)
    
    CoexistRank = 0 if N1 < 1 else 1
    
#     The original code from Yenni et al. replaced r1 with r2 (in this case, l1 and l2):
    S1 = l2 / (1 + (a12 / a22) * (l2 - 1))
    S2 = l1 / (1 + (a21 / a11) * (l1 - 1))
#     # If the Strength of Stabilization is fixed, the result is different:
#     S1 = l1 / (1 + (a12 / a22) * (l2 - 1))
#     S2 = l2 / (1 + (a21 / a11) * (l1 - 1))
    
    E1, E2 = l1 / l2, l2 / l1  # fitness equivalence
    Asy = S1 - S2
    if N1 == 0 and N2 == 0:
        Rare = 0
    else:
        Rare = N1 / (N1 + N2)

    x = np.array([N1, N2])
    y = np.array([S1, S2])
    cov_matrix = np.cov(x, y)
    cor = cov_matrix[0, 1]
    
    if N1 == 0 and N2 == 0:
        Rank = 0
    elif N1 / (N1 + N2) <= 0.25:
        Rank = 2 # choose N1 rare: frequency <= 0.25 
    else:
        Rank = 1
    
    return np.array([l1, l2, a11, a12, a21, a22, N1, N2, E1, E2, S1, S2, Rank, CoexistRank, Asy, cor, Rare])

# Run the simulation for each row in the DataFrame
for k in range(n_rows):
    results[k] = Sim(k, mesh[k])

# Convert the NumPy array back to a DataFrame
column_order = ['l1', 'l2', 'a11', 'a12', 'a21', 'a22', 'N1', 'N2', 'E1', 'E2', 'S1', 'S2', 'Rank', 'CoexistRank', 'Asy', 'cor', 'Rare']
simul = pd.DataFrame(results, columns=column_order)

# Save the DataFrame to a CSV file
simul.to_csv(outfile, index=False)

# -

# # cor_figure.r

# +
# Read the data
dat_det = pd.read_csv("csv/annplant_2spp_det_rare.csv")

# Filter the data using query for better readability and performance
dat_det = dat_det.query('Rank == 2 & S1 >= 1 & S2 >= 1')

# Reset the index (optional but recommended for clean data)
dat_det.reset_index(drop=True, inplace=True)

# Save the modified DataFrame back to the same CSV file (or a different one if you prefer)
dat_det.to_csv("csv/annplant_2spp_det_rare_filtered.csv", index=False)

# Calculate the correlation (covariance in this case) between abundance and stabilization
x = dat_det[['N1', 'N2']].values
y = dat_det[['S1', 'S2']].values

# Calculate covariance for each row and store it in a new 'cor' column
dat_det['cor'] = [np.cov(x[i], y[i])[0, 1] for i in range(len(x))]

# Save the DataFrame with the updated 'cor' column
dat_det.to_csv("csv/annplant_2spp_det_rare_filtered.csv", index=False)
# -

# # figures_det.r

# +
# Read the data
dat = pd.read_csv("csv/annplant_2spp_det_rare.txt") # Yenni et al. original result

# Effect on coexistence
X = dat[['S1', 'E1', 'cor']]
X = sm.add_constant(X)
y = dat['CoexistRank']
model = sm.GLM(y, X, family=sm.families.Binomial())
result = model.fit()
print(result.summary())

# +
# Read the data
dat = pd.read_csv("csv/annplant_2spp_det_rare_filtered.csv")

# Effect on coexistence
X = dat[['S1', 'E1', 'cor']]
X = sm.add_constant(X)
y = dat['CoexistRank']
model = sm.GLM(y, X, family=sm.families.Binomial())
result = model.fit()
print(result.summary())
