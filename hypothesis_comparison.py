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

# ### The code performs a hypothesis comparison:
# #### - nu<0 promotes coexistence
# #### - Dominant PGR1 promotes coexistence
# #### - N1 a better competitor promotes coexistence (CE1>CE2)

import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from joblib import Parallel, delayed
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, normalized_mutual_info_score


def preprocess_data():
    r1_v  = np.arange(15, 21, 1)
    r2_v  = np.arange(15, 21, 1)
    a11_v = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1, 1.5, 2, 2.5, 3])
    a12_v = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1])
    a21_v = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1])
    a22_v = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1])
    mesh = np.array(np.meshgrid(r1_v, r2_v, a11_v, a12_v, a21_v, a22_v)).T.reshape(-1, 6)
    return mesh


def SOS(r1, r2, a11, a12, a21, a22):
    S1 = r2 / (1 + (a12/a22)*(r2 - 1))
    S2 = r1 / (1 + (a21/a11)*(r1 - 1))
    return S1, S2


def check_coexistence(r1, r2, a11, a12, a21, a22, eps=5.0e-2):
    E1 = (r1 - 1) / a11
    E2 = (r2 - 1) / a22
    P  = (r1 - 1) / a12
    Q  = (r2 - 1) / a21
    if (P > E2 and E1 > Q) or (E2 > P and Q > E1):
        return 0      # competitive exclusion
    if (P > E2 and Q > E1):
        return 1      # stable coexistence
    N1, N2 = find_equilibrium(r1, r2, a11, a12, a21, a22)
    if (N1 < eps or N2 < eps):
        return 0      # competitive exclusion
    if (N1 > eps and N2 > eps):
        return 1      # stable coexistence


def compute_nu(N1, N2, S1, S2):
    x = np.array([N1, N2], dtype=float)
    y = np.array([S1, S2], dtype=float)
    cov = np.cov(x, y)
    return cov[0,1]


def find_equilibrium(r1, r2, a11, a12, a21, a22, N1_init=5.0, N2_init=5.0, tol=1e-10, max_iter=10000):
    y1 = np.array([N1_init], dtype=np.float64)
    y2 = np.array([N2_init], dtype=np.float64)
    for _ in range(max_iter):
        denom1 = 1 + a11*y1[-1] + a12*y2[-1]
        denom2 = 1 + a22*y2[-1] + a21*y1[-1]
        new_y1 = y1[-1]*(r1/denom1)
        new_y2 = y2[-1]*(r2/denom2)
        y1 = np.append(y1, new_y1)
        y2 = np.append(y2, new_y2)
        if abs(y1[-1] - y1[-2]) < tol and abs(y2[-1] - y2[-2]) < tol:
            break
    return y1[-1], y2[-1]


def compute_zero_growth_Ntot(r, a_ii, a_ij, F_star):
    return (r - 1)/(a_ii*F_star + a_ij*(1.0 - F_star))


def compute_logPGR(r, Ntot_star, a_ii, a_ij, F_array):
    return np.log(r) - np.log(1.0 + Ntot_star*(a_ii*F_array + a_ij*(1.0 - F_array)))


def setup_plot_style():
    plt.rcParams.update({
        'axes.titlesize': 14,
        'axes.labelsize': 18,
        'xtick.labelsize':16,
        'ytick.labelsize':16,
        'legend.fontsize':12,
        'font.size':18,
        'lines.linewidth':1.5
    })


def plot_and_save(F, logPGR1, logPGR2, F1_star, nu, coexist, params, S1, S2, save_fig, eps=1e-3):
    # Unpack parameters
    r1, r2, a11, a12, a21, a22, N1_eq, N2_eq = params
    # Determine rare/common species at leftmost frequency (F=0)
    if abs(N1_eq - N2_eq) <= 1e-6:
        # Equal densities - compare directly
        if logPGR1[0] > logPGR2[0]:
            left_flag = 1
        elif logPGR1[0] < logPGR2[0]:
            left_flag = -1
        else:
            left_flag = 0
    elif N1_eq < N2_eq:
        # Species 1 is rare, species 2 is common
        left_flag = 1 if logPGR1[0] > logPGR2[0] else (-1 if logPGR1[0] < logPGR2[0] else 0)
    else:
        # Species 2 is rare, species 1 is common
        left_flag = 1 if logPGR2[0] > logPGR1[0] else (-1 if logPGR2[0] < logPGR1[0] else 0)
    # nu_sign: -1, 0, or 1
    nu_sign = -1 if nu < -eps else 1 if nu > eps else 0
    if save_fig:
        # classify nu category
        if nu_sign < 0: nu_dir = "nu_negative"
        elif nu_sign > 0: nu_dir = "nu_positive"
        else:             nu_dir = "nu_zero"
        subdir = "coexistence" if coexist == 1 else "exclusion"
        outdir = os.path.join(nu_dir, subdir)
        os.makedirs(outdir, exist_ok=True)
        plt.figure()
        plt.plot(F, logPGR1, label='N1')
        plt.plot(F, logPGR2, linestyle='--', label='N2')
        plt.axhline(0, color='grey', linestyle='--')
        plt.axvline(F1_star, linestyle=':')
        plt.xlim(-0.01, 1.01)
        plt.ylim(-1.01, 1.01)
        plt.xlabel('Frequency')
        plt.ylabel('log(PGR)')
        plt.legend()
        plt.title(f"nu={nu:.2g}, coexist={coexist}\nr1={r1}, r2={r2}, a11={a11}, a12={a12}, a21={a21}, a22={a22}")
        plt.tight_layout()
        fname = f"r1_{r1}_r2_{r2}_a11_{a11}_a12_{a12}_a21_{a21}_a22_{a22}.png"
        plt.savefig(os.path.join(outdir, fname))
        plt.close()
    # Build and return result dictionary
    result_dict = {
        'r1': r1, 'r2': r2, 
        'a11': a11, 'a12': a12, 'a21': a21, 'a22': a22,
        'N1_eq': N1_eq, 'N2_eq': N2_eq,
        'left_PGR1': logPGR1[0], 'left_PGR2': logPGR2[0],
        'left_flag': left_flag,
        'nu': nu, 'nu_sign': nu_sign,
        'coexist': coexist,
        'S1': S1, 'S2': S2,
        'F1_star': F1_star
    }
    return result_dict


def compute_competitive_efficiency(result_dict, eps=1e-6):
    # Extract parameters
    r1 = result_dict['r1']
    r2 = result_dict['r2']
    a11 = result_dict['a11']
    a12 = result_dict['a12']
    a21 = result_dict['a21']
    a22 = result_dict['a22']
    # Calculate Competitive Efficiency
    CE1 = ((r1 - 1) / a12) - ((r2 - 1) / a22)
    CE2 = ((r2 - 1) / a21) - ((r1 - 1) / a11)
    # Determine rare/common species based on equilibrium densities
    N1 = result_dict['N1_eq']
    N2 = result_dict['N2_eq']
    # Compare competitive efficiencies based on rare/common status
    if abs(N1 - N2) <= eps:
        # Equal densities
        CE_rare = CE1
        CE_common = CE2
    elif N1 < N2:
        # Species 1 is rare, species 2 is common
        CE_rare = CE1
        CE_common = CE2
    else:
        # Species 2 is rare, species 1 is common
        CE_rare = CE2
        CE_common = CE1
    # Determine CE case based on rare vs common comparison
    if CE_rare - CE_common > eps:
        CE_case = 1
    elif CE_common - CE_rare > eps:
        CE_case = -1
    else:
        CE_case = 0
    return CE1, CE2, CE_case


def compute_rank(result_dict):
    N1_eq = result_dict['N1_eq']
    N2_eq = result_dict['N2_eq']
    if N1_eq == 0 and N2_eq == 0:
        return 0  # Global extinction
    total = N1_eq + N2_eq
    freq1 = N1_eq / total
    return 2 if freq1 <= 0.25 else 1


# Define a structure for our results
RESULT_KEYS = [ 'r1', 'r2', 'a11', 'a12', 'a21', 'a22', 'N1_eq', 'N2_eq', 'left_PGR1', 'left_PGR2', 'left_flag', 'nu', 'nu_sign', 'coexist', 'S1', 'S2', 'CE1', 'CE2', 'CE_case', 'Rank', 'F1_star' ]


def save_results(results, filename="csv/results.csv", eps=1.0e-3):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_KEYS)
        writer.writeheader()
        for result_dict in results:
            # Compute competitive efficiency metrics
            CE1, CE2, CE_case = compute_competitive_efficiency(result_dict, eps)
            result_dict['CE1'] = CE1
            result_dict['CE2'] = CE2
            result_dict['CE_case'] = CE_case
            # Compute rank
            result_dict['Rank'] = compute_rank(result_dict)
            # Write the row
            writer.writerow({k: result_dict.get(k, '') for k in RESULT_KEYS})


def calculate_nu_proportions(results, eps=1.0e-3):
    counts = {
        "nu_negative": {0:0, 1:0},
        "nu_zero":     {0:0, 1:0},
        "nu_positive": {0:0, 1:0},
    }
    for result in results:
        nu = result['nu']
        coexist = result['coexist']
        key = "nu_negative" if nu < -eps else "nu_positive" if nu > eps else "nu_zero"
        counts[key][1 if coexist == 1 else 0] += 1
    proportions = {}
    print("\n\n===== NU SIGN ANALYSIS =====")
    for key in counts:
        total = counts[key][0] + counts[key][1]
        proportions[key] = counts[key][1] / total if total > 0 else 0
        co = counts[key][1] / total * 100 if total > 0 else 0
        ex = counts[key][0] / total * 100 if total > 0 else 0
        print(f"{key}: coexist={co:.4g}%, exclusion={ex:.4g}%")
    sorted_keys = sorted(
        counts,
        key=lambda k: counts[k][1] / (counts[k][0] + counts[k][1]),
        reverse=True
    )
    top, mid, bot = sorted_keys
    top_p = counts[top][1] / (counts[top][0] + counts[top][1]) * 100
    mid_p = counts[mid][1] / (counts[mid][0] + counts[mid][1]) * 100
    bot_p = counts[bot][1] / (counts[bot][0] + counts[bot][1]) * 100
    print(
        f"\n{top} (coexist: {top_p:.4g}%) had a larger proportion of coexistence than "
        f"{mid} (coexist: {mid_p:.4g}%), \nwhile {mid} had a larger proportion than "
        f"{bot} (coexist: {bot_p:.4g}%)."
    )
    return proportions


def calculate_dominance_proportions(results):
    counts = {
        "left_PGR1_nondominant": {0:0, 1:0},
        "left_PGR1_equal":       {0:0, 1:0},
        "left_PGR1_dominant":    {0:0, 1:0},
    }
    for result in results:
        left_flag = result['left_flag']
        coexist = result['coexist']
        if left_flag == -1:
            key = "left_PGR1_nondominant"
        elif left_flag == 0:
            key = "left_PGR1_equal"
        else:
            key = "left_PGR1_dominant"
        counts[key][1 if coexist == 1 else 0] += 1
    proportions = {}
    print("\n\n===== PGR DOMINANCE ANALYSIS =====")
    for key in counts:
        total = counts[key][0] + counts[key][1]
        proportions[key] = counts[key][1] / total if total > 0 else 0
        co = counts[key][1] / total * 100 if total > 0 else 0
        ex = counts[key][0] / total * 100 if total > 0 else 0
        print(f"{key}: coexist={co:.4g}%, exclusion={ex:.4g}%")
    sorted_keys = sorted(
        counts,
        key=lambda k: counts[k][1] / (counts[k][0] + counts[k][1]),
        reverse=True
    )
    top, mid, bot = sorted_keys
    top_p = counts[top][1] / (counts[top][0] + counts[top][1]) * 100
    mid_p = counts[mid][1] / (counts[mid][0] + counts[mid][1]) * 100
    bot_p = counts[bot][1] / (counts[bot][0] + counts[bot][1]) * 100
    print(
        f"\n{top} (coexist: {top_p:.4g}%) had a larger proportion of coexistence than "
        f"{mid} (coexist: {mid_p:.4g}%), \nwhile {mid} had a larger proportion than "
        f"{bot} (coexist: {bot_p:.4g}%)."
    )
    return proportions


def calculate_ce_cases(results, eps=1.0e-6):
    counts = {
        "CE_rare > CE_common": {0:0, 1:0},
        "CE_rare = CE_common": {0:0, 1:0},
        "CE_rare < CE_common": {0:0, 1:0},
    }
    for result in results:
        CE_case = result['CE_case']  # Get precomputed case
        coexist = result['coexist']
        if CE_case == 1:
            key = "CE_rare > CE_common"
        elif CE_case == -1:
            key = "CE_rare < CE_common"
        else:  # CE_case == 0
            key = "CE_rare = CE_common"
        counts[key][1 if coexist == 1 else 0] += 1
    proportions = {}
    print("\n\n===== COMPETITIVE EFFICIENCY COMPARISON ANALYSIS =====")
    # First pass: calculate proportions and print basic stats
    for key in counts:
        total = counts[key][0] + counts[key][1]
        if total == 0:
            print(f"{key}: No representatives")
            proportions[key] = 0
            continue
        proportions[key] = counts[key][1] / total
        co_percent = proportions[key] * 100
        ex_percent = 100 - co_percent
        print(f"{key}: coexist={co_percent:.4g}%, exclusion={ex_percent:.4g}%")
    # Second pass: sort keys by coexistence proportion
    valid_keys = [k for k in counts if (counts[k][0] + counts[k][1]) > 0]
    sorted_keys = sorted(
        valid_keys,
        key=lambda k: proportions[k],
        reverse=True
    )
    # Handle cases with ties in proportions
    grouped_keys = {}
    for key in sorted_keys:
        prop = proportions[key]
        if prop not in grouped_keys:
            grouped_keys[prop] = []
        grouped_keys[prop].append(key)
    # Report with tie-handling
    if len(sorted_keys) == 0:
        print("\nNo valid cases found")
    elif len(sorted_keys) == 1:
        key = sorted_keys[0]
        print(f"\nOnly one group: {key} (coexist: {proportions[key]*100:.4g}%)")
    else:
        # Group by proportion value to handle ties
        unique_props = sorted(set(proportions[k] for k in sorted_keys), reverse=True)
        groups = []
        for prop in unique_props:
            groups.append((prop, grouped_keys[prop]))
        # Build comparison string
        parts = []
        for i in range(len(groups)-1):
            higher_group = ", ".join(groups[i][1])
            lower_group = ", ".join(groups[i+1][1])
            higher_p = groups[i][0] * 100
            lower_p = groups[i+1][0] * 100
            parts.append(
                f"{higher_group} (coexist: {higher_p:.4g}%) had a larger "
                f"proportion of coexistence than {lower_group} (coexist: {lower_p:.4g}%)"
            )
        print("\n" + ",\nwhile ".join(parts) + ".")
    return proportions


def compare_hypotheses(results, eps=1e-3):
    # Print metric explanations
    print("\nMETRIC EXPLANATIONS:")
    print("1. ARC (Aligned Rank Correlation):")
    print("   - Measures monotonic relationship between hypothesis categories and coexistence")
    print("   - Ranges from -1 (perfect inverse) to 1 (perfect alignment)")
    print("   - Interpretation: Positive = hypothesis supported, near 0 = no relationship")
    print("   - Based on: Rank correlation between category order and coexistence proportions")
    print("\n2. AUC (Area Under ROC Curve):")
    print("   - Measures classification performance (coexistence vs exclusion)")
    print("   - ROC = Receiver Operating Characteristic curve")
    print("   - Ranges from 0 (worst) to 1 (best), 0.5 = random guessing")
    print("   - Interpretation: Higher = better at distinguishing coexistence outcomes")
    print("\n3. NMI (Normalized Mutual Information):")
    print("   - Measures information gain about coexistence from hypothesis categories")
    print("   - Ranges from 0 (no information) to 1 (perfect prediction)")
    print("   - Interpretation: Higher = hypothesis captures more information about coexistence")
    # Hypothesis definitions using dictionary keys
    hypotheses = {
        'nu_sign': {
            'direction': -1,  # nu<0 -> more coexistence
            'extractor': lambda res: res['nu_sign']
        },
        'left_dominance': {
            'direction': 1,   # left-dominance -> more coexistence
            'extractor': lambda res: res['left_flag']
        },
        'ce_case': {
            'direction': 1,   # CE_rare>CE_common -> more coexistence
            'extractor': lambda res: res['CE_case']
        }
    }
    # Data collection for metrics
    data = {name: {} for name in hypotheses}
    per_sim_data = {name: {'categories': [], 'labels': []} for name in hypotheses}
    for result in results:
        coexist = result['coexist']
        for name, hyp in hypotheses.items():
            category = hyp['extractor'](result)
            # For per-category counts (ARC)
            if category not in data[name]:
                data[name][category] = [0, 0]  # [coexists, total]
            data[name][category][1] += 1
            if coexist == 1:
                data[name][category][0] += 1
            # For per-simulation data (AUC and NMI)
            per_sim_data[name]['categories'].append(category)
            per_sim_data[name]['labels'].append(coexist)
    # Metrics storage
    metrics = {'ARC': {}, 'AUC': {}, 'NMI': {}}
    name_map = {
        'nu_sign': 'nu sign', 
        'left_dominance': 'PGR dominance', 
        'ce_case': 'Competitive Efficiency'
    }
    # Calculate metrics for each hypothesis
    for name, hyp in hypotheses.items():
        categories = sorted(data[name].keys())
        n_cats = len(categories)
        direction = hyp['direction']
        # 1. ARC (Aligned Rank Correlation)
        if n_cats >= 2:
            proportions = [data[name][c][0] / data[name][c][1] for c in categories]
            ranks = list(range(1, n_cats + 1))
            rho, _ = spearmanr(ranks, proportions)
            arc = (0.0 if np.isnan(rho) else rho) * direction
        else:
            arc = 0.0
        metrics['ARC'][name] = arc
        # 2. AUC (ROC Area Under Curve)
        labels = per_sim_data[name]['labels']
        raw_categories = per_sim_data[name]['categories']
        if len(set(labels)) < 2:
            auc_value = 0.5  # Handle single-class case
        else:
            # Apply direction to align categories with hypothesis
            scores = [c * direction for c in raw_categories]
            auc_value = roc_auc_score(labels, scores)
        metrics['AUC'][name] = auc_value
        # 3. NMI (Normalized Mutual Information)
        nmi_value = normalized_mutual_info_score(labels, raw_categories)
        metrics['NMI'][name] = nmi_value
    # Find best hypothesis for each metric (with tie handling)
    best_per_metric = {}
    for metric in metrics:
        # For ARC and AUC, higher is better; for NMI, higher is better
        max_val = max(metrics[metric].values())
        best_hypotheses = [hyp for hyp, val in metrics[metric].items() if abs(val - max_val) < 1e-6]
        best_per_metric[metric] = (max_val, best_hypotheses)
    # Generate aligned report
    print("\nHYPOTHESIS EVALUATION REPORT")
    print("=" * 70)
    print(f"{'Hypothesis':<25} {'ARC':>10} {'AUC':>10} {'NMI':>10}")
    print("-" * 70)
    for name in hypotheses:
        arc_str = f"{metrics['ARC'][name]:.2g}"
        auc_str = f"{metrics['AUC'][name]:.2g}"
        nmi_str = f"{metrics['NMI'][name]:.2g}"
        print(f"{name_map[name]:<25} {arc_str:>10} {auc_str:>10} {nmi_str:>10}")
    print("=" * 70)
    # Report best per metric
    print("\nBEST HYPOTHESIS PER METRIC:")
    for metric, (max_val, best_list) in best_per_metric.items():
        names = [name_map[hyp] for hyp in best_list]
        if len(names) == 1:
            tie_str = names[0]
        else:
            tie_str = ", ".join(names[:-1]) + " and " + names[-1]
        print(f"- {metric}: {tie_str} ({max_val:.2g})")
    return best_per_metric


def cor_figure(results, filter_option, truncate=False):
    # Create in-memory filtered results
    filtered_results = []
    for result in results:
        rank = result['Rank']
        S1 = result['S1']
        S2 = result['S2']
        if filter_option == 'on':
            if rank == 2 and S1 >= 1 and S2 >= 1:
                filtered_results.append(result)
        else:  # 'off'
            if rank == 2:
                filtered_results.append(result)
    # Convert to DataFrame for CSV output
    df = pd.DataFrame(filtered_results)
    if truncate:
        num_cols = df.select_dtypes(include=[np.number]).columns
        df[num_cols] = np.trunc(df[num_cols] * 100) / 100.0
    df.sort_values(by=['a22', 'a21', 'a12', 'a11', 'r2', 'r1'], inplace=True)
    outfile = f"csv/results_filtered_{filter_option}.csv"
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    df.to_csv(outfile, index=False)
    return filtered_results


def process_set(idx, params, F, save_fig):
    r1, r2, a11, a12, a21, a22 = params
    S1, S2 = SOS(r1, r2, a11, a12, a21, a22)
    coexist = check_coexistence(r1, r2, a11, a12, a21, a22)
    N1_eq, N2_eq = find_equilibrium(r1, r2, a11, a12, a21, a22)
    total_eq = N1_eq + N2_eq
    # Calculate F1_star from equilibrium
    F1_star = N1_eq / total_eq if total_eq > 0 else 0.5
    nu = compute_nu(N1_eq, N2_eq, S1, S2)
    Ntot1 = compute_zero_growth_Ntot(r1, a11, a12, F1_star)
    Ntot2 = compute_zero_growth_Ntot(r2, a22, a21, 1 - F1_star)
    logPGR1 = compute_logPGR(r1, Ntot1, a11, a12, F)
    logPGR2 = compute_logPGR(r2, Ntot2, a22, a21, F)
    params_tuple = (r1, r2, a11, a12, a21, a22, N1_eq, N2_eq)
    return plot_and_save(F, logPGR1, logPGR2, F1_star, nu, coexist, params_tuple, S1, S2, save_fig)


def main():
    save_fig = False
    setup_plot_style()
    F = np.linspace(0, 1, 200)
    mesh = preprocess_data()
    # Process all parameter sets with progress tracking
    print("Processing parameter sets...")
    results = []
    # Create parallel generator
    parallel_generator = Parallel(n_jobs=-1, return_as="generator")(
        delayed(process_set)(i+1, params, F, save_fig)
        for i, params in enumerate(mesh)
    )
    # Process results with tqdm progress bar
    with tqdm(total=len(mesh), desc="Parameter sets") as pbar:
        for result in parallel_generator:
            results.append(result)
            pbar.update(1)
    # Save results to CSV
    print("Saving results...")
    save_results(results, filename="csv/results.csv")
    # Perform analysis for different filter options
    for filter_option in ['on', 'off']:
        print("\n\n================================================================================================")
        print("================================================================================================\n\n")
        print(f"\n===== ANALYSIS FOR FILTER: {filter_option.upper()} =====")
        # Get filtered results and save to CSV
        filtered_results = cor_figure(results, filter_option, truncate=False)
        print(f"Number of cases: {len(filtered_results)}")
        # Only proceed if we have cases to analyze
        if len(filtered_results) > 0:
            nu_props = calculate_nu_proportions(filtered_results)
            dom_props = calculate_dominance_proportions(filtered_results)
            ce_props = calculate_ce_cases(filtered_results)
            print("\n\n================================================\n\n")
            print("\nHypothesis comparison:")
            best_hypothesis = compare_hypotheses(filtered_results)
        else:
            print("No cases meet the filter criteria")


if __name__ == '__main__':
    main()
