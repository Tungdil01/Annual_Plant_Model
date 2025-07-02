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
from sklearn.metrics import matthews_corrcoef


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


def check_coexistence(r1, r2, a11, a12, a21, a22, eps=0.05):
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


def PGR_dominance(F, logPGR1, logPGR2, F1_star, nu, coexist, params, S1, S2, eps=1e-3):
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


def save_results(results, filename="csv/annplant_2spp_det_rare.csv", eps=1.0e-3):
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
    print("1. Risk Difference (Prop. Difference):")
    print("   - Measures difference in coexistence probability")
    print("   - Formula: P(coexist|condition) - P(coexist|¬condition)")
    print("   - Range: [-1, 1]; >0 means condition promotes coexistence")
    print("\n2. Matthews Correlation Coefficient (MCC):")
    print("   - Measures quality of binary classification")
    print("   - Considers all confusion matrix elements")
    print("   - Range: [-1, 1]; 1 = perfect prediction, 0 = random\n")
    # Hypothesis definitions (binary conditions)
    hypotheses = {
        'nu_sign': {
            'description': 'Strength of Self-limitation (\u03BD<0)',
            'extractor': lambda res: 1 if res['nu'] < 0 else 0
        },
        'left_dominance': {
            'description': 'Per Capita Growth Rate (PGR1>PGR2)',
            'extractor': lambda res: 1 if res['left_flag'] == 1 else 0
        },
        'ce_case': {
            'description': 'Competitive Efficiency (CE1>CE2)',
            'extractor': lambda res: 1 if res['CE_case'] == 1 else 0
        }
    }
    # Initialize metrics storage
    metrics = {'Risk Difference': {}, 'MCC': {}}
    condition_data = {name: {'true': [], 'false': []} for name in hypotheses}
    # Collect coexistence outcomes per condition
    for res in results:
        coexist = int(res['coexist'])
        for name, hyp in hypotheses.items():
            condition = hyp['extractor'](res)
            if condition == 1:
                condition_data[name]['true'].append(coexist)
            else:
                condition_data[name]['false'].append(coexist)
    # Calculate metrics
    for name in hypotheses:
        true_vals = condition_data[name]['true']
        false_vals = condition_data[name]['false']
        # Risk Difference (Risk Difference)
        if not true_vals or not false_vals:
            metrics['Risk Difference'][name] = float('nan')
        else:
            p_true = np.mean(true_vals)
            p_false = np.mean(false_vals)
            metrics['Risk Difference'][name] = p_true - p_false
        # MCC calculation with FIXED degenerate case
        all_labels = []
        all_preds = []
        for res in results:
            all_labels.append(int(res['coexist']))
            all_preds.append(hypotheses[name]['extractor'](res))
        if len(set(all_preds)) > 1:  # MCC requires both classes present
            metrics['MCC'][name] = matthews_corrcoef(all_labels, all_preds)
        else:
            metrics['MCC'][name] = float('nan')  # Degenerate case
    # Find best hypothesis per metric
    best_per_metric = {}
    for metric_name, values in metrics.items():
        # Filter out NaN values
        valid_vals = {k: v for k, v in values.items() if not np.isnan(v)}
        if not valid_vals:
            best_per_metric[metric_name] = (float('nan'), [])
            continue
        max_val = max(valid_vals.values())
        best_hypotheses = [hyp for hyp, val in valid_vals.items() 
                          if abs(val - max_val) < eps]
        best_per_metric[metric_name] = (max_val, best_hypotheses)
    # Generate report
    print("HYPOTHESIS EVALUATION REPORT")
    print("=" * 65)
    print(f"{'Hypothesis':<40} {'Risk.Diff':>10} {'MCC':>10}")
    print("-" * 65)
    for name, hyp in hypotheses.items():
        rd = metrics['Risk Difference'][name]
        mcc = metrics['MCC'][name]
        # Format NaN values explicitly
        rd_str = "NaN" if np.isnan(rd) else f"{rd:>10.2g}"
        mcc_str = "NaN" if np.isnan(mcc) else f"{mcc:>10.2g}"
        print(f"{hyp['description']:<40} {rd_str} {mcc_str}")
    print("=" * 65)
    # Report best per metric (with NaN handling)
    print("\nBEST HYPOTHESIS PER METRIC:")
    for metric, (max_val, best_list) in best_per_metric.items():
        if not best_list:
            print(f"- {metric}: No valid comparisons (all hypotheses had NaN)")
            continue
        names = [hypotheses[name]['description'] for name in best_list]
        max_val_str = f"{max_val:.2g}" if not np.isnan(max_val) else "NaN"
        if len(names) == 1:
            print(f"- {metric}: {names[0]} ({max_val_str})")
        else:
            print(f"- {metric}: Tie between {', '.join(names)} ({max_val_str})")
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
    outfile = f"csv/annplant_2spp_det_rare_filtered_{filter_option}.csv"
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    df.to_csv(outfile, index=False)
    return filtered_results


def process_set(idx, params, F):
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
    return PGR_dominance(F, logPGR1, logPGR2, F1_star, nu, coexist, params_tuple, S1, S2)


def main():
    F = np.linspace(0, 1, 200)
    mesh = preprocess_data()
    # Process all parameter sets with progress tracking
    print("Processing parameter sets...")
    results = []
    # Create parallel generator
    parallel_generator = Parallel(n_jobs=-1, return_as="generator")(
        delayed(process_set)(i+1, params, F)
        for i, params in enumerate(mesh)
    )
    # Process results with tqdm progress bar
    with tqdm(total=len(mesh), desc="Parameter sets") as pbar:
        for result in parallel_generator:
            results.append(result)
            pbar.update(1)
    # Save results to CSV
    print("Saving results...")
    save_results(results, filename="csv/annplant_2spp_det_rare.csv")
    # Perform analysis for different filter options
    for filter_option in ['on', 'off']:
        print("\n\n================================================================================================")
        print("================================================================================================\n\n")
        print(f"\n===== ANALYSIS FOR FILTER: {filter_option.upper()} =====")
        # Get filtered results and save to CSV
        filtered_results = cor_figure(results, filter_option, truncate=False)
        print(f"Number of cases: {len(filtered_results)}")
        # Only proceed if we have cases to analyse
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
