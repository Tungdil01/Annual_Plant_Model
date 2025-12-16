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
import numpy as np
import pandas as pd
from scipy.stats import qmc, norm
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_curve, auc

plt.rcParams.update({
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'axes.titlesize': 16,
    'font.size': 16,
    'axes.grid': False,
    'text.usetex': False,
    'mathtext.fontset': 'stix',
    'font.family': 'STIXGeneral'
})


def compute_equilibria(params, model, eps=1e-8):
    r1, r2, a11, a22, a12, a21 = params
    if model == 'bevertonHolt':
        det = a11 * a22 - a12 * a21
        if abs(det) <= eps:
            return {}
        N1 = (a22 * (r1 - 1) - a12 * (r2 - 1)) / det
        N2 = (a11 * (r2 - 1) - a21 * (r1 - 1)) / det
        # Strength of Stabilization Adler et al. (2007)
        S1 = r2 / (1.0 + (a12 / a22) * (r2 - 1))
        S2 = r1 / (1.0 + (a21 / a11) * (r1 - 1))
        # Competition Ability Hart et al. (2018)
        CA1 = (r1 - 1) / np.sqrt(a12 * a11) if (a12 > eps and a11 > eps) else np.nan
        CA2 = (r2 - 1) / np.sqrt(a21 * a22) if (a21 > eps and a22 > eps) else np.nan
        # Competition Effect Streipert and Wolkowicz (2022)
        CE1 = (r1 - 1)/a12 - (r2 - 1)/a22 if (a12 > eps and a22 > eps) else np.nan
        CE2 = (r2 - 1)/a21 - (r1 - 1)/a11 if (a21 > eps and a11 > eps) else np.nan
        # Carrying capacity for Beverton-Holt: (r-1)/a
        N1_potential = (r1 - 1) / a11 if (a11 > eps and (r1 - 1) > 0.0) else eps
        N2_potential = (r2 - 1) / a22 if (a22 > eps and (r2 - 1) > 0.0) else eps
    elif model == 'ricker':
        det = a11 * a22 - a12 * a21
        if abs(det) <= eps:
            return {}
        N1 = (r1 * a22 - r2 * a12) / det
        N2 = (r2 * a11 - r1 * a21) / det
        S1 = np.exp(r2 * (1.0 - a12 / a22))
        S2 = np.exp(r1 * (1.0 - a21 / a11))
        # Carrying capacity for Ricker: r/a
        N1_potential = r1 / a11 if a11 > eps else eps
        N2_potential = r2 / a22 if a22 > eps else eps
    else:
        raise ValueError("Unknown model: %s" % model)
    nu_a = (N1 - N2) * (a11 - a22) / 2.0
    nu = (N1 - N2) * (S1 - S2) / 2.0 # Strength of Self-limitation Yenni et al. (2012)
    N1_realized = N1 if N1 > 0.0 else 0.0
    N2_realized = N2 if N2 > 0.0 else 0.0
    N1_potential = np.maximum(N1_potential, eps)
    N2_potential = np.maximum(N2_potential, eps)
    ASL1 = a11 * np.sqrt(N1_realized**2 + N1_potential**2)
    ASL2 = a22 * np.sqrt(N2_realized**2 + N2_potential**2)
    nu_ASL = (N1 - N2) * (ASL1 - ASL2) / 2.0
    if model == 'ricker':
        return {
            'N1': N1, 'N2': N2, 'S1': S1, 'S2': S2,
            'nu': nu, 'nu_ASL': nu_ASL, 'nu_a': nu_a
        }
    else:
        nu_CA = (N1 - N2) * (CA1 - CA2) / 2.0
        nu_CE = (N1 - N2) * (CE1 - CE2) / 2.0
        return {
            'N1': N1, 'N2': N2, 'S1': S1, 'S2': S2,
            'nu': nu, 'nu_ASL': nu_ASL, 'nu_a': nu_a,
            'nu_CA': nu_CA, 'nu_CE': nu_CE
        }


def check_analytical_scenarios_beverton_holt(params):
    r1, r2, a11, a22, a12, a21 = params
    if r1 <= 1 or r2 <= 1:
        return 'invalid' # Avoid division by zero
    # Calculate the analytical conditions
    cond1_left = a12
    cond1_right = a22 * (r1 - 1) / (r2 - 1)
    cond2_left = a21
    cond2_right = a11 * (r2 - 1) / (r1 - 1)
    # Check the four scenarios
    if cond1_left < cond1_right and cond2_left > cond2_right:
        return 'species1_wins'
    elif cond1_left > cond1_right and cond2_left < cond2_right:
        return 'species2_wins'
    elif cond1_left < cond1_right and cond2_left < cond2_right:
        return 'stable_coexistence'
    elif cond1_left > cond1_right and cond2_left > cond2_right:
        return 'saddle_point'
    else:
        return 'borderline' # Edge cases where inequalities are equal


def check_analytical_scenarios_ricker(params):
    r1, r2, a11, a22, a12, a21 = params
    if r1 <= 0 or r2 <= 0:
        return 'invalid'
    # Calculate the analytical conditions
    cond1 = a12 < (r1 * a22 / r2)
    cond2 = a21 < (r2 * a11 / r1)
    # Check the four scenarios
    if cond1 and cond2:
        return 'stable_coexistence'
    elif cond1 and not cond2:
        return 'species1_wins'
    elif not cond1 and cond2:
        return 'species2_wins'
    else:
        return 'borderline'


def bootstrap_percentile_proportion(event_mask, condition_mask=None, replicates=1000, seed=1234, alpha=0.05):
    if replicates <= 0:
        raise ValueError("bootstrap_percentile_proportion: replicates must be > 0")
    rng = np.random.default_rng(seed)
    event_mask = np.asarray(event_mask, dtype=bool)
    n = len(event_mask)
    if condition_mask is None:
        condition_mask = np.ones(n, dtype=bool)
    condition_mask = np.asarray(condition_mask, dtype=bool)
    if len(condition_mask) != n:
        raise ValueError("bootstrap_percentile_proportion: event_mask and condition_mask must have same length.")
    denom_obs = int(condition_mask.sum())
    if denom_obs == 0:
        raise ValueError("bootstrap_percentile_proportion: conditioning set is empty")
    est = float(np.count_nonzero(event_mask & condition_mask)) / float(denom_obs) # Observed estimate
    rep = np.empty(replicates, dtype=float) # Bootstrap replicates
    for i in range(replicates):
        idx = rng.integers(0, n, size=n)
        denom = int(condition_mask[idx].sum())
        if denom == 0: # Skip this replicate if condition empty
            rep[i] = np.nan
            continue
        num = int(np.count_nonzero(event_mask[idx] & condition_mask[idx]))
        rep[i] = float(num) / float(denom)
    rep = rep[~np.isnan(rep)] # Remove NaN values
    if len(rep) == 0:
        return est, np.nan, np.nan, np.array([])
    lower = float(np.percentile(rep, 100.0 * (alpha / 2.0)))
    upper = float(np.percentile(rep, 100.0 * (1.0 - alpha / 2.0)))
    return est, lower, upper, rep


def sample_parameters_lhs(n_samples=10000, seed=1234):
    r_range = (1.01, 20.0)
    a_range = (0.01, 3.0)
    bounds = np.array([r_range, r_range, a_range, a_range, a_range, a_range])
    low = bounds[:, 0]
    high = bounds[:, 1]
    sampler = qmc.LatinHypercube(d=6, seed=seed)
    u = sampler.random(n=n_samples)
    params = qmc.scale(u, low, high)
    return params


def analyze_parameter_set(params, model, metrics, eps = 1e-8):
    r1, r2, a11, a22, a12, a21 = params
    if model == 'bevertonHolt':
        scenario = check_analytical_scenarios_beverton_holt(params)
        coexistence = (scenario == 'stable_coexistence')
        competitive_exclusion = (scenario == 'species1_wins' or scenario == 'species2_wins')
        valid_outcome = (coexistence or competitive_exclusion)
    else:
        scenario = check_analytical_scenarios_ricker(params)
        coexistence = (scenario == 'stable_coexistence')
        competitive_exclusion = (scenario == 'species1_wins' or scenario == 'species2_wins')
        valid_outcome = (coexistence or competitive_exclusion)
    if not valid_outcome:
        result = {'valid_outcome': False, 'coexistence': False, 'competitive_exclusion': False, 'has_rare_species': False}
        for metric in metrics:
            result[metric] = np.nan
        return result
    if coexistence:
        equilibria = compute_equilibria(params, model)
        if not equilibria or any(np.isnan(equilibria.get(metric, np.nan)) for metric in metrics):
            result = {'valid_outcome': False, 'coexistence': False, 'competitive_exclusion': False, 'has_rare_species': False}
            for metric in metrics:
                result[metric] = np.nan
            return result
        N1 = equilibria.get('N1', 0.0)
        N2 = equilibria.get('N2', 0.0)
    else:
        if model == 'bevertonHolt':
            if scenario == 'species1_wins':
                N1 = (r1 - 1) / a11 if (a11 > eps and (r1 - 1) > 0.0) else 0.0
                N2 = 0.0
                N1_potential = (r1 - 1) / a11 if (a11 > eps and (r1 - 1) > 0.0) else eps
                N2_potential = (r2 - 1) / a22 if (a22 > eps and (r2 - 1) > 0.0) else eps
            else:
                N1 = 0.0
                N2 = (r2 - 1) / a22 if (a22 > eps and (r2 - 1) > 0.0) else 0.0
                N1_potential = (r1 - 1) / a11 if (a11 > eps and (r1 - 1) > 0.0) else eps
                N2_potential = (r2 - 1) / a22 if (a22 > eps and (r2 - 1) > 0.0) else eps
            S1 = r2 / (1.0 + (a12 / a22) * (r2 - 1))
            S2 = r1 / (1.0 + (a21 / a11) * (r1 - 1))
            CA1 = (r1 - 1) / np.sqrt(a12 * a11) if (a12 > eps and a11 > eps) else np.nan
            CA2 = (r2 - 1) / np.sqrt(a21 * a22) if (a21 > eps and a22 > eps) else np.nan
            CE1 = (r1 - 1)/a12 - (r2 - 1)/a22 if (a12 > eps and a22 > eps) else np.nan
            CE2 = (r2 - 1)/a21 - (r1 - 1)/a11 if (a21 > eps and a11 > eps) else np.nan
        else:
            if scenario == 'species1_wins':
                N1 = r1 / a11 if a11 > eps else 0.0
                N2 = 0.0
                N1_potential = r1 / a11 if a11 > eps else eps
                N2_potential = r2 / a22 if a22 > eps else eps
            else:
                N1 = 0.0
                N2 = r2 / a22 if a22 > eps else 0.0
                N1_potential = r1 / a11 if a11 > eps else eps
                N2_potential = r2 / a22 if a22 > eps else eps
            S1 = np.exp(r2 * (1.0 - a12 / a22))
            S2 = np.exp(r1 * (1.0 - a21 / a11))
            CA1 = np.nan
            CA2 = np.nan
            CE1 = np.nan
            CE2 = np.nan
        nu = (N1 - N2) * (S1 - S2) / 2.0
        nu_a = (N1 - N2) * (a11 - a22) / 2.0
        N1_realized = N1 if N1 > 0.0 else 0.0
        N2_realized = N2 if N2 > 0.0 else 0.0
        N1_potential = np.maximum(N1_potential, eps)
        N2_potential = np.maximum(N2_potential, eps)
        ASL1 = a11 * np.sqrt(N1_realized**2 + N1_potential**2)
        ASL2 = a22 * np.sqrt(N2_realized**2 + N2_potential**2)
        nu_ASL = (N1 - N2) * (ASL1 - ASL2) / 2.0
        if model == 'bevertonHolt':
            nu_CA = (N1 - N2) * (CA1 - CA2) / 2.0
            nu_CE = (N1 - N2) * (CE1 - CE2) / 2.0
            equilibria = {
                'N1': N1, 'N2': N2, 'S1': S1, 'S2': S2,
                'nu': nu, 'nu_ASL': nu_ASL, 'nu_a': nu_a,
                'nu_CA': nu_CA, 'nu_CE': nu_CE
            }
        else:
            equilibria = {
                'N1': N1, 'N2': N2, 'S1': S1, 'S2': S2,
                'nu': nu, 'nu_ASL': nu_ASL, 'nu_a': nu_a
            }
    result = {
        'valid_outcome': True,
        'coexistence': coexistence,
        'competitive_exclusion': competitive_exclusion,
        'has_rare_species': False
    }
    for metric in metrics:
        result[metric] = equilibria.get(metric, np.nan) if equilibria else np.nan
    N1_pos = max(N1, 0.0)
    N2_pos = max(N2, 0.0)
    total_pop = N1_pos + N2_pos
    if total_pop > eps:
        frac1 = N1_pos / total_pop
        frac2 = N2_pos / total_pop
        rarity_threshold = 0.25
        result['has_rare_species'] = (frac1 < rarity_threshold) or (frac2 < rarity_threshold) or (N1_pos <= eps) or (N2_pos <= eps)
    return result


def calculate_confusion_matrix(y_true, y_pred):
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    return TP, TN, FP, FN


def calculate_comparison_metrics(results, params, metrics, scenario, bootstrap_replicates=1000):
    valid_mask = results['valid_outcome']
    if not np.any(valid_mask):
        print("Warning: No valid outcomes for %s scenario" % scenario)
        return None, None, None, None
    if scenario == 'general':
        mask = valid_mask
    elif scenario == 'rarity':
        rare_mask = results['has_rare_species'][valid_mask]
        if np.sum(rare_mask) == 0:
            print("Warning: No rare species cases for %s scenario" % scenario)
            return None, None, None, None
        mask = valid_mask.copy()
        mask[valid_mask] = rare_mask
    else:
        raise ValueError("Unknown scenario: %s" % scenario)
    metrics_data = {}
    for metric in metrics:
        metrics_data[metric] = results[metric][mask]
    outcomes_raw = results['coexistence'][mask].astype(int)
    clean_mask = np.ones(len(outcomes_raw), dtype=bool)
    for metric in metrics:
        clean_mask = clean_mask & (~np.isnan(metrics_data[metric]))
    if not np.any(clean_mask):
        print("Warning: No non-NaN rows after cleaning for %s scenario" % scenario)
        return None, None, None, None
    for metric in metrics:
        metrics_data[metric] = np.asarray(metrics_data[metric])[clean_mask]
    outcomes = np.asarray(outcomes_raw)[clean_mask].astype(int)
    if len(outcomes) == 0:
        print("Warning: No valid data for %s scenario after NaN cleaning" % scenario)
        return None, None, None, None
    optimal_predictions = {}
    optimal_thresholds = {}
    optimal_directions = {}
    optimal_mcc_values = {}
    optimal_accuracies = {}
    auc_values = {}
    for metric in metrics:
        metric_vals = metrics_data[metric]
        if len(np.unique(outcomes)) < 2:
            optimal_predictions[metric] = np.zeros_like(outcomes)
            optimal_thresholds[metric] = 0.0
            optimal_directions[metric] = 'below'
            optimal_mcc_values[metric] = 0.0
            optimal_accuracies[metric] = 0.0
            auc_values[metric] = 0.5
            continue
        sorted_vals = np.sort(metric_vals)
        percentiles = np.linspace(1, 99, 99)
        thresholds = np.percentile(sorted_vals, percentiles)
        best_mcc = -1
        best_threshold = 0.0
        best_direction = 'below'
        best_pred = None
        for threshold in thresholds:
            for direction in ['below', 'above']:
                if direction == 'below':
                    predictions = (metric_vals < threshold).astype(int)
                else:
                    predictions = (metric_vals > threshold).astype(int)
                if len(np.unique(predictions)) < 2:
                    continue
                try:
                    mcc = matthews_corrcoef(outcomes, predictions)
                except:
                    mcc = -1
                if mcc > best_mcc:
                    best_mcc = mcc
                    best_threshold = threshold
                    best_direction = direction
                    best_pred = predictions
        if best_pred is None:
            best_pred = (metric_vals < 0).astype(int)
            best_threshold = 0.0
            best_direction = 'below'
            best_mcc = matthews_corrcoef(outcomes, best_pred) if len(np.unique(best_pred)) >= 2 else 0.0
        optimal_predictions[metric] = best_pred
        optimal_thresholds[metric] = best_threshold
        optimal_directions[metric] = best_direction
        optimal_mcc_values[metric] = best_mcc
        optimal_accuracies[metric] = float(np.sum(best_pred == outcomes)) / float(len(outcomes)) if len(outcomes) > 0 else 0.0
        if len(np.unique(metric_vals)) > 1:
            fpr, tpr, _ = roc_curve(outcomes, metric_vals)
            auc_val = auc(fpr, tpr)
            if auc_val < 0.5:
                auc_val = 1 - auc_val
            auc_values[metric] = auc_val
        else:
            auc_values[metric] = 0.5
    candidate_items = [(n, auc_values[n]) for n in metrics if n in auc_values]
    if len(candidate_items) == 0:
        best_name = None
        best_score = 0.0
        worse_score = 0.0
    else:
        best_name = max(candidate_items, key=lambda x: x[1])[0]
        best_score = auc_values.get(best_name, 0.0)
        sorted_scores = sorted([v for n, v in candidate_items], reverse=True)
        worse_score = sorted_scores[1] if len(sorted_scores) > 1 else 0.0
    metrics_dict = {
        'scenario': scenario,
        'n_cases': len(outcomes),
        'n_cases_raw': int(np.sum(mask)),
        'coexistence_count': int(np.sum(outcomes == 1)),
        'competitive_exclusion_count': int(np.sum(outcomes == 0)),
        'better_metric': best_name,
        'better_score': best_score,
        'worse_score': worse_score
    }
    for metric in metrics:
        metrics_dict[f'{metric}_auc'] = auc_values.get(metric, 0.5)
        metrics_dict[f'{metric}_mcc'] = optimal_mcc_values.get(metric, 0.0)
        metrics_dict[f'{metric}_accuracy'] = optimal_accuracies.get(metric, 0.0)
        metrics_dict[f'{metric}_threshold'] = optimal_thresholds.get(metric, 0.0)
        metrics_dict[f'{metric}_direction'] = optimal_directions.get(metric, 'below')
        pred_opt = optimal_predictions[metric]
        TP_opt, TN_opt, FP_opt, FN_opt = calculate_confusion_matrix(outcomes, pred_opt)
        metrics_dict[f'TP_{metric}'] = int(TP_opt)
        metrics_dict[f'TN_{metric}'] = int(TN_opt)
        metrics_dict[f'FP_{metric}'] = int(FP_opt)
        metrics_dict[f'FN_{metric}'] = int(FN_opt)
    return_values = [metrics_dict]
    for metric in metrics:
        return_values.append(metrics_data[metric])
    return tuple(return_values)


def compute_conditional_probs(event_mask, condition_mask, bootstrap_replicates, alpha, seed_offset):
    if np.sum(condition_mask) > 0:
        p_event_given_cond, p_lower, p_upper, _ = bootstrap_percentile_proportion(
            event_mask=event_mask,
            condition_mask=condition_mask,
            replicates=bootstrap_replicates,
            seed=seed_offset,
            alpha=alpha
        )
        return float(p_event_given_cond), float(p_lower), float(p_upper)
    return np.nan, np.nan, np.nan


def compute_probability_analysis(results, params, scenario, model, metrics, bootstrap_replicates=1000, confidence_level=0.95):
    coexistence_mask = np.asarray(results['coexistence'], dtype=bool)
    competitive_exclusion_mask = np.asarray(results['competitive_exclusion'], dtype=bool)
    has_rare_species = np.asarray(results['has_rare_species'], dtype=bool)
    valid_mask = np.asarray(results['valid_outcome'], dtype=bool)
    n_total = int(len(coexistence_mask))
    alpha = 1.0 - float(confidence_level)
    prob_results = {
        'model': model,
        'scenario': scenario,
        'n_total': n_total,
        'bootstrap_replicates': int(bootstrap_replicates),
        'confidence_level': float(confidence_level),
        'coexistence_count': int(np.sum(coexistence_mask)),
        'competitive_exclusion_count': int(np.sum(competitive_exclusion_mask))
    }
    if np.sum(valid_mask) > 0:
        p_val, p_low, p_up = compute_conditional_probs(coexistence_mask, valid_mask, bootstrap_replicates, alpha, 0)
        prob_results['P_coexistence_given_valid'] = p_val
        prob_results['P_coexistence_given_valid_lower'] = p_low
        prob_results['P_coexistence_given_valid_upper'] = p_up
    if scenario == 'general':
        base_mask = valid_mask
        seed_base = 1
    elif scenario == 'rarity':
        base_mask = valid_mask & has_rare_species
        seed_base = 10
    else:
        raise ValueError("Unknown scenario: %s" % scenario)
    if np.sum(base_mask) == 0:
        return prob_results
    for metric_name in metrics:
        metric_vals = np.asarray(results[metric_name], dtype=float)
        metric_defined_mask = (~np.isnan(metric_vals)) & base_mask
        metric_neg_mask = (metric_vals < 0) & metric_defined_mask
        metric_pos_mask = (metric_vals >= 0) & metric_defined_mask
        p_val, p_low, p_up = compute_conditional_probs(coexistence_mask, metric_neg_mask, bootstrap_replicates, alpha, seed_base)
        prob_results[f'P_coexistence_given_{metric_name}_negative'] = p_val
        prob_results[f'P_coexistence_given_{metric_name}_negative_lower'] = p_low
        prob_results[f'P_coexistence_given_{metric_name}_negative_upper'] = p_up
        p_val, p_low, p_up = compute_conditional_probs(coexistence_mask, metric_pos_mask, bootstrap_replicates, alpha, seed_base + 1)
        prob_results[f'P_coexistence_given_{metric_name}_positive'] = p_val
        prob_results[f'P_coexistence_given_{metric_name}_positive_lower'] = p_low
        prob_results[f'P_coexistence_given_{metric_name}_positive_upper'] = p_up
        coex_metric_defined = coexistence_mask & (~np.isnan(metric_vals)) & base_mask
        if np.sum(coex_metric_defined) > 0:
            p_val, p_low, p_up = compute_conditional_probs((metric_vals < 0), coex_metric_defined, bootstrap_replicates, alpha, seed_base + 2)
            prob_results[f'P_{metric_name}_negative_given_coexistence'] = p_val
            prob_results[f'P_{metric_name}_negative_given_coexistence_lower'] = p_low
            prob_results[f'P_{metric_name}_negative_given_coexistence_upper'] = p_up
            p_val, p_low, p_up = compute_conditional_probs((metric_vals >= 0), coex_metric_defined, bootstrap_replicates, alpha, seed_base + 3)
            prob_results[f'P_{metric_name}_positive_given_coexistence'] = p_val
            prob_results[f'P_{metric_name}_positive_given_coexistence_lower'] = p_low
            prob_results[f'P_{metric_name}_positive_given_coexistence_upper'] = p_up
        prob_results[f'{metric_name}_negative_coexistence_count'] = int(np.sum((metric_vals < 0) & coexistence_mask & (~np.isnan(metric_vals)) & base_mask))
        prob_results[f'{metric_name}_positive_coexistence_count'] = int(np.sum((metric_vals >= 0) & coexistence_mask & (~np.isnan(metric_vals)) & base_mask))
    return prob_results


def get_metric_config(metric_name):
    if metric_name.startswith('nu'):
        if '_' in metric_name:
            parts = metric_name.split('_')
            if len(parts) == 2:
                base, sub = parts
                return f'$\\nu_{{{sub}}}$'
        return '$\\nu$'
    elif metric_name in ['CA', 'CE']:
        return f'${metric_name}$'
    elif metric_name.startswith('ASL'):
        if len(metric_name) > 3:
            return f'$ASL_{{{metric_name[3:]}}}$'
        return '$ASL$'
    elif metric_name in ['S1', 'S2', 'N1', 'N2']:
        return f'${metric_name[0]}_{{{metric_name[1]}}}$'
    return f'${metric_name}$'


def plot_mcc_comparison(metrics_dict, model_name, metrics, scenario):
    if metrics_dict is None:
        print(f"No metrics to plot MCC comparison for {scenario} scenario")
        return
    auc_values = []
    mcc_values = []
    valid_metrics = []
    for metric in metrics:
        auc_val = metrics_dict.get(f'{metric}_auc', np.nan)
        mcc_val = metrics_dict.get(f'{metric}_mcc', np.nan)
        if not np.isnan(auc_val) and not np.isnan(mcc_val):
            auc_values.append(auc_val)
            mcc_values.append(mcc_val)
            valid_metrics.append(metric)
    if not valid_metrics:
        return
    fig, ax1 = plt.subplots(figsize=(10, 6))
    x = np.arange(len(valid_metrics))
    width = 0.35
    display_names_list = [get_metric_config(metric) for metric in valid_metrics]
    auc_max = max(auc_values) if auc_values else 1.0
    mcc_max = max(mcc_values) if mcc_values else 1.0
    bars_auc = ax1.bar(x - width/2, auc_values, width, color='blue', edgecolor='black', alpha=0.7, label='AUC')
    ax1.set_ylabel("Area Under Curve (AUC)", color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_ylim(0, auc_max * 1.1)
    ax1.set_xlabel("Metric")
    ax1.set_xticks(x)
    ax1.set_xticklabels(display_names_list, ha='center')
    ax2 = ax1.twinx()
    bars_mcc = ax2.bar(x + width/2, mcc_values, width, color='green', edgecolor='black', alpha=0.7, label='MCC')
    ax2.set_ylabel("Matthews Correlation Coefficient (MCC)", color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    ax2.set_ylim(0, mcc_max * 1.1)
    for i, (bar_auc, bar_mcc) in enumerate(zip(bars_auc, bars_mcc)):
        height_auc = bar_auc.get_height()
        height_mcc = bar_mcc.get_height()
        if not np.isnan(height_auc):
            ax1.text(bar_auc.get_x() + bar_auc.get_width()/2., height_auc + 0.01, f'{height_auc:.3g}', ha='center', va='bottom', color='blue')
        if not np.isnan(height_mcc):
            ax2.text(bar_mcc.get_x() + bar_mcc.get_width()/2., height_mcc + 0.01, f'{height_mcc:.3g}', ha='center', va='bottom', color='green')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    plt.tight_layout()
    os.makedirs('img', exist_ok=True)
    filename = f'img/metric_comparison_{model_name}_{scenario}'
    fig.savefig(f'{filename}.pdf', bbox_inches='tight', dpi=300)
    plt.close()


def plot_roc_curves(metrics_data_dict, outcomes, model_name, metrics, scenario):
    if len(outcomes) == 0:
        print(f"No data to plot ROC curves for {scenario} scenario")
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    for metric in metrics:
        if metric not in metrics_data_dict:
            continue
        metric_data = metrics_data_dict[metric]
        if len(metric_data) != len(outcomes):
            print(f"Warning: Metric {metric} data length ({len(metric_data)}) doesn't match outcomes length ({len(outcomes)})")
            continue
        valid_mask = ~np.isnan(metric_data)
        metric_data_clean = metric_data[valid_mask]
        outcomes_clean = outcomes[valid_mask]
        if len(metric_data_clean) == 0:
            print(f"Warning: No valid data for metric {metric} after removing NaN values")
            continue
        fpr, tpr, _ = roc_curve(outcomes_clean, -metric_data_clean)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, label=f"{get_metric_config(metric)} (AUC = {roc_auc:.3g})")
    if len(ax.lines) == 0:
        print(f"No ROC curves to plot for {scenario} scenario")
        plt.close(fig)
        return
    ax.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.legend(loc="lower right")
    plt.tight_layout()
    os.makedirs('img', exist_ok=True)
    filename = f'img/roc_curves_{model_name}_{scenario}.pdf'
    fig.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()


def plot_probability_analysis(prob_results, model_name, metrics, scenario):
    for metric in metrics:
        has_data = f'P_coexistence_given_{metric}_negative' in prob_results
        if not has_data:
            continue
        config = get_metric_config(metric)
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        coex_given_neg = prob_results[f'P_coexistence_given_{metric}_negative']
        coex_given_pos = prob_results.get(f'P_coexistence_given_{metric}_positive', np.nan)
        neg_given_coex = prob_results.get(f'P_{metric}_negative_given_coexistence', np.nan)
        pos_given_coex = prob_results.get(f'P_{metric}_positive_given_coexistence', np.nan)
        coex_given_neg_lower = prob_results.get(f'P_coexistence_given_{metric}_negative_lower', np.nan)
        coex_given_neg_upper = prob_results.get(f'P_coexistence_given_{metric}_negative_upper', np.nan)
        coex_given_pos_lower = prob_results.get(f'P_coexistence_given_{metric}_positive_lower', np.nan)
        coex_given_pos_upper = prob_results.get(f'P_coexistence_given_{metric}_positive_upper', np.nan)
        neg_given_coex_lower = prob_results.get(f'P_{metric}_negative_given_coexistence_lower', np.nan)
        neg_given_coex_upper = prob_results.get(f'P_{metric}_negative_given_coexistence_upper', np.nan)
        pos_given_coex_lower = prob_results.get(f'P_{metric}_positive_given_coexistence_lower', np.nan)
        pos_given_coex_upper = prob_results.get(f'P_{metric}_positive_given_coexistence_upper', np.nan)
        x = np.arange(2)
        width = 0.35
        conditions = [f'{config} < 0', f'{config} \u2265 0']
        coex_given_cond = [coex_given_neg, coex_given_pos]
        cond_given_coex = [neg_given_coex, pos_given_coex]
        bars1 = ax.bar(x - width/2, coex_given_cond, width, label='P(Coexistence | Condition)', edgecolor='black', alpha=0.7)
        bars2 = ax.bar(x + width/2, cond_given_coex, width, label='P(Condition | Coexistence)', edgecolor='black', alpha=0.7)
        error_bars_data = [
            (x[0] - width/2, coex_given_neg, coex_given_neg_lower, coex_given_neg_upper),
            (x[1] - width/2, coex_given_pos, coex_given_pos_lower, coex_given_pos_upper),
            (x[0] + width/2, neg_given_coex, neg_given_coex_lower, neg_given_coex_upper),
            (x[1] + width/2, pos_given_coex, pos_given_coex_lower, pos_given_coex_upper)
        ]
        for x_pos, val, lower, upper in error_bars_data:
            if not np.isnan(lower) and not np.isnan(upper):
                ax.errorbar(x_pos, val, yerr=[[val - lower], [upper - val]], fmt='none', ecolor='black', capsize=3, capthick=1)
        for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
            height1 = bar1.get_height()
            height2 = bar2.get_height()
            if not np.isnan(height1):
                ax.text(bar1.get_x() + bar1.get_width()/2., height1 + 0.04, f'{height1:.3g}', ha='center', va='bottom')
            if not np.isnan(height2):
                ax.text(bar2.get_x() + bar2.get_width()/2., height2 + 0.04, f'{height2:.3g}', ha='center', va='bottom')
        ax.set_xticks(x)
        ax.set_xticklabels(conditions, ha='center')
        ax.set_ylabel('Probability')
        ax.legend()
        ax.set_ylim(0, 1.1)
        plt.tight_layout()
        os.makedirs('img', exist_ok=True)
        filename = f'img/probability_analysis_{scenario}_{metric}_{model_name}.pdf'
        fig.savefig(filename, bbox_inches='tight', dpi=300)
        plt.close(fig)


# +
# def format_pct(value):
#     if abs(value - 100) < 0.01:
#         return "100%"
#     else:
#         return f"{value:.3g}%"

def format_prob(value):
    if np.isnan(value):
        return "NaN"
    if abs(value - 1.0) < 0.001:
        return "1.000"
    else:
        return f"{value:.3g}"


# -

def print_probability_analysis(prob_results, metrics, scenario):
    print(f"\n{'='*50}")
    print(f"PROBABILITY ANALYSIS (Coexistence vs Competitive Exclusion)")
    print(f"Scenario: {scenario.upper()}")
    print(f"{'='*50}")
    print(f"\n1. BASELINE PROBABILITY:")
    base_val = prob_results.get('P_coexistence_given_valid', np.nan)
    base_low = prob_results.get('P_coexistence_given_valid_lower', np.nan)
    base_up = prob_results.get('P_coexistence_given_valid_upper', np.nan)
    print(f"   P(Coexistence | Valid outcome) = {format_prob(base_val)} "
          f"[{format_prob(base_low)}, {format_prob(base_up)}]")
    total_valid = prob_results.get('coexistence_count', 0) + prob_results.get('competitive_exclusion_count', 0)
    print(f"   Total valid cases: {total_valid:,}")
    print(f"   Coexistence cases: {prob_results.get('coexistence_count', 0):,}")
    print(f"   Competitive exclusion cases: {prob_results.get('competitive_exclusion_count', 0):,}")
    print(f"\n2. HYPOTHESIS TESTS (P(Coexistence | Condition)):")
    for metric in metrics:
        neg_key = f'P_coexistence_given_{metric}_negative'
        pos_key = f'P_coexistence_given_{metric}_positive'
        has_data = neg_key in prob_results
        if not has_data:
            continue
        neg_val = prob_results[neg_key]
        pos_val = prob_results.get(pos_key, np.nan)
        metric_label = get_metric_config(metric)
        print(f"\n      {metric_label}:")
        print(f"        P(Coexistence | {metric_label} < 0) = {format_prob(neg_val)}")
        print(f"        P(Coexistence | {metric_label} \u2265 0) = {format_prob(pos_val)}")
        if not np.isnan(neg_val) and not np.isnan(pos_val):
            diff = neg_val - pos_val
            if diff > 0:
                print(f"        -> {metric_label} < 0 predicts {abs(diff):.3g} ({abs(diff)*100:.3g}%) MORE coexistence")
            elif diff < 0:
                print(f"        -> {metric_label} \u2265 0 predicts {abs(diff):.3g} ({abs(diff)*100:.3g}%) MORE coexistence")
            else:
                print(f"        -> No difference in prediction")
    print(f"\n3. COUNTS:")
    for metric in metrics:
        neg_key = f'{metric}_negative_coexistence_count'
        pos_key = f'{metric}_positive_coexistence_count'
        if neg_key in prob_results:
            print(f"      {get_metric_config(metric)} < 0 in coexistence: {prob_results.get(neg_key, 0):,}")
            print(f"      {get_metric_config(metric)} \u2265 0 in coexistence: {prob_results.get(pos_key, 0):,}")
    print(f"\n{'='*50}")


def print_classification_analysis(metrics_dict, metrics, scenario):
    if metrics_dict is None:
        print(f"No classification metrics for {scenario} scenario")
        return
    print(f"\n{'='*50}")
    print(f"COMPREHENSIVE METRIC ANALYSIS - {scenario.upper()} SCENARIO")
    print(f"{'='*50}")
    print(f"\n1. SAMPLE STATISTICS:")
    n_cases = metrics_dict['n_cases']
    coexistence_count = metrics_dict['coexistence_count']
    competitive_exclusion_count = metrics_dict['competitive_exclusion_count']
    coexistence_pct = (coexistence_count / n_cases * 100) if n_cases > 0 else 0
    print(f"   Total cases: {n_cases:,}")
    print(f"   Coexistence cases: {coexistence_count:,} ({coexistence_pct:.2g}%)")
    print(f"   Competitive exclusion cases: {competitive_exclusion_count:,} ({100-coexistence_pct:.2g}%)")
    print(f"\n2. METRIC COMPARISON (Hypothesis Independent):")
    print(f"   Which metric best separates classes, regardless of direction?")
    print(f"   {'Metric':<15} {'AUC':<10} {'MCC':<10} {'Threshold':<12} {'Direction':<10} {'Accuracy':<10}")
    print(f"   {'-'*15} {'-'*10} {'-'*10} {'-'*12} {'-'*10} {'-'*10}")
    for metric in metrics:
        auc_val = metrics_dict.get(f'{metric}_auc', np.nan)
        mcc_val = metrics_dict.get(f'{metric}_mcc', np.nan)
        threshold = metrics_dict.get(f'{metric}_threshold', 0.0)
        direction = metrics_dict.get(f'{metric}_direction', 'below')
        acc_val = metrics_dict.get(f'{metric}_accuracy', np.nan)
        label = get_metric_config(metric)
        auc_str = f'{auc_val:.3g}' if not np.isnan(auc_val) else 'NaN'
        mcc_str = f'{mcc_val:.3g}' if not np.isnan(mcc_val) else 'NaN'
        threshold_str = f'{threshold:.3g}'
        direction_str = f'{direction} threshold'
        acc_str = f'{acc_val:.3g}' if not np.isnan(acc_val) else 'NaN'
        print(f"   {label:<15} {auc_str:>9} {mcc_str:>9} {threshold_str:>11} {direction_str:>9} {acc_str:>9}")
    print(f"\n3. BEST METRIC ANALYSIS:")
    best_metric = metrics_dict.get('better_metric', 'none')
    best_score = metrics_dict.get('better_score', 0.0)
    if best_metric != 'none':
        best_direction = metrics_dict.get(f'{best_metric}_direction', 'below')
        best_threshold = metrics_dict.get(f'{best_metric}_threshold', 0.0)
        best_auc = metrics_dict.get(f'{best_metric}_auc', 0.5)
        best_mcc = metrics_dict.get(f'{best_metric}_mcc', 0.0)
        print(f"   Best metric (by AUC): {get_metric_config(best_metric)}")
        print(f"   AUC: {best_auc:.3g}")
        print(f"   MCC: {best_mcc:.3g}")
        print(f"   Optimal rule: {get_metric_config(best_metric)} {best_direction} {best_threshold:.3g}")
        auc_values = []
        for metric in metrics:
            if metric != best_metric:
                auc_val = metrics_dict.get(f'{metric}_auc', 0.5)
                auc_values.append(auc_val)
        if auc_values:
            second_best = max(auc_values)
            improvement = best_auc - second_best
            print(f"   AUC improvement over second best: {improvement:.3g}")
    print(f"\n4. HYPOTHESIS ALIGNMENT CHECK:")
    for metric in metrics:
        direction = metrics_dict.get(f'{metric}_direction', 'below')
        threshold = metrics_dict.get(f'{metric}_threshold', 0.0)
        label = get_metric_config(metric)
        if direction == 'below' and abs(threshold) < 0.1:
            alignment = "Aligns with hypothesis"
        else:
            alignment = f"Differs: {direction} threshold {threshold:.3g}"
        print(f"   {label}: {alignment}")
    print(f"\n{'='*50}")


def save_results_to_csv(results, params, prob_results, metrics_dict, model_name, scenario, metrics):
    df_results = pd.DataFrame({
        'r1': params[:, 0],
        'r2': params[:, 1],
        'a11': params[:, 2],
        'a22': params[:, 3],
        'a12': params[:, 4],
        'a21': params[:, 5],
        'valid_outcome': results['valid_outcome'],
        'coexistence': results['coexistence'],
        'competitive_exclusion': results['competitive_exclusion'],
        'has_rare_species': results['has_rare_species']
    })
    for metric in metrics:
        df_results[metric] = results[metric]
    os.makedirs('csv', exist_ok=True)
    results_filename = f'csv/results_{model_name}_{scenario}.csv'
    df_results.to_csv(results_filename, index=False)
    if prob_results:
        prob_df = pd.DataFrame([prob_results])
        prob_filename = f'csv/probability_{model_name}_{scenario}.csv'
        prob_df.to_csv(prob_filename, index=False)
    if metrics_dict:
        metrics_dict['model'] = model_name
        metrics_dict['scenario'] = scenario
        metrics_df = pd.DataFrame([metrics_dict])
        metrics_filename = f'csv/classification_metrics_{model_name}_{scenario}.csv'
        metrics_df.to_csv(metrics_filename, index=False)
    return df_results


def analyze_chunk(params_chunk, model, metrics):
    n = len(params_chunk)
    results = {
        'valid_outcome': np.zeros(n, dtype=bool),
        'coexistence': np.zeros(n, dtype=bool),
        'competitive_exclusion': np.zeros(n, dtype=bool),
        'has_rare_species': np.zeros(n, dtype=bool)
    }
    for metric in metrics:
        results[metric] = np.zeros(n, dtype=float)
    for i in range(n):
        res = analyze_parameter_set(params_chunk[i], model, metrics)
        results['valid_outcome'][i] = res['valid_outcome']
        results['coexistence'][i] = res['coexistence']
        results['competitive_exclusion'][i] = res['competitive_exclusion']
        results['has_rare_species'][i] = res['has_rare_species']
        for metric in metrics:
            results[metric][i] = res[metric]
    return results


def run_parallel_analysis(model, metrics, n_samples=10000, n_jobs=-1):
    print(f"Generating {n_samples:,} parameter samples...")
    params = sample_parameters_lhs(n_samples=n_samples)
    chunk_size = max(1, n_samples // n_jobs)
    print(f"Running analysis with {n_jobs} parallel jobs (chunk size: {chunk_size:,})...")
    chunks = []
    for i in range(0, n_samples, chunk_size):
        chunks.append(params[i:i + chunk_size])
    results_list = Parallel(n_jobs=n_jobs)(
        delayed(analyze_chunk)(chunk, model, metrics) for chunk in chunks
    )
    combined_results = {
        'valid_outcome': np.concatenate([r['valid_outcome'] for r in results_list]),
        'coexistence': np.concatenate([r['coexistence'] for r in results_list]),
        'competitive_exclusion': np.concatenate([r['competitive_exclusion'] for r in results_list]),
        'has_rare_species': np.concatenate([r['has_rare_species'] for r in results_list])
    }
    for metric in metrics:
        combined_results[metric] = np.concatenate([r[metric] for r in results_list])
    return combined_results, params


def analyze_model(model_name, metrics, scenario, n_samples=10000, n_jobs=-1, bootstrap_replicates=1000, confidence_level=0.95):
    metrics_dict = None
    np.random.seed(1234)
    print(f"\n{'='*50}")
    print(f"COMPREHENSIVE ANALYSIS FOR {model_name.upper()} MODEL")
    print(f"Scenario: {scenario.upper()}")
    print(f"{'='*50}")
    if model_name == 'ricker':
        metrics_for_model = [m for m in metrics if m not in ('nu_CA', 'nu_CE')]
    else:
        metrics_for_model = list(metrics)
    results, params = run_parallel_analysis(
        model=model_name,
        metrics=metrics_for_model,
        n_samples=n_samples,
        n_jobs=n_jobs
    )
    print(f"\nANALYSIS COMPLETED")
    print(f"Total samples: {n_samples:.3g}")
    valid_count = np.sum(results['valid_outcome'])
    print(f"Valid outcomes (coexistence or exclusion): {valid_count:.3g} ({valid_count/n_samples*100:.3g}%)")
    coexist_count = np.sum(results['coexistence'])
    excl_count = np.sum(results['competitive_exclusion'])
    if valid_count > 0:
        print(f"Coexistence cases: {coexist_count:.3g} ({coexist_count/valid_count*100:.3g}% of valid)")
        print(f"Competitive exclusion cases: {excl_count:.3g} ({excl_count/valid_count*100:.3g}% of valid)")
    else:
        print(f"Coexistence cases: {coexist_count:.3g}")
        print(f"Competitive exclusion cases: {excl_count:.3g}")
    print(f"\n{'='*50}")
    print("RUNNING COMPREHENSIVE PROBABILITY ANALYSIS...")
    print(f"Bootstrap replicates: {bootstrap_replicates:.3g}")
    prob_results = compute_probability_analysis(
        results, params, scenario, model=model_name, metrics=metrics_for_model,
        bootstrap_replicates=bootstrap_replicates,
        confidence_level=confidence_level
    )
    print_probability_analysis(prob_results, metrics_for_model, scenario)
    plot_probability_analysis(prob_results, model_name, metrics_for_model, scenario)
    print(f"\n{'='*50}")
    print("RUNNING CLASSIFICATION ANALYSIS...")
    classification_results = calculate_comparison_metrics(
        results, params, metrics_for_model, scenario, bootstrap_replicates
    )
    if classification_results and classification_results[0] is not None:
        metrics_dict = classification_results[0]
        metric_data_arrays = classification_results[1:]
        print_classification_analysis(metrics_dict, metrics_for_model, scenario)
        metrics_data_dict = dict(zip(metrics_for_model, metric_data_arrays))
        if scenario == 'general':
            mask = results['valid_outcome']
        elif scenario == 'rarity':
            mask = results['valid_outcome'] & results['has_rare_species']
        else:
            mask = results['valid_outcome']
        outcomes = np.asarray(results['coexistence'][mask])
        if len(outcomes) > 0:
            min_len = min(len(outcomes), len(metrics_data_dict[metrics_for_model[0]]))
            outcomes = outcomes[:min_len]
            for metric in metrics_for_model:
                metrics_data_dict[metric] = metrics_data_dict[metric][:min_len]
            plot_roc_curves(metrics_data_dict, outcomes, model_name, metrics_for_model, scenario)
        plot_mcc_comparison(metrics_dict, model_name, metrics_for_model, scenario)
    else:
        metrics_data_dict = None
    save_results_to_csv(results, params, prob_results,
                        metrics_dict if classification_results else None,
                        model_name, scenario, metrics_for_model)
    print(f"\n{'='*50}")
    print(f"ANALYSIS FOR {model_name.upper()} - {scenario.upper()} COMPLETE!")
    print(f"{'='*50}")
    return {
        'results': results,
        'params': params,
        'prob_results': prob_results,
        'metrics_dict': metrics_dict
    }


def main():
    n_samples = 20000
    n_jobs = -1
    bootstrap_replicates = int(0.1*n_samples)
    confidence_level = 0.95
    models = ['bevertonHolt'] # , 'ricker'
    metrics = ['nu', 'nu_CE'] # , 'nu_CA', 'nu_ASL', 'nu_a'
    scenarios = ['rarity'] # , 'general'
    print("="*50)
    print("COEXISTENCE vs COMPETITIVE EXCLUSION ANALYSIS")
    print("="*50)
    print(f"Number of samples: {n_samples:,}")
    print(f"Bootstrap replicates: {bootstrap_replicates:,}")
    print(f"Confidence level: {confidence_level:.0%}")
    print(f"Parallel jobs: {n_jobs if n_jobs > 0 else 'all cores'}")
    print(f"Scenarios: {', '.join(scenarios)}")
    print("="*50)
    all_results = {}
    for model in models:
        model_results = {}
        for scenario in scenarios:
            scenario_results = analyze_model(
                model_name=model,
                metrics=metrics,
                scenario=scenario,
                n_samples=n_samples,
                n_jobs=n_jobs,
                bootstrap_replicates=bootstrap_replicates,
                confidence_level=confidence_level
            )
            model_results[scenario] = scenario_results
        all_results[model] = model_results
    print(f"\n{'='*50}")
    print("FINAL SUMMARY")
    print(f"{'='*50}")
    for model, model_results in all_results.items():
        print(f"\n{model.upper()} Model:")
        for scenario in scenarios:
            if scenario in model_results:
                prob = model_results[scenario]['prob_results']
                metrics_dict = model_results[scenario].get('metrics_dict', {})
                print(f"  {scenario.upper()} scenario:")
                print(f"    P(Coexistence | Valid outcome): {prob.get('P_coexistence_given_valid', np.nan):.3g}")
                if metrics_dict:
                    for metric in metrics:
                        print(f"    {get_metric_config(metric)} MCC: {metrics_dict.get(f'{metric}_mcc', 0):.3g}")
    print(f"\n{'='*50}")
    print("ANALYSIS COMPLETE! All results saved to 'csv/' directory.")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()

# def time_simul(r1, r2, a11, a22, a12, a21, y01=5.0, y02=5.0, eps=1e-3):
#     y1 = np.array([y01], dtype=np.float64)
#     y2 = np.array([y02], dtype=np.float64)
#     stop_run = False
#     i = 0
#     while not stop_run and i < 1000:
#         denom1 = 1 + a11 * y1[i] + a12 * y2[i]
#         denom2 = 1 + a22 * y2[i] + a21 * y1[i]
#         per_cap1 = r1 / denom1
#         per_cap2 = r2 / denom2
#         new_y1 = y1[i] * per_cap1
#         new_y2 = y2[i] * per_cap2
#         y1 = np.append(y1, new_y1)
#         y2 = np.append(y2, new_y2)
#         if i >= 1:
#             if (abs(y1[-1] - y1[-2]) < eps and abs(y2[-1] - y2[-2]) < eps):
#                 stop_run = True
#         i += 1
#     return y1, y2

# def resolve_saddle_with_simulation(r1, r2, a11, a22, a12, a21, y01=5.0, y02=5.0):
#     try:
#         y1, y2 = time_simul(r1, r2, a11, a22, a12, a21, y01, y02)
#         final_N1 = y1[-1]
#         final_N2 = y2[-1]
#         # Check if both species persist
#         if final_N1 > 1e-6 and final_N2 > 1e-6:
#             return 'coexistence'
#         elif final_N1 > 1e-6 and final_N2 <= 1e-6:
#             return 'extinction_species2'  # Species 1 wins
#         elif final_N1 <= 1e-6 and final_N2 > 1e-6:
#             return 'extinction_species1'  # Species 2 wins
#         else:
#             return 'extinction_both'
#     except:
#         return 'simulation_error'
