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

import numpy as np


def compute_equilibria(params, model='bevertonHolt', eps=1e-8):
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
        # Competition Effect Streipert and Wolkowicz (2022)
        CE1 = (r1 - 1)/a12 - (r2 - 1)/a22 if (a12 > eps and a22 > eps) else np.nan
        CE2 = (r2 - 1)/a21 - (r1 - 1)/a11 if (a21 > eps and a11 > eps) else np.nan
        # nu Yenni et al. (2012)
        nu = (N1 - N2) * (S1 - S2) / 2.0
        nu_C = (N1 - N2) * (CE1 - CE2) / 2.0 if (not np.isnan(CE1) and not np.isnan(CE2)) else np.nan
        return {'N1': N1, 'N2': N2, 'S1': S1, 'S2': S2, 'CE1': CE1, 'CE2': CE2, 'nu': nu, 'nu_C': nu_C}


def format_value(value):
    if isinstance(value, (int, float, np.float64, np.float32)):
        return format(value, '.3g')
    return str(value)


def print_case(name, params, expected):
    print(f"\n{'='*50}")
    print(f"Parameter Set: {name}")
    print(f"{'='*50}")
    r1, r2, a11, a22, a12, a21 = params
    print(f"Parameters: r1={format_value(r1)}, r2={format_value(r2)}, a11={format_value(a11)}, a22={format_value(a22)}, a12={format_value(a12)}, a21={format_value(a21)}")
    results = compute_equilibria(params, model='bevertonHolt')
    formatted = {k: format_value(v) for k, v in results.items()}
    print(f"\nResults:")
    print(f"N1_eq = {formatted['N1']}")
    print(f"N2_eq = {formatted['N2']}")
    print(f"SOS1 = {formatted['S1']}")
    print(f"SOS2 = {formatted['S2']}")
    print(f"CE1 = {formatted['CE1']}")
    print(f"CE2 = {formatted['CE2']}")
    print(f"nu = {formatted['nu']}")
    print(f"nu_C = {formatted['nu_C']}")
    return results


def main():
    print("BEVERTON-HOLT MODEL ANALYSIS")
    print("="*50)
    parameter_sets = {
        'nu < 0': (15, 4, 2.3, 1, 0.91, 0.26),
        'nu_C < 0': (16, 9, 0.51, 1.6, 1.5, 0.049),
        'nu >= 0': (8.6, 9.8, 0.92, 1.8, 0.57, 0.88),
        'nu_C >= 0': (15, 4, 2.3, 1, 0.91, 0.26),
        'new': (20, 15, 1.0, 0.4, 0.5, 0.4),
    }
    all_results = {}
    for name, params in parameter_sets.items():
        all_results[name] = print_case(name, params, None)
    print(f"\n{'='*50}")
    print("SUMMARY TABLE")
    print(f"{'='*50}")
    print(f"{'Case':<12} {'N1':<8} {'N2':<8} {'nu':<10} {'nu_C':<10} {'S1':<8} {'S2':<8}")
    print("-"*50)
    for name, results in all_results.items():
        formatted = {k: format_value(v) for k, v in results.items()}
        print(f"{name:<12} {formatted['N1']:<8} {formatted['N2']:<8} {formatted['nu']:<10} {formatted['nu_C']:<10} {formatted['S1']:<8} {formatted['S2']:<8}")


if __name__ == "__main__":
    main()
