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


def compute_Nstar(r1, r2, a11, a12, a21, a22):
    denom = a11 * a22 - a12 * a21
    if np.isclose(denom, 0.0):
        raise ValueError("Division by zero in compute_Nstar: denominator (a11*a22 - a12*a21) is zero.")
    N1 = ((r1 - 1) * a22 - (r2 - 1) * a12) / denom
    N2 = ((r2 - 1) * a11 - (r1 - 1) * a21) / denom
    return N1, N2


def time_simul(r1, r2, a11, a22, a12, a21, y01=5.0, y02=5.0, eps=1e-3):
    y1 = np.array([y01], dtype=np.float64)
    y2 = np.array([y02], dtype=np.float64)
    stop_run = False
    i = 0
    while not stop_run and i < 1000:
        denom1 = 1 + a11 * y1[i] + a12 * y2[i]
        denom2 = 1 + a22 * y2[i] + a21 * y1[i]
        if np.isclose(denom1, 0.0):
            raise ValueError("time_simul: denominator1 is zero.")
        if np.isclose(denom2, 0.0):
            raise ValueError("time_simul: denominator2 is zero.")
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


def compute_S(r_j, a_ij, a_jj):
    if np.isclose(a_jj, 0.0):
        raise ValueError("Division by zero in compute_S: a_jj is zero.")
    rj_minus_1 = r_j - 1
    denom = 1 + rj_minus_1 * a_ij / a_jj
    if np.isclose(denom, 0.0):
        raise ValueError("Division by zero in compute_S: denominator (1 + (r_j-1)*a_ij/a_jj) is zero.")
    return r_j / denom


def compute_nu(N1, N2, S1, S2):
    return (N1 - N2) * (S1 - S2) / 2.0


def compute_C(r_i, a_ij, r_j, a_jj):
    if np.isclose(a_ij, 0.0):
        raise ValueError("Division by zero in compute_C: a_ij is zero.")
    if np.isclose(a_jj, 0.0):
        raise ValueError("Division by zero in compute_C: a_jj is zero.")
    return (r_i - 1) / a_ij - (r_j - 1) / a_jj


def check_coexistence(r1, r2, a11, a12, a21, a22):
    if r1 <= 1.0 or r2 <= 1.0:
        raise ValueError("check_coexistence: both r1 and r2 must be > 1.")
    if np.isclose(r1, 1.0) or np.isclose(r2, 1.0):
        raise ValueError("check_coexistence: r1 or r2 equals 1, division by zero in (r-1).")
    left12 = a22 * (r1 - 1) / (r2 - 1)
    left21 = a11 * (r2 - 1) / (r1 - 1)
    if a12 < left12 and a21 < left21:
        return ("stable coexistence", None, None)
    elif a12 < left12 and a21 > left21:
        return ("exclusion", 1, 2)
    elif a12 > left12 and a21 < left21:
        return ("exclusion", 2, 1)
    elif a12 > left12 and a21 > left21:
        return ("unstable saddle", None, None)
    elif np.isclose(a12, left12) and np.isclose(a21, left21):
        return ("borderline", None, None)
    else:
        raise ValueError("check_coexistence: unexpected combination of parameters.")


def get_equilibrium_populations(r1, r2, a11, a12, a21, a22):
    status, winner, loser = check_coexistence(r1, r2, a11, a12, a21, a22)
    if status == "stable coexistence":
        return compute_Nstar(r1, r2, a11, a12, a21, a22)
    elif status == "exclusion":
        if winner == 1:
            if np.isclose(a11, 0.0):
                raise ValueError("get_equilibrium_populations: a11 is zero, cannot compute single-species equilibrium.")
            N1 = (r1 - 1) / a11
            N2 = 0.0
        else:
            if np.isclose(a22, 0.0):
                raise ValueError("get_equilibrium_populations: a22 is zero, cannot compute single-species equilibrium.")
            N1 = 0.0
            N2 = (r2 - 1) / a22
        return N1, N2
    else: # unstable saddle or borderline
        y1, y2 = time_simul(r1, r2, a11, a22, a12, a21)
        return y1[-1], y2[-1]


def main():
    # r1, r2 = 20.0, 20.0
    # a11, a22 = 2.0, 2.0
    # a12, a21 = 1.0, 1.0
    r1, r2 = 2.0, 2.0
    a11, a22 = 2.0, 2.0
    a12, a21 = 1.0, 1.0
    print(f"a12 = {a12:.3g}")
    print(f"a22 * (r1 - 1) / (r2 - 1) = {a22 * (r1 - 1) / (r2 - 1):.3g}")
    print(f"a21 = {a21:.3g}")
    print(f"a11 * (r2 - 1) / (r1 - 1) = {a11 * (r2 - 1) / (r1 - 1):.3g}")
    print("\n---\n")
    if r1 <= 1.0 or r2 <= 1.0:
        raise ValueError("Both r1 and r2 must be > 1 for coexistence equilibrium to be defined.")
    print(f"r1 = {r1:.3g}")
    print(f"r2 = {r2:.3g}")
    print(f"a11 = {a11:.3g}")
    print(f"a12 = {a12:.3g}")
    print(f"a21 = {a21:.3g}")
    print(f"a22 = {a22:.3g}")
    print("\n---\n")
    N1, N2 = get_equilibrium_populations(r1, r2, a11, a12, a21, a22)
    S1 = compute_S(r1, a21, a11)
    S2 = compute_S(r2, a12, a22)
    nu = compute_nu(N1, N2, S1, S2)
    C1 = compute_C(r1, a12, r2, a22)
    C2 = compute_C(r2, a21, r1, a11)
    nu_C = compute_nu(N1, N2, C1, C2)
    status, _, _ = check_coexistence(r1, r2, a11, a12, a21, a22)
    print(f"N1* = {N1:.3g}, N2* = {N2:.3g}")
    print(f"S1 = {S1:.3g}, S2 = {S2:.3g}")
    print(f"nu = {nu:.3g}")
    print(f"C1 = {C1:.3g}, C2 = {C2:.3g}")
    print(f"nu_C = {nu_C:.3g}")
    print("\n---\n")
    print(f"Coexistence condition: {status}")


if __name__ == "__main__":
    main()
