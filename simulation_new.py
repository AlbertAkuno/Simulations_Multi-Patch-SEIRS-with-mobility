#import time
import json
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from odeintw import odeintw

plt.rcParams['font.family'] = ['serif'] # default is sans-serif
plt.rcParams['font.serif'] = ['Times New Roman']

rng = np.random.default_rng()

def SEIR_system(M, t, var_dict, pstar):
    S, E, I, R = M
    S, E, I, R = S.reshape(-1, 1), E.reshape(-1, 1), I.reshape(-1, 1), R.reshape(-1, 1)
    N_tilde = pstar.T @ var_dict["N_bar"]
    beta, gamma, kappa, lamda, mu, phi, tau = (
        var_dict["beta"].reshape(-1, 1),
        var_dict["gamma"].reshape(-1, 1),
        var_dict["kappa"].reshape(-1, 1),
        var_dict["lamda"].reshape(-1, 1),
        var_dict["mu"].reshape(-1, 1),
        var_dict["phi"].reshape(-1, 1),
        var_dict["tau"].reshape(-1, 1),
    )

    # import pdb;pdb.set_trace()
    dSdt = (
        lamda
        - np.diag(S.reshape(-1,)) @ pstar @ np.diag(beta.reshape(-1,)) @ np.linalg.inv(np.diag(N_tilde.reshape(-1,))) @ pstar.T @ I
        - np.diag(mu.reshape(-1,)) @ S
        + np.diag(tau.reshape(-1,)) @ R
    )

    dEdt = (
        np.diag(S.reshape(-1,)) @ pstar @ np.diag(beta.reshape(-1,)) @ np.linalg.inv(np.diag(N_tilde.reshape(-1,))) @ pstar.T @ I
        - np.diag((kappa + mu).reshape(-1,)) @ E
    )

    dIdt = np.diag(kappa.reshape(-1,)) @ E - np.diag((gamma + phi + mu).reshape(-1,)) @ I

    dRdt = (
        np.diag(gamma.reshape(-1,)) @ I
        - np.diag((tau + mu).reshape(-1,)) @ R
    )

    dSdt, dEdt, dIdt, dRdt = dSdt.reshape(-1,), dEdt.reshape(-1,), dIdt.reshape(-1,), dRdt.reshape(-1,)
    return np.array([dSdt, dEdt, dIdt, dRdt])


def initialise_variables(n):
    var_dict = {}
    var_dict["beta"] = rng.uniform(low=0, high=3, size=(n, 1))
    var_dict["gamma"] = np.ones((n, 1)) * 1 / 14
    var_dict["kappa"] = np.ones((n, 1)) * 1 / 7
    var_dict["lamda"] = np.ones((n, 1)) * 9
    var_dict["mu"] = np.ones((n, 1)) * 1 / 140
    var_dict["phi"] = np.ones((n, 1)) * 0.003
    var_dict["tau"] = np.ones((n, 1)) * 1 / 10
    var_dict["N_bar"] = var_dict["lamda"] * 1 / var_dict["mu"]

    var_dict["E0"] = np.zeros((n, 1))
    var_dict["I0"] = rng.integers(low=0, high=10, size=(n, 1))
    var_dict["R0"] = np.zeros((n, 1))
    var_dict["S0"] = var_dict["N_bar"] - (
        var_dict["E0"] + var_dict["I0"] + var_dict["R0"]
    )
    var_dict["M0"] = np.array(
        [var_dict["S0"], var_dict["E0"], var_dict["I0"], var_dict["R0"]]
    )

    return var_dict


def generate_random_p(n):
    p = rng.random(size=(n, n))
    p = p / p.sum(axis=1).reshape(-1, 1)
    return p


def generate_random_alpha(n):
    return rng.uniform(low=0, high=1, size=(n, 1))


def generate_random_pstar(n):
    p = generate_random_p(n)
    alpha = generate_random_alpha(n)
    pstar = p * alpha.reshape(-1, 1) + np.diag(
        1
        - alpha.reshape(
            -1,
        )
    )
    return pstar


def main():
    # residence_matrix = json.load(open('/workspace/CHAHAK/bbmm/normal_avg_res_mat/albert_avg_res_mat_First_First.json', 'r'))['residence_matrix']
    # residence_matrix = np.array(residence_matrix)
    # n = residence_matrix.shape[0]
    # n = 500
    for n in tqdm(range(23, 523, 20)):
        import time
        start = time.time()
        t = np.linspace(0, 300, 100)
        var_dict = initialise_variables(n)
        pstar = generate_random_pstar(n)
        sol = odeintw(SEIR_system, var_dict["M0"], t, args=(var_dict, pstar))
        # json.dump({"sol": sol}, open(f'/workspace/CHAHAK/bbmm/forward_simulation_n_{n}.json', 'r'), indent=2)

        fig, ax = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
        ax[0][0].plot(t, sol[:, 0, :, 0])  # all susceptible curves
        ax[0][0].set_title("Susceptible", fontsize=25)
        ax[0][0].set_ylabel(r"count", fontsize=20)
        ax[0][0].set_xlabel(r"time", fontsize=20)
        ax[0][1].plot(t, sol[:, 1, :, 0])  # all exposed curves
        ax[0][1].set_title("Exposed", fontsize=25)
        ax[0][1].set_ylabel(r"count", fontsize=20)
        ax[0][1].set_xlabel(r"time", fontsize=20)
        ax[1][0].plot(t, sol[:, 2, :, 0])  # all infected curves
        ax[1][0].set_title("Infected", fontsize=25)
        ax[1][0].set_xlabel(r"time", fontsize=20)
        ax[1][0].set_ylabel(r"count", fontsize=20)
        ax[1][1].plot(t, sol[:, 3, :, 0])  # all recovered curves
        ax[1][1].set_title("Recovered", fontsize=25)
        ax[1][1].set_xlabel(r"time", fontsize=20)
        ax[1][1].set_ylabel(r"count", fontsize=20)
        fig.savefig(f"/Users/albertakuno/Desktop/random_simulation_plots/random_simulation_n_{n}.jpg", format="jpg", dpi=300)
        end = time.time()
        if end - start > 15*60:
            print(f"Exceeded 300 seconds for the run: n: {n}, t: {end-start}")
            break

if __name__ == '__main__':
    main()

    
"""
Forward map simulation using the 503x503 residence matrix and 503x1 alpha values estimated 
from the AGEPS in Hermosillo.

"""

def generate_random_pstar_res_mat(n):
    p = residence_matrix
    alpha = alfa
    pstar = p * alpha.reshape(-1, 1) + np.diag(
        1
        - alpha.reshape(
            -1,
        )
    )
    return pstar


alfa=json.load(open('/Users/albertakuno/Desktop/albert_alpha_values_First_First.json', 'r'))['alpha_values']
alfa=np.array(alfa)
residence_matrix = json.load(open('/Users/albertakuno/Desktop/albert_avg_res_mat_First_First.json', 'r'))['residence_matrix']
residence_matrix = np.array(residence_matrix)
n=residence_matrix.shape[0]
#for n in tqdm(range(20, 520, 20)):
#start = time.time()
t = np.linspace(0, 300, 100)
var_dict = initialise_variables(n)
pstar = generate_random_pstar_res_mat(n)

import time
start = time.time()
sol_res_mat = odeintw(SEIR_system, var_dict["M0"], t, args=(var_dict, pstar))
end = time.time()

time=end-start
# json.dump({"sol": sol}, open(f'/workspace/CHAHAK/bbmm/forward_simulation_n_{n}.json', 'r'), indent=2)

fig, ax = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
ax[0][0].plot(t, sol_res_mat[:, 0, :, 0])  # all susceptible curves
ax[0][0].set_title("Susceptible", fontsize=25)
ax[0][0].set_ylabel(r"count", fontsize=20)
ax[0][0].set_xlabel(r"time", fontsize=20)
ax[0][1].plot(t, sol_res_mat[:, 1, :, 0])  # all exposed curves
ax[0][1].set_title("Exposed", fontsize=25)
ax[0][1].set_ylabel(r"count", fontsize=20)
ax[0][1].set_xlabel(r"time", fontsize=20)
ax[1][0].plot(t, sol_res_mat[:, 2, :, 0])  # all infected curves
ax[1][0].set_title("Infected", fontsize=25)
ax[1][0].set_xlabel(r"time", fontsize=20)
ax[1][0].set_ylabel(r"count", fontsize=20)
ax[1][1].plot(t, sol_res_mat[:, 3, :, 0])  # all recovered curves
ax[1][1].set_title("Recovered", fontsize=25)
ax[1][1].set_xlabel(r"time", fontsize=20)
ax[1][1].set_ylabel(r"count", fontsize=20)
plt.savefig("/Users/albertakuno/Desktop/res_mat_simulation/res_mat_simulation_n.jpg", format="jpg", dpi=300)


plt.plot(t,sol_res_mat[:, 2, :, 0].sum(axis=1))






