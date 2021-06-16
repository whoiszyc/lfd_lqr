import numpy as np
from scipy.linalg import solve_discrete_are
import cvxpy as cp
from algorithms import policy_fitting, policy_fitting_with_a_kalman_constraint
import warnings
# warnings.filterwarnings('ignore')

n, m = 4, 2
A = np.array([
    [.99997, .039, 0, -.322],
    [-.065, .99681, 7.74, 0],
    [.02, -.101, .99571, 0],
    [0, 0, 1, 1]
])
B = np.array([
    [.0001, 0 ],
    [-.0018, -.0004 ],
    [-.0116, .00598 ],
    [0, 0]
])
W = .1*np.array([
    [ 1.00092109, -0.02610491,  0.016055  ,  0.        ],
    [-0.02610491,  0.99785518, -0.10197781,  0.        ],
    [ 0.016055  , -0.10197781,  0.010601  ,  0.        ],
    [ 0.        ,  0.        ,  0.        ,  0.        ]
])
Q_true = np.diag([1, 1, 1, 1])
R_true = np.diag([1, 1])
P_true = solve_discrete_are(A, B, Q_true, R_true)
K_true = -np.linalg.solve(R_true + B.T @ P_true @ B, B.T @ P_true @ A)


def simulate(K, N=10, seed=None, add_noise=False):
    if seed is not None:
        np.random.seed(seed)
    x = np.random.multivariate_normal(np.zeros(n), W)
    xs = []
    us = []
    cost = 0.0
    for _ in range(N):
        u = K @ x
        if add_noise:
            u += 5 * np.random.randn(m)
        xs.append(x)
        us.append(u)
        cost += (x @ Q_true @ x + u @ R_true @ u) / N
        x = A @ x + B @ u + np.random.multivariate_normal(np.zeros(n), W)
    xs = np.array(xs)
    us = np.array(us)

    return cost, xs, us

N_test = 10
# cost_true = simulate(K_true, N=N_test, seed=0)[0]
# cost_noise = simulate(K_true, N=N_test, seed=0, add_noise=True)[0]
# cost_true, np.trace(P_true @ W), cost_noise
cost_true = np.trace(P_true @ W)
cost_noise = np.trace(P_true @ (W + 25 * B @ B.T))

costs_lr = []
costs_admm = []
Ns = np.arange(1, 51)
for N in Ns:
    costs_lr += [[]]
    costs_admm += [[]]
    for k in range(1, 11):
        print("Inner iterative {}".format(k))

        _, xs, us = simulate(K_true, N=N, seed=k, add_noise=True)


        def L(K):
            return cp.sum_squares(xs @ K.T - us)


        def r(K):
            return 1. * cp.sum_squares(K), []


        Klr = policy_fitting(L, r, xs, us)
        Kadmm = policy_fitting_with_a_kalman_constraint(L, r, xs, us, A, B, n_random=5)

        cost_lr = simulate(Klr, N=N_test, seed=0)[0]
        cost_admm = simulate(Kadmm, N=N_test, seed=0)[0]

        if np.isnan(cost_lr) or cost_lr > 1e5 or cost_lr == np.inf:
            cost_lr = np.nan

        costs_lr[-1].append(cost_lr)
        costs_admm[-1].append(cost_admm)

    print(" %03d | %3.3f | %3.3f | %3.3f | %3.3f" %
          (N, cost_true, cost_noise, np.nanmean(costs_lr[-1]), np.nanmean(costs_admm[-1])))


costs_lr = np.array(costs_lr)
costs_admm = np.array(costs_admm)

mean_lr = np.nanmean(costs_lr, axis=1)
std_lr = np.nanstd(costs_lr, axis=1)

mean_admm = np.nanmean(costs_admm, axis=1)
std_admm = np.nanstd(costs_admm, axis=1)

from utils import latexify
import matplotlib.pyplot as plt

latexify(fig_width=6, fig_height=2.7)
plt.axhline(cost_noise, ls='--', c='k', label='expert')
plt.scatter(np.arange(1,51), mean_lr, s=4, marker='o', c='blue', label='PF')
plt.fill_between(np.arange(1,51), mean_lr - std_lr / 3, mean_lr + std_lr / 3, alpha=.5, color='blue')
plt.scatter(np.arange(1,51), mean_admm, s=4, marker='*', c='green', label='ours')
plt.fill_between(np.arange(1,51), mean_admm - std_admm / 3, mean_admm + std_admm / 3, alpha=.5, color='green')
plt.semilogy()
plt.axhline(cost_true, ls='-', c='k', label='optimal')
plt.ylabel('cost')
plt.xlabel('demonstrations')
plt.legend()
plt.tight_layout()
plt.savefig("figs/aircraft_control.pdf")
plt.show()

latexify(fig_width=6, fig_height=2)
plt.plot(np.arange(1, 51), np.isnan(costs_lr).mean(axis=1), c='k')
plt.ylim(0,1.1)
plt.xlabel("demonstrations")
plt.ylabel("fraction infinite")
plt.tight_layout()
plt.savefig("figs/inf_aircraft_control.pdf")







