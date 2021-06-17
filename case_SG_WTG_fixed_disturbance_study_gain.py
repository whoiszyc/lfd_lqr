import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_discrete_are, eig
import cvxpy as cp
from algorithms import policy_fitting, policy_fitting_with_a_kalman_constraint
from dyn_sim import dyn_sim_discrete_time, dyn_sim_feedback_discrete_time
from dyn_noc import dyn_noc
import warnings
# warnings.filterwarnings('ignore')
import sys
sys.setrecursionlimit(100000000) # 10000 is an example, try with different values

A = np.array([
    [0.9999, 0.2926, 0.0071, 0.1544],
    [-0.0008, 0.9512, 0.0464, -0.0000],
    [-0.0317, -0.0048, 0.9048, -0.0025],
    [0,     0,      0,      0.9972]
])
B = np.array([
    [-0.1081, -0.3000],
    [0.0000, 0.0001],
    [0.0017, 0.0048],
    [0.0019, 0]
])
Bu = B[:, [0]]
Bd = B[:, [1]]

# simulation test
delta_t = 0.01  # time step size (seconds)
t_max = 3.0
T_series_control = np.linspace(0.0, t_max, int(t_max / delta_t))  # Get timesteps
T_series_sim = np.linspace(0.0, 10, int(10 / delta_t))  # Get timesteps
x0 = np.array([[0], [0], [0], [0]])  # Initial condition

# open-loop
Xt, Ut, _ = dyn_sim_discrete_time(A, Bu, Bd, x0, np.array([[0]]), np.array([[0.1]]), T_series_sim)

# NOC control
U_noc = dyn_noc(A, Bu, Bd, x0, [0.1], T_series_control)  # solve NOC problem
Xt_noc, Ut_noc, _ = dyn_sim_discrete_time(A, Bu, Bd, x0, U_noc, np.array([[0.1]]), T_series_control)  # simulate NOC


# Curve fitting NOC signal
time_interval_learn = 80
def L(K):
    return cp.sum_squares(Xt_noc.T[0:time_interval_learn, :] @ K.T - U_noc.T[0:time_interval_learn, :])
def r(K):
    return 0 * cp.sum_squares(K), []

# curve fitting NOC signal directly
K_noc = policy_fitting(L, r, Xt_noc.T[0:time_interval_learn, :], U_noc.T[0:time_interval_learn, :])
Acl_noc = A + Bu @ K_noc  # closed-loop A
eig_noc, _ = eig(Acl_noc)
Xt_cl_noc, Ut_cl_noc, _ = dyn_sim_feedback_discrete_time(A, Bu, Bd, x0, np.array([[0]]), np.array([[0.1]]), T_series_sim, K_noc)

# curve fitting NOC signal with Kalman constraints
K_noc_Kalman = policy_fitting_with_a_kalman_constraint(L, r, Xt_noc.T[0:time_interval_learn, :], U_noc.T[0:time_interval_learn, :], A, Bu)
Acl_noc_Kalman = A + Bu @ K_noc_Kalman  # closed-loop A
eig_noc_Kalman, _ = eig(Acl_noc_Kalman)
Xt_cl_noc_Kalman, Ut_cl_noc_Kalman, _ = dyn_sim_feedback_discrete_time(A, Bu, Bd, x0, np.array([[0]]), np.array([[0.1]]), T_series_sim, K_noc_Kalman)

print("Learn NOC w/o Kalman constraint {}".format(K_noc))
print("Learn NOC w Kalman constraint {}".format(K_noc_Kalman))
print("Eigenvalue w/o Kalman constraint {}".format(eig_noc))
print("Eigenvalue w Kalman constraint {}".format(eig_noc_Kalman))

# compare all cases
f = plt.figure(figsize=(15, 9))
ax = f.add_subplot(411)
ax.plot(T_series_sim, Xt[0, :], label="Open-loop")
ax.plot(T_series_control, Xt_noc[0, :], label="NOC")
ax.plot(T_series_sim, Xt_cl_noc[0, :], label="Leanred NOC")
ax.plot(T_series_sim, Xt_cl_noc_Kalman[0, :], linestyle='--', label="Leanred NOC w Kalman")
ax.set_ylabel("Frequency (Hz)")
ax.legend(loc='lower right', fontsize=8, ncol=5)
bx = f.add_subplot(412)
bx.plot(T_series_sim, Xt[1, :], label="Open-loop")
bx.plot(T_series_control, Xt_noc[1, :], label="NOC")
bx.plot(T_series_sim, Xt_cl_noc[1, :], label="Leanred NOC")
bx.plot(T_series_sim, Xt_cl_noc_Kalman[1, :], linestyle='--', label="Leanred NOC w Kalman")
bx.set_ylabel("Mechenical (PU)")
bx.legend(loc='lower right', fontsize=8, ncol=5)
cx = f.add_subplot(413)
cx.plot(T_series_sim, Xt[3, :], label="Open-loop")
cx.plot(T_series_control, Xt_noc[3, :], label="NOC")
cx.plot(T_series_sim, Xt_cl_noc[3, :], label="Leanred NOC")
cx.plot(T_series_sim, Xt_cl_noc_Kalman[3, :], linestyle='--', label="Leanred NOC w Kalman")
cx.set_ylabel("WTG Speed (PU)")
cx.legend(loc='lower left', fontsize=8, ncol=5)
dx = f.add_subplot(414)
dx.plot(T_series_sim, Ut[0, :], label="Open-loop")
dx.plot(T_series_control, Ut_noc[0, :], label="NOC")
dx.plot(T_series_sim, Ut_cl_noc[0, :], label="Leanred NOC")
dx.plot(T_series_sim, Ut_cl_noc_Kalman[0, :], linestyle='--', label="Leanred NOC w Kalman")
dx.set_xlabel("Time (s)")
dx.set_ylabel("Control Signal")
dx.legend(loc='lower right', fontsize=8, ncol=5)
f.tight_layout(pad=0.2, w_pad=0.2, h_pad=0.2)
plt.savefig("fig_case_comp_inter80_r0.pdf")