import numpy as np
import cvxpy as cp
import time

def dyn_noc(A, Bu, Bd, x0, d, t_series, verbose_flag=True):
    """
    Simulate discrete-time ODE
    Args:
        A: discrete-time A
        Bu: discrete-time B for control
        Bd: discrete-time B for disturbance
        x0: Initial condition in numpy array nx*1
        xf: Final condition
        d: Disturbance signal in numpy array, [[d]] if constrant
        t_series: Time series
    Returns: Integration results in numpy array
    """
    Tf = len(t_series)
    nx = len(x0)

    # define variable
    u = cp.Variable((Bu.shape[1], Tf))
    # x = cp.Variable((nx, Tf + 1))

    # begin formulation
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~begin formulation~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    time_start = time.time()
    cost = 0
    constr = []
    # state and control constraints
    for t in range(Tf):
        cost += cp.sum_squares(u[:, t])
        x1 = A @ x0 + Bu @ u[:, [t]] + Bd * d[0]
        constr += [u[:, t] <= 0.2, u[:, t] >= -0.2]
        constr += [x1[0] <= 0.5, x1[0] >= -0.5]
        x0 = x1
    # control RoC constraint
    for t in range(1, Tf):
        constr += [u[:, t] - u[:, t - 1] <= 0.006]
        constr += [u[:, t] - u[:, t - 1] >= -0.006]

    # sums problem objectives and concatenates constraints.
    constr += [u[:, 0] == 0, u[:, -1] == 0]
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~complete formulation using {}~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~".format(time.time() - time_start))

    # solve the problem
    time_start = time.time()
    problem = cp.Problem(cp.Minimize(cost), constr)
    problem.solve(solver=cp.GUROBI, verbose=verbose_flag)
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~complete problem-solving using {}~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~".format(time.time() - time_start))

    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Control cost is {}~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~".format(cost.value))
    # in cvxpy, variable.value is numpy.ndarray
    return u.value


def dyn_noc_ode_constr(A, Bu, Bd, x0, d, t_series, verbose_flag=True):
    """
    Simulate discrete-time ODE
    Args:
        A: discrete-time A
        Bu: discrete-time B for control
        Bd: discrete-time B for disturbance
        x0: Initial condition in numpy array nx*1
        xf: Final condition
        d: Disturbance signal in numpy array, [[d]] if constrant
        t_series: Time series
    Returns: Integration results in numpy array
    """
    Tf = len(t_series)
    nx = len(x0)

    # define variable
    u = cp.Variable((Bu.shape[1], Tf))
    x = cp.Variable((nx, Tf + 1))

    # begin formulation
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ begin formulation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    time_start = time.time()
    cost = 0
    constr = []
    # initial condition constraint
    constr += [x[:, [0]] == x0]
    # ODE, state and control constraint
    for t in range(Tf):
        cost += cp.sum_squares(u[:, t])
        constr += [x[:, [t+1]] == A @ x[:, [t]] + Bu @ u[:, [t]] + Bd * d[0]]
        constr += [u[:, t] <= 0.2, u[:, t] >= -0.2]
        constr += [x[0, t] <= 0.5, x[0, t] >= -0.5]
    # control RoC constraint
    for t in range(1, Tf):
        constr += [u[:, t] - u[:, t - 1] <= 0.006]
        constr += [u[:, t] - u[:, t - 1] >= -0.006]
    # sums problem objectives and concatenates constraints.
    constr += [u[:, 0] == 0, u[:, -1] == 0]
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ complete formulation using {} ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~".format(time.time() - time_start))

    # solve the problem
    time_start = time.time()
    problem = cp.Problem(cp.Minimize(cost), constr)
    problem.solve(solver=cp.GUROBI, verbose=verbose_flag)
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ complete problem-solving using {} ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~".format(time.time() - time_start))

    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Control cost is {} ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~".format(cost.value))
    # in cvxpy, variable.value is numpy.ndarray
    return u.value

if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.linalg import solve_discrete_are, eig
    import cvxpy as cp
    from dyn_sim import dyn_sim_discrete_time, dyn_sim_feedback_discrete_time
    import sys
    sys.setrecursionlimit(100000000)  # 10000 is an example, try with different values

    A = np.array([
        [0.9999, 0.2926, 0.0071, 0.1544],
        [-0.0008, 0.9512, 0.0464, -0.0000],
        [-0.0317, -0.0048, 0.9048, -0.0025],
        [0, 0, 0, 0.9972]
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
    x0 = np.array([[0], [0], [0], [0]])  # Initial condition

    # NOC control
    # dyn_noc_ode_constr leads to much smaller Compilation time for cvxpy
    U_noc = dyn_noc_ode_constr(A, Bu, Bd, x0, [0.1], T_series_control)  # solve NOC problem
    Xt_noc, Ut_noc, _ = dyn_sim_discrete_time(A, Bu, Bd, x0, U_noc, np.array([[0.1]]), T_series_control)  # simulate NOC

    # open loop
    Xt, Ut, _ = dyn_sim_discrete_time(A, Bu, Bd, x0, np.array([[0]]), np.array([[0.1]]), T_series_control)

    # compare all cases
    f = plt.figure(figsize=(15, 9))
    ax = f.add_subplot(211)
    ax.plot(T_series_control, Xt[0, :], label="Open-loop")
    ax.plot(T_series_control, Xt_noc[0, :], label="NOC")
    ax.set_ylabel("Frequency (Hz)")
    ax.legend(loc='lower right', fontsize=8, ncol=5)
    bx = f.add_subplot(212)
    bx.plot(T_series_control, Ut_noc[0, :], label="NOC")
    bx.set_ylabel("Control")
    bx.legend(loc='lower right', fontsize=8, ncol=5)
