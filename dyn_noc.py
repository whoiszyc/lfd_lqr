import numpy as np
import cvxpy as cp

def dyn_noc(A, Bu, Bd, x0, xf, d, t_series):
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
    u = cp.Variable((Bu.shape[1], Tf))
    # x = cp.Variable((nx, Tf + 1))

    cost = 0
    constr = []
    for t in range(Tf):
        cost += cp.sum_squares(u[:, t])
        x1 = A @ x0 + Bu @ u[:, [t]] + Bd * d[0]
        constr += [u[:, t] <= 0.2, u[:, t] >= -0.2]
        constr += [x1[0] <= 0.5, x1[0] >= -0.5]
        x0 = x1

    for t in range(1, Tf):
        constr += [u[:, t] - u[:, t - 1] <= 0.006]
        constr += [u[:, t] - u[:, t - 1] >= -0.006]

    # sums problem objectives and concatenates constraints.
    constr += [u[:, 0] == 0, u[:, -1] == 0]
    problem = cp.Problem(cp.Minimize(cost), constr)
    problem.solve(solver=cp.GUROBI, verbose=True)

    print("Control cost is {}".format(cost.value))
    # in cvxpy, variable.value is numpy.ndarray
    return u.value




