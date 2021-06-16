import numpy as np


def dyn_sim_discrete_time(A, Bu, Bd, x0, u, d, t_series):
    """
    Simulate discrete-time ODE
    Args:
        A: discrete-time A
        Bu: discrete-time B for control
        Bd: discrete-time B for disturbance
        x0: Initial condition in numpy array nx*1
        u: Control signal in numpy array, [[u]] if constrant
        d: Disturbance signal in numpy array, [[d]] if constrant
        t_series: Time series
    Returns: Integration results in numpy array
    """
    steps = len(t_series)
    nx = x0.shape[0]
    xt = np.zeros((nx, steps))
    if u.shape[1] == 1 and d.shape[1] == 1:
        for ii in range(steps):
            xt[:, [ii]] = x0
            x1 = A @ x0 + Bu @ u[:, [0]] + Bd @ d[:, [0]]
            x0 = x1
    elif u.shape[1] == 1 and d.shape[1] > 1:
        for ii in range(steps):
            xt[:, [ii]] = x0
            x1 = A @ x0 + Bu @ u[:, [0]] + Bd @ d[:, [ii]]
            x0 = x1
    elif u.shape[1] > 1 and d.shape[1] == 1:
        for ii in range(steps):
            xt[:, [ii]] = x0
            x1 = A @ x0 + Bu @ u[:, [ii]] + Bd @ d[:, [0]]
            x0 = x1
    elif u.shape[1] > 1 and d.shape[1] > 1:
        for ii in range(steps):
            xt[:, [ii]] = x0
            x1 = A @ x0 + Bu @ u[:, [ii]] + Bd @ d[:, [ii]]
            x0 = x1
    else:
        print("Input dimensions do not match")

    return xt




