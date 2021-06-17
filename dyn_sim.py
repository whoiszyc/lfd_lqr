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
    nu = u.shape[0]
    nd = d.shape[0]
    Xt = np.zeros((nx, steps))
    Ut = np.zeros((nu, steps))
    Dt = np.zeros((nd, steps))
    if u.shape[1] == 1 and d.shape[1] == 1:
        for ii in range(steps):
            Xt[:, [ii]] = x0
            Ut[:, [ii]] = u[:, [0]]
            Dt[:, [ii]] = d[:, [0]]
            x1 = A @ x0 + Bu @ u[:, [0]] + Bd @ d[:, [0]]
            x0 = x1
    elif u.shape[1] == 1 and d.shape[1] > 1:
        for ii in range(steps):
            Xt[:, [ii]] = x0
            Ut[:, [ii]] = u[:, [0]]
            Dt[:, [ii]] = d[:, [ii]]
            x1 = A @ x0 + Bu @ u[:, [0]] + Bd @ d[:, [ii]]
            x0 = x1
    elif u.shape[1] > 1 and d.shape[1] == 1:
        for ii in range(steps):
            Xt[:, [ii]] = x0
            Ut[:, [ii]] = u[:, [ii]]
            Dt[:, [ii]] = d[:, [0]]
            x1 = A @ x0 + Bu @ u[:, [ii]] + Bd @ d[:, [0]]
            x0 = x1
    elif u.shape[1] > 1 and d.shape[1] > 1:
        for ii in range(steps):
            Xt[:, [ii]] = x0
            Ut[:, [ii]] = u[:, [ii]]
            Dt[:, [ii]] = d[:, [ii]]
            x1 = A @ x0 + Bu @ u[:, [ii]] + Bd @ d[:, [ii]]
            x0 = x1
    else:
        print("Input dimensions do not match")

    return Xt, Ut, Dt




def dyn_sim_feedback_discrete_time(A, Bu, Bd, x0, u, d, t_series, K):
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
    nu = u.shape[0]
    nd = d.shape[0]
    Xt = np.zeros((nx, steps))
    Ut = np.zeros((nu, steps))
    Dt = np.zeros((nd, steps))
    if d.shape[1] == 1:
        for ii in range(steps):
            Xt[:, [ii]] = x0
            Ut[:, [ii]] = K @ x0
            x1 = A @ x0 + Bu @ K @ x0 + Bd @ d[:, [0]]
            x0 = x1
    elif d.shape[1] > 1:
        for ii in range(steps):
            Xt[:, [ii]] = x0
            Ut[:, [ii]] = K @ x0
            x1 = A @ x0 + Bu @ K @ x0 + Bd @ d[:, [ii]]
            x0 = x1
    else:
        print("Disturbance dimensions do not match")

    return Xt, Ut, Dt
