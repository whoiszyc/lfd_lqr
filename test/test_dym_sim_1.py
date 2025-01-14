import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

def int_pendulum_sim(theta_init, t, L=1, m=1, b=0, g=9.81):
    theta_dot_1 = theta_init[1]
    theta_dot_2 = -b/m*theta_init[1] - g/L*np.sin(theta_init[0])
    return theta_dot_1, theta_dot_2

# Input constants
m = 1 # mass (kg)
L = 1 # length (m)
b = 0 # damping value (kg/m^2-s)
g = 9.81 # gravity (m/s^2)
delta_t = 0.02 # time step size (seconds)
t_max = 10.0 # max sim time (seconds)
theta1_0 = np.pi/2 # initial angle (radians)
theta2_0 = 0 # initial angular velocity (rad/s)
theta_init = (theta1_0, theta2_0)
# Get timesteps
t = np.linspace(0.0, t_max, int(t_max/delta_t))

theta_vals_int = integrate.odeint(int_pendulum_sim, theta_init, t)


f = plt.figure()
ax = f.add_subplot(111)
plt.plot(t, theta_vals_int)