import numpy as np
import matplotlib.pyplot as plt

# Forward Euler Method
w = 1
T = (2 * np.pi) / w
h = T / 100
to = 0
tf = 10 * T

omega_vals = np.zeros(1000)  # 1000 values if tf = 10T and h = T/100
theta_vals = np.zeros(1000)  # 1000 values if tf = 10T and h = T/100
t_vals = np.arange(to, tf, h)

omega_o = 0
theta_o = 0.1

for i in range(len(omega_vals)):
    if i == 0:
        omega_vals[i] = omega_o
        theta_vals[i] = theta_o
    else:
        omega_vals[i] = omega_vals[i - 1] + (h * (-np.sin(theta_vals[i - 1])))
        theta_vals[i] = theta_vals[i - 1] + (h * omega_vals[i - 1])

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 2)
plt.plot(
    t_vals,
    omega_vals,
    color="purple",
    linestyle="-",
    linewidth=2,
    label=r"$\omega(t)$ - Angular Velocity (rad/s)",
)
plt.plot(
    t_vals,
    theta_vals,
    color="green",
    linestyle="-",
    linewidth=2,
    label=r"$\theta(t)$ - Angular Position (rad)",
)
line1 = plt.axvline(x=tf, color="red", linestyle="--")
line1.set_label(r"$t = 10T$")

plt.xlabel("Time (s)", fontsize=12)
plt.ylabel("Angular Position (rad) &  Angular Velocity (rad/s)", fontsize=12)
plt.title("Simple Pendulum Motion Over Time (Forward Method)", fontsize=14)
plt.legend(fontsize=10, loc="best")
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

# Backward Euler Method
omega = 1
T = (2 * np.pi) / omega
dt = T / 20
to = 0
tf = 10 * T
h = T / 100

t_vals = np.arange(to, tf, h)
theta = np.zeros(len(t_vals))
Omega = np.zeros(len(t_vals))
theta_0 = 0.1
Omega_0 = 0


def newton_method(theta_k, Omega_k, h, omega, eps=1e-9, iterations=10000):
    theta_k1 = theta_k
    for i in range(iterations):
        f = theta_k1 - theta_k - h * Omega_k + h**2 * omega**2 * np.sin(theta_k1)
        fd = 1 + h**2 * omega**2 * np.cos(theta_k1)
        theta_k1_new = theta_k1 - (f / fd)
        if abs(theta_k1_new - theta_k1) < eps:
            break
        theta_k1 = theta_k1_new
    return theta_k1


for i in range(len(t_vals)):
    theta_new = newton_method(theta_0, Omega_0, h, omega)
    Omega_new = Omega_0 - h * omega**2 * np.sin(theta_new)
    theta[i] = theta_new
    Omega[i] = Omega_new
    theta_0 = theta_new
    Omega_0 = Omega_new

plt.subplot(1, 2, 1)
plt.plot(
    t_vals,
    Omega,
    color="purple",
    linestyle="-",
    linewidth=2,
    label=r"$\omega(t)$ - Angular Velocity (rad/s)",
)
plt.plot(
    t_vals,
    theta,
    color="green",
    linestyle="-",
    linewidth=2,
    label=r"$\theta(t)$ - Angular Position (rad)",
)
line2 = plt.axvline(x=tf, color="red", linestyle="--")
line2.set_label(r"$t = 10T$")

plt.xlabel("Time (s)", fontsize=12)
plt.ylabel("Angular Position (rad) & Angular Velocity (rad/s)", fontsize=12)
plt.title("Simple Pendulum Motion Over Time (Backward Method)", fontsize=14)
plt.legend(fontsize=10, loc="best")
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

plt.tight_layout()
plt.show()
