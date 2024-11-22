#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import lineStyles


def f(x):
    if x >= 0 or x < 1 / 3:
        return 1
    elif x >= 1 / 3 or x < 2:
        return 2
    elif x >= 2 / 3 or x < 1:
        return 3


def c(x):
    return 1


def c2(x, a):

    return 1 + a * x


def G(x, x0):
    fun = 0
    if x < x0 and 0 < x:
        fun = (1 - x0) * x
    elif x0 < x and x < 1:
        fun = x0 * (1 - x)

    return fun


def yquad(x, NN):
    hh = 1.0 / NN
    sumy = 0.0
    xh = hh / 2.0
    for i in range(NN):
        sumy += G(x, xh) * f(xh)
        xh += hh

    return sumy * hh


# def yexact(x):
#    return -x + math.log(1 + x)/math.log(2)

h_values = [0.2, 0.1, 0.03, 0.05, 0.01]
h_smooth = 0.01
y0 = []

a_values = [0.01, 0.1, 0.2, 0.5]

color_count = 0
colors = ["r", "b", "m", "c", "k", "g", "y"]


for h in h_values:
    N = int((1 / h) - 1)
    print(N)
    x_N = np.linspace(0, 1, N + 2)
    f_N = [f(x) for x in x_N]
    A = np.zeros((N + 2, N + 2))
    K = np.zeros((N + 2, N + 2))

    for i in range(N + 2):
        A[i, i] = 1.0 / h
        coeff = c(x_N[i])
        K[i, i] = A[i, i] * coeff
        if i > 0:
            A[i, i - 1] = -1.0 / h
            K[i, i - 1] = coeff * A[i, i - 1]

    K = A.transpose() @ K

    y_N = np.linalg.solve(K[1 : N + 1, 1 : N + 1], f_N[1 : N + 1])
    plt.plot(
        x_N[1 : N + 1], y_N, linewidth=0.5, color=colors[color_count], label=f"h = {h}"
    )
    color_count = color_count + 1
    if h == 0.01:
        y0 = y_N

print(y0)
color_count = 0

for a in a_values:
    N = int(1 / h_smooth - 1)
    x_N = np.linspace(0, 1, N + 2)
    f_N = [f(x) for x in x_N]
    A = np.zeros((N + 2, N + 2))
    K = np.zeros((N + 2, N + 2))

    for i in range(N + 2):
        A[i, i] = 1.0 / h
        coeff = c2(x_N[i], a)
        K[i, i] = A[i, i] * coeff
        if i > 0:
            A[i, i - 1] = -1.0 / h
            K[i, i - 1] = coeff * A[i, i - 1]

    K = A.transpose() @ K

    y_N = np.linalg.solve(K[1 : N + 1, 1 : N + 1], f_N[1 : N + 1])
    plt.plot(
        x_N[1 : N + 1],
        y_N,
        linewidth=1,
        color=colors[color_count],
        label=f"a = {a}",
        linestyle="--",
    )
    color_count = color_count + 1


G_N = [G(x, 0) for x in x_N]

NQ = 100
xq_N = np.linspace(0, 1, NQ + 2)
yquad = np.array([yquad(x, NQ) for x in xq_N])

plt.plot(xq_N, yquad, label="Exact (greens)", color="r", linestyle="dotted")

plt.legend()
plt.show()
