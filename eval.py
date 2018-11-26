# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

from module_simplex import simplex
from himmelblau import himmelblau


N_max = 1000
p = 1e-12

start = np.array([[1, 1], [-1, 1], [1, -1], [-1, -1]])
x_min = np.empty((4, 2))
y_min = np.empty(4)
N = np.empty(4)

for i in range(4):
    x_min[i], y_min[i], N[i] = simplex(himmelblau, start[i], N_max, p)

for i in range(4):
    print("For starting point: " + str(start[i]) + ", maximal Steps "
            + str(N_max) + " and upper bound " + str(p) + " the results are:")
    print("Minimum found by F(" + str(x_min[i]) + ") = " + str(y_min[i])
            + "\t with a step count of " + str(N[i]))
    print("\n----------------------------------------\n")

# Plot y_min over N_max with p = 0
p = 0
N_max = np.logspace(0.5, 4, 500)
y_min = np.empty(len(N_max))

for i in range(len(N_max)):
    x_min[0], y_min[i], N[0] = simplex(himmelblau, start[0], N_max[i], p)
    
plt.semilogx(N_max, y_min, "b-", label="Min. over $N_{max}$")
plt.legend(loc="best")
plt.show()
