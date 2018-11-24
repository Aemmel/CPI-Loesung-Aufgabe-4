# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

from module_simplex import simplex
from himmelblau import himmelblau

start_1 = [3, 2]
start_2 = [-2.805118, 3.131312]
start_3 = [-3.779310, -3.283186]
start_4 = [3.584428, -1.848126]

x_min, y_min, N_max = simplex(himmelblau, start_4, 1000, 0.01)

print(y_min)
print(x_min)