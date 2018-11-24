# -*- coding: utf-8 -*-
import numpy as np
from copy import copy

# what will the simplex look like?
# it will be an array (nxn), which looks like this:
# [ [x_1_1, ..., x_n_1, F(x_1_1, ..., x_n_1)],
#     .                                .
#     .                                .
#     .                                .   
#   [x_1_n, ..., x_n_n, F(x_1_n, ..., x_n_n)] ]


# Sort the simplex
# Not the prettiest, but it works
def sort_simplex(simp):
    while not all(simp[i][-1] <= simp[i+1][-1] for i in range(len(simp) - 1)):
        for i in range(len(simp)-1):
            if simp[i][-1] > simp[i+1][-1]:
                # Standard swapping doesn't work for some reason...
                # simp[i], simp[i+1] = simp[i+1], simp[i]
                temp = copy(simp[i])
                simp[i] = simp[i+1]
                simp[i+1] = temp
    
    return simp

# Helper functions for the operations on the simplexes
def reflect(simp, a, fhandle):
    """ Reflect simplex
    Input:
        simp ((n+1)x(n+1) array) - simplex to reflect, where the last element 
            is the maximum of the simplex
        a (float) - alpha, reflection factor
        fhandle (function handle) - function handle that takes two-tuple as arg
    Output:
        ndarray - 1x(n+1) vector with new simplex point
    """
    # Calculate middle point
    x_s = np.empty(len(simp) - 1)
    for i in range(len(x_s)):
        x_s[i] = np.sum(simp[:-1], axis=0)[i] / len(x_s)
    
    # Reflect
    x_r = np.empty(len(x_s))
    for i in range(len(x_s)):
        x_r[i] = x_s[i] - a * (simp[-1][i] - x_s[i])
    
    y_r = fhandle(x_r)
    
    return np.append(x_r, y_r)
    
def expand(simp, g, fhandle):
    """ Expand simplex
    Input:
        simp ((n+1)x(n+1) array) - simplex to expand, where the last element 
            is the new point x_r
        g (float) - gamma, expansion factor
        fhandle (function handle) - function handle that takes two-tuple as arg
    Output:
        ndarray - 1x(n+1) vector with new simplex point
    """
    # Calculate middle point
    x_s = np.empty(len(simp) - 1)
    for i in range(len(x_s)):
        x_s[i] = np.sum(simp[:-1], axis=0)[i] / len(x_s)    
    
    #expand
    x_e = np.empty(len(x_s))
    for i in range(len(x_s)):
        x_e[i] = x_s[i] + g*(simp[-1][i] + x_s[i])
    
    y_e = fhandle(x_e)
    
    return np.append(x_e, y_e)

def contract(simp, b, fhandle):
    """ contract simplex
    Input:
        simp ((n+1)x(n+1) array) - simplex to contract, where the last element
        is the maximum of the simplex
        b (float) - beta, contraction factor
        fhandle (function handle) - function handle that takes two-tuple as arg
    Output:
        ndarray - 1x(n+1) vector with new simplex point
    """
    # Calculate the middle point
    x_s = np.empty(len(simp) - 1)
    for i in range(len(x_s)):
        x_s[i] = np.sum(simp[:-1], axis=0)[i] / len(x_s)
    
    #contract
    x_c = np.empty(len(x_s))
    for i in range(len(x_s)):
        x_c[i] = x_s[i] + b*(simp[-1][i] - x_s[i])
    
    y_c = fhandle(x_c)
    
    return np.append(x_c, y_c)

def shrink(simp, s, fhandle):
    """ shrink simplex
    Input:
        simp ((n+1)x(n+1) array) - simplex to shrink, with the standard order
        s (float) - sigma, shrink factor
        fhandle (function handle) - function handle that takes tow-tuple as arg
    """    
    for i_s in range(1, len(simp)):
        for i in range(len(simp[i_s] - 1)):
            simp[i_s][i] = (simp[0][i] + simp[i_s][i]) / s
        simp[i_s][-1] = fhandle((simp[i_s][0], simp[i_s][1]))
    
    # Return nothing, because simplex is already shrunk

def check_precision(simp, p, fhandle):
    """ Check if the simplex is precise enough
    Input:
        simp ((n+1)x(n+1) array) - sorted simplex
        p (float) - precision factor
        fhandle (function handle) - used function
    Output:
        bool - is it close enough?
    """
    for i_s in range(len(simp) - 1):
        if simp[i_s][-1] > p:
            return False
        for i in range(len(simp[i_s]) - 1):
            if abs(simp[i_s][i] - simp[i_s + 1][i]) > p:
                return False
    
    # If no distance is lager than p or no y value, then it is precise enough
    return True

def simplex(fhandle, x_start, N_max, p):
    """ SIMPLEX Minimumssuche mittels des Downhill-Simplex Verfahrens.

    Beispiel
    --------

    fhandle = himmelblau
    x_start = [0 0]
    N_max   = 1e3
    p       = 1e-15
    x_min, f_min, N = simplex(fhandle, x_start, N_max, p)

    Argumente
    ---------
    fhandle : function
        Die zu minimierende Funktion

    x_start : n-tuple with starting point (with n being dimension)
        Startpunkt des Simplex

    N_max : int
        Maximale Anzahl an Iterationen

    p : float
        Genauigkeit in x oder f

    Output
    ------
    x_min : float
        Punkt (x_1, ..., x_n) des Funktionsminimums (n-tupel)

    f_min : float
        Funktionsminimum

    N_max : int
        Anzahl der benötigten Schritte
    """

    #==================================================
    # Initialisierung
    #==================================================

    # Die Skalierungsfaktoren des Downhill-Simplex Verfahrens
    alpha_  = 1.0  # empfohlener Faktor für die Spiegelung
    beta_   = 0.5  # empfohlener Faktor für die Kontraktion
    gamma_  = 2.0  # empfohlener Faktor für die Expansion
    sigma_ = 2.0   # Recommended factor for shrinking
    lambda_ = 0.1  # empfohlene Größe des Startsimplex
    
    # Create starting simplex
    simp = np.empty((len(x_start) + 1, len(x_start) + 1))
    
    # Initialize starting point
    for i in range(len(x_start)):
        simp[0][i] = x_start[i]
    simp[0][-1] = fhandle(simp[0][:-1])
    
    # Initialize the rest of the n points
    for i_s in range(1, len(simp)):
        # Create vector of length lambda_ in direction i_s
        e = np.zeros(len(simp) - 1)
        e[i_s - 1] = lambda_
        for i in range(len(e)):
            simp[i_s][i] = simp[0][i] + e[i]
        simp[i_s][-1] = fhandle(simp[i_s][:-1])
    
    sort_simplex(simp)
    
    # Main loop
    N = 0
    while N < N_max or not check_precision(simp, p, fhandle):  
        # Never change the original simplex during the algorithm! always 
        # save the returned values of the simplex modifcations methods as
        # individual points
        
        # Fist reflect the maximum
        point_r = reflect(simp, alpha_, fhandle)
        
        # Is the new value better than the original lowest?
        # Following it will always be: better <=> better or equal
        if point_r[-1] <= simp[0][-1]:
            # Yes? Then expand
            point_e = expand(np.append(simp[:-1], [point_r], axis=0), 
                    gamma_, fhandle)
            
            # point_e better?
            if point_e[-1] <= point_r[-1]:
                # Yes? Then take point_e as new point
                simp = np.append(simp[:-1], [point_e], axis=0)
            else:
                # No? Then take point_r as new point
                simp = np.append(simp[:-1], [point_r], axis=0)
        # Is the new value better than the worst old one
        #elif point_r[-1] <= simp[-1][-1]:
        #simp = np.append(simp[:-1])
        
        # Because of n-dimensionality, I need a Lööp
        else:
            # Check at what entry the new one is better than the next best one
            for i_s in range(1, len(simp) - 1):
                if point_r[-1] <= simp[i_s][-1]:
                    simp = np.append(simp[:-1], [point_r], axis=0)
                    break
                # Need a special case for the last value
                if i_s == len(simp) - 2:
                    # No value up to (but including) the second to last, was
                    # worse than the new value, so check if it is at least
                    # better than the last value
                    if point_r[-1] <= simp[-1][-1]:
                        # It is better than the last
                        simp = np.append(simp[:-1], [point_r], axis=0)
                        sort_simplex(simp)
                    # Esle do nothing and contract right away
                    
                    point_c = contract(simp, beta_, fhandle)
                    
                    # I point_c better?
                    if point_c[-1] <= simp[-1][-1]:
                        # Yes? Set as new value
                        simp = np.append(simp[:-1], [point_c], axis=0)
                    else:
                        shrink(simp, sigma_, fhandle)
                    
                    # If done, break out of loop (shouldnt matter, but still..)
                    break
        
        # At end of every iteration: set checking conditions right to ensure
        # correct evaluation of breaking conditions
        N += 1
        sort_simplex(simp)
        
    # Out of the loop => minimum was found or it took too long
    return (simp[0][:-1], simp[0][-1], N)
        
    
    
    
    
    
    
    
    
