# following from https://github.com/susobhang70/shapley_value
from itertools import combinations
import math
import bisect
import sys
import numpy as np
import random

def power_set(players, depth = None):
    """ Generates list of all permutations for list List
    :param list of all players. can be string, int
    :param depth depth to compute combinations. Defaults to all.
    :return power set of all combinations to depth
    """
    if (depth == None):
        depth = len(players)
    PS = [list(j) for i in range(depth+1) for j in combinations(players, i)]
    return PS

def compute_shapley_values(players, characteristic_function):
    """
    :param charteristic_function contains all possible permutations of the value function
    """
    n = len(players)
    tempList = list([i for i in range(n)])
    N = power_set(tempList)
    R_orderings = power_set(players)
    assert(len(characteristic_function) == len(R_orderings))
    
    shapley_values = []
    for i in range(n):
        shapley = 0
        for j in N:
            if i not in j:
                cmod = len(j)
                Cui = j[:]
                bisect.insort_left(Cui,i)
                l = N.index(j)
                k = N.index(Cui)
                temp = float(float(characteristic_function[k]) - float(characteristic_function[l])) *\
                           float(math.factorial(cmod) * math.factorial(n - cmod - 1)) / float(math.factorial(n))
                shapley += temp

        cmod = 0
        Cui = [i]
        k = N.index(Cui)
        temp = float(characteristic_function[k]) * float(math.factorial(cmod) * math.factorial(n - cmod - 1)) / float(math.factorial(n))
        shapley += temp

        shapley_values.append(shapley)

    return shapley_values