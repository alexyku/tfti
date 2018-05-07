# following from https://github.com/susobhang70/shapley_value
import math
import bisect
import sys
import numpy as np
import random
from itertools import chain, combinations 


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

def n_choose_k(n,k):
    f = math.factorial
    return f(n) / f(k) / f(n-k)

def power_set_n(n):
    """ Generates number of all permutations up to length
    :param n depth to compute combinations. Defaults to all.
    :return chained iterable of power set of all combinations to depth
    """
    count = 0
    for k in range(n):
        count += n_choose_k(n, k)
    return count

def compute_shapley_value(C, players, characteristic_function, depth, N = None):
    """ Computes generalized Shapley Value
    :param C list of players to compute shapley value for
    :param players list of all players
    :param characteristic_function 
    :param depth to compute
    :param N power set of all player combinations. Recomputes if none
    :return shapley value for the group C of players
    """
    if (not isinstance(C, list)):
        C = [C]
        
    # n players
    n = len(players)
    if (N == None):
        tempList = list([i for i in range(n)])
        N = list(filter(lambda x: len(x) <= depth, power_set(tempList)))
        
    shapley = 0
    
    # for all T in N\C
    for T in N:
        if sum([i in T for i in C]) == 0:
            cmod = len(T)
            Cui = T[:]
            # insert each c in C in Cui in sorted order
            for c in C:
                bisect.insort_left(Cui,c)
            # flatten nested lists, if necessary
            if (any(isinstance(i, list) for i in Cui)):
                Cui = [item for sublist in Cui for item in sublist]
                
            if (len(Cui) <= depth):
                l = N.index(T)
                k = N.index(Cui)
                # |T|!(n-|T|-|C|)!/(n-|C|+1)
                pref = float(math.factorial(cmod) * math.factorial(n - cmod - len(C))) / float(math.factorial(n-len(C)+1))
                
                #sum (S in C) (-1)^(|C|-|S|)v(S union T)
                suff = 0
                C_power_set = power_set(C)
                for S in C_power_set:
                    S_union_T = set(S).union(T) 
                    k = N.index(sorted(list(S_union_T)))
                    # get index
                    tmp = math.pow(-1,len(C)-len(S))* (characteristic_function[k])
                    suff+=tmp
                    
                # subtract the empty set characteristic function
                temp = pref * suff
                shapley += temp
                
    print(f"Shapley for {C} is {shapley}")
    return shapley

def compute_shapley_values_1d(players, characteristic_function, depth=None):
    """
    :param charteristic_function contains all possible permutations of the value function
    """
    n = len(players)
    if (depth == None):
        depth = n
        

    tempList = list([i for i in range(n)])
    N = list(filter(lambda x: len(x) <= depth, power_set(tempList)))
    assert(len(characteristic_function) == len(N))
    
    # compute shapley values for all players
    shapley_values = []
    for i in range(n):
        print(f"Computing Shapley for {i}")
        shapley = compute_shapley_value(i, players, characteristic_function, depth, N=N)
        shapley_values.append(shapley)

    return shapley_values

def compute_shapley_values_2d(players, characteristic_function, depth=None):
    """
    :param charteristic_function contains all possible permutations of the value function
    """
    n = len(players)
    if (depth == None):
        depth = n
        

    tempList = list([i for i in range(n)])
    p_set = power_set(tempList)
    N = list(filter(lambda x: len(x) <= depth, p_set))
    N_tuples = list(filter(lambda x: len(x) == 2, p_set))

    # this is used for filtering the characteristic function when
    # removing a player from the game 
    indexed_characteristic_function = list(zip(N, characteristic_function))
    
    assert(len(characteristic_function) == len(N))
    
    # compute two dimensional interactions
    shapleys = []
    interactions = []
    
    for C in N_tuples:
        C_players = [players[i] for i in C]
        print(f"Computing 2d Shapley value for {C_players}")
        
        # Compute shapley value of element 1
        y_i_j = compute_shapley_value(C, players, characteristic_function, depth, N = N)
        shapleys.append((C_players, y_i_j))
        
        # compute shapley value of element 1 in subgame without element 2
        characteristic_function_notj = filter(lambda x: players[C[1]] not in x[0], indexed_characteristic_function)
        characteristic_function_notj = list(map(lambda x: x[1], characteristic_function_notj))
        players_noj = players[:]
        players_noj.remove(C_players[1])
        y_i_notj = compute_shapley_value(players_noj.index(C_players[0]), players_noj, characteristic_function_notj, depth)

        # compute shapley value of element 2 in subgame without element 1
        characteristic_function_noti = filter(lambda x: players[C[0]] not in x[0], indexed_characteristic_function)
        characteristic_function_noti = list(map(lambda x: x[1], characteristic_function_noti))
        players_noi = players[:]
        players_noi.remove(C_players[0])
        y_j_noti = compute_shapley_value(players_noi.index(C_players[1]), players_noi, characteristic_function_noti, depth)
        interaction = y_i_j - y_i_notj - y_j_noti
        interactions.append((C_players, interaction))

    return (shapleys, interactions)
