from copy import deepcopy
import itertools
from collections import defaultdict
import numpy as np
import pandas as pd
from pyvis.network import Network
import networkx as nx

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

"""
If the best known solution is not improved for 20 iterations the search is terminated.
This is was empirically found in a few test runs.
Tabu queue length of 5 was found good.
"""

MIN_PER_LOCATION = 4

# extra data calculated from raw data
data2 = {}

# raw data from input file
data = None


def g(s):
    """
    global component
    each student has r characteristics
    c[i,j] = student i, ability j
    """
    # random accademic abilities data
    c = np.random.rand(91, 4)
    # max in each row
    cj = np.amax(c, axis=1)
    return min(cj) * sum(cj)


def gender_score(s):
    """return score in [0,1] where 1 is equal parts, 0 is homogenous"""
    global data
    g = data['Gender']
    sum_dev = 0
    for c in s:
        c = np.array(list(c))
        sum_dev += abs(sum(g[c] == 'M') / c.size - 0.5)
    avg_dev = sum_dev / len(s)
    return 1 - avg_dev / 0.5


def locations():
    global data, data2
    return data2.setdefault('locations', data['Location'].unique())


def location_score(s):
    all_locations = data['Location']
    vc = pd.value_counts(all_locations)
    per_class = [pd.value_counts(all_locations[x]) for x in s]
    pc = pd.DataFrame(per_class)
    num_classes = len(s)
    target = vc.map(lambda x: max(x / num_classes, MIN_PER_LOCATION))
    above = pc - target
    bad = above[above > 0].sum().sum()
    below = pc[pc < MIN_PER_LOCATION]
    bad += below.sum().sum()
    return -bad


def friend_enemy_score(s):
    global data
    cols = itertools.product(('Friend', 'Enemy'), '123')
    cols = list(map(''.join, cols))
    # avoid np.NaN since replace makes it float
    data.fillna({x: 'NA' for x in cols}, inplace=True)
    lookup = data.set_index('Student Name')['No']
    lookup['NA'] = -1
    data.replace({c: lookup for c in cols}, inplace=True)
    s_back = {n: i for i, x in enumerate(s) for n in x}
    # now replace each student id with her group id
    data.replace({c: s_back for c in cols}, inplace=True)
    # add student own group id
    b = pd.Series(s_back)
    data['Group'] = b
    score = [len(data[data[x] == data['Group']]) for x in cols]
    return sum(score[:3]) - sum(score[3:])


def behavior_score(s):
    global data
    s = pd.DataFrame(s)
    s.replace(data.TroubleMaker, inplace=True)
    t = s.sum(axis=1)
    return -t.var()


def inclusion_score(s):
    num_groups = len(s)
    cols = [f'Class {x} OK' for x in range(1, num_groups + 1)]
    s_back = {n: i for i, x in enumerate(s) for n in x}
    data['Group'] = pd.Series(s_back)

    def bad(x):
        return (data[cols[x]] == 0) & (data.Group == x)

    bad_rows = [data.loc[bad(x)] for x in range(num_groups)]
    return -sum(map(len, bad_rows))


def f(s):
    gs = gender_score(s)
    ls = location_score(s)
    fs = friend_enemy_score(s)
    bs = behavior_score(s)
    ns = inclusion_score(s)
    return 0.8 * gs + ls + 0.5 * fs + bs + ns


def apply(s, move):
    [(i, x), (j, y)] = move
    s[i].remove(x)
    s[i].add(y)
    s[j].remove(y)
    s[j].add(x)


def undo(s, move):
    [(i, x), (j, y)] = move
    apply(s, [(i, y), (j, x)])


def f(s, move):
    apply(s, move)
    v = f(s)
    undo(s, move)
    return v


def find_max(s, neighbors):
    best_move = neighbors.pop()
    best_value = f(s, best_move)
    for move in neighbors:
        v = f(s, move)
        if best_value < v:
            best_value = v
            best_move = move
    s1 = deepcopy(s)
    apply(s1, best_move)
    return s1, best_move


def tabu_search_sap(n, s, f):
    """
    Student Assignment Problem
    n students
    s initial partition
    f state evaluation function
    """
    delta = 6
    # groups sizes
    k = list(map(len, s))
    m = len(k)
    # INITIALIZE
    sbest = deepcopy(s)
    tc = 0
    tcbest = tc
    # defaults to non-tabu timestamp
    t = defaultdict(lambda: tc - delta)
    # STOP after X iterations
    while tc - tcbest > 20:
        # GENERATE
        neighbors = []
        for i in range(m):
            for j in range(m):
                if i != j:
                    for x in s[i]:
                        for y in s[j]:
                            move = [(i, x), (j, y)]
                            tabu = tc - t[x] < delta or tc - t[y] < delta
                            asp = f(s, move) > f(sbest)
                            if not tabu or asp:
                                neighbors += move
        # SELECT
        smax, best_move = find_max(s, neighbors)
        # TEST
        if f(smax) > f(sbest):
            sbest = deepcopy(smax)
            tcbest = tc
        # UPDATE
        x, y = best_move[0][1], best_move[1][1]
        t[x] = t[y] = tc
        tc += 1
        s = smax
    return sbest


def tabu(filename, num_groups):
    global data
    # YOU MUST PUT sheet_name=None TO READ ALL CSV FILES IN YOUR XLSM FILE
    data = pd.read_excel(filename, sheet_name='StudentInfo', skiprows=3)
    n = data.values.shape[0]
    print(data.keys())
    s = np.arange(n)
    np.random.shuffle(s)
    s = np.array_split(s, num_groups)
    s = list(map(set, s))
    return tabu_search_sap(n, s, f)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    tabu('demo.xlsm', 4)

    net = Network()
    nx_graph = nx.cycle_graph(10)
    nx_graph.nodes[1]['title'] = 'Number 1'
    nx_graph.nodes[1]['group'] = 1
    nx_graph.nodes[3]['title'] = 'I belong to a different group!'
    nx_graph.nodes[3]['group'] = 10
    nx_graph.add_node(20, size=20, title='couple', group=2)
    nx_graph.add_node(21, size=15, title='couple', group=2)
    nx_graph.add_edge(20, 21, weight=5)
    nx_graph.add_node(25, size=25, label='lonely', title='lonely node', group=3)
    nt = Network('500px', '500px')
    # populates the nodes and edges data structures
    nt.from_nx(nx_graph)
    nt.show('nx.html')
