from multiprocessing import Pool, cpu_count
from copy import deepcopy
import itertools
import functools
from operator import itemgetter
import numpy as np
import pandas as pd
from pyvis.network import Network

# import networkx as nx

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
# cache of values calculated from s
s2 = {}

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
    s = pd.DataFrame(s)
    s.replace(g, inplace=True)
    s.replace({'M': -1, 'F': 1}, inplace=True)
    index = s.mean(axis=1).abs().mean()
    return 1 - index


def locations():
    global data, data2
    return data2.setdefault('locations', data['Location'].unique())


def location_score(s):
    """
    return How many students are not according to location constraint.
    That is, distributed evenly, but 4 and above from each location, zeros allowed.
    :param s:
    :return:
    """
    all_locations = data['Location']
    vc = pd.value_counts(all_locations)
    per_class = [pd.value_counts(all_locations[list(x)]) for x in s]
    pc = pd.DataFrame(per_class)
    num_classes = len(s)
    target = vc.map(lambda x: max(x / num_classes, MIN_PER_LOCATION))
    above = pc - target
    bad = above[above > 0].sum().sum()
    below = pc[pc < MIN_PER_LOCATION]
    bad += below.sum().sum()
    return -bad


def fe_cols():
    global data
    cols = itertools.product(('Friend', 'Enemy'), '123')
    cols = list(map(''.join, cols))
    # avoid np.NaN since replace makes it float
    data.fillna({x: 'NA' for x in cols}, inplace=True)
    lookup = data.set_index('Student Name')['No']
    lookup['NA'] = -1
    data.replace({c: lookup for c in cols}, inplace=True)
    return cols


def friend_enemy_score(s):
    """
    :param s: Current division to groups
    :return:
    How many friend and enemy requests not satistfied
    """
    global data, data2, s2
    cols = data2.setdefault('fe_cols', fe_cols())
    cols_groups = [f'{c}_group' for c in cols]
    s_back = s2.setdefault('s_back', {n: i for i, x in enumerate(s) for n in x})
    # now replace each student id with her group id
    data[cols_groups] = data.replace({c: s_back for c in cols})[cols]
    # add student own group id
    data['Group'] = pd.Series(s_back)
    score = [len(data[data[x] == data['Group']]) for x in cols]
    return sum(score[:3]) - sum(score[3:])


def behavior_score(s):
    """
    :param s:
    :return:
    Value in range [0,1]
    where 1 is perfect even distribution of TroubleMakers
    and 0 is for most unbalanced distribution.
    """
    global data
    s = pd.DataFrame(s)
    s.replace(data.TroubleMaker, inplace=True)
    t = s.sum(axis=1)
    return -t.var()


def inclusion_score(s):
    """
    :param s:
    :return:
    How many students are assigned not in a suitable group.
    """
    num_groups = len(s)
    cols = [f'Class {x} OK' for x in range(1, num_groups + 1)]
    s_back = s2.setdefault('s_back', {n: i for i, x in enumerate(s) for n in x})
    data['Group'] = pd.Series(s_back)

    def bad(x):
        return (data[cols[x]] == 0) & (data.Group == x)

    bad_rows = [data.loc[bad(x)] for x in range(num_groups)]
    return -sum(map(len, bad_rows))


def eval_func(s):
    global s2  # cache of values calculated from s
    s2 = {}
    gs = gender_score(s)
    ls = location_score(s)
    fs = friend_enemy_score(s)
    bs = behavior_score(s)
    ns = inclusion_score(s)
    return 10 * gs + ls + 0.5 * fs + bs + ns


def apply(s, move):
    [(i, x, ix), (j, y, iy)] = move
    t = s[i][ix]
    s[i][ix] = s[j][iy]
    s[j][iy] = t


def undo(s, move):
    apply(s, move)


def scan(s, tc, t, delta, fbest, pair):
    """
    Scan 2 groups for best swap
    :param s:
    :param tc:
    :param t:
    :param delta:
    :param fbest:
    :param pair:
    :return:
    """
    f = eval_func
    i, j = pair
    cc = 0
    fmax = None
    for ix, x in enumerate(s[i]):
        print(cc, i, j)
        for iy, y in enumerate(s[j]):
            cc += 1
            move = [(i, x, ix), (j, y, iy)]
            # defaults to non-tabu timestamp
            tx = t.get(x, tc - delta)
            ty = t.get(y, tc - delta)
            tabu = tc - tx < delta or tc - ty < delta
            apply(s, move)
            fmove = f(s)
            asp = fmove > fbest
            if not tabu or asp:
                if fmax == None or fmax < fmove:
                    smax = deepcopy(s)
                    fmax = fmove
                    best_move = move
            undo(s, move)
    return fmax, smax, best_move


def tabu_search_sap(n, s):
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
    t = {}
    fbest = eval_func(sbest)
    print(fbest)
    show_graph(s)
    cores = cpu_count()
    # STOP after X iterations
    while tc - tcbest < 20:
        # GENERATE and SELECT best
        pairs = itertools.combinations(range(m), 2)
        print(list(pairs))
        pairs = itertools.combinations(range(m), 2)
        with Pool(processes=cores) as pool:
            results = pool.map(functools.partial(scan, s, tc, t, delta, fbest), pairs)
        fmax, smax, best_move = max(results, key=itemgetter(0))
        print(fmax, smax, best_move)
        # TEST
        if fmax > fbest:
            sbest = deepcopy(smax)
            tcbest = tc
        # UPDATE
        x, y = best_move[0][1], best_move[1][1]
        t[x] = t[y] = tc
        tc += 1
        s = smax
        show_graph(s)
    return sbest


def tabu(filename, num_groups):
    global data
    # YOU MUST PUT sheet_name=None TO READ ALL CSV FILES IN YOUR XLSM FILE
    # data = pd.read_excel(filename, sheet_name='StudentInfo', skiprows=3)
    data = pd.read_csv('demo.csv')
    n = data.values.shape[0]
    print(data.keys())
    s = np.arange(n)
    np.random.shuffle(s)
    s = np.array_split(s, num_groups)
    s = list(map(list, s))
    return tabu_search_sap(n, s)


def show_graph(s):
    colors = ['blue', 'orange', 'magenta', 'green', 'pink', 'purple', 'yellow', 'darkblue', 'red']
    # Enrich data with Group
    # and replace Friends name with their id
    friend_enemy_score(s)
    nt = Network(height='750px', width='100%', bgcolor='#222222', font_color='white', directed=True)
    nt.set_edge_smooth('dynamic')
    nt.options.edges.smooth.enabled = False
    # set the physics layout of the network
    # nt.barnes_hut(gravity=-20000, central_gravity=0.3, spring_length=250, spring_strength=0.001, damping=0.09)
    n = sum(map(len, s))
    m = len(s)
    ROWS = int(m ** .5 + .5)
    # Add a gravity center node per group
    [nt.add_node(1001 + i, hidden=False, value=10, mass=1, shape='circle', label=f'Group{i+1}',
                 color=colors[i], physics=False,
                 x=400 * (i % ROWS), y=400 * (i // ROWS)) for i in range(m)]

    # populates the nodes and edges data structures
    # https: // github.com / visjs / vis - network / blob / master / lib / network / modules / Groups.js
    nt.add_nodes(data['No'],
                 title=data['Student Name'],
                 color=data['Group'].replace(range(m), colors[:m]),
                 value=n * [1])
    [n.update({'mass': 1}) for n in nt.nodes if n['id'] < 1000]
    # [nt.add_node(i, title=data['Student Name'].iloc[i], group=data['Group'].iloc[i]) for i in range(n)]
    edges = zip(data['No'], 1001 + data['Group'])
    [nt.add_edge(s, t, hidden=True) for s, t in edges]
    # nt.add_edges(edges)
    sources = data['No']
    for i in '123':
        targets = data[f'Friend{i}'].drop(data[data[f'Friend{i}'] == -1].index)
        edges = zip(sources, targets)
        [nt.add_edge(s, t, physics=False) for s, t in edges]
    ## nt.add_edges(edges)
    nt.show('nx.html')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    state = tabu('demo.xlsm', 4)
    print(state)
    with open('output', 'wt') as f:
        f.write(state)
