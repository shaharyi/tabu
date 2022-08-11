# Run this way:
# python -i scratch.py

from collections import defaultdict
from copy import deepcopy
import numpy as np
import pandas as pd
from pyvis.network import Network
import networkx as nx

num_classes = 4
# data = pd.read_excel('demo.xlsm', sheet_name='StudentInfo', skiprows=3)
data = pd.read_csv('demo.csv')
n = data.values.shape[0]
print(data.keys())
s = np.arange(n)
np.random.shuffle(s)
s = np.array_split(s, num_classes)
s = list(map(list, s))
s_back = {n: i for i, x in enumerate(s) for n in x}

k = list(map(len, s))
m = len(k)
delta = 6
neighbors = []
sbest = deepcopy(s)
fbest = f(sbest)
cc = 0
tc = 0
tcbest = tc
# defaults to non-tabu timestamp
t = defaultdict(lambda: tc - delta)
for i in range(m):
    for j in range(i + 1, m):
        for ix, x in enumerate(s[i]):
            print(cc, i, j)
            for iy, y in enumerate(s[j]):
                cc += 1
                move = [(i, x, ix), (j, y, iy)]
                tabu = tc - t[x] < delta or tc - t[y] < delta
                asp = f2(s, move) > fbest
                if not tabu or asp:
                    neighbors += [move]
