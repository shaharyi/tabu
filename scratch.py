# Run this way:
# python -i scratch.py

from copy import deepcopy
import numpy as np
import pandas as pd
import main

num_classes = 4
# data = pd.read_excel('demo.xlsm', sheet_name='StudentInfo', skiprows=3)
data = main.data = pd.read_csv('demo.csv')
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
fbest = main.eval_func(sbest)
cc = 0
tc = 0
tcbest = tc
t = {}

main.friend_enemy_score(s)
# main.show_graph(s)
