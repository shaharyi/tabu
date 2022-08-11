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
s = list(map(set, s))
s_back = {n: i for i, x in enumerate(s) for n in x}
