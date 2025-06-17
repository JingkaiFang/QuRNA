# Copyright 2025 Jing-Kai Fang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Implements the function of encoding the weights of points and edges into 2-by-2 binary blocks."""

import numpy as np
import random
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
from matplotlib import pyplot as plt
import matplotlib.patches as patches

def binary_weight(n: int) -> np.ndarray:
    b = "{0:b}".format(n)
    b = "{:0>4}".format(b)
    x = np.zeros([4],int)
    for i in range(4):
        x[i] = b[i]
    x = x.reshape(2,2)
    return x

def adj_binary(adj: np.ndarray) -> np.ndarray:
    s = np.shape(adj)
    adj_bw = []
    for i in range(s[0]):
        adj_line = []
        for j in range(s[1]):
            bw = binary_weight(adj[i,j])
            if len(adj_line) == 0:
                adj_line = bw
            else:
                assert len(adj_line) == len(bw)
                adj_line = np.concatenate((adj_line, bw), 1)
        if len(adj_bw) == 0:
            adj_bw = adj_line
        else:
            assert np.shape(adj_bw)[1] == np.shape(adj_line)[1]
            adj_bw = np.concatenate((adj_bw, adj_line), 0)
    return adj_bw        

def adj_binary_symmetric(adj: np.ndarray) -> np.ndarray:
    s = np.shape(adj)
    adj_bw = []
    for i in range(s[0]):
        adj_line = []
        for j in range(s[1]):
            bw = binary_weight(adj[i,j])
            if i > j:
                bw = bw.T
            if len(adj_line) == 0:  
                adj_line = bw
            else:
                assert len(adj_line) == len(bw)
                adj_line = np.concatenate((adj_line, bw), 1)
        if len(adj_bw) == 0:
            adj_bw = adj_line
        else:
            assert np.shape(adj_bw)[1] == np.shape(adj_line)[1]
            adj_bw = np.concatenate((adj_bw, adj_line), 0)
    return adj_bw   
