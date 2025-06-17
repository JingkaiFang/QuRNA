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

"""Implement the function of identifying known motifs from RNA secondary structures."""

from qsubgisom_w import ansatz, observable, s4_ansatz, CustomVQE
from qsubgisom_w import sample_exact_thetas, perm_to_2line
from qiskit.algorithms import VQE
from qiskit.algorithms.optimizers import SLSQP, ADAM, COBYLA, NELDER_MEAD, SLSQP, CRS
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit import Aer
from qiskit.quantum_info import Operator
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import numpy as np
from tqdm import tqdm
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
from adj_weight import adj_binary, adj_binary_symmetric
import os
import matplotlib
matplotlib.use('Agg') 
import logging
import sys
import warnings
warnings.filterwarnings('ignore')

path = f'/jdfssz1/ST_BIOINTEL/P23Z10200N0816/fangjingkai/quRNA/result'

a1 = np.array([[ 0,  3,  10,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
               [ 3,  0,  0,  10,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
               [10,  0,  0,  3, 10,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
               [ 0,  10, 3,  0, 0,  10,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
               [ 0,  0,  10, 0,  0, 3,  10,  0,  0,  0,  0,  0,  0,  0,  0,  0],
               [ 0,  0,  0,  10, 3,  0, 0,  10,  0,  0,  0,  0,  0,  0,  0,  0],
               [ 0,  0,  0,  0,  10, 0,  0,  3,  10,  0,  0,  0,  0,  0,  0,  0],
               [ 0,  0,  0,  0,  0,  10, 3,  0, 0,  10,  0,  0,  0,  0,  0,  0],
               [ 0,  0,  0,  0,  0,  0,  10, 0,  0,  3,  10,  0,  0,  0,  0,  0],
               [ 0,  0,  0,  0,  0,  0,  0,  10, 3,  0,  0,  10,  0,  0,  0,  0],
               [ 0,  0,  0,  0,  0,  0,  0,  0,  10, 0,  0, 1,  10,  0,  0,  0],
               [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  10, 1,  0, 0,  10,  0,  0],
               [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  10, 0,  0, 0,  10,  0],
               [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  10, 0,  0,  0, 10],
               [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  10, 0,  0,  10],
               [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  10, 10,  0]])
a2 = np.array([[ 0,  3,  10,  0,  0,  0,  0,  0],
               [ 3,  0,  0,  10,  0,  0,  0,  0],
               [ 10, 0,  0,  1,  10,  0,  0,  0],
               [ 0,  10,  1,  0,  0,  10,  0,  0],
               [ 0,  0,  10,  0,  0,  0,  10,  0],
               [ 0,  0,  0,  10,  0,  0,  0,  10],
               [ 0,  0,  0,  0,  10,  0,  0,  10],
               [ 0,  0,  0,  0,  0,  10,  10,  0]])
adj1 = adj_binary_symmetric(a1)
g1 = nx.from_numpy_array(a1)
adj2 = adj_binary_symmetric(a2)
g2 = nx.from_numpy_array(a2)

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))
nx.draw(g1, with_labels=True, ax=axs[0][0])
nx.draw(g2, with_labels=True, ax=axs[0][1])
axs[0][0].set_title('Source graph ($\\mathcal{A}$)')
axs[0][1].set_title('Subgraph ($\\mathcal{B}$)')
axs[1][0].set_title('Adjacency matrix ($A$)')
axs[1][0].matshow(adj1)
axs[1][0].set_title('Adjacency matrix ($B$)')
axs[1][1].matshow(adj2)
plt.savefig(os.path.join(path, 'adj.png'))


seed = np.random.randint(101,26481379)
algorithm_globals.random_seed = seed
rng = np.random.default_rng(seed=seed)
pathseed = f'_{seed}'
qc, params = ansatz(adj1, adj2)

logger = logging.getLogger()


class StreamToLoggerAndFile:
    def __init__(self, logger, log_level=logging.INFO, file_path=None):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''
        self.encoding = 'utf-8'
        self.file = open(file_path, "a", encoding='utf-8') if file_path else None

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())
            if self.file:
                self.file.write(line )

    def flush(self):
        pass

    def close(self):
        if self.file:
            self.file.close()
file_path = path + '/' + f'loss_log.txt'
stream_logger = StreamToLoggerAndFile(logger, logging.INFO, file_path)
sys.stdout = stream_logger

max_trials = 3
qi = QuantumInstance(Aer.get_backend('statevector_simulator'),
                     seed_transpiler=seed, seed_simulator=seed,
                     shots=1024)
optim = SLSQP(maxiter=2, disp=True) # Init classical optimizer.
# optim = ADAM(maxiter=500, lr=0.001)
# optim = COBYLA(maxiter=100, disp=True)
def my_callback(eval_count, parameters, mean, std, iter):
    tanh_mean = np.tanh(mean)
    print(f"Evaluation: {eval_count}, Mean: {mean}, Std: {std}")
loss_per_iteration = []
def store_loss(iteration, parameters, loss, stepsize):
    loss_per_iteration.append([iteration, loss])

def tanh_expectation_value(result):
    # This function will extract the expectation value from the result
    expectation_value = result.eigenvalue.real
    # Apply tanh to the expectation value
    tanh_value = np.tanh(expectation_value)
    return tanh_value

def log_expectation_value(result):
    expectation_value = result.eigenvalue.real
    Log_value = np.log(1 + expectation_value)
    return Log_value

def trial():
    # Randomize initial parameters.
    initial_point = (rng.uniform(size=len(qc.parameters)) - 1/2) * np.pi
    vqe = VQE(qc, quantum_instance=qi, initial_point=initial_point,
              optimizer=optim, callback=store_loss)
    obj = vqe.compute_minimum_eigenvalue(observable(qc.num_qubits))
    final_tanh_value = log_expectation_value(obj)
    # return final_tanh_value, obj
    return obj.optimal_value, obj

def trial_tanh():
    initial_point = (rng.uniform(size=len(qc.parameters)) - 1/2) * np.pi
    custom_vqe = CustomVQE(qc, quantum_instance=qi, initial_point=initial_point,
              optimizer=optim, callback=my_callback)
    obj = custom_vqe.compute_minimum_eigenvalue(observable(qc.num_qubits))
    return obj.optimal_value, obj

# Run a number of trials and select the one presenting
# the least minimum eigenvalue.
results = [trial() for _ in tqdm(range(max_trials))]
results = sorted(results, key=lambda obj: obj[0])
result = results[0][1]

sys.stdout = sys.__stdout__
stream_logger.close()
f = open(file_path, "a")
f.write(f' seed = {seed} \n')
f.close()

f = open(path+'/'+f'loss_{seed}.txt', "a")
f.write(f'seed = {seed} \n')
for i in range(len(loss_per_iteration)):
    f.write(f'{loss_per_iteration[i][0]} {loss_per_iteration[i][1]} \n')
f.close()
# Prepare the part of the ansatz implementing the linear composition of permutations.
# Note that the topology must be consistent with that used for the optimization problem above. 

qc1 = s4_ansatz('circular', qreg=(qc.num_qubits - 1)//2 - 1, params=params)[0]

def cost_f(a1, a2, p):
    m = p @ a1 @ p.T
    m = m[:len(a2), :len(a2)]
    return np.linalg.norm(m - a2)

sampled_params_dicts = sample_exact_thetas(result.optimal_parameters,
                                           n=64, seed=seed)
min_cost = np.inf
v_min = []
f = open(path+'/'+f'permutation matrix_{seed}.txt', "a")
for v in sampled_params_dicts:
    p1 = np.abs(Operator(qc1.bind_parameters(v)).data)
    p1 = np.round(p1)
    cost = cost_f(a1, a2, p1)
    f.write(f'{cost} \n')
    if cost < min_cost:
        p2 = p1
        min_cost = cost
        v_min = list(v.values())
        if cost < 1.:
            break
f.write(f'{p2} \n')
f.close()

# Plot the permutation matrix that presents the least cost.
plt.matshow(p2, vmin=0, vmax=1)
plt.savefig(os.path.join(path, 'permutation matrix'+ pathseed+'.png'))


# Plot with highlighted solution

def plot_solution_edges(g1, *, pos, perm, ax):
    vx_sel = perm_to_2line(perm, inverse=True)
    vx_sel = vx_sel[1, :len(a2)]
    nx.draw_networkx_edges(g1.subgraph(vx_sel), pos,
                           width=5.0, alpha=0.5, ax=ax)

fig, ax = plt.subplots()
g1_pos = graphviz_layout(g1)
nx.draw(g1, g1_pos, with_labels=True, ax=ax)
plot_solution_edges(g1, pos=g1_pos, perm=p2, ax=ax)
plt.savefig(os.path.join(path, 'overlab'+ pathseed+'.png'))

fig, axs = plt.subplots(1, 3, figsize=(12, 5))
fig.patch.set_facecolor('white')
axs[0].matshow(a1)
axs[0].set_title('Adj $A$')
axs[1].matshow(a2)
axs[1].set_title('Adj $B$')
axs[2].matshow(p2 @ a1 @ p2.T)
axs[2].set_title('Permuted adj $PAP^T$')
rect = patches.Rectangle((-0.5, -0.5), len(a2), len(a2), linewidth=2,
                         edgecolor='red', facecolor='white', alpha=0.5)
axs[2].add_patch(rect)
plt.savefig(os.path.join(path, 'adj2'+pathseed+'.png'))

PAPT = p2 @ a1 @ p2.T
f = open(path+'/'+f'adj_PAPT_{seed}.txt',"a")
f.write(f'{PAPT} \n')
f.close()
