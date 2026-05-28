import warnings

warnings.filterwarnings("ignore")

from pyscf import ao2mo, tools

# Read in molecule from disk
mf_as = tools.fcidump.to_scf("FCIDUMP")

norb=num_orbitals = mf_as.mol.nao  # Number of atomic orbitals
num_electrons = mf_as.mol.nelectron
spin_sq = mf_as.mol.spin  # Spin multiplicity (2S)

open_shell = False
if spin_sq != 0:
    open_shell = True

num_elec_a = (num_electrons + spin_sq) // 2
num_elec_b = (num_electrons - spin_sq) // 2

n_qubits = 2*norb
# 打印信息
print(f"Number of orbitals: {num_orbitals}")
print(f"Number of electrons: {num_electrons}")
print(f"Spin (2S): {spin_sq}")
print(f"Number of alpha electrons: {num_elec_a}")
print(f"Number of beta electrons: {num_elec_b}")

hcore = mf_as.get_hcore()
eri = ao2mo.restore(1, mf_as._eri, num_orbitals)
nuclear_repulsion_energy = mf_as.mol.energy_nuc()

from pyscf import cc, mp

mf_as.kernel()
mc = cc.CCSD(mf_as)
mc.kernel()
t1 = mc.t1
t2 = mc.t2

# 2. 获得能量差
mo_energy = mf_as.mo_energy  # 获得所有分子轨道能量
mo_occ = mf_as.mo_occ  # 获得所有分子轨道的占据数

# 获得占据和虚拟轨道能量
n_occ = mo_energy[mo_occ > 0]  # 占据轨道的能量
n_vir = mo_energy[mo_occ == 0]  # 虚轨道的能量

import ffsim
from qiskit import QuantumCircuit, QuantumRegister
import numpy as np

# 获得占据轨道和虚轨道的数量
num_occ = len(n_occ)
num_vir = len(n_vir)

# 使用随机数生成器生成 t2 振幅张量
#t2 = np.random.rand(num_occ, num_occ, num_vir, num_vir)

pairs_aa = [(p, p + 1) for p in range(norb - 1)]
pairs_ab = [(p, p) for p in range(norb)]
interaction_pairs = (pairs_aa, pairs_ab)

#params=np.random.uniform(t2.shape)
# Construct UCJ operator
n_reps = 1
#ucj_op = ffsim.UCJOpSpinBalanced.from_t_amplitudes(t2, n_reps=n_reps,interaction_pairs=interaction_pairs)

params=np.load('final_parameters.npy')

#n_param=len(ucj_op.to_parameters(interaction_pairs=interaction_pairs))
ucj_op = ffsim.UCJOpSpinBalanced.from_parameters(params[:119], norb=norb, n_reps=n_reps,interaction_pairs=interaction_pairs)
n_param=len(ucj_op.to_parameters(interaction_pairs=interaction_pairs))

print('parameters: ', n_param)

nelec = (num_elec_a, num_elec_b)

# create an empty quantum circuit
qubits = QuantumRegister(2 * num_orbitals, name="q")
circuit = QuantumCircuit(qubits)

# prepare Hartree-Fock state as the reference state and append it to the quantum circuit
circuit.append(ffsim.qiskit.PrepareHartreeFockJW(num_orbitals, nelec), qubits)

# apply the UCJ operator to the reference state
circuit.append(ffsim.qiskit.UCJOpSpinBalancedJW(ucj_op), qubits)
circuit.measure_all()

from qiskit_aer import AerSimulator

backend =AerSimulator()

from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

pass_manager = generate_preset_pass_manager(
    optimization_level=3, backend=backend)

pass_manager.pre_init = ffsim.qiskit.PRE_INIT
isa_circuit = pass_manager.run(circuit)
print(f"Gate counts (w/ pre-init passes): {isa_circuit.count_ops()}")

n_shots_list = [1000, 10000, 100000, 1000000, 10000000, 100000000]
data_by_shots = {}

rand_seed = 12
from qiskit_addon_sqd.counts import counts_to_arrays

# Convert counts into bitstring and probability arrays
#bitstring_matrix_full, probs_arr_full = counts_to_arrays(counts)

from qiskit_addon_sqd.configuration_recovery import recover_configurations
from qiskit_addon_sqd.fermion import (
    flip_orbital_occupancies,
    solve_fermion,
)
from qiskit_addon_sqd.subsampling import postselect_and_subsample

# SQD options
iterations = 1
# Eigenstate solver options
n_batches = 1
max_davidson_cycles = 500

import matplotlib.pyplot as plt

# 初始化数据结构来存储 n_shots 对应的态和概率
data_by_shots = {}

# 假设 n_shots_list 和测量结果已经计算完毕
# 请确保在实际运行时，n_shots_list 和 backend 已经定义好
for n_shots in n_shots_list:
    job = backend.run([isa_circuit], shots=n_shots)
    primitive_result = job.result()
    counts = primitive_result.get_counts()

    # 计算总测量次数
    total_counts = sum(counts.values())

    # 按照频率对测量结果进行排序
    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)

    # 取前20个频率最大的态
    top_20_states = sorted_counts[:20]

    # 分别获取态和对应的概率
    states, probabilities = zip(*[(state, freq / total_counts) for state, freq in top_20_states])

    # 存储当前 n_shots 对应的态和概率
    data_by_shots[n_shots] = (states, probabilities)

# 合并所有 n_shots 的数据，确保状态顺序一致
all_states = list(set().union(*[data_by_shots[n_shots][0] for n_shots in n_shots_list]))

# 初始化存储不同 n_shots 对应的概率矩阵
probability_matrix = []
for n_shots in n_shots_list:
    states, probabilities = data_by_shots[n_shots]
    probability_dict = dict(zip(states, probabilities))
    probabilities_aligned = [probability_dict.get(state, 0) for state in all_states]
    probability_matrix.append(probabilities_aligned)

# 确保每个组态都有六个柱状图，并填充缺失值为0
probability_matrix = np.array(probability_matrix).T.tolist()

# 绘制柱状图
x = np.arange(len(all_states))  # 组态的位置
width = 0.15  # 每组柱的宽度

fig, ax = plt.subplots(figsize=(14, 8))

for i, n_shots in enumerate(n_shots_list):
    probabilities = [row[i] for row in probability_matrix]
    ax.bar(x + i * width, probabilities, width, label=f'n_shots={n_shots}')

# 设置横轴和纵轴
ax.set_xlabel('States')
ax.set_ylabel('Measurement Probability')
ax.set_title('Measurement Probabilities for Different n_shots')
ax.set_xticks(x + width * (len(n_shots_list) - 1) / 2)
ax.set_xticklabels(all_states, rotation=90, fontsize=8)
ax.legend()

plt.tight_layout()
plt.show()

import sys
sys.exit(0)

#print('counts: ',counts)
#from qiskit_addon_sqd.counts import generate_counts_uniform


##    bitstring_matrix_full, probs_arr_full = counts_to_arrays(counts)
##
##    # Run eigenstate solvers in a loop. This loop should be parallelized for larger problems.
##    energy_sci, coeffs_sci, avg_occs, spin = solve_fermion(
##        bitstring_matrix_full,
##        hcore,
##        eri,
##        open_shell=open_shell,
##        spin_sq=spin_sq,
##        max_davidson=max_davidson_cycles,
##    )
##    energy_sci += nuclear_repulsion_energy
##    print('energy_sci: ',energy_sci)


