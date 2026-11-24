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

#params=np.random.uniform([119])
# Construct UCJ operator
n_reps = 1
#ucj_op = ffsim.UCJOpSpinBalanced.from_t_amplitudes(t2, n_reps=n_reps,interaction_pairs=interaction_pairs)

params=np.load('final_parameters.npy')

#n_param=len(ucj_op.to_parameters(interaction_pairs=interaction_pairs))
ucj_op = ffsim.UCJOpSpinBalanced.from_parameters(params, norb=norb, n_reps=n_reps)
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

n_shots=10000

job = backend.run([isa_circuit], shots=n_shots)
primitive_result = job.result()
counts = primitive_result.get_counts()

# 按照频率对测量结果进行排序
sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)

# 分别获取态和对应的概率
states, probs = zip(*[(state, freq / n_shots) for state, freq in sorted_counts])

#print('states: ',states)

n_unique = len(states)
ci_probs=[]
abab_strings=states
ci_states=np.zeros([n_unique,n_qubits],dtype=int)
for i in range(n_unique):
    binary_str = states[i][::-1]
    half = len(binary_str) // 2
    left, right = binary_str[:half], binary_str[half:]
    interleaved = "".join(a + b for a, b in zip(left, right))

    ci_states[i, :] = [int(bit) for bit in interleaved]
    ci_probs.append(probs[i])

#print('ci_states: ',ci_states)

ci_probs = np.array(ci_probs)

np.savez('N2_3.0_lucj.npz',ci_probs=ci_probs,ci_states=ci_states)


