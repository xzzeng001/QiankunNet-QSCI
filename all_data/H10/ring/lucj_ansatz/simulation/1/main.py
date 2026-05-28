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

import ffsim
mf_as.kernel()

#mc = cc.UCCSD(mf_as)
#mc.kernel()
#t1 = mc.t1
#t2 = mc.t2

mol_data = ffsim.MolecularData.from_fcidump('FCIDUMP')
norb = mol_data.norb
nelec = mol_data.nelec
mol_hamiltonian = mol_data.hamiltonian

# 2. 获得能量差
mo_energy = mf_as.mo_energy  # 获得所有分子轨道能量
mo_occ = mf_as.mo_occ  # 获得所有分子轨道的占据数

# 获得占据和虚拟轨道能量
n_occ = mo_energy[mo_occ > 0]  # 占据轨道的能量
n_vir = mo_energy[mo_occ == 0]  # 虚轨道的能量

from qiskit import QuantumCircuit, QuantumRegister
import numpy as np

# 获得占据轨道和虚轨道的数量
num_occ = len(n_occ)
num_vir = len(n_vir)

# 使用随机数生成器生成 t2 振幅张量
t2 = np.random.rand(num_occ, num_occ, num_vir, num_vir)

pairs_aa = [(p, p + 1) for p in range(norb - 1)]
pairs_ab = [(p, p) for p in range(norb)]
interaction_pairs = (pairs_aa, pairs_ab)

# Construct UCJ operator
n_reps = 1
operator = ffsim.UCJOpSpinBalanced.from_t_amplitudes(t2, n_reps=n_reps,interaction_pairs=interaction_pairs)

# Construct the Hartree-Fock state to use as the reference state
reference_state = ffsim.hartree_fock_state(norb, nelec)

# Apply the operator to the reference state
ansatz_state = ffsim.apply_unitary(reference_state, operator, norb=norb, nelec=nelec)

# Compute the energy ⟨ψ|H|ψ⟩ of the ansatz state
hamiltonian = ffsim.linear_operator(mol_hamiltonian, norb=norb, nelec=nelec)
energy = np.real(np.vdot(ansatz_state, hamiltonian @ ansatz_state))
print(f"Energy at initialization: {energy}")

import scipy.optimize

def fun(x):
    # Initialize the ansatz operator from the parameter vector
    operator = ffsim.UCJOpSpinBalanced.from_parameters(x, norb=norb, n_reps=n_reps)
    # Apply the ansatz operator to the reference state
    final_state = ffsim.apply_unitary(reference_state, operator, norb=norb, nelec=nelec)
    # Return the energy ⟨ψ|H|ψ⟩ of the ansatz state
    return np.real(np.vdot(final_state, hamiltonian @ final_state))


# 定义 callback 函数用于每步输出
energies = []  # 用于存储每一步的能量

def callback(xk):
    energy = fun(xk)
    energies.append(energy)
    print(f"Iteration: {len(energies)}, Energy: {energy}")

# 优化过程
result = scipy.optimize.minimize(fun,
    x0=operator.to_parameters(),
    method="L-BFGS-B",
    options={'ftol': 1e-10,'maxiter':1000} ,
    #options=dict(maxiter=1000),
    tol=1e-10,
    callback=callback)

# 优化结果
print("Optimization completed. Final energy:", result.fun)

# 保存最终参数
final_parameters = result.x
np.save("final_parameters.npy", final_parameters)
print("Final parameters saved to final_parameters.npy")

