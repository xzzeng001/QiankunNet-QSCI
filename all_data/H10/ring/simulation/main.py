import numpy as np
import pyscf
import pyscf.mcscf
from pyscf import gto

import ffsim

from pyscf import cc
from pyscf.tools import fcidump
import scipy

mf = fcidump.to_scf('FCIDUMP',molpro_orbsym=True)
mf.mol.verbose = 4
scf=mf.run()

# Get CCSD t2 amplitudes for initializing the ansatz
ccsd = cc.CCSD(scf).run()

hcore = mf.get_hcore()
n_orb=mf.mol.nao
energy_core= mf.mol.energy_nuc()

one_body_mo = hcore
two_body_mo = pyscf.ao2mo.restore(1, mf._eri,n_orb)

mol_hamiltonian = ffsim.MolecularHamiltonian(
    one_body_tensor=one_body_mo,
    two_body_tensor=two_body_mo,
    constant=energy_core)

norb=mf.mol.nao
nelec=(5,5)

pairs_aa = [(p, p + 1) for p in range(norb - 2)]
pairs_ab = [(p, p) for p in range(norb-1)]
interaction_pairs = (pairs_aa, pairs_ab)

# Construct UCJ operator
n_reps = 1
operator = ffsim.UCJOpSpinBalanced.from_t_amplitudes(ccsd.t2, n_reps=n_reps,interaction_pairs=interaction_pairs)

print('parameters: ',len(operator.to_parameters(interaction_pairs=interaction_pairs)))

# Construct the Hartree-Fock state to use as the reference state
reference_state = ffsim.hartree_fock_state(norb, nelec)

# Apply the operator to the reference state
ansatz_state = ffsim.apply_unitary(reference_state, operator, norb=norb, nelec=nelec)

hamiltonian = ffsim.linear_operator(mol_hamiltonian, norb=norb, nelec=nelec)
energy = np.real(np.vdot(ansatz_state, hamiltonian @ ansatz_state))
print(f"Energy at initialization: {energy}")

import scipy.optimize

def fun(x):
    # Initialize the ansatz operator from the parameter vector
    operator = ffsim.UCJOpSpinBalanced.from_parameters(x, norb=norb, n_reps=n_reps,interaction_pairs=interaction_pairs)
    # Apply the ansatz operator to the reference state
    final_state = ffsim.apply_unitary(reference_state, operator, norb=norb, nelec=nelec)

    return np.real(np.vdot(final_state, hamiltonian @ final_state))


# 定义 callback 函数用于每步输出
energies = []  # 用于存储每一步的能量

def callback(xk):
    energy = fun(xk)
    energies.append(energy)
    print(f"Iteration: {len(energies)}, Energy: {energy}")

# 优化过程
result = scipy.optimize.minimize(fun, 
    x0=operator.to_parameters(interaction_pairs=interaction_pairs), 
    method="L-BFGS-B", 
    options={'ftol': 1e-10,'maxiter':1000} ,#dict(maxiter=1000),
    callback=callback)

# 优化结果
print("Optimization completed. Final energy:", result.fun)

# 保存最终参数
final_parameters = result.x
np.save("final_parameters.npy", final_parameters)
print("Final parameters saved to final_parameters.npy")
