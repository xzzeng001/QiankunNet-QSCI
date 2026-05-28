import pyscf
from pyscf import gto, scf, cc, tools
from pyscf.tools import fcidump
import numpy as np
import openfermion
from openfermion import MolecularData

from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit import transpile

import ffsim

import scipy

mf = fcidump.to_scf('FCIDUMP',molpro_orbsym=True)
mf.mol.verbose = 4
scf=mf.run()

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
nelec=(15,15)

pairs_aa = [(p, p + 1) for p in range(0,norb - 1,4)]
pairs_ab = [(p, p) for p in range(0,norb,4)]
interaction_pairs = (pairs_aa, pairs_ab)

#pairs_aa = [(p, p + 1) for p in range(norb - 1)]
#pairs_ab = [(p, p) for p in range(norb)]
#interaction_pairs = (pairs_aa, pairs_ab)

# Construct UCJ operator
n_reps = 1
#operator = ffsim.UCJOpSpinBalanced.from_t_amplitudes(ccsd.t2, n_reps=n_reps,interaction_pairs=interaction_pairs)

#print('parameters: ',len(operator.to_parameters(interaction_pairs=interaction_pairs)))
params=np.random.uniform(high=2*np.pi,size=(410))

operator = ffsim.UCJOpSpinBalanced.from_parameters(params, norb=norb, n_reps=n_reps,interaction_pairs=interaction_pairs)

# Construct the Hartree-Fock state to use as the reference state
reference_state = ffsim.hartree_fock_state(norb, nelec)

# Apply the operator to the reference state
ansatz_state = ffsim.apply_unitary(reference_state, operator, norb=norb, nelec=nelec)
hamiltonian = ffsim.linear_operator(mol_hamiltonian, norb=norb, nelec=nelec)

# Construct circuit
#qubits = QuantumRegister(2 * norb)
#circuit = QuantumCircuit(qubits)
#circuit.append(ffsim.qiskit.PrepareHartreeFockJW(norb, nelec), qubits)
#circuit.append(ffsim.qiskit.UCJOpSpinBalancedJW(operator), qubits)
#circuit.measure_all()

from collections import defaultdict

from ffsim.optimize import minimize_linear_method

# Define function that converts a list of parameters to the corresponding state vector
def params_to_vec(x: np.ndarray) -> np.ndarray:
    operator = ffsim.UCJOpSpinBalanced.from_parameters(
        x, norb=norb, n_reps=n_reps, interaction_pairs=interaction_pairs
    )
    return ffsim.apply_unitary(reference_state, operator, norb=norb, nelec=nelec)


# Define a callback function used to save optimization information (this is optional)
info = defaultdict(list)


def callback(intermediate_result: scipy.optimize.OptimizeResult):
    # The callback function is called after each iteration. It accepts
    # an OptimizeResult object storing the parameters and function value at
    # the current iteration, and possibly other information
    info["x"].append(intermediate_result.x)
    info["fun"].append(intermediate_result.fun)
    if hasattr(intermediate_result, "jac"):
        info["jac"].append(intermediate_result.jac)
    if hasattr(intermediate_result, "regularization"):
        info["regularization"].append(intermediate_result.regularization)
    if hasattr(intermediate_result, "variation"):
        info["variation"].append(intermediate_result.variation)


# Optimize with the linear method
result = minimize_linear_method(
    params_to_vec,
    hamiltonian,
    x0=operator.to_parameters(interaction_pairs=interaction_pairs),
    maxiter=10,
    callback=callback,
)

# Print some information
print(f"Number of parameters: {len(result.x)}")
print(result)
print()
for i, (fun, jac, regularization, variation) in enumerate(
    zip(info["fun"], info["jac"], info["regularization"], info["variation"])
):
    print(f"Iteration {i + 1}")
    print(f"    Energy: {fun}")
    print(f"    Norm of gradient: {np.linalg.norm(jac)}")
    print(f"    Regularization hyperparameter: {np.linalg.norm(regularization)}")
    print(f"    Variation hyperparameter: {np.linalg.norm(variation)}")

