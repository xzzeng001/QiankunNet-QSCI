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

##ccsd = cc.CCSD(scf).run()

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
n_reps=1
nelec=(15,15)

pairs_aa = [(p, p + 1) for p in range(0,norb - 1,4)]
pairs_ab = [(p, p) for p in range(0,norb,4)]
interaction_pairs = (pairs_aa, pairs_ab)

#pairs_aa = [(14, 15)]
#pairs_ab = [(14, 15)] # for p in range(norb)]
#interaction_pairs = (pairs_aa, pairs_ab)

# Construct UCJ operator
n_reps = 1
#ucj_op = ffsim.UCJOpSpinBalanced.from_t_amplitudes(ccsd.t2, n_reps=n_reps,interaction_pairs=interaction_pairs)
params=np.random.uniform(high=2*np.pi,size=(410))

ucj_op = ffsim.UCJOpSpinBalanced.from_parameters(params, norb=norb, n_reps=n_reps,interaction_pairs=interaction_pairs)
print('parameters: ',len(ucj_op.to_parameters(interaction_pairs=interaction_pairs)))

# Construct circuit
qubits = QuantumRegister(2 * norb)
circuit = QuantumCircuit(qubits)
circuit.append(ffsim.qiskit.PrepareHartreeFockJW(norb, nelec), qubits)
circuit.append(ffsim.qiskit.UCJOpSpinBalancedJW(ucj_op), qubits)
#circuit.measure_all()

from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

# Initialize quantum device backend
backend = GenericBackendV2(2 * norb, basis_gates=['cx','rx','ry','rz','h','x'])

from qiskit.transpiler import PassManager

# Create a pass manager for circuit transpilation
pass_manager = generate_preset_pass_manager(optimization_level=3, backend=backend)

# Set the pre-initialization stage of the pass manager with passes suggested by ffsim
pass_manager.pre_init = ffsim.qiskit.PRE_INIT

# Transpile the circuit
transpiled_qc = pass_manager.run(circuit)

from qiskit import qasm2
# Export to a file.
qasm2.dump(transpiled_qc, "circuit.qasm")

#transpiled.count_ops()

#transpiled_qc = transpile(circuit,basis_gates=['x','ry','rz','rx','cx','h'])

# 绘制量子电路并获取 Matplotlib 图形对象
##circuit_fig = transpiled_qc.draw(
##    output='mpl',
##    fold=100,
##    style={'backgroundcolor': '#FFFFFF'},
##    initial_state=True
##)
##
###circuit_fig = transpiled_qc.draw()
##circuit_fig.savefig('my_quantum_circuit.png')

def analyze_circuit(circuit):
    single_qubit_gates = 0
    two_qubit_gates = 0
    parameters = set()

    for instruction, qargs, cargs in circuit.data:
        if instruction.name in ['measure', 'barrier']:
            continue  # Skip measurement and barrier operations
        num_qubits = len(qargs)
        if num_qubits == 1:
            single_qubit_gates += 1
        elif num_qubits == 2:
            two_qubit_gates += 1
        # Collect parameters
#        for param in instruction.params:
#            if isinstance(param, Parameter):
#                parameters.add(param)

    circuit_depth = circuit.depth()
#    num_parameters = len(parameters)

    print("Circuit depth:", circuit_depth)
    print("Number of single-qubit gates:", single_qubit_gates)
    print("Number of two-qubit gates:", two_qubit_gates)

    return circuit_depth, single_qubit_gates, two_qubit_gates

circuit_depth, single_qubit_gates, two_qubit_gates=analyze_circuit(transpiled_qc)
#print("Number of parameters:", len(t2))

with open("info_circuit.txt", "w") as file:
    file.write(f"Circuit depth: {circuit_depth}\n")
    file.write(f"Number of single-qubit gates: {single_qubit_gates}\n")
    file.write(f"Number of two-qubit gates: {two_qubit_gates}\n")
#    file.write(f"Number of parameters: {len(t2)}\n")


# Construct the Hartree-Fock state to use as the reference state
reference_state = ffsim.hartree_fock_state(norb, nelec)

# Apply the operator to the reference state
ansatz_state = ffsim.apply_unitary(reference_state, ucj_op, norb=norb, nelec=nelec)

# Compute the energy ⟨ψ|H|ψ⟩ of the ansatz state
hamiltonian = ffsim.linear_operator(mol_hamiltonian, norb=norb, nelec=nelec)
energy = np.real(np.vdot(ansatz_state, hamiltonian @ ansatz_state))
print(f"Energy at initialization: {energy}")

