from qiskit import QuantumCircuit #, transpile, assemble, execute
from qiskit.quantum_info import Statevector
from qiskit.circuit import Parameter
import numpy as np
import matplotlib.pyplot as plt
import sys
from qiskit.quantum_info import SparsePauliOp, Pauli
from qiskit.circuit.library import PauliEvolutionGate


def read_target_configurations_and_coeff():
    # read the ci coefficients
    f=np.loadtxt('ci.txt',dtype=np.str_)
    
    # determinate the number of orbitals and determinant
    norb=len(f[0])-2
    ndet=len(f[:,0])

    target_states=[]
    coefficients=[]
    n_qubits=2*norb

    for i in range(ndet):
        psi_str=''
        for j in range(norb):
            sk=f[i,j+2]
            if sk.strip() == '2':
                psi_str += '11'
            elif sk.strip() == '0':
                psi_str += '00'
            elif sk.strip() == 'a':
                psi_str += '10'
            elif sk.strip() == 'b':
                psi_str += '01'
            else:
                print('read error string!!!!')
                exit(0)
    
        rr=f[i,1]
        target_states.append(psi_str)
        coefficients.append(rr)
        
    return n_qubits, target_states, coefficients

def excitation_analysis(n_qubits, target_states, coefficients, tol):
    # Step 1: filter the coeffs less than tol
    filtered_states = [(state, coef, idx) for idx, (state, coef) in enumerate(zip(target_states, coefficients)) if abs(np.float(coef)) >= tol]

    if not filtered_states:
        return []

    # the first configuration as the ref
    ref_state = filtered_states[0][0]

    excitations = []  # save the output results
    unique_diff_positions = set()

    # Helper function: 
    def get_diff_positions(state1, state2):
        return [i for i in range(n_qubits) if state1[i] != state2[i]]

    # Step 2: loop the left configurations 
    for state, coef, original_idx in filtered_states[1:]:
        diff_positions = get_diff_positions(ref_state, state)
        diff_count = len(diff_positions)

        index_list = [diff_count] + [diff_positions[i] for i in range(diff_count)]
        excitations.append(index_list)
        unique_diff_positions.update(diff_positions)

    # Get the list of positions not in unique_diff_positions
    remaining_positions = [i for i in range(n_qubits) if i not in unique_diff_positions]

    return filtered_states, excitations, list(unique_diff_positions), remaining_positions

def generate_circuit_from_ref_state(n_qubits, target_state):
    # construct n_qubits quantum circuit
    qc = QuantumCircuit(n_qubits)

    ind_=0
    # target_state 
    for i, bit in enumerate(target_state[0]):
        if bit == '1':
            qc.x(i)
            ind_ +=1

    return qc

def single_excitation(qc, qubit_pair, ind_param):
    """
    在指定的比特对之间实现单激发线路
    Args:
        qc: 量子电路
        qubit_pair: 包含两个量子比特的元组 (q_i, q_j)
        theta: 单激发角参数
    """
    qubit1, qubit2 = qubit_pair

    name_param='theta_'+str(ind_param)
    theta = Parameter(name_param)

    qc.cx(qubit1, qubit2)
    qc.ry(theta/2, qubit1)
    qc.cx(qubit2, qubit1)
    qc.ry(-theta/2, qubit1)
    qc.cx(qubit2, qubit1)
    qc.cx(qubit1, qubit2)

def double_excitation(qc, qubit_pairs, ind_param):
    """
    在指定的四个比特之间实现双激发线路
    Args:
        qc: 量子电路
        qubit_pairs: 两个包含量子比特的元组 [(q1, q2, q3, q4)]
        theta: 双激发角参数
    """
    qubit1, qubit2, qubit3, qubit4 = qubit_pairs

    name_param='theta_'+str(ind_param)
    theta = Parameter(name_param)

    qc.cx(qubit3, qubit4)
    qc.cx(qubit1, qubit3)
    qc.h(qubit1)
    qc.h(qubit4)

    qc.cx(qubit1, qubit2)
    qc.cx(qubit3, qubit4)

    qc.ry(-theta/8, qubit1)
    qc.ry(theta/8, qubit2)

    qc.cx(qubit1, qubit4)
    qc.h(qubit4)

    qc.cx(qubit4, qubit2)

    qc.ry(-theta/8, qubit1)
    qc.ry(theta/8, qubit2)

    qc.cx(qubit3, qubit2)
    qc.cx(qubit3, qubit1)

    qc.ry(theta/8, qubit1)
    qc.ry(-theta/8, qubit2)

    qc.cx(qubit4, qubit2)
    qc.h(qubit4)

    qc.cx(qubit1, qubit4)

    qc.ry(theta/8, qubit1)
    qc.ry(-theta/8, qubit2)

    qc.cx(qubit1, qubit2)
    qc.cx(qubit3, qubit1)

    qc.h(qubit1)
    qc.h(qubit4)

    qc.cx(qubit1, qubit3)
    qc.cx(qubit3, qubit4)

def generate_quantum_circuit_from_sci():
   
    # read the target configurations
    n_qubits, target_states, coefficients=read_target_configurations_and_coeff()

    tol=1.0e-1
    # do the excitation analysis
    filtered_states, excitations, unique_positions, remaining_positions = excitation_analysis(n_qubits, target_states, coefficients, tol)

    print('filtered_states: ',filtered_states)
    print('excitations: ',excitations)
    print('unique_position: ',unique_positions)
    print('remaining_positions: ',remaining_positions)
    # generate reference circuit
    qc = generate_circuit_from_ref_state(n_qubits, filtered_states[0])

    n_double=len(excitations)
    p_single=len(unique_positions)
    q_single=len(remaining_positions)
    n_tot=n_double + p_single * q_single

#    print('n_tot: ',n_tot)

#    sys.exit(0)
    ind_param=0
    # for the many-body coupling
    for results in excitations:
        if results[0] == 2:  # Single excitation
            excitation_ops = tuple(results[i + 1] for i in range(results[0]))
            single_excitation(qc, excitation_ops, ind_param)
            ind_param += 1
        elif results[0] == 4:  # Double excitation
            excitation_ops = tuple(results[i + 1] for i in range(results[0]))
            double_excitation(qc, excitation_ops, ind_param)
            ind_param += 1
        elif results[0] > 4:  # Mixed excitations
            for i in range(0, results[0], 4):
                if i + 4 <= results[0]:
                    excitation_ops = tuple(results[i + 1] for i in range(i, i + 4))
                    double_excitation(qc, excitation_ops, ind_param)
                    ind_param += 1
                else:
                    excitation_ops = tuple(results[i + 1] for i in range(i, results[0]))
                    single_excitation(qc, excitation_ops, ind_param)
                    ind_param += 1

    print('Num_params: ',ind_param)

#    sys.exit(0)
#    # for the orbital rotations
#    for p in unique_positions:
#        for q in remaining_positions:
#            excitation_ops = (p, q)
#            single_excitation(qc, excitation_ops, ind_param)
#            ind_param += 1
        
    return qc

qc = generate_quantum_circuit_from_sci()

from qiskit import QuantumCircuit, transpile, assemble
from qiskit_aer import AerSimulator

from qiskit.circuit import Parameter

# Assuming `qc` is the generated quantum circuit
parameter_values = np.random.random(len(qc.parameters))  # Example: Assign random values to parameters
parameter_binds = dict(zip(qc.parameters, parameter_values))

# Bind the parameters to the circuit
qc.assign_parameters(parameter_binds, inplace=True)

transpiled_qc = transpile(qc,basis_gates=['cx','ry','rz','rx','x','h'])

from qiskit import qasm2

# Export to a file.
qasm2.dump(transpiled_qc, "circuit.qasm")

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

# 绘制量子电路并获取 Matplotlib 图形对象
circuit_fig = transpiled_qc.draw(
    output='mpl',
    fold=100,
    style={'backgroundcolor': '#FFFFFF'},
    initial_state=True
)

#circuit_fig = transpiled_qc.draw()
circuit_fig.savefig('my_quantum_circuit.png')

