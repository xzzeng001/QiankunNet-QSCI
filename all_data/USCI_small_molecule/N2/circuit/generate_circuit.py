from qiskit import QuantumCircuit #, transpile, assemble, execute
from qiskit.quantum_info import Statevector
from qiskit.circuit import Parameter
import numpy as np
import matplotlib.pyplot as plt
import sys
from qiskit.quantum_info import SparsePauliOp, Pauli
from qiskit.circuit.library import PauliEvolutionGate

def read_target_configurations_and_coeff(file_path):
    # Determine the file type by checking the filename
    if 'cisd' in file_path.lower() or 'fci' in file_path.lower():
        # Read the ci coefficients from the SHCI file
        with open(file_path, 'r') as f:
            lines = f.readlines()

        target_states = []
        coefficients = []

        for line in lines:
            if "String:" in line and "Coeffs:" in line:
                parts = line.split(", Coeffs: ")
                state = parts[0].replace("String: ", "").strip()
                coeff = complex(parts[1].strip())
                target_states.append(state)
                coefficients.append(coeff.real)

        n_qubits = len(target_states[0]) if target_states else 0
        
    elif 'shci' in file_path.lower():
        # Read the ci coefficients from the SHCI file
        f=np.loadtxt(file_path,dtype=np.str_)

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
        
            rr=float(f[i,1])
            target_states.append(psi_str)
            coefficients.append(rr)
 
    else:
        raise ValueError("Unsupported file type. Please provide a valid 'cisd', 'fci', or 'shci' file.")

    return n_qubits, target_states, coefficients

def excitation_analysis(n_qubits, target_states, coefficients, tol, max_connections=16):
    # Step 1: filter the coeffs less than tol
    filtered_states = [(state, coef, idx) for idx, (state, coef) in enumerate(zip(target_states, coefficients)) if abs(coef) >= tol]

    if not filtered_states:
        return []

    excitations = []  # save the output results
    unique_diff_positions = set()
    connection_pairs = [set() for _ in range(n_qubits)]  # Track the connected qubits for each qubit

    # Helper function:
    def get_diff_positions(state1, state2):
        return [i for i in range(n_qubits) if state1[i] != state2[i]]

    # Step 2: generate all pairs of filtered states and sort by the product of their coefficients
    state_pairs = [
        (filtered_states[i], filtered_states[j], abs(filtered_states[i][1]) * abs(filtered_states[j][1]))
        for i in range(len(filtered_states))
        for j in range(i + 1, len(filtered_states))
    ]
    state_pairs.sort(key=lambda x: x[2], reverse=True)

    # Step 3: loop through sorted pairs of filtered states
    for state1, state2, _ in state_pairs:
        diff_positions = get_diff_positions(state1[0], state2[0])
        diff_count = len(diff_positions)

        # Filter out positions that already have exceeded the max connection limit
        valid_diff_positions = [pos for pos in diff_positions if len(connection_pairs[pos]) < max_connections]
        valid_diff_count = len(valid_diff_positions)

        # Skip if only one valid difference remains after applying the max connection limit
        if valid_diff_count <= 1:
            continue

#        if valid_diff_count == 0:
#            continue

        # Update connection pairs for the valid qubits involved in this excitation
        for pos in valid_diff_positions:
            for other_pos in valid_diff_positions:
                if pos != other_pos:
                    connection_pairs[pos].add(other_pos)
                    connection_pairs[other_pos].add(pos)

        # Record excitation information
        index_list = [valid_diff_count] + valid_diff_positions
        excitations.append(index_list)
        unique_diff_positions.update(valid_diff_positions)

    # Step 4: Filter out excitations with more than 4 excitations
#    excitations = [excitation for excitation in excitations if excitation[0] <= 4]

    # Get the list of positions not in unique_diff_positions
    remaining_positions = [i for i in range(n_qubits) if i not in unique_diff_positions]

    return filtered_states, excitations, list(unique_diff_positions), remaining_positions


def generate_circuit_from_ref_state(n_qubits, target_state, n_param):
    # construct n_qubits quantum circuit
    qc = QuantumCircuit(n_qubits)

    ind_=0
    # target_state 
    for i, bit in enumerate(target_state[0]):
        if bit == '1':
            qc.x(i)
            ind_ +=1

    return qc, n_param

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

def generate_quantum_circuit_from_sci(file_path, tol):
   
    # read the target configurations
    n_qubits, target_states, coefficients=read_target_configurations_and_coeff(file_path)

    # do the excitation analysis
    filtered_states, excitations, unique_positions, remaining_positions = excitation_analysis(n_qubits, target_states, coefficients, tol)

    print('filtered_states: ',filtered_states)
    print('excitations: ',excitations)
    print('unique_position: ',unique_positions)
    print('remaining_positions: ',remaining_positions)

    ind_=0
    # generate reference circuit
    qc,ind_param = generate_circuit_from_ref_state(n_qubits, filtered_states[0], ind_)

    n_double=len(excitations)
    p_single=len(unique_positions)
    q_single=len(remaining_positions)
    n_tot=n_double + p_single * q_single

#    print('n_tot: ',n_tot)

#    sys.exit(0)
#    ind_param=0
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


    for jj in range(0,n_qubits-1,2):
        excitation_ops = tuple([jj,jj+1])
        single_excitation(qc, excitation_ops, ind_param)
        ind_param += 1

    for jj in range(1,n_qubits-1,2):
        excitation_ops = tuple([jj,jj+1])
        single_excitation(qc, excitation_ops, ind_param)
        ind_param += 1

    print('Num_params: ',ind_param)

    return qc

def generate_quantum_circuit(file_path,L):
   
    # read the target configurations
    n_qubits, target_states, coefficients=read_target_configurations_and_coeff(file_path)

    ind_=0
    qc, ind_param = generate_circuit_from_ref_state(n_qubits, target_states, ind_)

    for ii in range(L):
#        for jj in range(0,n_qubits,2):
#            excitation_ops = tuple([jj,jj+1])
#            single_excitation(qc, excitation_ops, ind_param)
#            ind_param += 1

        for jj in range(0,n_qubits,4):
            excitation_ops = tuple([i for i in range(jj,jj+4)])
            double_excitation(qc, excitation_ops, ind_param)
            ind_param += 1

        for jj in range(3,n_qubits-1,4):
            excitation_ops = tuple([jj,jj+1])
            single_excitation(qc, excitation_ops, ind_param)
            ind_param += 1


    print('Num_params: ',ind_param)

    return qc


tol=0.046
file_path='shci_results.txt'
qc = generate_quantum_circuit_from_sci(file_path, tol)

#L=1 
#qc = generate_quantum_circuit(file_path,L)

from qiskit import QuantumCircuit, transpile, assemble
from qiskit_aer import AerSimulator

from qiskit.circuit import Parameter

# Assuming `qc` is the generated quantum circuit
parameter_values = np.random.random(len(qc.parameters))  # Example: Assign random values to parameters
parameter_binds = dict(zip(qc.parameters, parameter_values))

# Bind the parameters to the circuit
qc.assign_parameters(parameter_binds, inplace=True)

from qiskit.transpiler import CouplingMap

## 定义一个 4x5 的二维网格拓扑（每个量子比特最多与四个最近邻相连）
#coupling = CouplingMap([
#    [0, 1], [1, 0],   # (0, 1)相连
#    [1, 2], [2, 1],   # (1, 2)相连
#    [2, 3], [3, 2],   # (2, 3)相连
#    [3, 4], [4, 3],   # (3, 4)相连
#    [0, 5], [5, 0],   # (0, 5)相连
#    [1, 6], [6, 1],   # (1, 6)相连
#    [2, 7], [7, 2],   # (2, 7)相连
#    [3, 8], [8, 3],   # (3, 8)相连
#    [4, 9], [9, 4],   # (4, 9)相连
#    [5, 6], [6, 5],   # (5, 6)相连
#    [6, 7], [7, 6],   # (6, 7)相连
#    [7, 8], [8, 7],   # (7, 8)相连
#    [8, 9], [9, 8],   # (8, 9)相连
#    [5, 10], [10, 5], # (5, 10)相连
#    [6, 11], [11, 6], # (6, 11)相连
#    [7, 12], [12, 7], # (7, 12)相连
#    [8, 13], [13, 8], # (8, 13)相连
#    [9, 14], [14, 9], # (9, 14)相连
#    [10, 11], [11, 10], # (10, 11)相连
#    [11, 12], [12, 11], # (11, 12)相连
#    [12, 13], [13, 12], # (12, 13)相连
#    [13, 14], [14, 13], # (13, 14)相连
#    [10, 15], [15, 10], # (10, 15)相连
#    [11, 16], [16, 11], # (11, 16)相连
#    [12, 17], [17, 12], # (12, 17)相连
#    [13, 18], [18, 13], # (13, 18)相连
#    [14, 19], [19, 14], # (14, 19)相连
#    [15, 16], [16, 15], # (15, 16)相连
#    [16, 17], [17, 16], # (16, 17)相连
#    [17, 18], [18, 17], # (17, 18)相连
#    [18, 19], [19, 18]  # (18, 19)相连
#])
#
#transpiled_qc = transpile(qc, coupling_map=coupling, basis_gates=['cx','ry','rz','rx','x','h'])

transpiled_qc = transpile(qc, basis_gates=['cx','ry','rz','rx','x','h'])

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

