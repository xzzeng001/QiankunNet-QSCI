import sys
sys.path.append("../../../")  # 需要根据本机环境，将q2chemistry的目录添加至其中
import os
os.environ["OMP_NUM_THREADS"] = "8"  # 设置线程数

import q2chem
from q2chem.qcirc.circuit import QuantumCircuit
from q2chem.qcirc.gate import H, CX, X, RX, RY, RZ

import numpy as np
import matplotlib.pyplot as plt
import sys

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
#    qc = QuantumCircuit()

    ind_=0
    # target_state 
    for i, bit in enumerate(target_state[0]):
        if bit == '1':
            qc.append(X(target_qubits=[i]))
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

    qc.append(CX(target_qubits=[qubit1, qubit2]))
    qc.append(RY(target_qubits=[qubit1], parameter=q2chem.qcirc.gate.Parameter(prefactor=1.0/2, name=name_param)))
    qc.append(CX(target_qubits=[qubit2, qubit1]))
    qc.append(RY(target_qubits=[qubit1], parameter=q2chem.qcirc.gate.Parameter(prefactor=-1.0/2, name=name_param)))
    qc.append(CX(target_qubits=[qubit2, qubit1]))
    qc.append(CX(target_qubits=[qubit1, qubit2]))

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

    qc.append(CX(target_qubits=[qubit3, qubit4]))
    qc.append(CX(target_qubits=[qubit1, qubit3]))
    qc.append(H(target_qubits=[qubit1]))
    qc.append(H(target_qubits=[qubit4]))

    qc.append(CX(target_qubits=[qubit1, qubit2]))
    qc.append(CX(target_qubits=[qubit3, qubit4]))

    qc.append(RY(target_qubits=[qubit1], parameter=q2chem.qcirc.gate.Parameter(prefactor=-1.0/8, name=name_param)))
    qc.append(RY(target_qubits=[qubit2], parameter=q2chem.qcirc.gate.Parameter(prefactor=1.0/8, name=name_param)))

    qc.append(CX(target_qubits=[qubit1, qubit4]))
    qc.append(H(target_qubits=[qubit4]))

    qc.append(CX(target_qubits=[qubit4, qubit2]))

    qc.append(RY(target_qubits=[qubit1], parameter=q2chem.qcirc.gate.Parameter(prefactor=-1.0/8, name=name_param)))
    qc.append(RY(target_qubits=[qubit2], parameter=q2chem.qcirc.gate.Parameter(prefactor=1.0/8, name=name_param)))

    qc.append(CX(target_qubits=[qubit3, qubit2]))
    qc.append(CX(target_qubits=[qubit3, qubit1]))

    qc.append(RY(target_qubits=[qubit1], parameter=q2chem.qcirc.gate.Parameter(prefactor=1.0/8, name=name_param)))
    qc.append(RY(target_qubits=[qubit2], parameter=q2chem.qcirc.gate.Parameter(prefactor=-1.0/8, name=name_param)))

    qc.append(CX(target_qubits=[qubit4, qubit2]))
    qc.append(H(target_qubits=[qubit4]))

    qc.append(CX(target_qubits=[qubit1, qubit4]))

    qc.append(RY(target_qubits=[qubit1], parameter=q2chem.qcirc.gate.Parameter(prefactor=1.0/8, name=name_param)))
    qc.append(RY(target_qubits=[qubit2], parameter=q2chem.qcirc.gate.Parameter(prefactor=-1.0/8, name=name_param)))
    
    qc.append(CX(target_qubits=[qubit1, qubit2]))
    qc.append(CX(target_qubits=[qubit3, qubit1]))

    qc.append(H(target_qubits=[qubit1]))
    qc.append(H(target_qubits=[qubit4]))

    qc.append(CX(target_qubits=[qubit1, qubit3]))
    qc.append(CX(target_qubits=[qubit3, qubit4]))

def generate_quantum_circuit_from_sci(tol):
   
    # read the target configurations
    n_qubits, target_states, coefficients=read_target_configurations_and_coeff()

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

#    for p in remaining_positions[:-1]:
#        excitation_ops = (p, p+1)
#        single_excitation(qc, excitation_ops, ind_param)
#        ind_param += 1

    print('Num_params: ',ind_param)

    return qc

#qc = generate_quantum_circuit_from_sci()

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

