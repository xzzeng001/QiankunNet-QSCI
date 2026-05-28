import sys
sys.path.append("../../../../")  # 需要根据本机环境，将q2chemistry的目录添加至其中
import os
os.environ["OMP_NUM_THREADS"] = "64"  # 设置线程数

import q2chem
from generate_circuit import generate_quantum_circuit_from_sci,generate_quantum_circuit
import numpy as np
from diag import ciGen
import time

num_shots=10000
fcidump_file='FCIDUMP'

tol=0.1
file_path='shci_results.txt'
# Construct the circuit ansatz
qc = generate_quantum_circuit_from_sci(file_path,tol)

#L=1
#qc = generate_quantum_circuit(file_path,L)

#print("The circuit is:")
#qc.draw(max_cols=120)

# set up the simulator 
simulator = q2chem.qcirc.qpu.simulator.MatrixProductStateSimulator()
simulator.set_simulator_options(
    max_bond_dimension=64,
    cut_threshold=1e-6,
    n_threads=32)

print("simulator options:")
print(simulator.simulator_options)
print("mps options:")
print(simulator.mps_options)

# read the hamiltonian
ham_filename='ham_qubit.npy'
ham = np.load(ham_filename, allow_pickle=True).reshape([1])[0]
n_qubits = q2chem.qchem.transformation.count_qubits(ham)
# In case that the circuit is smaller than the Hamiltonian
#if n_qubits != qc.n_qubits:
#    qc.append(q2chem.qcirc.gate.I([n_qubits - 1]))

param_names = qc.parameter_names
n_params = len(param_names)
e_dict = {}

def _energy(x: np.ndarray, with_grad: bool = False):
    simulator.reset_quantum_state()
    qc.set_parameter_value_by_name(
        {param_names[i]: x[i] for i in range(n_params)})
    simulator.evolve_circuit(qc)

    qubit_indices = [i for i in range(n_qubits)]

    # get the shot results
    measurement_result = simulator.measure_state(
    qubit_indices=qubit_indices,
    num_shots=num_shots)

    unique_ns_parts = list(set(measurement_result))
    unique_count=len(unique_ns_parts)

    result_strings = np.array(unique_ns_parts)

    fixed_probability = 1 + 1j
    ci_probs=[]
    abab_strings=result_strings
    ci_states=np.zeros([unique_count,n_qubits],dtype=int)
    for i in range(len(abab_strings)):
        binary_string = abab_strings[i]
        ci_states[i, :] = [int(bit) for bit in binary_string]
        ci_probs.append(fixed_probability)

    ci_probs = np.array(ci_probs)

    # 获取当前时间戳并转为整数（秒级）
    current_time = int(time.time())

    state_path='qc_sim_confs_'+str(current_time)+'.npz'
    # 保存到文件
    np.savez(state_path, ci_probs=ci_probs, ci_states=ci_states)

    e=ciGen(fcidump_file,state_path)

    print("Energy: {}".format(e),flush=True)

    return e

def _callback(x: np.ndarray):
    print("Energy: {}".format(e_dict[hash(x.tobytes())]))
    return

print("Start optimization.")
print("Number of qubits: {}".format(n_qubits))
print("Number of parameters: {}".format(n_params))

#x0 = np.random.uniform(size=(n_params))
x0 = np.zeros(n_params)
maxiter = 1000
with_grad = False
result = q2chem.utils.minimize(
    _energy,
    x0,
    args=(with_grad),
    method="L-BFGS-B" if with_grad else "COBYLA",
    jac=with_grad,
    callback=_callback if with_grad else None,
    options={"maxiter": maxiter, "disp": False})

print("Optimized energy: {}".format(result.fun))
print("Optimized parameters: ", result.x.tolist())

