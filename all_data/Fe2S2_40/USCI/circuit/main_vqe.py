# 导入必要的库
import numpy as np
from pyscf import gto, scf, ao2mo

qubit_hamiltonian=np.load('ham_qubit.npy',allow_pickle=True).item()

from openfermion.utils import count_qubits
num_spin_orbitals = count_qubits(qubit_hamiltonian)
num_qubits=num_spin_orbitals

# 将 OpenFermion 的 QubitOperator 转换为 Qiskit 的 PauliSumOp
#from qiskit.opflow import PauliSumOp

def openfermion_to_qiskit(qubit_op, num_qubits):
    """将 OpenFermion 的 QubitOperator 转换为 Qiskit 的 PauliSumOp。"""
    from qiskit.quantum_info import SparsePauliOp

    pauli_strings = []
    coeffs = []

    for term, coeff in qubit_op.terms.items():
        pauli_label = ['I'] * num_qubits
        for idx, pauli in term:
            pauli_label[num_qubits - idx - 1] = pauli  # Qiskit 使用小端序
        pauli_strings.append(''.join(pauli_label))
        coeffs.append(coeff)
    
    sparse_pauli = SparsePauliOp.from_list(list(zip(pauli_strings, coeffs)))
#    pauli_sum_op = PauliSumOp(sparse_pauli)
    return  sparse_pauli

qubit_hamiltonian_qiskit = openfermion_to_qiskit(qubit_hamiltonian, num_spin_orbitals)

from generate_circuit import generate_quantum_circuit_from_sci 
ansatz=generate_quantum_circuit_from_sci()

# 定义回调函数
eval_counts = []
energies = []

def vqe_callback(eval_count, parameters, mean, std):
    eval_counts.append(eval_count)
    energies.append(mean)
    print(f"Iteration: {eval_count}, Energy: {mean}")
    
# 设置优化器和 VQE 算法
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import SLSQP,ADAM
from qiskit_algorithms.optimizers import SPSA
# define Aer Estimator for noiseless statevector simulation
from qiskit_algorithms.utils import algorithm_globals
from qiskit_aer.primitives import Estimator as AerEstimator

seed = 170
algorithm_globals.random_seed = seed

noiseless_estimator = AerEstimator(
    run_options={"seed": seed, "shots": 1024},
    transpile_options={"seed_transpiler": seed},
)

optimizer = SPSA(maxiter=1000)

vqe_solver = VQE(
    noiseless_estimator,
    ansatz,
    optimizer=optimizer,
    callback=vqe_callback)

# 执行 VQE 优化
result = vqe_solver.compute_minimum_eigenvalue(operator=qubit_hamiltonian_qiskit)

# 输出结果
print('基态能量（VQE）:', result.eigenvalue.real)
#print('核排斥能:', mol.energy_nuc())
print('总能量（VQE，包括核排斥能）:', result.eigenvalue.real)

# 绘制收敛过程（可选）
import matplotlib.pyplot as plt
plt.plot(eval_counts, energies, '-o')
plt.xlabel('Iteration')
plt.ylabel('Energy')
plt.title('VQE Convergence')
plt.show()
