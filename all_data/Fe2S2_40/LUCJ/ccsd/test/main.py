from qiskit import QuantumCircuit, transpile, assemble
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

def create_rotation_circuit(theta, K, J):
    # Determine the number of qubits needed
    num_qubits = max(len(K), len(J))
    qc = QuantumCircuit(num_qubits)

    # Step 1: Construct the operator matrix for |K><J| + |J><K|
    # Convert K and J from binary strings to integer indices
    K_idx = int(K, 2)
    J_idx = int(J, 2)

    # Initialize a 2^num_qubits x 2^num_qubits zero matrix
    dim = 2 ** num_qubits
    A = np.zeros((dim, dim))

    # Set the elements for |K><J| + |J><K|
    A[K_idx, J_idx] = 1
    A[J_idx, K_idx] = 1

    # Step 2: Diagonalize the matrix A
    eigenvalues, V = la.eigh(A)

    # Step 3: Construct the unitary operator e^(i * theta * A)
    D = np.diag(np.exp(1j * theta * eigenvalues))
    U = V @ D @ V.conj().T

    # Step 4: Decompose the unitary matrix into quantum gates
    qc.unitary(U, range(num_qubits))

    return qc

# Define the rotation angle theta
theta = np.pi / 4

# Define the configurations K and J (in binary string format)
K = '01'
J = '11'

# Create the quantum circuit with the specified rotation
qc = create_rotation_circuit(theta, K, J)

circuit_fig = qc.draw(
    output='mpl',
    fold=100,
    style={'backgroundcolor': '#FFFFFF'},
    initial_state=True
)

#circuit_fig = qc.draw()
circuit_fig.savefig('my_quantum_circuit.png')

# Draw the circuit
#qc.draw('mpl')
#plt.show()

## Simulate the circuit
#simulator = AerSimulator()
#job = simulator.run([qc], shots=1024)
#result = job.result()
#
## Get the unitary matrix of the circuit
#unitary = result.get_unitary(qc)
#print("Unitary matrix of the circuit:")
#print(unitary)
#
