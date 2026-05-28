import numpy as np
from openfermion import QubitOperator
from grouping import ham_grouping

hamiltonian_qubitOp = np.load('ham_qubit.npy', allow_pickle=True).item()

ham_grouping(hamiltonian_qubitOp)

# Task 1: Analyze the distribution of the coefficients
coefficients = []

for term, coeff in hamiltonian_qubitOp.terms.items():
    coefficients.append(coeff.real)  # Assuming coefficients are real numbers

# Convert the list to a numpy array for easier processing
coefficients = np.array(coefficients)

# Task 2: Analyze the lengths of the Pauli strings (excluding identities)
pauli_string_lengths = []

for term in hamiltonian_qubitOp.terms:
    # Each term is a tuple of (qubit index, Pauli operator)
    # We need to count the number of non-identity operators
    # Since identities are not explicitly stored, we can count the length directly
    # However, the identity operator is represented by an empty tuple ()
    if term == ():  # Identity term
        length = 0
    else:
        length = len(term)
    pauli_string_lengths.append(length)

# Convert the list to a numpy array
pauli_string_lengths = np.array(pauli_string_lengths)

# Output the data (coefficients and Pauli string lengths) for further use if needed
# For example, save them to files
np.savetxt('coefficients_data.txt', coefficients)
np.savetxt('pauli_string_lengths_data.txt', pauli_string_lengths)

