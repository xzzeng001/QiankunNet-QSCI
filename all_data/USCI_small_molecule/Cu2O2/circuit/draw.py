import networkx as nx
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from collections import Counter, defaultdict

# Step 1: Read QASM file
def read_qasm_file(qasm_file):
    qc = QuantumCircuit.from_qasm_file(qasm_file)  # Read quantum circuit directly from QASM file
    return qc

# Step 2: Parse the quantum circuit and extract qubit connections
def extract_topology_from_circuit(qc):
    # Initialize the graph
    G = nx.Graph()

    # Used to record the connection order and frequency
    connections = []

    # Iterate over all quantum gates
    for gate in qc.data:
        operation, qubits, _ = gate  # Use operation and qubits to get gate-related information
        if len(qubits) == 2:  # Only handle two-qubit gates, such as CNOT gates
            qubit1 = qc.find_bit(qubits[0]).index
            qubit2 = qc.find_bit(qubits[1]).index
            # Use ordered pair (min, max) to maintain consistency in qubit connections
            edge = tuple(sorted((qubit1, qubit2)))
            G.add_edge(*edge)  # Add the connection as an edge in the graph
            connections.append(edge)  # Record the connection

    return G, connections

# Step 3: Draw the topology graph
def draw_topology_graph(G):
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=700, node_color='skyblue', font_size=12, font_weight='bold', edge_color='gray')
    plt.title('Quantum Circuit Topology')
    plt.show()

# Step 4: Print the connection order and frequency
def print_connections_info(connections):
    # Print the connection order
    print("Connection Order:")
    for i, connection in enumerate(connections, start=1):
        print(f"Connection {i}: Qubit {connection[0]} <-> Qubit {connection[1]}")

    # Calculate the frequency of each connection
    connection_counter = Counter(connections)
    print("\nFrequency of Connections Between Qubits:")
    for connection, count in connection_counter.items():
        print(f"Qubit {connection[0]} <-> Qubit {connection[1]}: {count} times")

# Step 5: Print each qubit's connections
def print_qubit_connections(connections):
    qubit_connections = defaultdict(set)

    # Collect connections for each qubit
    for qubit1, qubit2 in connections:
        qubit_connections[qubit1].add(qubit2)
        qubit_connections[qubit2].add(qubit1)

    # Print each qubit's connections
    print("\nConnections for Each Qubit:")
    for qubit, neighbors in qubit_connections.items():
        neighbors_list = ', '.join(str(neighbor) for neighbor in sorted(neighbors))
        print(f"Qubit {qubit} is connected to qubits: {neighbors_list}")

# Main function: Read QASM file, extract and draw qubit connections, print connection info
def main(qasm_file):
    qc = read_qasm_file(qasm_file)  # Read the quantum circuit
    topology_graph, connections = extract_topology_from_circuit(qc)  # Extract qubit connections
    draw_topology_graph(topology_graph)  # Draw the topology graph
    print_connections_info(connections)  # Print the connection order and frequency
    print_qubit_connections(connections)  # Print each qubit's connections

# Specify QASM file path
qasm_file = 'circuit.qasm'  # Replace with your QASM file path

# Execute the main function
main(qasm_file)

