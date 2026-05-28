import sys
sys.path.append("../../../")
import os
if "OMP_NUM_THREADS" not in os.environ:
    os.environ["OMP_NUM_THREADS"] = "4"

import numpy

import q2chem


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: {} qasm_filename ham_filename".format(sys.argv[0]))
        exit()

    qasm_filename = sys.argv[1]
    f = open(qasm_filename, "r")
    all_lines = f.readlines()
    f.close()
    circuit_tmp = q2chem.qcirc.circuit.parse_qasm(all_lines)

    # Construct parametric circuit from QASM's input circuit.
    # For R(t), if t satisfies:
    #     1. t / numpy.pi * 2 is not an integer,
    # then t is a trainable parameter. In addition, if ti satisfies:
    #     1. abs(ti) is in [abs(t1), abs(t2), abs(t3), ....], for example,
    #         abs(ti) = abs(t10)
    # then ti and t10 are the same parameter.
    tmp_gates = circuit_tmp.quantum_gates
    n_gates = len(tmp_gates)
    abs_param_val = []
    param_prefactor = []  # Assume the prefactor is only 1 or -1.
    new_gates = []
    for i in range(n_gates):
        gate_i: q2chem.qcirc.gate.QuantumGate = tmp_gates[i]
        # Non-RX/RY/RZ gates
        if gate_i.name not in ["RX", "RY", "RZ"]:
            new_gates.append(gate_i.copy())
            continue

        # RX/RY/RZ gates.
        param_i_ori = gate_i.parameters[0]
        param_i_val_ori = param_i_ori.get_true_value()
        is_multiple_half_pi = param_i_val_ori / numpy.pi * 2
        # RX/RY/RZ gates, but not trainable
        if numpy.isclose(is_multiple_half_pi, int(is_multiple_half_pi)):
            # Multiple of 0.5 pi. Assumed as a non-trainable gate.
            new_gates.append(gate_i.copy())
            continue
        # RX/RY/RZ gates, trainable
        param_i_val = abs(param_i_val_ori)
        param_i_prefactor = param_i_val_ori / param_i_val
        param_i_name = "p_{}".format(len(abs_param_val))
        is_new_param = True
        for j in range(len(abs_param_val)):
            # For floating-point comparison, cannot simply use 'if param_i_val in abs_param_val'.
            # Need to use numpy.isclose()
            if numpy.isclose(param_i_val, abs_param_val[j]):
                param_i_name = "p_{}".format(j)
                is_new_param = False
                break
        if is_new_param:
            abs_param_val.append(param_i_val)
        print("Found parameter {}. Value is {}, prefactor is {}.\
".format(param_i_name, param_i_val, param_i_prefactor))
        param_i = q2chem.qcirc.gate.Parameter(
            value=param_i_val,
            prefactor=param_i_prefactor,
            name=param_i_name,
            requires_grad=True)
        gate_i = gate_i.copy()
        gate_i.set_parameters([param_i])
        new_gates.append(gate_i)

    circuit = q2chem.qcirc.circuit.QuantumCircuit(quantum_gates=new_gates)
    global_phase, qasm_str = circuit.to_qasm()
    f = open("tmp_circuit_qasm.log", "w")
    f.write(qasm_str)
    f.close()
    print("The circuit read by q2chem is saved in tmp_circuit_qasm.log for testing purpose.")
    print("The circuit is:")
    circuit.draw(max_cols=120)

    simulator = q2chem.qcirc.qpu.simulator.MatrixProductStateSimulator()
    simulator.set_simulator_options(
        max_bond_dimension=128,
        cut_threshold=1e-6)
    print("simulator options:")
    print(simulator.simulator_options)
    print("mps options:")
    print(simulator.mps_options)

    ham_filename = sys.argv[2]
    ham = numpy.load(ham_filename, allow_pickle=True).reshape([1])[0]
    print('ham: ',ham)
    n_qubits = q2chem.qchem.transformation.count_qubits(ham)
    # In case that the circuit is smaller than the Hamiltonian
    if n_qubits != circuit.n_qubits:
        circuit.append(q2chem.qcirc.gate.I([n_qubits - 1]))

    param_names = circuit.parameter_names
    n_params = len(param_names)
    e_dict = {}

    def _energy(x: numpy.ndarray, with_grad: bool = False):
        simulator.reset_quantum_state()
        circuit.set_parameter_value_by_name(
            {param_names[i]: x[i] for i in range(n_params)})
        if with_grad:
            e, g = simulator.get_expectation_value_with_gradient(
                circuit=circuit,
                operator=ham,
                gradient_method="parameter_shift_rule")
            assert numpy.isclose(e.imag, 0.0)
            e = e.real
            grad = numpy.zeros(n_params)
            for i in range(n_params):
                assert numpy.isclose(g[param_names[i]].imag, 0.0)
                grad[i] = g[param_names[i]].real
            e_dict.update({hash(x.tobytes()): e})
            return e, grad
        else:
            simulator.evolve_circuit(circuit)
            e = simulator.get_expectation_value(ham)
            assert numpy.isclose(e.imag, 0.0)
            e = e.real
            print("Energy: {}".format(e))
            return e

    def _callback(x: numpy.ndarray):
        print("Energy: {}".format(e_dict[hash(x.tobytes())]))
        return

    print("Start optimization.")
    print("Number of qubits: {}".format(n_qubits))
    print("Number of parameters: {}".format(n_params))

    #x0 = numpy.array(abs_param_val)
    x0 = numpy.zeros(n_params)
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

