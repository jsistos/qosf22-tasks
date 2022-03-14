import numpy as np

# Importing standard Qiskit libraries
from qiskit import *
from qiskit.circuit.library import QFT, ZGate
from qiskit.visualization import plot_histogram

def draper_adder(value, num_qubits):
    circuit = QuantumCircuit(num_qubits)
    
    #value_bin = bin(value)[2:]
    #if num_qubits < len(value_bin):
    #    print(f"Can't add {value} using only {len(qubits)} qubits!")
    #    return None
    for i in range(num_qubits):
        circuit.rz(2 * np.pi * value / (2**(num_qubits-i)), i)

    return circuit.to_gate(label=f' Adder({value})')
    
#This is the circuit that uses QFT and Draper Adders to calculate a sum
def controlled_addition(target, values, memReg, resReg, phasReg):
    #Number of qubits in result register
    num_qubits = len(resReg)
    
    #Number of qubits in memory register
    num_values = len(memReg)
    
    #Create the circuit template
    circuit = QuantumCircuit(memReg, resReg, phasReg)
    
    #Put the result state in Fourier basis
    circuit.append(QFT(num_qubits), resReg)
    
    #Add the complement of our target
    #The complement has to shift our target state to the state composed of all 1's,
    # or 2**num_qubits - 1
    complement = 2**num_qubits - 1 - target
    circuit.append(draper_adder(complement, num_qubits), resReg)
    
    #Control the addition of the numbers in the QRAM
    for i in range(num_values):
        circuit.append(draper_adder(values[i], num_qubits).control(1), [memReg[i], *resReg])
        
    #Return the result state to computational basis
    circuit.append(QFT(num_qubits, inverse=True), resReg)

    return circuit

#This circuit combines the adder circuit, a controlled Z-Gate, and resets the result register
def reflection(target, values, memReg, resReg, phasReg):
    #Number of qubits in result register
    num_qubits = len(resReg)
    
    #Create the adder circuit, flipping the list of values to account for Qiskit's ordering
    adder_oracle = controlled_addition(target, values[::-1], memReg, resReg, phasReg)
    
    #Create circuit template
    circuit = QuantumCircuit(memReg, resReg, phasReg)
    circuit.compose(adder_oracle, inplace=True)
    
    #Use a controlled Z-Gate to change the phase of the ancillary qubit
    circuit.append(ZGate().control(num_qubits), [*resReg, *phasReg])
    
    #Finally, we reset the result register, so that its state doesn't interfere with
    # next iterations
    circuit.compose(adder_oracle.inverse(), inplace=True)
    
    return circuit

def diffuser(num_qubits):
    circuit = QuantumCircuit(num_qubits)
    
    #The Hadamard gates change the basis from the superposition state to the zero's state.
    circuit.h(range(num_qubits))
    
    #This sends the zero's state to the one's state, which is affected by the MCZ Gate.
    circuit.x(range(num_qubits))
    
    #A multi-controlled Z-Gate flips the phase
    circuit.append(ZGate().control(num_qubits - 1), range(num_qubits))
    
    #The following operations return the quantum state to the superposition state,
    # while preserving the phase flipping
    circuit.x(range(num_qubits))
    circuit.h(range(num_qubits))
    
    return circuit

#This function is the standard process to simulate a quantum circuit.
def test_circuit(circuit):
    # Use Aer's qasm_simulator
    backend_sim = Aer.get_backend('qasm_simulator')

    # Execute the circuit on the qasm simulator.
    # We've set the number of repeats of the circuit
    # to be 1024, which is the default.
    job_sim = backend_sim.run(transpile(circuit, backend_sim), shots=1024)

    # Grab the results from the job.
    result_sim = job_sim.result()

    counts = result_sim.get_counts(circuit)
    
    return counts


def general_finder(values, target, iterations):
    #Obtaining the size of the registers
    size_qram = len(values)
    
    #We will base the size of the result register on the bigger of the total sum of the values arg or the
    # sum we want to find.
    max_sum = np.amax([np.sum(values), target])
    
    #Once we determine the maximum value, we use their binary expression to know how many qubits will be needed.
    size_result = len(bin(max_sum)[2:])
    
    #Creating the registers
    memory = QuantumRegister(size_qram)
    result = QuantumRegister(size_result)
    phaser = QuantumRegister(1)
    
    #Measurement register is to measure the qram states.
    meas = ClassicalRegister(size_qram)
    
    #Initializing the circuit
    qc = QuantumCircuit(memory, result, phaser, meas)
    
    """
    STEP 1. Creating an equal superposition of all possible sums.
    """
    #Initialize equal superpostion of all possible elements to add
    qc.h(memory)

    #Put our ancilla qubit in the 1 state to flip its phase later
    qc.x(phaser)

    """
    STEP 2. A Full grover Oracle
    """
    reflector = reflection(target, values, memory, result, phaser)
    diffuse = diffuser(size_qram)

    #Grover needs approximately sqrt(2**n) iterations, but we can allow the user to control this.
    for _ in range(iterations):
        qc.append(reflector, [*memory, *result, *phaser])
        qc.append(diffuse, memory)
        
    """
    STEP 3. Finding out if our circuit works!
    """
    qc.measure(memory, meas)
    
    return qc

"""
SHORT DOCS
values - List of values given to the algorithm.
solutions - Binary strings returned by the experiment.
"""
def translate_from_qram(values, solutions):
    #Subsets is the list where we'll store all possible answers.
    subsets = []
    
    for string in solutions:
        #The value is included in the sublist only if the qubit was measured in the state 1.
        subsets.append(tuple([values[i] for i in range(len(values)) if string[i] == '1']))
        
    return subsets

"""
SHORT DOCS
values - List of values given to the algorithm.
target - Value to be found as the sum of elements in the values arg
iterations - Number of times that the Grover oracle will be repeated
net - Allows the user to choose between bitstrings, or the translated subsets from the values list.
"""
def find_solutions(values, target, iterations, net=False):
    #Run the circuit & evaluate results
    circuit = general_finder(values, target, iterations)
    counts = test_circuit(circuit)
    
    #Create an iterable from the results
    net_counts = [val for val in counts.values()]
    
    #Obtain the maximum counts & define treshold
    top_value = np.amax(net_counts)
    treshold = 3 * top_value / 5
    
    #Get the states whose counts are higher than the treshold
    solutions = [state for (state, count) in counts.items() if count >= treshold]
    
    if net:
        return solutions
    else:
        return translate_from_qram(values, solutions)

if __name__ == "__main__":
    print( "The solution to the challenge is " +
        str(find_solutions([5,7,8,9,1], 16, 4))
        )