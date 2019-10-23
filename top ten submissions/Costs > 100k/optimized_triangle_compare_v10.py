# Team name: Costs > 100k
# Team Members:
# Ji Liu (jliu45@ncsu.edu) and Zachary Johnston (ztjohnst@ncsu.edu)
from qiskit import IBMQ
from qiskit.compiler import transpile
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import Unroller
from qiskit import execute
from qiskit import BasicAer
from qiskit.tools.visualization import plot_state_city
from qiskit.tools.visualization import plot_histogram
from qiskit.tools.visualization import circuit_drawer
from qiskit.aqua.circuits.gates import mct
import sys
import numpy as np
from collections import Counter
from qiskit.quantum_info.operators import Operator, Pauli
np.set_printoptions(threshold=sys.maxsize)


def bit_compare(circ,q):
    '''
    Logic to see if two vertices are colored differently.
    '''
    circ.cx(q[0],q[4])
    circ.cx(q[2],q[4])
    circ.cx(q[1],q[5])
    circ.cx(q[3],q[5])

def bit_compare_inverse(circ,q):
    '''
    This inverts bit_compare
    '''
    circ.cx(q[3],q[5])
    circ.cx(q[1],q[5])
    circ.cx(q[2],q[4])
    circ.cx(q[0],q[4])

def and_gate(circ,q):
    circ.ccx(q[0],q[1],q[2])
    return q[2]

def triangle_compare_2edge_dirty(circ,v1,v2,v3,ancilla_bits,output_bit):
    '''
    Does the triangle compare optimization, but only for two edges. It also leaves the ancilla bits dirty.
    '''
    ancilla_list = [ancilla_bits[0],ancilla_bits[1],output_bit]
    circ.x(output_bit)
    vertex_compare(circ, v1 + v3, ancilla_bits, output_bit)
    vertex_compare_dirty(circ, v2 + v3, ancilla_bits, output_bit)
    
    

def triangle_compare_2edge_dirty_inverse(circ,v1,v2,v3,ancilla_bits,output_bit):
    '''
    Inverts triangle_compare_2edge_dirty
    '''
    ancilla_list = [ancilla_bits[0],ancilla_bits[1],output_bit]
    vertex_compare_dirty_inverse(circ, v2 + v3, ancilla_bits, output_bit)
    vertex_compare(circ, v1 + v3, ancilla_bits, output_bit)
    circ.x(output_bit)

def triangle_compare_dirty(circ,v1,v2,v3,ancilla_bits,output_bit):
    '''
    Performs triangle compare for all edges in a triangle, and leaves the ancilla bits dirty.
    '''
    ancilla_list = [ancilla_bits[0],ancilla_bits[1],output_bit]
    bit_compare(circ,v1+v2+ancilla_bits[0:2])
    and_gate(circ,ancilla_list)
    bit_compare_inverse(circ,v1+v2+ancilla_bits[0:2])

    bit_compare(circ,v1+v3+ancilla_bits[0:2])
    and_gate(circ,ancilla_list)
    bit_compare_inverse(circ,v1+v3+ancilla_bits[0:2])

    bit_compare(circ,v3+v2+ancilla_bits[0:2])
    and_gate(circ,ancilla_list)
    return output_bit[0]


def triangle_compare_dirty_inverse(circ,v1,v2,v3,ancilla_bits,output_bit):
    '''
    Inverts triangle_compare_dirty
    '''
    ancilla_list = [ancilla_bits[0],ancilla_bits[1],output_bit]
    and_gate(circ,ancilla_list)
    bit_compare_inverse(circ,v3+v2+ancilla_bits[0:2])
    
    bit_compare(circ,v1+v3+ancilla_bits[0:2])
    and_gate(circ,ancilla_list)
    bit_compare_inverse(circ,v1+v3+ancilla_bits[0:2])

    bit_compare(circ,v1+v2+ancilla_bits[0:2])
    and_gate(circ,ancilla_list)
    bit_compare_inverse(circ,v1+v2+ancilla_bits[0:2])

def vertex_compare(circ,q,ancilla_bits,output_bit):
    '''
    This compares two vertices that are connected with an edge.
    The output bit is 0 when the two vertices are the same.
    ''' 
    bit_compare(circ,q[0:4]+ancilla_bits[0:2])
    circ.x(ancilla_bits[0])
    circ.x(ancilla_bits[1])
    circ.ccx(ancilla_bits[0], ancilla_bits[1],output_bit)
    circ.x(ancilla_bits[0])
    circ.x(ancilla_bits[1])
    circ.x(output_bit)
    bit_compare_inverse(circ,q[0:4]+ancilla_bits[0:2])
    return output_bit

def vertex_compare_inverse(circ,q,ancilla_bits,output_bit):
    '''
    Inverts vertex_compare
    '''
    bit_compare(circ,q[0:4]+ancilla_bits[0:2])
    circ.x(output_bit)
    circ.x(ancilla_bits[0])
    circ.x(ancilla_bits[1])
    circ.ccx(ancilla_bits[0], ancilla_bits[1],output_bit)
    circ.x(ancilla_bits[0])
    circ.x(ancilla_bits[1])
    bit_compare_inverse(circ,q[0:4]+ancilla_bits[0:2])
    return output_bit

def vertex_compare_dirty(circ,q,ancilla_bits,output_bit):
    '''
    This is the same as vertex_compare, but it leaves the ancilla bits dirty.
    '''
    bit_compare(circ,q[0:4]+ancilla_bits[0:2])
    circ.x(ancilla_bits[0])
    circ.x(ancilla_bits[1])
    circ.ccx(ancilla_bits[0], ancilla_bits[1],output_bit)
    circ.x(output_bit)
    return output_bit

def vertex_compare_dirty_inverse(circ,q,ancilla_bits,output_bit):
    '''
    Inverts vertex_compare_dirty
    '''
    circ.x(output_bit)
    circ.ccx(ancilla_bits[0], ancilla_bits[1],output_bit)
    circ.x(ancilla_bits[0])
    circ.x(ancilla_bits[1])
    bit_compare_inverse(circ,q[0:4]+ancilla_bits[0:2])
    return output_bit

def B_triangle(circ,v,output_bit):
    '''
    This is the logic for the edges around the B vertex initial condition.
    '''
    circ.x(v[0]) 
    circ.ccx(v[0],v[1],output_bit)
    circ.x(v[0]) 

def D_triangle(circ,v,output_bit):
    '''
    This is the logic for the edges around the D vertex initial condition.
    '''
    circ.ccx(v[0],v[1],output_bit)

def triangle_compare_B_dirty(circ,v0,v2,ancilla_bits,output_bit):
    '''
    This is the logic for the triangle around the B vertex initial condition.
    This function also leaves the ancilla bits dirty.
    '''
    B_triangle(circ,v0,output_bit)
    B_triangle(circ,v2,output_bit)
    vertex_compare_dirty(circ,v0+v2,ancilla_bits,output_bit)

def triangle_compare_B_dirty_inverse(circ,v0,v2,ancilla_bits,output_bit):
    '''
    Inverts triangle_compare_B_dirty
    '''
    vertex_compare_dirty_inverse(circ,v0+v2,ancilla_bits,output_bit)
    B_triangle(circ,v2,output_bit)
    B_triangle(circ,v0,output_bit)

def triangle_compare_D_dirty(circ,v0,v2,ancilla_bits,output_bit):
    '''
    This is the logic for the triangle around the D vertex initial condition.
    '''
    D_triangle(circ,v0,output_bit)
    D_triangle(circ,v2,output_bit)
    vertex_compare_dirty(circ,v0+v2,ancilla_bits,output_bit)

def triangle_compare_D_dirty_inverse(circ,v0,v2,ancilla_bits,output_bit):
    '''
    Inverts triangle_compare_D_dirty
    '''
    vertex_compare_dirty_inverse(circ,v0+v2,ancilla_bits,output_bit)
    D_triangle(circ,v2,output_bit)
    D_triangle(circ,v0,output_bit)

def inversion_about_average(circ,q,ancilla_bits):
    '''
    This is the inversion about the average amplitude in Grover's alorithm.
    '''
    circ.h(q[8:])
    circ.x(q[8:])
    circ.h(q[13])
    circ.mct(q[8:13],q[13],ancilla_bits,mode="basic")
    circ.h(q[13])
    circ.x(q[8:])
    circ.h(q[8:])

def encode_0123(circ, v0,v1, v2, v3):
    '''
    This function propagates the initial conditions by creating a constrained
    superposition of colors for certain vertices.
    '''
    # Contraints for vertex 2
    circ.x(v2[1])
    circ.h(v2[0])

    # Contraints for vertex 0 and 3
    circ.h(v0[1])
    circ.cx(v0[1], v3[0])
    circ.x(v0[1])
    circ.cx(v0[1], v0[0])
    circ.x(v0[1])
    circ.cx(v0[0], v3[1])
    circ.x(v2[0])
    circ.ccx(v2[0], v0[1], v0[0])
    circ.ccx(v2[0], v3[1], v3[0])
    circ.x(v2[0])

    # Contraints for vertex 1
    circ.ch(v2[0],v1[1])
    circ.cx(v1[1],v1[0])


q = QuantumRegister(14,'q')
ancilla_bits = QuantumRegister(11,'ab')
input_bit = QuantumRegister(1,'i')
output_bit = QuantumRegister(4,'o')
check_bit = QuantumRegister(2,'cb')
c = ClassicalRegister(14,'c')

circ = QuantumCircuit(q,ancilla_bits,input_bit,output_bit,check_bit,c)

circ.h(q[8:]) # Create equal superposition for vertices with no constraints.

encode_0123(circ, q[0:2],q[2:4], q[4:6], q[6:8]) # Apply constraints

# Explaination of the optimizations we used
'''
Triangle optimization explaination:
We noticed that when the graph creates a triangle there is either 1 edge with an error, or all edges have an error.
So, our triangle_compare functions use this assumption to reduce the number of output qubits needed. 
Each traingle only needs 1 output qubit that represents error (0) or no error (1).
This ultimate shrinks the MCT comparison that is made after the edge compare to detect if there are any errors.
'''

'''
Graph Reduction optimization:
Since we were given initial condition in this problem. You can propagate those condition inward on the graph,
and not have to check as many edges. For example, if you look at the A,2,C part of the graph, vertex 2
can only be B or D. So, you can Hadamard the LSB and NOT gate the MSB. Then you don't have to check the 
edges between 2,A and 2,C.
'''

'''
Other optimizations:
Since the two optimization above freed up a lot of qubits. We were able to leave a lot of dirty ancilla qubits.
This reduces the cost since we don't have to invert the computation to clean up ancilla bits.
'''

for i in range(0,5):
    # Comparison for triangle 5,6,D.
    triangle_compare_D_dirty(circ,q[10:12],q[12:14],check_bit[0:2],output_bit[0])
    # Comparison for triangle 1,4,B.
    triangle_compare_B_dirty(circ,q[2:4],q[8:10],[output_bit[1]] + [output_bit[3]],output_bit[2])
    # Comparison for 2 edges 1,5 and 2,5 in the triangle 1,2,5.
    triangle_compare_2edge_dirty(circ,q[4:6],q[6:8],q[10:12],ancilla_bits[6:8],ancilla_bits[10])
    # Comparison for last triangle left over, which is 3,4,6.
    triangle_compare_dirty(circ,q[6:8],q[8:10],q[12:14],ancilla_bits[4:6],ancilla_bits[9])

    # Comparison for egde between 2 and 6.
    vertex_compare_dirty(circ,q[4:6]+q[12:14],ancilla_bits[2:4],ancilla_bits[8])

    # Invert the phase for the correct colorings.
    circ.h(ancilla_bits[8])
    circ.mct([output_bit[0],output_bit[2],ancilla_bits[10],ancilla_bits[9]],ancilla_bits[8],[ancilla_bits[0], ancilla_bits[1]],mode='basic')
    circ.h(ancilla_bits[8])

    # Invert the comparison
    vertex_compare_dirty_inverse(circ,q[4:6]+q[12:14],ancilla_bits[2:4],ancilla_bits[8])
    triangle_compare_dirty_inverse(circ,q[6:8],q[8:10],q[12:14],ancilla_bits[4:6],ancilla_bits[9])
    triangle_compare_2edge_dirty_inverse(circ,q[4:6],q[6:8],q[10:12],ancilla_bits[6:8],ancilla_bits[10])
    triangle_compare_B_dirty_inverse(circ,q[2:4],q[8:10],[output_bit[1]] + [output_bit[3]],output_bit[2])
    triangle_compare_D_dirty_inverse(circ,q[10:12],q[12:14],check_bit[0:2],output_bit[0])

    # Perform inversion about the average amplitude.
    inversion_about_average(circ,q,ancilla_bits)

for x in range(0, 7):
    circ.measure(q[2*x],c[2*x])
    circ.measure(q[(2*x)+1],c[2*x + 1])


provider = IBMQ.load_account()
backend = provider.get_backend('ibmq_qasm_simulator')
job = execute(circ, backend=backend, shots=8000, seed_simulator=12345, backend_options={"fusion_enable":True})
result = job.result()
count = result.get_counts()
print(count)

print(circ)

k = Counter(count)

high = k.most_common(9)


print(high)


# Cost
pass_ = Unroller(['u3','cx'])
pm = PassManager(pass_)
new_circuit = pm.run(circ)
unrolled = new_circuit.count_ops()
print(unrolled)

cost = unrolled['u3']*1 + unrolled['cx']*10
print("Circuit cost: ",cost)


# Input your quantum circuit
#circuit='Input your circuit'
circuit=circ
# Input your result of the execute(groverCircuit, backend=backend, shots=shots).result()
#results = 'Input your result'
results = result
count=results.get_counts()
# Provide your team name
name='Costs > 100k'
# Please indicate the number of times you have made a submission so far. 
# For example, if it's your 1st time to submit your answer, write 1. If it's your 5th time to submit your answer, write 5.
times='14'

import json
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import Unroller

# Unroll the circuit
pass_ = Unroller(['u3', 'cx'])
pm = PassManager(pass_)
new_circuit = pm.run(circuit) 

# obtain gates
gates=new_circuit.count_ops()

#sort count
count_sorted = sorted(count.items(), key=lambda x:x[1], reverse=True)

# collect answers with Top 9 probability
ans_list = count_sorted[0:9]

# reverse ans_list
ans_reversed = []
for i in ans_list:
    ans_temp=[i[0][::-1],i[1]]
    ans_reversed.append(ans_temp)

# convert each 2 bits into corresponding color. Add node0(0),node3(1),node8(2) and node11(3)
ans_shaped = []
for j in ans_reversed:
    ans_temp=j[0]
    nodeA = 0
    node0 = int(ans_temp[0] + ans_temp[1], 2)
    node1 = int(ans_temp[2] + ans_temp[3], 2)
    nodeB = 1
    node2 = int(ans_temp[4] + ans_temp[5], 2)
    node3 = int(ans_temp[6] + ans_temp[7], 2)
    node4 = int(ans_temp[8] + ans_temp[9], 2)
    nodeC = 2
    node5 = int(ans_temp[10] + ans_temp[11], 2)
    node6 = int(ans_temp[12] + ans_temp[13], 2)
    nodeD = 3
    nodes_color = str(nodeA) + str(node0) + str(node1) + str(nodeB) + str(node2) + str(node3) + str(node4) + str(nodeC) + str(node5) + str(node6) + str(nodeD) 
    if node0 == 0 or node0 == node1 or node0 == node2 or node0 == node3 or node1 == 1 or node1 == node4 or node1 == node3 or node2 == 0 or node2 == node3 or node2 == node5 or node2 == node6 or node2 == 2 or node5 == node3 or node5 == 3 or node5 == node6 or node6 == 3 or node6 == node3 or node6 == node4 or node4 == node3 or node4 == 1:
        print("ERROR! " + nodes_color)
    ans_shaped.append([nodes_color,j[1]])

# write the result into '[your name]_final_output.txt'

filename=name+'_'+times+'_final_output.txt'
dct={'ans':ans_shaped,'costs':gates}
with open(filename, 'w') as f:
    json.dump(dct, f)
