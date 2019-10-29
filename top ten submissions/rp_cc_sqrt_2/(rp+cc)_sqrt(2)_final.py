from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit import IBMQ, Aer, execute

# importing Qiskit
from qiskit import IBMQ, BasicAer
from qiskit.providers.ibmq import least_busy
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, execute


####################################################
# For the search problem there are 7 unset districts with id's from 0 to 6 and to encode it directly we need 14 qubits
# (7 * 2). But there are some constraints:
# 1. the 2 districts cannot be A and C so the only possible values are 01 and 11 so we can encode it in 1 qubit
# 2. for 0 and 3 we know that both of them cannot be equal to 00 so there are 9=3*3 possibilities for bot of this
#    districts but because this boot is connected they cannot be equal so we have only 6 possibilities
# 3. similar is for districts 1 and 4, and 5 and 6 so in total there are 2 * 6 * 6 * 6 = 432 possible combinations and
#    we can store the whole problem in 9 qubits ( log2(432) ), but regarding the encoding/decoding issues I've decided
#    to use 10 qubits (1 for 2 districts, 3 for 0 and 3 districts, 3 for 1 and 4, and 3 for 5 and 6
#
# So, the main idea of my solution is:
# 1. encode the problem in 10 qubits
# 2. create oracle function and constraint on 10 qubits search space
# 3. do the Grover search
# 4. decode the 10 qubits used in the algorithm to 14 qubits final solution required by the problem definition
#
# What we gain in this case, the original problem definition has 2^14 search space (16384) so the optimal number of the
# Grover iteration is 33 (but we can find the answer in 5 iterations). After the search space reduction, there is only
# 2^10 (1024) possible state so the optimal Grover iteration is 8 (and we can see the answer in 3 iterations only).
#
# The main problem, which I have was the to less number of the ancilla qubits, so in the constraints, after the calculate
# the boolean function I have to revert the operation on the temporary qubits to avoid entanglements.
#
#
####################################################
# The mapping
# q[0] => district 2
#         0 -> B
#         1 -> D
# q[1:3] => district 0 and 3
#                03
#         000 -> BC
#         001 -> BD
#         011 -> DB
#         100 -> DC
#         101 -> CD
#         111 -> CB
#
# q[4:6] => district 1 and 4
#                14
#         000 -> AD
#         001 -> AC
#         011 -> CA
#         100 -> CD
#         101 -> DC
#         111 -> DA
#
# q[7:9] => district 5 and 6
#                56
#         000 -> AB
#         001 -> AC
#         011 -> CA
#         100 -> CB
#         101 -> BC
#         111 -> BA
#
####################################################

####################################################
#                                                  #
#           Example function definitions           #
#                                                  #
####################################################
def check_allowed_state(qc, idx, out):
    """
    There are only two states which are not allowed 010 and 110 so to calculate this we need the boolean function
    below:
        not (idx[1] and not(idx[0])

    we assume that the 'out' cubit is in |1> state so ccx work for us as the NAND gate

    :param qc: Quantum Circuit
    :param idx: the index of the first qubit in the 2 districts node:
                        1 for districts 0 and 3
                        4 for districts 1 and 4
                        7 for districts 5 and 6
    :param out: the qubit where the calculated value will be stored, it has to be in |1> state before the function is
                called
    """
    # the state 010 and 110 are not allowed
    qc.x(idx)
    qc.ccx(idx, idx + 1, out)  # NAND
    # reverse x
    qc.x(idx)


def check_356(qc, _03, _56, out, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8):
    """
    For all constraint we have to do the same algorithm, I will show it on the check of node 3, 5 and 6. It is one of
    the most complicated checks. The other can be done similarly.

    1. First, we create the truth table for the constraint.
        hint: when you look on the mapping for node 03 you will see that the value of 3 districts is independent of the
              value in index 0, so we can skip this in the true table

    3  | 56  | 3 | 56 | value
    ---+-----+---+----|-------
    00 | 000 | C | AB | 1
    00 | 001 | C | AC | 0
    00 | 011 | C | CA | 0
    ---+-----+---+----|-------
    00 | 100 | C | CB | 0
    00 | 101 | C | BC | 0
    00 | 111 | C | BA | 1
    ---+-----+---+----|-------
    11 | 000 | B | AB | 0
    11 | 001 | B | AC | 1
    11 | 011 | B | CA | 1
    ---+-----+---+----|-------
    11 | 100 | B | CB | 0
    11 | 101 | B | BC | 0
    11 | 111 | B | BA | 0
    ---+-----+---+----|-------
    01 | XXX | D |    | 1
    ---+-----+---+----|-------

    So the needed fuction is:
        or (
            and ( not(_03[1]), _03[2])
            and ( not(_03[1]),
                 or (
                        and( not(_56[0]), not(_56[1]))
                        and (_56[0], _56[1])
                    )
            )
            and (_03[1],
                 and (not(_56[0]), _56[2])
            )
        )


    :param qc: Quantum Circuit
    :param _03: The index of for districts 0 and 3
    :param _56: 4 for districts 1 and 4
    :param out: the qubit where the calculated value will be stored
    :param tmp1-tmp8: the temporary qubits, we have to invert all changes of it after the calculate the output cubit
    """
    # 3 - D
    qc.x(_03 + 1)
    qc.ccx(q[_03 + 1], q[_03 + 0], tmp1)

    # 3 - C
    # 56 - AB
    qc.x(_56 + 0)
    qc.x(_56 + 2)
    qc.ccx(q[_56 + 0], q[_56 + 2], tmp2)
    qc.x(_56 + 2)
    qc.x(_56 + 0)
    # 56 -BA
    qc.ccx(q[_56 + 1], q[_56 + 2], tmp3)
    gate_or(qc, tmp2, tmp3, tmp4)
    qc.ccx(_03 + 1, tmp4, tmp5)
    qc.x(_03 + 1)

    # 3 - B
    # 56 - AC, CA
    qc.x(_56 + 2)
    qc.ccx(_56 + 2, _56 + 0, tmp6)
    qc.x(_56 + 2)
    qc.ccx(_03 + 1, tmp6, tmp7)

    gate_or(qc, tmp1, tmp5, tmp8)

    #######################################
    # Calculate the output cubit          #
    #######################################
    gate_or(qc, tmp7, tmp8, out)

    #######################################
    # Invert all changes on temporary     #
    # qubits                              #
    #######################################
    inv_gate_or(qc, tmp1, tmp5, tmp8)
    qc.ccx(_03 + 1, tmp6, tmp7)
    qc.x(_56 + 2)
    qc.ccx(_56 + 2, _56 + 0, tmp6)
    qc.x(_56 + 2)
    qc.x(_03 + 1)
    qc.ccx(_03 + 1, tmp4, tmp5)
    inv_gate_or(qc, tmp2, tmp3, tmp4)
    qc.ccx(q[_56 + 1], q[_56 + 2], tmp3)
    qc.x(_56 + 0)
    qc.x(_56 + 2)
    qc.ccx(q[_56 + 0], q[_56 + 2], tmp2)
    qc.x(_56 + 2)
    qc.x(_56 + 0)
    qc.ccx(q[_03 + 1], q[_03 + 0], tmp1)
    qc.x(_03 + 1)


####################################################
#                                                  #
#           Other function definitions             #
#                                                  #
####################################################
def gate_or(qc, in0, in1, out):
    qc.x(in0)
    qc.x(in1)
    qc.ccx(in0, in1, out)
    qc.x(out)
    qc.x(in0)
    qc.x(in1)


def inv_gate_or(qc, in0, in1, out):
    qc.x(in1)
    qc.x(in0)
    qc.x(out)
    qc.ccx(in0, in1, out)
    qc.x(in1)
    qc.x(in0)




def inv_check_allowed_state(qc, idx, out):
    qc.x(idx)
    qc.ccx(idx, idx + 1, out)
    qc.x(idx)


def check_203(qc, _2, _03, out, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6):
    # not(2) and _03 == 10
    qc.x(_2)
    qc.x(_03 + 1)
    qc.ccx(_03 + 2, _03 + 1, tmp1)
    qc.ccx(tmp1, _2, tmp2)
    qc.x(_2)
    qc.x(_03 + 1)

    # 2 and (_03 == 00 or _03 == 11)
    qc.x(_03 + 2)
    qc.x(_03 + 0)
    qc.ccx(_03 + 2, _03 + 0, tmp3)
    qc.x(_03 + 0)
    qc.x(_03 + 2)
    qc.ccx(_03 + 2, _03 + 1, tmp4)
    gate_or(qc, tmp3, tmp4, tmp5)
    qc.ccx(_2, tmp5, tmp6)
    ###
    gate_or(qc, tmp2, tmp6, out)
    ###
    qc.ccx(_2, tmp5, tmp6)
    inv_gate_or(qc, tmp3, tmp4, tmp5)
    qc.ccx(_03 + 2, _03 + 1, tmp4)
    qc.x(_03 + 2)
    qc.x(_03 + 0)
    qc.ccx(_03 + 2, _03 + 0, tmp3)
    qc.x(_03 + 0)
    qc.x(_03 + 2)
    qc.x(_03 + 1)
    qc.x(_2)
    qc.ccx(tmp1, _2, tmp2)
    qc.ccx(_03 + 2, _03 + 1, tmp1)
    qc.x(_03 + 1)
    qc.x(_2)


def inv_check_203(qc, _2, _03, out, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6):
    # not(2) and _03 == 10
    qc.x(_2)
    qc.x(_03 + 1)
    qc.ccx(_03 + 2, _03 + 1, tmp1)
    qc.ccx(tmp1, _2, tmp2)
    qc.x(_2)
    qc.x(_03 + 1)

    # 2 and (_03 == 00 or _03 == 11)
    qc.x(_03 + 2)
    qc.x(_03 + 0)
    qc.ccx(_03 + 2, _03 + 0, tmp3)
    qc.x(_03 + 0)
    qc.x(_03 + 2)
    qc.ccx(_03 + 2, _03 + 1, tmp4)
    gate_or(qc, tmp3, tmp4, tmp5)
    qc.ccx(_2, tmp5, tmp6)
    ###
    inv_gate_or(qc, tmp2, tmp6, out)
    ###
    qc.ccx(_2, tmp5, tmp6)
    inv_gate_or(qc, tmp3, tmp4, tmp5)
    qc.ccx(_03 + 2, _03 + 1, tmp4)
    qc.x(_03 + 2)
    qc.x(_03 + 0)
    qc.ccx(_03 + 2, _03 + 0, tmp3)
    qc.x(_03 + 0)
    qc.x(_03 + 2)
    qc.x(_03 + 1)
    qc.x(_2)
    qc.ccx(tmp1, _2, tmp2)
    qc.ccx(_03 + 2, _03 + 1, tmp1)
    qc.x(_03 + 1)
    qc.x(_2)

def check_01(qc, _03, _14, out, tmp1, tmp2, tmp3, tmp4, tmp5):
    gate_or(qc, q[_03 + 2], q[_03 + 1], tmp1)
    gate_or(qc, q[_14 + 2], q[_14 + 1], tmp2)
    qc.x(tmp1)
    qc.x(tmp2)
    gate_or(qc, tmp1, tmp2, tmp3)
    qc.ccx(q[_03 + 2], q[_03 + 0], tmp4)
    qc.ccx(q[_14 + 2], q[_14 + 0], tmp5)
    qc.cx(tmp4, tmp5)
    qc.x(tmp5)
    gate_or(qc, tmp3, tmp5, out)
    qc.x(tmp5)
    qc.cx(tmp4, tmp5)
    qc.ccx(q[_14 + 2], q[_14 + 0], tmp5)
    qc.ccx(q[_03 + 2], q[_03 + 0], tmp4)
    inv_gate_or(qc, tmp1, tmp2, tmp3)
    qc.x(tmp2)
    qc.x(tmp1)
    inv_gate_or(qc, q[_14 + 2], q[_14 + 1], tmp2)
    inv_gate_or(qc, q[_03 + 2], q[_03 + 1], tmp1)


def inv_check_01(qc, _03, _14, out, tmp1, tmp2, tmp3, tmp4, tmp5):
    gate_or(qc, q[_03 + 2], q[_03 + 1], tmp1)
    gate_or(qc, q[_14 + 2], q[_14 + 1], tmp2)
    qc.x(tmp1)
    qc.x(tmp2)
    gate_or(qc, tmp1, tmp2, tmp3)
    qc.ccx(q[_03 + 2], q[_03 + 0], tmp4)
    qc.ccx(q[_14 + 2], q[_14 + 0], tmp5)
    qc.cx(tmp4, tmp5)
    qc.x(tmp5)
    inv_gate_or(qc, tmp3, tmp5, out)
    qc.x(tmp5)
    qc.cx(tmp4, tmp5)
    qc.ccx(q[_14 + 2], q[_14 + 0], tmp5)
    qc.ccx(q[_03 + 2], q[_03 + 0], tmp4)
    inv_gate_or(qc, tmp1, tmp2, tmp3)
    qc.x(tmp2)
    qc.x(tmp1)
    inv_gate_or(qc, q[_14 + 2], q[_14 + 1], tmp2)
    inv_gate_or(qc, q[_03 + 2], q[_03 + 1], tmp1)


def inv_check_356(qc, _03, _56, out, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8):
    # 3 - D
    qc.x(_03 + 1)
    qc.ccx(q[_03 + 1], q[_03 + 0], tmp1)

    # 3 - C
    # 56 - AB
    qc.x(_56 + 0)
    qc.x(_56 + 2)
    qc.ccx(q[_56 + 0], q[_56 + 2], tmp2)
    qc.x(_56 + 2)
    qc.x(_56 + 0)
    # 56 -BA
    qc.ccx(q[_56 + 1], q[_56 + 2], tmp3)
    gate_or(qc, tmp2, tmp3, tmp4)
    qc.ccx(_03 + 1, tmp4, tmp5)
    qc.x(_03 + 1)

    # 3 - B
    # 56 - AC, CA
    qc.x(_56 + 2)
    qc.ccx(_56 + 2, _56 + 0, tmp6)
    qc.x(_56 + 2)
    qc.ccx(_03 + 1, tmp6, tmp7)

    gate_or(qc, tmp1, tmp5, tmp8)
    ####
    inv_gate_or(qc, tmp7, tmp8, out)
    ###
    inv_gate_or(qc, tmp1, tmp5, tmp8)
    qc.ccx(_03 + 1, tmp6, tmp7)
    qc.x(_56 + 2)
    qc.ccx(_56 + 2, _56 + 0, tmp6)
    qc.x(_56 + 2)
    qc.x(_03 + 1)
    qc.ccx(_03 + 1, tmp4, tmp5)
    inv_gate_or(qc, tmp2, tmp3, tmp4)
    qc.ccx(q[_56 + 1], q[_56 + 2], tmp3)
    qc.x(_56 + 0)
    qc.x(_56 + 2)
    qc.ccx(q[_56 + 0], q[_56 + 2], tmp2)
    qc.x(_56 + 2)
    qc.x(_56 + 0)
    qc.ccx(q[_03 + 1], q[_03 + 0], tmp1)
    qc.x(_03 + 1)

def check_314(qc, _03, _14, out, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8):
    # 3 - B
    qc.ccx(q[_03 + 1], q[_03 + 0], tmp1)

    # 3 - C
    # 14 - AB
    qc.x(_14 + 0)
    qc.x(_14 + 2)
    qc.ccx(q[_14 + 0], q[_14 + 2], tmp2)
    qc.x(_14 + 2)
    qc.x(_14 + 0)
    # 56 -BA
    qc.ccx(q[_14 + 1], q[_14 + 2], tmp3)
    gate_or(qc, tmp2, tmp3, tmp4)
    qc.x(_03 + 0)
    qc.ccx(_03 + 0, tmp4, tmp5)
    qc.x(_03 + 0)

    # 3 - D
    # 56 - AC, CA
    qc.x(_14 + 2)
    qc.ccx(_14 + 2, _14 + 0, tmp6)
    qc.x(_14 + 2)
    qc.ccx(_03 + 0, tmp6, tmp7)

    gate_or(qc, tmp1, tmp5, tmp8)
    ####
    gate_or(qc, tmp7, tmp8, out)
    ###
    inv_gate_or(qc, tmp1, tmp5, tmp8)
    qc.ccx(_03 + 0, tmp6, tmp7)
    qc.x(_14 + 2)
    qc.ccx(_14 + 2, _14 + 0, tmp6)
    qc.x(_14 + 2)
    qc.x(_03 + 0)
    qc.ccx(_03 + 0, tmp4, tmp5)
    qc.x(_03 + 0)
    inv_gate_or(qc, tmp2, tmp3, tmp4)
    qc.ccx(q[_14 + 1], q[_14 + 2], tmp3)
    qc.x(_14 + 0)
    qc.x(_14 + 2)
    qc.ccx(q[_14 + 0], q[_14 + 2], tmp2)
    qc.x(_14 + 2)
    qc.x(_14 + 0)
    qc.ccx(q[_03 + 1], q[_03 + 0], tmp1)


def inv_check_314(qc, _03, _14, out, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8):
    # 3 - B
    qc.ccx(q[_03 + 1], q[_03 + 0], tmp1)

    # 3 - C
    # 14 - AB
    qc.x(_14 + 0)
    qc.x(_14 + 2)
    qc.ccx(q[_14 + 0], q[_14 + 2], tmp2)
    qc.x(_14 + 2)
    qc.x(_14 + 0)
    # 56 -BA
    qc.ccx(q[_14 + 1], q[_14 + 2], tmp3)
    gate_or(qc, tmp2, tmp3, tmp4)
    qc.x(_03 + 0)
    qc.ccx(_03 + 0, tmp4, tmp5)
    qc.x(_03 + 0)

    # 3 - D
    # 56 - AC, CA
    qc.x(_14 + 2)
    qc.ccx(_14 + 2, _14 + 0, tmp6)
    qc.x(_14 + 2)
    qc.ccx(_03 + 0, tmp6, tmp7)

    gate_or(qc, tmp1, tmp5, tmp8)
    ####
    inv_gate_or(qc, tmp7, tmp8, out)
    ###
    inv_gate_or(qc, tmp1, tmp5, tmp8)
    qc.ccx(_03 + 0, tmp6, tmp7)
    qc.x(_14 + 2)
    qc.ccx(_14 + 2, _14 + 0, tmp6)
    qc.x(_14 + 2)
    qc.x(_03 + 0)
    qc.ccx(_03 + 0, tmp4, tmp5)
    qc.x(_03 + 0)
    inv_gate_or(qc, tmp2, tmp3, tmp4)
    qc.ccx(q[_14 + 1], q[_14 + 2], tmp3)
    qc.x(_14 + 0)
    qc.x(_14 + 2)
    qc.ccx(q[_14 + 0], q[_14 + 2], tmp2)
    qc.x(_14 + 2)
    qc.x(_14 + 0)
    qc.ccx(q[_03 + 1], q[_03 + 0], tmp1)


def check_46(qc, _14, _56, out, tmp1, tmp2, tmp3):
    # 4 -> D
    qc.x(_14 + 0)
    qc.x(_14 + 1)
    qc.ccx(_14 + 0, _14 + 1, tmp1)
    # 6 -> B
    qc.x(_56 + 0)
    qc.x(_56 + 1)
    qc.ccx(_56 + 0, _56 + 1, tmp2)
    gate_or(qc, tmp1, tmp2, tmp3)
    # 4 code xor 6 code
    qc.cx(_14 + 1, _56 + 1)
    ###
    gate_or(qc, tmp3, _56 + 1, out)
    ###
    qc.cx(_14 + 1, _56 + 1)
    # inv_gate_or(qc, tmp1, tmp2, tmp3)
    # qc.ccx(_56 + 0, _56 + 1, tmp2)
    qc.x(_56 + 1)
    qc.x(_56 + 0)
    # qc.ccx(_14 + 0, _14 + 1, tmp1)
    qc.x(_14 + 1)
    qc.x(_14 + 0)


def inv_check_46(qc, _14, _56, out, tmp1, tmp2, tmp3):
    # # 4 -> D
    qc.x(_14 + 0)
    qc.x(_14 + 1)
    # qc.ccx(_14 + 0, _14 + 1, tmp1)
    # # 6 -> B
    qc.x(_56 + 0)
    qc.x(_56 + 1)
    # qc.ccx(_56 + 0, _56 + 1, tmp2)
    # gate_or(qc, tmp1, tmp2, tmp3)
    # # 4 code xor 6 code
    qc.cx(_14 + 1, _56 + 1)
    ###
    inv_gate_or(qc, tmp3, _56 + 1, out)
    ###
    qc.cx(_14 + 1, _56 + 1)
    inv_gate_or(qc, tmp1, tmp2, tmp3)
    qc.ccx(_56 + 0, _56 + 1, tmp2)
    qc.x(_56 + 1)
    qc.x(_56 + 0)
    qc.ccx(_14 + 0, _14 + 1, tmp1)
    qc.x(_14 + 1)
    qc.x(_14 + 0)


def check_256(qc, _2, _56, out, tmp1):
    qc.x(_56 + 2)
    qc.ccx(_56 + 2, _56 + 0, tmp1)
    gate_or(qc, tmp1, _2, out)
    #qc.ccx(_56 + 2, _56 + 0, tmp1)
    qc.x(_56 + 2)


def inv_check_256(qc, _2, _56, out, tmp1):
    qc.x(_56 + 2)
    #qc.ccx(_56 + 2, _56 + 0, tmp1)
    inv_gate_or(qc, tmp1, _2, out)
    qc.ccx(_56 + 2, _56 + 0, tmp1)
    qc.x(_56 + 2)



####################################################
#                                                  #
#              Circuit definitions                 #
#                                                  #
####################################################
from datetime import datetime
print(datetime.today())

n_qubits = 32
q = QuantumRegister(n_qubits)
c = ClassicalRegister(14)
qc = QuantumCircuit(q, c)

# Circuit config
n = 10  # The number of data qubits
ite = 3 #number of iteration


_2 = 0
_03 = 1
_14 = 4
_56 = 7
oracle = n

# Temporary values
val_reg_id = oracle + 1
val_allowed_state_03 = val_reg_id
val_reg_id += 1
val_check_203 = val_reg_id
val_reg_id += 1
val_allowed_state_14 = val_reg_id
val_reg_id += 1
val_check_01 = val_reg_id
val_reg_id += 1
val_check_314 = val_reg_id
val_reg_id += 1
val_allowed_state_56 = val_reg_id
val_reg_id += 1
val_check_356 = val_reg_id
val_reg_id += 1
val_check_46 = val_reg_id
val_reg_id += 1
val_check_256 = val_reg_id
val_reg_id += 1
print("Veriable reg: ", val_reg_id)

# Temporary registers
tmp_reg_id = val_reg_id
# Zeros registers
_0_tmp1 = tmp_reg_id
tmp_reg_id += 1
_0_tmp2 = tmp_reg_id
tmp_reg_id += 1
_0_tmp3 = tmp_reg_id
tmp_reg_id += 1
_0_tmp4 = tmp_reg_id
tmp_reg_id += 1
_0_tmp5 = tmp_reg_id
tmp_reg_id += 1
_0_tmp6 = tmp_reg_id
tmp_reg_id += 1
_0_tmp7 = tmp_reg_id
tmp_reg_id += 1
_0_tmp8 = tmp_reg_id
tmp_reg_id += 1
_0_tmp9 = tmp_reg_id
tmp_reg_id += 1
_0_tmp10 = tmp_reg_id
tmp_reg_id += 1
_0_tmp11 = tmp_reg_id
tmp_reg_id += 1
print("Temporary reg: ", tmp_reg_id)



# Initialization of data qubits
qc.h(q[0:n])
qc.x(q[val_allowed_state_03])
qc.x(q[val_allowed_state_14])
qc.x(q[val_allowed_state_56])

for i in range(ite):
    # oracle part
    check_allowed_state(qc, _03, val_allowed_state_03) # 71
    check_allowed_state(qc, _14, val_allowed_state_14) # 71
    check_allowed_state(qc, _56, val_allowed_state_56) # 71
    check_01(qc, _03, _14, val_check_01, _0_tmp1, _0_tmp2, _0_tmp3, _0_tmp4, _0_tmp5)  # 820
    check_203(qc, _2, _03, val_check_203, _0_tmp1, _0_tmp2, _0_tmp3, _0_tmp4, _0_tmp5, _0_tmp6)  # 928
    check_356(qc, _03, _56, val_check_356, _0_tmp1, _0_tmp2, _0_tmp3, _0_tmp4, _0_tmp5, _0_tmp6, _0_tmp7, _0_tmp8)  #1214
    check_314(qc, _03, _14, val_check_314, _0_tmp1, _0_tmp2, _0_tmp3, _0_tmp4, _0_tmp5, _0_tmp6, _0_tmp7, _0_tmp8)  # 1214
    check_256(qc, _2, _56, val_check_256, _0_tmp11)  # 214
    check_46(qc, _14, _56, val_check_46, _0_tmp8, _0_tmp9, _0_tmp10)  # 526

    # 573
    qc.mct([q[val_allowed_state_03], q[val_allowed_state_14], q[val_allowed_state_56], q[val_check_256],
            q[val_check_46], q[val_check_356], q[val_check_314], q[val_check_01], q[val_check_203]],
           q[oracle],
           [q[_0_tmp1], q[_0_tmp2], q[_0_tmp3], q[_0_tmp4], q[_0_tmp5], q[_0_tmp6], q[_0_tmp7]],
           mode='basic')

    # inverse
    inv_check_46(qc, _14, _56, val_check_46, _0_tmp8, _0_tmp9, _0_tmp10)  # 526
    inv_check_256(qc, _2, _56, val_check_256, _0_tmp11)  # 214
    inv_check_314(qc, _03, _14, val_check_314, _0_tmp1, _0_tmp2, _0_tmp3, _0_tmp4, _0_tmp5, _0_tmp6, _0_tmp7, _0_tmp8)  # 1214
    inv_check_356(qc, _03, _56, val_check_356, _0_tmp1, _0_tmp2, _0_tmp3, _0_tmp4, _0_tmp5, _0_tmp6, _0_tmp7, _0_tmp8)  # 1214
    inv_check_203(qc, _2, _03, val_check_203, _0_tmp1, _0_tmp2, _0_tmp3, _0_tmp4, _0_tmp5, _0_tmp6)  # 928
    inv_check_01(qc, _03, _14, val_check_01, _0_tmp1, _0_tmp2, _0_tmp3, _0_tmp4, _0_tmp5)  # 820
    inv_check_allowed_state(qc, _56, val_allowed_state_56)  # 71
    inv_check_allowed_state(qc, _14, val_allowed_state_14)  # 71
    inv_check_allowed_state(qc, _03, val_allowed_state_03) # 71



    # diffusion part
    qc.h(q[0:n])
    qc.x(q[0:n])
    qc.h(q[n - 1])

    # reuse qubits
    tmp7 = val_check_01

    qc.mct([q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8]], q[9],
           [q[_0_tmp1], q[_0_tmp2], q[_0_tmp3], q[_0_tmp4], q[_0_tmp5], q[_0_tmp6], q[tmp7]], mode='basic')
    qc.h(q[n - 1])
    qc.x(q[0:n])
    qc.h(q[0:n])


####################################################
#                                                  #
#              Decode the solution                 #
#                                                  #
####################################################
cl_reg_idx = 0

####################################################
#
# The decode part is similar to the constraint check, we have to find to boolean function to map our internal state
# to the state required by the final solution. This is done in the code below.
#
####################################################

######
qc.x(_0_tmp1)
gate_or(qc, _03 + 1, _03 + 2, _0_tmp2)
qc.ccx(q[_03 + 2], q[_03 + 0], _0_tmp3)
qc.x(_0_tmp3)
qc.x(q[_03 + 1])
######
gate_or(qc, _14 + 1, _14 + 2, _0_tmp4)
qc.ccx(q[_14 + 2], q[_14 + 0], _0_tmp5)
qc.x(q[_14 + 0])
qc.x(q[_14 + 1])
######
# reuse qubits
tmp7 = val_check_01
tmp8 = _0_tmp8
tmp9 = _0_tmp9
# end of reuse
qc.ccx(q[_56 + 2], q[_56 + 0], _0_tmp6)
gate_or(qc, _56 + 1, _56 + 2, tmp7)
qc.x(_0_tmp6)
qc.ccx(_0_tmp6, tmp7, tmp8)
qc.x(_0_tmp6)
qc.x(_56 + 1)
qc.ccx(q[_56 + 1], q[_56 + 0], tmp9)
qc.x(_56)
######


# 0
qc.measure(q[_0_tmp2], c[cl_reg_idx])
cl_reg_idx += 1
qc.measure(q[_0_tmp3], c[cl_reg_idx])
cl_reg_idx += 1
# 1
qc.measure(q[_0_tmp4], c[cl_reg_idx])
cl_reg_idx += 1
qc.measure(q[_0_tmp5], c[cl_reg_idx])
cl_reg_idx += 1
# 2
qc.measure(q[_2], c[cl_reg_idx])
cl_reg_idx += 1
qc.measure(_0_tmp1, c[cl_reg_idx])
cl_reg_idx += 1
# 3
qc.measure(q[_03 + 1], c[cl_reg_idx])
cl_reg_idx += 1
qc.measure(q[_03], c[cl_reg_idx])
cl_reg_idx += 1
# 4
qc.measure(q[_14 + 1], c[cl_reg_idx])
cl_reg_idx += 1
qc.measure(q[_14 + 0], c[cl_reg_idx])
cl_reg_idx += 1
# 5
qc.measure(q[tmp8], c[cl_reg_idx])
cl_reg_idx += 1
qc.measure(q[_0_tmp6], c[cl_reg_idx])
cl_reg_idx += 1
# 6
qc.measure(q[tmp9], c[cl_reg_idx])
cl_reg_idx += 1
qc.measure(q[_56], c[cl_reg_idx])
cl_reg_idx += 1


####################################################
#                                                  #
#               Execute the JOB                    #
#                                                  #
####################################################
#print(cost(qc))

#provider = IBMQ.load_account()

backend = provider.get_backend('ibmq_qasm_simulator')
job = execute(qc, backend=backend, shots=8000, seed_simulator=12345, backend_options={"fusion_enable":True})
result = job.result()
count = result.get_counts()
print(count)

#`shots` are set to 8000 to increase sampling
#`seed_simulator`` is set to 12345 to 'lock' its value, and
#`backend_options={"fusion_enable":True}` is specified to improve simulator performance.




