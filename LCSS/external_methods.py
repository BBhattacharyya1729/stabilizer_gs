# from LCSS_core import * 
import stim
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from qiskit.quantum_info import SparsePauliOp

  
def circ_to_stim(circ):
    """
    Convert circuit instruction to stim circuit

    Args:
        circ (circuit instruction): circuit to be converted

    Returns:
        stim.Circuit: stim circuit
    """
    s = ("".join(["".join([str(i)+" " for i in d]) + "\n" for d in circ.data])) 
    c=stim.Circuit()
    c.append_from_stim_program_text(s)
    c.append("I",circ.n-1)
    return c

def circ_to_unitary(circ):
    """
    Convert circuit instruction to operation unitary

    Args:
        circ (circuit instruction): circuit to be converted

    Returns:
        stim.Circuit: stim circuit
    """
    c = QuantumCircuit(circ.n)
    for i in circ.data:
        if(i[0]=="H"):
            c.h(i[1])
        elif(i[0]=="S"):
            c.s(i[1])
        elif(i[0]=="CX"):
            c.cx(i[1],i[2])
        elif(i[0]=="CZ"):
            c.cz(i[1],i[2])
    return Operator(c.reverse_bits()).data

def to_matrix(H):
    coeffs = H.coeffs 
    paulis = [p for p in H.paulis]
    l = []
    for p in paulis:
        s=""
        for _ in p:
            if _ != "_":
                s+=_ 
            else:
                s+="I"
        l.append(s)
    return Operator(SparsePauliOp(coeffs=coeffs,data=l)).to_matrix()