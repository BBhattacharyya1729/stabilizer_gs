import sys 
sys.path.append("../")
sys.path.append("../py")
from py.state import Hamiltonian, StateMachine, generate_Sright_all
from py.DMRG import *
from py.pauli import Pauli
import numpy as np
from functools import reduce
from scipy.linalg import eigh

from gf2_linalg import *
from pauli_methods_c import * 
from triple_methods import triple

### Linear Combination of Pauli Strings Hamiltonians  
class Pauli_Hamiltonian():
    
    def __init__(self,coeffs,paulis):
        """
        Initialize

        Args:
            coeffs (np.array[float]): coefficients
            paulis (list[string]): paulis
        """
        self.coeffs = coeffs
        self.paulis = paulis
    
    def convert_to_spare_hamiltonian(self):
        strings = []
        indicies = []
        for p in self.paulis:
            s = ""
            idx = []
            for (i,v) in enumerate(p):
                if(v != '_'):
                    idx.append(i)
                    s+=v 
            strings.append(s)
            indicies.append(idx)
        n = len(self.paulis[0])
        diffs = [np.max(i)-np.min(i)+1 for i in indicies]
        k = np.max(diffs)
        
        H  = Hamiltonian(n,k)
        for i in range(len(strings)):
            H.add_Pauli(Pauli.from_string(strings[i],indicies[i]),self.coeffs[i])
        
        return H 
    
    def __get_ref_energy__(stab, H):
        return sum([stab.get_energy(p) * c for p, c in H.original_paulis])

    def __stab_to_str__(stab,n_qubits):
        s1 = (stab.begin - 0) * "_"
        s2 = (n_qubits - stab.end) * "_"
        return [s1+"".join(i)+s2 for i in np.array(['_', 'X', 'Z', 'Y'])[stab.Ps.array4]]

    def get_excited_states(self,k):
        sparseH = self.convert_to_spare_hamiltonian()
        Sright_all = generate_Sright_all(sparseH)
        lists = []

        SM2 = StateMachine(sparseH, Sright_all, nexts=k)
        while True:
            SM2.evolve()
            lists.append(list(SM2.state_dict.values()))
            if SM2.m == sparseH.n:
                break
        result = SM2.generate_gs()
        result = [(stab,energy) for (stab,energy) in result if stab is not None]
        for stab, energy in result:
            assert abs(Pauli_Hamiltonian.__get_ref_energy__(stab, sparseH) - energy) < 1e-8
        
        groups = [Pauli_Hamiltonian.__stab_to_str__(r[0],sparseH.n) for r in result]
        phases = [i[0].Ps.phase   for i in result]
        energies = [r[1] for r in result]
        
            
        return [['+'+x if (1+_) else '-'+x for (x,_) in zip(g,p)] for (g,p) in zip(groups, phases)], energies 

### Tester Methods (exponential runtume because explicit vectors/matrices) ###
class test:
    paulis_dict = {
        "_": np.eye(2),
        "X": np.array([[0,1],[1,0]], dtype=complex),
        "Y": np.array([[0,-1j],[1j,0]], dtype=complex),
        "Z": np.array([[1,0],[0,-1]], dtype=complex)
    }

    @staticmethod
    def pauli_to_mat(str):
        c = 1 if str[0]=='+' else -1
        l = [test.paulis_dict[s] for s in str[1:]]
        return c * reduce(lambda a,b: np.kron(a,b), l)

    @staticmethod
    def get_state(triple):
        n = len(triple.z0)
        k = len(triple.V)
        psi = np.zeros(2**n,dtype=complex)
        
        if(k == 0):
            term = sum(b << (n - 1 - i) for i, b in enumerate(triple.z0 % 2))
            psi[term] = 1
            return psi
        
        for x in range(1 << k):       
            bits = [(x >> i) & 1 for i in range(k)] 
            coeff = (1j)**((triple.l @ bits) % 2) * (-1+0j)**(bits @ triple.Q @ bits)
            term = sum(b << (n - 1 - i) for i, b in enumerate((bits @ triple.V + triple.z0) % 2))
            psi[term] = coeff/(2**(k/2))
        return psi

### Full LCSS 
class LCSS:
    def __init__(self, H):
        self.H = H 
    
    def get_states(self, N):
        states, energies = self.H.get_excited_states(N)
        extended_states = []
        extended_energies = []
        for i,s in enumerate(states):
            extended = full_extend_group(s, self.H.paulis)
            if extended is not None:
                extended_states.append(extended)
                extended_energies.append(energies[i])
        return extended_states, extended_energies 
    
    def run_LCSS(self, N):
        print("------------extension--------------------")
        states, energies  = self.get_states(N)
        triples = [triple.get_triple(s) for s in states]
        N_states = len(states)
        print("------------overlap--------------------")
        O = np.eye(N_states,dtype=complex)
        for i in range(0,N_states):
            for j in range(i+1,N_states):
                O[i,j] = triple.inner(triples[i], triples[j])
                O[j,i] = O[i,j].conjugate()
                
        new_states= [[t.apply_pauli(p) for p in self.H.paulis] for t in triples]
        print("------------energies--------------------")
        M = np.array(np.diag(energies),dtype=complex)
        for i in range(N_states):
            for j in range(i+1,N_states):
                M[i,j] = np.dot(self.H.coeffs,[c*triple.inner(triples[i],t) for (c,t) in (new_states[j])])
                M[j,i] = M[i,j].conjugate() 
        return M, O 
    
    @staticmethod
    def basis_diagonalize(M,O):
        O_data = eigh(O)
        valid = [i for i,v in enumerate(O_data[0]) if not np.isclose(v,0)]

        U_plus = O_data[1][:, valid]
        data = eigh(U_plus.conjugate().T @ M @ U_plus, np.diag(O_data[0][valid]))

        E = data[0][0]
        v = U_plus @ data[1][:,0]
        return E,v
    
### Classical Methods 
from itertools import combinations
def ZXZ_Z_ising_model(n, g):
    init = ["_"] * n
    paulis = ["".join(init[:i] + ["Z", "X", "Z"] + init[i+3:]) for i in range(n-2)]
    paulis += ["".join(init[:i] + ["Z"] + init[i+1:]) for i in range(n)]
    coeffs = np.array([1] * (n - 2) + [g] * n)
    return coeffs, paulis

def ZXZ_ZZ_ising_model(n, g):
    init = ["_"] * n
    paulis = ["".join(init[:i] + ["Z", "X", "Z"] + init[i+3:]) for i in range(n-2)]
    paulis += ["".join(init[:i] + ["Z", "Z"] + init[i+2:]) for i in range(n-1)]
    coeffs = np.array([1] * (n - 2) + [g] * (n - 1))
    return coeffs, paulis

def ZZ_X_ising_model(n, g):
    init = ["_"] * n
    paulis = ["".join(init[:i] + ["Z", "Z"] + init[i+2:]) for i in range(n-1)]
    paulis += ["".join(init[:i] + ["X"] + init[i+1:]) for i in range(n)]
    coeffs = np.array([1] * (n - 1) + [g] * n)
    return coeffs, paulis

def XX_Z_ising_model(n, g):
    init = ["_"] * n
    paulis = ["".join(init[:i] + ["X", "X"] + init[i+2:]) for i in range(n-1)]
    paulis += ["".join(init[:i] + ["Z"] + init[i+1:]) for i in range(n)]
    coeffs = np.array([1] * (n - 1) + [g] * n)
    return coeffs, paulis


### Methods for Dealing with the Classical Part of the Hamiltonian ###
def Z_states(n, m):
    out = []
    for k in range(n, -1, -1):
        for ones in combinations(range(n), k):
            mask = 0
            for i in ones:
                mask |= 1 << (n - 1 - i) 
            out.append(mask)
            if len(out) == m:
                return out
    return out

def ZZ_states(n, m):
    basis_states = []
    
    for k in range(n):
        l = []
        for combo in combinations(range(n-1), k):
            v = [0]
            val = 0
            for i in range(n-1):
                if i in combo:
                    v.append(v[-1])
                else:
                    v.append(1 - v[-1])
            # Convert list to integer directly
            for bit in v:
                val = (val << 1) | bit
            l.append(val)
        
        for combo in combinations(range(n-1), k):
            v = [1]
            val = 0
            for i in range(n-1):
                if i in combo:
                    v.append(v[-1])
                else:
                    v.append(1 - v[-1])
            # Convert list to integer directly
            for bit in v:
                val = (val << 1) | bit
            l.append(val)
        l.sort()
        basis_states += l
        if len(basis_states) > m:
            break
    return basis_states[:m]

### Matrices for Pauli Hamiltonians in a list of computational basis states
def pack_paulis(H):
    arr = np.array([pauli_to_bin('+'+p)[:-1] for p in H.paulis])
    N, total = arr.shape
    n = (total) // 2

    # Split
    bits1 = arr[:, :n]
    bits2 = arr[:, n:2*n]

    # Precompute bit weights (MSB first)
    weights = (1 << np.arange(n-1, -1, -1, dtype=np.uint64))

    # Convert bits to integers
    A = bits1 @ weights
    B = bits2 @ weights
    return A, B 

def get_matrix(a,b, states):
    N = len(states)
    M = np.zeros((N,N), dtype=complex)
    
    for i1, s1 in enumerate(states):
        new_state = s1 ^ a
        coeff = (1j)**(bin(a & b).count('1')) * (-1)**(bin(b & s1).count('1'))
        for i2, s2 in enumerate(states):
            if s2 == new_state:
                M[i2, i1] = coeff 
    return M

def full_matrix(H, states):
    A, B = pack_paulis(H)
    N = len(states)
    out =  np.zeros((N,N), dtype=complex)
    for i, (a,b) in enumerate(zip(A, B)):
        out += H.coeffs[i] * get_matrix(a,b,states)
    return out 