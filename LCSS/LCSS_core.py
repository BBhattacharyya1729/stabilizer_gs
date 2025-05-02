import numpy as np 
from py.state import Hamiltonian, StateMachine, generate_Sright_all
from py.pauli import Pauli
from joblib import Parallel, delayed
import numpy as np

###RREF methods
def __swap_rows__(A,i,j):
    """
    Swap rows i and j of matrix A

    Args:
        A (np.array[int]): matrix
        i (int): index 
        j (int): index 
    """
    A[[i,j]]=A[[j,i]]
    
def __add_rows__(A,i,j):
    """
    Add two rows of a matrix

    Args:
        A (np.array[int]): matrix
        i (int): index 
        j (int): index 
    """
    A[j] = (A[j]+A[i])%2

def __elim_rows__(A, pivot_row, col,tracker):
    """
    Gaussian Elimination

    Args:
        A (np.array[int]): Matrix
        pivot_row (int): pivot index
        col (int): current column
        tracker (np.array[int]): store operations
    """
    for row in range(0, A.shape[0]):
        if(A[row, col] !=0 and row != pivot_row):
            __add_rows__(A,pivot_row,row)
            __add_rows__(tracker,pivot_row,row)
            

def __find_nonzero_row__(A, pivot_row, col):
    """
    Find next pivot

    Args:
        A (np.array[int]): Matrix
        pivot_row (int): pivot index
        col (int): current column

    Returns:
        int: row
    """
    for row in range(pivot_row, A.shape[0]):
        if A[row, col] != 0:
            return row
    return None


def __rref__(A_init):
    """
    rref of matrix
    
    Args:
        A_init (np.array[int]): matrix

    Returns:
        (np.array[int],np.array[int],int): rref matrix, product of elementary matrices, rank
    """
    A  = np.copy(A_init)
    pivot_row = 0
    tracker = np.eye(A.shape[0],dtype=int)
    for col in range(A.shape[1]):
        nonzero_row = __find_nonzero_row__(A, pivot_row, col)
        if nonzero_row is not None:
            __swap_rows__(A, pivot_row, nonzero_row)
            __swap_rows__(tracker, pivot_row, nonzero_row)
            __elim_rows__(A, pivot_row, col,tracker)
            
            pivot_row += 1
    return A,tracker,pivot_row

def __commuting_space__(A):
    """Get a basis for the symplectic complement of the column space of A setminus A

    Args:
        A (np.array[int]): matrix

    Returns:
        np.array[int]: matrix with columns as the basis vectors
    """
    P = np.block([[np.zeros((A.shape[0]//2,A.shape[0]//2)),np.eye(A.shape[0]//2)],[np.eye(A.shape[0]//2), np.zeros((A.shape[0]//2,A.shape[0]//2))]])
    r,_,_ = __rref__(A.T @ P)
    pivot = np.unique([np.argmax(i) for i in r])
    
    free = [i for i in range(len(A)) if i not in pivot]
    v_list = []
    for f in free:
        v = np.zeros(len(A))
        v[pivot] = r[:,f] 
        v[f] = 1
        v_list.append(v)
    
    full_space = np.hstack([A,np.array(v_list).T])
    r,_,_ = __rref__(full_space)
    pivot = np.unique([np.argmax(i) for i in r][len(A.T):])
    return full_space[:,[i for i in np.unique([np.argmax(i) for i in r]) if i>=len(A.T)]]

###Class for circuit representations
class circuit_instruct():

    def __init__(self,n):
        """
        Initialize

        Args:
            n (int): _description_
        """
        self.n  = n
        self.data = []

    def H(self,idx):
        """
        Hadamard gate
        
        Args:
            idx (int): index
        """
        if(isinstance(idx, int) or isinstance(idx,np.integer)):
            if(idx >= self.n ):
                return "ERROR: invalid index"
            self.data.append(("H",idx))
        else:
            for i in idx:
                self.H(i)
    
    def S(self,idx):
        """
        S gate

        Args:
            idx (int): index
        """
        if(isinstance(idx, int) or isinstance(idx,np.integer)):
            if(idx >= self.n ):
                return "ERROR: invalid index"
            self.data.append(("S",idx))
        else:
            for i in idx:
                self.S(i)
                
    def CZ(self,idx1,idx2):
        """
        CZ gate

        Args:
            idx1 (int): index 1
            idx2 (int): index 2
        """
        if(idx1 >= self.n or idx2 >= self.n):
                return "ERROR: invalid index"
        self.data.append(("CZ",idx1,idx2))
        
    def CX(self,idx1,idx2):
        """
        CZ gate

        Args:
            idx1 (int): index 1
            idx2 (int): index 2
        """
        if(idx1 >= self.n or idx2 >= self.n):
                return "ERROR: invalid index"
        self.data.append(("CX",idx1,idx2))
    
    def append(self,other_circ,before=True):
        """
        compose circuits

        Args:
            other_circ (circuit instruct): other circuit 
            before (bool, optional): whether current circuit goes before. Defaults to True.
        """
        if(before):
            self.data += other_circ.data
        else:
            self.data = other_circ.data + self.data
    
    def copy(self):
        """
        Return a new copy of current circuit

        Returns:
            circuit instruct: copy of circuit
        """
        out = circuit_instruct(self.n)
        out.data = self.data.copy() 
        return out
    
    def invert(self):
        """
        Return inverse of circuit

        Returns:
            circuit instruct: inverse of circuit
        """
        out = circuit_instruct(self.n)
        for i in self.data[::-1]:
            if(i[0] != 'S'):
                out.data.append(i)
            else:
                out.data.append(i)
                out.data.append(i)
                out.data.append(i)
        return out 

    @staticmethod
    def random_circuit(gate_list, n, depth):
        """
        Create a random 

        Args:
            gate_list (list[string]): list of allowed gates
            n (int): number of qubits
            depth (_type_): depth

        Returns:
            _type_: random circuit
        """
        circ = circuit_instruct(n)
        for _ in range(depth):
            g = np.random.randint(0,len(gate_list))
            gate_lbl = gate_list[g]
            idx = np.random.randint(0,n,len(gate_lbl))
            while(len(idx)>1 and np.all(idx == idx[0]) ):
                idx = np.random.randint(0,n,len(gate_lbl))
            getattr(circ,gate_lbl)(*idx)
        return circ
        
                    

class stabilizer_group():
    
    pauli_map = {"_":[0,0],"X":[0,1],"Y":[1,1],"Z":[1,0]}
    #bin_map = {[0,0]:"_",[0,1]:"X",[1,1]:"Y",[1,0]:"Z"}

    
    @staticmethod
    def __pauli_to_bin__(str):
        """
        Convert pauli operator to F_2 homomorphism

        Args:
            str (string): pauli string

        Returns:
            np.array: _description_
        """
        s1 = []
        s2 = []
        for i in str:
            s1.append(stabilizer_group.pauli_map[i][0])
            s2.append(stabilizer_group.pauli_map[i][1])
        return np.array(s1+s2,dtype=int)

    @staticmethod
    def __pauli_list_to_arr__(str_list):
        """
        Convert list of pauli strings to F_2 matrix

        Args:
            str_list (list[string]): list of strings

        Returns:
            np.array: F_2 array
        """
        return np.array([stabilizer_group.__pauli_to_bin__(str) for str in str_list]).T
    

    def __init__(self,str_list):
        """
        Initialize with a list of pauli strings

        Args:
            str_list (list[string]): pauli strings
        """
        self.pauli_list = str_list
        
    
    @staticmethod
    def __bad_rows__(A,r):
        """
        Given the rank of A, find the rows which are not independent 

        Args:
            A (np.array[int]): matrix
            r (int): rank

        Returns:
            list[int]: dependent rows
        """
        indices = []
        right_idx = -1
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                if(A.data[i,j] > 0 and j > right_idx):
                    right_idx = j
                    indices.append(i)
                    break
            if(len(indices) == r):
                break
        return [i for i in range(A.shape[0]) if i not in indices]


    @staticmethod
    def __had__(n,k):
        """
        F_2 representation of Hadamard conjugation
        Args:
            n (_type_): number of qubits
            k (_type_): qubit index

        Returns:
            np.array: F_2 operator
        """
        H = np.eye(2*n)
        __swap_rows__(H,k,n+k)
        return H

    @staticmethod
    def __flip__(n,k):
        """
        F_2 representation of S conjugation
        Args:
            n (_type_): number of qubits
            k (_type_): qubit index

        Returns:
            np.array: F_2 operator
        """
        F = np.eye(2*n)
        F[k,n+k]=1
        return F

    @staticmethod
    def __graph_equivalence__(stabilizer_generators):
        """
        Find the graph state which under local transformations realizes the given pauli stabilizers (ignores phase)

        Args:
            stabilizer_generators (list[string]): pauli strings

        Returns:
            (np.array[int],list[int],list[int],np.array[int]): F_2 representation of new stabilizers, indicies for S transforms, indicies for H transforms, change of basis matrix
        """
        n = len(stabilizer_generators)
        S = stabilizer_group.__pauli_list_to_arr__(stabilizer_generators)
        X_init  = S[n:,]

        _,t,r = __rref__(X_init.T)
        S_new = (S @ t.T)% 2 

        HS_new = np.copy(S_new)
        h_idx = stabilizer_group.__bad_rows__(S_new[n:],r)
        for i in h_idx:
            HS_new = (stabilizer_group.__had__(n,i) @ HS_new)%2

        _,ht,hr=__rref__(HS_new[n:])    

        if(hr!=n):
            print("Error")
            
        S_Graph = (HS_new @ ht) % 2

        diags = []
        for i in range(n):
            if(S_Graph[i,i]>0):
                diags.append(i)

        S_Graph_final = np.copy(S_Graph)     
        for i in diags:
            S_Graph_final = (stabilizer_group.__flip__(n,i) @ S_Graph_final)%2
            
        return S_Graph_final, diags, h_idx, (t.T @ ht) % 2

    @staticmethod
    def __arr_to_pauli_list__(op):
        """
        Convert F_2 array to pauli list

        Args:
            op (np.array[int]): F_2 array

        Returns:
            list[string]: pauli list
        """
        n = op.shape[-1]
        
        p = []
        for i in range(n):
            s=""
            for k in range(n):
                d = [op[k,i],op[k+n,i]]
                if(d == [0,0]):
                    s+="_"
                elif(d == [1,0]):
                    s+="Z"
                elif(d == [1,1]):
                    s+="Y"
                else:
                    s+="X"
            p.append(s)
        return p

    @staticmethod
    def __product__(p1,p2):
        """
        Compute pauli product 

        Args:
            p1 (string): pauli string 1
            p2 (string):  pauli string 1

        Returns:
            (complex, string): phase, pauli product
        """
        n = len(p1)
        order = {"X":0,"Y":1,"Z":2}
        sign = 0
        out = ""
        for i in range(n):
            s1 = p1[i]
            s2 = p2[i]
                    
            
            if(s1 == "_"):
                out+= s2  
            elif(s2 == "_"):
                out+= s1
            elif(s1==s2):
                out+="_"
            else:
                out += list(order.keys())[3-(order[s1]+order[s2])]
                diff = (order[s1]-order[s2]) % 3
                if(diff == 2):
                    sign += 1
                else:
                    sign +=3
                
        return (-1j) ** (sign), out

    @staticmethod
    def __arr_product__(p_list):
        """
        Compute pauli product of array

        Args:
            p_list (list[string]): pauli strings
            
        Returns:
            (complex, string): phase, pauli product
        """
        if(len(p_list) == 1):
            return 1, p_list[0]
        
        sign, out = stabilizer_group.__product__(p_list[0],p_list[1])
        for i in range(2,len(p_list)):
            s, out = stabilizer_group.__product__(out,p_list[i])
            sign *= s
        return sign, out


    @staticmethod
    def __basis_signs__(p_list,basis):
        """
        Find signs generated from products in basis change

        Args:
            p_list (list[string]): paulis list
            basis (np.array[int]): basis change matrix

        Returns:
            np.array: list of signs
        """
        signs = []
        for i in basis:
            basis_paulis = []
            for (j,v) in enumerate(i):
                if(v):
                    basis_paulis.append(p_list[j])
            signs.append(stabilizer_group.__arr_product__(basis_paulis)[0])
        return np.array(np.array(signs).real,dtype=int)


    def make_circuit(self,output_signs):
        """
        Construct a circuit satisfying the pauli strings as stabilizers with given signs

        Args:
            output_signs (np.array[int]): List of signs (+ or -) for each string

        Returns:
            circuit instruct: circuit
        """
        n = len(self.pauli_list)
        SG,diags,h_idx,change = stabilizer_group.__graph_equivalence__(self.pauli_list)
        adj = SG[:n]
        
        
        circ = circuit_instruct(n)
        circ.H(range(n))

        for i in range(n):
            for j in range(i+1,n):
                if(adj[i,j]>0):
                    circ.CZ(i,j)
        
        op = np.copy(SG)
        
        for d in diags:
            op = stabilizer_group.__flip__(n,d) @ op

        for h in h_idx:
            op = stabilizer_group.__had__(n,h) @ op
        
        basis = __rref__(change)[1].T
        signs = stabilizer_group.__basis_signs__(stabilizer_group.__arr_to_pauli_list__(op),basis)
        z_idx = (change.T@((1-output_signs)//2  - (1-signs)//2))%2

        for i,v in enumerate(z_idx):
            if(v):
                circ.S(i)
                circ.S(i)
            

        circ.S(diags)
        circ.H(h_idx)

        
        return circ

### Class for CH simulation
class CHform():
    
    @staticmethod
    def __get_abc__(vq,yq,zq,delta):
        """
        Reduce operators of the form H^vq (|yq> + i^delta |zq>)

        Args:
            vq (int): vq
            yq (int): yq
            zq (int): zq
            delta (int): delta

        Returns:
            (complex,int,int,int): data for the omega * S^a H^b |c> form
        """
        omega = 1
        a=0
        b=0
        c=0
        
        if(yq==zq):
            return "ERROR yq and zq same"
        
        psi =  np.zeros(2,dtype=complex)
        psi[zq] = (1j)**delta
        psi[yq]  = 1
        
        H = np.array([[1,1],[1,-1]])/np.sqrt(2)
        
        if(vq):
            psi = H @ psi
        
        if(psi[0] != 0 and psi[1] !=0 ):
            omega = psi[0] 
            psi/= omega
            b=1
            omega*=np.sqrt(2)
            
            if(psi[1] == 1j):
                a=1
            elif(psi[1] == -1):
                c = 1
            elif(psi[1] == -1j):
                a=1
                c=1
            
        elif(psi[0] == 0):
            omega = psi[1]
            c=1
        
        elif(psi[1] == 0):
            omega = psi[0]
        
        return omega , a, b, c
    
    @staticmethod
    def __Vc__(v,t,u):
        """
        Compute Vc operator given |t>, |u> and v

        Args:
            v (np.array[int]): v
            t (np.array[int]): t
            u (np.array[int]): u

        Returns:
            (circuit instruct, int): Vc, q
        """
        n = len(t)
        if(np.all(t==u)):
            return "ERROR: identical states"

        Vc = circuit_instruct(n)
        q = -1

        V0  = []
        V1 = []
        for i in range(n):
            if(t[i] != u[i]):
                if(v[i] == 0):
                    V0.append(i)
                else:
                    V1.append(i)


        if(len(V0) > 0):
            q = V0[0]
            for i in V1:
                Vc.CZ(q,i)
            for i in V0[1:]:
                Vc.CX(q,i)
        else:
            q = V1[0]
            for i in V1[1:]:
                Vc.CX(i,q)
        
        return Vc, q
    
    @staticmethod
    def __yz__(t,u,q):
        """
        Compute y,z states given |t>, |u> and q

        Args:
            t (np.array[int]): t
            u (np.array[int]): u
            q (np.array[int]): q


        Returns:
            (np.array[int],np.array[int]) y,q
        """
        if(t[q] == 1):
            y = np.copy(u)
            y[q] = (y[q]+1)%2    
            z=np.copy(u)
            return y,z
        else:
            y=np.copy(t)
            z=np.copy(t)
            z[q] = (z[q]+1)%2  
            return y,z
    
    @staticmethod
    def __had_transform__(v,t,u,delta):
        """
        Reduce H_v(|t> + i^delta |u>)
        Args:
            v (np.array[int]): v
            t (np.array[int]): t
            u (np.array[int]): u
            delta (int): delta

        Returns:
            (complex,circuit instruct,np.array[int],np.array[int]): omega, Wc, v_new, s_new
        """
        n = len(v)
        Vc,q = CHform.__Vc__(v,t,u)
        y,z = CHform.__yz__(t,u,q)
        
        vq = v[q]
        yq = y[q]
        zq = z[q]
        
        omega,a,b,c = CHform.__get_abc__(vq,yq,zq,delta)

        s_new = np.copy(y)
        s_new[q] = c
        
        v_new = np.copy(v)
        v_new[q] = b % 2
        
        Wc = circuit_instruct(n)
        if(a):
            Wc.S(q)
        
        Wc.append(Vc)
        
        return omega, Wc, v_new, s_new
    
    def __init__(self,n):
        """
        Initialize CH form data

        Args:
            n (int): number pf qubits
        """
        self.n = n
        self.control_circ = circuit_instruct(n)
        
        self.G = np.eye(n,dtype=int)
        self.F = np.eye(n,dtype=int)
        self.M = np.zeros((n,n),dtype = int)
        
        self.g = np.zeros(n,dtype=int)
        self.v = np.zeros(n,dtype=int)
        self.s = np.zeros(n,dtype=int)
        
        self.omega = 1+0j
    
    def __copy_data__(self):
        """
        Copy current data

        Returns:
            np.array[int],np.array[int],np.array[int],np.array[int],np.array[int],np.array[int]: G,F,M,gamma,v,s
        """
        return np.copy(self.G),np.copy(self.F),np.copy(self.M),np.copy(self.g),np.copy(self.v),np.copy(self.s)    
        
    def S(self,q,left=True):
        """
        Apply S gate

        Args:
            q (int): index
            left (bool, optional): left multiply (as opposed to right). Defaults to True.
        """
        oldG,oldF,oldM,oldg,oldv,olds = self.__copy_data__()
        
        if(left):
            self.M[q] = (oldM[q] + oldG[q]) % 2
            self.g[q] = (oldg[q]-1)%4
            self.control_circ.S(q)
 
            
        else:
            self.M[:,q] = (oldM[:,q] + oldF[:,q]) % 2
            self.g = (oldg-oldF[:,q]) % 4
        
        
    def CZ(self,q,r,left=True):
        """
        Apply CZ gate

        Args:
            q (int): index
            r (int): index
            left (bool, optional): left multiply (as opposed to right). Defaults to True.
        """
        oldG,oldF,oldM,oldg,oldv,olds = self.__copy_data__()
        
        if(left):
            self.M[q] = (oldM[q]+oldG[r])%2
            self.M[r] = (oldM[r]+oldG[q])%2
            self.control_circ.CZ(q,r)
        else:
            self.M[:,q] = (oldM[:,q]+oldF[:,r]) % 2
            self.M[:,r] = (oldM[:,r]+oldF[:,q]) % 2
            self.g = (oldg+ 2 * oldF[:,q]  * oldF[:,r]) % 4

    def CX(self,q,r,left=True):
        """
        Apply CX gate

        Args:
            q (int): index
            r (int): index
            left (bool, optional): left multiply (as opposed to right). Defaults to True.
        """
        oldG,oldF,oldM,oldg,oldv,olds = self.__copy_data__()
        
        if(left):
            self.G[r] = (oldG[r]+oldG[q]) % 2
            self.F[q] = (oldF[q]+oldF[r]) % 2
            self.M[q] = (oldM[q]+oldM[r]) % 2
            self.g[q] = (oldg[q]+oldg[r]+2*(oldM @ oldF.T)[q,r]) % 4
            self.control_circ.CX(q,r)

        else:
            self.G[:,q] = (oldG[:,q] +oldG[:,r]) % 2
            self.F[:,r] = (oldF[:,r]+oldF[:,q]) % 2
            self.M[:,q] = (oldM[:,q]+oldM[:,r]) % 2
                

        
    def H(self, p):
        """
        Apply H gate

        Args:
            p (int): index
        """
        
        t = (self.s + self.G[p] * self.v)%2 
        u = (self.s+ self.F[p]*(1-self.v)+self.M[p] * self.v) %2 
        
        alpha = np.sum(self.G[p] * (1-self.v) * self.s)%2
        beta = np.sum(self.M[p]*(1-self.v)*self.s+self.F[p]*self.v*(self.M[p]+self.s)) % 2
            
        if(np.all(t==u)):
            self.s = t
            self.omega *= 1/np.sqrt(2) * ((-1)**alpha + (1j)**self.g[p]*(-1)**beta)
        
        else:
            omega, Wc, v_new, s_new = CHform.__had_transform__(self.v,t,u,(self.g[p]+2 * (alpha+beta)) % 4)
            
            self.omega = 1/np.sqrt(2) * (-1)**alpha * omega * self.omega
            
            for wc in Wc.data[::-1]:
                getattr(self,wc[0])(*wc[1:],False)
            self.v = v_new 
            self.s = s_new 
            self.control_circ.append(Wc,before=False)
            
        
    @staticmethod
    def from_circuit(circ):
        """
        Get CH form from circuit

        Args:
            circ (circuit instruct): circuit

        Returns:
            CHform: obtained CH form
        """
        out = CHform(circ.n)
        for c in circ.data:
            getattr(out,c[0])(*c[1:])
        return out     
        
    @staticmethod
    def inner(circ1,circ2):
        """
        Inner product of two circuits <0|circuit 1 | circuit 2|0>

        Args:
            circ1 (circuit instruct): circuit 1
            circ2 (circuit instruct): circuit 2

        Returns:
            complex: value of product
        """
        inner_circ = circ1.invert()
        inner_circ.append(circ2,before=False)

        CH = CHform.from_circuit(inner_circ)
        
        prod = 1
        for i in range(CH.n):
            if(CH.v[i] == 0 and CH.s[i]==1):
                prod *= 0 
                break
            elif(CH.v[i]==1):
                prod *= 1/np.sqrt(2) 

        return prod* CH.omega
    
    @staticmethod
    def pauli_matrix_element(circ1,circ2,pauli_string):
        """
        Pauli element of two circuits <0|circuit 1 | P |circuit 2|0>

        Args:
            circ1 (circuit instruct): circuit 1
            circ2 (circuit instruct): circuit 2
            pauli_string (string) : pauli string

        Returns:
            complex: value of product
        """
        pauli_circ = circuit_instruct(circ1.n)
        sign = 1
        for i,v in enumerate(pauli_string):
            if(v=="X"):
                pauli_circ.H(i)
                pauli_circ.S(i)
                pauli_circ.S(i)
                pauli_circ.H(i)
            if(v=="Y"):
                pauli_circ.S(i)
                pauli_circ.H(i)
                pauli_circ.S(i)
                pauli_circ.S(i)
                pauli_circ.H(i)
                pauli_circ.S(i)
                pauli_circ.S(i)
                pauli_circ.S(i)
                sign *= -1
            if(v=="Z"):
                pauli_circ.S(i)
                pauli_circ.S(i)
        
        pauli_circ2 = circ2.copy()
        pauli_circ2.append(pauli_circ)
        return CHform.inner(circ1,pauli_circ2) * sign


###Class for linear combo of pauli hamiltonians
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
        
    @staticmethod
    def random_pauli_generators(n_groups,n,depth = 50):
        """
        Get "random" pauli generator groups by simulating a circuit of a given depth and size

        Args:
            n_groups (int): number of groups
            n (int): number of qubits
            depth (int, optional): circuit depth. Defaults to 50.

        Returns:
            (list[list[string],np.array[int]]): group generators and associated random +/- signs
        """
        groups = []
        for _ in range(n_groups):
            pauli_strings = []
            C = circuit_instruct.random_circuit(["CZ","S","CX","H"], n,depth)
            CH = CHform.from_circuit(C)
            
            for i in range(CH.n):
                s= ""
                for j in range(CH.n):
                    if(CH.F[i][j] and CH.M[i][j]):
                        s+="Y"
                    elif(CH.F[i][j]):
                        s+="X"
                    elif(CH.M[i][j]):
                        s+="Z"
                    else:
                        s+="_"
                pauli_strings.append(s)
            groups.append((pauli_strings,1-np.random.randint(0,2,n)*2))
        return groups

    @staticmethod
    def __groups_to_circs__(groups):
        """
        Get circuits from generators and signs

        Args:
            groups (list[list[string],np.array[int]]): group generators and associated +/- signs

        Returns:
            list[circuit instruct]: list of circuits
        """
        circs = []
        for i in groups:
            sg = stabilizer_group(i[0])
            circs.append(sg.make_circuit(i[1]))
        return circs

    @staticmethod
    def __overlap_elements__(circs):
        """
        Find overlap elements

        Args:
            circs (list[circuit instruct]): circuits for each state

        Returns:
            np.array[complex]: Overlap matrix with O_ij = <s_i | s_j>
        """
        output = np.zeros((len(circs),len(circs)),dtype=complex)
        for i in range(len(circs)):
            for j in range(i+1,len(circs)):
                output[i,j] = CHform.inner(circs[i],circs[j])
        output = output+output.conjugate().T + np.diag(np.ones(len(output)))
                
        return output 
    
    
    @staticmethod
    def __p_overlap_elements__(circs):
        """
        Find overlap elements

        Args:
            circs (list[circuit instruct]): circuits for each state

        Returns:
            np.array[complex]: Overlap matrix with O_ij = <s_i | s_j>
        """
        n = len(circs)
        output = np.zeros((n, n), dtype=complex)

        # Define task
        def compute_overlap(i, j):
            return i, j, CHform.inner(circs[i], circs[j])

        # Parallel computation over upper triangle (i < j)
        results = Parallel(n_jobs=-1)(
            delayed(compute_overlap)(i, j)
            for i in range(n)
            for j in range(i + 1, n)
        )

        # Fill the output matrix
        for i, j, val in results:
            output[i, j] = val

        # Complete the Hermitian matrix
        output = output + output.conjugate().T + np.diag(np.ones(n))
        
        return output
    

    @staticmethod
    def __pauli_elements__(circs,p):
        """
        Find pauli matrix elements

        Args:
            circs (list[circuit instruct]): circuits for each state

        Returns:
            np.array[complex]: pauli matrix with P_ij = <s_i | P | s_j>
        """
        output = np.zeros((len(circs),len(circs)),dtype=complex)
        for i in range(len(circs)):
            for j in range(i,len(circs)):
                output[i,j] = CHform.pauli_matrix_element(circs[i],circs[j],p)
                if(i != j):
                    output[j,i] = output[i,j].conjugate()
        return output
    
    
    @staticmethod
    def __p_pauli_elements__(circs, p):
        """
        Find pauli matrix elements

        Args:
            circs (list[circuit instruct]): circuits for each state

        Returns:
            np.array[complex]: pauli matrix with P_ij = <s_i | P | s_j>
        """
        n = len(circs)
        output = np.zeros((n, n), dtype=complex)

        # Task for parallel execution
        def compute_pauli_element(i, j):
            val = CHform.pauli_matrix_element(circs[i], circs[j], p)
            return (i, j, val)

        # Parallel computation over upper triangle (i <= j)
        results = Parallel(n_jobs=-1)(
            delayed(compute_pauli_element)(i, j)
            for i in range(n)
            for j in range(i, n)
        )

        # Fill the output matrix
        for i, j, val in results:
            output[i, j] = val
            if i != j:
                output[j, i] = val.conjugate()

        return output

    def __total_elements__(self, circs):
        """
        Find full matrix elements

        Args:
            circs (list[circuit instruct]): circuits for each state

        Returns:
            np.array[complex]: matrix with M_ij = <s_i | H | s_j>
        """
        pauli_elements = [Pauli_Hamiltonian.__pauli_elements__(circs,p) for p in self.paulis]
        return np.einsum("ijk,i->jk",pauli_elements,self.coeffs)
    
    def __p_total_elements__(self, circs):
        """
        Find full matrix elements

        Args:
            circs (list[circuit instruct]): circuits for each state

        Returns:
            np.array[complex]: matrix with M_ij = <s_i | H | s_j>
        """
        pauli_elements = Parallel(n_jobs=-1)(
            delayed(Pauli_Hamiltonian.__pauli_elements__)(circs, p) 
            for p in self.paulis
        )
        return np.einsum("ijk,i->jk", pauli_elements, self.coeffs)
            
    def LCSS_opt(self, circs):
        """
        Optimize over linear combination of stabilizer states

        Args:
            circs (list[circuit instruct]): list of circuits for each state

        Returns:
            (float,np.array[complex]): minimum value, coefficients
        """
        O = Pauli_Hamiltonian.__overlap_elements__(circs)
        M = self.__total_elements__(circs)
        L = np.linalg.cholesky(O+1e-14 * np.eye(len(O))).conjugate().T

        U = np.linalg.inv(L).conjugate().T @ M @ np.linalg.inv(L)
        return np.linalg.eigvalsh(U)[0],np.linalg.inv(L) @ np.linalg.eigh(U)[1][:,0]
    
    def p_LCSS_opt(self, circs):
        """
        Optimize over linear combination of stabilizer states

        Args:
            circs (list[circuit instruct]): list of circuits for each state

        Returns:
            (float,np.array[complex]): minimum value, coefficients
        """
        O = Pauli_Hamiltonian.__p_overlap_elements__(circs)
        M = self.__p_total_elements__(circs)
        L = np.linalg.cholesky(O+1e-14 * np.eye(len(O))).conjugate().T

        U = np.linalg.inv(L).conjugate().T @ M @ np.linalg.inv(L)
        return np.linalg.eigvalsh(U)[0],np.linalg.inv(L) @ np.linalg.eigh(U)[1][:,0]
                
                
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
############################################


bin_map = {(0,0):"_",(0,1):"X",(1,1):"Y",(1,0):"Z"}

def __bin_to_paulis__(v):
    # print(v)
    return "".join([(bin_map[tuple(i)]) for i in np.reshape(v,(2,len(v)//2)).T])


from copy import copy
def __solveable_system__(A,b):
    r,m,_  = __rref__(A)
    r_zeros = [np.all(i==0) for i in r]
    sol = (m @ np.array(b)) % 2 
    return np.all(sol[r_zeros]==0)

def __count_overlap__(L,p_list):
    A= stabilizer_group.__pauli_list_to_arr__(L)
    c=0
    for p in p_list:
        b = stabilizer_group.__pauli_to_bin__(p)
        c+=int(__solveable_system__(A,b))
    return c
    
def __extra_overlap__(L,s,p_list):
    __count_overlap__(L,p_list)
    __count_overlap__(L+[s],p_list)
    return __count_overlap__(L,p_list) != __count_overlap__(L+[s],p_list)

def __indepenent__(L,s):
    A = stabilizer_group.__pauli_list_to_arr__(L+[s])
    return __rref__(A)[-1] == A.shape[-1]

def __enumerate_span__(v_list,idx):
    coeff = [idx//(2**i) % 2 for i in range(len(v_list))[::-1]]
    return np.dot(coeff,v_list)

def extend_group(L, p_list,n):
    L_new = copy(L)
    fail = False
    while(len(L_new) < n):
        space = __commuting_space__(stabilizer_group.__pauli_list_to_arr__(L_new)).T
        for idx in range(1,2**(len(space))):
            s = __bin_to_paulis__(__enumerate_span__(space,idx))
            if(not __extra_overlap__(L_new,s,p_list) and __indepenent__(L_new,s)):
                L_new.append(s)
                break 
            if(idx == 2**(len(space))-1 ):
                fail = True 
        if(fail):
            break
    return L_new,fail

######## 

def __get_ref_energy__(stab, H):
    return sum([stab.get_energy(p) * c for p, c in H.original_paulis])

def __stab_to_str__(stab,n_qubits):
    s1 = (stab.begin - 0) * "_"
    s2 = (n_qubits - stab.end) * "_"
    return [s1+"".join(i)+s2 for i in np.array(['_', 'X', 'Z', 'Y'])[stab.Ps.array4]]

def get_excited_states(sparseH,nexts):
    Sright_all = generate_Sright_all(sparseH)
    lists = []

    SM2 = StateMachine(sparseH, Sright_all, nexts=nexts)
    while True:
        SM2.evolve()
        lists.append(list(SM2.state_dict.values()))
        if SM2.m == sparseH.n:
            break
    result = SM2.generate_gs()
    for stab, energy in result:
        assert abs(__get_ref_energy__(stab, sparseH) - energy) < 1e-8
    
    groups = [__stab_to_str__(r[0],sparseH.n) for r in result]
    phases = [ np.concatenate( [i[0].Ps.phase,np.ones(sparseH.n-len(i[0].Ps.phase))])   for i in result]
    energies = [r[1] for r in result]
    
    return result, groups, phases, energies 

def get_circs(groups,phases,H):
    groups = [extend_group(g,H.paulis,len(H.paulis[0])) for g in groups]
    groups  = [g[0] for g in groups if not g[1]]
    circs = H.__groups_to_circs__(zip(groups,phases))
    return circs

def LCSS_full(H,n):
    result,groups,phases,energies = get_excited_states(H.convert_to_spare_hamiltonian(),n)
    circs = get_circs(groups,phases,H)
    return H.LCSS_opt(circs),circs,groups

def p_LCSS_full(H,n):
    result,groups,phases,energies = get_excited_states(H.convert_to_spare_hamiltonian(),n)
    circs = get_circs(groups,phases,H)
    return H.p_LCSS_opt(circs),circs,groups


##### Classical Methods
from itertools import combinations
from copy import deepcopy

def is_Z(s):
    return np.all(s[len(s)//2:] == 0)

def bit_extend_group(L, p_list,n):
    L_new = copy(L)
    fail = False
    while(len(L_new) < n):
        space = __commuting_space__(stabilizer_group.__pauli_list_to_arr__(L_new)).T
        space = [s for s in space if is_Z(s)]
        for idx in range(1,2**(len(space))):
            s = __bin_to_paulis__(__enumerate_span__(space,idx))
            if(not __extra_overlap__(L_new,s,p_list) and __indepenent__(L_new,s)):
                L_new.append(s)
                break 
            if(idx == 2**(len(space))-1 ):
                fail = True 
        if(fail):
            break
    return L_new,fail


def get_bitstring(Z_list, signs):
    bit_signs = (1 - np.array(signs,dtype=int))//2
    matrix = np.zeros((len(Z_list), len(Z_list)))
    for i in range(len(Z_list)):
        for j in range(len(Z_list)):
            matrix[i,j] = int(Z_list[i][j] == "Z")
            
    _,inv,r = __rref__(matrix)
    assert r == len(Z_list) ##full rank!
    return (inv @ bit_signs) % 2

def get_energy(coeffs, Z_matrix,bitstring):
    return np.dot(coeffs, (-1)**((Z_matrix @ bitstring ) % 2))

def get_local_excitation_states(HZ,k,n_states):
    assert k < len(HZ.paulis[0])//2
    result, groups, phases, energies  = get_excited_states(HZ.convert_to_spare_hamiltonian(),1)
    groups = bit_extend_group(groups[0],HZ.paulis,len(HZ.paulis[0]))
    assert not groups[-1]
    groups = groups[0]
    coeffs = HZ.coeffs 
    Z_list = groups
    phases = np.concatenate([phases[0],np.ones(len(Z_list) - len(phases[0]))])
    gs = get_bitstring(Z_list,phases)
    
    Z_matrix = np.zeros((len(HZ.paulis), len(Z_list)))
    for i in range(len(HZ.paulis)):
        for j in range(len(Z_list)):
            Z_matrix[i,j] = int(HZ.paulis[i][j] == "Z")
    
    
    states = [(get_energy(coeffs,Z_matrix,gs),gs)] 
    
    for i in range(1,k+1):
        for combo in combinations(range(len(Z_list)), i):  
            temp_state = deepcopy(gs)
            temp_state[[combo]] +=1
            temp_state %=2 

            states.append((get_energy(coeffs,Z_matrix,temp_state),temp_state))

    states.sort(key = lambda x: x[0])
    return states[:n_states]

def get_local_excitation_states_parallel(HZ, k, n_states, n_jobs=-1):
    assert k < len(HZ.paulis[0]) // 2
    result, groups, phases, energies = get_excited_states(HZ.convert_to_spare_hamiltonian(), 1)
    groups = bit_extend_group(groups[0], HZ.paulis, len(HZ.paulis[0]))
    assert not groups[-1]
    groups = groups[0]
    coeffs = HZ.coeffs
    Z_list = groups
    phases = np.concatenate([phases[0], np.ones(len(Z_list) - len(phases[0]))])
    gs = get_bitstring(Z_list, phases)

    Z_matrix = np.zeros((len(HZ.paulis), len(Z_list)))
    for i in range(len(HZ.paulis)):
        for j in range(len(Z_list)):
            Z_matrix[i, j] = int(HZ.paulis[i][j] == "Z")

    # Base states (ground state)
    states = [(get_energy(coeffs, Z_matrix, gs), gs)]

    def evaluate_state(combo):
        temp_state = deepcopy(gs)
        temp_state[list(combo)] += 1
        temp_state %= 2
        return [
            (get_energy(coeffs, Z_matrix, temp_state), temp_state.copy())        ]

    # Run in parallel
    for i in range(1,k+1):
        results = Parallel(n_jobs=n_jobs)(delayed(evaluate_state)(combo) for combo in combinations(range(len(Z_list)), i))

        # Flatten results and append to states
        for pair in results:
            states.extend(pair)

    states.sort(key=lambda x: x[0])
    return states[:n_states]

def bit_string_to_circ(bitstring):
    circ = circuit_instruct(len(bitstring))
    for i,v in enumerate(bitstring):
        if(v):
            circ.H(i)
            circ.S(i)
            circ.S(i)
            circ.H(i)
    return circ

def LCBS(H,HZ,k,n_states):
    states = get_local_excitation_states(HZ,k,n_states)
    circs = [bit_string_to_circ(s[-1]) for s in states]
    energy_matrix = H.__total_elements__(circs)
    vals,vecs = np.linalg.eigh(energy_matrix)
    return vals[0],vecs[:,0]

def p_LCBS(H,HZ,k,n_states):
    states = get_local_excitation_states_parallel(HZ,k,n_states)
    circs = [bit_string_to_circ(s[-1]) for s in states]
    energy_matrix = H.__p_total_elements__(circs)
    vals,vecs = np.linalg.eigh(energy_matrix)
    return vals[0],vecs[:,0]