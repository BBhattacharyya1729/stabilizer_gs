from LCSS_new import *
import time
np.random.seed(42)

H = Pauli_Hamiltonian.TI_local_H(["XX","Z"],20)
N  = 50
instance = LCSS(H)

print("LCSS States")
print("--------------------------")
t = time.time()
states  = instance.get_states(N)
triples = [triple.get_triple(s) for s in states]
t1 = time.time()
print(f"State Generation Time: {t1-t}")

N_states = len(states)
t = time.time()
O = np.eye(N_states,dtype=complex)
for i in range(0,N_states):
    for j in range(i+1,N_states):
        O[i,j] = triple.inner(triples[i], triples[j])
        O[j,i] = O[i,j].conjugate()
t1 = time.time()
print(f"Overlap Generation Time: {t1-t}")

t = time.time()
new_states= [[t.apply_pauli(p) for p in instance.H.paulis] for t in triples]
M = np.eye(N_states,dtype=complex)
for i in range(N_states):
    M[i,i] = np.dot(instance.H.coeffs,[pauli_expect(p, states[i]) for p in instance.H.paulis])
    for j in range(i+1,N_states):
        M[i,j] = np.dot(instance.H.coeffs,[c*triple.inner(triples[i],t) for (c,t) in (new_states[j])])
        M[j,i] = M[i,j].conjugate() 
t1 = time.time()
print(f"Energy Generation Time: {t1-t}")
print("--------------------------")

print("Classical States")
print("--------------------------")
t = time.time()
states_Z  = instance.get_Z_states(N)  
t1 = time.time()
print(f"State Generation Time: {t1-t}")

t = time.time()
O_Z = np.eye(len(states_Z),dtype=complex)
for i,s1 in enumerate(states_Z):
    for j,s2 in enumerate(states_Z[i+1:]):
        if s1 == s2:
            O_Z[i,j+i+1] = 1
            O_Z[j+i+1,i] = 1
t1 = time.time()
print(f"State Generation Time: {t1-t}")

t = time.time()
M_Z=full_matrix(instance.H, states_Z)
t1 = time.time()
print(f"State Generation Time: {t1-t}")
print("--------------------------")

print("Post Proccessing")
rank_list = [np.linalg.matrix_rank(O[:i,:i], hermitian=True) for i in range(1,len(O))]
energies = [LCSS.basis_diagonalize(M[:i,:i],O[:i,:i])[0] for i in range(1,len(O))]
increases =  [0]+[i for i in range(1, len(rank_list)) if rank_list[i] > rank_list[i - 1]] 
energies = np.array(energies)[increases]

rank_list_Z = [np.linalg.matrix_rank(O_Z[:i,:i], hermitian=True) for i in range(1,len(O_Z))]
energies_Z = [LCSS.basis_diagonalize(M_Z[:i,:i],O_Z[:i,:i])[0] for i in range(1,len(O_Z))]
increases_Z =  [0]+[i for i in range(1, len(rank_list_Z)) if rank_list_Z[i] > rank_list_Z[i - 1]] 
energies_Z = np.array(energies_Z)[increases_Z]

g = run_dmrg(convert_to_mpos(H.paulis,H.coeffs),50)


import matplotlib.pyplot as plt 
plt.plot(np.abs((energies-g)/g),label='Stabilizer')
plt.plot(np.abs((energies_Z-g)/g),label='Computational')
plt.savefig("Test_Plot.pdf",dpi=200)
print("Plot Saved")
