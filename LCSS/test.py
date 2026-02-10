from LCSS_new import *
np.random.seed(42)

H = Pauli_Hamiltonian.TI_local_H(["XX","Z"],10)
N  = 100
states, energies = H.get_excited_states(N)

instance = LCSS(H)
M,O = instance.run_LCSS(N)
MZ,OZ = instance.run_Z_LCSS(N)

exact_ground_state = run_dmrg(convert_to_mpos(H.paulis,H.coeffs),50)

np.save('test_result.npy',[{
    'M':M, 
    'O':O, 
    'MZ':MZ, 
    'OZ':OZ, 
    'dmrg':exact_ground_state
}],allow_pickle= True)