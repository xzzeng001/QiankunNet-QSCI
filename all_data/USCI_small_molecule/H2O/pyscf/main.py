import sys
from pyscf import gto, scf,fci
from pyscf import mp, ci, cc
import os
import scipy
import numpy as np

# Initialize H2O molecule
mol = gto.Mole()
mol.build(
        verbose = 5,
        atom = [('O', (0, 0, 0)), ('H', (0.2774, 0.8929, 0.2544)), ('H', (0.6068, -0.2383, -0.7169))],
        basis = 'sto-3g',
        spin = 0)

# Create HF molecule
n_elec=sum(mol.nelec)
n_orb=mol.nao_nr()
print('n_elec: ',n_elec)
print('n_orb: ',n_orb)
mf = scf.RHF(mol)
myhf=mf.kernel()

# for the MP2 energy
mymp = mp.MP2(mf).run()
mp_energy=mymp.e_tot
print('mp energy: ',mp_energy)

# for the CCSD energy
mycc=cc.CCSD(mf).run()
cc_energy=mycc.e_tot
print('cc energy: ',cc_energy)

# for the CISD energy
myci = ci.CISD(mf).run()
ci_energy=myci.e_tot
print('ci energy: ',ci_energy)

# for the FCI energy
myfci = fci.FCI(mf).run()
fci_energy=myfci.e_tot

print('fci: ',fci_energy)

file_name='energy_pes.txt'
np.savetxt(file_name,['# HF   MP2   CISD   CCSD   FCI'],fmt='%s')
with open(file_name,'a') as f:
    np.savetxt(f,[[myhf, mp_energy, ci_energy, cc_energy, fci_energy]])


###########################################
# Generate the configuration coefficient
###########################################
def Generate_ci_coeffs(n_orb,n_elec,fcivec,filename):
    ###########################################
    # Generate the configuration coefficient
    ###########################################

    nn=scipy.special.comb(n_orb, n_elec//2)
    ndet=int(nn*nn)
    
    occslst = fci.cistring.gen_occslst(range(n_orb), n_elec//2)
    ik=0
    for i,occsa in enumerate(occslst):
        for j,occsb in enumerate(occslst):
            if abs(fcivec[i,j]) > 1e-10:
                ik += 1
    nn=ik
    
    n_qubits=2*n_orb
    ci_probs=np.zeros([ndet],dtype=complex)
    ci_states=np.zeros([ndet,n_qubits],dtype=int)
    
    ik=0
    for i,occsa in enumerate(occslst):
        for j,occsb in enumerate(occslst):
    
            if abs(fcivec[i,j]) > 1e-10:
                for kk in range(n_qubits):
                    ci_states[ik][kk]=0

# for the abab
#                for ii in occsa:
#                    ci_states[ik][2*ii]=1
#                    for jj in occsb:
#                        ci_states[ik][2*jj+1]=1
# for the aabb
                for ii in occsa:
                    ci_states[ik][ii]=1
                    for jj in occsb:
                        ci_states[ik][jj+n_orb]=1

                ci_probs[ik]=fcivec[i,j]
    
                ik +=1
    
    # Sort ci_probs and ci_states based on the absolute value of ci_probs
    sorted_indices = np.argsort(-np.abs(ci_probs[:ik]))
    ci_probs_sorted = ci_probs[sorted_indices]
    ci_states_sorted = ci_states[sorted_indices]
    
    # Convert ci_states to strings
    ci_states_strings = [''.join(map(str, row)) for row in ci_states_sorted]
    
    # Save sorted ci_probs and ci_states as strings to a file
    with open(filename, 'w') as f:
        for idx in range(ik):
            f.write(f'String: {ci_states_strings[idx]}, Coeffs: {ci_probs_sorted[idx]}\n')


myci = ci.CISD(mf)
ecisd, civec = myci.kernel()
fcivec = myci.to_fcivec(civec)
filename='cisd_results_aabb.txt'
Generate_ci_coeffs(n_orb,n_elec,fcivec,filename)

myfci = fci.FCI(mf)
fci_energy, fcivec = myfci.kernel()
filename='fci_results_aabb.txt'
Generate_ci_coeffs(n_orb,n_elec,fcivec,filename)

