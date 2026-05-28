import numpy as np

def get_h10ring_geom(r):
    phi_o = 2 * np.pi / 10
    rad_denom = 2 * np.sin(np.pi/10)
    rad = r / rad_denom

    a1x = rad * np.cos(phi_o * 0)
    a2x = rad * np.cos(phi_o * 1)
    a3x = rad * np.cos(phi_o * 2)
    a4x = rad * np.cos(phi_o * 3)
    a5x = rad * np.cos(phi_o * 4)
    a6x = rad * np.cos(phi_o * 5)
    a7x = rad * np.cos(phi_o * 6)
    a8x = rad * np.cos(phi_o * 7)
    a9x = rad * np.cos(phi_o * 8)
    a10x = rad * np.cos(phi_o * 9)


    a1y = rad * np.sin(phi_o * 0)
    a2y = rad * np.sin(phi_o * 1)
    a3y = rad * np.sin(phi_o * 2)
    a4y = rad * np.sin(phi_o * 3)
    a5y = rad * np.sin(phi_o * 4)
    a6y = rad * np.sin(phi_o * 5)
    a7y = rad * np.sin(phi_o * 6)
    a8y = rad * np.sin(phi_o * 7)
    a9y = rad * np.sin(phi_o * 8)
    a10y = rad * np.sin(phi_o * 9)

    g1 = '         '
    g2 = '     '

    geom =  'H' + g1 + str(a1x)  + g2 + str(a1y)  + g2 + str(0.00)  + '\n' + \
            'H' + g1 + str(a2x)  + g2 + str(a2y)  + g2 + str(0.00)  + '\n' + \
            'H' + g1 + str(a3x)  + g2 + str(a3y)  + g2 + str(0.00)  + '\n' + \
            'H' + g1 + str(a4x)  + g2 + str(a4y)  + g2 + str(0.00)  + '\n' + \
            'H' + g1 + str(a5x)  + g2 + str(a5y)  + g2 + str(0.00)  + '\n' + \
            'H' + g1 + str(a6x)  + g2 + str(a6y)  + g2 + str(0.00)  + '\n' + \
            'H' + g1 + str(a7x)  + g2 + str(a7y)  + g2 + str(0.00)  + '\n' + \
            'H' + g1 + str(a8x)  + g2 + str(a8y)  + g2 + str(0.00)  + '\n' + \
            'H' + g1 + str(a9x)  + g2 + str(a9y)  + g2 + str(0.00)  + '\n' + \
            'H' + g1 + str(a10x) + g2 + str(a10y) + g2 + str(0.00)

    return geom


import pyscf
import pyscf.mcscf
from pyscf import gto

import ffsim

# Build an ethene molecule
# Initialize N2 molecule
mol = gto.Mole()
mol.build(
        verbose = 5,
        atom = get_h10ring_geom(1.0),
        basis = 'sto-6g',
        spin = 0)

n_elec=sum(mol.nelec)
n_orb=mol.nao_nr()
print('n_elec: ',n_elec)
print('n_orb: ',n_orb)
print('n_qubits: ',2*n_orb)

# Get molecular data and molecular Hamiltonian (one- and two-body tensors)
scf = pyscf.scf.RHF(mol).run()
mol_data = ffsim.MolecularData.from_scf(scf)
norb = mol_data.norb
nelec = mol_data.nelec
mol_hamiltonian = mol_data.hamiltonian

# Compute FCI energy
mol_data.run_fci()

import numpy as np
from pyscf import cc

# Get CCSD t2 amplitudes for initializing the ansatz
ccsd = cc.CCSD(scf).run()
mycc=ccsd

# Extract t1 and t2 amplitudes (spatial orbitals)
t1 = mycc.t1
t2 = mycc.t2

# Get the number of occupied and virtual orbitals (spatial orbitals)
nocc = t1.shape[0]
nvir = t1.shape[1]
nmo = nocc + nvir  # Total number of spatial orbitals

# Total number of spin orbitals
nso = nmo * 2

# Set the threshold
threshold = 1e-2

# Initialize t1 and t2 amplitudes in spin orbital form
t1_so = np.zeros((nocc * 2, nvir * 2))
t2_so = np.zeros((nocc * 2, nocc * 2, nvir * 2, nvir * 2))

# Expand t1 amplitudes to spin orbital form
for i in range(nocc):
    for a in range(nvir):
        # Occupied orbital indices (spin orbitals)
        i_alpha = 2 * i
        i_beta = 2 * i + 1
        # Virtual orbital indices (spin orbitals), need to add total number of occupied orbitals * 2
        a_alpha = 2 * a + nocc * 2
        a_beta = 2 * a + 1 + nocc * 2
        # Fill the amplitudes
        t1_so[i_alpha, a_alpha - nocc * 2] = t1[i, a]       # α spin
        t1_so[i_beta, a_beta - nocc * 2] = t1[i, a]         # β spin

# Expand t2 amplitudes to spin orbital form
for i in range(nocc):
    for j in range(nocc):
        for a in range(nvir):
            for b in range(nvir):
                # Occupied orbital indices (spin orbitals)
                i_alpha = 2 * i
                i_beta = 2 * i + 1
                j_alpha = 2 * j
                j_beta = 2 * j + 1
                # Virtual orbital indices (spin orbitals), need to add total number of occupied orbitals * 2
                a_alpha = 2 * a + nocc * 2
                a_beta = 2 * a + 1 + nocc * 2
                b_alpha = 2 * b + nocc * 2
                b_beta = 2 * b + 1 + nocc * 2
                # Fill the amplitudes
                # αα spin pair
                t2_so[i_alpha, j_alpha, a_alpha - nocc * 2, b_alpha - nocc * 2] = t2[i, j, a, b]
                # ββ spin pair
                t2_so[i_beta, j_beta, a_beta - nocc * 2, b_beta - nocc * 2] = t2[i, j, a, b]

# Define spin orbital label function
def spin_label(idx):
    spin = 'alpha ' if idx % 2 == 0 else 'beta '
    orb = idx // 2
    return f"{spin}{orb}"

# Output significant t1 amplitudes (single excitations with amplitude > threshold)
print("\nSignificant t1 amplitudes (single excitations with amplitude > threshold):")
t1_indices = np.argwhere(np.abs(t1_so) > threshold)
for idx in t1_indices:
    i = idx[0]  # Occupied orbital index (spin orbital)
    a = idx[1] + nocc * 2  # Virtual orbital index (spin orbital)
    amp = t1_so[idx[0], idx[1]]
    print(f"Excitation from  {spin_label(i)} to {spin_label(a)}: {amp}")

# Filter and sort t2 amplitudes
t2_list = []
for i in range(nocc * 2):
    for j in range(nocc * 2):
        for a in range(nvir * 2):
            for b in range(nvir * 2):
                amp = t2_so[i, j, a, b]
                if abs(amp) > threshold:
                    # Restore global indices for virtual orbitals
                    a_global = a + nocc * 2
                    b_global = b + nocc * 2
                    t2_list.append(((i, j, a_global, b_global), amp))

# Sort t2 amplitudes by the absolute value of amplitudes in descending order
t2_list_sorted = sorted(t2_list, key=lambda x: abs(x[1]), reverse=True)

# Output the sorted t2 amplitudes (double excitations)
print("\nSignificant t2 amplitudes (double excitations), sorted by amplitude magnitude:")
for (i, j, a, b), amp in t2_list_sorted:
    print(f"Excitation from {spin_label(i)}, {spin_label(j)} to {spin_label(a)}, {spin_label(b)}: {amp}")

