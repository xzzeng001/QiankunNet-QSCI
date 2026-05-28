from pyscf import gto, scf, cc
import numpy as np

# Define the molecule
mol = gto.Mole()
mol.atom = '''
O  0.000000   0.000000   0.000000
H  0.000000  -0.757160   0.586260
H  0.000000   0.757160   0.586260
'''
mol.basis = 'sto-3g'
mol.spin = 0  # Closed-shell system
mol.build()

# Perform RHF calculation
mf = scf.RHF(mol)
mf.kernel()

# Perform CCSD calculation
mycc = cc.CCSD(mf)
mycc.kernel()

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
threshold = 1e-6

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

