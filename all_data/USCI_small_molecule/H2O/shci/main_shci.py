import sys
from pyscf import gto, scf,fci

#os.system('cp settings.py /data/home/xzzeng/.local/lib/python3.8/site-packages/pyscf/shciscf/settings.py')

from pyscf.shciscf import shci
import os

# Initialize N2 molecule
mol = gto.Mole()
mol.build(
        verbose = 5,
#        output = None,
        atom = [('N', (0, 0, 0)), ('N', (0, 0, 1.12079733))],
        basis = 'sto-3g',
#        symmetry = True,
#        symmetry_subgroup = 'D2h',
        spin = 0
        )

# Create HF molecule
n_elec=sum(mol.nelec)
n_orb=mol.nao_nr()
print('n_elec: ',n_elec)
print('n_orb: ',n_orb)
mf = scf.RHF( mol )
mf.conv_tol = 1e-12
mf.scf()

# for the FCI energy
myfci = fci.FCI(mf).run()
fci_energy=myfci.e_tot

print('fci: ',fci_energy)
#sys.exit(0)


# Number of orbital and electrons
norb = n_orb
nelec = n_elec

# Create SHCI molecule for just variational opt.
# Active spaces chosen to reflect valence active space.
mch = shci.SHCISCF( mf, norb, nelec )
mch.fcisolver.mpiprefix = ''
mch.fcisolver.stochastic = True
mch.fcisolver.nPTiter = 0
mch.fcisolver.sweep_iter = [ 3 ]
mch.fcisolver.DoRDM = True
mch.fcisolver.sweep_epsilon = [ 1e-3]
#mch.fcisolver.sweep_epsilon = [ 5e-3, 1e-3 ]
e_shci = mch.mc1step()[0]

