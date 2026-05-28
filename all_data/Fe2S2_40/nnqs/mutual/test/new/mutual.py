from pyscf import gto, scf, ci, fci, mcscf
#from pyscf import dmrgscf
#from pyscf.dmrgscf.dmrgci import DMRGCI
import numpy as np
import math
import os

#mol = gto.Mole()
#mol.build(
#    atom = 'Li 0 0 0; H 0 0 1.2',
#    basis = 'sto-3g'
#)
#
#mf = scf.RHF(mol)
#mf.kernel()
#
####casscf###
#
#ncas =  mf.mo_coeff.shape[1]
#nelecas = mol.nelectron
#mcas = mcscf.CASSCF(mf, ncas, nelecas)
#e1 = mcas.kernel()
#
#d1, d2, d3, d4 = fci.rdm.make_dm1234('FCI4pdm_kern_sf', mcas.ci, mcas.ci, mcas.ncas, mcas.nelecas)
#rd1, rd2, rd3, rd4 = fci.rdm.reorder_dm1234(d1, d2, d3, d4)
###sd1, sd2, sd3 = mcas.fcisolver.make_rdm123s(mcas.ci, mcas.ncas, mcas.nelecas)
#
#sd1,sd2,sd3 = fci.direct_spin1.make_rdm123s(mcas.ci, mcas.ncas, mcas.nelecas)

###cisd###

#mci = ci.CISD(mf)
#e1 = mci.kernel()

#d1, d2, d3, d4 = fci.rdm.make_dm1234('FCI4pdm_kern_sf', mci.to_fcivec(mci.ci), mci.to_fcivec(mci.ci), mci.nmo, (mci.nocc, mci.nocc))
#rd1, rd2, rd3, rd4 = fci.rdm.reorder_dm1234(d1, d2, d3, d4)
#sd1, sd2, sd3 = fci.direct_spin1.make_rdm123s(mci.to_fcivec(mci.ci), mci.nmo, (mci.nocc, mci.nocc))

###fci###

#mci = fci.FCI(mf)
#e1 = mci.kernel()

#d1, d2, d3, d4 = fci.rdm.make_dm1234('FCI4pdm_kern_sf', mci.ci, mci.ci, mci.norb, mci.nelec)
#rd1, rd2, rd3, rd4 = fci.rdm.reorder_dm1234(d1, d2, d3, d4)
#sd1, sd2, sd3 = mci.make_rdm123s(mci.ci, mci.norb, mci.nelec)

######

def von_Neumann_1(rdm1, rdm2, i):
    g1 = rdm1[0][i][i]
    g2 = rdm1[1][i][i]
    G = rdm2[1][i][i][i][i]
    eigv = [(1-g1-g2+G), (g1-G), (g2-G), G]
    entr = 0
    for j in range(len(eigv)):
        if (eigv[j] > 0):
            entr = entr - eigv[j] * math.log(eigv[j])
    return entr

def von_Neumann_2(rdm1, rdm2, rdm3, rdm4, i, j):
    rho = np.zeros((16, 16))
    rho[1][2] = rdm1[0][j][i] - rdm2[1][j][i][i][i] - rdm2[1][j][i][j][j] + rdm3[1][i][j]
    rho[2][1] = np.conj(rho[1][2])
    rho[3][4] = rdm1[1][j][i] - rdm2[1][i][i][j][i] - rdm2[1][j][j][j][i] + rdm3[0][i][j]
    rho[4][3] = np.conj(rho[3][4])
    rho[5][5] = rdm4[i][j][1][1][0][0]
    rho[6][6] = rdm4[i][j][0][0][1][1]
    rho[7][7] = rdm4[i][j][0][1][0][1]
    rho[7][8] = rdm2[1][j][i][j][j] - rdm3[1][i][j]
    rho[8][7] = np.conj(rho[7][8])
    rho[7][9] = - rdm2[1][j][j][j][i] + rdm3[0][i][j]
    rho[9][7] = np.conj(rho[7][9])
    rho[7][10] = rdm2[1][j][i][j][i]
    rho[10][7] = np.conj(rho[7][10])
    rho[8][8] = rdm4[i][j][1][0][0][1]
    rho[8][9] = - rdm2[1][i][j][j][i]
    rho[9][8] = np.conj(rho[8][9])
    rho[8][10] = rdm2[1][i][i][j][i] - rdm3[0][i][j]
    rho[10][8] = np.conj(rho[8][10])

    rho[9][9] = rdm4[i][j][0][1][1][0]
    rho[9][10] = - rdm2[1][j][i][i][i] + rdm3[1][i][j]
    rho[10][9] = np.conj(rho[9][10])
    rho[10][10] = rdm4[i][j][1][0][1][0]
    rho[11][11] = rdm4[i][j][1][1][0][1]
    rho[11][12] = - rdm3[0][i][j]
    rho[12][11] = np.conj(rho[11][12])
    rho[12][12] = rdm4[i][j][1][1][1][0]
    rho[13][13] = rdm4[i][j][0][1][1][1]
    rho[13][14] = - rdm3[1][i][j]
    rho[14][13] = np.conj(rho[13][14])
    rho[14][14] = rdm4[i][j][1][0][1][1]
    rho[15][15] = rdm4[i][j][1][1][1][1]

    rho[0][0] = rdm4[i][j][0][0][0][0]
    rho[1][1] = rdm4[i][j][0][1][0][0]
    rho[2][2] = rdm4[i][j][1][0][0][0]
    rho[3][3] = rdm4[i][j][0][0][0][1]
    rho[4][4] = rdm4[i][j][0][0][1][0]
    
    eigv2, eigvec2 = np.linalg.eig(rho)
    entr2 = 0
    for k in range(len(eigv2)):
        if (eigv2[k] > 0):
            entr2 = entr2 - eigv2[k] * math.log(eigv2[k])
    return entr2

def mutual(rdm1, rdm2, rdm3, rdm4, i, j):
    delt = 0
    if (i == j) :
        delt = 1
    s1i = von_Neumann_1(rdm1, rdm2, i)
    s1j = von_Neumann_1(rdm1, rdm2, j)
    s2ij = von_Neumann_2(rdm1, rdm2, rdm3, rdm4, i, j)
    Iij = - 0.5 * (s2ij - s1i - s1j) * (1 - delt)
    return Iij

#for num in range(len(d1)):
#    print(num, von_Neumann_1(sd1, sd2, num))
#
#for num1 in range(len(d1)):
#    for num2 in range((num1 + 1), len(d1)):
#        print(num1, num2, mutual(sd1, sd2, sd3, rd4, num1, num2))
