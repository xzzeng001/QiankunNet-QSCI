from pyscf import gto, scf, ci, fci, mcscf, dmrgscf
from pyscf.dmrgscf.dmrgci import DMRGCI
import numpy as np
import math
import os
import time

origin_time = time.time()#test

mol = gto.Mole()
mol.build(
    atom = 'B 0 0 0; N 0 0 1.2',
    basis = 'sto-3g'
)

mf = scf.RHF(mol)
mf.kernel()
print('mo_energy: ',mf.mo_energy)

###CAS###

#ncas = 8   #number of orbitals
#nelecas = 8   #number of electrons
#mci = mcscf.CASSCF(mf, ncas,nelecas)
#e1 = mci.kernel()
#sd1, sd2 = mci.fcisolver.make_rdm12s(mci.ci, mci.ncas, mci.nelecas)

#cist_temp = fci.cistring.make_strings(range(ncas), int(nelecas/2))
#cist = np.zeros((len(cist_temp), ncas))
#for i in range(len(cist_temp)):
    #temp = (bin(cist_temp[i])[2:].zfill(ncas))
    #for j in range(ncas):
        #cist[i][j] = temp[ncas-j-1]

###CISD###

#mci = ci.CISD(mf)
#e1 = mci.kernel()
#sd1, sd2 = fci.direct_spin1.make_rdm12s(mci.to_fcivec(mci.ci), mci.nmo, (mci.nocc, mci.nocc))
#mci.ci = mci.to_fcivec(mci.ci)

###FCI###

mci = fci.FCI(mf)
e1 = mci.kernel()
sd1, sd2 = mci.make_rdm12s(mci.ci, mci.norb, mci.nelec)

cist_temp = fci.cistring.make_strings(range(mf.mo_coeff.shape[1]), int(mol.nelectron/2))
cist = np.zeros((len(cist_temp), mf.mo_coeff.shape[1]))
for i in range(len(cist_temp)):
    temp = (bin(cist_temp[i])[2:].zfill(mf.mo_coeff.shape[1]))
    for j in range(mf.mo_coeff.shape[1]):
        cist[i][j] = temp[mf.mo_coeff.shape[1]-j-1]

current_time = time.time()#test
print("FCI time=", current_time-origin_time, flush=True)#test

origin_time = time.time()#test

######

vN2_diag = np.zeros((len(cist[0]),len(cist[0]),2,2,2,2))
for i in range(len(cist[0])):
    for j in range(len(cist[0])):
        for k in range(len(cist)):
            for l in range(len(cist)):
                vN2_diag[i][j][int(cist[k][i])][int(cist[k][j])]\
                    [int(cist[l][i])][int(cist[l][j])] += mci.ci[k][l] ** 2

current_time = time.time()#test
print("diag time=", current_time-origin_time, flush=True)#test

origin_time = time.time()#test

off_diag = np.zeros((len(cist[0]),len(cist[0]),2,2))#only for RHF!

for i in range(len(cist[0])):
    for j in range(len(cist[0])):
        for k in range(len(cist)):
            for l in range(len(cist)):
                if cist[k][i] == 0 and cist[k][j] == 1 and cist[l][i] == 1 and cist[l][j] == 0 and (cist[k].T @ cist[l] == sum(cist[l]) - 1):
                    for m in range(len(cist)):
                        if i <= j: off_diag[i][j][int(cist[m][i])][int(cist[m][j])] += mci.ci[k][m].conj() * mci.ci[l][m] * (-1) ** (sum(cist[k][i+1:j]))
                        else: off_diag[i][j][int(cist[m][i])][int(cist[m][j])] += mci.ci[k][m].conj() * mci.ci[l][m] * (-1) ** (sum(cist[k][j+1:i]))

current_time = time.time()#test
print("off_diag time=", current_time-origin_time, flush=True)#test

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
    rho[1][2] = rdm3[i][j][0][0]
    rho[2][1] = np.conj(rho[1][2])
    rho[3][4] = rdm3[i][j][0][0]
    rho[4][3] = np.conj(rho[3][4])
    rho[5][5] = rdm4[i][j][1][1][0][0]
    rho[6][6] = rdm4[i][j][0][0][1][1]
    rho[7][7] = rdm4[i][j][0][1][0][1]
    rho[7][8] = rdm3[i][j][0][1]
    rho[8][7] = np.conj(rho[7][8])
    rho[7][9] = rdm3[i][j][0][1]
    rho[9][7] = np.conj(rho[7][9])
    rho[7][10] = rdm2[1][j][i][j][i]
    rho[10][7] = np.conj(rho[7][10])
    rho[8][8] = rdm4[i][j][1][0][0][1]
    rho[8][9] = rdm2[1][i][j][j][i]
    rho[9][8] = np.conj(rho[8][9])
    rho[8][10] = rdm3[i][j][1][0]
    rho[10][8] = np.conj(rho[8][10])

    rho[9][9] = rdm4[i][j][0][1][1][0]
    rho[9][10] = rdm3[i][j][1][0]
    rho[10][9] = np.conj(rho[9][10])
    rho[10][10] = rdm4[i][j][1][0][1][0]
    rho[11][11] = rdm4[i][j][1][1][0][1]
    rho[11][12] = rdm3[i][j][1][1]
    rho[12][11] = np.conj(rho[11][12])
    rho[12][12] = rdm4[i][j][1][1][1][0]
    rho[13][13] = rdm4[i][j][0][1][1][1]
    rho[13][14] = rdm3[i][j][1][1]
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

origin_time = time.time()#test

one_orb_vN = np.zeros(len(sd1[0]))
two_orb_mu = np.zeros([len(sd1[0]), len(sd1[0])])
for num1 in range(len(sd1[0])):
    one_orb_vN[num1] = von_Neumann_1(sd1, sd2, num1)
    for num2 in range(len(sd1[0])):
        two_orb_mu[num1][num2] = mutual(sd1, sd2, off_diag, vN2_diag, num1, num2)
print(one_orb_vN)
print(two_orb_mu)

current_time = time.time()#test
print("mutual time=", current_time-origin_time, flush=True)#test
