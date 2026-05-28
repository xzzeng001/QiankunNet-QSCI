#!/usr/bin/env python

'''
Broken-symmetry DFT 
'''
from pyscf import dft
import functools
from functools import reduce
import numpy as np
import pyscf
from pyscf import gto
from pyscf import scf
from pyscf import mp,mcscf
from pyscf import fci
from pyscf.mcscf import avas
import openfermion
import scipy
#from grouping import ham_grouping
import sys
import psutil
from openfermion import MolecularData
import os
from pyscf.tools import fcidump
import utils
#from pennylane_qchem.qchem import convert_observable
#
# First converge a high-spin UKS calculation
#
mf = fcidump.to_scf('FCIDUMP',molpro_orbsym=True)
#mf.mol.verbose = 4
#mf.run()

hcore = mf.get_hcore()
#mo_coeff = mf.mo_coeff

n_orb=mf.mol.nao
energy_core= mf.mol.energy_nuc()
#print('energy_core: ',energy_core)
#np.savetxt('mo_energy.txt',[mf.mo_energy])

#sys.exit(0)

#print('nelectron: ',mf.mol.nelectron)
#print('n_orb: ',mf.mol.nao)
#print('hcore: ',hcore)
#print('mo_coeff: ',mo_coeff)
#one_body_mo = functools.reduce(np.dot, (mo_coeff.T, hcore, mo_coeff))
#two_body_mo = pyscf.ao2mo.restore(1, pyscf.ao2mo.get_mo_eri(
#    mf._eri, mo_coeff, compact=False),
#    n_orb)

one_body_mo = hcore
two_body_mo = pyscf.ao2mo.restore(1, mf._eri,n_orb)

one_body_int = np.zeros([n_orb * 2] * 2)
two_body_int = np.zeros([n_orb * 2] * 4)

## for the aabb
for p in range(n_orb):
    for q in range(n_orb):
        one_body_int[p][q] = one_body_mo[p][q]
        one_body_int[n_orb + p][n_orb + q] = one_body_mo[p][q]
        for r in range(n_orb):
            for s in range(n_orb):
                two_body_int[p][q][r][s] = two_body_mo[p][s][q][r]
                two_body_int[n_orb + p][n_orb + q][n_orb + r][n_orb + s] = two_body_mo[p][s][q][r]
                two_body_int[n_orb + p][q][r][n_orb + s] = two_body_mo[p][s][q][r]
                two_body_int[p][n_orb + q][n_orb + r][s] = two_body_mo[p][s][q][r]

## for the abab
##for p in range(n_orb):
##    for q in range(n_orb):
##        one_body_int[2 * p][2 * q] = one_body_mo[p][q]
##        one_body_int[2 * p + 1][2 * q + 1] = one_body_mo[p][q]
##        for r in range(n_orb):
##            for s in range(n_orb):
##                two_body_int[2 * p][2 * q][2 * r][2 * s] = two_body_mo[p][s][q][r]
##                two_body_int[2 * p + 1][2 * q + 1][2 * r + 1][2 * s + 1] = two_body_mo[p][s][q][r]
##                two_body_int[2 * p + 1][2 * q][2 * r][2 * s + 1] = two_body_mo[p][s][q][r]
##                two_body_int[2 * p][2 * q + 1][2 * r + 1][2 * s] = two_body_mo[p][s][q][r]

hamiltonian_fermOp_1 = openfermion.FermionOperator()
hamiltonian_fermOp_2 = openfermion.FermionOperator()

for p in range(n_orb * 2):
    for q in range(n_orb * 2):
        hamiltonian_fermOp_1 += openfermion.FermionOperator(
            ((p, 1), (q, 0)),
            one_body_int[p][q]
        )
for p in range(n_orb * 2):
    for q in range(n_orb * 2):
        for r in range(n_orb * 2):
            for s in range(n_orb * 2):
                hamiltonian_fermOp_2 += openfermion.FermionOperator(
                    ((p, 1), (q, 1), (r, 0), (s, 0)),
                    two_body_int[p][q][r][s] * 0.5
                )

hamiltonian_fermOp_1 = openfermion.normal_ordered(hamiltonian_fermOp_1)
hamiltonian_fermOp_2 = openfermion.normal_ordered(hamiltonian_fermOp_2)
hamiltonian_fermOp = hamiltonian_fermOp_1 + hamiltonian_fermOp_2
hamiltonian_fermOp += energy_core

np.save('ham_fermi_aabb.npy',hamiltonian_fermOp)

hamiltonian_qubitOp = openfermion.jordan_wigner(hamiltonian_fermOp)
n_qubits = openfermion.count_qubits(hamiltonian_qubitOp)

#H =  openfermion.get_sparse_operator(hamiltonian_qubitOp)
#H=convert_observable(hamiltonian_qubitOp)

#fci, fci_vec = scipy.sparse.linalg.eigsh(H, k=1, which="SA")
#print('e: ',fci)

print('Number of terms: ',len(hamiltonian_qubitOp.terms))

np.save('ham_qubit_aabb.npy',hamiltonian_qubitOp)

filename = "qubit_op_aabb.data"
utils.save_binary_qubit_op(hamiltonian_qubitOp, filename=filename)
print("Qubit Hamiltonian saved to %s." % (filename))

print('ham_aabb saved!')
print('number of qubits: ',n_qubits)
print('All Done !!!')

