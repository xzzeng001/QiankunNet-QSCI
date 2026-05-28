#!/usr/bin/env python
# encoding: utf-8
import numpy as np
from pyscf import lib

from datetime import datetime
import argparse
from concurrent.futures import ProcessPoolExecutor
import sys
from dataclasses import dataclass, field
from scipy.linalg import eigh
import scipy
from scipy.sparse.linalg import eigs, eigsh, lobpcg
from typing import List
import os

import time
import concurrent.futures

import torch
import sys

sys.path.append("../../../")

import cpp_lib.build.local_energy_module as lem

def transfer_id_state_optimized_vectorized(state, qubit_length):
    state_np = np.array(state, dtype=np.uint64)  # 确保有足够的位来处理位运算
    # 生成一个每行为0到qubit_length-1的矩阵，用于计算每位的状态
    bit_indices = np.arange(qubit_length, dtype=np.uint64)
    # 扩展state_np数组和bit_indices，使它们具有可广播的形状
    state_expanded = state_np[:, None]
    # 计算二进制表示中每位的状态，然后使用where转换为1或-1
    bits = np.where((state_expanded >> bit_indices) & 1, 1, -1)
    return bits

def transfer_state_id_numpy(states):
    states_np = np.array(states)
    qubit_length = states_np.shape[1]
    # 将 -1 转换为 0
    states_converted = np.where(states_np == -1, 0, states_np)
    two_powers = 2**np.arange(qubit_length)
    ids = np.sum(states_converted * two_powers.reshape(1, -1), axis=-1)
    return ids

def transfer_state_id_optimized(states):
    states_np = np.array(states)
    qubit_length = states_np.shape[1]
    states_converted = np.where(states_np == -1, 0, states_np)
    # 使用 np.einsum 进行高效的矩阵乘法和求和操作
    ids = np.einsum('ij,j->i', states_converted, 2**np.arange(qubit_length))
    return ids


def construct_complex(amps, phases):
    return torch.complex(amps * phases.cos(), amps * phases.sin())

def WF_lossFN(psi1: torch.complex, psi2: torch.complex):
    S = (psi1.conj() * psi2).sum().abs().pow(2)
    loss = -S
    return loss

def Coeff_lossFn(psis, ci_amps, weights):
    loss_real_err = (psis.real - ci_amps).pow(2)
    loss_imag_err = (psis.imag - 0.0).pow(2)
    loss = torch.dot(loss_real_err + loss_imag_err, weights)
    #loss_list += ((loss_real_err + loss_imag_err)*weights).tolist()
    return loss


def ciGen(fcidump_file, state_path):
#    fcidump_file =  "FCIDUMP"
#    state_path = "exp1_vqe_confs.npz"
#    print(f"Fcidump File: {fcidump_file}")

    fci = lem.Fcidump(fcidump_file)
    ham = lem.Hamiltonian(fci)
    threshold = 1e-8
#    print(f"Load {state_path}",flush=True)
    core_states = np.load(state_path)
    core_states = core_states["ci_states"]
        
    use_aabb_pretrain = False
    if use_aabb_pretrain:
        L = core_states.shape[1]
        perm_idx=np.array([t for t in zip(np.arange(L//2), np.arange(L//2)+L//2)]).reshape(-1)
        core_states = core_states[:, perm_idx] # aabb -> abab
        print(f"using aabb pretrain")

    core_states = transfer_state_id_optimized(core_states)



    # Sort the state  small -> big : core states dec
    sorted_indices = np.argsort(core_states)
    core_states = core_states[sorted_indices]
    log_diag_length = core_states.shape[0]
    core_states = core_states.tolist()
    #print(f"core_states length = {len(core_states)}", flush=True)

    # 2. Diagonalization: A
    # 2.1 get ham matrix
    cpp_st = time.time()
    result = lem.getSubHamMatrix(ham, core_states, 0.0)
#    print(f"Get Sub Ham Matrix Elapse: {time.time() - cpp_st} seconds", flush=True)

    np2list_st = time.time()
    ori_states_np = np.array(result.ori_states, dtype=int) # 十进制
    coupled_states_length_np = np.array(result.coupled_states_length, dtype=int)
    coupled_states_np = np.array(result.coupled_states, dtype=int)# 十进制
    coeffs_np = np.array(result.coeffs, dtype=np.float64)
#    print(f"Numpy to List Elapse: {time.time() - np2list_st} seconds", flush=True)

    diag_id = np.cumsum(coupled_states_length_np)[:-1]
    diag_id = np.insert(diag_id, 0, 0)
    diag = coeffs_np[diag_id]

    buildMatrix_st = time.time()
    matrixRow_index_np = np.arange(ori_states_np.shape[0], dtype=int)
    matrixRow_index_np_repeat = np.repeat(matrixRow_index_np, coupled_states_length_np)
    matrixCol_index_np = np.searchsorted(ori_states_np, coupled_states_np)
    length = ori_states_np.shape[0]
    subHamMatrix = scipy.sparse.coo_matrix((coeffs_np, (matrixRow_index_np_repeat, matrixCol_index_np)), shape=(length,length)).astype(np.float64)
#    print(f"Build Ham Matrix Elapse: {time.time() - buildMatrix_st} seconds", flush=True)
    # 2.2 diagonalize subHamMatrix
#    print(f"Sub Ham Matrix shape = {length}*{length}, Dense Matrix Memory: {length*length*8/1024/1024/1024} GB, Sparse Matrix Memory: {3*matrixCol_index_np.shape[0]*8/1024/1024/1024} GB", flush=True)
#    print(f"Dense Matrix Elem: {length*length}, Sparse Matrix Elem: {matrixCol_index_np.shape[0]}, perc: {matrixCol_index_np.shape[0]/(length*length)*100}%", flush=True)
    # scipy.sparse.linalg.eigs
    eig_st = time.time()
    #X = np.random.rand(length)
    X = np.ones(length)
    X /= np.linalg.norm(X)
    def matvec(x):
        return subHamMatrix @ x
    def precond_with_diag(residual, e_guess, x0):
        denom = diag - e_guess
        denom[np.abs(denom) < 1e-8] = 1e-8  # 避免分母为零
        return residual / denom
    eigenvalues, eigenvalues_vec  = lib.davidson(matvec, X, precond_with_diag, nroots=1, tol=0.0, max_cycle=500)
    #eigenvalues, eigenvalues_vec  = eigs(subHamMatrix, k=1, which='SR', tol=0, return_eigenvectors=True)
    psis = eigenvalues_vec.real
    #psis = eigenvalues_vec[:,0].real
    local_E = eigenvalues.real
    #local_E = eigenvalues[0].real

    return local_E

#        print(f"{i}-th eigen elapse: {time.time() - eig_st}", flush=True)
#        PT_psis = psis.tolist()
#
#        now = datetime.now()
#        formatted_time = now.strftime('%Y-%m-%d %H:%M:%S')
#        microseconds = now.microsecond // 100
#        print(f"{i}-th, VMCE = {local_E}, time: {formatted_time}.{microseconds:04d}", flush=True)
#        weights_sorted = sorted(psis*psis, reverse=True) 
#        weights_str = '  '.join('{:.8f}'.format(w) for w in weights_sorted[:8])
#        print(f"{i}-th, Weights[:8] = [{weights_str}]", flush=True)
#
#
#        pt_st = time.time()
#        PT = lem.computePT_real(ham, local_E, PT_psis, core_states, 1e-8) + local_E
#        print(f"{i}-th PT cost: {time.time() - pt_st}", flush=True)
#        print(f"{i}-th, PTE = {PT}", flush=True)

#if __name__ == "__main__":
#    print(f"process id: {os.getpid()}",flush=True)
#    ciGen(None)
