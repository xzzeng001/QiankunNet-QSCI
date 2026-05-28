#!/usr/bin/env python3
"""
Script to compute quantum entropies and mutual information from CI amplitudes,
aligned with FCI determinant ordering, then plot as heatmap.
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from pyscf import fci
from pyscf.fci.cistring import make_strings
import time

def load_ci_data(npz_filename):
    """
    Load CI states and probabilities from .npz, normalize amplitudes.
    Returns:
      ci_states: shape (M, L) int array of bitstrings
      ci_coeff : shape (M,) complex128 amplitudes
    """
    data = np.load(npz_filename)
    ci_states = data['ci_states']    # (M, L)
    ci_probs  = data['ci_probs']
    ci_coeff  = ci_probs.astype(np.complex64)
    ci_coeff /= np.linalg.norm(ci_coeff)
    return ci_states, ci_coeff

def build_ci_vector(ci_states, ci_coeff, norb, na, nb):
    """
    Build CI vector matrix of shape (n_alpha_str, n_beta_str) in PySCF FCI order,
    by inverting the Generate_ci_coeffs procedure.
    ci_states: (M, 2*norb) interleaved alpha-beta bits; ci_coeff: (M,) amplitudes.
    """
    # Generate all occupancy lists for alpha and beta
    occslst = fci.cistring.gen_occslst(range(norb), na)
    occblst = fci.cistring.gen_occslst(range(norb), nb)
    n_a, n_b = len(occslst), len(occblst)
    # Initialize CI vector (FCI-format)
    ci_vector = np.zeros((n_a, n_b), dtype=np.complex64)
    # Build mapping from bitstring to coefficient index
    mapping = {tuple(state): idx for idx, state in enumerate(ci_states)}
    # Fill ci_vector by matching interleaved bit patterns
    for i, occsa in enumerate(occslst):
        for j, occsb in enumerate(occblst):
            # construct interleaved state bits
            bits = [0] * (2*norb)
            for ii in occsa:
                bits[2*ii] = 1
            for jj in occsb:
                bits[2*jj + 1] = 1
            idx = mapping.get(tuple(bits))
            if idx is not None:
                ci_vector[i, j] = ci_coeff[idx]
    # Renormalize vector
    norm = np.linalg.norm(ci_vector)
    if norm > 0:
        ci_vector /= norm
    return ci_vector

def write_entropy(filename, entropy_list):
    """
    将每个轨道的纠缠熵写入文件，每行格式：
      Orbital_Index Entropy
    """
    with open(filename, "w") as f:
        f.write("Orbital_Index Entropy\n")
        for i, s in enumerate(entropy_list):
            f.write(f"{i} {s}\n")

def write_mutual(filename, mutual_mat):
    """
    将互信息矩阵写入文件，每行对应一个轨道，列间以空格分隔。
    """
    n_rows, n_cols = mutual_mat.shape
    with open(filename, "w") as f:
        for i in range(n_rows):
            line = " ".join(f"{mutual_mat[i,j]}" for j in range(n_cols))
            f.write(line + "\n")

def plot_ent_mut(S, I_uv, title=None):
    """
    Plot heatmap of entropy (diag) and mutual info (off-diag).
    """
    norb = S.size
    M = I_uv.copy()
    np.fill_diagonal(M, S)
    Mlog = np.log10(M + 1e-20)
    plt.figure(figsize=(6, 5))
    im = plt.imshow(
        Mlog, cmap='bwr', origin='upper',
        norm=colors.Normalize(vmin=Mlog.min(), vmax=Mlog.max())
    )
    for u in range(norb):
        for v in range(norb):
            plt.text(v, u, f"{M[u, v]:.2e}", ha='center', va='center', fontsize=10)
    plt.xticks(range(norb), [str(i) for i in range(norb)])
    plt.yticks(range(norb), [str(i) for i in range(norb)])
    plt.xlabel("Orbital index $u$")
    plt.ylabel("Orbital index $v$")
    plt.title(title or "One-orbital entropies (diag) & mutual info (off-diag)")
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label(r"$\log_{10}\,$(Entropy / Mutual info)")
    plt.tight_layout()
    plt.show()

def main():

    npz_file = "fe2s2_nnqs_0.0_107437784states.npz"

    #npz_file='Lih.npz'

    start_time = time.time()
    # Load CI data from .npz
    ci_states, ci_coeff = load_ci_data(npz_file)

    end1_time=time.time()
    print("time for load_ci_data: {:.4f}s".format(end1_time - start_time),flush=True)

    # Determine norb and nelec
    L = ci_states.shape[1]
    if L % 2 != 0:
        raise ValueError("Spin orbital count must be even.")
    norb = L // 2
    total_occ = int(np.sum(ci_states[0]))
    if total_occ % 2 != 0:
        raise ValueError("Total electron number must be even for singlet.")
    na = nb = total_occ // 2
    nelec = (na, nb)

    strs_a = make_strings(range(norb), na)
    strs_b = make_strings(range(norb), nb)
    n_det  = len(strs_a) * len(strs_b)

    end2_time=time.time()
    print("time for make_strings: {:.4f}s".format(end2_time - end1_time),flush=True)
    
    # build ci_vector
    ci_vec = build_ci_vector(ci_states, ci_coeff, norb, na, nb)

    end3_time=time.time()
    print("time for build_ci_vector: {:.4f}s".format(end3_time - end2_time),flush=True)
    
    # Probabilities and occupation table
    p_k = np.abs(ci_vec.reshape(-1))**2
    p_k /= p_k.sum()
   
    occ = np.zeros((n_det, 2*norb), dtype=int)
    for ia, a in enumerate(strs_a):
        for ib, b in enumerate(strs_b):
            idx = ia*len(strs_b) + ib
            for p in range(norb):
                occ[idx, p       ] = (a >> p) & 1
                occ[idx, p+norb ] = (b >> p) & 1

    end4_time=time.time()
    print("time for occupation table: {:.4f}s".format(end4_time - end3_time),flush=True)


    # 5. One?orbital entropies
    S = np.zeros(norb)
    eps = 1e-15
    for u in range(norb):
        P = np.zeros((2,2))
        for na, nb, pr in zip(occ[:,u], occ[:,u+norb], p_k):
            P[na, nb] += pr
        S[u] = -np.sum(P * np.log(P + eps))
#        print(u,S[u])

    end5_time=time.time()
    print("time for one-orbital entropies: {:.4f}s".format(end5_time - end4_time),flush=True)

    # 6. Two?orbital entropies & mutual info
    I_uv = np.zeros((norb, norb))
    for u in range(norb):
        for v in range(u, norb):
            P2 = np.zeros((2,2,2,2))
            for row, pr in zip(occ, p_k):
                nau, nbu = row[u],       row[u+norb]
                nav, nbv = row[v],       row[v+norb]
                P2[nau,nbu,nav,nbv] += pr
            S_uv = -np.sum(P2 * np.log(P2 + eps))
            I = S[u] + S[v] - S_uv
            I_uv[u,v] = I_uv[v,u] = I
    
#            print(u,v,I)

    end6_time=time.time()
    print("time for mutual info: {:.4f}s".format(end6_time - end5_time),flush=True)

    write_entropy("out_entropy.dat", S)
    write_mutual("out_mutual.dat", I_uv)

    end7_time=time.time()
    print("time for writing file: {:.4f}s".format(end7_time - end6_time),flush=True)

    # Plot
    title='figure'
    plot_ent_mut(S, I_uv, title)

if __name__ == '__main__':
    main()

