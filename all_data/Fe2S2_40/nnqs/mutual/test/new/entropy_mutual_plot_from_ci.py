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
from mutual import von_Neumann_1, mutual

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
    ci_coeff  = ci_probs.astype(np.complex128)
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
    ci_vector = np.zeros((n_a, n_b), dtype=np.complex128)
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

def build_rdms(ci_coeff, norb, nelec):
    """
    Build 1- to 4-RDMs from CI amplitudes via PySCF.
    nelec = (n_alpha, n_beta)
    """

    sd1, sd2 = fci.direct_spin1.make_rdm12s(ci_coeff, norb, nelec)

    cist_temp = fci.cistring.make_strings(range(norb), int(nelec[0]))
    cist = np.zeros((len(cist_temp), norb))
    for i in range(len(cist_temp)):
        temp = (bin(cist_temp[i])[2:].zfill(norb))
        for j in range(norb):
            cist[i][j] = temp[norb-j-1]

    vN2_diag = np.zeros((len(cist[0]),len(cist[0]),2,2,2,2))
    for i in range(len(cist[0])):
        for j in range(len(cist[0])):
            for k in range(len(cist)):
                for l in range(len(cist)):
                    vN2_diag[i][j][int(cist[k][i])][int(cist[k][j])]\
                        [int(cist[l][i])][int(cist[l][j])] += ci_coeff[k][l] ** 2
                
    dm3_simple = np.zeros((2,len(cist[0]),len(cist[0])))
    for i in range(len(cist[0])):
        for j in range(len(cist[0])):
            for k in range(len(cist)):
                for l in range(len(cist)):
                    if cist[k][i] == 1 and cist[k][j] == 1 and cist[l][i] == 0 and cist[l][j] == 1:
                        for m in range(len(cist)):
                            if cist[m][i] == 1 and cist[m][j] == 0 and (cist[m].T @ cist[l] == sum(cist[l]) - 1):
                                if i <= j: dm3_simple[0][i][j] += ci_coeff[k][l].conj() * ci_coeff[k][m] * (-1) ** (sum(cist[l][i+1:j]))
                                if i > j: dm3_simple[0][i][j] += ci_coeff[k][l].conj() * ci_coeff[k][m] * (-1) ** (sum(cist[l][j+1:i]))
                    if cist[k][i] == 0 and cist[k][j] == 1 and cist[l][i] == 1 and cist[l][j] == 1:
                        for m in range(len(cist)):
                            if cist[m][i] == 1 and cist[m][j] == 0 and (cist[m].T @ cist[k] == sum(cist[k]) - 1):
                                if i <= j: dm3_simple[1][i][j] += ci_coeff[k][l].conj() * ci_coeff[m][l] * (-1) ** (sum(cist[k][i+1:j]))
                                if i > j: dm3_simple[1][i][j] += ci_coeff[k][l].conj() * ci_coeff[m][l] * (-1) ** (sum(cist[k][j+1:i]))
    #dm3_simple[0,i,j]=sd3[1][iijjji],dm3_simple[1,i,j]=sd3[1][jiiijj]

    return sd1, sd2, dm3_simple, vN2_diag

def compute_ent_mut(rd1, rd2, rd3, rd4):
    """
    Compute one-orbital entropies S and two-orbital mutual information I_uv.
    """
    norb = rd1[0].shape[0]
    S = np.array([von_Neumann_1(rd1, rd2, i) for i in range(norb)])
    I_uv = np.zeros((norb, norb))
    for i in range(norb):
        for j in range(i+1, norb):
            val = mutual(rd1, rd2, rd3, rd4, i, j)
            I_uv[i, j] = I_uv[j, i] = val
    return S, I_uv

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

    npz_file = "Lih.npz"

    #npz_file='Lih.npz'

    # Load CI data from .npz
    ci_states, ci_coeff = load_ci_data(npz_file)

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

    # build ci_vector
    ci_vec = build_ci_vector(ci_states, ci_coeff, norb, na, nb)

#    print('ci_vec: ',ci_vec.real)

#    import sys
#    sys.exit(0)
    # rdms
    rd1, rd2, rd3, rd4 = build_rdms(ci_vec.real, norb, nelec)

    # entropy & mutual
    S, I_uv = compute_ent_mut(rd1, rd2, rd3, rd4)

    write_entropy("out_entropy.dat", S)
    write_mutual("out_mutual.dat", I_uv)

    # Plot
    title='figure'
    plot_ent_mut(S, I_uv, title)

if __name__ == '__main__':
    main()

