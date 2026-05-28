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
    d1, d2, d3, d4 = fci.rdm.make_dm1234('FCI4pdm_kern_sf', ci_coeff, ci_coeff, norb, nelec)
    rd1, rd2, rd3, rd4 = fci.rdm.reorder_dm1234(d1, d2, d3, d4)

    sd1,sd2,sd3 = fci.direct_spin1.make_rdm123s(ci_coeff, norb, nelec)

    return sd1, sd2, sd3, rd4

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

    npz_file = "fe2s2_nnqs_0.0_107437784states.npz"

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

