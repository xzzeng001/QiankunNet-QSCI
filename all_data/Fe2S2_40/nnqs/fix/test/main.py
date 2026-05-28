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

# === 1. 新增：快速构造 CI 向量的函数 ===
def build_ci_vector_fast(ci_states, ci_coeff, norb, na, nb):
    """
    用“正向映射”方式把稀疏的 spin-orbital 组态(ci_states)铺入 FCI 格式的 ci_vector，
    并返回对应的 alpha_ints, beta_ints。此处仅修改了 make_strings 调用。
    """
    # 1.1 取出所有 α、β 轨道组合的整数编码（bit-int）
    alpha_strings = make_strings(range(norb), na)  # (而非 make_strings(range(norb), na))
    beta_strings = make_strings(range(norb), nb)

    n_a = len(alpha_strings)
    n_b = len(beta_strings)

    # 1.2 建映射字典：bit-int → 在 FCI 格式矩阵中的行/列索引
    dict_alpha = { alpha_strings[i]: i for i in range(n_a) }
    dict_beta  = { beta_strings[j]: j  for j in range(n_b)  }

    # 1.3 初始化 CI 向量矩阵
    ci_vector = np.zeros((n_a, n_b), dtype=np.complex64)

    # 1.4 预生成位权重，用于把 0/1 数组转成 bit-int
    bit_weights = (1 << np.arange(norb, dtype=np.int64))  # [1,2,4,...,2^(norb-1)]

    M = ci_states.shape[0]
    alpha_ints = np.empty(M, dtype=np.int64)
    beta_ints  = np.empty(M, dtype=np.int64)

    for k in range(M):
        row = ci_states[k]  # 长度 2*norb，交错排列 [α0,β0,α1,β1,...]
        bits_a = row[0::2].astype(np.int64)
        bits_b = row[1::2].astype(np.int64)

        a_int = int(bits_a.dot(bit_weights))
        b_int = int(bits_b.dot(bit_weights))

        alpha_ints[k] = a_int
        beta_ints[k]  = b_int

        ia = dict_alpha.get(a_int)
        ib = dict_beta .get(b_int)
        if ia is not None and ib is not None:
            ci_vector[ia, ib] = ci_coeff[k]
        # （若找不到，表示 ci_states 中的组态不在这个 norb/na/nb 范围内，可忽略）

    # 归一化
    norm = np.linalg.norm(ci_vector)
    if norm > 0:
        ci_vector /= norm

    return ci_vector, alpha_ints, beta_ints


# === 2. 保留原始 load_ci_data、plot_ent_mut、write_entropy、write_mutual 等函数，略 ===

def load_ci_data(npz_filename):
    data = np.load(npz_filename)
    ci_states = data['ci_states']    # (M, 2*norb)
    ci_probs  = data['ci_probs']

    ci_states_new = []
    for i in range(ci_states.shape[0]):
        row = ci_states[i]
        # 将 -1 → 0，如果已经只有 0/1，则不受影响
        row0 = np.where(row == -1, 0, row).astype(np.int8)
        # 转成 length=40 的字符串
        bin_state = ''.join(str(int(bit)) for bit in row0)
        ci_states_new.append(bin_state)

    ci_coeff  = ci_probs.astype(np.complex64)
    ci_coeff /= np.linalg.norm(ci_coeff)
    return np.array(ci_states_new), ci_coeff

def plot_ent_mut(S, I_uv, title=None):
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
            plt.text(v, u, f"{M[u, v]:.2e}", ha='center', va='center', fontsize=6)
    plt.xticks(range(norb), [str(i) for i in range(norb)])
    plt.yticks(range(norb), [str(i) for i in range(norb)])
    plt.xlabel("Orbital index $u$")
    plt.ylabel("Orbital index $v$")
    plt.title(title or "One-orbital entropies (diag) & mutual info (off-diag)")
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label(r"$\log_{10}\,$(Entropy / Mutual info)")
    plt.tight_layout()
    plt.show()

def write_entropy(filename, entropy_list):
    with open(filename, "w") as f:
        f.write("Orbital_Index Entropy\n")
        for i, s in enumerate(entropy_list):
            f.write(f"{i} {s}\n")

def write_mutual(filename, mutual_mat):
    n_rows, n_cols = mutual_mat.shape
    with open(filename, "w") as f:
        for i in range(n_rows):
            line = " ".join(f"{mutual_mat[i,j]}" for j in range(n_cols))
            f.write(line + "\n")


# === 3. main() 入口：仅替换 build_ci_vector → build_ci_vector_fast，其余保持原始循环版本 ===

def main():
    npz_file = "fe2s2_nnqs_0.0_107437784states.npz"
    #npz_file='Lih.npz'
    start_time = time.time()

    # 3.1 加载 CI 数据
    ci_states, ci_coeff = load_ci_data(npz_file)
    t1 = time.time()
    print("time for load_ci_data: {:.4f}s".format(t1 - start_time), flush=True)

    L =20
    norb=20
    na=nb=15

##    # 3.2 推断 norb、na、nb
##    L = ci_states.shape[1]
##    if L % 2 != 0:
##        raise ValueError("Spin orbital count must be even.")
##    norb = L // 2
##    total_occ = int(np.sum(ci_states[0]))
##    if total_occ % 2 != 0:
##        raise ValueError("Total electron number must be even for singlet.")
##    na = nb = total_occ // 2

#    # 3.3 原始脚本中对 make_strings 的调用（只是为了后面构造 occ 表用，计算 n_det）
#    strs_a = make_strings(range(norb), na)
#    strs_b = make_strings(range(norb), nb)
#    n_det  = len(strs_a) * len(strs_b)
    t2 = time.time()
    print("time for make_strings: {:.4f}s".format(t2 - t1), flush=True)

    # 3.4 —— 关键替换：用加速版构造 CI 向量 —— 
    #     build_ci_vector_fast 返回：(ci_vec, alpha_ints, beta_ints)
#    ci_vec, alpha_ints, beta_ints = build_ci_vector_fast(ci_states, ci_coeff, norb, na, nb)
    t3 = time.time()
    print("time for build_ci_vector_fast: {:.4f}s".format(t3 - t2), flush=True)

    # 3.3 直接由 ci_coeff 计算每个配置的概率分布 p_k
    #    load_ci_data 已经做了归一化，故 ∑|ci_coeff|^2 = 1

    p_k = np.abs(ci_coeff)**2   # 长度为 M
    M = ci_states.shape[0]
    #assert np.isclose(p_k.sum(), 1.0, atol=1e-6)

    cutoff = 1e-8   # 根据你的体系大小和精度要求调整
    mask = p_k > cutoff
    num_before = p_k.size
    # 筛选
    p_k = p_k[mask]
    ci_states = ci_states[mask]
    # 重新归一化
    p_k /= p_k.sum()
    num_after = p_k.size
    print(f"Applied cutoff {cutoff:.1e}: kept {num_after} / {num_before} configurations")

    # 3.4 —— 基于筛选后的 ci_states、p_k 计算单轨道熵 ——
    eps = 1e-15
    S = np.zeros(norb)
    for u in range(norb):
        P = np.zeros((2, 2))
        for k in range(num_after):
            na_bit = int(ci_states[k,     2*u])      # 若是“先全α后全β”格式
            nb_bit = int(ci_states[k, 2*u+1])   # β 部分偏移 norb
            pr = p_k[k]
            P[na_bit, nb_bit] += pr
        S[u] = -np.sum(P * np.log(P + eps))

    # 3.5 —— 基于筛选后的 ci_states、p_k 计算两轨道互信息 ——
    I_uv = np.zeros((norb, norb))
    for u in range(norb):
        for v in range(u, norb):
            P2 = np.zeros((2, 2, 2, 2))
            for k in range(num_after):
                nau = int(ci_states[k,  2*u])
                nbu = int(ci_states[k, 2*u+1])
                nav = int(ci_states[k,  2*v])
                nbv = int(ci_states[k, 2*v+1])
                pr  = p_k[k]
                P2[nau, nbu, nav, nbv] += pr
            S_uv = -np.sum(P2 * np.log(P2 + eps))
            I = S[u] + S[v] - S_uv
            I_uv[u, v] = I_uv[v, u] = 0.5 * I


    t3 = time.time()
    print("time for mutual info: {:.4f}s".format(t3 - t2), flush=True)

    # 3.6 写文件并绘图
    write_entropy("out_entropy.dat", S)
    write_mutual("out_mutual.dat", I_uv)
    t4 = time.time()
    print("time for writing file: {:.4f}s".format(t4 - t3), flush=True)

    plot_ent_mut(S, I_uv, title="Figure: One-orbital entropies & mutual info")

if __name__ == '__main__':
    main()

