import sys
sys.path.append("../../../")
import numpy as np
from diag import ciGen

# 1. 读取 alpha / beta 的配置和频率
alpha_data = np.load('exp1_alpha_confs.npz')
alpha_states = alpha_data['ci_states']    # shape = [N_alpha, n_qubits]
alpha_freqs  = alpha_data['ci_probs']     # shape = [N_alpha,]

beta_data  = np.load('exp1_beta_confs.npz')
beta_states = beta_data['ci_states']       # shape = [N_beta, n_qubits]
beta_freqs  = beta_data['ci_probs']        # shape = [N_beta,]

n_qubits = alpha_states.shape[1]  # 每边的 qubit 数

# 2. 定义要抽样的大小列表
sample_sizes = [100, 200, 400, 600, 800, 1000]

fcidump_file='FCIDUMP'
for n in sample_sizes:
    # 2.1 随机抽取 n 条 alpha / beta
    idx_a = np.random.choice(alpha_states.shape[0], size=n, replace=False)
    idx_b = np.random.choice(beta_states.shape[0],  size=n, replace=False)
    a_sel = alpha_states[idx_a]   # [n, n_qubits]
    b_sel = beta_states[idx_b]    # [n, n_qubits]

    # 2.2 生成所有 n*n 对 (i,j)，并按 “abab” 交叉模式组合
    # 重复 alpha，每一行对应一个 beta
    a_rep = np.repeat(a_sel, repeats=n, axis=0)    # [n*n, n_qubits]
    # 平铺 beta，使其对齐 alpha_rep
    b_tile = np.tile(b_sel, reps=(n, 1))           # [n*n, n_qubits]

    # 交叉：位置 0,2,4,… 放 alpha；位置 1,3,5,… 放 beta
    full_states = np.empty((n*n, 2*n_qubits), dtype=int)
    full_states[:, 0::2] = a_rep
    full_states[:, 1::2] = b_tile

    # 2.3 所有组合的概率都设为 1.0
    ci_probs = np.ones(n*n, dtype=float)

    # 2.4 保存到 exp_full_{n}_{n}.npz
    outfname = f'exp_full_{n}_{n}.npz'
    np.savez(outfname,
             ci_states=full_states,
             ci_probs=ci_probs)

    print(f"Saved {full_states.shape[0]} states (+probs) to {outfname}")
#    state_path=outfname
#    e=ciGen(fcidump_file,state_path)

#    print(f"Total state {full_states.shape[0]}")
#    print("Energy: {}".format(e),flush=True)
