import numpy as np
from pyscf import gto, scf, fci
from pyscf import ao2mo
from pyscf.fci import cistring, direct_spin1

# 1. 构建 LiH 分子（键长 1.2 Å）并做 RHF 计算
mol = gto.Mole()
mol.atom = [
    ['Li', (0.0, 0.0, 0.0)],
    ['H' , (0.0, 0.0, 1.2)]
]
mol.basis = 'sto-3g'
mol.unit  = 'Angstrom'
mol.build()

mf = scf.RHF(mol).run()

# 2. 做 FCI 计算
nelec      = mol.nelectron
nalpha     = nbeta = nelec // 2
norb       = mf.mo_coeff.shape[1]

fci_solver = fci.FCI(mol, mf.mo_coeff)
e_fci, ci_vec = fci_solver.kernel()  # ci_vec 维度 (nα_cfgs, nβ_cfgs)

# 3. 生成 α、β 电子组态（整数编码）
alpha_ints = cistring.make_strings(range(norb), nalpha)
beta_ints = cistring.make_strings(range(norb), nbeta)

# 4. 转为二进制并反转，使得 bit[i] 对应第 i 号轨道的占据（从左到右）
fmt = f'0{norb}b'
alpha_bits = [format(det, fmt)[::-1] for det in alpha_ints]
beta_bits  = [format(det, fmt)[::-1] for det in beta_ints]

#alpha_bits = [format(det, fmt) for det in alpha_ints]
#beta_bits  = [format(det, fmt) for det in beta_ints]

# 5. 按 (α,β) 交错模式构建全组态：alpha0,beta0,alpha1,beta1,...
full_configs = []
for a_bits in alpha_bits:
    for b_bits in beta_bits:
        # 将每个轨道的 α、β 占据依次拼接
        interleaved = ''.join(''.join(pair) for pair in zip(a_bits, b_bits))
        full_configs.append(interleaved)

# 6. 将 ci 矩阵按行主序拉平，对应 above full_configs 列表
full_coeffs = ci_vec.flatten()

print('full_configs: ',full_configs[:10])
print('full_coeffs: ',full_coeffs[:10])

# 7. 按系数模值降序排序
##order = np.argsort(-np.abs(full_coeffs)).tolist()
n = len(full_coeffs)

# 生成一个 0…n-1 的随机排列
order = np.random.permutation(n)

sorted_configs = [full_configs[i] for i in order]
sorted_coeffs = full_coeffs[order]

#sorted_configs=full_configs
#sorted_coeffs=full_coeffs
print('sorted_configs: ',sorted_configs[:10])
print('sorted_coeffs: ',sorted_coeffs[:10])

config_bits = np.array([[int(bit) for bit in cfg] for cfg in sorted_configs], dtype=int)

print('config_bits: ',config_bits[:10])
# 8. 保存结果
np.savez('Lih.npz',
         ci_states=config_bits,
         ci_probs =sorted_coeffs)

print(f"FCI 基态能量 = {e_fci:.8f} Ha")
print(f"共计 {len(sorted_configs)} 条组态，已按 |coeff| 降序保存至 Lih.npz")

## -------------------------------
## 4. 测试：从 Lih.npz 加载波函数并重构 civec，计算能量
## -------------------------------
#data = np.load('Lih.npz', allow_pickle=True)
#configs = data['ci_states']   # 一维字符串列表
#coeffs  = data['ci_probs']    # 对应的系数
#
## 重新生成 α、β 整数组态列表，用于索引
#alpha_ints = cistring.make_strings(range(norb), nalpha)
#beta_ints = cistring.make_strings(range(norb), nbeta)
#
#alpha_list   = list(alpha_ints)
#beta_list    = list(beta_ints)
#alpha_map    = {det: idx for idx, det in enumerate(alpha_list)}
#beta_map     = {det: idx for idx, det in enumerate(beta_list)}
#
## 重建二维 civec 矩阵
#civec2d = np.zeros((len(alpha_list), len(beta_list)), dtype=coeffs.dtype)
#for cfg, c in zip(configs, coeffs):
#    a_int = sum(int(cfg[2*i])   << i for i in range(norb))
#    b_int = sum(int(cfg[2*i+1]) << i for i in range(norb))
#    a_idx = alpha_map[a_int]
#    b_idx = beta_map[b_int]
#    civec2d[a_idx, b_idx] = c
#
## 直接从 RHF 结果重算一、二电子积分（MO 基）
#h1e_ao = mf.get_hcore()
#h1e    = mf.mo_coeff.T.dot(h1e_ao.dot(mf.mo_coeff))
#eri    = ao2mo.full(mol, mf.mo_coeff)
#
#eri_packed = ao2mo.kernel(mol, mf.mo_coeff)
#
## 调用 direct_spin1.energy，自动在内部 restore 正确的 4 维 h2e
#e_loaded = direct_spin1.energy(h1e, eri_packed, norb, (nalpha, nbeta), civec2d)
#
#print(f"从 Lih.npz 重载的波函数期望能量 = {e_loaded:.8f} Ha")
#
