from pyscf import gto, scf, mcscf, tools
from pyscf.mcscf import addons

def write_fcidump(name,
                  atom,
                  basis,
                  charge=0,
                  spin=0,
                  frozen: list = None,   # 要冻结的 MO 全空间索引列表，例如 [0,1] 冻结 1s,2s
                  act_orb: list = None,  # 要纳入 CAS 的全空间 MO 索引列表
                  cas_elec: int = None): # 活性电子总数
    if act_orb is None or cas_elec is None:
        raise ValueError("请同时指定 act_orb（MO 索引列表）和 cas_elec（活性电子数）。")
    # 1) 构建分子并完成 RHF
    mol = gto.M(atom=atom, basis=basis,
                charge=charge, spin=spin, unit='Angstrom')
    mf = scf.RHF(mol).run()

    # 2) 冻结核心：从 mo_coeff_full 中去掉 frozen 指定的列
    mo_full = mf.mo_coeff         # shape = (n_ao, n_mo_full)
    nmo_full = mo_full.shape[1]
    if frozen:
        valence_idx = [i for i in range(nmo_full) if i not in frozen]
    else:
        valence_idx = list(range(nmo_full))
    mo_valence = mo_full[:, valence_idx]  # shape = (n_ao, n_valence)

    # 3) 在“价轨道”空间中重新映射用户给定的 act_orb
    #    这里 act_orb 中的索引是针对原始全空间的，所以要找出它在 valence_idx 中的位置
    new_act = [ valence_idx.index(i) for i in act_orb ]
    ncas = len(new_act)

    # 4) 用截断后的 MO 进行 CASCI 排序（仅为了使用 sort_mo）
    #    我们先把 mf.mo_coeff 替换为 mo_valence，让 CASCI 在“价轨道”基上工作
    mf.mo_coeff = mo_valence
    mc = mcscf.CASCI(mf, ncas, cas_elec)

    e_tot, e_cas, fcivec, mo, mo_energy=mc.kernel()
    
    one_body_mo, energy_core = mc.get_h1eff(mo)  # 获得活性空间的单电子积分
    two_electron_compressed = mc.get_h2eff(mo)  # 获得活性空间的双电子积分（压缩形式）
    
    # 保存为FCIDUMP文件
    tools.fcidump.from_integrals(name, one_body_mo, two_electron_compressed, mc.ncas, mc.nelecas, energy_core)


cu2o2_geom = '''
Cu  0.000  0.000  1.900
Cu  0.000  0.000 -1.900
O   0.000  1.527  0.000
O   0.000 -1.527  0.000
'''
write_fcidump(
    name='Cu2O2_dimer_CAS16e12o.FCIDUMP',
    atom=cu2o2_geom,
    basis='def2-TZVP',
    charge=2,
    spin=0,                # 根据构型调整
    frozen=[0,1,2,3,4,5],  # 冻结 Cu [Ar] 核
    act_orb=[              # Cu 3d(10) + O₂ σ/π(2)
        20,21,22,23,24,25,26,27,28,29,30,31
    ],
    cas_elec=16
)
