import numpy as np
import math

MIN_WEIGHT = 1e-10   # 权重过滤下限

def read_and_filter_npz(npz_filename):
    """
    读取 npz 文件中的 ci_states 和 ci_probs，
    对于每个确定式：
      - 将 ci_states 中的 -1 转换为 0，生成40位二进制字符串
      - 计算权重 w = |ci_prob|^2
    仅保留 w >= MIN_WEIGHT 的项，并按权重降序排序。
    返回列表，每个元素为 (bin_state, weight)。
    """
    data = np.load(npz_filename)
    ci_states = data["ci_states"]  # 假设形状为 (N, 40)
    ci_probs  = data["ci_probs"]   # 假设形状为 (N,)
    
    filtered = []
    for i in range(ci_states.shape[0]):
        state_row = ci_states[i]
        # 将 -1 转换为 0
        state_converted = np.where(state_row == -1, 0, state_row)
        # 生成二进制字符串（注意：这里保证每行有40个数）
        bin_state = ''.join(str(int(bit)) for bit in state_converted)
        weight = abs(ci_probs[i])**2
        if weight >= MIN_WEIGHT:
            filtered.append((bin_state, weight))
    # 按权重从大到小排序
    filtered_sorted = sorted(filtered, key=lambda x: x[1], reverse=True)
    return filtered_sorted

def excitation_analysis(filtered_states):
    """
    生成 HF 态（40位：前30位'1', 后10位'0'），
    对于每个二进制状态，计算与 HF 态的不同位数 diff，
    定义激发级别为 diff//2。
    返回一个字典，键为激发级别，值为该级别的计数。
    """
    hf_state = "1" * 30 + "0" * 10
    exc_counts = {}
    for bin_state, _ in filtered_states:
        # 逐位比较
        diff = sum(1 for a, b in zip(bin_state, hf_state) if a != b)
        excitation_level = diff // 2  # 假设 diff 总为偶数
        exc_counts[excitation_level] = exc_counts.get(excitation_level, 0) + 1
    return exc_counts

def write_excitation(filename, exc_counts):
    """
    将激发统计结果写入文件，每行格式：
      Excitation_Level Count
    按激发级别升序输出。
    """
    with open(filename, "w") as f:
        f.write("Excitation_Level Count\n")
        for level in sorted(exc_counts.keys()):
            f.write(f"{level} {exc_counts[level]}\n")

def normalize_wavefunction(filtered_states):
    """
    归一化过滤后的波函数（确定式权重）。
    返回：
      - norm_states: 列表，每个元素为 (bin_state, normalized_weight)
      - norm_total: 总归一化因子（应为1）
    """
    total_weight = sum(w for _, w in filtered_states)
    norm_states = [(state, w/total_weight) for state, w in filtered_states]
    return norm_states, total_weight

def compute_orbital_entropy(norm_states, n_orb=40):
    """
    计算每个轨道的占据概率和轨道纠缠熵。
    对于轨道 i:
      p_i = sum_{det, 如果 det[i]=='1'} (normalized_weight)
      s_i = -[ p_i ln(p_i) + (1-p_i) ln(1-p_i) ]
      其中约定 0 ln0 = 0。
    返回一个长度为 n_orb 的列表，记录各轨道的熵。
    """
    entropy_list = []
    # 累计每个轨道的占据概率
    p_occ = np.zeros(n_orb)
    for state, w in norm_states:
        for i in range(n_orb):
            if state[i] == '1':
                p_occ[i] += w
    # 计算熵
    for i in range(n_orb):
        p = p_occ[i]
        # 避免 log(0) 情况
        term1 = -p * math.log(p) if p > 0 else 0.0
        term2 = -(1-p) * math.log(1-p) if (1-p) > 0 else 0.0
        s_i = term1 + term2
        entropy_list.append(s_i)
    return entropy_list

def compute_mutual_information(norm_states, norm_states_total, orbital_entropy, n_orb=40):
    """
    计算 40 个轨道两两之间的互信息。
    对于轨道 i 和 j，先计算联合概率分布：
      p00, p01, p10, p11（分别对应 det[i], det[j] 为 (0,0), (0,1), (1,0), (1,1)）
    两轨道联合熵：
      s_ij = - Σ_{a,b in {0,1}} p_{ab} ln(p_{ab})
    互信息： I(i,j) = s_i + s_j - s_ij
    返回一个 40x40 的二维 numpy 数组。
    """
    # 先计算每对轨道的联合概率（归一化后的波函数已经归一化，总和为1）
    mutual_mat = np.zeros((n_orb, n_orb))
    
    # 对每一对轨道 i,j
    for i in range(n_orb):
        for j in range(n_orb):
            # 联合概率字典
            p00 = p01 = p10 = p11 = 0.0
            for state, w in norm_states:
                a = state[i]
                b = state[j]
                if a == '0' and b == '0':
                    p00 += w
                elif a == '0' and b == '1':
                    p01 += w
                elif a == '1' and b == '0':
                    p10 += w
                elif a == '1' and b == '1':
                    p11 += w
            # 联合熵 s_ij
            s_ij = 0.0
            for p in (p00, p01, p10, p11):
                if p > 0:
                    s_ij -= p * math.log(p)
            # 互信息：I(i,j)= s_i + s_j - s_ij，其中 s_i, s_j 已由 orbital_entropy 得到
            mutual_mat[i, j] = orbital_entropy[i] + orbital_entropy[j] - s_ij
    return mutual_mat

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

def main():
    npz_filename = "fe2s2_cdfci_0.0_107437784states.npz"
    # 1. 读取并过滤排序
    filtered = read_and_filter_npz(npz_filename)
    # 按权重降序得到确定式列表：列表中每项为 (bin_state, weight)
    
    # 2. 激发分析：仅对过滤后的确定式进行，与 HF 态比较
    exc_counts = excitation_analysis(filtered)
    write_excitation("out_excitation_2.dat", exc_counts)
    print("激发分析结果已写入 out_excitation_2.dat")
    
    # 3. 构造归一化波函数
    norm_states, total_weight = normalize_wavefunction(filtered)
    # 此时 norm_states 中每个元素为 (bin_state, normalized_weight)，归一化后权重之和=1
    
    # 4. 计算各轨道的纠缠熵
    orbital_entropy = compute_orbital_entropy(norm_states, n_orb=40)
    write_entropy("out_entropy.dat", orbital_entropy)
    print("轨道纠缠熵已写入 out_entropy.dat")
    
    # 5. 计算轨道互信息
    mutual_mat = compute_mutual_information(norm_states, total_weight, orbital_entropy, n_orb=40)
    write_mutual("out_mutual.dat", mutual_mat)
    print("轨道互信息已写入 out_mutual.dat")

if __name__ == "__main__":
    main()

