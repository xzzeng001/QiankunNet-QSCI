import numpy as np

def read_output_file(filename):
    """
    读取 output.dat 文件，跳过表头，返回记录列表，每条记录为 (state_int, ci_prob_mod2)
    """
    records = []
    with open(filename, "r") as f:
        lines = f.readlines()
    # 假设第一行为表头，从第二行开始读取
    for line in lines[1:]:
        parts = line.strip().split()
        if len(parts) < 3:
            continue
        try:
            state_int = int(parts[1])
            prob_mod2 = float(parts[2])
            records.append((state_int, prob_mod2))
        except Exception as e:
            print("解析行时出错：", line, e)
    return records

def analyze_prob_histogram(records, nbins=50):
    """
    对 records 中的 ci_prob_mod2 值进行直方图统计，
    返回 bin_center 数组与计数数组
    """
    # 提取第三列数据
    prob_values = np.array([rec[1] for rec in records])
    counts, bin_edges = np.histogram(prob_values, bins=nbins)
    # 计算每个 bin 的中心值
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return bin_centers, counts

def write_histogram(filename, bin_centers, counts):
    """
    将直方图统计结果写入文件，每行写入：bin_center  count
    """
    with open(filename, "w") as f:
        f.write("Bin_Center Count\n")
        for center, cnt in zip(bin_centers, counts):
            f.write(f"{center} {cnt}\n")

def int_to_bin_str(n, length=40):
    """
    将整数 n 转换为固定长度的二进制字符串，不足位数前面补0
    """
    return format(n, f"0{length}b")

def analyze_excitations(records, hf_state):
    """
    对 records 中的状态进行激发分析：
      - 将状态整数转换为二进制字符串（长度固定为 40）
      - 与 HF 状态比较，统计不同位数 diff
      - 如果 diff==0 则认为激发级别为 0；如果 diff==2 则为单激发（标记为 1），diff==4 则为双激发（标记为 2），依此类推
    返回一个字典，键为激发级别，值为数量统计
    """
    excitation_counts = {}
    for state_int, _ in records:
        bin_str = int_to_bin_str(state_int, length=40)
        # 统计与 hf_state 不同的位数
        diff = sum(1 for a, b in zip(bin_str, hf_state) if a != b)
        # 激发级别：假设差异数总为偶数，diff==0 则级别 0，diff==2 则级别 1，依此类推
        excitation_level = diff // 2
        excitation_counts[excitation_level] = excitation_counts.get(excitation_level, 0) + 1
    return excitation_counts

def write_excitations(filename, excitation_counts):
    """
    将激发统计结果写入文件，每行写入：Excitation_Level  Count
    """
    with open(filename, "w") as f:
        f.write("Excitation_Level Count\n")
        # 按激发级别从小到大排序输出
        for level in sorted(excitation_counts.keys()):
            f.write(f"{level} {excitation_counts[level]}\n")

def main():
    # 1. 读取 output.dat 文件
    output_file = "output.dat"
    records = read_output_file(output_file)
    
    # 2. 对第三列（ci_prob_mod2）数据进行直方图统计
    bin_centers, counts = analyze_prob_histogram(records, nbins=50)
    out_freq_file = "out_freq.dat"
    write_histogram(out_freq_file, bin_centers, counts)
    print(f"直方图统计结果已写入 {out_freq_file}")
    
    # 3. 激发分析：
    # 将第二列中的整数转换成40位二进制字符串
    # 自动生成 HF 态：长度为40，前30位为1，后10位为0
    hf_state = "1" * 30 + "0" * 10
    excitation_counts = analyze_excitations(records, hf_state)
    out_excitation_file = "out_excitation.dat"
    write_excitations(out_excitation_file, excitation_counts)
    print(f"激发分析结果已写入 {out_excitation_file}")

if __name__ == "__main__":
    main()

