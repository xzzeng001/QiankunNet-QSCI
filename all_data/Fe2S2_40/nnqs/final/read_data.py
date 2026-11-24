import numpy as np

MIN_THRESHOLD = 1e-10  # 模方过滤下限

def state_to_int(state_row):
    """
    将一行状态转换为整数表示：
      1. 将 -1 转换为 0
      2. 拼接为二进制字符串（最高位在最左边）
      3. 转换为整数
    """
    # 将 -1 替换为 0
    state_converted = np.where(state_row == -1, 0, state_row)
    # 拼接成二进制字符串
    bit_str = ''.join(str(int(bit)) for bit in state_converted)
    return int(bit_str, 2)

def main():
    input_file = "fe2s2_cdfci_0.0_107437784states.npz"
    output_file = "output.dat"
    
    # 加载 npz 文件
    data = np.load(input_file)
    ci_states = data["ci_states"]
    ci_probs = data["ci_probs"]
    
    records = []
    # 遍历每一行，转换状态为整数并计算 ci_probs 的模方
    for i in range(ci_states.shape[0]):
        state_int = state_to_int(ci_states[i])
        prob = ci_probs[i]
        prob_mod2 = abs(prob)**2
        # 仅保留模方大于或等于 MIN_THRESHOLD 的记录
        if prob_mod2 >= MIN_THRESHOLD:
            records.append((state_int, prob_mod2))
    
    # 根据模方由大到小排序
    records_sorted = sorted(records, key=lambda x: x[1], reverse=True)
    
    # 将结果写入 output.dat 文件，格式为：
    # 序号  状态整数  ci_probs 的模方
    with open(output_file, "w") as f:
        f.write("Index State_Int CI_Prob_Mod2\n")
        for idx, (state_int, prob_mod2) in enumerate(records_sorted, start=1):
            f.write(f"{idx} {state_int} {prob_mod2}\n")
    
    print(f"处理完成，结果已保存至 {output_file}")

if __name__ == "__main__":
    main()

