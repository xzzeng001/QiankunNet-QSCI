#!/bin/bash
#SBATCH --job-name=Nxzzeng
#SBATCH --ntasks=64  # 总核数
#SBATCH --nodes=1  # 总节点数
#SBATCH --output=%j.log
#SBATCH --partition=fat  # 队列名，可选debug，normal，long等

# 提交作业之前，先加载环境：
# module use /public/software/modulefiles/
# module load vasp/6.4.2/intelmpi-intelmkl

# 运行VASP
#rm -rf /public/home/xzzeng/.cache/torch_extensions/py38_cu121
stdbuf -o0 -e0 python -u entropy_mutual_plot_from_ci.py
#mpirun vasp_std > runlog

