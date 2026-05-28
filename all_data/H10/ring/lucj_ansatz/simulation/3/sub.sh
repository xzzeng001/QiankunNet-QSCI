#!/bin/bash
#SBATCH --job-name=Nxzzeng
#SBATCH --ntasks=64  # 总核数
#SBATCH --nodes=1  # 总节点数
#SBATCH --output=%j.log
#SBATCH --partition=normal  # 队列名，可选debug，normal，long等

# 提交作业之前，先加载环境：
# module use /public/software/modulefiles/
# module load vasp/6.4.2/intelmpi-intelmkl

# 运行VASP
stdbuf -o0 -e0 python main.py
#mpirun vasp_std > runlog

