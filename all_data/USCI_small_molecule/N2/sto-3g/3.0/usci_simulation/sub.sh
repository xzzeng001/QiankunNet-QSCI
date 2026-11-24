#!/bin/bash
#SBATCH --job-name=Nxzzeng
#SBATCH --ntasks=64  # 总核数
#SBATCH --nodes=1  # 总节点数
#SBATCH --ntasks-per-node=64  # 每个节点使用的核数
#SBATCH --cpus-per-task=1  # 每个核的线程数。对VASP而言总为1
#SBATCH --output=%j.log
#SBATCH --partition=normal  # 队列名，可选debug，normal，long等

# 提交作业之前，先加载环境：
# module use /public/software/modulefiles/
# module load vasp/6.4.2/intelmpi-intelmkl

# 运行VASP
stdbuf -o0 -e0 python q2chem_vqe.py
#mpirun vasp_std > runlog

