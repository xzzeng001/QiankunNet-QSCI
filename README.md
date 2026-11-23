# QiankunNet-QSCI

Research code, quantum circuits, and calculation data accompanying **[Transformer refined quantum sampling for strongly correlated electronic structure](https://doi.org/10.48550/arXiv.2605.24617)**.

QiankunNet-QSCI combines quantum-selected configuration interaction (QSCI) with the QiankunNet transformer model. The unitary selected configuration interaction (USCI) ansatz prepares circuits for sampling electronic configurations, which are used to guide neural-network wavefunction refinement.

This repository contains calculation-specific scripts and archived results for the study, including USCI simulations, circuit generation, selected-CI calculations, and analysis of molecular and iron-sulfur systems. Dependencies and input files vary by calculation; the archive does not provide a single installation or end-to-end training command.

## Repository contents

| Directory | Contents |
| --- | --- |
| [`all_data/USCI_small_molecule/`](all_data/USCI_small_molecule/) | H2O, N2, O2, C4H4, Cu2O2, and Cr2 calculations, including molecular inputs, selected-CI results, USCI circuits, and simulation scripts. |
| [`all_data/H10/ring/`](all_data/H10/ring/) | H10 ring calculations: Hamiltonian preparation, CCSD references, USCI and LUCJ circuits and simulations, sampling, and post-processing. |
| [`all_data/Fe2S2_40/`](all_data/Fe2S2_40/) | The 40-qubit Fe2S2 study, with USCI and LUCJ calculations, circuit variants, energy records, and neural-wavefunction analysis and plotting scripts. |
| [`all_data/Pcluster/`](all_data/Pcluster/) | P-cluster geometry, SHCI calculations, USCI circuits, configuration-expansion scripts and archives, experimental circuit mappings, and reference energies. |

Within the small-molecule directories, common subdirectories are:

- `pyscf/`: molecular calculations and integral generation.
- `shci/`: selected-CI inputs, outputs, and Hamiltonian conversion scripts.
- `circuit/`: circuit generation, exported circuits where available, and visualization scripts.
- `simulation/`: classical USCI simulation and optimization scripts.
- `qsci2/`: sampling and selected-subspace diagonalization scripts, where present.

The files available in each subdirectory differ between systems.

## Get the repository

```bash
git clone https://github.com/xzzeng001/QiankunNet-QSCI.git
cd QiankunNet-QSCI
```

The repository includes stored calculation outputs and binary data as well as source code. You can inspect these results without rerunning the simulations.

## Software requirements

The repository does not currently include a pinned environment or package manifest. Choose dependencies for the calculation you intend to run and check the imports in its scripts.

| Component | Used for |
| --- | --- |
| Python, NumPy, SciPy | Numerical calculations and data processing. |
| PySCF, OpenFermion | Electronic-structure calculations and Hamiltonian preparation. |
| Qiskit, Qiskit Aer | Circuit construction, export, and simulation in scripts that import these packages. |
| `ffsim` | Fermionic simulations and LUCJ calculations. |
| Matplotlib, NetworkX | Plotting and circuit-connectivity analysis. |
| PyTorch | Selected configuration-analysis and diagonalization scripts. |
| Qiskit Algorithms, Qiskit addon SQD | Additional workflows in scripts that import these packages. |

Some workflows also need separately configured software:

- `q2chem`: imported by USCI simulation scripts such as `q2chem_vqe.py`; its source is not bundled here. Adjust the scripts' import paths to your installation.
- `cpp_lib.build.local_energy_module`: a compiled extension imported by several `diag.py` files; its build sources are not included in this repository.
- SHCI and DMRG backends and the relevant PySCF extensions for scripts that invoke these solvers. Some directories contain an `shci.e` executable; compatibility depends on the target machine.
- MATLAB for the `.m` plotting scripts. Some Python plots also enable LaTeX text rendering and require a working LaTeX installation or a change to that plotting setting.

Several circuit scripts import older Qiskit interfaces such as `assemble`. Use a compatible environment or adapt those calls before running them; compatibility with arbitrary current package versions has not been established.

## Inspect and run a calculation

### Inspect archived data

From the repository root, this read-only example prints the header of the stored N2 circuit using only the Python standard library:

```bash
python -c "from pathlib import Path; p = Path('all_data/USCI_small_molecule/N2/circuit/circuit.qasm'); print('\n'.join(p.read_text().splitlines()[:6]))"
```

Other useful starting points are the [P-cluster energy summary](all_data/Pcluster/pcluster_data.txt), the [Fe2S2 analysis directory](all_data/Fe2S2_40/nnqs/), and the P-cluster [alpha](all_data/Pcluster/Fe8S7_alpha/README.txt) and [beta](all_data/Pcluster/Fe8S7_beta/README.txt) circuit notes.

### Run a system-specific script

Run scripts from their own directories because many input paths are relative to the working directory. For example, after configuring `q2chem` and the imports used by the N2 simulation:

```bash
cd all_data/USCI_small_molecule/N2/simulation
python q2chem_vqe.py
```

This script reads `shci_results.txt` and `ham_qubit.npy` from that directory, constructs the USCI circuit, and starts a matrix-product-state simulation and optimization. Review its thread counts, bond dimension, convergence settings, and output paths before launching a calculation. It is a research calculation, not a lightweight installation test.

The `sub.sh` files are SLURM batch-job examples with machine-specific resource settings. Update them for your cluster before submitting jobs. Scripts may write or overwrite result files in their working directories; use a separate working copy when preserving archived outputs.

## Data formats and availability

- `FCIDUMP` and `*.FCIDUMP`: electronic Hamiltonian integrals; some larger inputs are stored in ZIP archives.
- `*.qasm`: exported OpenQASM circuits.
- `QCIS_circuit.txt` and mapping text files: hardware circuit instructions and qubit/coupler mappings.
- `*.npy` and `*.npz`: numerical arrays, Hamiltonian objects, parameters, or configuration data, depending on the file and its loading script.
- `*.xyz`: molecular geometries.
- `*.txt`, `*.dat`, and `*.log`: calculation inputs, configuration coefficients, energies, and execution records; layouts are specific to each calculation.

Check a file's consuming script before interpreting its array layout, orbital ordering, or energy convention. Some Hamiltonian files are loaded as pickled Python objects and require the corresponding software classes.

For P-cluster calculations, inspect and extract the required ZIP archives before running scripts that expect uncompressed inputs. The experimental notes describe `single_qubit.npy`, but that file is not included in the alpha or beta directory. Likewise, `config_expansion/main.py` expects `exp1_alpha_confs.npz` and `exp1_beta_confs.npz`, which are not present alongside the script. These inputs and the external modules above must be supplied before the affected workflows can run.

## Citation

If you use this code or data, please cite the accompanying [paper](https://arxiv.org/abs/2605.24617):

```bibtex
@misc{zeng2026transformer,
  title         = {Transformer refined quantum sampling for strongly correlated electronic structure},
  author        = {Xiongzhi Zeng and Ming Gong and Bowen Kan and Yi Fan and Huan Ma and
                   Jianbin Cai and Yancheng Liu and Naibin Zhou and Tao Jiang and
                   Shaojun Guo and Zhijie Fan and Zongkang Zhang and Yuan Li and
                   Sirui Cao and Kai Yan and Xiaobo Zhu and Yi Luo and Honghui Shang and
                   Zhenyu Li and Jian-Wei Pan and Jinlong Yang},
  year          = {2026},
  eprint        = {2605.24617},
  archivePrefix = {arXiv},
  primaryClass  = {quant-ph},
  doi           = {10.48550/arXiv.2605.24617},
  url           = {https://arxiv.org/abs/2605.24617}
}
```

## License

The repository is distributed under the [MIT License](license.txt).
