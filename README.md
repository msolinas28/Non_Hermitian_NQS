# Non-Hermitian Neural Quantum States

This repository contains the code that accompanies the framework developed in our paper (see https://arxiv.org/abs/2508.01072). It implements neural-network variational methods for non-Hermitian quantum spin systems and provides practical tools to:

- build and optimize biorthogonal left/right variational states,
- compute JAX-accelerated biorthogonal expectation values and variance-based cost functions

The implementation is built on NetKet and JAX. See the paper for the theoretical background; the `Tutorial/nh_driver.ipynb` notebook provides a hands-on demo of common workflows.

## Highlights
- Non-Hermitian Transverse-Field Ising Model implementation (`hamiltonian.cTFIM`).
- JAX-accelerated routines to compute biorthogonal expectation values between left and right variational states (`Custom_nk/nh_expect.py`, `Custom_nk/cc_nh_expect.py`).
- 
- Example notebook: `Tutorial/nh_driver.ipynb` demonstrating common workflows.

## Non-Hermitian VMC utilities

This repository contains utilities to run variational Monte Carlo (VMC) for non-Hermitian, biorthogonal quantum problems:

- `Custom_nk/nh_mcstate.py` — `NHMCState` extends NetKet's `MCState` to represent left/right variational states in a biorthogonal pair. It adds convenience methods to compute non-Hermitian expectations (symmetry-aware and generic biorthogonal expectations) and a registry for symmetry-specific expectation handlers.

- `Custom_nk/nh_variance_opt.py` — VMC drivers (`NHDriver`, `NHDriverSymm`) that optimize variational parameters using a generalized variance cost function for non-Hermitian Hamiltonians. The drivers support different optimization phases (fixed / transition / self-consistent) and can optimize left/right states either jointly or with symmetry-specific routines.

- `Custom_nk/variance.py` — `Variance` constructs the left and right variance operators from a Hamiltonian and its adjoint.

## Data and Figures

- `Data/` — contains all the datasets used in the paper (raw and processed numerical results, CSVs, and any auxiliary data files used to produce the results).
- `Data_plot/` — contains the figures shown in the paper and the notebooks used to generate them (`Data_plot/paper_plot.ipynb` reproduces the plots used in the manuscript).