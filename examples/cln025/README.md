# CLN025 (Chignolin) Committor Training Example

End-to-end example for training a committor function on the CLN025
mini-protein (10 residues, sequence YYDPETGTWY) and validating against
the DESRES brute-force folding trajectories.

## Prerequisites

**Easiest:** from the **repository root**, create the project Conda environment (OpenMM, `libstdc++`, PyTorch, PDBFixer, mdtraj, dev tools, and the editable `committorch` install):

```bash
conda env create -f environment.yml
conda activate tpstorch
```

Or install manually:

- **OpenMM** >= 8.0: prefer `conda install -c conda-forge openmm` (with `libstdcxx-ng` from the same channel) so native libraries stay consistent. **GPU (CUDA):** if `--platform CUDA` errors or falls back, keep OpenMM from conda-forge and align CUDA/NVRTC; see the troubleshooting section below.
- **PDBFixer** (recommended for RCSB PDBs): `conda install -c conda-forge pdbfixer` (or the env above) — repairs missing atoms/residues and adds hydrogens before solvation. `setup_system.py` uses it by default; use `--no-pdbfixer` only if you have an all-atom, protonated structure.
- **openmm-torch** (NN forces, conda-forge) / **openmmtorch** (PyPI name): included in the root `environment.yml` as the Conda package `openmm-torch` (avoids a pip OpenMM stack alongside Conda)
- **mdtraj**: included in `environment.yml`, or `conda install -c conda-forge mdtraj` / `pip install mdtraj`
- **committorch**: `pip install -e .` from the repository root (already done in `environment.yml`)

Optional:
- **mace-torch**: `pip install mace-torch` (for MACE encoder)

### GPU: make OpenMM CUDA work (`CUDA_ERROR_UNSUPPORTED_PTX_VERSION` / 222)

The driver must be able to JIT the PTX that ships inside your **conda-forge** `openmm` build. Mismatches (very common) produce **error 222**.

1. **Check driver vs stack:** from the repository root, run `python scripts/verify_openmm_cuda.py` (and `python check_openmm_gpu.py` in this directory). The number **CUDA Version** in `nvidia-smi` is the *maximum* your driver supports.
2. **This repo’s default `environment.yml` pins `openmm` to 8.2.x** (with `openmm-torch` 1.5). Conda-forge **OpenMM 8.5** needs a driver whose supported CUDA is **~12.9+**; if you only see 12.4 in `nvidia-smi`, **upgrade the NVIDIA driver** or use the 8.2 stack, not 8.5.
3. **Older / fixed drivers (Linux):** create the env from **`environment-cuda-linux.yml`** in the repo root, or `conda install -c conda-forge cudatoolkit=11.8 "openmm>=8.2,<8.3"`.
4. One **conda prefix** for OpenMM and CUDA user libs; avoid a **pip** `openmm` in the same env.
5. For a quick unblocked run, `setup_system.py --platform CPU` is small; the scripts fall back to OpenCL/CPU if CUDA `Context` creation fails.

## Data Requirements

1. **CLN025 PDB structure**: `5AWL.pdb` is included in this directory (RCSB: [5AWL](https://www.rcsb.org/structure/5AWL), crystal structure of the CLN025 chignolin variant).

2. **DESRES trajectories** (for validation only):
   Lindorff-Larsen et al., Science 2011. There is **no** stable public direct URL in this repository: you must download the official trajectories from D. E. Shaw Research via their portal (`https://www.deshawresearch.com/downloads/download_trajectory_sarcomere.cgi/`) and accept their terms. Place the resulting directory (for example `DESRES-Trajectory_CLN025-0` with `*.dcd` or `*.xtc` plus a matching `*.pdb` topology) on disk and pass its path to `validate.py`. For a **trajectory-only** smoke test of the committor along a folding path (not comparable to DESRES numerically), you can use another public dataset such as the Zenodo chignolin MD set (DOI `10.5281/zenodo.6349893`, GROMACS `xtc` + topology), keeping in mind different force fields and conditions.

3. **MACE foundation weights** (training with `--foundation_model mace-off-small`, default):
   Downloaded automatically on first use by `mace-torch` and cached under `~/.cache/mace/` (see `src/committorch/models/equivariant/mace.py`).

4. **Topology match for validation**:
   `validate.py` loads the same MACE architecture as training and requires a **reference PDB** whose atom count and order match the trained system. A DESRES all-atom explicit-solvated trajectory will **not** match a model trained on an implicit or vacuum system unless you preprocess frames to the same topology. Use the same solvated PDBs you trained on (for example `output/minimized.pdb`) as `--reference_pdb` when it matches your trajectory.

## Workflow

### Step 1: Build the OpenMM System

**Explicit solvent** (TIP3P, ~4,800 atoms—requires a large GPU for MACE training):

```bash
python setup_system.py --pdb 5AWL.pdb --forcefield amber --platform CUDA --strict_openmm_platform
```

**Implicit solvent** (GBn2, protein-only, ~166 atoms—fits on consumer GPUs):

```bash
# First strip crystallographic waters from 5AWL.pdb (HOH residues)
python -c "from openmm.app import PDBFile, Modeller; import openmm.app as app; \
    pdb = PDBFile('5AWL.pdb'); m = Modeller(pdb.topology, pdb.positions); \
    m.delete([r for r in m.topology.residues() if r.name == 'HOH']); \
    app.PDBFile.writeFile(m.topology, m.positions, open('5AWL_dry.pdb','w'))"

# Then build implicit system
python setup_system.py --pdb 5AWL_dry.pdb --solvent_model implicit --platform CUDA --strict_openmm_platform
```

Use `--strict_openmm_platform` if you want OpenMM to **fail** when the requested `--platform` is unavailable (no silent fallback to OpenCL/CPU). Omit it to allow the usual fallback chain.

This creates:
- `output/system.xml`: serialized OpenMM System
- `output/minimized.pdb`: energy-minimized structure

### Step 2: Prepare boundary structures (solvated PDBs)

`train_committor.py` loads **`system.xml`** (solvated box) and builds a
`Simulation` from that serialized system. Every PDB you pass must have the
**same number of atoms** as that system. **Do not** use the raw RCSB `5AWL.pdb`
(protein only) as `pdb_folded` or `pdb_unfolded`—it will not match the solvated
`system.xml`.

- **Folded (A):** use the solvated minimized structure from Step 1, e.g.
  `output/minimized.pdb`.
- **Unfolded (B):** generate a solvated snapshot in an unfolded basin, e.g. by
  short high-temperature MD starting from `minimized.pdb`:

  ```bash
  python make_unfolded_snapshot.py \
      --input_pdb output/minimized.pdb \
      --system_xml output/system.xml \
      --output output/unfolded_solvated.pdb \
      --platform CUDA \
      --strict_openmm_platform
  ```

  By default the script **drops the barostat** from `system.xml` (NPT at very high
  *T* is prone to NaN coordinates), **minimizes**, **warms at 340 K**, then runs at
  **450 K** with a **1 fs** timestep. Override with `--temperature`, `--n_steps`,
  `--keep_barostat` only if you know you need them.

Optional: if you use a different folded coordinate file but want the topology
from `minimized.pdb`, pass `--reference_pdb output/minimized.pdb` to
`train_committor.py` in addition to `--pdb_folded` / `--pdb_unfolded`.

### Step 3: Train the Committor

**GPU-only PyTorch + strict OpenMM:** add `--require_cuda` (requires `--platform CUDA`) and optionally `--strict_openmm_platform` so OpenMM does not fall back if CUDA context creation fails.

Using V_K + OPES (Paper 2 method) with the default **MACE-OFF** small backbone:

```bash
python train_committor.py \
    --system_xml output/system.xml \
    --pdb_folded output/minimized.pdb \
    --pdb_unfolded output/unfolded_solvated.pdb \
    --method opes \
    --foundation_model mace-off-small \
    --n_iterations 100 \
    --n_sim_steps 1000 \
    --platform CUDA \
    --require_cuda \
    --strict_openmm_platform
```

**VRAM:** MACE on the default **explicit-solvated** system (~4.8k atoms) is memory-heavy during TorchScript tracing of the committor bias. Expect to need a **large** GPU (often 24 GB+ class) for full runs; a lightweight alternative for debugging is `--foundation_model distance_encoder` (test-only, \(O(N^2)\) cost).

Test-only distance encoder (no MACE; small-GPU or CPU-friendly for plumbing checks):

```bash
python train_committor.py \
    --system_xml output/system.xml \
    --pdb_folded output/minimized.pdb \
    --pdb_unfolded output/unfolded_solvated.pdb \
    --method opes \
    --foundation_model distance_encoder \
    --n_iterations 100 \
    --n_sim_steps 1000 \
    --platform CUDA
```

Using umbrella sampling (Paper 1 method):

```bash
python train_committor.py \
    --system_xml output/system.xml \
    --pdb_folded output/minimized.pdb \
    --pdb_unfolded output/unfolded_solvated.pdb \
    --method umbrella \
    --n_iterations 100 \
    --platform CUDA
```

### Step 4: Validate Against DESRES (or other trajectories)

```bash
python validate.py \
    --model_path output/committor_final.pt \
    --reference_pdb output/minimized.pdb \
    --desres_dir /path/to/DESRES-Trajectory_CLN025-0 \
    --foundation_model mace-off-small \
    --stride 100
```

The trajectory and `--reference_pdb` must have the **same number of atoms** as the model (see “Topology match for validation” above). For `--foundation_model distance_encoder`, add `--n_atoms` equal to the atom count of the training system.

For GPU inference only, add `--require_cuda` when PyTorch should run on CUDA.

## Expected Results

- **State assignment accuracy**: >80% agreement between committor-based
  (q < 0.5 = folded) and RMSD-based state labels
- **Folding rate**: should be within an order of magnitude of the DESRES
  brute-force rate (~1/600 ns for CLN025 at 340 K)

## Runtime Estimates

| Step | CPU | GPU (A100) |
|------|-----|-----------|
| System setup | ~1 min | ~1 min |
| Training (100 iter, 1000 steps/iter) | ~hours | ~30 min |
| Validation | ~10 min | ~5 min |

## References

1. Hasyim, Batton, Mandadapu. arXiv:2107.13522 (2022).
2. Trizio, Kang, Parrinello. arXiv:2410.17029 (2025).
3. Lindorff-Larsen et al. Science 334, 517-520 (2011).
