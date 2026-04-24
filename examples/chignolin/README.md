# Chignolin (1UAO) -- Committor Training with MACE Foundation Model

Chignolin is a 10-residue mini-protein (sequence: GYDPETGTWG) that folds
into a beta-hairpin.  With implicit solvent (GBn2), the system has ~138
atoms -- small enough for the MACE-OFF23 foundation model on a 12 GB GPU.

## Quick Start

```bash
# 1. Setup system (auto-downloads 1UAO.pdb from RCSB)
python setup_system.py --fetch_pdb 1UAO --solvent_model implicit --platform CPU

# 2. Generate unfolded snapshot
python make_unfolded_snapshot.py \
    --input_pdb output/minimized.pdb \
    --system_xml output/system.xml \
    --output output/unfolded.pdb

# 3. Train committor (MACE-OFF23 small backbone, auto-downloaded)
python train_committor.py \
    --system_xml output/system.xml \
    --pdb_folded output/minimized.pdb \
    --pdb_unfolded output/unfolded.pdb \
    --foundation_model mace-off-small \
    --platform CUDA \
    --n_iterations 100
```

## Solvent Options

| `--solvent_model` | Atoms  | Description                          |
|-------------------|--------|--------------------------------------|
| `implicit`        | ~138   | GBn2 implicit solvent (recommended)  |
| `vacuum`          | ~138   | Gas phase (no solvent)               |
| `explicit`        | ~3000+ | TIP3P water box                      |

## Foundation Model Options

| `--foundation_model` | Params | License | Best for               |
|----------------------|--------|---------|------------------------|
| `mace-off-small`     | ~694K  | ASL     | Proteins (recommended) |
| `mace-off-medium`    | ~4.7M  | ASL     | Higher accuracy        |
| `mace-off-large`     | ~5.7M  | ASL     | Highest accuracy       |
| `mace-mp-small`      | ~694K  | MIT     | Inorganic materials    |
| `distance_encoder`   | varies | --      | Testing only           |
