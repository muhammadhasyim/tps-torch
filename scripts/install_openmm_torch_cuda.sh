#!/bin/bash
# Install openmm-torch with CUDA platform kernels in a pip-based environment.
#
# The PyPI 'openmmtorch' package is a Python stub without the C++/CUDA
# platform plugins that TorchForce requires.  This script uses micromamba
# to extract the conda-forge build and copies the necessary files into
# the active pip environment.
#
# Prerequisites: curl, tar, Python 3.12, PyTorch with CUDA, OpenMM (pip)
#
# Usage:
#   bash scripts/install_openmm_torch_cuda.sh
#
# After running, set LD_LIBRARY_PATH before launching training:
#   source scripts/openmm_torch_env.sh
#   python train_committor.py --platform CUDA ...

set -euo pipefail

PYTHON=${PYTHON:-python3}
SITE_PACKAGES=$($PYTHON -c "import site; print(site.getsitepackages()[0])")
OPENMM_PLUGINS="$SITE_PACKAGES/OpenMM.libs/lib/plugins"
OPENMM_LIB="$SITE_PACKAGES/OpenMM.libs/lib"
TORCH_LIB=$($PYTHON -c "import torch; import os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))")

echo "=== openmm-torch CUDA installer ==="
echo "Site-packages: $SITE_PACKAGES"
echo "OpenMM plugins: $OPENMM_PLUGINS"
echo "Torch lib: $TORCH_LIB"

# Step 1: Install micromamba if not present
MAMBA=/tmp/bin/micromamba
if [ ! -x "$MAMBA" ]; then
    echo "Installing micromamba..."
    mkdir -p /tmp/bin
    $PYTHON -c "
import urllib.request, tarfile
urllib.request.urlretrieve('https://micro.mamba.pm/api/micromamba/linux-64/latest', '/tmp/micromamba.tar.bz2')
with tarfile.open('/tmp/micromamba.tar.bz2', 'r:bz2') as t:
    t.extractall('/tmp', filter='data')
"
    chmod +x "$MAMBA"
fi
echo "micromamba: $($MAMBA --version)"

# Step 2: Create minimal conda env with openmm-torch
echo "Creating conda env with openmm-torch..."
$MAMBA create -y -n _openmm_torch \
    -c conda-forge \
    "openmm-torch>=1.5" \
    "python=3.12" \
    --root-prefix /tmp/mamba 2>&1 | tail -5

CONDA_SP="/tmp/mamba/envs/_openmm_torch/lib/python3.12/site-packages"
CONDA_LIB="/tmp/mamba/envs/_openmm_torch/lib"

# Step 3: Copy files
echo "Copying openmm-torch files..."

# Python wrapper + SWIG C extension
cp "$CONDA_SP/openmmtorch.py" "$SITE_PACKAGES/"
cp "$CONDA_SP"/_openmmtorch.cpython-*.so "$SITE_PACKAGES/"

# TorchForce plugin shared libraries
for lib in libOpenMMTorchCUDA.so libOpenMMTorchOpenCL.so libOpenMMTorchReference.so; do
    if [ -f "$CONDA_LIB/plugins/$lib" ]; then
        cp "$CONDA_LIB/plugins/$lib" "$OPENMM_PLUGINS/"
        echo "  Copied plugin: $lib"
    fi
done

# Main libOpenMMTorch.so
cp "$CONDA_LIB/libOpenMMTorch.so" "$OPENMM_LIB/"
echo "  Copied libOpenMMTorch.so"

# Step 4: Create shim directory for missing runtime libs
SHIM_DIR=/tmp/openmm_torch_shims
mkdir -p "$SHIM_DIR"
for lib in libcudart.so.12 libstdc++.so.6; do
    src=$(find "$CONDA_LIB" -maxdepth 1 -name "$lib" | head -1)
    if [ -n "$src" ]; then
        ln -sf "$src" "$SHIM_DIR/$lib"
        echo "  Shimmed: $lib"
    fi
done

# Step 5: Write env script
ENV_SCRIPT="$(dirname "$0")/openmm_torch_env.sh"
cat > "$ENV_SCRIPT" <<ENVEOF
# Source this before running training with TorchForce on CUDA:
#   source scripts/openmm_torch_env.sh
export LD_LIBRARY_PATH="$TORCH_LIB:$OPENMM_LIB:$OPENMM_PLUGINS:$SHIM_DIR:\${LD_LIBRARY_PATH:-}"
ENVEOF
echo "Wrote $ENV_SCRIPT"

# Step 6: Verify
echo ""
echo "=== Verification ==="
LD_LIBRARY_PATH="$TORCH_LIB:$OPENMM_LIB:$OPENMM_PLUGINS:$SHIM_DIR" \
    $PYTHON -c "
from openmmtorch import TorchForce
print('TorchForce import: OK')
import openmm, torch, tempfile
class D(torch.nn.Module):
    def forward(self, p): return p.sum() * 0.0
t = torch.jit.trace(D(), torch.randn(2,3))
f = tempfile.NamedTemporaryFile(suffix='.pt', delete=False)
t.save(f.name); f.close()
s = openmm.System(); s.addParticle(1.0); s.addParticle(1.0)
s.addForce(TorchForce(f.name))
ctx = openmm.Context(s, openmm.VerletIntegrator(0.001), openmm.Platform.getPlatformByName('CUDA'))
ctx.setPositions([[0,0,0],[1,0,0]])
print(f'TorchForce on CUDA: OK (energy={ctx.getState(getEnergy=True).getPotentialEnergy()})')
"
echo "=== Done ==="
