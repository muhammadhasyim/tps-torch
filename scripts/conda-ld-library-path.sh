#!/usr/bin/env bash
# Prepend Conda’s lib/ to LD_LIBRARY_PATH (Linux) so Conda’s libstdc++ is used
# before the system’s when loading OpenMM and similar native libraries.
# Usage (after: conda activate tpstorch):
#   source scripts/conda-ld-library-path.sh

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "Source this file instead of running it:  source $0" >&2
  exit 1
fi

if [[ "$(uname -s)" != "Linux" ]]; then
  return 0
fi
if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "conda-ld-library-path: CONDA_PREFIX is unset; run: conda activate <env>" >&2
  return 0
fi
_lib="${CONDA_PREFIX}/lib"
if [[ ! -d "$_lib" ]]; then
  return 0
fi
case ":${LD_LIBRARY_PATH:-}:" in
  *":$_lib:"*) ;;
  *)
    export LD_LIBRARY_PATH="${_lib}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    ;;
esac
unset _lib
