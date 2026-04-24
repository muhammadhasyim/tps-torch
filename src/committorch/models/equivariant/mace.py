"""MACE backbone adapter for committor prediction.

Wraps a pre-trained MACE foundation model (MACE-OFF23, MACE-MP-0, or
MACE-OMOL-0), extracts invariant node embeddings, pools them to a
graph-level feature vector, and feeds into the committor head MLP.

The ``download_mace_model`` function auto-downloads pre-trained
checkpoints via the ``mace-torch`` package (cached in ``~/.cache/mace``).

When ``mace-torch`` is not installed, a lightweight placeholder encoder
allows running tests and CI without the heavy dependency.

Requires (optional): ``pip install mace-torch``

References
----------
.. [1] Batatia et al., MACE. NeurIPS 2022 / arXiv:2206.07697.
.. [2] Kovacs et al., MACE-OFF23. arXiv:2312.15211.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import torch
import torch.nn as nn

from .backbone import PretrainedBackbone
from .graph_input import build_batch_graphs, scatter_mean

logger = logging.getLogger(__name__)

HAS_MACE = False
try:
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    from mace.calculators.foundations_models import mace_mp, mace_off
    from mace.tools import torch_tools as mace_torch_tools

    HAS_MACE = True
except (ImportError, Exception):
    pass

_MACE_FAMILIES = {
    "mace-off": {"loader": "mace_off", "sizes": ("small", "medium", "large")},
    "mace-mp": {"loader": "mace_mp", "sizes": ("small", "medium", "large")},
}

AVAILABLE_FOUNDATION_MODELS: list[str] = []
for fam, info in _MACE_FAMILIES.items():
    for sz in info["sizes"]:
        AVAILABLE_FOUNDATION_MODELS.append(f"{fam}-{sz}")


def _require_mace() -> None:
    if not HAS_MACE:
        raise ImportError(
            "mace-torch is required for MACE foundation models. "
            "Install with: pip install mace-torch"
        )


def download_mace_model(
    model_name: str = "mace-off",
    size: str = "small",
    device: str = "cpu",
    default_dtype: str = "float64",
) -> nn.Module:
    """Download a pre-trained MACE foundation model.

    Uses the ``mace-torch`` auto-download API which caches checkpoints
    in ``~/.cache/mace/``.

    Parameters
    ----------
    model_name : str
        Foundation model family: ``"mace-off"`` (organic/bio, ASL license),
        ``"mace-mp"`` (materials, MIT license).
    size : str
        Model size: ``"small"``, ``"medium"``, or ``"large"``.
    device : str
        Target device (``"cpu"`` or ``"cuda"``).
    default_dtype : str
        PyTorch default dtype during loading (``"float32"`` or ``"float64"``).
        MACE models are distributed in float64; use ``"float64"`` to avoid
        dtype mismatches during loading, then cast afterward if needed.

    Returns
    -------
    nn.Module
        Raw MACE PyTorch model (not an ASE calculator).
    """
    _require_mace()

    if model_name not in _MACE_FAMILIES:
        raise ValueError(
            f"Unknown model family '{model_name}'. "
            f"Available: {list(_MACE_FAMILIES.keys())}"
        )
    info = _MACE_FAMILIES[model_name]
    if size not in info["sizes"]:
        raise ValueError(
            f"Unknown size '{size}' for {model_name}. Available: {list(info['sizes'])}"
        )

    with mace_torch_tools.default_dtype(default_dtype):
        if model_name == "mace-off":
            model = mace_off(
                model=size,
                device=device,
                default_dtype=default_dtype,
                return_raw_model=True,
            )
        elif model_name == "mace-mp":
            model = mace_mp(
                model=size,
                device=device,
                default_dtype=default_dtype,
                return_raw_model=True,
            )
        else:
            raise ValueError(f"No loader for {model_name}")

    logger.info(
        "Loaded %s-%s (%d params, r_max=%.1f A)",
        model_name,
        size,
        sum(p.numel() for p in model.parameters()),
        float(model.r_max),
    )
    return model


def parse_foundation_model_spec(spec: str) -> tuple[str, str]:
    """Parse a foundation model specifier like ``"mace-off-small"`` into
    ``("mace-off", "small")``.

    Parameters
    ----------
    spec : str
        Model specifier in the format ``"family-size"`` where family is
        ``"mace-off"`` or ``"mace-mp"`` and size is ``"small"``,
        ``"medium"``, or ``"large"``.

    Returns
    -------
    tuple of (str, str)
        ``(model_name, size)``
    """
    for fam in sorted(_MACE_FAMILIES, key=len, reverse=True):
        if spec.startswith(fam + "-"):
            size = spec[len(fam) + 1 :]
            return fam, size
    raise ValueError(
        f"Cannot parse foundation model spec '{spec}'. "
        f"Expected format like 'mace-off-small'. Available: {AVAILABLE_FOUNDATION_MODELS}"
    )


class MACECommittor(PretrainedBackbone):
    """Committor model using a MACE foundation model encoder.

    The MACE encoder extracts SE(3)-invariant per-atom features from atomic
    graphs (within a radial cutoff), pools them to a graph-level vector, and
    projects to a committor head MLP.  The encoder is frozen by default for
    transfer learning.

    Parameters
    ----------
    mace_model : nn.Module or None
        Pre-trained MACE model. If None, a placeholder identity encoder
        is used (for testing only).
    atomic_numbers : torch.Tensor or None
        Atomic numbers for the system, shape ``(n_atoms,)``.
    cutoff : float
        Neighbor list cutoff in angstroms.  If a real MACE model is
        provided, this is overridden by ``model.r_max``.
    feature_dim : int
        Dimension of the pooled graph-level features.
    head_hidden_dims : tuple of int
        Committor head architecture.
    sigmoid_steepness : float
        Output sigmoid steepness.
    freeze_encoder : bool
        Whether to freeze the MACE encoder weights.
    """

    def __init__(
        self,
        mace_model: nn.Module | None = None,
        atomic_numbers: torch.Tensor | None = None,
        cutoff: float = 5.0,
        feature_dim: int = 128,
        head_hidden_dims: tuple[int, ...] = (32, 32),
        sigmoid_steepness: float = 1.0,
        freeze_encoder: bool = True,
    ) -> None:
        if mace_model is None:
            encoder = _PlaceholderEncoder(feature_dim)
        else:
            actual_cutoff = float(mace_model.r_max) if hasattr(mace_model, "r_max") else cutoff
            encoder = _MACEFeatureExtractor(
                mace_model,
                feature_dim,
                actual_cutoff,
                atomic_numbers=atomic_numbers,
            )
            cutoff = actual_cutoff

        super().__init__(
            encoder=encoder,
            feature_dim=feature_dim,
            head_hidden_dims=head_hidden_dims,
            sigmoid_steepness=sigmoid_steepness,
            freeze_encoder=freeze_encoder,
        )

        if atomic_numbers is not None:
            self.register_buffer("atomic_numbers", atomic_numbers.clone())
        else:
            self.register_buffer(
                "atomic_numbers",
                torch.zeros(0, dtype=torch.long),
            )
        self.cutoff = cutoff

    @classmethod
    def from_foundation(
        cls,
        model_name: str = "mace-off",
        size: str = "small",
        atomic_numbers: torch.Tensor | None = None,
        feature_dim: int = 128,
        head_hidden_dims: tuple[int, ...] = (32, 32),
        sigmoid_steepness: float = 1.0,
        freeze_encoder: bool = True,
        device: str = "cpu",
    ) -> MACECommittor:
        """Build a MACECommittor from an auto-downloaded foundation model.

        Parameters
        ----------
        model_name : str
            Foundation model family (``"mace-off"``, ``"mace-mp"``).
        size : str
            Model size (``"small"``, ``"medium"``, ``"large"``).
        atomic_numbers : torch.Tensor or None
            Atomic numbers for the system.
        device : str
            Target device.
        """
        mace_model = download_mace_model(model_name, size, device=device)
        cutoff = float(mace_model.r_max) if hasattr(mace_model, "r_max") else 5.0
        return cls(
            mace_model=mace_model,
            atomic_numbers=atomic_numbers,
            cutoff=cutoff,
            feature_dim=feature_dim,
            head_hidden_dims=head_hidden_dims,
            sigmoid_steepness=sigmoid_steepness,
            freeze_encoder=freeze_encoder,
        )

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str | None = None,
        atomic_numbers: torch.Tensor | None = None,
        cutoff: float = 5.0,
        feature_dim: int = 128,
        head_hidden_dims: tuple[int, ...] = (32, 32),
        sigmoid_steepness: float = 1.0,
        freeze_encoder: bool = True,
        *,
        model_name: str | None = None,
        size: str = "small",
        device: str = "cpu",
    ) -> MACECommittor:
        """Load a MACECommittor from a checkpoint or foundation model.

        Supports two modes:

        1. **Foundation model** (recommended): set ``model_name`` (e.g.
           ``"mace-off"``) and ``size`` (e.g. ``"small"``).  The checkpoint
           is auto-downloaded.
        2. **Custom checkpoint**: set ``checkpoint_path`` to a ``.model``
           or ``.pt`` file.

        Parameters
        ----------
        checkpoint_path : str or None
            Path to a MACE checkpoint file.  Mutually exclusive with
            ``model_name``.
        model_name : str or None
            Foundation model name for auto-download.
        size : str
            Model size (only used with ``model_name``).
        atomic_numbers : torch.Tensor or None
            Atomic numbers for the system.
        cutoff : float
            Neighbor list cutoff (overridden by model's r_max).
        device : str
            Target device.
        """
        if model_name is not None:
            return cls.from_foundation(
                model_name=model_name,
                size=size,
                atomic_numbers=atomic_numbers,
                feature_dim=feature_dim,
                head_hidden_dims=head_hidden_dims,
                sigmoid_steepness=sigmoid_steepness,
                freeze_encoder=freeze_encoder,
                device=device,
            )

        if checkpoint_path is None:
            raise ValueError("Either model_name or checkpoint_path must be provided")

        _require_mace()
        os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
        mace_model = torch.load(checkpoint_path, map_location=device)
        if isinstance(mace_model, dict):
            raise ValueError(
                "Expected a model object, got a state dict. "
                "Load the full model, not just the state dict."
            )

        actual_cutoff = float(mace_model.r_max) if hasattr(mace_model, "r_max") else cutoff
        return cls(
            mace_model=mace_model,
            atomic_numbers=atomic_numbers,
            cutoff=actual_cutoff,
            feature_dim=feature_dim,
            head_hidden_dims=head_hidden_dims,
            sigmoid_steepness=sigmoid_steepness,
            freeze_encoder=freeze_encoder,
        )


class _PlaceholderEncoder(nn.Module):
    """Placeholder for when mace-torch is not installed (testing only)."""

    def __init__(self, output_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(1, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dummy = x.reshape(x.shape[0], -1).mean(dim=-1, keepdim=True)
        return self.linear(dummy)


class _MACEFeatureExtractor(nn.Module):
    """Extract invariant features from a MACE model.

    MACE produces per-atom node features (``node_feats`` key in the output
    dict).  These are mean-pooled over atoms to get a graph-level invariant
    vector, then projected to ``feature_dim``.

    Parameters
    ----------
    mace_model : nn.Module
        Pre-trained MACE model (e.g. from ``download_mace_model``).
    feature_dim : int
        Output dimension after projection.
    cutoff : float
        Neighbor cutoff in angstroms.
    atomic_numbers : torch.Tensor or None
        Fixed atomic numbers for graph building.
    """

    def __init__(
        self,
        mace_model: nn.Module,
        feature_dim: int,
        cutoff: float = 5.0,
        atomic_numbers: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.mace = mace_model
        self.cutoff = cutoff

        if hasattr(mace_model, "atomic_numbers"):
            z_table = mace_model.atomic_numbers
            self.num_species = len(z_table)
            z_to_index = torch.full((int(z_table.max()) + 1,), -1, dtype=torch.long)
            for idx, z in enumerate(z_table):
                z_to_index[int(z)] = idx
            self.register_buffer("_z_to_index", z_to_index)
        else:
            self.num_species = 100
            self.register_buffer(
                "_z_to_index", torch.arange(100, dtype=torch.long)
            )
            z_to_index = self._z_to_index

        if atomic_numbers is not None:
            zn = atomic_numbers.clone().long()
            if hasattr(mace_model, "atomic_numbers"):
                z_t = mace_model.atomic_numbers
                fallback = int(z_t[0])
                oov: set[int] = set()
                for i in range(zn.numel()):
                    z = int(zn[i])
                    if z >= z_to_index.shape[0] or int(z_to_index[z]) < 0:
                        oov.add(z)
                        zn[i] = fallback
                if oov:
                    logger.warning(
                        "MACE has no index for Z=%s; remapped to Z=%d. "
                        "Ions/OOV elements are approximated; remove extra ions for "
                        "best fidelity and smaller graphs.",
                        sorted(oov),
                        fallback,
                    )
            self.register_buffer("atomic_numbers", zn)
        else:
            self.register_buffer("atomic_numbers", torch.zeros(0, dtype=torch.long))

        mace_out_dim = self._detect_output_dim(mace_model)
        self.proj = nn.Linear(mace_out_dim, feature_dim)

    @staticmethod
    def _detect_output_dim(mace_model: nn.Module) -> int:
        """Detect MACE's per-atom node_feats dimension.

        The node_feats output is the concatenation of invariant features from
        all interaction blocks.  This is computed by summing the scalar (l=0)
        irreps dimension across all product blocks.

        Falls back to 128 if the model structure cannot be inspected.
        """
        if hasattr(mace_model, "products"):
            total = 0
            for prod in mace_model.products:
                if hasattr(prod, "linear") and hasattr(prod.linear, "irreps_out"):
                    total += prod.linear.irreps_out.dim
            if total > 0:
                return total
        return 128

    def _build_mace_input(
        self,
        positions: torch.Tensor,
        atomic_numbers: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Build input dict expected by MACE models.

        For non-periodic systems (molecules, proteins) the ``cell`` is set to
        zeros and ``unit_shifts`` are zeros -- no periodic images.

        Parameters
        ----------
        positions : torch.Tensor, shape ``(batch, n_atoms, 3)``
        atomic_numbers : torch.Tensor, shape ``(n_atoms,)``

        Returns
        -------
        dict
            Keys matching the MACE ``forward`` / ``prepare_graph`` signature:
            ``positions``, ``node_attrs``, ``edge_index``, ``shifts``,
            ``unit_shifts``, ``cell``, ``batch``, ``ptr``.
        """
        if positions.dim() == 2:
            positions = positions.unsqueeze(0)

        batch_size = positions.shape[0]
        n_atoms_per = positions.shape[1]
        z = atomic_numbers.to(device=positions.device, dtype=torch.long)
        graph = build_batch_graphs(positions, z, self.cutoff)

        z_idx = self._z_to_index.to(positions.device)
        species_indices = z_idx[graph.atomic_numbers.to(z_idx.device, dtype=torch.long)]

        n_edges = graph.edge_index.shape[1]

        return {
            "positions": graph.positions,
            "node_attrs": nn.functional.one_hot(
                species_indices, num_classes=self.num_species
            ).to(positions.dtype),
            "edge_index": graph.edge_index,
            "shifts": graph.edge_vectors,
            "unit_shifts": torch.zeros(
                (n_edges, 3), dtype=positions.dtype, device=positions.device
            ),
            "cell": torch.zeros(
                (batch_size, 3, 3), dtype=positions.dtype, device=positions.device
            ),
            "batch": graph.batch_index,
            "ptr": torch.tensor(
                [i * n_atoms_per for i in range(batch_size + 1)],
                dtype=torch.long,
                device=positions.device,
            ),
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract and pool MACE features.

        Parameters
        ----------
        x : torch.Tensor, shape ``(batch, n_atoms*3)`` or ``(batch, n_atoms, 3)``
            Positions in **nanometers** (OpenMM, training scripts, and mdtraj).

        Returns
        -------
        torch.Tensor, shape ``(batch, feature_dim)``
        """
        if self.atomic_numbers.numel() == 0:
            raise RuntimeError(
                "atomic_numbers not set. Provide them at construction."
            )

        n_atoms = self.atomic_numbers.shape[0]
        if x.dim() == 2 and x.shape[-1] == n_atoms * 3:
            positions = x.reshape(-1, n_atoms, 3)
        elif x.dim() == 3:
            positions = x
        else:
            positions = x.reshape(-1, n_atoms, 3)

        mace_dtype = next(self.mace.parameters()).dtype
        positions = positions.to(dtype=mace_dtype)
        # MACE and :func:`build_batch_graphs` use angstroms; our pipeline is nm.
        positions = positions * 10.0

        batch_size = positions.shape[0]
        mace_input = self._build_mace_input(positions, self.atomic_numbers)

        out = self.mace(mace_input, training=False, compute_force=False)

        if isinstance(out, dict):
            node_feats = out.get("node_feats")
            if node_feats is None:
                node_feats = out.get("node_energy")
        else:
            node_feats = out

        if node_feats is None or not isinstance(node_feats, torch.Tensor):
            raise RuntimeError(
                "Could not extract node features from MACE output. "
                f"Output keys: {list(out.keys()) if isinstance(out, dict) else type(out)}. "
                "Check MACE model version compatibility."
            )

        pooled = scatter_mean(node_feats, mace_input["batch"], batch_size)
        # Ensure pooled is on the same device as proj (handles cross-device cases)
        pooled = pooled.to(self.proj.weight.device, dtype=self.proj.weight.dtype)
        return self.proj(pooled)
