"""Registry for loading pre-trained MLIP models as committor backbones.

Usage::

    from committorch.models.pretrained import load_pretrained

    model = load_pretrained("distance_encoder", n_atoms=10, freeze_encoder=True)

For actual MLIP models (MACE, SchNet, PaiNN), the corresponding
packages must be installed.
"""

from __future__ import annotations

from typing import Any

from .base import CommittorModel
from .equivariant.backbone import DistanceEncoder, PretrainedBackbone
from .equivariant.mace import MACECommittor
from .equivariant.schnet import SchNetCommittor
from .equivariant.painn import PaiNNCommittor


_REGISTRY: dict[str, type] = {
    "mace": MACECommittor,
    "schnet": SchNetCommittor,
    "painn": PaiNNCommittor,
}


def load_pretrained(
    name: str,
    freeze_encoder: bool = True,
    sigmoid_steepness: float = 1.0,
    head_hidden_dims: tuple[int, ...] = (32, 32),
    **kwargs: Any,
) -> CommittorModel:
    """Load a pre-trained MLIP backbone and attach a committor head.

    Parameters
    ----------
    name : str
        Model name.  Built-in options:
        - "distance_encoder": lightweight pairwise-distance MLP (no deps).
        - "mace": MACE backbone (requires mace-torch).
        - "schnet": SchNet backbone (requires schnetpack).
        - "painn": PaiNN backbone (requires schnetpack).
    freeze_encoder : bool
        Whether to freeze the encoder parameters.
    sigmoid_steepness : float
        Steepness of the output sigmoid.
    head_hidden_dims : tuple
        Hidden dims for the committor head MLP.
    **kwargs
        Additional keyword arguments passed to the model constructor.
        For "distance_encoder": n_atoms, hidden_dim, output_dim.
        For MLIP models: the pre-trained model object or checkpoint path.

    Returns
    -------
    CommittorModel
        Ready-to-train committor model.
    """
    if name == "distance_encoder":
        n_atoms = kwargs.get("n_atoms", 10)
        hidden_dim = kwargs.get("hidden_dim", 64)
        output_dim = kwargs.get("output_dim", 32)
        encoder = DistanceEncoder(n_atoms, hidden_dim, output_dim)
        return PretrainedBackbone(
            encoder=encoder,
            feature_dim=output_dim,
            head_hidden_dims=head_hidden_dims,
            sigmoid_steepness=sigmoid_steepness,
            freeze_encoder=freeze_encoder,
        )

    if name in _REGISTRY:
        cls = _REGISTRY[name]
        return cls(
            freeze_encoder=freeze_encoder,
            sigmoid_steepness=sigmoid_steepness,
            head_hidden_dims=head_hidden_dims,
            **kwargs,
        )

    raise ValueError(
        f"Unknown model name '{name}'. Available: "
        f"{list(_REGISTRY.keys()) + ['distance_encoder']}"
    )
