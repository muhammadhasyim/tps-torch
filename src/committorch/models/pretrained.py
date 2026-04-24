"""Registry for loading pre-trained MLIP models as committor backbones.

Usage::

    # Foundation model (recommended) -- auto-downloads MACE-OFF23 small
    model = load_pretrained("mace", model_name="mace-off", size="small",
                            atomic_numbers=z)

    # Shorthand
    model = load_pretrained("mace-off-small", atomic_numbers=z)

    # Test-only distance encoder (no external deps)
    model = load_pretrained("distance_encoder", n_atoms=10)
"""

from __future__ import annotations

import warnings
from typing import Any

from .base import CommittorModel
from .equivariant.backbone import DistanceEncoder, PretrainedBackbone
from .equivariant.mace import (
    AVAILABLE_FOUNDATION_MODELS,
    MACECommittor,
    parse_foundation_model_spec,
)


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
        Model identifier.  Options:

        - ``"distance_encoder"``: lightweight all-pairs-distance MLP.
          Test-only; scales as O(N^2) and is not suitable for production.
        - ``"mace"``: MACE backbone.  Pass ``model_name`` and ``size``
          kwargs (or ``checkpoint_path`` for a custom checkpoint).
        - ``"mace-off-small"``, ``"mace-off-medium"``, etc.: shorthand
          that auto-parses the family and size.
        - ``"schnet"``, ``"painn"``: deprecated stubs.

    freeze_encoder : bool
        Whether to freeze the encoder parameters.
    sigmoid_steepness : float
        Steepness of the output sigmoid.
    head_hidden_dims : tuple
        Hidden dims for the committor head MLP.
    **kwargs
        Additional keyword arguments.  For ``"mace"``: ``model_name``,
        ``size``, ``atomic_numbers``, ``cutoff``, ``feature_dim``,
        ``device``.  For ``"distance_encoder"``: ``n_atoms``,
        ``hidden_dim``, ``output_dim``.

    Returns
    -------
    CommittorModel
        Ready-to-train committor model.
    """
    # --- Distance encoder (test-only) ---
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

    # --- Shorthand: "mace-off-small", "mace-mp-medium", etc. ---
    if name in AVAILABLE_FOUNDATION_MODELS:
        model_name, size = parse_foundation_model_spec(name)
        return MACECommittor.from_pretrained(
            model_name=model_name,
            size=size,
            atomic_numbers=kwargs.get("atomic_numbers"),
            feature_dim=kwargs.get("feature_dim", 128),
            head_hidden_dims=head_hidden_dims,
            sigmoid_steepness=sigmoid_steepness,
            freeze_encoder=freeze_encoder,
            device=kwargs.get("device", "cpu"),
        )

    # --- Explicit "mace" with kwargs ---
    if name == "mace":
        model_name_kw = kwargs.pop("model_name", None)
        size_kw = kwargs.pop("size", "small")
        checkpoint_path = kwargs.pop("checkpoint_path", None)
        atomic_numbers = kwargs.pop("atomic_numbers", None)
        cutoff = kwargs.pop("cutoff", 5.0)
        feature_dim = kwargs.pop("feature_dim", 128)
        device = kwargs.pop("device", "cpu")

        if model_name_kw is not None:
            return MACECommittor.from_pretrained(
                model_name=model_name_kw,
                size=size_kw,
                atomic_numbers=atomic_numbers,
                feature_dim=feature_dim,
                head_hidden_dims=head_hidden_dims,
                sigmoid_steepness=sigmoid_steepness,
                freeze_encoder=freeze_encoder,
                device=device,
            )
        if checkpoint_path is not None:
            if atomic_numbers is None:
                raise ValueError(
                    "atomic_numbers is required when loading MACE from a checkpoint"
                )
            return MACECommittor.from_pretrained(
                checkpoint_path=checkpoint_path,
                atomic_numbers=atomic_numbers,
                cutoff=cutoff,
                feature_dim=feature_dim,
                head_hidden_dims=head_hidden_dims,
                sigmoid_steepness=sigmoid_steepness,
                freeze_encoder=freeze_encoder,
                device=device,
            )
        raise ValueError(
            "For name='mace', provide either model_name (e.g. 'mace-off') "
            "or checkpoint_path."
        )

    # --- Deprecated stubs ---
    if name in ("schnet", "painn"):
        warnings.warn(
            f"'{name}' is a deprecated placeholder adapter with no real "
            f"pre-trained model support. Use 'mace' or a 'mace-off-*' "
            f"shorthand instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if name == "schnet":
            from .equivariant.schnet import SchNetCommittor

            return SchNetCommittor(
                freeze_encoder=freeze_encoder,
                sigmoid_steepness=sigmoid_steepness,
                head_hidden_dims=head_hidden_dims,
                **kwargs,
            )
        from .equivariant.painn import PaiNNCommittor

        return PaiNNCommittor(
            freeze_encoder=freeze_encoder,
            sigmoid_steepness=sigmoid_steepness,
            head_hidden_dims=head_hidden_dims,
            **kwargs,
        )

    available = ["distance_encoder", "mace"] + AVAILABLE_FOUNDATION_MODELS
    raise ValueError(
        f"Unknown model name '{name}'. Available: {available}"
    )
