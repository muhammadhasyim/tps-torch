r"""Importance-sampling reweighting for biased simulations.

Two reweighting strategies are implemented:

1. **FEP (Free Energy Perturbation)**: Stratified product of adjacent
   window ratios for umbrella sampling (Paper 1, Section 3.1).

2. **Master equation**: Equilibrium cell weights from the master equation
   using crossing rates between Voronoi cells (Paper 1, Section 3.2).

References
----------
.. [1] Hasyim, Batton, Mandadapu. arXiv:2107.13522, Eqs. 10-12, Appendix B.
"""

from __future__ import annotations

import torch


def fep_reweighting(
    configs_per_window: list[torch.Tensor],
    bias_energies_per_window: list[torch.Tensor],
    kT: float,
) -> list[torch.Tensor]:
    r"""Compute FEP reweighting factors via stratified ratios.

    For umbrella sampling with M windows, the ratio between adjacent
    windows alpha and alpha+1 is:

    .. math::
        \frac{z_{\alpha+1}}{z_\alpha} =
            \frac{\langle e^{-\beta W_\alpha} \rangle_{\alpha+1}}
                 {\langle e^{-\beta W_{\alpha+1}} \rangle_\alpha}

    Parameters
    ----------
    configs_per_window : list of Tensor
        configs_per_window[alpha] has shape (n_alpha, dim).
    bias_energies_per_window : list of Tensor
        bias_energies_per_window[alpha] has shape (n_alpha,), the bias
        energy W_alpha(x) for each sample in window alpha.
    kT : float
        Thermal energy.

    Returns
    -------
    list of Tensor
        z_alpha values (partition function ratios), normalized so sum = 1.
    """
    beta = 1.0 / kT
    n_windows = len(configs_per_window)
    log_z = torch.zeros(n_windows)
    log_z[0] = 0.0

    for alpha in range(n_windows - 1):
        w_alpha = bias_energies_per_window[alpha]
        w_alpha_next = bias_energies_per_window[alpha + 1]

        log_num = torch.logsumexp(-beta * w_alpha, dim=0) - torch.log(
            torch.tensor(float(len(w_alpha)))
        )
        log_den = torch.logsumexp(-beta * w_alpha_next, dim=0) - torch.log(
            torch.tensor(float(len(w_alpha_next)))
        )
        log_z[alpha + 1] = log_z[alpha] + log_num - log_den

    z = torch.exp(log_z - log_z.max())
    z = z / z.sum()
    return [z[i] for i in range(n_windows)]


def master_equation_weights(
    crossing_counts: torch.Tensor,
    n_steps_per_cell: torch.Tensor,
) -> torch.Tensor:
    r"""Compute equilibrium cell weights from the master equation.

    The crossing rate from cell alpha to alpha' is estimated as:

    .. math::
        k_{\alpha\alpha'} = N_{\alpha\alpha'} / N_\mathrm{steps}^\alpha

    The equilibrium distribution satisfies:

    .. math::
        \sum_{\alpha'} k_{\alpha'\alpha} z_{\alpha'} = 0 \quad (\text{with } \sum z_\alpha = 1)

    Parameters
    ----------
    crossing_counts : torch.Tensor, shape (n_cells, n_cells)
        crossing_counts[alpha, alpha'] = number of crossings from cell
        alpha to cell alpha'.
    n_steps_per_cell : torch.Tensor, shape (n_cells,)
        Total number of simulation steps in each cell.

    Returns
    -------
    torch.Tensor, shape (n_cells,)
        Equilibrium weights z_alpha, normalized to sum to 1.
    """
    n = crossing_counts.shape[0]
    rates = crossing_counts.float() / (n_steps_per_cell.float().unsqueeze(1) + 1e-30)

    K = rates.clone().double()
    K.fill_diagonal_(0.0)
    K -= torch.diag(K.sum(dim=1))

    K_T = K.T.clone()
    K_T[-1, :] = 1.0
    b = torch.zeros(n, dtype=torch.float64)
    b[-1] = 1.0

    z = torch.linalg.solve(K_T, b)
    z = z.clamp(min=0.0)
    z = z / z.sum()
    return z
