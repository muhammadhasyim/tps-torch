"""Worker process for actor-learner architecture.

Workers run local simulations with the current bias, periodically
ship data to the learner, and pull updated model parameters.
"""

from __future__ import annotations

from typing import Callable

import torch

from ..models.base import CommittorModel
from ..simulators.base import Simulator
from .comm import CommBackend


class Worker:
    """Simulation worker that collects data and ships to learner.

    Parameters
    ----------
    worker_id : int
        Unique worker identifier.
    model : CommittorModel
        Local copy of the committor (parameters synced from learner).
    simulator : Simulator
        Dynamics engine for this worker.
    comm : CommBackend
        Communication backend.
    bias_fn_factory : callable or None
        Function that creates a bias_fn from the current model.
        Called after each parameter update to refresh the bias.
    n_steps_per_round : int
        Simulation steps between data shipments.
    """

    def __init__(
        self,
        worker_id: int,
        model: CommittorModel,
        simulator: Simulator,
        comm: CommBackend,
        bias_fn_factory: Callable[[CommittorModel], Callable] | None = None,
        n_steps_per_round: int = 100,
    ) -> None:
        self.worker_id = worker_id
        self.model = model
        self.simulator = simulator
        self.comm = comm
        self.bias_fn_factory = bias_fn_factory
        self.n_steps_per_round = n_steps_per_round
        self._current_position: torch.Tensor | None = None

    def sync_params(self) -> None:
        """Pull updated parameters from the learner."""
        self.comm.broadcast_params(self.model)
        if self.bias_fn_factory is not None:
            self.simulator.set_bias(self.bias_fn_factory(self.model))

    def collect_round(
        self, initial_position: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        """Run one round of simulation and return collected data.

        Parameters
        ----------
        initial_position : torch.Tensor or None
            Starting position.  If None, uses the last position.

        Returns
        -------
        dict
            {"configs": (n_steps, dim), "worker_id": scalar}
        """
        if initial_position is not None:
            self._current_position = initial_position
        if self._current_position is None:
            raise ValueError("No initial position set for worker")

        configs_list = []
        pos = self._current_position
        for _ in range(self.n_steps_per_round):
            pos = self.simulator.step(pos, n_steps=1)
            configs_list.append(pos.detach().clone())

        self._current_position = pos.detach()
        configs = torch.cat(configs_list, dim=0)

        return {
            "configs": configs,
            "worker_id": torch.tensor([self.worker_id]),
        }

    def collect_and_send(
        self, initial_position: torch.Tensor | None = None
    ) -> None:
        """Collect data and ship to the learner."""
        data = self.collect_round(initial_position)
        self.comm.send_data(data, dest=0)


class SingleProcessActorLearner:
    """Convenience class running both learner and workers in one process.

    Useful for debugging and small-scale experiments.

    Parameters
    ----------
    learner : Learner
    workers : list of Worker
    """

    def __init__(self, learner, workers: list[Worker]) -> None:
        from .learner import Learner

        self.learner = learner
        self.workers = workers

    def run_round(self) -> float:
        """Run one sample-train round.

        1. Each worker collects data and sends to queue.
        2. Learner collects and trains.
        3. Learner broadcasts updated params.

        Returns
        -------
        float
            Loss value from training.
        """
        for worker in self.workers:
            worker.collect_and_send()

        loss = self.learner.collect_and_train(n_workers=len(self.workers))
        self.learner.broadcast_params()

        for worker in self.workers:
            worker.sync_params()

        return loss

    def run(self, n_rounds: int) -> list[float]:
        """Run multiple sample-train rounds.

        Parameters
        ----------
        n_rounds : int
            Number of rounds.

        Returns
        -------
        list of float
            Loss history.
        """
        losses = []
        for _ in range(n_rounds):
            loss = self.run_round()
            losses.append(loss)
        return losses
