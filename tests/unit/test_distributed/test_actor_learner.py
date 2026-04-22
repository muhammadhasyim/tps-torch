"""Tests for actor-learner distributed architecture.

Uses the single-process fallback to test the full flow without
launching multiple processes.
"""

import torch
import pytest

from committorch.models.mlp import CommittorMLP
from committorch.simulators.langevin import OverdampedLangevin
from committorch.simulators.potentials import muller_brown
from committorch.distributed.comm import SingleProcessBackend
from committorch.distributed.learner import Learner
from committorch.distributed.worker import Worker, SingleProcessActorLearner


class TestSingleProcessBackend:
    def test_rank_and_world_size(self):
        backend = SingleProcessBackend()
        assert backend.rank == 0
        assert backend.world_size == 1
        assert backend.is_learner

    def test_send_recv(self):
        backend = SingleProcessBackend()
        data = {"configs": torch.randn(5, 2), "weights": torch.ones(5)}
        backend.send_data(data)
        received = backend.recv_data()
        torch.testing.assert_close(received["configs"], data["configs"])
        torch.testing.assert_close(received["weights"], data["weights"])

    def test_queue_order(self):
        backend = SingleProcessBackend()
        backend.send_data({"configs": torch.tensor([[1.0, 2.0]])})
        backend.send_data({"configs": torch.tensor([[3.0, 4.0]])})
        d1 = backend.recv_data()
        d2 = backend.recv_data()
        assert d1["configs"][0, 0].item() == 1.0
        assert d2["configs"][0, 0].item() == 3.0

    def test_empty_recv(self):
        backend = SingleProcessBackend()
        assert backend.recv_data() == {}


class TestLearner:
    def test_train_on_batch(self):
        model = CommittorMLP(input_dim=2, hidden_dims=[16])
        comm = SingleProcessBackend()
        x_a = torch.tensor([[-0.558, 1.442]])
        x_b = torch.tensor([[0.623, 0.028]])
        learner = Learner(model, comm, lr=1e-3, x_a=x_a, x_b=x_b)

        configs = torch.randn(20, 2)
        loss = learner.train_on_batch(configs)
        assert loss > 0
        assert len(learner.loss_history) == 1

    def test_collect_and_train(self):
        model = CommittorMLP(input_dim=2, hidden_dims=[16])
        comm = SingleProcessBackend()
        x_a = torch.tensor([[-0.558, 1.442]])
        x_b = torch.tensor([[0.623, 0.028]])
        learner = Learner(model, comm, lr=1e-3, x_a=x_a, x_b=x_b)

        comm.send_data({"configs": torch.randn(10, 2)})
        loss = learner.collect_and_train(n_workers=1)
        assert loss > 0


class TestWorker:
    def test_collect_round(self):
        model = CommittorMLP(input_dim=2, hidden_dims=[16])
        sim = OverdampedLangevin(
            dim=2,
            potential_fn=muller_brown.energy,
            gradient_fn=muller_brown.gradient,
            kT=15.0,
            dt=1e-4,
        )
        comm = SingleProcessBackend()
        worker = Worker(0, model, sim, comm, n_steps_per_round=10)

        data = worker.collect_round(initial_position=torch.zeros(1, 2))
        assert "configs" in data
        assert data["configs"].shape == (10, 2)
        assert data["worker_id"].item() == 0

    def test_collect_and_send(self):
        model = CommittorMLP(input_dim=2, hidden_dims=[16])
        sim = OverdampedLangevin(
            dim=2,
            potential_fn=muller_brown.energy,
            gradient_fn=muller_brown.gradient,
            kT=15.0,
            dt=1e-4,
        )
        comm = SingleProcessBackend()
        worker = Worker(0, model, sim, comm, n_steps_per_round=10)

        worker.collect_and_send(initial_position=torch.zeros(1, 2))
        data = comm.recv_data()
        assert "configs" in data
        assert data["configs"].shape[0] == 10


class TestSingleProcessActorLearner:
    def test_full_round(self):
        model = CommittorMLP(input_dim=2, hidden_dims=[16])
        comm = SingleProcessBackend()

        x_a = muller_brown.REACTANT_MINIMUM.unsqueeze(0)
        x_b = muller_brown.PRODUCT_MINIMUM.unsqueeze(0)

        learner = Learner(model, comm, lr=1e-3, x_a=x_a, x_b=x_b)

        sim = OverdampedLangevin(
            dim=2,
            potential_fn=muller_brown.energy,
            gradient_fn=muller_brown.gradient,
            kT=15.0,
            dt=1e-4,
        )
        worker = Worker(0, model, sim, comm, n_steps_per_round=20)
        worker._current_position = torch.zeros(1, 2)

        system = SingleProcessActorLearner(learner, [worker])
        losses = system.run(n_rounds=5)
        assert len(losses) == 5
        assert all(l > 0 for l in losses)

    def test_multiple_workers(self):
        model = CommittorMLP(input_dim=2, hidden_dims=[16])
        comm = SingleProcessBackend()

        x_a = muller_brown.REACTANT_MINIMUM.unsqueeze(0)
        x_b = muller_brown.PRODUCT_MINIMUM.unsqueeze(0)

        learner = Learner(model, comm, lr=1e-3, x_a=x_a, x_b=x_b)

        workers = []
        for i in range(3):
            sim = OverdampedLangevin(
                dim=2,
                potential_fn=muller_brown.energy,
                gradient_fn=muller_brown.gradient,
                kT=15.0,
                dt=1e-4,
            )
            w = Worker(i, model, sim, comm, n_steps_per_round=10)
            w._current_position = torch.randn(1, 2)
            workers.append(w)

        system = SingleProcessActorLearner(learner, workers)
        losses = system.run(n_rounds=3)
        assert len(losses) == 3
