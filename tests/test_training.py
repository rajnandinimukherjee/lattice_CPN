from types import SimpleNamespace
import unittest

import torch
import torch.nn as nn

from training import Trainer


class ConstantNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.value = nn.Parameter(torch.zeros(1))

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        del grid
        return self.value


class ConstantVarianceLoss(nn.Module):
    selection_metric = "var"
    target_exp = 0.5
    exp_err = 0.0
    initial_var = 0.25

    def __init__(self, model: SimpleNamespace) -> None:
        super().__init__()
        self.model = model
        self.obs = SimpleNamespace(grid=torch.ones(1))

    def get_obs(
        self,
        alphas: torch.Tensor,
        sampletype: str,
        batch_idx: torch.Tensor | slice = slice(None),
    ) -> torch.Tensor:
        if sampletype == "train":
            size = len(self.model.samples["train"][batch_idx])
        else:
            size = len(self.model.samples[sampletype])
        values = torch.arange(size, dtype=torch.float32) % 2
        return values + 0.0 * alphas.sum()

    def forward(self, alphas: torch.Tensor, **kwargs: object) -> torch.Tensor:
        del kwargs
        return 1.0 + 0.0 * alphas.sum()


def make_trainer(epochs: int = 100) -> tuple[Trainer, ConstantNet]:
    model = SimpleNamespace(
        samples={
            "train": torch.zeros(8, 1),
            "test": torch.zeros(4, 1),
        }
    )
    loss = ConstantVarianceLoss(model)
    network = ConstantNet()
    optimizer = torch.optim.SGD(network.parameters(), lr=0.1)
    trainer = Trainer(model, loss, optimizer, epochs=epochs, batch_size=2)
    trainer.warmup_epochs = 0
    return trainer, network


class TrainerControlTests(unittest.TestCase):
    def test_plateau_monitoring_uses_disjoint_blocks_and_stops_early(
        self,
    ) -> None:
        trainer, network = make_trainer()

        trainer.train(
            network,
            plateau_window=2,
            plateau_patience=3,
            plateau_min_delta=0.0,
            early_stopping=True,
            lr_patience=1,
            lr_cooldown=0,
            min_lr=1e-4,
        )

        self.assertEqual(trainer.stop_reason, "early_stopping")
        self.assertEqual(trainer.monitor_epochs, [2, 4, 6, 8])
        self.assertEqual(trainer.lr_change_epochs, [6])
        self.assertEqual(trainer.epochs, 9)
        self.assertEqual(len(trainer.lr_history), trainer.epochs)

    def test_best_alphas_searches_full_smoothed_validation_history(
        self,
    ) -> None:
        trainer, _ = make_trainer(epochs=6)
        trainer.alpha_record = [
            torch.tensor(float(i)) for i in range(6)
        ]
        trainer.test_var = [
            float("nan"), 5.0, 4.0, 3.0, 10.0, 10.0
        ]
        trainer.train_var = [6.0, 5.0, 4.0, 3.0, 2.0, 1.0]

        best = trainer.best_alphas(window=2)

        self.assertEqual(best.item(), 3.0)
        self.assertEqual(trainer.best_epoch, 3)


if __name__ == "__main__":
    unittest.main()
