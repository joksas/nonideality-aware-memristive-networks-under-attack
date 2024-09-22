from pathlib import Path
from typing import Optional

import torch
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger
from ignite.engine import (
    Engine,
    Events,
    create_supervised_evaluator,
    create_supervised_trainer,
)
from ignite.handlers import ModelCheckpoint, global_step_from_engine
from ignite.metrics import Accuracy, ConfusionMatrix, Loss
from torch.utils.data import DataLoader

from memnet import layers

from .data import Dataset


class MemristiveNet(torch.nn.Module):
    def __init__(
        self,
        dataset: Dataset,
        hidden_layer_sizes: list[int],
        memristive_params: Optional[layers.MemristiveParams],
    ) -> None:
        super().__init__()
        input_size, output_size = dataset.units()
        self.flatten = torch.nn.Flatten()
        sequential_layers = []
        if len(hidden_layer_sizes) == 0:
            sequential_layers.append(
                layers.MemristiveLayer(input_size, output_size, memristive_params)
            )
        else:
            sequential_layers.append(
                layers.MemristiveLayer(
                    input_size, hidden_layer_sizes[0], memristive_params
                )
            )
            for i in range(len(hidden_layer_sizes) - 1):
                sequential_layers.append(torch.nn.ReLU())
                sequential_layers.append(
                    layers.MemristiveLayer(
                        hidden_layer_sizes[i],
                        hidden_layer_sizes[i + 1],
                        memristive_params,
                    )
                )
            sequential_layers.append(torch.nn.ReLU())
            sequential_layers.append(
                layers.MemristiveLayer(
                    hidden_layer_sizes[-1],
                    output_size,
                    memristive_params,
                )
            )
        self.stack = torch.nn.Sequential(*sequential_layers)

        self.dataset_name = dataset.name
        self.memristive_params = memristive_params
        self.hidden_layer_sizes = hidden_layer_sizes

    def forward(self, x):
        x = self.flatten(x)
        logits = self.stack(x)
        return logits

    def label(self) -> str:
        label = self.dataset_name.name
        label += "_"
        if self.memristive_params is not None:
            label += self.memristive_params.label()

        for size in self.hidden_layer_sizes:
            label += f"_h{size}"

        return label

    def model_dir(self) -> str:
        return Path(f"runs/{self.label()}").as_posix()

    def model_path(self) -> str:
        return Path(f"{self.model_dir()}/model.pt").as_posix()


def logger(label: str | Path) -> TensorboardLogger:
    return TensorboardLogger(log_dir=label)


def trainer(
    model: torch.nn.Module,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    label: str | Path,
    logger: TensorboardLogger,
    num_classes: int,
) -> Engine:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainer = create_supervised_trainer(model, optimizer, criterion, device=device)

    val_metrics = {
        "accuracy": Accuracy(),
        "confusion_matrix": ConfusionMatrix(num_classes),
        "loss": Loss(criterion),
    }

    val_evaluator = create_supervised_evaluator(
        model, metrics=val_metrics, device=device
    )

    LOG_INTERVAL = 100

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_val_results(_engine):
        val_evaluator.run(val_loader)

    def score_function(engine):
        val_loss = engine.state.metrics["loss"]
        return -val_loss

    model_checkpoint = ModelCheckpoint(
        label,
        n_saved=1,
        score_function=score_function,
        score_name="-val_loss",
        global_step_transform=global_step_from_engine(trainer),
        filename_pattern="model.pt",
    )

    val_evaluator.add_event_handler(
        Events.COMPLETED, model_checkpoint, {"memnet": model}
    )

    logger.attach_output_handler(
        trainer,
        event_name=Events.ITERATION_COMPLETED(every=LOG_INTERVAL),
        tag="training",
        metric_names="all",
        output_transform=lambda loss: {"batchloss": loss},
    )

    logger.attach_output_handler(
        val_evaluator,
        event_name=Events.EPOCH_COMPLETED,
        tag="validation",
        metric_names="all",
        global_step_transform=global_step_from_engine(trainer),
    )

    return trainer


def evaluator(
    model: torch.nn.Module,
    logger: TensorboardLogger,
    criterion: torch.nn.Module,
    num_classes: int,
) -> Engine:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_metrics = {
        "accuracy": Accuracy(),
        "confusion_matrix": ConfusionMatrix(num_classes),
        "loss": Loss(criterion),
    }

    test_evaluator = create_supervised_evaluator(
        model, metrics=test_metrics, device=device
    )

    logger.attach_output_handler(
        test_evaluator,
        event_name=Events.COMPLETED,
        tag="test",
        metric_names="all",
        global_step_transform=global_step_from_engine(test_evaluator),
    )

    return test_evaluator
