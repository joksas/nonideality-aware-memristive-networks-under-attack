import logging
import os
from typing import Optional

import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from . import networks
from .data import Dataset
from .layers import MemristiveParams


def pretty_print_list(lst: list[float]) -> str:
    return ", ".join([f"{x:.3f}" for x in lst])


class TrainingResults:
    def __init__(
        self,
        training_batches: list[float],
        training_losses: list[float],
        validation_epochs: list[float],
        validation_losses: list[float],
        validation_accuracies: list[float],
    ):
        self.training_batches = training_batches
        self.training_losses = training_losses  # shape: (num_batches)

        self.validation_epochs = validation_epochs
        self.validation_losses = validation_losses  # shape: (num_epochs)
        self.validation_accuracies = validation_accuracies  # shape: (num_epochs)

    def __str__(self):
        return f"Training batches: {pretty_print_list(self.training_batches)} \n\
Training losses: {pretty_print_list(self.training_losses)} \n\
Validation epochs: {pretty_print_list(self.validation_epochs)} \n\
Validation losses: {pretty_print_list(self.validation_losses)} \n\
Validation accuracies: {pretty_print_list(self.validation_accuracies)}"


class InferenceResults:
    def __init__(
        self,
        test_loss: float,
        test_accuracy: float,
    ):
        self.test_loss = test_loss
        self.test_accuracy = test_accuracy

    def __str__(self):
        return f"Test accuracy: {self.test_accuracy:.3f}"


def training_results(dir_name: str) -> TrainingResults:
    dir_name = os.path.join(dir_name)

    event_acc = EventAccumulator(dir_name)
    event_acc.Reload()

    batchloss_params = event_acc.Scalars("training/batchloss")
    training_batches, training_losses = map(
        list, zip(*map(lambda x: (x.step, x.value), batchloss_params))
    )

    validation_loss_params = event_acc.Scalars("validation/loss")
    validation_epochs, validation_losses = map(
        list, zip(*map(lambda x: (x.step, x.value), validation_loss_params))
    )

    validation_accuracy_params = event_acc.Scalars("validation/accuracy")
    validation_accuracies = list(map(lambda x: x.value, validation_accuracy_params))

    return TrainingResults(
        training_batches,
        training_losses,
        validation_epochs,
        validation_losses,
        validation_accuracies,
    )


def inference_results(
    model: networks.MemristiveNet,
    dir_name: str,
    train_model: Optional[networks.MemristiveNet] = None,
) -> InferenceResults:
    if train_model is None:
        dir_name = os.path.join(model.model_dir(), dir_name)
    else:
        dir_name = os.path.join(train_model.model_dir(), dir_name)

    event_acc = EventAccumulator(dir_name)
    event_acc.Reload()

    test_loss_params = event_acc.Scalars("test/loss")
    test_loss = test_loss_params[0].value

    test_accuracy_params = event_acc.Scalars("test/accuracy")
    test_accuracy = test_accuracy_params[0].value

    return InferenceResults(
        test_loss,
        test_accuracy,
    )


def train(
    model: networks.MemristiveNet,
    dataset: Dataset,
    criterion: torch.nn.Module,
    num_epochs: int,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print(model.model_dir())

    if os.path.exists(model.model_dir()):
        logging.warning(
            f"Directory {model.model_dir()} already exists, loading weights from file"
        )
        model.load_state_dict(
            torch.load(f"{model.model_path()}", map_location=torch.device("cpu"))
        )
        return

    logging.info(f"Training {model.model_dir()}")

    logger = networks.logger(model.model_dir())

    optimizer = torch.optim.RMSprop(model.parameters(), lr=5e-3)
    trainer = networks.trainer(
        model,
        dataset.val_loader,
        optimizer,
        criterion,
        model.model_dir(),
        logger,
        dataset.units()[1],
    )
    trainer.run(
        dataset.train_loader,
        num_epochs,
    )

    logger.close()

    model.train()


def evaluate(
    model: networks.MemristiveNet,
    dataset: Dataset,
    criterion: torch.nn.Module,
    dir_name: str,
    train_model: Optional[networks.MemristiveNet] = None,
):
    if train_model is None:
        dir_name = f"{model.model_dir()}/{dir_name}"
    else:
        dir_name = f"{train_model.model_dir()}/{dir_name}"

    if os.path.exists(f"{dir_name}"):
        logging.warning(f"Directory {dir_name} already exists, skipping")
        return

    logging.info(f"Evaluating {dir_name}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    logger = networks.logger(dir_name)

    evaluator = networks.evaluator(
        model,
        logger,
        criterion,
        10,
    )
    evaluator.run(dataset.test_loader)

    logger.close()


class Stage:
    def __init__(
        self,
        num_repeats: int,
        dataset: Dataset,
        memristive_params: Optional[MemristiveParams],
    ):
        self.num_repeats = num_repeats
        self.memristive_params = memristive_params
        self.dataset = dataset
        self.current_repeat: int = 0

    def label(self, label: Optional[str] = None):
        if self.memristive_params is None:
            label_str = "ideal"
        else:
            label_str = self.memristive_params.label()

        if label is not None:
            label_str = f"{label_str}__{label}"

        return f"{label_str}/{self.current_repeat+1}"


class TrainingStage(Stage):
    def __init__(
        self,
        num_repeats: int,
        num_epochs: int,
        criterion: torch.nn.Module,
        dataset: Dataset,
        memristive_params: Optional[MemristiveParams],
    ):
        super().__init__(num_repeats, dataset, memristive_params)
        self.num_epochs = num_epochs
        self.criterion = criterion


class TrainingResultsOld:
    def __init__(
        self,
        training_batches: torch.Tensor,
        training_losses: torch.Tensor,
        validation_epochs: torch.Tensor,
        validation_losses: torch.Tensor,
        validation_accuracies: torch.Tensor,
    ):
        self.training_batches = training_batches
        self.training_losses = training_losses  # shape: (num_repeats, num_batches)

        self.validation_epochs = validation_epochs
        self.validation_losses = validation_losses  # shape: (num_repeats, num_epochs)
        self.validation_accuracies = (
            validation_accuracies  # shape: (num_repeats, num_epochs)
        )


class InferenceResultsOld:
    def __init__(
        self,
        test_losses: torch.Tensor,
        test_accuracies: torch.Tensor,
    ):
        self.test_losses = (
            test_losses  # shape: (num_training_repeats, num_inference_repeats)
        )
        self.test_accuracies = (
            test_accuracies  # shape: (num_training_repeats, num_inference_repeats)
        )


class InferenceStage(Stage):
    def __init__(
        self,
        num_repeats: int,
        dataset: Dataset,
        memristive_params: Optional[MemristiveParams],
        label: Optional[str] = None,
    ):
        super().__init__(num_repeats, dataset, memristive_params)
        self._label = label

    def label(self):
        return super().label(self._label)


class Experiment:
    def __init__(
        self, training_stage: TrainingStage, inference_stage: Optional[InferenceStage]
    ):
        self.training_stage = training_stage
        self.inference_stage = inference_stage

    def train(self):
        self.training_stage.current_repeat = 0
        for _ in range(self.training_stage.num_repeats):
            label = self.training_stage.label()
            dir = self.training_dir()
            if os.path.exists(dir):
                logging.warning(f"Directory {dir} already exists, skipping")
                self.training_stage.current_repeat += 1
                continue

            logging.info(f"Training {label}")

            model = networks.MemristiveNet(
                self.training_stage.dataset,
                [],
                self.training_stage.memristive_params,
            )
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)

            logger = networks.logger(label)

            optimizer = torch.optim.RMSprop(model.parameters(), lr=5e-3)
            trainer = networks.trainer(
                model,
                self.training_stage.dataset.val_loader,
                optimizer,
                self.training_stage.criterion,
                label,
                logger,
                self.training_stage.dataset.units()[1],
            )
            trainer.run(
                self.training_stage.dataset.train_loader,
                self.training_stage.num_epochs,
            )

            self.training_stage.current_repeat += 1

            logger.close()

    def evaluated_model(self, use_inference_stage: bool):
        if not use_inference_stage:
            memristive_params = self.training_stage.memristive_params
        else:
            assert self.inference_stage is not None
            memristive_params = self.inference_stage.memristive_params

        model = networks.MemristiveNet(
            self.training_stage.dataset, [], memristive_params
        )
        model.load_state_dict(
            torch.load(
                self.trained_model_path(),
            )
        )

        return model

    def evaluate(self):
        self.training_stage.current_repeat = 0
        for _ in range(self.training_stage.num_repeats):
            training_label = self.training_stage.label()

            if self.inference_stage is None:
                return

            model = self.evaluated_model(True)
            self.inference_stage.current_repeat = 0
            for _ in range(self.inference_stage.num_repeats):
                dir = self.inference_dir()
                if os.path.exists(dir):
                    logging.warning(f"Directory {dir} already exists, skipping")
                    self.inference_stage.current_repeat += 1
                    continue

                inference_label = self.inference_stage.label()
                label = f"{training_label}/{inference_label}"
                logging.info(f"Evaluating {label}")

                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model.to(device)

                logger = networks.logger(label)

                evaluator = networks.evaluator(
                    model,
                    logger,
                    self.training_stage.criterion,
                    10,
                )
                evaluator.run(self.inference_stage.dataset.test_loader)

                self.inference_stage.current_repeat += 1

                logger.close()

            self.training_stage.current_repeat += 1

    @staticmethod
    def base_dir() -> str:
        return os.getcwd()

    def training_dir(self) -> str:
        return os.path.join(self.training_stage.label())

    def trained_model_path(self) -> str:
        return os.path.join(self.training_dir(), "model.pt")

    def inference_dir(self) -> str:
        if self.inference_stage is None:
            raise ValueError("No inference stage specified")

        return os.path.join(self.training_stage.label(), self.inference_stage.label())

    def training_results(self) -> TrainingResultsOld:
        self.training_stage.current_repeat = 0

        training_batches = []
        training_losses = []

        validation_epochs = []
        validation_losses = []
        validation_accuracies = []

        for _ in range(self.training_stage.num_repeats):
            event_acc = EventAccumulator(self.training_dir())
            event_acc.Reload()

            batchloss_params = torch.tensor(event_acc.Scalars("training/batchloss"))
            if len(training_batches) == 0:
                training_batches = batchloss_params[:, 1].tolist()
            training_losses.append(batchloss_params[:, 2].tolist())

            validation_loss_params = torch.tensor(event_acc.Scalars("validation/loss"))
            if len(validation_epochs) == 0:
                validation_epochs = validation_loss_params[:, 1].tolist()
            validation_losses.append(validation_loss_params[:, 2].tolist())

            validation_accuracy_params = torch.tensor(
                event_acc.Scalars("validation/accuracy")
            )
            validation_accuracies.append(validation_accuracy_params[:, 2].tolist())

            self.training_stage.current_repeat += 1

        return TrainingResultsOld(
            torch.Tensor(training_batches),
            torch.Tensor(training_losses),
            torch.Tensor(validation_epochs),
            torch.Tensor(validation_losses),
            torch.Tensor(validation_accuracies),
        )

    def inference_results(self) -> InferenceResultsOld:
        test_losses = []
        test_accuracies = []

        if self.inference_stage is None:
            raise ValueError("No inference stage specified")

        self.training_stage.current_repeat = 0
        for _ in range(self.training_stage.num_repeats):
            self.inference_stage.current_repeat = 0
            iter_test_losses = []
            iter_test_accuracies = []

            for _ in range(self.inference_stage.num_repeats):
                event_acc = EventAccumulator(self.inference_dir())
                event_acc.Reload()

                test_loss_params = event_acc.Scalars("test/loss")
                iter_test_losses.append(test_loss_params[0].value)

                test_accuracy_params = event_acc.Scalars("test/accuracy")
                iter_test_accuracies.append(test_accuracy_params[0].value)

                self.inference_stage.current_repeat += 1

            self.training_stage.current_repeat += 1
            test_losses.append(iter_test_losses)
            test_accuracies.append(iter_test_accuracies)

        return InferenceResultsOld(
            torch.Tensor(test_losses), torch.Tensor(test_accuracies)
        )
