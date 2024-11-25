import logging
import os
from datetime import datetime
from enum import Enum
from typing import Literal

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

from . import data, iterators, layers, networks, nonidealities, experiments

plt.style.use(os.path.join(os.path.dirname(__file__), "style.mplstyle"))

ONE_COLUMN_WIDTH = 8.5 / 2.54
TWO_COLUMNS_WIDTH = 17.8 / 2.54


class Color(Enum):
    """Okabe-Ito colorblind-friendly palette."""

    ORANGE = "#E69F00"
    SKY_BLUE = "#56B4E9"
    BLUISH_GREEN = "#009E73"
    YELLOW = "#F0E442"
    BLUE = "#0072B2"
    VERMILION = "#D55E00"
    REDDISH_PURPLE = "#CC79A7"
    BLACK = "#000000"

    def __get__(self, instance, owner):
        return self.value


def figure(
    num_columns: Literal[1, 2], height_to_width_ratio: float = 1.0
) -> tuple[plt.Figure, plt.Axes]:
    if num_columns == 1:
        width = ONE_COLUMN_WIDTH
    elif num_columns == 2:
        width = TWO_COLUMNS_WIDTH

    fig, axes = plt.subplots(figsize=(width, width * height_to_width_ratio))
    fig.tight_layout()

    return fig, axes


def boxplot(
    ax: plt.Axes, labels: list[str], data_list: list[list[float]], colors: list[Color]
) -> None:
    for idx, (label, data, color) in enumerate(zip(labels, data_list, colors)):
        boxplot = plt.boxplot(data, positions=[idx], sym=color, labels=[label])
        plt.setp(boxplot["fliers"], marker="x", markersize=2, markeredgewidth=0.5)
        for element in ["boxes", "whiskers", "fliers", "means", "medians", "caps"]:
            plt.setp(boxplot[element], color=color, linewidth=0.75)


def color_list() -> list[Color]:
    colors = [
        Color.ORANGE,
        Color.SKY_BLUE,
        Color.BLUISH_GREEN,
        Color.YELLOW,
        Color.BLUE,
        Color.VERMILION,
        Color.REDDISH_PURPLE,
        Color.BLACK,
    ]
    return colors


def save_fig(fig, name: str):
    dir_name = "plots"
    os.makedirs(dir_name, exist_ok=True)
    path = os.path.join(dir_name, f"{name}.pdf")
    if os.path.exists(path):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        new_path = os.path.join(dir_name, f".{name}__{timestamp}.pdf")
        logging.warning(
            f'File "{path}" already exists. Renaming old file to "{new_path}".'
        )
        os.rename(path, new_path)
        save_fig(fig, name)
        return

    fig.savefig(path, bbox_inches="tight", transparent=True)
    logging.info(f'Saved file "{path}".')


def add_boxplot_legend(axis, boxplots, labels, linewdith=1.0, loc="upper right"):
    leg = axis.legend(
        [boxplot["boxes"][0] for boxplot in boxplots],
        labels,
        fontsize=8,
        frameon=False,
        loc=loc,
    )
    for line in leg.get_lines():
        line.set_linewidth(linewdith)


def fmnist_intensities():
    """Generated using ChatGPT (GPT-4)."""

    # Load FashionMNIST dataset
    transform = transforms.ToTensor()
    train_data = datasets.FashionMNIST(
        root="~/.pytorch-datasets", train=True, download=True, transform=transform
    )

    # Create a dictionary to store an example from each class
    class_examples = dict()

    # Loop over the dataset until we have an example from each class
    for image, label in train_data:
        if label not in class_examples:
            class_examples[label] = image

        # Break the loop if we have all classes
        if len(class_examples) == 10:
            break

    # Plot each example along with its histogram
    fig, axs = plt.subplots(
        2, 10, figsize=(TWO_COLUMNS_WIDTH, 0.35 * TWO_COLUMNS_WIDTH), sharey="row"
    )
    fig.tight_layout()

    # Loop over classes and plot
    for i, (label, image) in enumerate(class_examples.items()):
        # Plot image
        axs[0, i].imshow(image.squeeze(), cmap="gray")
        axs[0, i].axis("off")

        # Plot histogram
        axs[1, i].hist(image.flatten(), bins=10, color=Color.BLUE)
        axs[1, i].set_xlim(0, 1)

    fig.suptitle("Fashion MNIST pixel intensities")

    filename = "fmnist-intensities"
    save_fig(fig, filename)


def sample_colormap(cmap_name, num_colors):
    cmap = plt.get_cmap(cmap_name)
    return [
        mcolors.rgb2hex(cmap(i)) for i in range(0, 256, int(256 / (num_colors - 1)))
    ]


def standard_ax_settings(fig, ax, dataset_name, colors, x, experiment: experiments.Experiment):
    attack_name = experiments.get_attack_name(experiment)
    ax.set_xlabel(rf"$\epsilon$ in {attack_name} attack")
    ax.set_ylabel(f"{dataset_name.pretty()} test accuracy (%)")
    ax.set_xlim(min(x), max(x))
    ax.set_ylim(0, 100)
    ax.xaxis.set_ticks(experiments.get_xticks(experiment))
    legend = fig.legend(
        frameon=False, loc="center", bbox_to_anchor=(0.5, 0.9), ncol=len(colors)
    )
    # Increase the linewidth of the legend lines.
    for line in legend.get_lines():
        line.set_linewidth(1.5)


def effect_of_nonidealities(experiment: experiments.Experiment):
    G_OFF, G_ON = 1e-4, 1e-3
    K_V = 0.5
    NUM_HIDDEN = 32
    epsilons = experiments.get_epsilons(experiment)
    dataset_name = experiments.get_dataset_name(experiment)

    dataset = data.Dataset(dataset_name)

    memristive_params_ideal = layers.MemristiveParams(G_OFF, G_ON, K_V, [])
    ideal_model = networks.MemristiveNet(
        dataset,
        [NUM_HIDDEN],
        memristive_params_ideal,
    )
    ideal_accuracies = []

    memristive_params_stuck_low = layers.MemristiveParams(
        G_OFF, G_ON, K_V, [nonidealities.StuckAtGOff(G_OFF, 0.10)]
    )
    stuck_low_model = networks.MemristiveNet(
        dataset,
        [NUM_HIDDEN],
        memristive_params_stuck_low,
    )
    stuck_low_accuracies = []

    memristive_params_stuck_high = layers.MemristiveParams(
        G_OFF, G_ON, K_V, [nonidealities.StuckAtGOff(G_OFF, 0.20)]
    )
    stuck_high_model = networks.MemristiveNet(
        dataset,
        [NUM_HIDDEN],
        memristive_params_stuck_high,
    )
    stuck_high_accuracies = []

    for eps in epsilons:
        attack_instance = experiments.get_attack_instance(experiment, eps)

        label = f"ideal-attack-sees-ideal-{attack_instance.label()}"
        results = iterators.inference_results(ideal_model, label)
        ideal_accuracies.append(100 * results.test_accuracy)

        label = f"nonideal-stuck-low-attack-sees-ideal-{attack_instance.label()}"
        results = iterators.inference_results(stuck_low_model, label, ideal_model)
        stuck_low_accuracies.append(100 * results.test_accuracy)

        label = f"nonideal-stuck-high-attack-sees-ideal-{attack_instance.label()}"
        results = iterators.inference_results(stuck_high_model, label, ideal_model)
        stuck_high_accuracies.append(100 * results.test_accuracy)

    fig, ax = figure(num_columns=2, height_to_width_ratio=9 / 16)
    colors = [Color.BLUISH_GREEN, Color.ORANGE, Color.REDDISH_PURPLE]
    labels = ["Ideal", "10% stuck in OFF", "20% stuck in OFF"]

    for accuracies, color in zip(
        [ideal_accuracies, stuck_low_accuracies, stuck_high_accuracies], colors
    ):
        ax.plot(epsilons, accuracies, color=color, label=labels.pop(0))

    standard_ax_settings(fig, ax, dataset_name, colors, epsilons, experiment)

    filename = f"{experiment.name}-effect-of-nonidealities"
    save_fig(fig, filename)


def aware_training(experiment: experiments.Experiment):
    G_OFF, G_ON = 1e-4, 1e-3
    K_V = 0.5
    NUM_HIDDEN = 32
    epsilons = experiments.get_epsilons(experiment)
    dataset_name = experiments.get_dataset_name(experiment)

    dataset = data.Dataset(dataset_name)

    memristive_params_ideal = layers.MemristiveParams(G_OFF, G_ON, K_V, [])
    ideal_model = networks.MemristiveNet(
        dataset,
        [NUM_HIDDEN],
        memristive_params_ideal,
    )

    memristive_params_stuck_high = layers.MemristiveParams(
        G_OFF, G_ON, K_V, [nonidealities.StuckAtGOff(G_OFF, 0.20)]
    )

    stuck_high_model = networks.MemristiveNet(
        dataset,
        [NUM_HIDDEN],
        memristive_params_stuck_high,
    )
    stuck_high_accuracies = []

    aware_stuck_high_model = networks.MemristiveNet(
        dataset,
        [NUM_HIDDEN],
        memristive_params_stuck_high,
    )
    aware_stuck_high_accuracies = []

    for eps in epsilons:
        attack_instance = experiments.get_attack_instance(experiment, eps)

        label = f"nonideal-stuck-high-attack-sees-ideal-{attack_instance.label()}"
        results = iterators.inference_results(stuck_high_model, label, ideal_model)
        stuck_high_accuracies.append(100 * results.test_accuracy)

        label = f"aware-stuck-high-attack-sees-ideal-{attack_instance.label()}"
        results = iterators.inference_results(aware_stuck_high_model, label)
        aware_stuck_high_accuracies.append(100 * results.test_accuracy)

    fig, ax = figure(num_columns=2, height_to_width_ratio=9 / 16)
    colors = [Color.REDDISH_PURPLE, Color.BLUISH_GREEN]
    labels = ["Ignores 20% stuck devices", "Assumes 20% stuck devices"]

    for accuracies, color in zip(
        [stuck_high_accuracies, aware_stuck_high_accuracies], colors
    ):
        ax.plot(epsilons, accuracies, color=color, label=labels.pop(0))

    standard_ax_settings(fig, ax, dataset_name, colors, epsilons, experiment)
    ax.set_title("Network designer's training assumptions")

    filename = f"{experiment.name}-aware-training"
    save_fig(fig, filename)


def defender_assumptions(experiment: experiments.Experiment):
    G_OFF, G_ON = 1e-4, 1e-3
    K_V = 0.5
    NUM_HIDDEN = 32
    epsilons = experiments.get_epsilons(experiment)
    dataset_name = experiments.get_dataset_name(experiment)
    attack_name = experiments.get_attack_name(experiment)

    dataset = data.Dataset(dataset_name)

    memristive_params_ideal = layers.MemristiveParams(G_OFF, G_ON, K_V, [])
    ideal_model = networks.MemristiveNet(
        dataset,
        [NUM_HIDDEN],
        memristive_params_ideal,
    )

    memristive_params_stuck_low = layers.MemristiveParams(
        G_OFF, G_ON, K_V, [nonidealities.StuckAtGOff(G_OFF, 0.10)]
    )
    memristive_params_stuck_high = layers.MemristiveParams(
        G_OFF, G_ON, K_V, [nonidealities.StuckAtGOff(G_OFF, 0.20)]
    )

    stuck_low_model = networks.MemristiveNet(
        dataset,
        [NUM_HIDDEN],
        memristive_params_stuck_low,
    )

    stuck_high_model = networks.MemristiveNet(
        dataset,
        [NUM_HIDDEN],
        memristive_params_stuck_high,
    )

    aware_stuck_low_model = networks.MemristiveNet(
        dataset,
        [NUM_HIDDEN],
        memristive_params_stuck_low,
    )

    aware_stuck_high_model = networks.MemristiveNet(
        dataset,
        [NUM_HIDDEN],
        memristive_params_stuck_high,
    )

    aware_stuck_low_model_exposed_to_high = networks.MemristiveNet(
        dataset,
        [NUM_HIDDEN],
        memristive_params_stuck_high,
    )

    aware_stuck_high_model_exposed_to_low = networks.MemristiveNet(
        dataset,
        [NUM_HIDDEN],
        memristive_params_stuck_low,
    )

    stuck_low_accuracies = []
    stuck_high_accuracies = []
    aware_stuck_low_accuracies = []
    aware_stuck_high_accuracies = []
    aware_stuck_low_exposed_to_high_accuracies = []
    aware_stuck_high_exposed_to_low_accuracies = []

    for eps in epsilons:
        attack_instance = experiments.get_attack_instance(experiment, eps)


        label = f"nonideal-stuck-low-attack-sees-ideal-{attack_instance.label()}"
        results = iterators.inference_results(stuck_low_model, label, ideal_model)
        stuck_low_accuracies.append(100 * results.test_accuracy)

        label = f"nonideal-stuck-high-attack-sees-ideal-{attack_instance.label()}"
        results = iterators.inference_results(stuck_high_model, label, ideal_model)
        stuck_high_accuracies.append(100 * results.test_accuracy)

        label = f"aware-stuck-low-attack-sees-ideal-{attack_instance.label()}"
        results = iterators.inference_results(aware_stuck_low_model, label)
        aware_stuck_low_accuracies.append(100 * results.test_accuracy)

        label = f"aware-stuck-high-attack-sees-ideal-{attack_instance.label()}"
        results = iterators.inference_results(aware_stuck_high_model, label)
        aware_stuck_high_accuracies.append(100 * results.test_accuracy)

        label = f"aware-stuck-low-exposed-to-high-attack-sees-ideal-{attack_instance.label()}"
        results = iterators.inference_results(
            aware_stuck_low_model_exposed_to_high, label
        )
        aware_stuck_low_exposed_to_high_accuracies.append(100 * results.test_accuracy)

        label = f"aware-stuck-high-exposed-to-low-attack-sees-ideal-{attack_instance.label()}"
        results = iterators.inference_results(
            aware_stuck_high_model_exposed_to_low, label
        )
        aware_stuck_high_exposed_to_low_accuracies.append(100 * results.test_accuracy)

    fig, ax = figure(num_columns=2, height_to_width_ratio=9 / 16)
    colors = [
        Color.ORANGE,
        Color.ORANGE,
        Color.ORANGE,
        Color.REDDISH_PURPLE,
        Color.REDDISH_PURPLE,
        Color.REDDISH_PURPLE,
    ]
    linestyle = ["-", "--", ":", "-", "--", ":"]
    labels = [
        "Assumes ideal, finds 10% stuck",
        "Assumes 10% stuck, finds 10% stuck",
        "Assumes 10% stuck, finds 20% stuck",
        "Assumes ideal, finds 20% stuck",
        "Assumes 20% stuck, finds 20% stuck",
        "Assumes 20% stuck, finds 10% stuck",
    ]

    for accuracies, color, style in zip(
        [
            stuck_low_accuracies,
            aware_stuck_low_accuracies,
            aware_stuck_low_exposed_to_high_accuracies,
            stuck_high_accuracies,
            aware_stuck_high_accuracies,
            aware_stuck_high_exposed_to_low_accuracies,
        ],
        colors,
        linestyle,
    ):
        ax.plot(epsilons, accuracies, color=color, linestyle=style, label=labels.pop(0))

    ax.set_xlabel(rf"$\epsilon$ in {attack_name} attack")
    ax.set_ylabel(f"{dataset_name.pretty()} test accuracy (%)")
    ax.set_xlim(min(epsilons), max(epsilons))
    ax.set_ylim(0, 100)
    ax.xaxis.set_ticks(experiments.get_xticks(experiment))
    legend = fig.legend(frameon=False, loc="center", bbox_to_anchor=(0.75, 0.8), ncol=1)
    # Increase the linewidth of the legend lines.
    for line in legend.get_lines():
        line.set_linewidth(1.5)

    ax.set_title("Network designer's training assumptions vs reality")

    filename = f"{experiment.name}-defender-assumptions"
    save_fig(fig, filename)
