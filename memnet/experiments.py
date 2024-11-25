from copy import deepcopy
from enum import Enum

import torch

from . import attacks, data, iterators, layers, networks, nonidealities

class Experiment(Enum):
    FMNIST_FGSM = "FMNIST_FGSM"
    MNIST_FGSM = "MNIST_FGSM"
    FMNIST_PGD = "FMNIST_PGD"

def get_dataset_name(experiment: Experiment) -> data.DatasetName:
    if experiment == Experiment.FMNIST_FGSM or experiment == Experiment.FMNIST_PGD:
        return data.DatasetName.FASHION_MNIST

    return data.DatasetName.MNIST
    
def get_epsilons(experiment: Experiment) -> list[float]:
    if experiment == Experiment.FMNIST_FGSM or experiment == Experiment.MNIST_FGSM:
        return [0.01 * i for i in range(21)]

    return [0.0, 0.2, 0.5, 0.7, 1.0, 2.0, 5.0]
    
def get_attack_instance(experiment: Experiment, eps: float) -> attacks.Attack:
    if experiment == Experiment.FMNIST_FGSM or experiment == Experiment.MNIST_FGSM or eps == 0.0:
        return attacks.FGSM(attacks.AttackType.UNTARGETED, eps)

    return attacks.PGD_L2(attacks.AttackType.UNTARGETED, eps, 0.2, 10)

def get_attack_name(experiment: Experiment) -> str:
    if experiment == Experiment.FMNIST_FGSM or experiment == Experiment.MNIST_FGSM:
        return "FGSM"
    
    return "PGD"

def get_xticks(experiment: Experiment) -> list[float]:
    if experiment == Experiment.FMNIST_FGSM or experiment == Experiment.MNIST_FGSM:
        return [0.0, 0.05, 0.10, 0.15, 0.20]

    return [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]


def run(experiment: Experiment):
    CRITERION = torch.nn.CrossEntropyLoss()
    NUM_EPOCHS = 10
    G_OFF, G_ON = 1e-4, 1e-3
    K_V = 0.5
    NUM_HIDDEN = 32
    epsilons = get_epsilons(experiment)
    dataset_name= get_dataset_name(experiment)

    dataset = data.Dataset(dataset_name)

    memristive_params_ideal = layers.MemristiveParams(G_OFF, G_ON, K_V, [])
    ideal_model = networks.MemristiveNet(
        dataset,
        [NUM_HIDDEN],
        memristive_params_ideal,
    )
    iterators.train(ideal_model, dataset, CRITERION, NUM_EPOCHS)

    memristive_params_stuck_low = layers.MemristiveParams(
        G_OFF, G_ON, K_V, [nonidealities.StuckAtGOff(G_OFF, 0.10)]
    )

    stuck_low_model = networks.MemristiveNet(
        dataset,
        [NUM_HIDDEN],
        memristive_params_stuck_low,
    )
    stuck_low_model.load_state_dict(
        torch.load(
            ideal_model.model_path(),
            map_location=torch.device("cpu"),
        )
    )

    aware_stuck_low_model = networks.MemristiveNet(
        dataset,
        [NUM_HIDDEN],
        memristive_params_stuck_low,
    )
    iterators.train(aware_stuck_low_model, dataset, CRITERION, NUM_EPOCHS)

    memristive_params_stuck_high = layers.MemristiveParams(
        G_OFF, G_ON, K_V, [nonidealities.StuckAtGOff(G_OFF, 0.20)]
    )

    stuck_high_model = networks.MemristiveNet(
        dataset,
        [NUM_HIDDEN],
        memristive_params_stuck_high,
    )
    stuck_high_model.load_state_dict(
        torch.load(
            ideal_model.model_path(),
            map_location=torch.device("cpu"),
        )
    )

    aware_stuck_high_model = networks.MemristiveNet(
        dataset,
        [NUM_HIDDEN],
        memristive_params_stuck_high,
    )
    iterators.train(aware_stuck_high_model, dataset, CRITERION, NUM_EPOCHS)

    aware_stuck_low_model_exposed_to_high = networks.MemristiveNet(
        dataset,
        [NUM_HIDDEN],
        memristive_params_stuck_high,
    )
    aware_stuck_low_model_exposed_to_high.load_state_dict(
        torch.load(
            aware_stuck_low_model.model_path(),
            map_location=torch.device("cpu"),
        )
    )

    aware_stuck_high_model_exposed_to_low = networks.MemristiveNet(
        dataset,
        [NUM_HIDDEN],
        memristive_params_stuck_low,
    )
    aware_stuck_high_model_exposed_to_low.load_state_dict(
        torch.load(
            aware_stuck_high_model.model_path(),
            map_location=torch.device("cpu"),
        )
    )

    for eps in epsilons:
        print(f"\teps: {eps}")
        attack_instance = get_attack_instance(experiment, eps)

        label = f"ideal-attack-sees-ideal-{attack_instance.label()}"
        ideal_attack_dataset = make_attack_test_dataset(
            dataset, attack_instance, ideal_model
        )
        iterators.evaluate(ideal_model, ideal_attack_dataset, CRITERION, label)
        ideal_results = iterators.inference_results(ideal_model, label)
        print(f"Ideal (attacker sees ideal): {int(100*ideal_results.test_accuracy)}%")

        label = f"nonideal-stuck-low-attack-sees-ideal-{attack_instance.label()}"
        nonideal_attack_dataset = make_attack_test_dataset(
            dataset, attack_instance, ideal_model
        )
        iterators.evaluate(
            stuck_low_model,
            nonideal_attack_dataset,
            CRITERION,
            label,
            train_model=ideal_model,
        )
        nonideal_results = iterators.inference_results(
            stuck_low_model, label, ideal_model
        )
        print(
            f"Nonideal stuck low (attacker sees ideal): {int(100*nonideal_results.test_accuracy)}%"
        )

        label = f"nonideal-stuck-low-attack-sees-nonideal-{attack_instance.label()}"
        nonideal_attack_dataset = make_attack_test_dataset(
            dataset, attack_instance, stuck_low_model
        )
        iterators.evaluate(
            stuck_low_model,
            nonideal_attack_dataset,
            CRITERION,
            label,
            train_model=ideal_model,
        )
        nonideal_results = iterators.inference_results(
            stuck_low_model, label, ideal_model
        )
        print(
            f"Nonideal stuck low (attacker sees nonideal): {int(100*nonideal_results.test_accuracy)}%"
        )

        label = f"aware-stuck-low-attack-sees-ideal-{attack_instance.label()}"
        aware_attack_dataset = make_attack_test_dataset(
            dataset, attack_instance, ideal_model
        )
        iterators.evaluate(
            aware_stuck_low_model, aware_attack_dataset, CRITERION, label
        )
        aware_results = iterators.inference_results(aware_stuck_low_model, label)
        print(
            f"Aware stuck low (attacker sees ideal): {int(100*aware_results.test_accuracy)}%"
        )

        label = f"aware-stuck-high-attack-sees-nonideal-{attack_instance.label()}"
        aware_attack_dataset = make_attack_test_dataset(
            dataset, attack_instance, stuck_high_model
        )
        iterators.evaluate(
            aware_stuck_high_model, aware_attack_dataset, CRITERION, label
        )
        aware_results = iterators.inference_results(aware_stuck_high_model, label)
        print(
            f"Aware stuck high (attacker sees nonideal): {int(100*aware_results.test_accuracy)}%"
        )

        label = f"aware-stuck-high-attack-sees-aware-{attack_instance.label()}"
        aware_attack_dataset = make_attack_test_dataset(
            dataset, attack_instance, aware_stuck_high_model
        )
        iterators.evaluate(
            aware_stuck_high_model, aware_attack_dataset, CRITERION, label
        )
        aware_results = iterators.inference_results(aware_stuck_high_model, label)
        print(
            f"Aware stuck high (attacker sees aware): {int(100*aware_results.test_accuracy)}%"
        )

        label = f"nonideal-stuck-high-attack-sees-ideal-{attack_instance.label()}"
        nonideal_attack_dataset = make_attack_test_dataset(
            dataset, attack_instance, ideal_model
        )
        iterators.evaluate(
            stuck_high_model,
            nonideal_attack_dataset,
            CRITERION,
            label,
            train_model=ideal_model,
        )
        nonideal_results = iterators.inference_results(
            stuck_high_model, label, ideal_model
        )
        print(
            f"Nonideal stuck high (attacker sees ideal): {int(100*nonideal_results.test_accuracy)}%"
        )

        label = f"aware-stuck-high-attack-sees-ideal-{attack_instance.label()}"
        aware_attack_dataset = make_attack_test_dataset(
            dataset, attack_instance, ideal_model
        )
        iterators.evaluate(
            aware_stuck_high_model, aware_attack_dataset, CRITERION, label
        )
        aware_results = iterators.inference_results(aware_stuck_high_model, label)
        print(
            f"Aware stuck high (attacker sees ideal): {int(100*aware_results.test_accuracy)}%"
        )

        label = f"aware-stuck-high-exposed-to-low-attack-sees-ideal-{attack_instance.label()}"
        aware_attack_dataset = make_attack_test_dataset(
            dataset, attack_instance, ideal_model
        )
        iterators.evaluate(
            aware_stuck_high_model_exposed_to_low,
            aware_attack_dataset,
            CRITERION,
            label,
            train_model=aware_stuck_low_model,
        )
        aware_results = iterators.inference_results(
            aware_stuck_high_model_exposed_to_low, label, aware_stuck_low_model
        )
        print(
            f"Aware stuck high exposed to low (attacker sees ideal): {int(100*aware_results.test_accuracy)}%"
        )

        label = f"aware-stuck-low-exposed-to-high-attack-sees-ideal-{attack_instance.label()}"
        aware_attack_dataset = make_attack_test_dataset(
            dataset, attack_instance, ideal_model
        )
        iterators.evaluate(
            aware_stuck_low_model_exposed_to_high,
            aware_attack_dataset,
            CRITERION,
            label,
            train_model=aware_stuck_high_model,
        )
        aware_results = iterators.inference_results(
            aware_stuck_low_model_exposed_to_high, label, aware_stuck_high_model
        )
        print(
            f"Aware stuck low exposed to high (attacker sees ideal): {int(100*aware_results.test_accuracy)}%"
        )


def make_attack_test_dataset(
    dataset: data.Dataset,
    attack: attacks.Attack,
    model: networks.MemristiveNet,
) -> data.Dataset:
    attack_dataset = deepcopy(dataset)
    attack_loader = attack.apply(
        model,
        attack_dataset,
        data.Subset.TEST,
    )
    attack_dataset.set_loader(data.Subset.TEST, attack_loader)
    return attack_dataset