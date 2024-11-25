## Reproduce qualitative results

### 1. Install dependencies

```
pip install -r requirements.txt
```

### 2. Run Python script

```python
from memnet import experiments, plotting
from memnet.experiments import Experiment

for experiment in [Experiment.FMNIST_FGSM, Experiment.MNIST_FGSM, Experiment.FMNIST_PGD]:
    experiments.run(experiment)
    plotting.effect_of_nonidealities(experiment)
    plotting.aware_training(experiment)
    plotting.defender_assumptions(experiment)
```
