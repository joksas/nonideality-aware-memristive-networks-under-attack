## Reproduce qualitative results

```python
from memnet import data, experiments, plotting

for dataset_name in [data.DatasetName.FASHION_MNIST, data.DatasetName.MNIST]:
    experiments.aware(dataset_name)
    plotting.effect_of_nonidealities(dataset_name)
    plotting.aware_training(dataset_name)
    plotting.defender_assumptions(dataset_name)
```