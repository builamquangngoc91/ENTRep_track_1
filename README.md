## Overview

Endoscopy image classification using ConvNeXt with phased training, focal loss, and ensemble methods.

## Results Folder Structure

```
paper_results/
├── exp1_baseline_convnext_frozen/
├── exp2_focal_augmentation/
├── exp3_phased_training/
├── exp4_full_method/
│   └── ensemble_models/
├── exp5_convnext_basic_finetuning/
└── summary/
    ├── table1_ablation_study.csv
    ├── table2_architecture_comparison.csv
    └── table3_per_class_performance.csv
```

## Training (test_mode = False)

Run file `EXPERIMENT.ipynb` with config:

```python
config = Config(
    data_root="./dataset",
    test_mode=False,
    baseline_epochs=50,
    phase1_epochs=15,
    phase2_epochs=40, 
    phase3_epochs=30,
    n_folds=5,
    tta_augmentations=5
)

set_seed(config.seed)
runner = PaperExperimentRunner(config)
runner.run_all_experiments()
```

## Inference (test_mode = True)

Run file `EXPERIMENT.ipynb` with config:

```python
config = Config(
    data_root="./dataset",
    test_mode=True,
    baseline_epochs=50,
    phase1_epochs=15,
    phase2_epochs=40, 
    phase3_epochs=30,
    n_folds=5,
    tta_augmentations=5
)

set_seed(config.seed)
runner = PaperExperimentRunner(config)
runner.run_all_experiments()
```