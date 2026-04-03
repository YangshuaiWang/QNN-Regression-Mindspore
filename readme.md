# QNN Regression Benchmark

This repository contains a benchmark setup for hybrid quantum neural network (QNN) regression on three UCI datasets:

- Yacht Hydrodynamics
- Energy Efficiency
- Concrete Compressive Strength

## MindSpore Requirement

This project keeps the QNN implementation in the MindSpore stack:

- model definition in `mindspore.nn.Cell`
- optimization with `mindspore.nn` optimizers
- gradients with `mindspore.value_and_grad`
- tensors with `mindspore.Tensor`
- quantum layer integration through MindQuantum `MQLayer`

The main training script is `benchmark_final_strategy.py`, which is the MindSpore-based QNN pipeline.

## Repository Layout

- `benchmark_final_strategy.py`: trains the Hybrid QNN model and reports RMSE.
- `classical_baseline.py`: runs classical baselines (MLP, SVR, RandomForest) and compares with Hybrid QNN if available.
- `data/`: raw dataset files.

## Requirements

- Python 3.9+
- `mindspore`
- `mindquantum`
- `numpy`
- `pandas`
- `scikit-learn`

For Excel files, install:

- `openpyxl` for `.xlsx`
- `xlrd` for `.xls`

## Run

1. Train and evaluate Hybrid QNN:

```bash
python benchmark_final_strategy.py
```

This creates:

- `results/qnn_benchmark_results.json`

2. Run classical baselines and print side-by-side RMSE:

```bash
python classical_baseline.py
```

If `results/qnn_benchmark_results.json` exists, the script also prints the RMSE delta against the MLP baseline.

## Notes

- All reported values are computed directly from model predictions.
- No synthetic benchmark data or manually adjusted plotting values are included.
