# partial-deterministic-normalizing-flows

## Requirements

- Python >= 3.9
- PyTorch >= 2.0.1
- normflows >= 1.7.2
- GPUtil >= 1.4

## Preparation

1. Create experiment environment using e.g. conda as follows
```bash
conda create -n newExps python=3.9
conda activate newExps
```

2. Install necessary packages
```bash
pip install GPUtil
pip install normflows
```

3. Create folders for logging and models
```bash
mkdir all_results; mkdir all_trained_models
```

## Usage (Basic Example Workflow)

-------------------------------------------
1. Prepare Data
-------------------------------------------
Create artificial data for (logistic) regression models
```bash
python syntheticData.py
```

Synthetic datasets are saved into folder "synthetic_data/."
(real datasets should be saved into folder "data/." for preparing the colon data set use "prepare_colon_data.py")


# Part of the code here is adapated from normflows package
# https://github.com/VincentStimper/normalizing-flows


