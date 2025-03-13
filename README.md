# Tutorial for High-dimensional Variational Inference

This tutorial explains the basics of variational inference (VI) from mean-field VI to normalizing flows, and 
then shows how to use/implement VI with PyTorch. The goal of this tutorial is to provide a good starting point for graduate students/researchers interested in applying VI to their Bayesian models, and also to acquire skills in analysis/debugging of VI. 

[Slides for this tutorial](https://drive.google.com/file/d/1ahZAMMzsyEZejL-f3gORfgpA9tmC7uZH/view?usp=drive_link)

The code introduced here is based on the models and methods discussed in [Stabilizing Training of Affine Coupling Layers for High-dimensional Variational Inference](https://doi.org/10.1088/2632-2153/ad9a39), Machine Learning: Science and Technology, 2024.
Most of the code should be scalabe to up to around 10000 dimensions.
However, for high-dimensions (d > 500) recommend usage of GPU.

Part of the code here is adapated from the normflows package
https://github.com/VincentStimper/normalizing-flows


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

2. Install necessary packages (like Pytorch) and the following
```bash
pip install -U numpy scikit-learn GPUtil normflows tqdm matplotlib pandas
```

3. Create folders for logging and models
```bash
mkdir all_results && mkdir all_trained_models
```

## Usage 

See [Slides for this tutorial](https://drive.google.com/file/d/1ahZAMMzsyEZejL-f3gORfgpA9tmC7uZH/view?usp=drive_link)
and *simple_example.py* for various examples.


## Citation 

If you are using part of the code in your work please cite the following papers:

- Andrade, Daniel. "Stabilizing training of affine coupling layers for high-dimensional variational inference." Machine Learning: Science and Technology 5.4 (2024): 045066, https://doi.org/10.1088/2632-2153/ad9a39

- Stimper et al., (2023). normflows: A PyTorch Package for Normalizing Flows. Journal of Open Source Software, 8(86), 5361, https://doi.org/10.21105/joss.05361
