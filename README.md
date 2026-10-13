# Uncertainty-Aware Driving-Style Inference

Official implementation of **“Uncertainty-Aware Driving-Style Inference Using Interpretable Continuous Traits”** by Dejin Wang and Seyede Fatemeh Ghoreishi.

## Overview

We propose a teacher–student framework for inferring six continuous driving-style traits—aggressiveness, impulsivity, risk tolerance, rule conformity, prospectiveness, and expertness—from short state-only trajectory windows. A style-conditioned PPO teacher generates labeled trajectories; a probabilistic Transformer student predicts style estimates and uncertainty intervals. These intervals are constrained to the valid style range and propagated to one-step action envelopes.

## Environments and Code

Experiments use **highway-env 1.10.1** with three tasks: `highway-v0`, `roundabout-v0`, and `intersection-v0`. The ego vehicle is controlled by a style-conditioned PPO policy (RLlib); surrounding vehicles use IDM/MOBIL. Real-world evaluation uses **NGSIM US-101** trajectories (07:50–08:05).

```text
highway_env/
  teacher/         # PPO training and labeled-trajectory generation
  student/         # Transformer training, evaluation, robustness, action bounds, NGSIM
roundabout_env/    # Roundabout teacher/student training and testing
intersection_env/  # Intersection teacher/student training and testing
```

## Installation

Tested with Python 3.10. A CUDA GPU is recommended for teacher training (the default RLlib configuration requests one GPU).

```bash
conda create -n style-infer python=3.10 -y
conda activate style-infer
pip install highway-env==1.10.1 gymnasium "ray[rllib]==2.9.3" torch numpy pandas scipy scikit-learn statsmodels matplotlib shapely
```

Run scripts from their respective directories. From the repository root, set the highway module paths:

```bash
export PYTHONPATH="$(pwd)/highway_env/teacher:$(pwd)/highway_env/student:$PYTHONPATH"
```

## Reproducing Results

**1. Train PPO teachers** (from the repository root):

```bash
(cd highway_env/teacher && python main2.py)
(cd roundabout_env && python main2.py)
(cd intersection_env && python main2.py)
```

**2. Generate style-labeled trajectories** using the trained highway teacher:

```bash
(cd highway_env/teacher && python store_for_trans.py)
```

The generator restores a configured PPO checkpoint and writes trajectory–style CSV files. Update its checkpoint and output paths as needed; roundabout/intersection student training also requires the corresponding generated datasets.

**3. Train and evaluate Transformer students:**

```bash
(cd highway_env/student && python transformer_1.py)
(cd highway_env/student && python Transformer_test.py)
(cd roundabout_env && python transformer_1.py)
(cd roundabout_env && python transformer_test1.py)
(cd intersection_env && python transformer_1.py)
(cd intersection_env && python transformer_test1.py)
```

**4. Additional highway evaluations:**

```bash
cd highway_env/student
python coverage_evaluation.py       # Uncertainty coverage
python residual_correlation.py      # Residual correlation
python reachability_analysis.py     # One-step action envelopes
python scenario_style_evolution.py  # Abrupt style change
```

**5. NGSIM US-101 evaluation:** Place `trajectories-0750am-0805am.txt` in `highway_env/student/archive/`, then run from `highway_env/student/`:

```bash
python cluster_style_ngsim.py
python ngsim_heldout_analysis.py
```

**Note:** Some scripts reference checkpoint and CSV filenames in their source code. Ensure the required datasets/checkpoints exist and update these paths before running. Results may depend on the trained checkpoints and experiment settings.

## Scope

The six style traits are operational modeling variables, not psychometrically validated personality measures. NGSIM evaluation uses behavioral proxies, not psychological ground truth. Action envelopes bound one-step policy outputs under style uncertainty; they do not guarantee closed-loop safety.

## Contact

Dejin Wang — wang.dej@northeastern.edu
