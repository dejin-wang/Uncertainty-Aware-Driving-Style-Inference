# Uncertainty-Aware Driving-Style Inference

Official implementation of the paper:

**“Uncertainty-Aware Driving-Style Inference Using Interpretable Continuous Traits”**

**Authors:** Dejin Wang and Seyede Fatemeh Ghoreishi

## Overview

This work introduces a teacher–student framework for online driving-style inference from short state-only observation windows, with an explicit representation of inference uncertainty.

Driving style is represented by six bounded continuous operational coordinates:

| Coordinate | Symbol | Support |
|---|---|---|
| Aggressiveness | z1 | [0, 0.9] |
| Impulsivity | z2 | [0, 0.9] |
| Risk tolerance | z3 | [0, 0.9] |
| Rule conformity | z4 | [0, 0.9] |
| Prospectiveness | z5 | [0.6, 0.99] |
| Expertness | z6 | [0.75, 1.0] |

A style-conditioned teacher policy generates labeled trajectory–style pairs over the continuous style space. A Transformer-based probabilistic student estimates a factorized posterior over the six coordinates in a single forward pass, producing both style estimates and dimension-wise uncertainty intervals.

A boundary-aware training objective discourages predicted intervals from extending beyond the valid style bounds. At deployment, these intervals are intersected with the valid style support and propagated through the style-conditioned policy to construct conservative one-step action envelopes.

## Simulation Environment

All simulation experiments use **highway-env 1.10.1** (no other simulator is used) and cover three tasks:

- **Highway** (`highway-v0`): Car following and lane-changing interactions.
- **Roundabout** (`roundabout-v0`): Merging and priority negotiation.
- **Intersection** (`intersection-v0`): Crossing interactions and conflict resolution.

Surrounding vehicles use the Intelligent Driver Model (IDM) for longitudinal motion and MOBIL for lane-changing decisions. A learned style-conditioned teacher policy (PPO, RLlib) controls the ego vehicle to generate style-labeled trajectories.

Real-world evaluation uses the **NGSIM US-101** dataset (on-ramp vehicles, 07:50–08:05 segment).

## Repository Structure

```
highway_env/
  teacher/
    problem_env1.py          # Style-conditioned highway environment (wraps highway-v0)
    custom_model.py          # Teacher policy network (RLlib custom model "custom_fcnet")
    register_env.py          # Registers the env with RLlib
    main2.py                 # Train the style-conditioned PPO teacher
    store_for_trans.py       # Roll out the teacher and log (state, style) trajectories to CSV
  student/
    register_env1.py         # Registers the env with RLlib and gymnasium
    transformer_1.py         # Train the probabilistic Transformer student (boundary-aware loss)
    Transformer_test.py      # Evaluate student NLL / MSE on held-out trajectories
    coverage_evaluation.py   # Empirical coverage of the per-dimension credible intervals
    residual_correlation.py  # Residual correlation matrix (factorized-head check)
    gaussian_head_verification.py  # Factorized vs. low-rank Cholesky covariance head
    crown_simplified.py      # CROWN-style linear bound propagation through the policy
    reachability_analysis.py # One-step action envelopes from style uncertainty intervals
    scenario_style_evolution.py    # Style tracking under abrupt (phase-switched) behavior change
    IDM.py, IDM_mobile.py, IDM_four_styles.py   # IDM / IDM+MOBIL rule-based rollouts
    regression_singe_state_prediction.py, regression_trajectory_style.py,
    action_prediction_generated.py, feature_test.py   # Action-prediction baselines / ablations
    cluster_ngsim_500m.py    # NGSIM: cluster on-ramp drivers (0–500 m) with reward-aligned features
    cluster_style_ngsim.py   # NGSIM: re-clustering + Transformer style inference per cluster
    ngsim_heldout_analysis.py      # NGSIM: held-out behavioral-proxy correlations (Spearman, BH)
roundabout_env/
  roundabout1.py, custom_model.py, register_env.py, main2.py   # Roundabout teacher
  transformer_1.py, transformer_test1.py                       # Roundabout student train / test
intersection_env/
  intersection2.py, custom_model.py, register_env.py, main2.py # Intersection teacher
  transformer_1.py, transformer_test1.py                       # Intersection student train / test
```

## Installation

Tested with Python 3.10. A CUDA GPU is recommended for teacher training and required for the default
`num_gpus=1` setting in the RLlib configs.

```bash
conda create -n style-infer python=3.10 -y
conda activate style-infer
pip install highway-env==1.10.1 gymnasium "ray[rllib]==2.9.3" torch numpy pandas scipy scikit-learn statsmodels matplotlib shapely
```

The teacher scripts use the RLlib "old API stack" (`.rollouts(num_rollout_workers=...)`,
`sgd_minibatch_size`), so use a Ray 2.x release that still supports it (e.g. 2.9.x).

### Running scripts

Scripts use plain module imports and relative file paths, so run each one **from inside its folder**.
The highway student scripts import `problem_env1` and `custom_model` from `highway_env/teacher`, and
`store_for_trans.py` imports `register_env1` from `highway_env/student`, so put both highway folders
on `PYTHONPATH`:

```bash
export PYTHONPATH="$(pwd)/highway_env/teacher:$(pwd)/highway_env/student:$PYTHONPATH"
```

(PowerShell: `$env:PYTHONPATH = "$PWD\highway_env\teacher;$PWD\highway_env\student"`.)

## Reproducing the Main Results

The full pipeline is: **(1)** train the style-conditioned teacher → **(2)** generate style-labeled
trajectories → **(3)** train the student → **(4)** evaluate. File names below match the defaults
hard-coded in the scripts; edit the constants at the top of each script if you use different paths.

### 1. Train the style-conditioned teacher (PPO)

```bash
cd highway_env/teacher && python main2.py      # checkpoints -> ppo_model_checkpoint/ppo_model_<iter>
cd roundabout_env      && python main2.py      # checkpoints -> ppo_model_checkpoint1/
cd intersection_env    && python main2.py      # checkpoints -> ppo_model_checkpoint1/
```

Each episode samples a style vector z from the supports above; the discount factor of each trajectory is
set to z5 (prospectiveness) by `DynamicGammaCallbacks`. The default config uses 40 rollout workers and one GPU;
reduce `num_rollout_workers` for smaller machines.

### 2. Generate style-labeled trajectories

```bash
cd highway_env/teacher && python store_for_trans.py
```

This restores `ppo_model_checkpoint/ppo_model_9815` and writes `Episode, Step, Reward, Obs(flat), Style`
rows to CSV. Run it several times (changing the output name) to produce the training files
`PPO_obs_and_rewards{,1,2,3,4,5,10}.csv` and a separate held-out file `PPO_obs_and_rewards_test.csv`.
Roundabout and intersection data are produced the same way with the corresponding environment and checkpoint
(`PPO_obs_and_rewards250K.csv` for training; `PPO_obs_and_rewards500.csv` / `PPO_obs_and_rewards200.csv` for testing).

### 3. Train the probabilistic Transformer student

```bash
cd highway_env/student && python transformer_1.py   # -> transformer_weights_gaussian/epoch_<n>.pth
cd roundabout_env      && python transformer_1.py
cd intersection_env    && python transformer_1.py
```

The student is trained with the Gaussian NLL plus the boundary-aware penalty (`gaussian_nll_loss_soft`),
which penalizes μ ± kσ falling outside the style support. Observation dimensions are 16 (highway),
24 (roundabout) and 18 (intersection).

### 4. Evaluate

Checkpoints used in the paper: highway `epoch_120400.pth`, roundabout `epoch_95600.pth`,
intersection `epoch_145600.pth` (all under `transformer_weights_gaussian/`).

**Style-inference accuracy (NLL / per-dimension MSE)**

```bash
cd highway_env/student && python Transformer_test.py
cd roundabout_env      && python transformer_test1.py
cd intersection_env    && python transformer_test1.py
```

**Uncertainty calibration and the factorized posterior (highway)**

```bash
cd highway_env/student
python coverage_evaluation.py               # empirical coverage at 80 / 90 / 95 % nominal levels
python residual_correlation.py              # 6×6 residual correlation matrix, max |rho_ij|
python gaussian_head_verification.py --train   # NLL gap vs. rank-2 Cholesky covariance head
```

**Action envelopes under style uncertainty (highway)**

```bash
cd highway_env/student && python reachability_analysis.py   # -> PPO_CROWN_style_analysis.csv
```

**Response to abrupt behavior change (highway)**

```bash
cd highway_env/student && python scenario_style_evolution.py   # -> scenario_style_evolution.png, scenario_style_predictions.csv
```

**NGSIM US-101 real-data analysis**

Download the NGSIM US-101 trajectory data and place `trajectories-0750am-0805am.txt` in
`highway_env/student/archive/`. Then:

```bash
cd highway_env/student
python cluster_style_ngsim.py      # reward-feature clustering + Transformer style per cluster
python ngsim_heldout_analysis.py   # held-out behavioral-proxy correlations -> ngsim_heldout_results.csv
```

## Interpretation and Scope

The six style coordinates are cognitively motivated operational modeling variables rather than psychometrically validated personality traits. The NGSIM experiments assess alignment with observable behavioral proxies and do not establish psychological ground truth.

The action-envelope module bounds policy outputs under a specified style uncertainty set at a fixed observation. It provides an interface for downstream safety-oriented reasoning, rather than a closed-loop safety guarantee.

## Contact

- **Dejin Wang:** wang.dej@northeastern.edu
