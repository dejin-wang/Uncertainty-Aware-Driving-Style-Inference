# Uncertainty-Aware Driving-Style Inference

Official implementation of the paper:

**“Uncertainty-Aware Driving-Style Inference Using Interpretable Continuous Traits”**

**Authors:** Dejin Wang and Seyede Fatemeh Ghoreishi

## Overview

This work introduces a teacher–student framework for online driving-style inference from short state-only observation windows, with an explicit representation of inference uncertainty.

Driving style is represented by six bounded continuous operational coordinates:

- Aggressiveness
- Impulsivity
- Risk tolerance
- Rule conformity
- Prospectiveness
- Expertness

A style-conditioned teacher policy generates labeled trajectory–style pairs over the continuous style space. A Transformer-based probabilistic student estimates a factorized posterior over the six coordinates in a single forward pass, producing both style estimates and dimension-wise uncertainty intervals.

A boundary-aware training objective discourages predicted intervals from extending beyond the valid style bounds. At deployment, these intervals are intersected with the valid style support and propagated through the style-conditioned policy to construct conservative one-step action envelopes.

## Simulation Environment

The simulation experiments use **highway-env 1.10.1** and cover three tasks:

- **Highway:** Car following and lane-changing interactions.
- **Roundabout:** Merging and priority negotiation.
- **Intersection:** Crossing interactions and conflict resolution.

Surrounding vehicles use the Intelligent Driver Model (IDM) for longitudinal motion and MOBIL for lane-changing decisions. A learned style-conditioned teacher policy controls the ego vehicle to generate style-labeled trajectories.


## Interpretation and Scope

The six style coordinates are cognitively motivated operational modeling variables rather than psychometrically validated personality traits. The NGSIM experiments assess alignment with observable behavioral proxies and do not establish psychological ground truth.

The action-envelope module bounds policy outputs under a specified style uncertainty set at a fixed observation. It provides an interface for downstream safety-oriented reasoning, rather than a closed-loop safety guarantee.

## Contact

- **Dejin Wang:** wang.dej@northeastern.edu


