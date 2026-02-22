# Particle Filter — Remaining Useful Life Estimation

A Sequential Monte Carlo (SMC) implementation for fatigue crack growth prognosis using Paris' Law, with a generic particle filter library that can be adapted to any state-space problem.

## Overview

This project estimates the **Remaining Useful Life (RUL)** of a structural component subject to fatigue crack growth. A particle filter assimilates crack size measurements and propagates an ensemble of particles forward in time to predict when the crack will exceed a failure threshold.

## Architecture

```
ParticleFilter.py
├── StateSpaceModel      — Abstract base class defining the model interface
├── ParticleFilter       — Generic Sequential Monte Carlo algorithm
├── CrackGrowthModel     — Paris' Law implementation of StateSpaceModel
├── RemainingUsefulLife  — Post-hoc RUL estimator
└── main()               — Driver: assimilation + prediction + plotting
```

### `StateSpaceModel` (ABC)

Defines the interface any domain model must implement:

| Method | Description |
|---|---|
| `sample_initial(N)` | Returns `(N, d)` array of initial particles |
| `transition(particles)` | Propagates particles one step forward, returns `(N, d)` |
| `likelihood(measurement, particles)` | Returns `(N,)` likelihoods |
| `estimate(particles, weights)` | Returns `(mean, var)` — has a sensible weighted default |

### `ParticleFilter`

Generic SMC filter — no domain knowledge baked in:

| Method | Description |
|---|---|
| `predict()` | Calls `model.transition()` on all particles |
| `update(measurement)` | Weights particles by likelihood, normalises |
| `resample()` | Systematic resampling |
| `resample_if_needed()` | Resamples when `neff < N/2` |
| `neff()` | Effective sample size: `1 / Σwᵢ²` |
| `estimate()` | Delegates to `model.estimate()` |
| `prognosis(threshold)` | Fraction of particles (crack column) exceeding threshold |

### `CrackGrowthModel`

Implements Paris' Law crack growth:

```
da/dN = C · (ΔS √(π·a))^m
```

State vector per particle: `[crack_size, m, c]`

| Parameter | Default | Description |
|---|---|---|
| `sigma` | 0.001 | Measurement noise (log-normal std) |
| `stress_range` | 78 | Cyclic stress range ΔS |
| `dN` | 50 | Cycle increment per step |
| `threshold` | 0.015 | Failure crack size |

## Installation

```bash
pip install numpy scipy matplotlib
```

## Usage

```bash
python ParticleFilter.py
```

This runs with 100,000 particles and two crack size measurements, producing three plots:

1. **Crack mean ± 95% CI** over cycles
2. **Probability of failure** over cycles
3. **RUL histogram** across all particles

### Custom measurements

```python
from ParticleFilter import ParticleFilter, CrackGrowthModel

model = CrackGrowthModel()
pf = ParticleFilter(num_particles=10000, model=model)

for measurement in my_measurements:
    pf.predict()
    pf.update(measurement)
    pf.resample_if_needed()
    mean, var = pf.estimate()
    print(f"Crack estimate: {mean:.4f} ± {var**0.5:.4f}")
```

### Custom state-space model

Subclass `StateSpaceModel` to apply the filter to any domain:

```python
from ParticleFilter import StateSpaceModel, ParticleFilter
import numpy as np

class MyModel(StateSpaceModel):
    def sample_initial(self, N):
        return np.random.randn(N, 2)          # (N, d)

    def transition(self, particles):
        particles[:, 0] += 0.1 * particles[:, 1]
        return particles

    def likelihood(self, measurement, particles):
        return np.exp(-0.5 * (particles[:, 0] - measurement)**2)

pf = ParticleFilter(5000, MyModel())
```

## Reference

- Gordon, N.J., Salmond, D.J., Smith, A.F.M. (1993). *Novel approach to nonlinear/non-Gaussian Bayesian state estimation.* IEE Proceedings F.
- Paris, P., Erdogan, F. (1963). *A critical analysis of crack propagation laws.* Journal of Basic Engineering.

## Author

Spandan Mishra
