# CLAUDE.md
This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.


## Workflow Orchestration

### 1. Plan Node Default
- Enter plan mode for ANY non-trivial task (3+ steps or architectural decisions)
- If something goes sideways, STOP and re-plan immediately – don't keep pushing
- Use plan mode for verification steps, not just building
- Write detailed specs upfront to reduce ambiguity

### 2. Subagent Strategy
- Use subagents liberally to keep main context window clean
- Offload research, exploration, and parallel analysis to subagents
- For complex problems, throw more compute at it via subagents
- One task per subagent for focused execution

### 3. Self-Improvement Loop
- After ANY correction from the user: update `tasks/lessons.md` with the pattern
- Write rules for yourself that prevent the same mistake
- Ruthlessly iterate on these lessons until mistake rate drops
- Review lessons at session start for relevant project

### 4. Verification Before Done
- Never mark a task complete without proving it works
- Diff behavior between main and your changes when relevant
- Ask yourself: "Would a staff engineer approve this?"
- Run tests, check logs, demonstrate correctness

### 5. Demand Elegance (Balanced)
- For non-trivial changes: pause and ask "is there a more elegant way?"
- If a fix feels hacky: "Knowing everything I know now, implement the elegant solution"
- Skip this for simple, obvious fixes – don't over-engineer
- Challenge your own work before presenting it

### 6. Autonomous Bug Fixing
- When given a bug report: just fix it. Don't ask for hand-holding
- Point at logs, errors, failing tests – then resolve them
- Zero context switching required from the user
- Go fix failing CI tests without being told how

---

## Task Management

1. **Plan First**: Write plan to `tasks/todo.md` with checkable items
2. **Verify Plan**: Check in before starting implementation
3. **Track Progress**: Mark items complete as you go
4. **Explain Changes**: High-level summary at each step
5. **Document Results**: Add review section to `tasks/todo.md`
6. **Capture Lessons**: Update `tasks/lessons.md` after corrections

---

## Core Principles

- **Simplicity First**: Make every change as simple as possible. Impact minimal code.
- **No Laziness**: Find root causes. No temporary fixes. Senior developer standards.
- **Minimal Impact**: Changes should only touch what's necessary. Avoid introducing bugs.



## Running the Code

```bash
python ParticleFilter.py
```

Dependencies: `numpy`, `scipy`, `matplotlib`

## Architecture

Single-file implementation (`ParticleFilter.py`) with two classes and a `main()` driver.

**`ParticleFilter`** — Core sequential Monte Carlo estimator for fatigue crack growth (Paris' Law model):
- State: particle arrays for crack size (`actual_crack`), Paris Law exponent (`m`), and coefficient (`c`)
- `predict()` → propagates particles forward one cycle increment (50 cycles, stress range 78) via `getCrack()`
- `update(weights, measured)` → weights particles using log-normal likelihood
- `resampleFromIndex(weights)` → systematic resampling when `neff < N/2`
- `get_posterior_pred()` → posterior predictive sampling (used in pure-prediction phase)
- `prognosis` property → probability of failure (fraction of particles exceeding `threshold = 0.015`)

**`RemainingUsefulLife`** — Post-hoc RUL estimator:
- Takes the full `all_predictions` matrix (time steps × particles) and finds per-particle first-passage time through a threshold (default `0.043`)
- `getRUL(t)` returns a list of RUL values relative to current time step `t`

**`main()` loop** — Two-phase execution:
1. **Assimilation phase**: iterates over `measured_crack` observations, running predict → update → conditional resample
2. **Prediction phase**: once measurements are exhausted, runs predict-only until mean crack exceeds threshold

Outputs three plots: crack mean ± 95% CI over cycles, probability of failure over cycles, and RUL histogram.

