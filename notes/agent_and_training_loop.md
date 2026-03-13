# Agent & Training Loop

Notes on `src/agent.py` and `src/run_single_agent.py`.

---

## `src/agent.py` — The Agent

### Architecture: `PrioritizedSweepingAgent`

This is a **model-based tabular agent**. Instead of learning a policy directly from rewards (like vanilla Q-learning), it:
1. Maintains an explicit **learned model** of the environment (`CountBasedTransitionModel`)
2. Uses that model to **plan** Q-values globally after every single step

It stores two core tables:
- `q_table`: shape `(|S|, |A|)` — Q-values for every state-action pair
- `U`: shape `(|S|,)` — the current value estimate `V(s) = max_a Q(s,a)`

**Action selection** is greedy over Q-values (ties broken randomly). Softmax/temperature exploration is commented out — the design philosophy is that *intrinsic rewards should drive exploration*, not stochastic action selection.

---

### How it's trained: Prioritized Sweeping

After every environment step, `prioritized_sweeping()` runs. The key insight is: when one state's value changes, neighbouring states' values may also need updating. Rather than sweeping all states uniformly, it prioritises states with the *largest pending value change* `|U(s) − V(s)|`.

The vectorised Bellman update is:

```
Q(s,a) = Σ_s' T(s'|s,a) · [(1−γ)·R(s,a,s') + γ·U(s')·(1 − terminal)]
```

The `(1−γ)` scaling keeps Q-values in [0,1] when rewards are in [0,1]. Terminal transitions zero out the future value term.

The sweep loop then:
1. Picks `s*` = argmax of pending priorities
2. Commits `U[s*] = V[s*]`
3. Finds all predecessor states `s` where `T(s*|s,a) · |ΔV|` is large enough to matter
4. Incrementally updates those Q-values and recomputes their priorities
5. Repeats up to `T_PS = 100 × |S|` times, with early stopping when relative change `< 1e-8`

---

## `src/run_single_agent.py` — The Entry Point

This is a CLI script for launching one training run on a cluster. What it does:

**1. Parse arguments** — specifies agent type (`ps` = PrioritizedSweeping), rewards (`info_gain`, `empowerment`, `novelty`, or combinations), γ, learning rate, Q-init, seed, number of steps.

**2. Build the environment** — creates `MixedEnv` wrapped with `AgentPosAndDirWrapper` (so the agent sees `(x, y, dir)` tuples). Calls `env.reset()` *before* `get_all_states()` — this is important because the grid isn't initialised until reset, so calling it before would incorrectly include wall positions.

**3. Build the agent** — instantiates `PrioritizedSweepingAgent` with a hardcoded reward config:
```python
DEFAULT_REWARD_CFG = {
    "empowerment": {"num_steps": 1, "method": "blahut_arimoto"},
    "info_gain":   {"method": "LittleSommerPIG"},
    "novelty":     {}
}
```

**4. Run and save** — delegates to `run_or_load()` which either loads an existing pickle from disk (if the same hyperparameters were already run) or runs `run_agent()` and saves the result.

---

## The Full Training Loop (in `run_utils.run_agent`)

```
for each step:
    action = agent.select_action(state)          # greedy Q
    next_state, reward, ... = env.step(action)
    agent.update(state, action, next_state, ...)
        → model.update(...)                      # increment count[s,a,s']
            → recompute intrinsic reward R       # empowerment/info_gain/novelty
        → prioritized_sweeping()                 # re-plan Q globally
    if eval checkpoint:
        snapshot Q-values, empowerment heatmap, etc.
```

The agent never receives an extrinsic reward — it is driven entirely by the intrinsic reward signal baked into `model.R`. This is the core experimental setup: does an agent motivated purely by empowerment / info-gain / novelty learn to explore the environment efficiently?
