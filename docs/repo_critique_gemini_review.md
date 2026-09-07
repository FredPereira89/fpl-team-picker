# FPL Team Picker - In-Depth Codebase Critique & Proposed Improvements (Gemini Review)

## Overview
The `FPL Team Picker` is built on a robust Mixed-Integer Linear Programming (MILP) solver (`pulp`) to find mathematically optimal teams based on the expected points (xP) model.

However, to push this engine to the highest FPL ranks, the mathematical model must reflect the intricacies of FPL scoring rules and the reality of football more accurately. Currently, the solver has some mathematical blind spots where it simplifies FPL mechanics. This leads to suboptimal budget allocation, risk-averse captaincy choices, and slightly inaccurate expected points generation for specific positions.

Here is a detailed, component-level critique of the repository alongside concrete implementation proposals.

---

## 1. Vice-Captaincy & The Conditional Captaincy Problem
**File**: `fpl/optimize/squad.py` and `fpl/optimize/transfers.py`

### The Current Flaw
The MILP model correctly doubles the captain's xP in the objective function:
```python
prob += pulp.lpSum(
    xp[i] * start[i] + bench_w * xp[i] * (squad[i] - start[i]) + xp[i] * cap[i]
    for i in ids
)
```
However, it completely ignores the Vice-Captain. The `xp[i]` passed to the solver is *unconditional* expected points (already multiplying the points if they play by `p_start`). 
If your Captain is a rotation risk (e.g., `p_start=0.5`), the solver assumes you get 0 captaincy points for the other 50% of the time. In FPL reality, the Vice-Captain inherits the armband. 

Because the solver doesn't model the Vice-Captain, it heavily penalizes high-upside rotation risks (like Pep Roulette players) for the captaincy, incorrectly favoring highly-nailed players with lower ceilings.

### Proposed Implementation
Introduce the Vice-Captain (`vcap[i]`) into the MILP constraints and objective function.
1. **New Variables**: Create a boolean decision variable array for `vcap[i]`.
2. **Constraints**:
   ```python
   prob += pulp.lpSum(vcap[i] for i in ids) == 1
   for i in ids:
       prob += vcap[i] <= start[i]
       prob += cap[i] + vcap[i] <= 1  # Cannot be both Cap and VCap
   ```
3. **Objective Adjustments**: Approximate the expected captaincy points by giving the Vice-Captain a bonus weighted by the probability that the main captain misses out. 
   *(Note: This requires passing `p_start` from `xp_df` into the objective).*
   ```python
   # Approximation: Assume the average captain has a ~90% chance of starting
   # To be completely accurate, you'd need quadratic programming (cap_start * vcap_points), 
   # but a linear approximation using a global `p_start_captain_avg` or 
   # pre-calculated conditional expectations works well in MILP.
   ```

---

## 2. Bench Ordering & Substitutions
**File**: `fpl/optimize/squad.py` and `fpl/optimize/transfers.py`

### The Current Flaw
The solver uses a flat `bench_w` for all non-starters:
```python
bench_w = float(np.mean(cfg.bench_weight)) # Averages [0.20, 0.15, 0.05, 0.02]
```
FPL bench order strictly matters. Bench 1 comes on first. If all your starters have a 90% chance of playing, Bench 1 has an ~70% chance of being substituted in, while Bench 3 has almost a 0% chance. 

By flattening the weights, the solver overvalues Bench 3 and undervalues Bench 1. This leads the solver to waste critical budget on the 3rd bench spot (e.g., buying a £4.5m player when a £4.0m would do) instead of upgrading the starting XI or Bench 1.

### Proposed Implementation
Make the MILP model respect strict bench slots:
1. **New Variables**: Instead of `start[i]` and `squad[i]`, explicitly model the bench slots: `bench_1[i]`, `bench_2[i]`, `bench_3[i]`, `bench_gk[i]`.
2. **Constraints**:
   ```python
   prob += pulp.lpSum(bench_1[i] for i in ids) == 1
   prob += pulp.lpSum(bench_2[i] for i in ids) == 1
   # ...etc
   ```
3. **Objective Adjustments**: Apply the specific configuration weights to the specific slots.
   ```python
   prob += pulp.lpSum(
       xp[i] * start[i] + 
       cfg.bench_weight[0] * xp[i] * bench_1[i] +
       cfg.bench_weight[1] * xp[i] * bench_2[i] + ...
   )
   ```

---

## 3. Goalkeeper Modeling (The Missing Edge)
**File**: `fpl/model/scoring.py` and `fpl/model/strength.py`

### The Current Flaw
The backtest explicitly calls out that Goalkeeper predictions have "no measurable skill (Spearman 0.034)". 
The current model relies on `saves90`, shrinking a player's historical saves toward a positional mean. 

Goalkeeper saves are highly dependent on the opponent. A goalkeeper playing against Arsenal will face significantly more Shots on Target (SOT) and make more saves than one playing against Southampton. The current model ignores fixture difficulty when calculating save points.

### Proposed Implementation
Model opponent attacking threat to dynamically predict save points:
1. **Team Attacking Strength**: In `fpl/model/strength.py`, generate a metric for "Expected Shots on Target For" based on team history.
2. **Matchup Simulation**: In `fpl/model/xp.py`, when iterating through fixtures, calculate the Expected Shots on Target Against (xSOTA) for the goalkeeper's team.
3. **Save Points**: Multiply `xSOTA` by the goalkeeper's historical Save Percentage to predict dynamic save points per fixture.

---

## 4. Co-optimizing Transfers and Hits
**File**: `fpl/optimize/transfers.py`

### The Current Flaw
The `optimize_transfers` function wraps the solver in a `for` loop, solving the entire MILP problem for `0`, `1`, `2`... up to `free_transfers + max_paid_hits` transfers.
This is computationally inefficient and prevents the solver from naturally finding the absolute best trade-off globally.

### Proposed Implementation
Introduce the number of transfers as a decision variable in a single MILP pass.
1. **Track Changes**: 
   ```python
   # 1 if player i was in current squad and is NO LONGER in squad
   prob += pulp.lpSum((1 - squad[i]) for i in current_set) == transfers_made
   ```
2. **Model Free vs. Paid Transfers**:
   ```python
   # paid_transfers = max(0, transfers_made - free_transfers)
   # Requires standard MILP max constraints
   prob += paid_transfers >= transfers_made - free_transfers
   prob += paid_transfers >= 0
   ```
3. **Objective Penalty**:
   ```python
   # Subtract the hit cost from the total objective
   - (paid_transfers * cfg.hit_cost)
   ```

---

## 5. Data Pipeline Inefficiencies
**File**: `fpl/data/client.py` and `fpl/pipeline.py`

### The Current Flaw
Data fetching is blocking and synchronous:
```python
summaries = client.element_summaries(players["player_id"].tolist(), progress=progress)
```
Fetching player history sequentially takes ~10-12 minutes on a cold cache (assuming 1 request per second). This dramatically slows down the iteration, debugging, and backtesting loop.

### Proposed Implementation
Refactor data fetching to use `aiohttp` and `asyncio`.
1. Use an `asyncio.Semaphore` to cap concurrency (e.g., 5-10 concurrent requests) to respect FPL API rate limits.
2. Batch fetch all `element_summaries` concurrently. This will reduce a 10-minute cold cache warmup to roughly 30-60 seconds.
