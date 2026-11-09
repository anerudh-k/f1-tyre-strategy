# F1 Tyre Degradation & Pit Strategy Optimiser

Fits tyre degradation curves from real F1 race laps ([OpenF1](https://openf1.org)) and uses them to find the fastest pit strategy, including how robust it is when a safety car shows up.

## Results

Trained on 90 races from 2023 to 2026 and tested by predicting every 2025-26 race from earlier races only.

- Fuel burn plus track evolution is worth about -0.06 s/lap, the same size as tyre wear. Separating the two gives median wear of 0.055 (Soft), 0.052 (Medium) and 0.045 s/lap (Hard).
- A history model (the circuit's past races, adjusted toward the current season) predicts race tyre wear with an error of 0.024 s/lap. Ridge, decision tree, random forest and gradient boosting on circuit features did not beat it, and the tree models overfit.
- For Las Vegas 2026 the best plan is a one-stop on Soft and Hard, fastest in about 52% of 5,000 simulated races with random safety cars and uncertain wear. A two-stop never pays at this track.

![Model comparison](figures/model_comparison.png)
![Decision tree depth vs error](figures/tree_depth_curve.png)
![Predicted tyre wear, Las Vegas 2026](figures/predicted_target.png)
![Monte Carlo strategy comparison](figures/monte_carlo.png)

## What it does

**Part 1: Tyre degradation** (`notebooks/01_tyre_degradation.ipynb`)
- Loads a race and keeps only representative laps: green flag, no in/out laps, no lap 1, no traffic outliers
- Removes the fuel-burn effect (~3 s over a race), which otherwise hides tyre wear
- Pools drivers by measuring each lap against that driver's own baseline pace
- Fits a linear and a quadratic model per compound, chooses between them with 5-fold CV, and detects a degradation "cliff" with a piecewise-linear breakpoint search
- Checks how sensitive the fitted deg rates are to the fuel assumptions

**Part 2: Pit strategy** (`notebooks/02_pit_strategy.ipynb`)
- Brute-forces every legal 1-stop and 2-stop plan (two dry compounds required, stint lengths capped near what the data supports)
- Plots the one-stop pit window and a two-stop pit-lap map
- Finds the pit-loss threshold at which a two-stop beats a one-stop
- Runs a Monte Carlo simulation with random safety cars (pit under SC if a stop is due within 6 laps) and, for generic-model predictions, uncertain tyre wear; reports the time distribution and win rate for each plan
- Compares the optimum against what the top 10 finishers actually ran

**Part 3: Generic model, any circuit** (`notebooks/03_generic_model.ipynb`)
- Downloads every race on OpenF1 (2023 onwards, ~90 dry races), with a data-quality report per race
- EDA, then layered outlier removal (wet races, short/impossible stints, robust MAD z-score on lap residuals, rare tyre ages) with a report of what each layer removed
- **Per-race tyre model:** `lap time = driver pace + compound offset + (fuel + track evolution) × lap + wear × tyre age`, fitted with robust regression. Fuel and track evolution (about −0.06 s/lap) are the same size as tyre wear and get mixed into it when stints are fitted one at a time
- **Prediction:** compares a history model (the circuit's past races shrunk toward the season average) with Ridge, a decision tree, a random forest and gradient boosting on circuit features. Uses **forward-chaining validation**: every 2025–26 race is predicted from earlier races only. Overfitting is checked with the train/test gap and a tree-depth curve
- Predicts wear, compound offsets, pit loss and safety-car rate for any GP, plus the model's error, which notebook 02 uses as wear uncertainty in its Monte Carlo

## Model

Race time relative to a driver's baseline:

$$T = \sum_{\text{stints}} \sum_{a=1}^{n_i} \Delta_{c_i}(a) + N_{\text{stops}} \cdot t_{\text{pit}}$$

where $\Delta_c(a)$ is the fitted lap-time loss of compound $c$ at tyre age $a$. Fuel burn and safety-car lap times are identical across strategies, so they drop out of the comparison.

## Run it

```bash
pip install -r requirements.txt
jupyter notebook notebooks/01_tyre_degradation.ipynb   # one race, then 02
jupyter notebook notebooks/03_generic_model.ipynb       # all races -> predict any GP, then 02
```

On Colab, open a notebook and do Runtime → Run all; the first cell clones this repo. Set `DEMO = True` in notebook 01 to test the pipeline on a synthetic race without downloading data.

## Repo layout

```
f1strat/
  data.py          # OpenF1 loading, lap cleaning, fuel correction, pit-loss estimate
  degradation.py   # per-compound fits, CV model selection, cliff detection
  strategy.py      # strategy enumeration, Monte Carlo safety-car simulation
  generic.py       # multi-circuit dataset, outlier removal, per-race tyre model, history model, validation
notebooks/         # 01 single-race degradation / 03 generic model -> 02 strategy
data/              # cached OpenF1 responses + fitted models (json)
figures/           # saved plots
```

## Data source

Lap times, tyre stints, pit stops and race-control messages come from the free [OpenF1](https://openf1.org) API (2023 onwards). FastF1 was dropped because F1's live-timing server returns 403 to cloud machines such as Colab. Responses are cached as JSON in `data/cache/`.

## Assumptions and limitations

- Fuel: 100 kg at the start, 0.03 s/kg (both are configurable and sensitivity-checked)
- Notebook 01 fits stints one at a time, so track evolution gets folded into the deg rate (notebook 03's per-race model separates them)
- Tyre wear is modelled as linear in tyre age (no cliff)
- OpenF1 reports SOFT/MEDIUM/HARD, not Pirelli's C1–C5, so the same label can be a different compound at different tracks; the circuit features absorb part of this
- 2026 is a new regulation era, so check whether pre-2026 seasons help or hurt when predicting 2026 races
- Stint order only matters through the safety-car timing, because track position, overtaking difficulty and the undercut/overcut are not modelled yet
- At most one safety car per race, and no VSC

## Next steps

- Undercut model (out-lap on new tyres vs in-lap on old tyres)
- Track-position / dirty-air penalty
- Add Pirelli C-number per race as a feature
