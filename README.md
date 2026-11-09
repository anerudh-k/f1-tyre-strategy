# F1 Tyre Degradation & Pit Strategy Optimiser

Fits tyre degradation curves from real F1 race laps ([FastF1](https://docs.fastf1.dev/)) and uses them to find the fastest pit strategy, including how robust it is when a safety car shows up.

<!-- After running on real data, drop your key figures here, e.g.
![Degradation](figures/degradation_curves.png)
![One-stop window](figures/one_stop_window.png)
-->

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
- Runs a Monte Carlo safety-car simulation (pit under SC if a stop is due within 6 laps) and reports the time distribution and win rate for each plan
- Compares the optimum against what the top 10 finishers actually ran

## Model

Race time relative to a driver's baseline:

$$T = \sum_{\text{stints}} \sum_{a=1}^{n_i} \Delta_{c_i}(a) + N_{\text{stops}} \cdot t_{\text{pit}}$$

where $\Delta_c(a)$ is the fitted lap-time loss of compound $c$ at tyre age $a$. Fuel burn and safety-car lap times are identical across strategies, so they drop out of the comparison.

## Run it

```bash
pip install -r requirements.txt
jupyter notebook notebooks/01_tyre_degradation.ipynb   # then 02
```

On Colab, set `REPO` in the first cell to this repo's URL. Set `DEMO = True` in notebook 01 to test the pipeline on a synthetic race without downloading data.

## Repo layout

```
f1strat/
  data.py          # FastF1 loading, lap cleaning, fuel correction, pit-loss estimate
  degradation.py   # per-compound fits, CV model selection, cliff detection
  strategy.py      # strategy enumeration, Monte Carlo safety-car simulation
notebooks/         # 01 degradation -> 02 strategy
data/              # cached FastF1 data + fitted models (json)
figures/           # saved plots
```

## Assumptions and limitations

- Fuel: 100 kg at the start, 0.03 s/kg (both are configurable and sensitivity-checked)
- Degradation is fitted from a single race, and track evolution is folded into the deg rate
- Stint order only matters through the safety-car timing, because track position, overtaking difficulty and the undercut/overcut are not modelled yet
- At most one safety car per race, and no VSC

## Next steps

- Undercut model (out-lap on new tyres vs in-lap on old tyres)
- Track-position / dirty-air penalty
- Pool several races or seasons for each track
