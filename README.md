# Semi-Markov Expected Threat (xT)

A Python package for computing **Expected Threat (xT)** in football using a **Semi-Markov model** that incorporates sojourn time distributions fitted via Accelerated Failure Time (AFT) models.

---

## Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/Sebastianlecoz/semi-markov.git
cd semi-markov
pip install -r requirements.txt
```

---

## Quick Start

See [`example.py`](example.py) for a full working walkthrough. The main steps are:

```python
from build_dataset import build_dataset, list_competitions
from Class_Foot_xt import Foot_semi_xt

# 1. Build the dataset (uses StatsBomb open data)
sequences, df, features_df = build_dataset(
    competitions={55: [43]},  # EURO 2020
    verbose=True,
)

# 2. Initialise the model
model = Foot_semi_xt(sequences, df, features_df)

# 3. Fit sojourn distributions
model.evaluate_sejourn_distributions(min_obs=10, n_boot=50)

# 4. Inspect a specific transition
model.Sejourn_distribution_table("Pass", "Carry")
model.QQ_plot("Pass", "Carry", top_n=4)
model.Survival_plot("Pass", "Carry", top_n=3)

# 5. Train AFT models
model.train_aft()

# 6. Inspect AFT parameters for a transition
model.aft_parameters("Pass", "Carry")
```

---

## Main API

| Method | Description |
|--------|-------------|
| `evaluate_sejourn_distributions(min_obs, n_boot)` | Fit and rank parametric sojourn distributions for all transitions |
| `Sejourn_distribution_table(from, to)` | Print a summary table for one transition |
| `QQ_plot(from, to, top_n)` | QQ plot comparing top fitted distributions to empirical data |
| `Survival_plot(from, to, top_n)` | Survival curve plot for one transition |
| `train_aft()` | Train AFT models for all transitions using the best-fit distribution |
| `aft_parameters(from, to)` | Print AFT coefficient table for one transition |
| `train_XT_baseline(train_seqs, test_seqs, delta_t, horizon, seed)` | Train the timed Markov baseline xT model |
| `predict_XT_baseline(state, x, y)` | Predict xT from the baseline model |
| `train_XT_semi(train_seqs, test_seqs, delta_t, horizon, seed)` | Train the semi-Markov xT model |
| `predict_XT_semi(state, x, y)` | Predict xT from the semi-Markov model |

---

## Data

By default the package uses **StatsBomb open data** via the `statsbombpy` library. `build_dataset()` handles the download and formatting automatically. To use your own data, format it as a list of possession sequences and pass it directly to `Foot_semi_xt`.

---

## Requirements

See [`requirements.txt`](requirements.txt). Key dependencies: `lifelines`, `statsbombpy`, `scipy`, `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `rich`.
