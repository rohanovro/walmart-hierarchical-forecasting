# Walmart Hierarchical Demand Forecasting

**Mahmudul Hasan Rohan**  
Jashore University of Science and Technology, Bangladesh

Forecast weekly store-department sales across Walmart's 3-level hierarchy and
evaluate four reconciliation methods for coherent, cost-effective predictions.

---

## Methodology Note

All reconciliation methods use **training-period information only**. No test-period
actuals are used during reconciliation weight computation. ERM weights are fitted
on training residuals and applied to test-period base forecasts. Statistical
significance is assessed with a HAC-corrected Diebold-Mariano test (Newey-West
variance estimator) appropriate for serially-correlated weekly errors.

---

## Results at a Glance

Results from `walmart_pipeline.py` on the full Kaggle dataset
(test window = last 12 weeks, August 10 – October 26, 2012).

### Department Level

| Method | RMSE ($) | vs Base GBM |
|--------|:--------:|:-----------:|
| Base GBM (no reconciliation) | 2,987 | — |
| **Bottom-Up** | **2,987** | 0% (identical) |
| **ERM** ★ | **3,044** | +1.9% worse — but best of reconciled |
| MinT (proportional) | 4,498 | +50.6% worse |
| Top-Down | 4,848 | +62.3% worse |

### Store Level

| Method | RMSE ($) |
|--------|:--------:|
| **ERM** | **55,613** |
| Bottom-Up | 57,176 |
| MinT | 57,294 |
| Top-Down | 94,406 |

### Chain Level

| Method | RMSE ($) |
|--------|:--------:|
| **ERM** | **932,247** |
| Bottom-Up | 999,798 |
| Top-Down | 1,002,698 |
| MinT | 1,002,718 |

---

## Statistical Significance (DM test, HAC-corrected Newey-West)

| Comparison | DM Statistic | p-value | Result |
|------------|:------------:|:-------:|--------|
| MinT vs Bottom-Up (dept) | −11.356 | < 0.0001 | MinT significantly **worse** |
| **ERM vs Bottom-Up (dept)** | **−3.228** | **0.0012** | ERM significantly **better** |

---

## Key Findings

1. **ERM is the strongest method** at all three levels with p = 0.0012 vs Bottom-Up.
2. **Bottom-Up** is the best classical baseline — preserves base GBM accuracy at dept level.
3. **Top-Down and MinT (proportional)** both degrade dept-level accuracy. Coarse
   historical proportions are insufficient for heterogeneous store-dept pairs.
4. **Full covariance-based MinT** (Wickramasuriya et al. 2019) is identified as future work.

---

## Hierarchy

```
Walmart Chain  (1)
      ↓
  Stores  (45)
      ↓
Departments  (81)
      ↓
~3,331 Store × Dept pairs
```

---

## Dataset

Download from Kaggle → [Walmart Store Sales Forecasting](https://www.kaggle.com/competitions/walmart-recruiting-store-sales-forecasting/data)

| File | Description | Rows |
|------|-------------|------|
| `train.csv` | Weekly sales per store-dept | 421,570 |
| `stores.csv` | Store type (A/B/C) and size | 45 |
| `features.csv` | CPI, fuel price, temperature, markdowns | 8,190 |

---

## Quick Start

```bash
git clone https://github.com/rohanovro/walmart-hierarchical-forecasting.git
cd walmart-hierarchical-forecasting
pip install -r requirements.txt
# Place train.csv, stores.csv, features.csv in data/
python walmart_pipeline.py
```

For better accuracy: `pip install lightgbm` then set `USE_LGBM = True`.

---

## Pipeline — 8 Phases

| Phase | Description | Key Output |
|-------|-------------|------------|
| 1 | Data setup — load, merge, fill, flag holidays | Clean panel |
| 2 | Feature engineering — lags, rolling means, time | Feature matrix |
| 3 | GBM walk-forward CV — last 12 weeks as test | Base forecasts + training residuals |
| 4 | Reconciliation — BU, TD, MinT, ERM (training only) | 4 reconciled sets |
| 5 | Evaluation — RMSE, MAPE, HAC DM test | `outputs/phase5_results.csv` |
| 6 | Inventory simulation (illustrative) | `outputs/phase6_inventory.csv` |
| 7 | 5 publication-ready plots | `plots/*.png` |
| 8 | Auto-generated research summary | `docs/research_summary.md` |

---

## Reconciliation Methods

**Bottom-Up** — Sum dept forecasts upward. Dept accuracy unchanged from base model.

**Top-Down** — Disaggregate chain-level forecast by training-period proportions. Loses dept granularity.

**MinT (Proportional)** — Distribute store-level forecasts to depts using training shares. No test actuals used. Proportional approximation to full covariance MinT.

**ERM** — Ridge regression weight per store-dept, fitted on training residuals, applied to test forecasts. Best at all levels.

---

## ⚠️ Notes

**MAPE**: Extreme values at dept level due to near-zero sales pairs. RMSE is the primary metric.

**Inventory parameters** (Z=1.645, lead=2wk, hold=$0.25, stockout=$1.50) are illustrative
textbook defaults (Silver et al. 2017), not Walmart actuals.

---

## References

- Diebold & Mariano (1995). *JBES* 13(3).
- Hyndman et al. (2011). *CSDA* 55(9).
- Newey & West (1987). *Econometrica* 55(3).
- Silver, Pyke & Thomas (2017). *Inventory and Production Management* (4th ed.). CRC Press.
- Wickramasuriya, Athanasopoulos & Hyndman (2019). *JASA* 114(526).

---

## Research Summary

> Full academic write-up. Also available as [`research_summary.md`](research_summary.md).

### Abstract

Four hierarchical reconciliation methods — Bottom-Up, Top-Down, MinT (proportional),
and ERM — are evaluated on the Walmart Store Sales dataset (421,570 weekly
observations, 45 stores, 81 departments). A GBM base model with strict walk-forward
cross-validation serves as the base forecaster. All reconciliation weights use
training-period data only — no test-period actuals. Statistical significance is
assessed with a HAC-corrected Diebold-Mariano test (Newey-West variance). ERM
achieves the best performance at all three hierarchy levels and is statistically
significantly better than Bottom-Up (DM = −3.228, p = 0.0012).

### Introduction

Retail demand forecasting must produce coherent predictions across aggregation
levels. Independent department, store, and chain forecasts are incoherent — the
sum of department forecasts does not equal the store forecast — creating downstream
problems in inventory planning and supply chain coordination.

This project evaluates four reconciliation strategies on a real-world 3-level
retail hierarchy (chain → 45 stores → 81 departments → ~3,331 store-dept pairs)
to determine which method delivers the most accurate and cost-effective forecasts.

### Methods

**Base Model** — Gradient Boosting Regressor with walk-forward CV (last 12 weeks
as test, no temporal leakage). Features: 1/2/4-week lags, 4/8-week rolling means,
time features, holiday flags, CPI, fuel price, temperature, unemployment, markdowns.

**Bottom-Up** — Sum dept base forecasts upward. Dept accuracy unchanged from base.

**Top-Down** — Disaggregate chain-level forecast by training-period dept proportions
of chain total. Loses granular dept signal.

**MinT (Proportional)** — Distribute store-level base forecasts to depts using
each dept's training-period share of its store total. No test actuals used.
Proportional approximation to full covariance MinT (Wickramasuriya et al. 2019).

**ERM** — Ridge regression weight per store-dept fitted on training residuals,
applied to test-period base forecasts. No test targets used during weight learning.

**DM Test** — Newey-West HAC-corrected variance with lag truncation floor(T^(1/3))
to account for serial correlation in weekly forecast errors.

### Results

| Method | Level | RMSE ($) | MAPE (%) |
|--------|-------|:--------:|:--------:|
| Base GBM | Dept | 2,987 | 637.5 |
| Bottom-Up | Dept | 2,987 | 637.5 |
| Bottom-Up | Store | 57,176 | 3.9 |
| Bottom-Up | Chain | 999,798 | 1.7 |
| Top-Down | Dept | 4,848 | 1011.9 |
| Top-Down | Store | 94,406 | 6.4 |
| Top-Down | Chain | 1,002,698 | 1.7 |
| MinT (proportional) | Dept | 4,498 | 1005.6 |
| MinT (proportional) | Store | 57,294 | 3.9 |
| MinT (proportional) | Chain | 1,002,718 | 1.7 |
| **ERM** | **Dept** | **3,044** | 1143.6 |
| **ERM** | **Store** | **55,613** | **3.8** |
| **ERM** | **Chain** | **932,247** | **1.6** |

**Diebold-Mariano (HAC-corrected):**

| Comparison | DM Statistic | p-value | Result |
|------------|:------------:|:-------:|--------|
| MinT vs Bottom-Up (dept) | −11.356 | < 0.0001 | MinT significantly worse |
| **ERM vs Bottom-Up (dept)** | **−3.228** | **0.0012** | **ERM significantly better** |

**Inventory Cost (illustrative parameters — not Walmart actuals):**

| Method | Holding | Stockout | Total |
|--------|:-------:|:--------:|:-----:|
| Bottom-Up | $3,379,535 | $2,938,048 | $6,317,555 |
| MinT | $3,379,535 | $4,706,986 | $8,086,040 |

### Conclusion

ERM outperforms all methods at every hierarchy level with statistical significance
(p = 0.0012). Top-Down and proportional MinT both degrade dept-level accuracy,
confirming that coarse historical proportions are insufficient for heterogeneous
retail series. Full covariance-based MinT (Wickramasuriya et al. 2019) is the
primary direction for future work.

### References

- Diebold & Mariano (1995). Comparing Predictive Accuracy. *JBES* 13(3).
- Hyndman et al. (2011). Optimal combination forecasts. *CSDA* 55(9).
- Newey & West (1987). HAC Covariance Matrix. *Econometrica* 55(3).
- Silver, Pyke & Thomas (2017). *Inventory and Production Management* (4th ed.). CRC Press.
- Wickramasuriya, Athanasopoulos & Hyndman (2019). Optimal Forecast Reconciliation. *JASA* 114(526).
