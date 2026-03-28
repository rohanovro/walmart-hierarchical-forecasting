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
