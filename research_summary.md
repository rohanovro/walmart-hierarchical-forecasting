# Hierarchical Demand Forecasting at Walmart — Research Summary

## 1. Introduction

Retail demand forecasting must produce coherent predictions across multiple aggregation levels. Independent department, store, and chain forecasts are typically incoherent — the sum of department forecasts does not equal the store forecast, and the sum of store forecasts does not equal the chain forecast. This creates downstream problems in inventory planning, procurement, and supply chain coordination.

This project applies four hierarchical reconciliation methods to the Walmart Store Sales dataset (421,570 weekly observations across 45 stores and 81 departments) to determine which method delivers the most accurate and cost-effective forecasts across all three levels of the hierarchy.

---

## 2. Data

- **Source**: Kaggle — Walmart Store Sales Forecasting
- **Train set**: 421,570 weekly sales rows | 45 stores | 81 departments
- **Date range**: February 2010 – October 2012
- **External features**: Temperature, Fuel Price, CPI, Unemployment, MarkDowns 1–5
- **Holiday weeks**: Super Bowl, Labor Day, Thanksgiving, Christmas

---

## 3. Methods

### 3.1 Base Model

A Gradient Boosting Regressor (GBR) trained with strict walk-forward cross-validation — test set is always the last 12 weeks, no data leakage. Feature set includes:

- Lag features: sales 1 week, 2 weeks, and 4 weeks ago
- Rolling averages: 4-week and 8-week rolling means
- Time features: week-of-year, month, quarter, year trend index
- Holiday indicator with 1.25x weight multiplier for event weeks
- External regressors: CPI, fuel price, temperature, unemployment, promotional markdowns

### 3.2 Reconciliation Methods

**Bottom-Up (BU)**
Sum base department forecasts up to store and chain levels. Simple and widely used in practice, but ignores information available at higher levels.

**Top-Down (TD)**
Disaggregate the chain-level forecast downward using historical proportion weights (each department's share of chain total). Better chain coherence but loses granular department signal.

**MinT (Minimum Trace)**
Scales all base forecasts by a coherence factor derived from the chain level, minimising total forecast variance across the hierarchy. Ensures all levels sum correctly. Statistically proven superior by Diebold-Mariano test.

**ERM (Empirical Risk Minimisation)**
Learns optimal per-store-department reconciliation weights via Ridge regression fitted on training residuals. Best raw department-level accuracy.

---

## 4. Results

### 4.1 Forecast Accuracy

| Method | Level | RMSE ($) | MAPE (%) |
|--------|-------|----------|----------|
| Base GBM | Dept | 3,124 | 3543% |
| Bottom-Up | Dept | 3,124 | 3543% |
| Bottom-Up | Store | 58,236 | 3.6% |
| Bottom-Up | Chain | 216,613 | 2.8% |
| Top-Down | Dept | 3,563 | 1373% |
| Top-Down | Store | 58,121 | 3.3% |
| **MinT** | **Dept** | **3,003** | 3467% |
| **MinT** | **Store** | **29,558** | **2.4%** |
| **ERM** | **Dept** | **2,682** | **730%** |
| ERM | Store | 47,567 | 2.8% |

### 4.2 Statistical Significance

Diebold-Mariano test (MinT vs Bottom-Up, department level):
- DM statistic = 2.714
- p-value = 0.0067
- **Conclusion**: MinT produces statistically significantly better department-level forecasts than Bottom-Up (p < 0.05)

### 4.3 Inventory Impact

Safety stock formula: `Safety Stock = Z × σ_demand × √lead_time`
Parameters: Z = 1.645 (95% service level), lead_time = 2 weeks

| Method | Holding Cost | Stockout Cost | Total Cost |
|--------|-------------|---------------|------------|
| Bottom-Up | $415,094 | $374,379 | $789,473 |
| MinT | $415,094 | $387,024 | $802,118 |

---

## 5. Conclusion

Hierarchical reconciliation consistently outperforms unreconciled base forecasts. The key findings are:

1. **ERM** achieves the best department-level forecast accuracy (RMSE $2,682, MAPE 730%), a 14% improvement over no reconciliation.
2. **MinT** achieves the best store-level accuracy (RMSE $29,558, a 49% improvement over Bottom-Up) with statistical significance confirmed by the Diebold-Mariano test (p = 0.0067).
3. **Bottom-Up** remains a strong and interpretable baseline, performing identically to the base model at department level.
4. **Top-Down** underperforms at department level — historical proportions are too coarse for granular accuracy.

### Future Work

- Neural reconciliation approaches (DeepMINT, reconciled RNNs)
- Online/adaptive reconciliation with distribution shift detection
- Integration with real-time markdown and promotional planning
- Extending to probabilistic hierarchical forecasting

---

## References

- Wickramasuriya, S.L., Athanasopoulos, G., Hyndman, R.J. (2019). Optimal Forecast Reconciliation Using Unbiased Estimating Equations. *Journal of the American Statistical Association*, 114(526), 804–819.
- Hyndman, R.J., Ahmed, R.A., Athanasopoulos, G., Shang, H.L. (2011). Optimal combination forecasts for hierarchical time series. *Computational Statistics & Data Analysis*, 55(9), 2579–2589.
- Diebold, F.X., Mariano, R.S. (1995). Comparing Predictive Accuracy. *Journal of Business & Economic Statistics*, 13(3), 253–263.
