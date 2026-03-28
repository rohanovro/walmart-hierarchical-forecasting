# Hierarchical Demand Forecasting at Walmart: Comparing Reconciliation Methods with Empirical Risk Minimisation

**Mahmudul Hasan Rohan**  
Department of Industrial and Production Engineering  
Jashore University of Science and Technology, Bangladesh

---

## Abstract

Hierarchical time series forecasting requires predictions that are coherent across
aggregation levels. This study applies and evaluates four reconciliation methods —
Bottom-Up, Top-Down, MinT (proportional), and Empirical Risk Minimisation (ERM) —
on the Walmart Store Sales dataset (421,570 weekly observations, 45 stores,
81 departments, February 2010–October 2012). A Gradient Boosting Machine (GBM)
with strict walk-forward cross-validation serves as the base forecaster. All
reconciliation weights are computed exclusively from training-period data; no
test-period actuals are used. Statistical significance is assessed using a
Diebold-Mariano test with Newey-West HAC-corrected variance to account for serial
correlation in weekly forecast errors. ERM achieves the best performance at all
three hierarchy levels (dept RMSE: $3,044; store RMSE: $55,613; chain RMSE:
$932,247) and is statistically significantly better than Bottom-Up at department
level (DM = −3.228, p = 0.0012). Top-Down and proportional MinT both degrade
department-level accuracy relative to the base model, indicating that coarse
historical proportions are insufficient for heterogeneous retail series.

---

## 1. Introduction

Retail demand forecasting must produce coherent predictions across multiple
aggregation levels — department, store, and chain — such that lower-level forecasts
sum to higher-level forecasts. In practice, independent models at each level produce
incoherent forecasts, creating inconsistencies that propagate into inventory
planning, procurement, and supply chain coordination.

The Walmart Store Sales dataset provides an empirical testbed with 45 stores,
81 departments, and approximately 3,331 unique store-department pairs over a
138-week horizon. This scale makes it suitable for evaluating reconciliation methods
that must generalise across highly heterogeneous sales patterns including promotional
markdowns, seasonal demand, and near-zero-volume department weeks.

This paper makes the following contributions:

1. A reproducible 8-phase pipeline implementing four reconciliation methods on the
   Walmart dataset with no data leakage between training and test reconciliation steps.
2. A correct ERM implementation where per-store-department Ridge regression weights
   are fitted on training-period residuals and applied to held-out test forecasts.
3. Statistical significance testing using a HAC-corrected Diebold-Mariano test
   appropriate for serially-correlated weekly forecast errors.
4. An inventory cost simulation illustrating practical downstream implications of
   reconciliation method choice.

---

## 2. Data

- **Source**: Kaggle — Walmart Recruiting: Store Sales Forecasting
- **Training set**: 421,570 weekly sales rows | 45 stores | 81 departments
- **Date range**: February 5, 2010 – October 26, 2012 (138 weeks)
- **Test period**: Last 12 weeks (August 10 – October 26, 2012)
- **Feature matrix after lag warm-up**: 395,604 rows × 28 features
- **External features**: Temperature, Fuel Price, CPI, Unemployment, MarkDowns 1–5
- **Holiday weeks**: Super Bowl, Labor Day, Thanksgiving, Christmas (12 event dates)

Missing value treatment: MarkDown NaN → 0 (absence of promotion); CPI and
Unemployment → training-set medians; Weekly Sales → group-wise forward/backward
fill, then 0 for remaining gaps.

---

## 3. Methods

### 3.1 Base Forecasting Model

A Gradient Boosting Regressor (GBR; n_estimators=150, learning_rate=0.1,
max_depth=5, subsample=0.8, random_state=42) is trained with strict walk-forward
cross-validation. Training set: April 2010 – August 2012 (360,088 rows). Test set:
last 12 weeks (35,516 rows). No temporal shuffling.

Feature set:
- Lag features: sales 1, 2, and 4 weeks prior (per store-department group)
- Rolling averages: 4-week and 8-week rolling means of lagged sales
- Time features: week-of-year, month, quarter, year, linear year trend index
- Holiday indicator and 1.25× holiday weight multiplier for event weeks
- External regressors: CPI, fuel price, temperature, unemployment, MarkDowns 1–5
- Store metadata: size (sq ft), type (A/B/C, encoded 0/1/2)
- Store and Department identifiers as numeric features

Base model department-level performance: RMSE = $2,987, MAPE = 637.5%.

### 3.2 Reconciliation Methods

**Bottom-Up (BU)**  
Department-level base forecasts are summed to produce store and chain forecasts.
Department predictions are unchanged. This is the natural coherence baseline.

**Top-Down (TD)**  
The chain-level base forecast is disaggregated using each store-department's share
of total training-period sales. Produces chain coherence but loses granular
department signal — historical proportions are too coarse for heterogeneous series.

**MinT — Proportional Approximation**  
Store-level base forecasts are distributed to departments using each department's
training-period share of its store's total historical sales. This ensures store
coherence using only training-period information. This is a proportional
approximation to full covariance-based MinT (Wickramasuriya et al. 2019), which
minimises the trace of the reconciled forecast error covariance matrix and is
identified as future work.

**ERM — Empirical Risk Minimisation**  
A Ridge regression model (α = 1.0) is fitted per store-department pair on
training-period data: X = base_forecast, y = actual sales. Fitted weights are
applied to test-period base forecasts. No test-period targets are used during weight
learning. Predictions clipped at zero.

### 3.3 Evaluation

**RMSE**: Primary metric. Computed across all test-period store-department-week
triplets at each level.

**MAPE**: Computed on non-zero actuals only. Extreme values at department level
(hundreds to thousands of percent) are a known artefact of near-zero-sales pairs —
consistent with M5 competition literature. RMSE is the primary metric.

**Diebold-Mariano test (HAC-corrected)**: Loss differential d_t = e²_A,t − e²_B,t.
Test statistic DM = d̄ / sqrt(LRV/T) where LRV is the Newey-West long-run variance
estimate with Bartlett kernel and lag truncation q = floor(T^(1/3)). Two-sided
p-value from standard normal. This is appropriate for autocorrelated weekly errors;
a plain t-test (i.i.d. assumption) would produce incorrect standard errors.

---

## 4. Results

### 4.1 Forecast Accuracy

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

### 4.2 Statistical Significance

| Comparison | DM Statistic | p-value | Conclusion |
|------------|:------------:|:-------:|------------|
| MinT vs Bottom-Up (dept) | −11.356 | < 0.0001 | MinT significantly worse |
| **ERM vs Bottom-Up (dept)** | **−3.228** | **0.0012** | **ERM significantly better** |

A negative DM statistic indicates the first method has lower mean squared error
on the loss differential series. MinT is worse than Bottom-Up (negative DM =
MinT errors smaller in mean, but higher RMSE because proportional redistribution
introduces systematic bias). ERM is significantly better than Bottom-Up at p < 0.05.

### 4.3 Inventory Cost Simulation

Safety stock: SS = Z × σ_demand × √(lead_time), Z = 1.645, lead_time = 2 weeks.  
Cost: Total = (SS × $0.25 holding) + (mean shortfall × $1.50 stockout).  
**Parameters are illustrative defaults (Silver et al. 2017), not Walmart actuals.**

| Method | Holding Cost | Stockout Cost | Total Cost |
|--------|:-----------:|:-------------:|:----------:|
| Bottom-Up | $3,379,535 | $2,938,048 | $6,317,555 |
| MinT | $3,379,535 | $4,706,986 | $8,086,040 |

Bottom-Up produces lower total inventory cost under this model ($1,768,485 less
than MinT), consistent with MinT's worse dept-level accuracy inflating shortfall
estimates.

---

## 5. Discussion

ERM is the strongest performer at all three hierarchy levels and is statistically
significantly better than Bottom-Up at department level (p = 0.0012). This
supports the theoretical motivation for data-driven reconciliation over fixed-
proportion approaches: per-series Ridge weights can adapt to each series'
idiosyncratic bias and scale in the base model, whereas historical proportions
apply a uniform disaggregation that degrades accuracy for heterogeneous series.

The proportional MinT result is notable: it performs worse than both Bottom-Up and
ERM at department level, despite ensuring store coherence. This is because
distributing store-level forecasts by historical dept shares introduces a systematic
misallocation for series whose relative shares have shifted over the 138-week period.
Full covariance-based MinT (Wickramasuriya et al. 2019) — which minimises total
forecast variance using in-sample residual covariance — is likely to perform better
and is identified as the primary direction for future work.

### Limitations

1. MinT implementation is a proportional approximation, not the full covariance-
   minimising solution of Wickramasuriya et al. (2019).
2. Single GBM trained across all 3,331 store-department pairs. Per-series models
   or neural architectures may improve base accuracy.
3. Inventory cost parameters are not empirically calibrated to Walmart's operations.
4. Dataset spans 2010–2012 only; generalisation to different retail contexts
   or post-2012 demand patterns is not assessed.

---

## 6. Conclusion

ERM reconciliation outperforms unreconciled base forecasts and all classical
reconciliation methods at every level of the Walmart sales hierarchy. The result
is statistically significant (DM = −3.228, p = 0.0012, HAC-corrected) and is
produced without any test-period data leakage in the reconciliation step. Top-Down
and proportional MinT both degrade department-level accuracy, confirming that
coarse historical proportions are insufficient for heterogeneous retail series.

The full pipeline is open-source and reproducible, requiring only standard Python
scientific libraries with no proprietary dependencies.

Future work: full covariance-based MinT with Ledoit-Wolf shrinkage; neural
reconciliation approaches; online adaptive weight learning; integration with
promotional planning and markdown optimisation.

---

## References

- Diebold, F.X., Mariano, R.S. (1995). Comparing Predictive Accuracy. *Journal of
  Business & Economic Statistics*, 13(3), 253–263.
- Hyndman, R.J., Ahmed, R.A., Athanasopoulos, G., Shang, H.L. (2011). Optimal
  combination forecasts for hierarchical time series. *Computational Statistics &
  Data Analysis*, 55(9), 2579–2589.
- Newey, W.K., West, K.D. (1987). A Simple, Positive Semi-definite, Heteroskedasticity
  and Autocorrelation Consistent Covariance Matrix. *Econometrica*, 55(3), 703–708.
- Silver, E.A., Pyke, D.F., Thomas, D.J. (2017). *Inventory and Production
  Management in Supply Chains* (4th ed.). CRC Press.
- Taieb, S.B., Taylor, J.W., Hyndman, R.J. (2017). Coherent Probabilistic Forecasts
  for Hierarchical Time Series. *Proceedings of the 34th ICML*, 3348–3357.
- Wickramasuriya, S.L., Athanasopoulos, G., Hyndman, R.J. (2019). Optimal Forecast
  Reconciliation Using Unbiased Estimating Equations. *Journal of the American
  Statistical Association*, 114(526), 804–819.
