# Benchmark Evaluation Summary Report

Evaluated over **2 identical seed episodes**.

### 📊 Metric Performance Matrix (Mean ± Std)

| metric                    | marl_mean_std   | baseline_mean_std   | pct_improvement   |   p_value |
|:--------------------------|:----------------|:--------------------|:------------------|----------:|
| task_completion_rate      | 0.260 ± 0.260   | 0.411 ± 0.219       | -36.86%           |       0.5 |
| coordination_efficiency   | 0.314 ± 0.024   | 0.556 ± 0.016       | -43.62%           |       0.5 |
| resource_utilization_rate | 0.683 ± 0.000   | 0.667 ± 0.040       | +2.40%            |       1   |
| avg_response_time         | 28.035 ± 21.965 | 17.475 ± 1.125      | +60.43%           |       1   |

### 📈 Key Insights
- **MARL Task Completion Rate:** 0.260 vs Baseline 0.411
- **MARL Coordination Efficiency:** 0.314 vs Baseline 0.556
