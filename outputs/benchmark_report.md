# VectorFlow Network Attack Forecasting — Model Benchmark Report

## 1. Overall Performance Comparison (Test Set)

| Model               |   PR-AUC |   ROC-AUC |   F1-Score |   Precision |   Recall |   Specificity |   Balanced Acc |   Threshold |   TP |   FP |   FN |   TN |
|:--------------------|---------:|----------:|-----------:|------------:|---------:|--------------:|---------------:|------------:|-----:|-----:|-----:|-----:|
| LogReg              |   0.2263 |    0.5045 |     0.1066 |      0.0635 |   0.3309 |        0.8202 |         0.5756 |      0.4538 |   46 |  678 |   93 | 3093 |
| RandomForest        |   0.0639 |    0.6856 |     0.1438 |      0.0855 |   0.4532 |        0.8213 |         0.6373 |      0.2535 |   63 |  674 |   76 | 3097 |
| XGBoost             |   0.0531 |    0.6573 |     0.0383 |      0.0308 |   0.0504 |        0.9417 |         0.496  |      0.1407 |    7 |  220 |  132 | 3551 |
| LSTM                |   0.0404 |    0.5794 |     0.0451 |      0.0329 |   0.0719 |        0.922  |         0.497  |      0.3367 |   10 |  294 |  129 | 3477 |
| Ensemble (XGB+LSTM) |   0.0496 |    0.6081 |     0.0404 |      0.0281 |   0.0719 |        0.9082 |         0.4901 |      0.2019 |   10 |  346 |  129 | 3425 |

## 2. Benchmark Visualizations

- **Executive Dashboard**: `outputs/benchmark_summary_dashboard.png`
- **Precision-Recall Curves**: `outputs/benchmark_pr_curves.png`
- **ROC Curves**: `outputs/benchmark_roc_curves.png`
- **Key Metrics Bar Chart**: `outputs/benchmark_metrics_barchart.png`
- **Confusion Matrices**: `outputs/benchmark_confusion_matrices.png`
