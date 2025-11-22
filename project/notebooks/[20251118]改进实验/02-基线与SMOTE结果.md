# 基线与 VIV-SMOTE 实验结果

本文件由 `src/imbalance_experiments.py` 自动生成。

## 指标说明

- `overall_r2` / `overall_rmse`: 全体样本 (190 座桥梁 + 合成样本) 的指标。
- `high_risk_r2` / `high_risk_rmse`: 高风险子集 (振幅 > 60mm) 的指标。
- `oversampling_factor`: 高风险样本在特征空间线性插值的扩增倍数。

## 实验结果表

| experiment   |   oversampling_factor |   overall_r2 |   overall_rmse |   high_risk_r2 |   high_risk_rmse |
|:-------------|----------------------:|-------------:|---------------:|---------------:|-----------------:|
| baseline     |                   1   |     0.639019 |        13.0522 |      -0.193345 |          16.1976 |
| viv_smote    |                   1.5 |     0.650618 |        12.8407 |      -0.100299 |          15.5533 |
| viv_smote    |                   2   |     0.622812 |        13.3419 |      -0.122631 |          15.7103 |
| viv_smote    |                   3   |     0.534588 |        14.8204 |      -0.298644 |          16.8971 |
