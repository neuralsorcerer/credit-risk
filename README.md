# Credit Risk Prediction Pipeline

This repository contains an end‑to‑end **credit risk prediction** workflow implemented in a Jupyter notebook (`CreditRisk.ipynb`).  
The goal is to predict whether a loan applicant is **good** (will repay) or **bad** (may default) using the classic **German Credit** dataset from the UCI Machine Learning Repository.

> **Key features**
>
> - Rich exploratory data analysis with visualizations
> - Automated feature engineering and encoding
> - Handling of class imbalance with SMOTE
> - Model training & hyper‑parameter tuning for **XGBoost, LightGBM, CatBoost, and TabTransformer**
> - Model interpretability with SHAP values
> - Threshold optimisation & fairness diagnostics
> - Ready‑to‑ship pipeline with artefact saving

## Getting Started

Create a fresh environment and install the dependencies:

```bash
conda create -n credit-risk python=3.11
conda activate credit-risk

# core libs
pip install numpy pandas scikit-learn matplotlib seaborn shap

# gradient boosting models
pip install xgboost lightgbm catboost

# TabTransformer (PyTorch)
pip install torch tab_transformer_pytorch

# optional: jupyter
pip install notebook
```

### Run the Notebook

```bash
jupyter notebook notebook/CreditRisk.ipynb
```

The notebook will:

1. **Download** the German Credit dataset if it is not present in `./data`.
2. **Clean & preprocess** the data (encoding, scaling, variance filtering).
3. **Generate new features** such as log‑transformed loan amount and interaction terms.
4. **Balance** the training set with SMOTE‑Tomek.
5. **Train** and **tune** four candidate models with cross‑validated grid / Bayesian search.
6. **Evaluate** each model on a held‑out test set and compute metrics (Accuracy, Precision, Recall, F1).
7. **Interpret** predictions using SHAP summary & dependence plots.
8. **Persist** the best model (`CatBoost` by default) as `models/catboost_creditrisk.cbm`.

## Interpreting the Model

- Global feature importance via SHAP shows that **Duration in Months**, **Credit Amount**, and **Age** are the most influential predictors.
- Dependence plots reveal non‑linear risk patterns, e.g., credit risk rises sharply beyond 36‑month durations.
- Fairness checks across **Gender** and **Personal Status** found no statistically significant disparate impact at the chosen 0.5 probability threshold (see notebook section _Fairness Checks_).

## Production Considerations

- **Model serving**: export CatBoost model as a `cbm` file and serve via FastAPI or a CatBoost native server.
- **Data validation**: use `pandera` or `great_expectations` to enforce schema/feature ranges at inference.
- **Monitoring**: track prediction distributions and key metrics (KS, PSI, recall) to detect data drift.

## License

This project is released under the **MIT License** — see the [LICENSE](LICENSE) file for details.
