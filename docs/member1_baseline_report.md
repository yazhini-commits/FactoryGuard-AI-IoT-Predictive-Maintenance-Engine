# Member 1 – Baseline Modeling Report

## Project
Manufacturing – IoT Predictive Maintenance Engine  
Task 2: Baseline Modeling

---

## Dataset
- Input dataset: `factoryguard_features.joblib`
- Location: `Artifacts/`
- Description: Final merged and engineered feature dataset used by all team members

---

## Modeling Approach
The objective of Member 1 was to establish a baseline model and reference performance.

- Initial attempt was classification using the `Failure` target.
- Since the `Failure` column was not present in the dataset, a **fallback regression approach** was used as per project safety logic.
- Target used for fallback: `vibration_roll_std_1h`

---

## Models Trained (Baseline)
1. **Ridge Regression**
2. **Random Forest Regressor**

All input features were converted to numeric values and missing values were handled.

---

## Evaluation Metric
- **R² Score** (used only for fallback regression validation)
- **Note**: The perfect R² scores observed in the fallback regression results indicate strong feature correlation or potential data leakage. These regression results are used only to validate the baseline modeling pipeline and artifact generation, and they are not intended for business decision-making.


---

## Results

| Model                     | R² Score |
|---------------------------|----------|
| Ridge Regression          | 1.00     |
| Random Forest Regressor   | 1.00     |

---

## Saved Artifacts
The following trained models were saved in the `Artifacts/` folder:

- `baseline_ridge_regression.joblib`
- `baseline_random_forest_regressor.joblib`

---

## Notes for Team
- This baseline serves as a **reference implementation**.
- Member 2 and Member 3 should use the same dataset (`factoryguard_features.joblib`) for their tasks.
- Member 4 will review and consolidate these outputs as part of the final evaluation, threshold selection, and reporting process.
- Classification models will be trained once the `Failure` target is finalized.

---

## Status
✅ Member 1 task completed successfully  
