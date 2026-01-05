# Week 3 – Model Explainability using SHAP

## Objective
The objective of Week 3 is to interpret and explain the predictions made by the predictive maintenance model. Since machine failure prediction is a high-risk domain, understanding *why* a model predicts failure is as important as prediction accuracy.

This week focuses on **Explainable AI (XAI)** using **SHAP (SHapley Additive Explanations)**.

---

## Why Explainability is Important
- Builds trust in AI systems
- Helps engineers understand root causes of failures
- Supports preventive maintenance decisions
- Required for industrial compliance and audits

---

## Explainability Technique Used
**SHAP (SHapley Additive Explanations)**

SHAP explains model predictions by:
- Computing feature contributions for each prediction
- Showing how each feature pushes prediction higher or lower
- Providing both **global** and **local** explanations

---

## Model Used
- Final trained production model (`final_production_model.joblib`)
- Input features: vibration, temperature, pressure, lag features, rolling statistics
- Target variable: machine failure (0 = No Failure, 1 = Failure)

---

## Global Explainability

### SHAP Summary Plot
- Displays overall feature importance
- Features ranked by average impact on predictions

**Key Insights:**
- Pressure and temperature are dominant predictors
- Vibration plays a secondary but consistent role
- Lag and rolling features improve stability of predictions

📌 Output File:
- `shap_outputs/shap_summary_plot.png`

---

### SHAP Bar Plot
- Aggregated mean absolute SHAP values
- Shows top contributing features across dataset

📌 Output File:
- `shap_outputs/shap_bar_plot.png`

---

## Local Explainability (Failure Case Analysis)

Local explanations were generated for **three failure cases**.

### Waterfall Plots
- Show how each feature contributes to final prediction
- Starts from base value and adds feature contributions

📌 Output Files:
- `waterfall_failure_case_1.png`
- `waterfall_failure_case_2.png`
- `waterfall_failure_case_3.png`

---

### Force Plots
- Visualizes pushing effect of features
- Red → increases failure probability
- Blue → decreases failure probability

📌 Output Files:
- `force_failure_case_1.png`
- `force_failure_case_2.png`
- `force_failure_case_3.png`

---

## Observations from Failure Cases
- High pressure consistently increases failure probability
- Elevated temperature strongly contributes to failures
- Sudden vibration spikes act as early indicators
- Lag features help capture temporal trends

---

## Business Value
- Enables early root cause detection
- Helps maintenance teams prioritize inspections
- Reduces unplanned downtime
- Improves confidence in AI predictions

---

## Conclusion
Week 3 successfully integrated explainable AI into the predictive maintenance pipeline. Using SHAP, the model's decisions are transparent, interpretable, and actionable for industrial use cases.
