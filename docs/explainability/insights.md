# Week-3: Model Explainability (SHAP)

## Objective
The objective of this module is to integrate SHAP-based model explainability into the predictive maintenance system to improve transparency, trust, and interpretability. This helps maintenance engineers understand why a failure is predicted rather than relying on black-box outputs.

---

## Explainability Approach
SHAP (SHapley Additive exPlanations) is used to quantify the contribution of each feature to the model’s predictions. SHAP supports both global (overall feature importance) and local (instance-level) explanations, making it suitable for industrial decision-support systems.

---

## Global Feature Importance
The SHAP summary and bar plots show that the most influential features for predicting machine failure include:

- Failure lag features that capture historical failure behavior
- Pressure lag features indicating mechanical or hydraulic stress
- Rolling temperature metrics reflecting thermal overload conditions
- Vibration variability signaling wear, imbalance, or degradation

These features consistently rank highest in SHAP importance, validating their relevance.

---

## Engineering Interpretation
From an engineering perspective:

- High rolling temperature suggests prolonged thermal stress and accelerated wear
- Increased vibration variance indicates mechanical instability or bearing issues
- Pressure fluctuations reflect subsystem instability
- Lag-based features enable early failure detection

These interpretations align with real-world industrial failure mechanisms.

---

## Local Explainability
SHAP force and waterfall visualizations explain individual predictions by showing how specific features push the model toward a failure or non-failure outcome. This enables targeted maintenance actions.

---

## Business Impact
The explainability layer improves trust in AI-driven maintenance, reduces unplanned downtime, and supports confident engineering decision-making.

---

## Explainability Outcome
The SHAP explainability module transforms the predictive model from a black-box system into an interpretable decision-support tool that bridges data science outputs with actionable engineering insights.

---

### Completion Status
- Global feature importance analyzed
- Key failure-driving features identified
- Engineering interpretations documented
- Explainability artifacts organized
