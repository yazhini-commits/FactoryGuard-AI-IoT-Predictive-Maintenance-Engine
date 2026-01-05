# FactoryGuard – AI IoT Predictive Maintenance Engine

## Project Overview
FactoryGuard is an AI-powered IoT predictive maintenance system designed to detect machine failures in advance using sensor data. The system combines robust data preprocessing, machine learning, explainable AI, and real-time deployment.

---

## Problem Statement
Unexpected machine failures cause:
- Production downtime
- Increased maintenance costs
- Safety risks

Traditional reactive maintenance is inefficient. FactoryGuard addresses this using predictive analytics.

---

## Dataset Description
- Sensor data: vibration, temperature, pressure
- Time-series structure
- Highly imbalanced failure events

---

## Week-wise Implementation Summary

### Week 1 – Data Understanding & Preprocessing
- Cleaned raw sensor data
- Handled missing values
- Standardized features
- Generated clean dataset

---

### Week 2 – Feature Engineering & Modeling
- Created lag features
- Generated rolling statistics
- Handled class imbalance
- Trained baseline and advanced models
- Evaluated using PR-AUC and precision

---

### Week 3 – Explainable AI (SHAP)
- Implemented global explainability
- Generated SHAP summary and bar plots
- Analyzed local failure cases
- Improved trust and interpretability

---

### Week 4 – Deployment & Inference
- Built REST API using Flask
- Enabled real-time predictions
- Designed scalable architecture
- Prepared production-ready model

---

## Technologies Used
- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost
- SHAP
- Flask
- Joblib

---

## Key Results
- High precision for rare failures
- Explainable predictions
- Low-latency inference
- Industry-ready deployment

---

## Business Impact
- Reduced downtime
- Cost savings
- Predictive maintenance planning
- Improved operational efficiency

---

## Future Enhancements
- Dashboard integration
- Edge deployment
- Model retraining automation
- Alerting system integration

---

## Conclusion
FactoryGuard demonstrates a complete end-to-end AI pipeline for predictive maintenance, combining accuracy, explainability, and deployability into a single robust solution.
