# Week 4 – Model Deployment & Real-Time Inference

## Objective
The goal of Week 4 is to demonstrate that the predictive maintenance model developed in previous weeks can be deployed in a real-time production environment with low latency and high reliability.

This phase validates the transition from an experimental ML model to an industry-ready system.


## Deployment Architecture Overview

The deployment pipeline consists of the following components:

1. **Sensor Data Input**
   - Real-time readings from vibration, temperature, and pressure sensors
   - Data received in JSON format

2. **Feature Preprocessing Pipeline**
   - Ensures consistency between training and inference
   - Handles scaling, rolling statistics, and lag features

3. **Trained ML Model**
   - Serialized using `joblib`
   - Loaded once during API startup

4. **Flask REST API**
   - Provides real-time prediction via HTTP endpoint

---

## API Workflow

### Endpoint

### Input Payload (Example)
```json
{
  "vibration": 0.82,
  "temperature": 78.5,
  "pressure": 310.2
}
{
  "failure_probability": 0.87,
  "prediction": "Failure"
}