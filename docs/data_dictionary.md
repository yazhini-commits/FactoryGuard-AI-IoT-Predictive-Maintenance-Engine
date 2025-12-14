# Data Dictionary – FactoryGuard AI

## Dataset: IoT Sensor Readings

### machine_id
- Description: Unique identifier for each robotic arm
- Type: Categorical (String)
- Example: M001, M002

### timestamp
- Description: Time when sensor reading was recorded
- Type: Datetime (YYYY-MM-DD HH:MM:SS)
- Frequency: 1 minute

### vibration
- Description: RMS vibration level of machine
- Unit: mm/s
- Type: Float

### temperature
- Description: Operating temperature of motor
- Unit: Degree Celsius (°C)
- Type: Float

### pressure
- Description: Hydraulic pressure reading
- Unit: kPa
- Type: Float

### failure
- Description: Machine failure indicator
- Type: Binary
- Values:
  - 0 → Normal operation
  - 1 → Failure detected
