import requests

url = "http://127.0.0.1:5000/predict"

data = {
    "rolling_mean_temperature": 78.5,
    "rolling_std_vibration": 0.12,
    "pressure_trend": 101.3
}

response = requests.post(url, json=data)

print(response.status_code)
print(response.json())
