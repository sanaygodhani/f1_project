import fastf1
import pandas as pd
import numpy as np
import requests
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split


fastf1.Cache.enable_cache("f1_cache")


session = fastf1.get_session(2024, "Japan", "R")
session.load()
laps = session.laps[["Driver", "LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]].copy()
laps.dropna(inplace=True)

for timings in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]:
    laps[f'{timings} (s)'] = laps[timings].dt.total_seconds()

sector_avg = laps.groupby("Driver")[["Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)"]].mean().reset_index()

quali_2025 = pd.DataFrame({
    "Driver": ["VER", "NOR", "PIA", "LEC", "RUS", "HAM", "GAS", "ALO", "TSU", "SAI", "HUL", "OCO", "STR"],
    "QualifyingTime (s)": [86.983, 86.995, 87.027, 87.299, 87.318, 87.610, 87.822, 87.897, 88.000, 87.836, 88.570, 88.696, 89.271]
})
#rain conditions possible giving wet performance rating to driver -- see wetperformance.py
# some drivers n/a
wet_timings = {
    "VER": 0.975196, 
    "HAM": 0.976464,  
    "LEC": 0.975862,  
    "NOR": 0.978179,  
    "ALO": 0.972655,  
    "RUS": 0.968678,  
    "SAI": 0.978754,  
    "TSU": 0.996338,  
    "OCO": 0.981810,  
    "GAS": 0.978832,  
    "STR": 0.979857   
}

quali_2025["WetPerformance"] = quali_2025["Driver"].map(wet_timings)

API_KEY = "0d3b7f76b09afefe4f5253866dd3ab15"
weather_url = f"http://api.openweathermap.org/data/2.5/forecast?lat=34.8823&lon=136.5845&appid={API_KEY}&units=metric"

response = requests.get(weather_url)
weather_data = response.json()
# print(weather_data)

forecast_time = "2025-04-05 14:00:00"
forecast_data = None
for data in weather_data["list"]:
    if data["dt_txt"] == forecast_time:
        forecast_data = data
        break

if forecast_data:
    rain = forecast_data["pop"]
    temp = forecast_data["main"]["temp"]
else:
    rain = 0
    temp = 20

merged = quali_2025.merge(sector_avg, left_on="Driver", right_on="Driver", how="left")
merged["ChanceForRain"] = rain
merged["Temperature"] = temp
X = merged[["QualifyingTime (s)", "Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)",  "WetPerformance", "ChanceForRain", "Temperature"]].fillna(0)

y = merged.merge(laps.groupby("Driver")["LapTime (s)"].mean(), left_on="Driver", right_index=True)["LapTime (s)"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1)
model.fit(X_train, y_train)

prediction = model.predict(X)
quali_2025["PredictedTime (s)"] = prediction
quali_2025.sort_values(by="PredictedTime (s)")

print("Predicted positions taking weather conditions into account!: ")
print(quali_2025[["Driver", "PredictedTime (s)"]])


y_pred = model.predict(X_test)
print(f"Error: {mean_absolute_error(y_test, y_pred):.2f} seconds")

