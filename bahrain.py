import fastf1
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import requests
#location of your f1_cache folder below
fastf1.Cache.enable_cache("f1_cache")
import matplotlib.pyplot as plt

session = fastf1.get_session(2024, "Bahrain", "R")
session.load()

laps = session.laps[["Driver", "LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]].copy()
laps.dropna(inplace=True)

for timings in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]:
    laps[f"{timings} (s)"] = laps[timings].dt.total_seconds()

sector_avg = laps.groupby("Driver")[["Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)"]].mean().reset_index()


quali = pd.DataFrame({
    "Driver": ["VER", "NOR", "PIA", "LEC", "RUS", "HAM", "GAS", "ALO", "TSU", "SAI", "HUL", "OCO", "STR"],
    "QualifyingTime (s)": [90.423, 90.267, 89.841, 90.175, 90.009, 90.772, 90.216, 91.886, 91.303, 90.680, 92.067, 91.886, 92.283]
})

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

quali["WetPerformance"] = quali["Driver"].map(wet_timings)

drivers_ranking = {
    "VER": 61,
    "NOR": 62,    # high! McLaren is strong
    "PIA": 49,    # high! McLaren is strong
    "LEC": 20,
    "RUS": 45,
    "HAM": 15,
    "GAS": 0,
    "ALO": 0,
    "TSU": 3,
    "SAI": 1,
    "HUL": 6,
    "OCO": 10,
    "STR": 10
}

quali["SeasonPoints"] = quali["Driver"].map(drivers_ranking)


API_KEY = "0d3b7f76b09afefe4f5253866dd3ab15"
weather_url = f"http://api.openweathermap.org/data/2.5/forecast?lat=34.8823&lon=136.5845&appid={API_KEY}&units=metric"

response = requests.get(weather_url)
weather_data = response.json()
# print(weather_data)

forecast_time = "2025-04-30 15:00:00"
forecast_data = None
for data in weather_data["list"]:
    if data["dt_txt"] == forecast_time:
        forecast_data = data
        break


rain = forecast_data["pop"] if forecast_data else 0
temp = forecast_data["main"]["temp"] if forecast_data else 20

merged = quali.merge(sector_avg, how="left", on="Driver")

merged["ChanceForRain"] = rain
merged["Temperature"] = temp
X = merged[["QualifyingTime (s)","Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)", "SeasonPoints", "WetPerformance",
            "ChanceForRain", "Temperature"]].fillna(0)

y = merged.merge(sector_avg.groupby("Driver")[["LapTime (s)"]].mean(), left_on="Driver", right_index=True)["LapTime (s)"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1)

model.fit(X_train, y_train)

prediction = model.predict(X)
merged["Prediction (s)"] = prediction
merged.sort_values(by="Prediction (s)")

print("Results: ")
print(merged[["Driver", "Prediction (s)"]])

y_pred = model.predict(X_test)

print(f"Model Error: {mean_absolute_error(y_test, y_pred):.2f} seconds")

feature_importance = model.feature_importances_
features = X.columns

plt.figure(figsize=(8,5))
plt.barh(features, feature_importance, color='skyblue')
plt.xlabel("Importance")
plt.title("Feature Importance in Race Time Prediction")
plt.tight_layout()
plt.show()

