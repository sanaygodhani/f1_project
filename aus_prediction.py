import fastf1
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib

fastf1.Cache.enable_cache("f1_cache")


session = fastf1.get_session(2024, "Australia", "R") #last year aus gp was 3rd race
session.load()

lap_timings = session.laps[["Driver", "LapTime"]].copy()
lap_timings.dropna(subset=["LapTime"], inplace=True)#clearing dsq lap times
lap_timings["LapTime (s)"] = lap_timings["LapTime"].dt.total_seconds()#creating new field which follows the standard, i.e, time in seconds
quali_2025 = pd.DataFrame({
    "Driver": ["Lando Norris", "Oscar Piastri", "Max Verstappen", "George Russell", "Yuki Tsunoda",
               "Alexander Albon", "Charles Leclerc", "Lewis Hamilton", "Pierre Gasly", "Carlos Sainz",
                "Isack Hadjar","Fernando Alonso", "Lance Stroll", "Jack Doohan", "Gabriel Bortoleto", 
                "Kimi Antonelli", "Nico Hülkenberg", "Liam Lawson", "Esteban Ocon", "Oliver Bearman"],
    "QualifyingTime (s)": [75.096, 75.180, 75.481, 75.546, 75.670,
                           75.737, 75.755, 75.973, 75.980, 76.062, 
                           76.175, 76.453, 76.483, 76.863, 77.52, 
                           76.525, 76.579, 77.094, 77.147, None] 
})
#current drivers and their mapping below
driver_mapping = {
    "Oscar Piastri": "PIA", "George Russell": "RUS", "Lando Norris": "NOR", "Max Verstappen": "VER",
    "Lewis Hamilton": "HAM", "Charles Leclerc": "LEC", "Isack Hadjar": "HAD", "Andrea Kimi Antonelli": "ANT",
    "Yuki Tsunoda": "TSU", "Alexander Albon": "ALB", "Esteban Ocon": "OCO", "Nico Hülkenberg": "HUL",
    "Fernando Alonso": "ALO", "Lance Stroll": "STR", "Carlos Sainz Jr.": "SAI", "Pierre Gasly": "GAS",
    "Oliver Bearman": "BEA", "Jack Doohan": "DOO", "Gabriel Bortoleto": "BOR", "Liam Lawson": "LAW"
}

quali_2025["DriverCode"] = quali_2025["Driver"].map(driver_mapping)
#merging the two dataset
final_data = quali_2025.merge(lap_timings, left_on="DriverCode", right_on="Driver")

X = final_data[["QualifyingTime (s)"]]
y = final_data["LapTime (s)"]  #predictions based on last year's positions


if X.shape[0] == 0:
    raise ValueError("Dataset transfer error")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1025)
model.fit(X_train, y_train)

prediction = model.predict(quali_2025[["QualifyingTime (s)"]])
quali_2025["Prediction (s)"] = prediction
quali_2025.sort_values(by="Prediction (s)")

print("Results for Australian GP:\n")
print(quali_2025[["Driver", "Prediction (s)"]])
y_pred = model.predict(X_test)
print(f"\n🔍 Model Error (MAE): {mean_absolute_error(y_test, y_pred):.2f} seconds")