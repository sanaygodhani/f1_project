import fastf1
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error


fastf1.Cache.enable_cache("f1_cache")

prev_session =  fastf1.get_session(2024, "China", "R")
prev_session.load()


lap_timings = prev_session.laps[["Driver", "LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]].copy()
lap_timings.dropna(inplace=True)
for timings in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]:
    lap_timings[f'{timings} (s)'] = lap_timings[timings].dt.total_seconds()



sector_avg = lap_timings.groupby("Driver")[["Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)"]].mean().reset_index()

quali_2025 = pd.DataFrame({
    "Driver": ["Oscar Piastri", "George Russell", "Lando Norris", "Max Verstappen", "Lewis Hamilton",
               "Charles Leclerc", "Isack Hadjar", "Andrea Kimi Antonelli", "Yuki Tsunoda", "Alexander Albon",
               "Esteban Ocon", "Nico Hülkenberg", "Fernando Alonso", "Lance Stroll", "Carlos Sainz Jr.",
               "Pierre Gasly", "Oliver Bearman", "Jack Doohan", "Gabriel Bortoleto", "Liam Lawson"],
    "QualifyingTime (s)": [90.641, 90.723, 90.793, 90.817, 90.927,
                           91.021, 91.079, 91.103, 91.638, 91.706,
                           91.625, 91.632, 91.688, 91.773, 91.840,
                           91.992, 92.018, 92.092, 92.141, 92.174]
})

driver_mapping = {
    "Oscar Piastri": "PIA", "George Russell": "RUS", "Lando Norris": "NOR", "Max Verstappen": "VER",
    "Lewis Hamilton": "HAM", "Charles Leclerc": "LEC", "Isack Hadjar": "HAD", "Andrea Kimi Antonelli": "ANT",
    "Yuki Tsunoda": "TSU", "Alexander Albon": "ALB", "Esteban Ocon": "OCO", "Nico Hülkenberg": "HUL",
    "Fernando Alonso": "ALO", "Lance Stroll": "STR", "Carlos Sainz Jr.": "SAI", "Pierre Gasly": "GAS",
    "Oliver Bearman": "BEA", "Jack Doohan": "DOO", "Gabriel Bortoleto": "BOR", "Liam Lawson": "LAW"
}

quali_2025["DriverCode"] = quali_2025["Driver"].map(driver_mapping)
#merging the two dataset
final_data = quali_2025.merge(sector_avg, left_on="DriverCode", right_on="Driver", how="left")

X = final_data[["QualifyingTime (s)", "Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)"]].fillna(0)
y = lap_timings.groupby("Driver")["LapTime (s)"].mean().reset_index()["LapTime (s)"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1)
model.fit(X_train, y_train)

prediction = model.predict(X)

quali_2025["Prediction (s)"] = prediction
quali_2025.sort_values(by="Prediction (s)")
print("Chinese GP Winner:\n")
print(quali_2025[["Driver", "Prediction (s)"]])

y_pred = model.predict(X_test)
print(f"\n🔍 Model Error (MAE): {mean_absolute_error(y_test, y_pred):.2f} seconds")