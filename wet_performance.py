import fastf1
import numpy as np
import pandas as pd

fastf1.Cache.enable_cache('f1_cache')  

session_23 = fastf1.get_session(2023, "Canada", "R")
session_23.load()
session_22 = fastf1.get_session(2022, "Canada", "R")
session_22.load()

lap_23 = session_23.laps[["Driver", "LapTime"]].copy()
lap_22 = session_22.laps[["Driver", "LapTime"]].copy()

lap_23.dropna(inplace=True)
lap_22.dropna(inplace=True)

lap_23["LapTime (s)"] = lap_23["LapTime"].dt.total_seconds()
lap_22["LapTime (s)"] = lap_22["LapTime"].dt.total_seconds()

avg_23 = lap_23.groupby("Driver")["LapTime (s)"].mean().reset_index()
avg_22 = lap_22.groupby("Driver")["LapTime (s)"].mean().reset_index()

merged = pd.merge(avg_22, avg_23, on="Driver", suffixes=('_2022', '_2023'))
merged["LapTimeDiff (s)"] = merged["LapTime (s)_2023"] - merged["LapTime (s)_2022"]

merged["PerformanceChange (%)"] = merged["LapTimeDiff (s)"] / merged["LapTime (s)_2022"]

merged["WetPerformance"] = 1 + (merged["PerformanceChange (%)"])

print(merged)