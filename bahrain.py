import fastf1
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

#location of your f1_cache folder below
fastf1.Cache.enable_cache("f1_cache")





