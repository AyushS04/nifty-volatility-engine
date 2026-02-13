import pandas as pd
from src.surface_builder import VolatilitySurface

options = pd.read_csv("data/options_sample.csv", parse_dates=['date','expiry'])
spot = pd.read_csv("data/spot_sample.csv", parse_dates=['date'])

surface = VolatilitySurface()
iv_surface = surface.build_surface(options, spot)

print(iv_surface.head())
