import pandas as pd
import os

URL = "https://docs.google.com/spreadsheets/d/1tAUaQrEHYgRsm1Lvrnj14HFHDwJWl0Bd9x0QePewNco/edit#gid=1977693859"
csv_url = URL.replace('/edit#gid=', '/export?format=csv&gid=')
if os.path.exists("spreadsheet.csv"):
    cached_data = pd.read_csv("spreadsheet.csv")
else:
    cached_data = pd.read_csv(csv_url)
    cached_data.to_csv("spreadsheet.csv", index=False)
models = {}

for url, filename in zip(cached_data['URL'], cached_data['Filename']):
    models[filename] = url