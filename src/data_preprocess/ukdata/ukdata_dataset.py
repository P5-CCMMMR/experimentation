import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Paths
DATASETFOLDER = "src/data_preprocess/ukdata/data_root/"

PROPERTY_PATH_0001 = DATASETFOLDER + "Property_ID=EOH0001.csv"

CLEAN_UKDATA_PATH = "src/data_preprocess/dataset/UKDATA_cleaned.csv"

# Data parameters
SAMPLE_TIME = "15min"
TIMESTAMP = "Timestamp"
USE_UTC = True
MAX_TEMP_DELTA = 15

# Consumption in Watt
watt_energy_df = pd.read_csv(PROPERTY_PATH_0001)
watt_energy_df = watt_energy_df[[TIMESTAMP, "PowerConsumption"]]
watt_energy_df.Timestamp = pd.to_datetime(watt_energy_df.Timestamp, utc=USE_UTC)
watt_energy_df = watt_energy_df.resample(SAMPLE_TIME, on=TIMESTAMP).last().reset_index()

# Indoor
indoor_df = pd.read_csv(PROPERTY_PATH_0001)
indoor_df = indoor_df[[TIMESTAMP, "IndoorTemp"]]
indoor_df.Timestamp = pd.to_datetime(indoor_df.Timestamp, utc=USE_UTC)
indoor_df = indoor_df.resample(SAMPLE_TIME, on=TIMESTAMP).last().reset_index()

# Outdoor
outdoor_df = pd.read_csv(PROPERTY_PATH_0001)
outdoor_df = outdoor_df[[TIMESTAMP, "OutdoorTemp"]]
outdoor_df.Timestamp = pd.to_datetime(outdoor_df.Timestamp, utc=USE_UTC)
outdoor_df = outdoor_df.resample(SAMPLE_TIME, on=TIMESTAMP).last().reset_index()

df = watt_energy_df
df = df.join(indoor_df.set_index(TIMESTAMP), on=TIMESTAMP)
df = df.join(outdoor_df.set_index(TIMESTAMP), on=TIMESTAMP)

df = df[(df.IndoorTemp >= 10) & (df.IndoorTemp <= 30)]
df = df[(df.OutdoorTemp >= -50) & (df.OutdoorTemp <= 50)]
series = df[(df.IndoorTemp.diff().abs().astype(float) <= MAX_TEMP_DELTA) & (df.OutdoorTemp.diff().abs().astype(float) <= MAX_TEMP_DELTA)]
df = df[(df.PowerConsumption <= 5000)]

df = df.dropna()

df.to_csv(CLEAN_UKDATA_PATH, index=False)

values = df.values

power_consumption = [i[1] for i in values]
indoor_temp = [i[2] for i in values]
outdoor_temp = [i[3] for i in values]

timestamps = df[TIMESTAMP]

fig = make_subplots(rows=3, cols=1, shared_xaxes=True, subplot_titles=("Power Consumption", "Indoor Temperature", "Outdoor Temperature"))
# Power Consumption
fig.add_trace(go.Scatter(x=timestamps, y=power_consumption, mode='lines', name='Power Consumption'), row=1, col=1)

# Indoor Temperature
fig.add_trace(go.Scatter(x=timestamps, y=indoor_temp, mode='lines', name='Indoor Temperature', line=dict(color='red')), row=2, col=1)

# Outdoor Temperature
fig.add_trace(go.Scatter(x=timestamps, y=outdoor_temp, mode='lines', name='Outdoor Temperature', line=dict(color='green')), row=3, col=1)

fig.update_layout(height=800, width=800, title_text="UKDATA Cleaned Graphs", showlegend=False)
fig.update_xaxes(title_text="Timestamps")
fig.update_yaxes(title_text="Values")

fig.write_image("graph/UKDATA_cleaned_graph.png")
