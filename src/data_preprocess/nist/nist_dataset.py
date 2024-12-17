import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Paths
DATASETFOLDER = "src/data_preprocess/nist/data_root/"
HVAC_MIN_PATH_2014 = DATASETFOLDER + "HVAC-minute-2014.csv"
HVAC_MIN_PATH_2015 = DATASETFOLDER + "HVAC-minute-2015.csv"

INDENV_MIN_PATH_2014 = DATASETFOLDER + "IndEnv-minute-2014.csv"
INDENV_MIN_PATH_2015 = DATASETFOLDER + "IndEnv-minute-2015.csv"

OUTENV_MIN_PATH_2014 = DATASETFOLDER + "OutEnv-minute-2014.csv"
OUTENV_MIN_PATH_2015 = DATASETFOLDER + "OutEnv-minute-2015.csv"

CLEAN_NIST_PATH = "src/data_preprocess/dataset/NIST_cleaned.csv"

# Data parameters
SAMPLE_TIME = "15min"
TIMESTAMP = "Timestamp"
USE_UTC = True
MAX_TEMP_DELTA = 15

# HVAC
hvac_df = pd.read_csv(HVAC_MIN_PATH_2015) #pd.concat([pd.read_csv(HVAC_MIN_PATH_2014), pd.read_csv(HVAC_MIN_PATH_2015)])
hvac_df = hvac_df[[TIMESTAMP, "HVAC_HeatPumpIndoorUnitPower", "HVAC_HeatPumpOutdoorUnitPower"]]
hvac_df["PowerConsumption"] = hvac_df.HVAC_HeatPumpIndoorUnitPower + hvac_df.HVAC_HeatPumpOutdoorUnitPower
hvac_df = hvac_df.drop(["HVAC_HeatPumpIndoorUnitPower", "HVAC_HeatPumpOutdoorUnitPower"], axis="columns")
hvac_df.Timestamp = pd.to_datetime(hvac_df.Timestamp, utc=USE_UTC)
hvac_df = hvac_df.resample(SAMPLE_TIME, on=TIMESTAMP).mean().reset_index()

# Indoor
indoor_df = pd.concat([pd.read_csv(INDENV_MIN_PATH_2014), pd.read_csv(INDENV_MIN_PATH_2015)])
indoor_df = indoor_df[[TIMESTAMP, "IndEnv_RoomTempBasementNW", "IndEnv_RoomTempBasementNE","IndEnv_RoomTempBasementSE", "IndEnv_RoomTempBasementSW"]]
indoor_df["IndoorTemp"] = indoor_df[["IndEnv_RoomTempBasementNW", "IndEnv_RoomTempBasementNE","IndEnv_RoomTempBasementSE", "IndEnv_RoomTempBasementSW"]].mean(axis="columns")
indoor_df = indoor_df.drop(["IndEnv_RoomTempBasementNW", "IndEnv_RoomTempBasementNE","IndEnv_RoomTempBasementSE", "IndEnv_RoomTempBasementSW"], axis="columns")
indoor_df.Timestamp = pd.to_datetime(indoor_df.Timestamp, utc=USE_UTC)
indoor_df = indoor_df.resample(SAMPLE_TIME, on=TIMESTAMP).last().reset_index()

# Outdoor
outdoor_df = pd.concat([pd.read_csv(OUTENV_MIN_PATH_2014), pd.read_csv(OUTENV_MIN_PATH_2015)])
outdoor_df = outdoor_df[[TIMESTAMP, "OutEnv_OutdoorAmbTemp"]]
outdoor_df.Timestamp = pd.to_datetime(outdoor_df.Timestamp, utc=USE_UTC)
outdoor_df = outdoor_df.resample(SAMPLE_TIME, on=TIMESTAMP).last().reset_index()
outdoor_df = outdoor_df.rename(columns={"OutEnv_OutdoorAmbTemp": "OutdoorTemp"})

df = hvac_df
df = df.join(indoor_df.set_index(TIMESTAMP), on=TIMESTAMP)
df = df.join(outdoor_df.set_index(TIMESTAMP), on=TIMESTAMP)

df = df[(df.IndoorTemp >= 10) & (df.IndoorTemp <= 30)]
df = df[(df.OutdoorTemp >= -50) & (df.OutdoorTemp <= 50)]
df = df[(df.IndoorTemp.diff().abs().astype(float) <= MAX_TEMP_DELTA) & (df.OutdoorTemp.diff().abs().astype(float) <= MAX_TEMP_DELTA)]
df = df[df.PowerConsumption > 0]

df = df.dropna()

df.to_csv(CLEAN_NIST_PATH, index=False)

values = df.values

power_consumption = [i[1] for i in values]
indoor_temp = [i[2] for i in values]
outdoor_temp = [i[3] for i in values]

timestamps = df[TIMESTAMP]

fig = make_subplots(rows=3, cols=1, shared_xaxes=True, subplot_titles=("Power Consumption", "Indoor Temperature", "Outdoor Temperature"))

# Power Consumption
fig.add_trace(go.Scatter(x=df[TIMESTAMP], y=df["PowerConsumption"], mode='lines', name='Power Consumption'), row=1, col=1)

# Indoor Temperature
fig.add_trace(go.Scatter(x=df[TIMESTAMP], y=df["IndoorTemp"], mode='lines', name='Indoor Temperature', line=dict(color='red')), row=2, col=1)

# Outdoor Temperature
fig.add_trace(go.Scatter(x=df[TIMESTAMP], y=df["OutdoorTemp"], mode='lines', name='Outdoor Temperature', line=dict(color='green')), row=3, col=1)

fig.update_layout(height=800, width=800, title_text="NIST Cleaned Graphs", showlegend=False)
fig.update_xaxes(title_text="Timestamps")
fig.update_yaxes(title_text="Values")

fig.write_image("graph/NIST_cleaned_graph.png")