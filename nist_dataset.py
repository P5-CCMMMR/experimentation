import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")

# Paths
DATASETFOLDER = "dataset/"
HVAC_MIN_PATH = DATASETFOLDER + "HVAC-minute.csv"
INDENV_MIN_PATH = DATASETFOLDER + "IndEnv-minute.csv"
OUTENV_MIN_PATH = DATASETFOLDER + "OutEnv-minute.csv"

SAMPLE_TIME = "15min"
TIMESTAMP = "Timestamp"
USE_UTC = True
COL_AXIS = 1
MAX_TEMP_DELTA = 15

# HVAC
hvac_df = pd.read_csv(HVAC_MIN_PATH)
hvac_df = hvac_df[[TIMESTAMP, "HVAC_HeatPumpIndoorUnitPower", "HVAC_HeatPumpOutdoorUnitPower"]]
hvac_df["PowerConsumption"] = hvac_df.HVAC_HeatPumpIndoorUnitPower + hvac_df.HVAC_HeatPumpOutdoorUnitPower
hvac_df = hvac_df.drop(["HVAC_HeatPumpIndoorUnitPower", "HVAC_HeatPumpOutdoorUnitPower"], axis=COL_AXIS)
hvac_df.Timestamp = pd.to_datetime(hvac_df.Timestamp, utc=USE_UTC)
hvac_df = hvac_df.resample(SAMPLE_TIME, on=TIMESTAMP).mean().reset_index()

# Indoor
indoor_df = pd.read_csv(INDENV_MIN_PATH)
indoor_df = indoor_df[[TIMESTAMP, "IndEnv_RoomTempBasementNW", "IndEnv_RoomTempBasementNE","IndEnv_RoomTempBasementSE", "IndEnv_RoomTempBasementSW"]]
indoor_df["IndoorTemp"] = indoor_df[["IndEnv_RoomTempBasementNW", "IndEnv_RoomTempBasementNE","IndEnv_RoomTempBasementSE", "IndEnv_RoomTempBasementSW"]].mean(axis=COL_AXIS)
indoor_df = indoor_df.drop(["IndEnv_RoomTempBasementNW", "IndEnv_RoomTempBasementNE","IndEnv_RoomTempBasementSE", "IndEnv_RoomTempBasementSW"], axis=COL_AXIS)
indoor_df.Timestamp = pd.to_datetime(indoor_df.Timestamp, utc=USE_UTC)
indoor_df = indoor_df.resample(SAMPLE_TIME, on=TIMESTAMP).last().reset_index()

# Outdoor
outdoor_df = pd.read_csv(OUTENV_MIN_PATH)
outdoor_df = outdoor_df[[TIMESTAMP, "OutEnv_OutdoorAmbTemp"]]
outdoor_df.Timestamp = pd.to_datetime(outdoor_df.Timestamp, utc=USE_UTC)
outdoor_df = outdoor_df.resample(SAMPLE_TIME, on=TIMESTAMP).last().reset_index()
outdoor_df = outdoor_df.rename(columns={"OutEnv_OutdoorAmbTemp": "OutdoorTemp"})

df = hvac_df
df = df.join(indoor_df.set_index(TIMESTAMP), on=TIMESTAMP)
df = df.join(outdoor_df.set_index(TIMESTAMP), on=TIMESTAMP)

df = df[(df.IndoorTemp >= 10) & (df.IndoorTemp <= 30)]
df = df[(df.OutdoorTemp >= -50) & (df.OutdoorTemp <= 50)]

df = df[df.IndoorTemp.diff().abs() <= MAX_TEMP_DELTA & (df.OutdoorTemp.diff().abs() <= MAX_TEMP_DELTA)]

df = df.dropna()

df.to_csv("NIST_cleaned.csv", index=False)

# Plotting of the cleaned data
fig, ax = plt.subplots(3)

values = df.values
power_consumption = [i[1] for i in values]
indoor_temp = [i[2] for i in values]
outdoor_temp = [i[3] for i in values]

ax[0].plot(power_consumption)
ax[0].set_title("Power Consumption")
ax[0].grid()

ax[1].plot(indoor_temp, color="r")
ax[1].set_title("Indoor Temperature")
ax[1].grid()

ax[2].plot(outdoor_temp, color="g")
ax[2].set_title("Outdoor Temperature")
ax[2].grid()

plt.subplots_adjust(hspace=1)
plt.savefig("NIST_cleaned_graph.png")
