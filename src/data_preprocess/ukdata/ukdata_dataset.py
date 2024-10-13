import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")

# Paths
DATASETFOLDER = "src/data_preprocess/ukdata/data_root/"

PROPERTY_PATH_0001 = DATASETFOLDER + "Property_ID=EOH0001.csv"
PROPERTY_PATH_0003 = DATASETFOLDER + "Property_ID=EOH0003.csv"
PROPERTY_PATH_0005 = DATASETFOLDER + "Property_ID=EOH0005.csv"
PROPERTY_PATH_0014 = DATASETFOLDER + "Property_ID=EOH0014.csv"
PROPERTY_PATH_00018 = DATASETFOLDER + "Property_ID=EOH0018.csv"

CLEAN_UKDATA_PATH = DATASETFOLDER + "UKDATA_cleaned.csv"

# Data parameters
SAMPLE_TIME = "2min"
TIMESTAMP = "Timestamp"
USE_UTC = True
MAX_TEMP_DELTA = 15

# HVAC
hvac_df = pd.concat([pd.read_csv(HVAC_MIN_PATH_2014), pd.read_csv(HVAC_MIN_PATH_2015)])
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
series = df[(df.IndoorTemp.diff().abs().astype(float) <= MAX_TEMP_DELTA) & (df.OutdoorTemp.diff().abs().astype(float) <= MAX_TEMP_DELTA)]
df = df[df.PowerConsumption > 0]

df = df.dropna()

df.to_csv(CLEAN_NIST_PATH, index=False)

# Plotting of the cleaned data
fig, ax = plt.subplots(3)

values = df.values

power_consumption = [i[1] for i in values]
indoor_temp = [i[2] for i in values]
outdoor_temp = [i[3] for i in values]

timestamps = df[TIMESTAMP]

ax[0].plot(timestamps, power_consumption)
ax[0].set_title("Power Consumption")
ax[0].grid()

ax[1].plot(timestamps, indoor_temp, color="r")
ax[1].set_title("Indoor Temperature")
ax[1].grid()

ax[2].plot(timestamps, outdoor_temp, color="g")
ax[2].set_title("Outdoor Temperature")
ax[2].grid()

plt.subplots_adjust(hspace=1)
plt.gcf().autofmt_xdate()
plt.savefig("graph/NIST_cleaned_graph.png")