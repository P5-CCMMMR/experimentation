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
PROPERTY_PATH_0018 = DATASETFOLDER + "Property_ID=EOH0018.csv"

CLEAN_UKDATA_PATH = DATASETFOLDER + "UKDATA_cleaned.csv"

# Data parameters
SAMPLE_TIME = "15min"
TIMESTAMP = "Timestamp"
USE_UTC = True
MAX_TEMP_DELTA = 15

# HVAC
# energy_df = pd.concat([pd.read_csv(PROPERTY_PATH_0001), pd.read_csv(PROPERTY_PATH_0003), pd.read_csv(PROPERTY_PATH_0005), pd.read_csv(PROPERTY_PATH_0014), pd.read_csv(PROPERTY_PATH_0018)])
energy_df = pd.read_csv(PROPERTY_PATH_0001)
energy_df = energy_df[[TIMESTAMP, "TotalPowerConsumption"]]
energy_df.Timestamp = pd.to_datetime(energy_df.Timestamp, utc=USE_UTC)
energy_df = energy_df.resample(SAMPLE_TIME, on=TIMESTAMP).mean().reset_index()

# Consumption in Watt
watt_energy_df = pd.read_csv(PROPERTY_PATH_0001)
watt_energy_df = watt_energy_df[[TIMESTAMP, "PowerConsumption"]]
watt_energy_df.Timestamp = pd.to_datetime(watt_energy_df.Timestamp, utc=USE_UTC)
watt_energy_df = watt_energy_df.resample(SAMPLE_TIME, on=TIMESTAMP).last().reset_index()

# Indoor
# indoor_df = pd.concat([pd.read_csv(PROPERTY_PATH_0001), pd.read_csv(PROPERTY_PATH_0003), pd.read_csv(PROPERTY_PATH_0005), pd.read_csv(PROPERTY_PATH_0014), pd.read_csv(PROPERTY_PATH_0018)])
indoor_df = pd.read_csv(PROPERTY_PATH_0001)
indoor_df = indoor_df[[TIMESTAMP, "IndoorTemp"]]
indoor_df.Timestamp = pd.to_datetime(indoor_df.Timestamp, utc=USE_UTC)
indoor_df = indoor_df.resample(SAMPLE_TIME, on=TIMESTAMP).last().reset_index()



# Outdoor
# outdoor_df = pd.concat([pd.read_csv(PROPERTY_PATH_0001), pd.read_csv(PROPERTY_PATH_0003), pd.read_csv(PROPERTY_PATH_0005), pd.read_csv(PROPERTY_PATH_0014), pd.read_csv(PROPERTY_PATH_0018)])
outdoor_df = pd.read_csv(PROPERTY_PATH_0001)
outdoor_df = outdoor_df[[TIMESTAMP, "OutdoorTemp"]]
outdoor_df.Timestamp = pd.to_datetime(outdoor_df.Timestamp, utc=USE_UTC)
outdoor_df = outdoor_df.resample(SAMPLE_TIME, on=TIMESTAMP).last().reset_index()

df = energy_df
df = df.join(watt_energy_df.set_index(TIMESTAMP), on=TIMESTAMP)
df = df.join(indoor_df.set_index(TIMESTAMP), on=TIMESTAMP)
df = df.join(outdoor_df.set_index(TIMESTAMP), on=TIMESTAMP)

df = df[(df.IndoorTemp >= 10) & (df.IndoorTemp <= 30)]
df = df[(df.OutdoorTemp >= -50) & (df.OutdoorTemp <= 50)]
series = df[(df.IndoorTemp.diff().abs().astype(float) <= MAX_TEMP_DELTA) & (df.OutdoorTemp.diff().abs().astype(float) <= MAX_TEMP_DELTA)]
df = df[(df.PowerConsumption <= 5000)]
df = df.drop(columns=["TotalPowerConsumption"])

df = df.dropna()

df.to_csv(CLEAN_UKDATA_PATH, index=False)

# Plotting of the cleaned data
fig, ax = plt.subplots(4)

values = df.values

total_power_consumption = [i[1] for i in values]
indoor_temp = [i[2] for i in values]
outdoor_temp = [i[3] for i in values]
#power_consumption = [i[4] for i in values]

timestamps = df[TIMESTAMP]

ax[0].plot(timestamps, total_power_consumption)
ax[0].set_title("Total Power Consumption of Heat Pump")
ax[0].grid()

ax[1].plot(timestamps, indoor_temp, color="r")
ax[1].set_title("Indoor Temperature")
ax[1].set_ylim(min(outdoor_temp), max(outdoor_temp))
ax[1].grid()

ax[2].plot(timestamps, outdoor_temp, color="g")
ax[2].set_title("Outdoor Temperature")
ax[2].set_ylim(min(outdoor_temp), max(outdoor_temp))
ax[2].grid()

#ax[3].plot(timestamps, power_consumption)
#ax[3].set_title("Power Consumption of Heat Pump")
#ax[3].grid()


plt.subplots_adjust(hspace=1)
plt.gcf().autofmt_xdate()
plt.savefig("graph/UKDATA_cleaned_graph.png")