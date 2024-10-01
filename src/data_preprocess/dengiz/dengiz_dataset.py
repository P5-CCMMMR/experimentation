import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")

# Paths
DATASETFOLDER = "dataset/"
KWH_CATEGORIES = ["kWh25", "kWh50", "kWh80"] 

CLEAN_NIST_PATH = DATASETFOLDER + "dengiz_cleaned.csv"

# Data preprocess parameters
SAMPLE_TIME = "15min"
USE_UTC = True
MAX_TEMP_DELTA = 15

# Data parameters 
WEEKS = 52
HOUSE_AMOUNT = 20
KWH_AMOUNT = 3


# Paths
PREPROSS_PATH = "src/data_preprocess/"
START_PATH = PREPROSS_PATH + "dengiz/data_root"
WEEK_PATH = "Week"
HOUSE_PATH = "BT4_HH"
MIDDLE_FOLDER_PATH = "Min_Costs_Scaled_PV_0_kWp_30_Min_A"

OUTPUT_PATH = PREPROSS_PATH + "dengiz/result_data/"

# Dataset Columns
TIME = "time of week"
INDOOR = "temperatureBufferStorage" 
OUTDOOR = "Outside Temperature [C]"
POWER_USAGE = "Space Heating [W]"
TIMESTAMP = "Timestamp"

START_DATE = "2023-01-01 00:00:00"

df_array = []
for kWh in range(0, KWH_AMOUNT):    
    for house in range(1, HOUSE_AMOUNT + 1):
        week_df_arr = []
        for week in range(1, WEEKS + 1):
            house_path = HOUSE_PATH + str(house)
            try:
                temp_df = pd.read_csv(f"{START_PATH}/{KWH_CATEGORIES[kWh]}/{MIDDLE_FOLDER_PATH}/{house_path}/{WEEK_PATH}{week}/{HOUSE_PATH}1.csv",
                                      sep=";",
                                      header=0)
            except:
                continue
            week_df_arr.append(temp_df)
        df = pd.concat(week_df_arr) 
        df = df[[TIME, POWER_USAGE, INDOOR, OUTDOOR]]

        date_range = pd.date_range(start=START_DATE, periods=len(df), freq='30min')
        df[TIME] = date_range
        df.columns = [TIMESTAMP, "PowerConsumption", "IndoorTemp", "OutdoorTemp"]

        df.to_csv(f"{OUTPUT_PATH}{KWH_CATEGORIES[kWh]}/{HOUSE_PATH}{house}.csv",
                  sep=";",
                  header=0,
                  index=False)
        
        df_array.append(df)


df = df[(df.IndoorTemp >= 10) & (df.IndoorTemp <= 30)]
df = df[(df.OutdoorTemp >= -50) & (df.OutdoorTemp <= 50)]
series = df[(df.IndoorTemp.diff().abs().astype(float) <= MAX_TEMP_DELTA) & (df.OutdoorTemp.diff().abs().astype(float) <= MAX_TEMP_DELTA)]
df = df[df.PowerConsumption > 0]

df = df.dropna()

df = pd.concat(df_array) 
df.to_csv(f"{PREPROSS_PATH}dataset/Dengiz_cleaned.csv",
          index=False)