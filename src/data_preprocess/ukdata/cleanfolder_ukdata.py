import pandas as pd
import os
def clean_csv(file_path, output_dir):
    
    # Check for empty and/or too small sets
    try: 
        if os.path.getsize(file_path) < 1000000:
            print(f"Warning: File {file_path} is too small. Skipping")
            return


        # Data parameters
        SAMPLE_TIME = "15min"
        TIMESTAMP = "Timestamp"
        USE_UTC = True
        MAX_TEMP_DELTA = 15

        # This could be done differently without making a "PowerConsumption" and just using the 
        # "Heat_Pump_Energy_Output" but I will just use the nist setup for now.
        # HVAC
        # energy_df = pd.concat([pd.read_csv(PROPERTY_PATH_0001), pd.read_csv(PROPERTY_PATH_0003), pd.read_csv(PROPERTY_PATH_0005), pd.read_csv(PROPERTY_PATH_0014), pd.read_csv(PROPERTY_PATH_0018)])
        energy_df = pd.read_csv(file_path)
        energy_df = energy_df[[TIMESTAMP, "Whole_System_Energy_Consumed"]]
        energy_df.Timestamp = pd.to_datetime(energy_df.Timestamp, utc=USE_UTC)
        energy_df = energy_df.resample(SAMPLE_TIME, on=TIMESTAMP).mean().reset_index()
        energy_df = energy_df.rename(columns={"Whole_System_Energy_Consumed": "TotalPowerConsumption"})


        # Circulation
        circulation_df = pd.read_csv(file_path)
        circulation_df = circulation_df[[TIMESTAMP, "Circulation_Pump_Energy_Consumed"]]
        circulation_df.Timestamp = pd.to_datetime(circulation_df.Timestamp, utc=USE_UTC)
        circulation_df = circulation_df.resample(SAMPLE_TIME, on=TIMESTAMP).mean().reset_index()
        circulation_df = circulation_df.rename(columns={"Circulation_Pump_Energy_Consumed" : "CirculationPowerConsumption"})

        # Convert to consumption instead of total consumption
        # also converts from kWh to W. The kWh are in intervals of 2 minutes
        energy_df['PowerConsumption'] = (energy_df['TotalPowerConsumption'].diff() - circulation_df['CirculationPowerConsumption'].diff()) * 30 * 1000 

        # Indoor
        # indoor_df = pd.concat([pd.read_csv(PROPERTY_PATH_0001), pd.read_csv(PROPERTY_PATH_0003), pd.read_csv(PROPERTY_PATH_0005), pd.read_csv(PROPERTY_PATH_0014), pd.read_csv(PROPERTY_PATH_0018)])
        indoor_df = pd.read_csv(file_path)
        indoor_df = indoor_df[[TIMESTAMP, "Internal_Air_Temperature"]]
        indoor_df.Timestamp = pd.to_datetime(indoor_df.Timestamp, utc=USE_UTC)
        indoor_df = indoor_df.resample(SAMPLE_TIME, on=TIMESTAMP).last().reset_index()
        indoor_df = indoor_df.rename(columns={"Internal_Air_Temperature": "IndoorTemp"})



        # Outdoor
        # outdoor_df = pd.concat([pd.read_csv(PROPERTY_PATH_0001), pd.read_csv(PROPERTY_PATH_0003), pd.read_csv(PROPERTY_PATH_0005), pd.read_csv(PROPERTY_PATH_0014), pd.read_csv(PROPERTY_PATH_0018)])
        outdoor_df = pd.read_csv(file_path)
        outdoor_df = outdoor_df[[TIMESTAMP, "External_Air_Temperature"]]
        outdoor_df.Timestamp = pd.to_datetime(outdoor_df.Timestamp, utc=USE_UTC)
        outdoor_df = outdoor_df.resample(SAMPLE_TIME, on=TIMESTAMP).last().reset_index()
        outdoor_df = outdoor_df.rename(columns={"External_Air_Temperature": "OutdoorTemp"})



        df = energy_df
        df = df.join(indoor_df.set_index(TIMESTAMP), on=TIMESTAMP)
        df = df.join(outdoor_df.set_index(TIMESTAMP), on=TIMESTAMP)

        df = df[(df.IndoorTemp >= 10) & (df.IndoorTemp <= 30)]
        df = df[(df.OutdoorTemp >= -50) & (df.OutdoorTemp <= 50)]
        series = df[(df.IndoorTemp.diff().abs().astype(float) <= MAX_TEMP_DELTA) & (df.OutdoorTemp.diff().abs().astype(float) <= MAX_TEMP_DELTA)]


        df = df.dropna()
        
        output_file = os.path.join(output_dir, os.path.basename(file_path))

        df.to_csv(output_file, index=False)
    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")

input_dir = 'src/data_preprocess/ukdata/9050csv_cleansed_data_set1_b693745c14a63a7ed1c6299c5abe1a19'
output_dir = 'src/data_preprocess/ukdata/UKDATA_CLEANED'

for file in os.listdir(input_dir):
    if file.endswith(".csv"):
        file_path = os.path.join(input_dir, file)
        clean_csv(file_path, output_dir)