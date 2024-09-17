import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")

df = pd.read_csv("HVAC-minute.csv")
df = df[["HVAC_HVACTempReturnAir", "HVAC_HVACTempSupplyAir", "HVAC_HeatPumpIndoorUnitPower"]]

fahrenHeitToCelsius = lambda f : (f - 32) * 5/9

df.to_csv("HVAC-minute_cleaned.csv", index=False)

values = df.values

tempReturnAir = [i[0] for i in values]
tempSupplyAir = [i[1] for i in values]
powerConsum = [i[2] for i in values]

fig, ax = plt.subplots(3)

ax[0].plot(tempReturnAir)
ax[0].set_title("Return air")
ax[0].grid()

ax[1].plot(tempSupplyAir, color="r")
ax[1].set_title("Supply air")
ax[1].grid()

ax[2].plot(powerConsum, color="g")
ax[2].set_title("Power consumption")
ax[2].grid()

plt.subplots_adjust(hspace=1)
plt.savefig("dataset_vis.png")
