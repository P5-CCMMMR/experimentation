from .cleaner import Cleaner

class TempCleaner(Cleaner):
    def __init__(self, pow_low_lim, in_low_lim, in_upper_lim, out_low_lim, out_upper_lim, delta_temp):
        self.pow_low_lim = pow_low_lim
        self.in_low_lim = in_low_lim
        self.in_upper_lim = in_upper_lim
        self.out_low_lim = out_low_lim
        self.out_upper_lim = out_upper_lim
        self.delta_temp = delta_temp

    def clean(self, df):
        df = df[(df.IndoorTemp >= self.in_low_lim) & (df.IndoorTemp <= self.in_upper_lim)]
        df = df[(df.OutdoorTemp >= self.out_low_lim) & (df.OutdoorTemp <= self.out_upper_lim)]
        df = df[(df.IndoorTemp.diff().abs().astype(float) <= self.delta_temp) & (df.OutdoorTemp.diff().abs().astype(float) <= self.delta_temp)]
        df = df[df.PowerConsumption > self.pow_low_lim]

        return df.dropna()