from datetime import datetime
import pandas as pd

data = pd.read_csv("dataset/NIST_cleaned.csv").values
train_size = int(len(data) * 0.8)

date_format = '%Y-%m-%d %H:%M:%S%z'

date_object = datetime.strptime(data[0][0], date_format)

last_date = datetime.strptime(data[0][0], date_format).date()
dayCounter = 0

train_data = []
test_data = []
for row in data:
    date = datetime.strptime(row[0], date_format).date()

    if dayCounter <= 16:
            train_data.append(row)
    else:
        test_data.append(row)

    if date > last_date:
        last_date = date
        dayCounter += 1
        if dayCounter == 20:
            dayCounter == 0