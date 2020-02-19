import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# load the file into DataFrame
file = 'TempData.csv'
df: pd.DataFrame = pd.read_csv(file)

# replace comas in the DataFrame with dots and convert to numbers
for col in df.columns[3:7]:
    df[col] = df[col].str.replace(',', '.').astype(float)
# print(df.dtypes)  # check columns data types

# make monthly temperatures by averaging month temperatures
df = df.groupby(['year', 'month'], as_index=False).mean()

# delete arbitrary columns
del df['day']
del df['SRA']


# MAKE PLOT
fig, ax = plt.subplots()
plt.subplot(projection='polar')
ln, = plt.plot([], [])
# plt.xlim((0, 13))
plt.ylim((-20, 30))

# adjust plot
# convert months numbers to radians
rad_months = df.copy()
rad_months.update(rad_months['month'].apply(lambda x: np.deg2rad(360 / 12 * x)))
plt.thetagrids(np.linspace(360 / 12, 360, len(df['month'].unique())), df['month'].unique())

xdata, ydata = pd.Series(dtype=np.float64), pd.Series(dtype=np.float64)


def update(frame):
    global xdata, ydata

    xdata = xdata.append(pd.Series(rad_months.iloc[frame]['month']))  # DND
    ydata = ydata.append(pd.Series(rad_months.iloc[frame]['T-AVG']))  # DND
    ln.set_data(xdata, ydata)  # DND
    return ln,


ani = FuncAnimation(fig, update, frames=range(0, len(rad_months)), interval=20, blit=True)
plt.show()
