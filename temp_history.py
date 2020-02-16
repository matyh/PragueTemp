import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# load the file into DataFrame
file = './TempPrecData.csv'
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

xdata, ydata = pd.Series(dtype=int), pd.Series(dtype=np.float64)


def update(frame):
    global xdata, ydata

    xdata = xdata.append(rad_months.loc[rad_months['year'] == frame]['month'])  # DND
    ydata = ydata.append(rad_months.loc[rad_months['year'] == frame]['T-AVG'])  # DND
    ln.set_data(xdata, ydata)  # DND
    return ln,

ani = FuncAnimation(fig, update, frames=df['year'].unique(), interval=20, blit=True)
plt.show()


def init():
    ax.set_xlim(0, 2 * np.pi)
    ax.set_ylim(-1, 1)
    return ln,


def update(frame):
    xdata.append(frame)
    ydata.append(np.sin(frame))
    ln.set_data(xdata, ydata)
    return ln,


ani = FuncAnimation(fig, update, frames=np.linspace(0, 2 * np.pi, 128),
                    init_func=init, blit=True, interval=12)
# plt.show()
