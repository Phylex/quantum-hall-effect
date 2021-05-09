import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys

data = pd.read_csv(sys.argv[1], skiprows=2, delimiter=';')
data.plot('Time / s', 'Voltage / V')
plt.show()
