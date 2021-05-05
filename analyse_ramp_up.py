import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import sys

def condense_data(df):
    means = []
    min_time = int(np.floor(df['Time / s'].min()))
    max_time = int(np.ceil(df['Time / s'].max()))
    for i in range(min_time, max_time-1):
        d = df.copy().where((data['Time / s'] > i) & (data['Time / s'] < i+1))
        d = d.dropna()
        means.append((d['Voltage / V'].mean(), d['Voltage / V'].std()))
    m, std = zip(*means)
    x = (np.array(range(min_time, max_time-1)) +
         np.array(range(min_time+1, max_time)))/2
    return pd.DataFrame({'V': m, 'stdV': std, 't': x})

def mean_error(stds):
    return np.sqrt(sum(np.array(stds)**2))/len(stds)

data = pd.read_csv(sys.argv[1], skiprows=2, delimiter=';')
voltage = pd.read_csv(sys.argv[2], skiprows=2, delimiter=';')

cd = condense_data(data)
vlt = condense_data(voltage)

min_time = int(np.floor(cd['t'].min()))
max_time = int(np.floor(cd['t'].max()))
ramp_start_time = 20
ramp_end_time = 730
plateaus = [(min_time, ramp_start_time-5), (ramp_end_time+5, max_time)]
ramp = (ramp_start_time+5, ramp_end_time-5)

plat_vals = []
for i, plateau in enumerate(plateaus):
    p = cd.copy().where((cd['t'] > plateau[0]) & (cd['t'] < plateau[1]))
    p = p.dropna()
    print("plateau {}: mean={:.3f} std={:.3f}".format(i, p['V'].mean(),
                                                      mean_error(p['stdV'])))
    plat_vals.append((p['V'].mean(), mean_error(p['stdV'])))
dist = plat_vals[1][0] - plat_vals[0][0]
dist_std = np.sqrt(plat_vals[1][1]**2 + plat_vals[0][1]**2)
cfactor = 6/dist
cfstd = cfactor*dist_std
print("Distance between plateaus: {:.3f} +/- {:.3f}".format(dist, dist_std))
print("Conversion factor U-> B: {:.5f} +/- {:.3f}".format(cfactor, cfstd))

r = cd.where((cd['t'] > ramp_start_time) & (cd['t'] < ramp_end_time)).dropna()
b = pd.Series(r['V']*cfactor, name='B')
bstd = [np.sqrt((stdV/V)**2 + (cfstd/cfactor)**2) * mf for mf, stdV, V in zip(b, r['stdV'], r['V'])]
bstd = pd.Series(bstd, name='stdB', index=r.index)
r = pd.concat([r, b, bstd], axis=1)
del r['V']
del r['stdV']
print(r)
ramp_start_time
def lin(x, a, b):
   return x*a+b


popt, pcov = curve_fit(lin, r['t'], r['B'], sigma=r['stdB'])
r.plot('t', 'B', linestyle='', marker='x')
plt.plot(r['t'], lin(r['t'], *popt))
plt.show()
print("Slope parameters: a={:.5f}, b={:.5f}".format(popt[0],popt[1]))

# now the field strength can be looked up for every value of the plateau
r = pd.concat([r, vlt['V'], vlt['stdV']], axis=1).dropna()
print(r)
r.plot('B', 'V')
plt.show()
