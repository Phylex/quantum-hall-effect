import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import sys

def mean_error(stds):
    return np.sqrt(sum(np.array(stds)**2))/len(stds)

data = pd.read_csv(sys.argv[1], skiprows=2, delimiter=';')
means = []
min_time = int(np.floor(data['Time / s'].min()))
max_time = int(np.ceil(data['Time / s'].max()))
plateaus = [(min_time, 15), (735, max_time)]
ramp = (25, 725)
for i in range(min_time, max_time-1):
    d = data.copy().where((data['Time / s'] > i) & (data['Time / s'] < i+1))
    d = d.dropna()
    means.append(d)

for i, d in enumerate(means):
    means[i] = (d['Voltage / V'].mean(), d['Voltage / V'].std())

m, std = zip(*means)
x = (np.array(range(min_time, max_time-1)) + np.array(range(min_time+1, max_time)))/2
cd = pd.DataFrame({'V': m, 'stdV': std, 't': x})

plat_vals = []
for i, plateau in enumerate(plateaus):
    p = cd.copy().where((cd['t'] > plateau[0]) & (cd['t'] < plateau[1]))
    p = p.dropna()
    print("plateau {}: mean={:.3f} std={:.3f}".format(i, p['V'].mean(),
                                              mean_error(p['stdV'])))
    plat_vals.append((p['V'].mean(), mean_error(p['stdV'])))
dist = plat_vals[1][0] - plat_vals[0][0]
dist_std = np.sqrt(plat_vals[1][1]**2 + plat_vals[0][1]**2)
print("Distance between plateaus: {:.3f} +/- {:.3f}".format(dist, dist_std))


r = cd.where((cd['t'] > ramp[0]) & (cd['t'] < ramp[1])).dropna()

def lin(x, a, b):
    return x*a+b


popt, pcov = curve_fit(lin, r['t'], r['V'], sigma=r['stdV'])
r.plot('t', 'V', linestyle='', marker='x')
plt.plot(r['t'], lin(r['t'], *popt))
print("Slope parameters: a={:.5f}, b={:.5f}".format(popt[0],popt[1]))
plt.show()
