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
        d = df.copy().where((df['Time / s'] > i) & (df['Time / s'] < i+1))
        d = d.dropna()
        means.append((d['Voltage / V'].mean(), d['Voltage / V'].std()))
    m, std = zip(*means)
    x = (np.array(range(min_time, max_time-1)) +
         np.array(range(min_time+1, max_time)))/2
    return pd.DataFrame({'V': m, 'stdV': std, 't': x})

def cut_data(df, var, rng):
    return df.copy().where((df[var] > rng[0]) &
                           (df[var] < rng[1])).dropna(axis=0)

def mean_error(stds):
    return np.sqrt(sum(np.array(stds)**2))/len(stds)

dufn = "./ramp_up_2K_B.dat"
vfn = "./ramp_up_2K_U.dat"
du = pd.read_csv(dufn, skiprows=2, delimiter=';')
voltage = pd.read_csv(vfn, skiprows=2, delimiter=';')

cd = condense_data(du)
vlt = condense_data(voltage)

min_time = int(np.floor(cd['t'].min()))
max_time = int(np.floor(cd['t'].max()))
ramp_start_time = 20
ramp_end_time = 730
current = 20e-6
plateaus = [(min_time, ramp_start_time-5), (ramp_end_time+5, max_time)]
ramp = (ramp_start_time+5, ramp_end_time-5)

plat_vals = []
for i, plateau in enumerate(plateaus):
    p = cd.copy().where((cd['t'] > plateau[0]) & (cd['t'] < plateau[1]))
    p = p.dropna()
    #print("plateau {}: mean={:.3f} std={:.3f}".format(i, p['V'].mean(),
    #                                                  mean_error(p['stdV'])))
    plat_vals.append((p['V'].mean(), mean_error(p['stdV'])))
dist = plat_vals[1][0] - plat_vals[0][0]
dist_std = np.sqrt(plat_vals[1][1]**2 + plat_vals[0][1]**2)
cfactor = 6/dist
cfstd = cfactor*dist_std
print()
print("Calculation of the conversion factor of Voltage to B-field-strength")
print("-------------------------------------------------------------------")
print("Voltage difference of B-ramp: {:.3f} +/- {:.3f}".format(dist, dist_std))
print("B-field difference of B-ramp: 6T")
print("Conversion factor U-> B: {:.5f} +/- {:.3f}".format(cfactor, cfstd))

r = cd.where((cd['t'] > ramp_start_time) & (cd['t'] < ramp_end_time)).dropna()
b = pd.Series(r['V']*cfactor, name='B')
bstd = [np.sqrt((stdV/V)**2 + (cfstd/cfactor)**2) * mf for mf, stdV, V in zip(b, r['stdV'], r['V'])]
bstd = pd.Series(bstd, name='stdB', index=r.index)
r = pd.concat([r, b, bstd], axis=1)
del r['V']
del r['stdV']
ramp_start_time
def lin(x, a, b):
   return x*a+b

figname = "2K_B_slope_fit.pdf"
popt, pcov = curve_fit(lin, r['t'], r['B'], sigma=r['stdB'])
r.plot('t', 'B', linestyle='', marker='x', label="magnetic field measurements (calculated)")
plt.plot(r['t'], lin(r['t'], *popt), label="linear fit with slope: {}".format(popt[0]))
plt.grid()
plt.legend()
plt.title("Linear fit to magnetic field")
plt.ylabel("Magnetic field Strength [T]")
plt.xlabel("Timestamp [s]")
plt.savefig(figname)
plt.close()


print()
print("Fitting a linear function to B-field Slope")
print("------------------------------------------")
print("Fitting function: a * x + b = f(x)")
print("Slope parameters: a={:.5f}, b={:.5f}".format(popt[0],popt[1]))

# now the field strength can be looked up for every value of the plateau
figname = "2K_hall_voltage_plot.pdf"
r = pd.concat([r, vlt['V'], vlt['stdV']], axis=1).dropna()
r.plot('B', 'V', label="Hall Voltage")
plt.grid()
plt.legend()
plt.title("Hall Voltage")
plt.xlabel("Magnetic Field [T]")
plt.ylabel("Hall Voltage [U]")
plt.savefig(figname)
plt.close()
print()
print("Plotting Results:")
print("-----------------")
print("Plotted slope fit in: {}".format(figname))
print("Plotted Hall Voltage over the magnetic field strength to {}".format(figname))
print()
print("Calculation of the Hall Plateau parameters:")
print("-------------------------------------------")
print("Hall Current: {:f} A".format(current))
hall_plateaus = [(4.1, 6.0), (2.2, 2.6), (1.5, 1.7), (1.20, 1.25)]
for i, hp in enumerate(hall_plateaus):
    hd = cut_data(r, 'B', hp)
    plat_U, plat_U_std = (hd['V'].mean(), mean_error(hd['stdV']))
    print("Plateau between {}T and {}T: U=({:.4f} +/- {:.4f})V".format(hp[0],
                                                               hp[1],
                                                               plat_U,
                                                               plat_U_std))
    print("Corresponding Hall resistance: ({} +/- {}) Ohm".format(round(plat_U/current, -1),
                                                                  round(plat_U_std/current, -1)))
    print()
