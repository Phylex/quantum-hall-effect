import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import norm
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

def gaussian(x, mu, sigma, A):
    """ A Gaussian function """
    return norm.pdf(x, loc=mu, scale=sigma)*A

def fwhm_gauss(sigma):
    return 2*np.sqrt(np.log(2)*2)*sigma
# this part is the ramp selection part of the good ramp
dbfn = "./ramp_up_ch2.dat"
dufn = './ramp_up_ch1.dat'
db = pd.read_csv(dbfn, skiprows=2, delimiter=';')
du = pd.read_csv(dufn, skiprows=2, delimiter=';')
cd = condense_data(du)
cb = condense_data(db)

min_time = int(np.floor(cb['t'].min()))
max_time = int(np.floor(cb['t'].max()))
ramp_start_time = 0
ramp_end_time = 545
bottom_plateau_start_time = 615
ramp_up_duration = ramp_end_time - ramp_start_time
plateaus = [(bottom_plateau_start_time, max_time), (ramp_end_time+5, bottom_plateau_start_time-5)]
ramp = (ramp_start_time, ramp_end_time-1)

plat_vals = []
for i, plateau in enumerate(plateaus):
    p = cut_data(cb, 't', plateau)
    plat_vals.append((p['V'].mean(), mean_error(p['stdV'])))

dist = plat_vals[1][0] - plat_vals[0][0]
dist_std = np.sqrt(plat_vals[1][1]**2 + plat_vals[0][1]**2)
cfactor = 6/dist
cfstd = cfactor*dist_std

print("Calculated U -> B conversion factor: {}".format(cfactor))
print()

r = cb.where((cb['t'] > ramp_start_time) & (cb['t'] < ramp_end_time)).dropna()
bd = pd.Series(r['V']*cfactor, name='B')
bstd = [np.sqrt((stdV/V)**2 + (cfstd/cfactor)**2) * mf for mf, stdV, V in zip(bd, r['stdV'], r['V'])]
bstd = pd.Series(bstd, name='stdB', index=r.index)
r = pd.concat([r, bd, bstd], axis=1)
del r['V']
del r['stdV']


def lin(x, a, b):
    """ linear fit function """
    return x*a+b


popt, pcov = curve_fit(lin, r['t'], r['B'], sigma=r['stdB'])
a = popt[0]
b = popt[1]

print("fitted linear fnction f(x)=a*x+b to the channel 2 slope")
print("the standard deviations are likely underestimated")
print("a={} +/- {}".format(a, pcov[0][0]))
print("b={} +/- {}".format(b, pcov[1][1]))
print()

u = pd.concat([cd, bd, bstd], axis=1)
print("Fitting the peaks of the Uxx voltage on the ramp up")
print("---------------------------------------------------")
peak_cuts = [(2.4, 4.5), (1.6, 2.2)]
for cut in peak_cuts:
    p = cut_data(u, 'B', cut)
    popt, pcov = curve_fit(gaussian, p['B'], p['V'], sigma=p['stdV'])
    print("for Peak between {} and {} T:".format(cut[0], cut[1]))
    print("mean = ({} +/- {}) T".format(popt[0], np.sqrt(pcov[0][0])))
    print("sigma = ({} +/- {}) T".format(popt[1], np.sqrt(pcov[1][1])))
    print("FWHM = ({} +/- {}) T".format(fwhm_gauss(popt[1]), fwhm_gauss(np.sqrt(pcov[1][1]))))
    p.plot('B', 'V', marker='x', linestyle='', color='blue', label='Uxx Hall voltage measurements')
    plt.plot(p['B'], gaussian(p['B'], *popt), label='fitted gaussian', color='orange')
    plt.grid()
    plt.legend()
    plt.xlabel('Magnetic Field Strength')
    plt.ylabel('Uxx Hall voltage')
    figname = 'uxx_4K_between_{}_and_{}T.pdf'.format(cut[0], cut[1])
    plt.savefig(figname)
    plt.close()
