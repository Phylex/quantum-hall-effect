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

r = cb.where((cb['t'] > ramp_start_time) & (cb['t'] < ramp_end_time)).dropna()
b = pd.Series(r['V']*cfactor, name='B')
bstd = [np.sqrt((stdV/V)**2 + (cfstd/cfactor)**2) * mf for mf, stdV, V in zip(b, r['stdV'], r['V'])]
bstd = pd.Series(bstd, name='stdB', index=r.index)
r = r.concat([r, b, bstd], axis=1)
del r['V']
del r['stdV']


def lin(x, a, b):
    """ linear fit function """
    return x*a+b


popt, pcov = curve_fit(lin, r['t'], r['B'], sigma=r['stdB'])
a = popt[0]
b = popt[1]

# load the dataset that we actually want to examine
low_plateau_start = 583
ramp_duration = 600

slope = pd.read_csv("ramp_down_2K_B.dat", skiprows=2, delimiter=';')
signal = pd.read_csv("ramp_down_2K_U.dat", skiprows=2, delimiter=';')
slope = condense_data(slope)
signal = condense_data(signal)
low_plateau = cut_data(slope, 't',
                       (low_plateau_start + 3, slope['t'].max()))
down_ramp = cut_data(slope, 't',
                     (slope['t'].min(), low_plateau_start - 3))
lp_mean = low_plateau['V'].mean()
lp_std = low_plateau['V'].std()

rd_popt, rd_pcov = curve_fit(lin, down_ramp['t'], down_ramp['V'], p0=[-0.006, 4])
print("Parameters for the downward slope:\na = {} +/- {}\nb = {} +/- {}".format(rd_popt[0],
                                                                                rd_pcov[0][0],
                                                                                rd_popt[1],
                                                                                rd_pcov[1][1]))
true_t0 = low_plateau_start - ramp_up_duration
high_plateau = lin(true_t0, *rd_popt)
high_plateau_std = np.sqrt(rd_pcov[1][1]**2 + (ramp_up_duration*rd_pcov[0][0])**2)
rd_dist = high_plateau - lp_mean
rd_dist_std = np.sqrt(high_plateau_std**2 + lp_std**2)
rd_cf = 6 / rd_dist
rd_cfstd = rd_dist_std * rd_cf
print('Conversion factor for the ramp:\n{} +/- {}'.format(rd_cf, rd_cfstd))
slope['V'] = slope['V'] - lp_mean
slope['stdV'] = np.sqrt(slope['stdV']**2 + lp_std**2)


b_down = pd.Series(slope['V'] * rd_cf, name='B', index=signal.index)
b_down_std = pd.Series(np.sqrt((slope['stdV']/slope['V'])**2 + (rd_cfstd/rd_cf)**2) * b,
                       name='stdB',
                       index=signal.index)
signal = pd.concat([signal, b_down, b_down_std], axis= 1)
signal['V'] *= (-1)
peak_borders = [(2.5, 4), (1.65, 2.15), (1.2, 1.5), (0.95, 1.125)]
peak_params = []
peak_covs = []
figname = "uxx_ramp_down.pdf"
signal.plot('B', 'V', label="Uxx Hall Voltage", linestyle='', marker='x', color='blue')
plt.grid()
plt.legend()
plt.xlabel('Magnetic Field [T]')
plt.ylabel('Hall Voltage [V]')
plt.savefig(figname)
plt.close()
print("Plotted Uxx Hall Voltage to: {}".format(figname))
print()
print("Fitting Gaussian curves to the Uxx Peaks")
print("----------------------------------------")
for peak in peak_borders:
    p = cut_data(signal, 'B', peak)
    popt, pcov = curve_fit(gaussian, p['B'], p['V'])
    print("Location of Peak: {:.5f} +/- {:.5f}".format(popt[0], np.sqrt(pcov[0][0])))
    print("width of peak (sigma): {:.5f} +/- {:.5f}".format(popt[1], np.sqrt(pcov[1][1])))
    p.plot('B', 'V', marker='x', color='blue', linestyle='', label='Uxx')
    x = np.linspace(p['B'].min(), p['B'].max(), 1000)
    plt.plot(x, gaussian(x, *popt), color='orange', linestyle='--', label='gaussian fit')
    plt.legend()
    plt.grid()
    plt.xlabel('Magnetic Field Strength [T]')
    plt.ylabel('Hall voltage [V]')
    figname = 'uxx_peak_{}-{}T.pdf'.format(peak[0], peak[1])
    plt.savefig(figname)
    plt.close()
    print("Saved plot of the fitted peak at: {}".format(figname))
    print()
