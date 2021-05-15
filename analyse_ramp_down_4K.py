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
db_file = "./ramp_down_4K_ch1.dat"
du_file = "./ramp_down_4K_ch2.dat"
du = pd.read_csv(du_file, skiprows=2, delimiter=';')
db = pd.read_csv(db_file, skiprows=2, delimiter=';')

cb = condense_data(db)
cu = condense_data(du)

min_time = int(np.floor(cb['t'].min()))
max_time = int(np.floor(cb['t'].max()))
ramp_start_time = 0
ramp_end_time = 530
ramp_up_duration = ramp_end_time - ramp_start_time
plateaus = [(615, 670), (675, max_time)]
ramp = (ramp_start_time+5, ramp_end_time-5)

plat_vals = []
for i, plateau in enumerate(plateaus):
    p = cut_data(cb, 't', plateau)
    plat_vals.append((p['V'].mean(), mean_error(p['stdV'])))
dist = np.abs(plat_vals[1][0] - plat_vals[0][0])
dist_std = np.sqrt(plat_vals[1][1]**2 + plat_vals[0][1]**2)
cfactor = 6/dist
cfstd = cfactor*dist_std
print("Conversion factor from Power supply voltage to Magnetic field strength")
print("cf = {} +/- {}".format(cfactor, cfstd))
print()

r = cut_data(cb, 't', ramp)
b = pd.Series(r['V']*cfactor, name='B')
bstd = [np.sqrt((stdV/V)**2 + (cfstd/cfactor)**2) * mf for mf, stdV, V in zip(b, r['stdV'], r['V'])]
bstd = pd.Series(bstd, name='stdB', index=r.index)
r = pd.concat([r, b, bstd], axis=1)
del r['V']
del r['stdV']
def lin(x, a, b):
   return x*a+b

cf_popt, cf_pcov = curve_fit(lin, r['t'], r['B'], sigma=r['stdB'])
print("Parameters for the downward slope:\na = {} +/- {}\nb = {} +/- {}".format(cf_popt[0],
                                                                                cf_pcov[0][0],
                                                                                cf_popt[1],
                                                                                cf_pcov[1][1]))
print()

# add the converted data to the hall voltage dataframe
cu = pd.concat([cu, b, bstd], axis=1)
# plot the hall voltage
cu.plot('B', 'V', label='Hall Voltage Uxy', color='blue')
plt.grid()
plt.xlabel('Magnetic Field Strength [T]')
plt.ylabel('Hall Voltage [V]')
plt.legend()
figname = "hall_voltage_uxy_4K.pdf"
plt.savefig(figname)
print("plotting the Hall voltage to {} ...".format(figname))
print()
hall_plateaus = [(2.1, 2.7), (4, 4.4)]
print("Calculating Hall plateau Voltages")
print("---------------------------------")
for hp in hall_plateaus:
    hd = cut_data(cu, 'B', hp)
    print('Hall Voltage for plateau between {} and {} T: {} +/- {} V'.format(hp[0],
                                                                           hp[1],
                                                                           hd['V'].mean(),
                                                                           mean_error(hd['stdV'])))
    print()
