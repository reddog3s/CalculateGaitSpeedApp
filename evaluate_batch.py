from myUtils import calculate_gait_velocity2
from myUtils import resample
from myUtils import apply_filters
from myUtils import change_orientation

import pandas
import matplotlib.pyplot as plt
from scipy import signal
from scipy import stats
import os
import numpy as np


fs = 100 # sampling frequency
l = 0.89 # leg length in meters [m]
number_of_meters = 7 # distance in meters [m]

base_path = os.path.join(os.path.abspath(os.getcwd()),'10m','medium')


dirs = [ name for name in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, name)) ] #list only directories


results = pandas.DataFrame({
    "Aprox gait v": [],
    "Real gait v": []
})


for directory in dirs:
    # load data
    path = os.path.join(base_path, directory,'Accelerometer.csv')
    accel = pandas.read_csv(path)
    accel.index = pandas.to_datetime(accel['time'], unit = 'ns')

    path = os.path.join(base_path, directory,'Orientation.csv')
    orient = pandas.read_csv(path)
    orient.index = pandas.to_datetime(orient['time'], unit = 'ns')


    # synchronize
    newindex = accel.index.union(orient.index)
    accel = accel.reindex(newindex)
    orient = orient.reindex(newindex)

    # resample
    accel, t_resampled = resample(accel, fs)
    orient, _ = resample(orient, fs)


    # change orientation
    coords = [accel['x'], accel['y'], accel['z']]
    coords = np.transpose(coords)

    euler = [orient['pitch'], orient['roll'], orient['yaw']]
    euler = np.transpose(euler)
    euler[:,2] = 0 # yaw equal to 0 to avoid changing direction of acceleration in y axis


    new_coords = change_orientation(coords, euler)
    az = new_coords[:,2] 
    #ay = orient_factor*new_coords[:,1] # orient factor because of specific phone coordinate system
    ay = new_coords[:,1] 



    ### filtration

    filtered_az, filtered_ay = apply_filters(az, ay, fs)

    ### find peaks 

    #peaks_ind, _ = signal.find_peaks(filtered_ay, height = 0.5) 
    peaks_ind, _ = signal.find_peaks(filtered_ay, distance=50, prominence=0.4) 

    ### calculate velocity
    gait_v, _, _ = calculate_gait_velocity2(t_resampled, filtered_az, peaks_ind, fs = fs, l = l, number_of_steps = 'all')



    #calcutate real velocity
    #real_time = t_resampled[-1] - t_resampled[peaks_ind[0]]
    real_time = t_resampled[-1] - t_resampled[0]
    real_gait_v = number_of_meters/real_time

    print('\n')
    print('real_gait_v')
    print(real_gait_v)
    print('aprox_gait_v')
    print(gait_v)

    results.loc[directory] = [gait_v, real_gait_v]

results = results.sort_values(by=['Real gait v'])
print(results)
#results = results.drop('10m_fast5')

aprox = results["Aprox gait v"]
real = results["Real gait v"]


mean_real = np.mean(real)
mean_aprox = np.mean(aprox)

print("\nReal mean = %.3f, Real std = %.3f" % (mean_real, np.std(real)))
print("Aprox mean = %.3f, Aprox std = %.3f" % (mean_aprox, np.std(aprox)))

mean_err = abs(mean_real - mean_aprox)
print("Mean error = %.3f, Mean relative error = %.3f %%" % (mean_err, (mean_err/mean_real)*100))
print(stats.pearsonr(aprox, real))



### Bland-Altman analysis


import scipy.stats as st

means = (aprox + real) / 2
diffs = aprox - real


# Average difference (aka the bias)
bias = np.mean(diffs)
# Sample standard deviation
s = np.std(diffs, ddof=1)  # Use ddof=1 to get the sample standard deviation

print(f'For the differences, x̄ = {bias:.2f} m/s and s = {s:.2f} m/s')

# Limits of agreement (LOAs)
upper_loa = bias + 1.96 * s
lower_loa = bias - 1.96 * s

print(f'The limits of agreement are {upper_loa:.2f} m/s and {lower_loa:.2f} m/s')


# Confidence level
C = 0.95  # 95%
# Significance level, α
alpha = 1 - C
# Number of tails
tails = 2
# Quantile (the cumulative probability)
q = 1 - (alpha / tails)
# Critical z-score, calculated using the percent-point function (aka the
# quantile function) of the normal distribution
z_star = st.norm.ppf(q)

print(f'95% of normally distributed data lies within {z_star}σ of the mean')

# Limits of agreement (LOAs)
loas = (bias - z_star * s, bias + z_star * s)

print(f'The limits of agreement are {loas} m/s')

# Limits of agreement (LOAs)
loas = st.norm.interval(C, bias, s)

print(np.round(loas, 2))


# Create plot
ax = plt.axes()
ax.scatter(means, diffs, c='k', s=20, alpha=0.6, marker='o')
# Plot the zero line
ax.axhline(y=0, c='k', lw=0.5)
# Plot the bias and the limits of agreement
ax.axhline(y=loas[1], c='grey', ls='--')
ax.axhline(y=bias, c='grey', ls='--')
ax.axhline(y=loas[0], c='grey', ls='--')
# Labels
ax.set_title('Bland-Altman Plot')
ax.set_xlabel('Mean (m/s)')
ax.set_ylabel('Difference (m/s)')
# Get axis limits
left, right = ax.get_xlim()
bottom, top = ax.get_ylim()
# Set y-axis limits
max_y = max(abs(bottom), abs(top))
ax.set_ylim(-max_y * 1.1, max_y * 1.1)
# Set x-axis limits
domain = right - left
ax.set_xlim(left, left + domain * 1.1)
# Annotations
ax.annotate('+LoA', (right, upper_loa), (0, 7), textcoords='offset pixels')
ax.annotate(f'{upper_loa:+4.2f}', (right, upper_loa), (0, -25), textcoords='offset pixels')
ax.annotate('Bias', (right, bias), (0, 7), textcoords='offset pixels')
ax.annotate(f'{bias:+4.2f}', (right, bias), (0, -25), textcoords='offset pixels')
ax.annotate('-LoA', (right, lower_loa), (0, 7), textcoords='offset pixels')
ax.annotate(f'{lower_loa:+4.2f}', (right, lower_loa), (0, -25), textcoords='offset pixels')
# Show plot
plt.show()