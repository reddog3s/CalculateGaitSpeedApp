import pandas
import matplotlib.pyplot as plt
from scipy import integrate
from scipy import signal
import numpy as np
import os
from myUtils import resample
from myUtils import change_orientation
from myUtils import find_max_displacement




### loading data
batch_name = 'fast' # name of the batch (medium, fast, slow) 
num_dir = 2 # number of measurement
base_path = os.path.join(os.path.dirname( __file__ ), os.pardir, 'DATA', batch_name)
dirs = [ name for name in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, name)) ] #list only directories
path = os.path.join(base_path, dirs[num_dir],'Accelerometer.csv')

#path = '1_km_from_start_to_1_km\Accelerometer.csv'

data = pandas.read_csv(path)

path = os.path.join(base_path, dirs[num_dir],'Orientation.csv')

orient = pandas.read_csv(path)
orient.index = pandas.to_datetime(orient['time'], unit = 'ns')


data.index = pandas.to_datetime(data['time'], unit = 'ns')
print(data.head())
print(data.shape)

# creating time vector in seconds
t = data['time'] /10**9 
t = t - t[0]

az = data['z'] # acceleration in z axis (vertical)
ay = data['y'] # acceleration in y axis (anterior-posterior, back-forth)
l = 0.89 # leg length in meters [m]





fig_index = 0
plt.figure(fig_index)
plt.plot(t, data['z'], label='z')
plt.plot(t, data['y'], label='y')
plt.plot(t, data['x'], label='x')
plt.xlabel('Time [s]')
plt.ylabel('Acceleration [m/s2]')
plt.legend()





### resampling

consecutive_deltas = data['seconds_elapsed'].diff()
avg_sampling_rate = 1 / np.mean(consecutive_deltas)
shortest_gap = np.min(consecutive_deltas)
maximum_gap = np.max(consecutive_deltas)
total_num_of_samples = len(data.index)

print('Avg sampling rate: ', avg_sampling_rate)
print('shortest_gap: ', 1/shortest_gap)
print('maximum_gap: ', 1/maximum_gap)



# synchronize
newindex = data.index.union(orient.index)
data = data.reindex(newindex)
orient = orient.reindex(newindex)


# resample
fs = 100 # sampling frequency

data, t_resampled = resample(data, fs)
orient, _ = resample(orient, fs)


fig_index +=1
plt.figure(fig_index)
plt.scatter(t, az, label='Raw')
plt.scatter(t_resampled, data['z'], label='Resampled',s=10)
plt.xlabel('Time [s]')
plt.ylabel('Acceleration [m/s2]')
plt.legend()

# resampled accelerations
az_res = data['z'] 
ay_res = data['y']




# change orientation
coords = [data['x'], data['y'], data['z']]
coords = np.transpose(coords)


euler = [orient['pitch'], orient['roll'], orient['yaw']]
euler = np.transpose(euler)
euler[:,2] = 0 # yaw equal to 0 to avoid changing direction of acceleration in y axis


new_coords = change_orientation(coords, euler)

az = new_coords[:,2] 
ay = new_coords[:,1] 

fig_index +=1
plt.figure(fig_index)
plt.plot(t_resampled, az, label='Changed orientation')
plt.plot(t_resampled, az_res, label='Resampled')
plt.xlabel('Time [s]')
plt.ylabel('Acceleration [m/s2]')
plt.legend()



### filtration

fs = 100

N = 4 # order
fc = 20
Wn = (2/fs)*fc
filt = signal.butter(N, Wn, 'lowpass', output='sos')
filtered_az = signal.sosfiltfilt(filt, az)
filtered_ay = signal.sosfiltfilt(filt, ay)

print(filtered_ay)
print(ay)
### additional filter for ay
N = 4 # order
fc = 2
Wn = (2/fs)*fc
filt = signal.butter(N, Wn, 'lowpass', output='sos')
filtered_ay = signal.sosfiltfilt(filt, filtered_ay)

# filtered_ay = filtered_ay - ay

peaks_ind, _ = signal.find_peaks(filtered_ay, distance=50, prominence=0.4) 

#print(filtered_ay)
#print(peaks_ind)
fig_index +=1
plt.figure(fig_index)
plt.plot(t_resampled, ay, color="blue", label='Raw accelaration')
plt.plot(t_resampled, filtered_ay, color="green", label='Filtered accelaration')
plt.scatter(t_resampled[peaks_ind], filtered_ay[peaks_ind], marker = "x", s = 80, color="red", label='Heel strikes')
plt.xlabel('Time [s]')
plt.ylabel('Acceleration [m/s2]')
plt.legend()


a = np.sqrt(filtered_az**2 + filtered_ay**2)
v = integrate.cumulative_trapezoid(a, t_resampled, initial = 0)
d = integrate.cumulative_trapezoid(v, t_resampled, initial = 0)


N = 4 # order
fc = 0.1
Wn = (2/fs)*fc
filt = signal.butter(N, Wn, 'highpass', output='sos')
d = signal.sosfiltfilt(filt, d)

fig_index +=1
plt.figure(fig_index)
plt.plot(t_resampled, d)
plt.xlabel('Time [s]')
plt.ylabel('Position of body [m]')
plt.legend()




steps = np.split(filtered_az, peaks_ind)
times = np.split(t_resampled, peaks_ind)

print('Number of steps: ', len(steps))

### computing displacement from acceleration

vz_steps = []
dz_steps = []
for i, step in enumerate(steps):
    vz = integrate.cumulative_trapezoid(step, times[i], initial = 0)
    vz_steps.append(vz)
    dz = integrate.cumulative_trapezoid(vz, times[i], initial = 0)
    dz_steps.append(dz)


### additional filter to remove integration drift
N = 4 # order
fc = 0.1
Wn = (2/fs)*fc
filt = signal.butter(N, Wn, 'highpass', output='sos')

dz_steps_filtered = []
for step in dz_steps:
    dz_steps_filtered.append(signal.sosfiltfilt(filt, step))

filtered_dz = np.concatenate(dz_steps_filtered, axis=0 )
vz = np.concatenate(vz_steps, axis=0 )
fig_index +=1
plt.figure(fig_index)
plt.plot(t_resampled, filtered_az, label = 'Acceleration')
plt.plot(t_resampled, vz, label = 'Velocity')
plt.plot(t_resampled, filtered_dz, label = 'Position')
plt.xlabel('Time [s]')
plt.ylabel('Acceleration [m/s2] / Velocity [m/s] / Position [m]')
plt.legend()


fig_index +=1
plt.figure(fig_index)
plt.plot(t_resampled, filtered_az, label = 'Acceleration')
plt.plot(t_resampled, filtered_dz, label = 'Position')
plt.scatter(t_resampled[peaks_ind], filtered_az[peaks_ind], marker = "x", s = 80, color="red", label='Heel strikes')
plt.xlabel('Time [s]')
plt.ylabel('Velocity [m/s] / Position [m]')
plt.legend()



fig_index +=1
plt.figure(fig_index)
plt.plot(t_resampled, vz, label = 'Velocity')
plt.plot(t_resampled, filtered_dz, label = 'Position')
plt.scatter(t_resampled[peaks_ind], filtered_dz[peaks_ind], marker = "x", s = 80, color="red", label='Heel strikes')
plt.xlabel('Time [s]')
plt.ylabel('Velocity [m/s] / Position [m]')
plt.legend()


### calcutating h - displacement - change in position 


h_list = []
args = []
le = 0
offset = 0 #abs(dz_steps_filtered[0][0])


peaks_ind_last = np.append(peaks_ind, len(t_resampled)-1)
peaks_ind_last = np.insert(peaks_ind_last, 0, 0)
differences = np.diff(peaks_ind_last)
min_diff = np.min(differences)



# h_list = np.array(h_list)

h_list, args = find_max_displacement(dz_steps_filtered, peaks_ind_last, True)

#min_diff = 80
ind, _ = signal.find_peaks(filtered_dz, distance = min_diff)


#colors = itertools.cycle(["r", "b", "g"])

fig_index +=1
plt.figure(fig_index)
plt.vlines(t_resampled[peaks_ind], -0.1, 0.1, color="C2", label = 'Start/end of step')
#plt.scatter(t_resampled[peaks_ind], filtered_dz[peaks_ind] + offset, marker = "x", s = 80, color="red", label = 'Heel strikes')
plt.scatter(t_resampled[args], filtered_dz[args] + offset, s = 120, color="orange", label = 'Custom algorithm')
plt.scatter(t_resampled[ind], filtered_dz[ind] + offset, s = 40, color="red", label = 'find_peaks')
plt.plot(t_resampled, filtered_dz + offset, label = 'Position')
plt.xlabel('Time [s]')
plt.ylabel('Position [m]')
plt.legend()



### calculating step length

step_lengths = 2*np.sqrt(2*h_list*l - h_list**2)
#print(step_lengths)
print("Distance: ", np.sum(step_lengths))
print(h_list)

### calculating step time
peaks_ind_last = np.append(peaks_ind, len(t_resampled)-1)
peaks_ind_last = np.insert(peaks_ind_last, 0, 0)
step_times = np.diff(t_resampled[peaks_ind_last])
#print(t_resampled[peaks_ind])
#print(step_times)

## calculating gait velocity
#print(len(step_lengths))
gait_v = (step_lengths / step_times)
print(gait_v)


number_of_steps = 2

number_of_arrays = len(gait_v) / number_of_steps

new_gait_v = np.split(gait_v, number_of_arrays)


#avg_gait_v = np.mean(new_gait_v)
print(np.mean(gait_v))





#print(gait_v)
#print(avg_gait_v)
number_of_meters = 7

#real_time = t_resampled[peaks_ind_last[-1]] - t_resampled[0]
real_time = t_resampled[-1] - t_resampled[0]
real_gait_v = number_of_meters/real_time
print(real_gait_v)


#print(stats.pearsonr(avg_gait_v, real_gait_v))
#print(np.corrcoef(avg_gait_v, real_gait_v))


consecutive_deltas = t_resampled.diff()
avg_sampling_rate = 1 / np.mean(consecutive_deltas)

#print('Avg sampling rate: ', avg_sampling_rate)

plt.show()