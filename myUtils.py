import pandas
from scipy import integrate
from scipy import signal
import numpy as np
from scipy.spatial.transform import Rotation as R

def resample(data, fs):
    """
    ## Arguments:
    data - dataframe to resample \n
    fs - sampling frequency [Hz] \n

    ## Returns:
    data - resampled dataframe with ['time_resampled'] column \n
    t_resampled - vector of resampled time starting from 0 \n
    """
    seconds = 1/fs

    new = str(seconds) + 'S'


    data = data.resample(new).median()
    data = data.interpolate(method='index')

    # if NaNs stayed (especially in the first index), fill them with next valid value
    data = data.fillna(method = 'backfill')

    unit_of_conv = seconds / 10

    unit_of_conv_str = str(unit_of_conv) + 'S'

    data['time_resampled'] = (data.index - pandas.Timestamp("1970-01-01")) // pandas.Timedelta(unit_of_conv_str) # conversion from new date to unix time
    t_resampled = data['time_resampled'] 
    t_resampled = t_resampled - data['time_resampled'][0] #normalization of time (start from 0)
    t_resampled = t_resampled / (1/unit_of_conv)


    return data, t_resampled


def change_orientation(coords, euler):
    '''
    # Arguments:
    coords = [ax, ay, az], shape N x 3 \n
    euler = (x, y, z) format, shape N x 4 \n

    # Returns:
    inv_coords = coordinates with invert transformation applied \n
    '''
    # r = R.from_quat(quats)
    r = R.from_euler('xyz', euler)
    inverted = r.inv()
    inv_coords = inverted.apply(coords)

    return inv_coords


def apply_filters(az, ay, fs):
    N = 4 # order
    fc = 20
    Wn = (2/fs)*fc
    filt = signal.butter(N, Wn, 'lowpass', output='sos')
    filtered_az = signal.sosfiltfilt(filt, az)
    filtered_ay = signal.sosfiltfilt(filt, ay)

    ### additional filter for ay
    N = 4 # order
    fc = 2 # maximum step rate
    Wn = (2/fs)*fc
    filt = signal.butter(N, Wn, 'lowpass', output='sos')
    filtered_ay = signal.sosfiltfilt(filt, filtered_ay)

    
    return filtered_az, filtered_ay


def getDivisors(n):
    i = 1
    divs = []
    while i <= (n/2) :
        if (n % i==0):
            divs.append(i)
        i = i + 1
    return divs

 
def calculate_gait_velocity2(t, az, peaks_ind, fs = 'assess', l = 0.89, number_of_steps = 2):
    """
    ## Arguments:
    t - time vector \n
    az - vertical acceleration \n
    peaks_ind - peaks indices \n
    fs - sampling frequency [Hz] \n
    l - leg length [m] \n
    number_of_steps - number of steps to calculate velocity from \n

    ## Returns:
    avg_gait_v - average gait velocity across number_of_steps \n
    num_of_steps - number of steps made \n
    tuple - velocities between steps and corresponding time vector (same as peaks indices with beginning and end of the signal).
            Last velocity value is doubled to provide equal vectors.
    """

    ### if fs == 'assess', assess fs based on average of intervals between samples
    if fs == 'assess':
        intervals = t.diff()
        fs = 1 / np.mean(intervals)



    steps = np.split(az, peaks_ind)
    times = np.split(t, peaks_ind)

    #print('Number of steps: ', len(steps))
    num_of_steps = len(steps)



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
    fc = 0.1 #cutoff frequency
    Wn = (2/fs)*fc #normalized cutoff frequency
    filt = signal.butter(N, Wn, 'highpass', output='sos')

    dz_steps_filtered = []
    for step in dz_steps:
        dz_steps_filtered.append(signal.sosfiltfilt(filt, step))






    h_list = []
    args = []
    le = 0


    peaks_ind_last = np.append(peaks_ind, len(t)-1)
    peaks_ind_last = np.insert(peaks_ind_last, 0, 0)
    differences = np.diff(peaks_ind_last)
    min_diff = np.min(differences)

    for dz in dz_steps_filtered:

        h_index = np.argmax(dz)

        if args:
            if (h_index - (args[-1] - le)) < min_diff:
                if dz[h_index] > previous_dz[args[-1]-le]:
                    args[-1] = h_index + le

                    half = int(len(dz)/2)
                    h_index = np.argmax(dz[half:]) + half
                else:
                    half = int(len(dz)/2)
                    h_index = np.argmax(dz[half:]) + half


        h = abs(dz[h_index]) + abs(min(dz))


        h_list.append(h)
        args.append(h_index + le)
        le += len(dz)
        previous_dz = dz


    h_list = np.array(h_list)

    ### calculating step length

    step_lengths = 2*np.sqrt(2*h_list*l - h_list**2)
    #print(step_lengths)
    #print("Distance: ", np.sum(step_lengths))
    #print(h_list)

    ### calculating step time
    step_times = np.diff(t[peaks_ind_last])
    #print(t_resampled[peaks_ind])
    #print(step_times)

    ## calculating gait velocity
    #print(len(step_lengths))
    gait_v = (step_lengths / step_times)
    #print(gait_v)



    
    ## if number of steps is 'all', calcutate average across all steps
    if number_of_steps == 'all':
        avg_gait_v = np.mean(gait_v)
    else:
        
        rest = len(gait_v) % int(number_of_steps)
        if rest:
            
            # get minimal divisor if cant divide len(gait_v) by number_of_steps
            divs = getDivisors(len(gait_v))
            number_of_arrays = len(gait_v) / min(divs) 
            splitted_gait_v = np.split(gait_v, number_of_arrays)
            avg_gait_v = np.mean(splitted_gait_v, 1)  
        else:
            number_of_arrays = len(gait_v) / number_of_steps
            splitted_gait_v = np.split(gait_v, number_of_arrays)
            avg_gait_v = np.mean(splitted_gait_v, 1)

    return avg_gait_v, num_of_steps, (t[peaks_ind_last], np.append(gait_v, gait_v[-1]))
