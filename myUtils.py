import pandas
from scipy import integrate
from scipy import signal
import numpy as np
from scipy.spatial.transform import Rotation as R

def resample(data, fs, cutoff = 0.7):
    """
    Resample and optionally cut first samples from data.
    ## Arguments:
    data - dataframe to resample \n
    fs - sampling frequency [Hz] \n
    cutoff - cutoff time in seconds [s], it cuts first seconds of data  \n

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
    #t_resampled = data['time_resampled'] 
    #t_resampled = t_resampled - data['time_resampled'][0] #normalization of time (start from 0)
    data['time_resampled'] -= data['time_resampled'][0] #normalization of time (start from 0)

    cutoff = cutoff / unit_of_conv
    df_closest = data.iloc[(data['time_resampled']-cutoff).abs().argsort()[:1]] # find time closest to cutoff in data
    closest_list = df_closest['time_resampled'].tolist()
    cut_data = data[data['time_resampled'] > closest_list[0]]
    t_resampled = cut_data['time_resampled'] 
    t_resampled = t_resampled / (1/unit_of_conv)


    return cut_data, t_resampled


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
    """
    Applies 4th order Butterworth lowpass filter with cutoff frequency of 20 Hz to both az and ay. \n
    After that applies 4th order Butterworth lowpass filter with cutoff frequency of 2 Hz to ay.
    ## Arguments:
    az - vertical acceleration \n
    ay - horizontal (anterior-posterior) acceleration \n
    fs - sampling frequency [Hz] \n
    ## Returns:
    filtered_az - filtered vertical acceleration \n
    filtered_ay - filtered horizontal (anterior-posterior) acceleration \n
    """
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

def find_max_displacement(dz_steps, peaks_ind):
    """
    Calculate max displacement h for every step in dz_steps
    ## Arguments:
    dz_steps - list steps' posistions, every element (step interval) is a vector of position values \n
    peaks_ind - indices of peaks \n
    ## Returns:
    h_list - list of max displacement for every step. Length is the same as dz_steps \n
    """
    h_list = []     # list of displacements h for each step
    h_indices = []       # list of indices of max values in each step interval 
    indices_length = 0          # length of indices which have been already checked 

    differences = np.diff(peaks_ind) # every step's length in number of samples (indices)
    min_diff = np.min(differences)   # shortest gap between peaks = shortest step

    for dz in dz_steps:

        h_index = np.argmax(dz) # index of max value for current step interval dz

        # if it's first iteration, skip that part
        if h_indices:

            # if difference between h_index and previous max value's index is less than min_diff, 
            # make sure that current max value is valid
            if (h_index - (h_indices[-1] - indices_length)) < min_diff:

                # if current max value is higher than previous one
                if dz[h_index] > previous_dz[h_indices[-1]-indices_length]:
                    
                    # assign current value to the previous one
                    h_indices[-1] = h_index + indices_length

                    # skip first half of current interval and search for max value in second half
                    half = int(len(dz)/2)
                    h_index = np.argmax(dz[half:]) + half
                #else:
                    # if current max value is lower than previous one
                    # skip first half of current interval and search for max value in second half
                half = int(len(dz)/2)
                h_index = np.argmax(dz[half:]) + half


        # displacement calculated as absolute value of max and min postion value in current step interval
        h = abs(dz[h_index] - min(dz))


        h_list.append(h)
        h_indices.append(h_index + indices_length)
        indices_length += len(dz)
        previous_dz = dz

    return np.array(h_list)
 
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


    # ### calculate displacement h from position dz

    peaks_ind_last = np.append(peaks_ind, len(t)-1)
    peaks_ind_last = np.insert(peaks_ind_last, 0, 0)

    h_list = find_max_displacement(dz_steps_filtered, peaks_ind_last)

    ### calculating step length
    step_lengths = 2*np.sqrt(2*h_list*l - h_list**2)


    ### calculating step time
    step_times = np.diff(t[peaks_ind_last])


    ## calculating gait velocity
    gait_v = (step_lengths / step_times)


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
