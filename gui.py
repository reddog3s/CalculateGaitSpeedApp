import PySimpleGUI as sg
import os
import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from myUtils import calculate_gait_velocity2
from myUtils import resample
from myUtils import apply_filters
from myUtils import change_orientation
from scipy import signal

# a default directory where a file is searched
working_directory = os.path.join(os.getcwd(),'data')



def create_raw_plot(acc_file):
    '''creates a plot of a raw signal'''

    df = pd.read_csv(acc_file)
    df.index = pd.to_datetime(df['time'], unit = 'ns')
    t = df['time'] /10**9 
    t = t - t[0]
    plt.figure(1)
    plt.figure(figsize=(7,5))
    plt.plot(t, df['z'], label='z')
    plt.plot(t, df['y'], label='y')
    plt.plot(t, df['x'], label='x')
    plt.xlabel('Time [s]')
    plt.ylabel('Acceleration [m/s2]')
    plt.title('Raw signal')
    plt.legend()
    return plt.gcf()

def draw_figure(canvas, figure):
    ''' draws plots'''
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg

def calculations(acc_file, leg_length):
    ''' calculations of average velocity and steps '''

    fs = 100 # sampling frequency
    df = pd.read_csv(acc_file)
    df.index = pd.to_datetime(df['time'], unit = 'ns')

    path = acc_file.replace("Accelerometer.csv", "Orientation.csv")
    orient = pd.read_csv(path)
    orient.index = pd.to_datetime(orient['time'], unit = 'ns')

    # synchronize
    newindex = df.index.union(orient.index)
    df = df.reindex(newindex)
    orient = orient.reindex(newindex)

    data, t_resampled = resample(df, fs)
    orient, _ = resample(orient, fs)

    # change orientation
    coords = [data['x'], data['y'], data['z']]
    coords = np.transpose(coords)

    euler = [orient['pitch'], orient['roll'], orient['yaw']]
    euler = np.transpose(euler)
    euler[:,2] = 0 # yaw equal to 0 to avoid changing direction of acceleration in y axis

    new_coords = change_orientation(coords, euler)

    az = new_coords[:,2] 
    ay = new_coords[:,1]


    filtered_az, filtered_ay = apply_filters(az, ay, fs)
    x = np.median(filtered_ay)
    filtered_ay = filtered_ay - x
    peaks_ind, _ = signal.find_peaks(filtered_ay, distance=50, prominence=0.4) 

 
    # l = 0.89 leg length in meters [cm]
    gait_v, steps, velocity = calculate_gait_velocity2(t_resampled, filtered_az, peaks_ind, fs = fs, l = leg_length/100, number_of_steps = 'all')

    plt.figure(2)
    plt.figure(figsize=(7,4))
    plt.plot(t_resampled, ay, color="blue", label='Raw accelaration')
    plt.plot(t_resampled, filtered_ay, color="green", label='Filtered accelaration')
    plt.scatter(t_resampled[peaks_ind], filtered_ay[peaks_ind], marker = "x", s = 80, color="red", label='Heel strikes')
    plt.title('Accelaration in Y axis after analysis')
    plt.xlabel('Time [s]')
    plt.ylabel('Accelaration [m/s\u00b2]')
    plt.legend()
    a = plt.gcf()

    plt.figure(3)
    plt.figure(figsize=(7,4))
    plt.scatter(velocity[0], velocity[1], color="g")
    plt.xlabel('Time [s]')
    plt.ylabel('Velocity [m/s]')
    plt.title('Instantaneous velocity')
    b = plt.gcf()


    return gait_v, steps, a, b



# Columns for the layout

first_column = [
    [
        sg.Text('Import an acceleration file to see your signal:', font=('default', 12))
    ], 
    [
        sg.InputText(key="-FILE_PATH-"),
        sg.FileBrowse(initial_folder=working_directory, file_types=[("CSV Files", "*.csv")], font=('default', 12))
    ],
    [
        sg.Button('Generate the plot', font=('default', 12))
    ],
    [
        sg.Canvas(key="-CANVAS_1-")
    ]

]

second_column = [
    [
        sg.Text('Specify the length of your leg and analyze your signal:', font=('default', 12))
    ],
    [
        sg.InputText(key='-INPUT_TEXT-'),
        sg.Text('[cm]', font=('default', 12)),
        sg.Button('Analyze', font=('default', 12))
    ],
    [
        sg.Text("", key='-TEXT_1-', font=('default', 12))
    ],
    [
        sg.Text("", key='-TEXT_2-', font=('default', 12))
    ],
    [
        sg.Canvas(key="-CANVAS_2-")
    ],
    [
        sg.Canvas(key="-CANVAS_3-")
    ]

]



# ----- Full layout -----
layout = [
    [
        sg.Column(first_column),
        sg.VSeperator(),
        sg.Column(second_column)

    ]
]


window = sg.Window(title='Gait velocity analysis', layout = layout, finalize=True, element_justification='top')



# Create an event loop
while True:
    event, values = window.read()
    # End program if user closes window
    if event == sg.WIN_CLOSED:
        break

    elif event == 'Generate the plot':
        path = values["-FILE_PATH-"]
        draw_figure(window['-CANVAS_1-'].TKCanvas, create_raw_plot(path))

    elif event == 'Analyze':
        path = values["-FILE_PATH-"]
        leg_length = float(values['-INPUT_TEXT-'])
        gait_v, steps, plot1, plot2 = calculations(path, leg_length)
        window["-TEXT_1-"].update('Your average velocity on the given distance: {:.2f} [m/s].'.format(gait_v))
        window["-TEXT_2-"].update(f'You have made {steps} steps on the given distance. Keep going!')
        draw_figure(window['-CANVAS_2-'].TKCanvas, plot1)
        draw_figure(window['-CANVAS_3-'].TKCanvas, plot2)
        


window.close()