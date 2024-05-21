"""
Get control parameters from motion capture data (Motive Optitrack file)

Two solid bodies:
    Bow: various markers, with two virtual markers at the extremes of the hair:
        hair_frog
        hair_tip
    Violin: various markers, with two virtual markers at the extremes of the
    G string:
        string_bridge
        string_nut

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import matplotlib.ticker as ticker
from matplotlib.ticker import MultipleLocator
import scipy.signal
import os

# deactivate latex
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "Times New Roman"
})

# activate latex
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Times New Roman"
})

# Working directory A CHANGER
main_path = os.path.expanduser('~') + '/Desktop/Stage Motion Capture/Inter-noise Data !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'
os.chdir(main_path)

# Sampling frequency
sf = 120

# Font for plots
font = font_manager.FontProperties(family='Times New Roman',
                                   style='normal', size=14)

def closest_point_on_line_r(p_r, v_r, p_s, v_s):
    """
    
    Parameters
    ----------
    p_r : numpy.ndarray shape (3,)
        point on line r.
    v_r : numpy.ndarray shape (3,)
        vector of line r.
    p_s : numpy.ndarray shape (3,)
        point on line s.
    v_s : numpy.ndarray shape (3,)
        vector of line s.

    Returns
    -------
    closest_point_on_r : numpy.ndarray shape (3,)
        closes point on line r to line s.

    """
    
    # Check if the lines are parallel by calculating the cross product of their
    # directional vectors
    if np.allclose(np.cross(v_r, v_s), [0, 0, 0]):
        # Lines are parallel, return any point on line r
        closest_point_on_r = p_r
    # elif
    else:
        # Lines are not intersecting, proceed with previous approach
        # Step 1: Find a vector normal to the plane containing both lines r
        # and s
        normal_vector = np.cross(v_r, v_s)

        # Step 2: Calculate the vector w from any point on line r to any point
        # on line s
        w = p_s - p_r

        # Step 3: Project the vector w onto the normal vector
        projection_length = np.dot(w, normal_vector) / np.dot(normal_vector,
                                                              normal_vector)
        projection = projection_length * normal_vector

        # Step 4: Subtract the projection from the vector w to get the vector v
        v = w - projection

        # Step 5: Add the vector v to the given point on line r to project
        # point s onto the plane parallel to s containing r
        projection_p_s = p_r + v
        
        # Step 6: Calculate direction perpendicular to s and parallel to the plane
        # perpendicular to r and s
        ortho_direction = np.cross(v_s, np.cross(v_r, v_s))
        
        # Step 7: Project the point projection_p_s on line r to obtain final point
        t = -np.dot(ortho_direction, p_r - projection_p_s) / \
        np.dot(ortho_direction, v_r)
        closest_point_on_r = p_r + t * v_r

    return closest_point_on_r

def load_trajectory_data(filename):
    """
    
    Parameters
    ----------
    filename : string
        name of the file to load data from.

    Returns
    -------
    hair_frog : numpy.ndarray
        position of the virtual marker of the hair at the frog.
    hair_tip : numpy.ndarray
        position of the virtual marker of the hair at the tip.
    string_bridge : numpy.ndarray
        position of the virtual marker of the string at the bridge.
    string_nut : numpy.ndarray
        position of the virtual marker of the string at the nut.
    time : numpy.ndarray
        time array.
    spiccato_ini : list, size 3
        Times at which the spiccato starts.
        Three trials -> three starting times.
    spiccato_end : list, size 3
        Times at which the spiccato ends.
        Three trials -> three ending times.

    """
    # Marker 6 is frog, marker 5 is tip
    trajectory_data = pd.read_csv(filename + '/' + filename +'-hair_frog.csv',
                                  delimiter=';')
    x_hair_frog = trajectory_data['X'].values
    y_hair_frog = trajectory_data['Y'].values
    z_hair_frog = trajectory_data['Z'].values
    hair_frog = np.array([x_hair_frog, y_hair_frog, z_hair_frog]).T;
    trajectory_data = pd.read_csv(filename + '/' + filename +'-hair_tip.csv',
                                  delimiter=';')
    x_hair_tip = trajectory_data['X'].values
    y_hair_tip = trajectory_data['Y'].values
    z_hair_tip = trajectory_data['Z'].values
    hair_tip = np.array([x_hair_tip, y_hair_tip, z_hair_tip]).T;
    # Marker 6 is bridge, marker 7 is nut
    trajectory_data = pd.read_csv(filename + '/' + filename +
                                  '-string_bridge.csv', delimiter=';')
    x_string_bridge = trajectory_data['X'].values
    y_string_bridge = trajectory_data['Y'].values
    z_string_bridge = trajectory_data['Z'].values
    string_bridge = np.array([x_string_bridge, y_string_bridge,
                              z_string_bridge]).T;
    trajectory_data = pd.read_csv(filename + '/' + filename +'-string_nut.csv',
                                  delimiter=';')
    x_string_nut = trajectory_data['X'].values
    y_string_nut = trajectory_data['Y'].values
    z_string_nut = trajectory_data['Z'].values
    string_nut = np.array([x_string_nut, y_string_nut, z_string_nut]).T;
    
    time = trajectory_data['Time'].values
    
    # Set limits of signals, from spiccato start to spiccato ending
    if filename == '120bpm-semiquavers_A-forte_clamp':
        spiccato_ini = [int(6.6*sf), int(60.8*sf), int(85.1*sf)]
        spiccato_end = [int(56.5*sf), int(80.4*sf), int(102.4*sf)]
    elif filename == '120bpm-semiquavers_A-forte_tip':
        spiccato_ini = [int(13*sf), int(31.3*sf), int(55.3*sf)]
        spiccato_end = [int(26.8*sf), int(50.7*sf), int(72.8*sf)]
    elif filename == '120bpm-semiquavers_A-piano_clamp':
        spiccato_ini = [int(14.2*sf), int(42.4*sf), int(64.3*sf)]
        spiccato_end = [int(38*sf), int(59.9*sf), int(84.2*sf)]
    elif filename == '120bpm-semiquavers_A-piano_tip':
        spiccato_ini = [int(5.2*sf), int(41.3*sf), int(69.5*sf)]
        spiccato_end = [int(35*sf), int(64.9*sf), int(103*sf)]
    elif filename == '120bpm-semiquavers_C-forte_clamp':
        spiccato_ini = [int(6*sf), int(32*sf), int(54.1*sf)]
        spiccato_end = [int(27.8*sf), int(49.8*sf), int(69.8*sf)]
    elif filename == '120bpm-semiquavers_C-forte_tip':
        spiccato_ini = [int(4.5*sf), int(32.5*sf), int(60.4*sf)]
        spiccato_end = [int(28.1*sf), int(56.2*sf), int(82.2*sf)]
    elif filename == '120bpm-semiquavers_C-piano_clamp':
        spiccato_ini = [int(7.2*sf), int(29.2*sf), int(57.2*sf)]
        spiccato_end = [int(23*sf), int(52.9*sf), int(77.1*sf)]
    elif filename == '120bpm-semiquavers_C-piano_tip':
        spiccato_ini = [int(5.5*sf), int(43.8*sf), int(69.8*sf)]
        spiccato_end = [int(39.2*sf), int(65.3*sf), int(101.3*sf)]
    
    return hair_frog, hair_tip, string_bridge, string_nut, time, \
        spiccato_ini, spiccato_end

def calculate_x_s(filename):
    """
    

    Parameters
    ----------
    filename : string
        name of the file to load data from.

    Returns
    -------
    time : numpy.ndarray
        time array.
    x_s : numpy.ndarray
        x_s array.
    spiccato_ini : list, size 3
        Times at which the spiccato starts.
        Three trials -> three starting times.
    spiccato_end : list, size 3
        Times at which the spiccato ends.
        Three trials -> three ending times.

    """
    [hair_frog, hair_tip, string_bridge, string_nut, time, \
     spiccato_ini, spiccato_end] = load_trajectory_data(filename)
    x_s = np.zeros_like(time)
    for i in range(len(time)):
        proy_string_hair = closest_point_on_line_r(hair_frog[i], hair_tip[i] - hair_frog[i],
                                        string_bridge[i], string_nut[i] - string_bridge[i])
        x_s[i] = np.linalg.norm(hair_frog[i] - proy_string_hair)
    
    return time, x_s, spiccato_ini, spiccato_end

def calculate_beta(filename):
    """
    

    Parameters
    ----------
    filename : string
        name of the file to load data from.

    Returns
    -------
    time : numpy.ndarray
        time array.
    beta : numpy.ndarray
        beta array.
    spiccato_ini : list, size 3
        Times at which the spiccato starts.
        Three trials -> three starting times.
    spiccato_end : list, size 3
        Times at which the spiccato ends.
        Three trials -> three ending times.
    string_length : numpy.float64
        Length of the string.

    """
    [hair_frog, hair_tip, string_bridge, string_nut, time, \
     spiccato_ini, spiccato_end] = load_trajectory_data(filename)
    # Calculate string length
    dist_bridgenut = np.zeros_like(time)
    for i in range(len(time)):
        dist_bridgenut[i] = np.linalg.norm(string_bridge[i]-string_nut[i])
    string_length = np.nanmean(dist_bridgenut[dist_bridgenut > 0])
    # Calculate beta
    beta = np.zeros_like(time)
    for i in range(len(time)):
        proy_hair_string = closest_point_on_line_r(string_bridge[i], string_nut[i] - string_bridge[i],
                                                   hair_frog[i], hair_tip[i] - hair_frog[i])
        beta[i] = np.linalg.norm(string_bridge[i] - proy_hair_string)/string_length
    
    return time, beta, spiccato_ini, spiccato_end, string_length

def calculate_dist_hairstring(filename):
    """
    

    Parameters
    ----------
    filename : string
        name of the file to load data from.

    Returns
    -------
    time : numpy.ndarray
        time array.
    dist_hairstring : numpy.ndarray
        dist_hairstring array.
    spiccato_ini : list, size 3
        Times at which the spiccato starts.
        Three trials -> three starting times.
    spiccato_end : list, size 3
        Times at which the spiccato ends.
        Three trials -> three ending times.

    """
    [hair_frog, hair_tip, string_bridge, string_nut, time, \
     spiccato_ini, spiccato_end] = load_trajectory_data(filename)
    v_plane = np.empty_like(hair_frog)
    dist_hairstring = np.zeros_like(time)
    for i in range(len(time)):
        v_plane[i] = np.cross(hair_tip[i] - hair_frog[i],
                              string_bridge[i] - string_nut[i])
        dist_hairstring[i] = np.dot(hair_frog[i]-string_nut[i],
                                    v_plane[i])/np.linalg.norm(v_plane[i])
    
    return time, dist_hairstring, spiccato_ini, spiccato_end

def plot_x_s_all():

    # size of the time segment from which mean distances are calculated, starting from the end of the spiccato
    time_segment = 5

    font = font_manager.FontProperties(family='Times New Roman',
                                       style='normal', size=16)

    def plot_x_s(filename, color):
        [time, x_s, spiccato_ini, spiccato_end] = calculate_x_s(filename)
        for i in range(3):
            # Remove NANs
            for j in range(len(x_s[spiccato_ini[i]:spiccato_end[i]])):
                if np.isnan(x_s[spiccato_ini[i]+j])==True:
                    x_s[spiccato_ini[i]+j] = np.nanmean(x_s[spiccato_ini[i]+j-60:spiccato_ini[i]+j-1])
            ax[0].plot(time[spiccato_ini[i]:spiccato_end[i]]-time[spiccato_ini[i]],
                     scipy.signal.savgol_filter(x_s[spiccato_ini[i]:spiccato_end[i]]*100, 51, 2),
                     # x_s[spiccato_ini[i]:spiccato_end[i]]*100,
                     color=color, alpha=0.5)
            ax[0].hlines(np.mean(scipy.signal.savgol_filter(x_s[spiccato_end[i]-time_segment*sf:spiccato_end[i]]*100, 51, 2)), time[spiccato_end[i]-time_segment*sf]-time[spiccato_ini[i]], time[spiccato_end[i]]-time[spiccato_ini[i]], color=color)

    def mean_std_x_s(filename):
        [time, x_s, spiccato_ini, spiccato_end] = calculate_x_s(filename)
        mean = np.zeros([1,3])[0]
        std = np.zeros([1,3])[0]
        for i in range(3):
            # Remove NANs
            for j in range(len(x_s[spiccato_ini[i]:spiccato_end[i]])):
                if np.isnan(x_s[spiccato_ini[i]+j])==True:
                    x_s[spiccato_ini[i]+j] = np.nanmean(x_s[spiccato_ini[i]+j-60:spiccato_ini[i]+j-1])
            mean[i] = np.mean(x_s[spiccato_end[i]-time_segment*sf:spiccato_end[i]]*100)
            std[i] = np.std(x_s[spiccato_end[i]-time_segment*sf:spiccato_end[i]]*100)
        return mean, std

    def mean_std_x_s_all(filename_trial):
        print('---------------------------------')
        print('--------> Trial', filename_trial[0][19:26])
        x_s_trial_all = np.array([])
        
        # Each trial has 2 cases: starting from clamp or from tip
        for case in [0,1]:
            filename = filename_trial[case]
            
            [time, x_s, spiccato_ini, spiccato_end] = calculate_x_s(filename)

            x_s_case_all = np.array([])
            
            for i in range(3):
                x_s_segment = x_s[spiccato_end[i]-time_segment*sf:spiccato_end[i]]
                # Remove NANs
                for j in range(len(x_s_segment)):
                    if np.isnan(x_s_segment[j])==True:
                        x_s_segment[j] = np.nanmean(x_s[spiccato_end[i]-time_segment*sf+j-60:spiccato_end[i]-time_segment*sf+j-1])
                x_s_segment = scipy.signal.savgol_filter(x_s_segment, 51, 2)
                x_s_case_all = np.append(x_s_case_all, x_s_segment*100)
                x_s_trial_all = np.append(x_s_trial_all, x_s_segment*100)
            print('Case: ', filename)
            print('All three trials : mean x_s = ', np.mean(x_s_case_all))
            print('All three trials : std x_s  = ', np.std(x_s_case_all))
        print('--------')
        print('All six trials : mean x_s = ', np.mean(x_s_trial_all))
        print('All six trials : std x_s  = ', np.std(x_s_trial_all))
        return np.mean(x_s_trial_all), np.std(x_s_trial_all)


    fig, ax = plt.subplots(1, 2, figsize=(12, 6),
                           gridspec_kw={'width_ratios': [6, 1]}, sharey=True)
    fig.subplots_adjust(wspace=0)

    filename = '120bpm-semiquavers_C-piano_clamp'
    plot_x_s(filename, color='red')
    [mean_clamp, std_clamp] = mean_std_x_s(filename)
    print(filename)
    print(mean_clamp)
    print(std_clamp)

    filename = '120bpm-semiquavers_C-piano_tip'
    plot_x_s(filename, color='red')
    [mean_tip, std_tip] = mean_std_x_s(filename)
    print(filename)
    print(mean_tip)
    print(std_tip)

    filename_trial = ['120bpm-semiquavers_C-piano_clamp',
     '120bpm-semiquavers_C-piano_tip']

    [mean_all, std_all] = mean_std_x_s_all(filename_trial)

    ax[1].hlines(mean_all, 0, 1, color='red',
                 label=r'Bow $C$, \textit{piano}')
    ax[1].errorbar(0.8, mean_all,
                   yerr=std_all, capsize=2, color='red')

    filename = '120bpm-semiquavers_A-piano_clamp'
    plot_x_s(filename, color='orange')
    [mean_clamp, std_clamp] = mean_std_x_s(filename)
    print(filename)
    print(mean_clamp)
    print(std_clamp)

    filename = '120bpm-semiquavers_A-piano_tip'
    plot_x_s(filename, color='orange')
    [mean_tip, std_tip] = mean_std_x_s(filename)
    print(filename)
    print(mean_tip)
    print(std_tip)

    filename_trial = ['120bpm-semiquavers_A-piano_clamp',
     '120bpm-semiquavers_A-piano_tip']

    [mean_all, std_all] = mean_std_x_s_all(filename_trial)

    ax[1].hlines(mean_all, 0, 1, color='orange',
                 label=r'Bow $A$, \textit{piano}')
    ax[1].errorbar(0.6, mean_all,
                   yerr=std_all, capsize=2, color='orange')

    filename = '120bpm-semiquavers_C-forte_clamp'
    plot_x_s(filename, color='green')
    [mean_clamp, std_clamp] = mean_std_x_s(filename)
    print(filename)
    print(mean_clamp)
    print(std_clamp)

    filename = '120bpm-semiquavers_C-forte_tip'
    plot_x_s(filename, color='green')
    [mean_tip, std_tip] = mean_std_x_s(filename)
    print(filename)
    print(mean_tip)
    print(std_tip)

    filename_trial = ['120bpm-semiquavers_C-forte_clamp',
     '120bpm-semiquavers_C-forte_tip']

    [mean_all, std_all] = mean_std_x_s_all(filename_trial)

    ax[1].hlines(mean_all, 0, 1, color='green',
                 label=r'Bow $C$, \textit{forte}')
    ax[1].errorbar(0.4, mean_all,
                   yerr=std_all, capsize=2, color='green')

    filename = '120bpm-semiquavers_A-forte_clamp'
    plot_x_s(filename, color='blue')
    [mean_clamp, std_clamp] = mean_std_x_s(filename)
    print(filename)
    print(mean_clamp)
    print(std_clamp)

    filename = '120bpm-semiquavers_A-forte_tip'
    plot_x_s(filename, color='blue')
    [mean_tip, std_tip] = mean_std_x_s(filename)
    print(filename)
    print(mean_tip)
    print(std_tip)

    filename_trial = ['120bpm-semiquavers_A-forte_clamp',
     '120bpm-semiquavers_A-forte_tip']

    [mean_all, std_all] = mean_std_x_s_all(filename_trial)

    ax[1].hlines(mean_all, 0, 1, color='blue',
                 label=r'Bow $A$, \textit{forte}')
    ax[1].errorbar(0.2, mean_all,
                   yerr=std_all, capsize=2, color='blue')

    ax[0].set_xlabel('Time (s)', fontname="Times New Roman", fontsize=18)
    ax[1].set_xlabel('Averages', fontname="Times New Roman", fontsize=16)
    ax[0].set_ylabel(r'Bowing point-frog distance $x_S$ (cm)', fontname="Times New Roman",
                 fontsize=18)
    ax[0].set_xlim([0, 51])
    ax[0].set_ylim([15, 65])
    ax[0].yaxis.set_major_locator(MultipleLocator(5))
    ax[0].yaxis.set_minor_locator(MultipleLocator(1))
    ax[0].set_yticklabels(ax[0].get_yticks(), FontProperties=font)
    ax[0].yaxis.set_major_formatter(ticker.FormatStrFormatter('%i'.format(0)))
    ax[0].xaxis.set_major_locator(MultipleLocator(10))
    ax[0].xaxis.set_minor_locator(MultipleLocator(5))
    ax[0].set_xticklabels(ax[0].get_xticks(), FontProperties=font)
    ax[0].xaxis.set_major_formatter(ticker.FormatStrFormatter('%i'.format(0)))
    ax[1].tick_params(
        axis='x',           # changes apply to the x-axis
        which='both',       # both major and minor ticks are affected
        bottom=False,       # ticks along the bottom edge are off
        top=False,          # ticks along the top edge are off
        labelbottom=False)  # labels along the bottom edge are off
    plt.legend(bbox_to_anchor=(-1.7, 0.65, 1.6, 1), loc="lower left",
                    mode="expand", borderaxespad=0, ncol=1,
                    prop=font, framealpha=1)
    ax[0].grid(axis='y')
    ax[1].grid(axis='y')

    # plt.savefig('../../../Inter-Noise 2024/Inter_Noise2024_Paper_Template_LaTeX/figure/Bowing point-frog distance.png', dpi=600)

def plot_beta_all():

    # size of the time segment from which mean distances are calculated, starting from the end of the spiccato
    time_segment = 5

    font = font_manager.FontProperties(family='Times New Roman',
                                       style='normal', size=16)

    def plot_beta(filename, color):
        [time, beta, spiccato_ini, spiccato_end, string_length] = calculate_beta(filename)
        for i in range(3):
            # Remove NANs
            for j in range(len(beta[spiccato_ini[i]:spiccato_end[i]])):
                if np.isnan(beta[spiccato_ini[i]+j])==True:
                    beta[spiccato_ini[i]+j] = np.nanmean(beta[spiccato_ini[i]+j-60:spiccato_ini[i]+j-1])
            ax[0].plot(time[spiccato_ini[i]:spiccato_end[i]]-time[spiccato_ini[i]],
                       scipy.signal.savgol_filter(beta[spiccato_ini[i]:spiccato_end[i]]*string_length*1000, 51, 2),
                       # beta[spiccato_ini[i]:spiccato_end[i]]*string_length*100,
                       color=color, alpha=0.5)
            ax[0].hlines(np.mean(scipy.signal.savgol_filter(beta[spiccato_end[i]-time_segment*sf:spiccato_end[i]]*string_length*1000, 51, 2)), time[spiccato_end[i]-time_segment*sf]-time[spiccato_ini[i]], time[spiccato_end[i]]-time[spiccato_ini[i]], color=color)

    def mean_std_beta(filename):
        [time, beta, spiccato_ini, spiccato_end, string_length] = calculate_beta(filename)
        mean = np.zeros([1,3])[0]
        std = np.zeros([1,3])[0]
        for i in range(3):
            # Remove NANs
            for j in range(len(beta[spiccato_ini[i]:spiccato_end[i]])):
                if np.isnan(beta[spiccato_ini[i]+j])==True:
                    beta[spiccato_ini[i]+j] = np.nanmean(beta[spiccato_ini[i]+j-60:spiccato_ini[i]+j-1])
            mean[i] = np.mean(beta[spiccato_end[i]-time_segment*sf:spiccato_end[i]]*string_length*1000)
            std[i] = np.std(beta[spiccato_end[i]-time_segment*sf:spiccato_end[i]]*string_length*1000)
        return mean, std

    def mean_std_beta_all(filename_trial):
        print('---------------------------------')
        print('--------> Trial', filename_trial[0][19:26])
        beta_trial_all = np.array([])
        
        # Each trial has 2 cases: starting from clamp or from tip
        for case in [0,1]:
            filename = filename_trial[case]
            
            [time, beta, spiccato_ini, spiccato_end, string_length] = calculate_beta(filename)

            beta_case_all = np.array([])
            
            for i in range(3):
                beta_segment = beta[spiccato_end[i]-time_segment*sf:spiccato_end[i]]
                # Remove NANs
                for j in range(len(beta_segment)):
                    if np.isnan(beta_segment[j])==True:
                        beta_segment[j] = np.nanmean(beta[spiccato_end[i]-time_segment*sf+j-60:spiccato_end[i]-time_segment*sf+j-1])
                beta_segment = scipy.signal.savgol_filter(beta_segment, 51, 2)
                beta_case_all = np.append(beta_case_all, beta_segment*string_length*1000)
                beta_trial_all = np.append(beta_trial_all, beta_segment*string_length*1000)
            print('Case: ', filename)
            print('All three trials : mean beta = ', np.mean(beta_case_all))
            print('All three trials : std beta  = ', np.std(beta_case_all))
        print('--------')
        print('All six trials : mean beta = ', np.mean(beta_trial_all))
        print('All six trials : std beta  = ', np.std(beta_trial_all))
        return np.mean(beta_trial_all), np.std(beta_trial_all)

    fig, ax = plt.subplots(1, 2, figsize=(12, 6),
                           gridspec_kw={'width_ratios': [6, 1]}, sharey=True)
    fig.subplots_adjust(wspace=0)

    filename = '120bpm-semiquavers_C-piano_clamp'
    plot_beta(filename, color='red')
    [mean_clamp, std_clamp] = mean_std_beta(filename)
    print(filename)
    print(mean_clamp)
    print(std_clamp)

    filename = '120bpm-semiquavers_C-piano_tip'
    plot_beta(filename, color='red')
    [mean_tip, std_tip] = mean_std_beta(filename)
    print(filename)
    print(mean_tip)
    print(std_tip)

    filename_trial = ['120bpm-semiquavers_C-piano_clamp',
     '120bpm-semiquavers_C-piano_tip']

    [mean_all, std_all] = mean_std_beta_all(filename_trial)

    ax[1].hlines(mean_all, 0, 1, color='red',
                 label=r'Bow $C$, \textit{piano}')
    ax[1].errorbar(0.8, mean_all,
                   yerr=std_all, capsize=2, color='red')

    filename = '120bpm-semiquavers_A-piano_clamp'
    plot_beta(filename, color='orange')
    [mean_clamp, std_clamp] = mean_std_beta(filename)
    print(filename)
    print(mean_clamp)
    print(std_clamp)

    filename = '120bpm-semiquavers_A-piano_tip'
    plot_beta(filename, color='orange')
    [mean_tip, std_tip] = mean_std_beta(filename)
    print(filename)
    print(mean_tip)
    print(std_tip)

    filename_trial = ['120bpm-semiquavers_A-piano_clamp',
     '120bpm-semiquavers_A-piano_tip']

    [mean_all, std_all] = mean_std_beta_all(filename_trial)

    ax[1].hlines(mean_all, 0, 1, color='orange',
                 label=r'Bow $A$, \textit{piano}')
    ax[1].errorbar(0.6, mean_all,
                   yerr=std_all, capsize=2, color='orange')

    filename = '120bpm-semiquavers_C-forte_clamp'
    plot_beta(filename, color='green')
    [mean_clamp, std_clamp] = mean_std_beta(filename)
    print(filename)
    print(mean_clamp)
    print(std_clamp)

    filename = '120bpm-semiquavers_C-forte_tip'
    plot_beta(filename, color='green')
    [mean_tip, std_tip] = mean_std_beta(filename)
    print(filename)
    print(mean_tip)
    print(std_tip)

    filename_trial = ['120bpm-semiquavers_C-forte_clamp',
     '120bpm-semiquavers_C-forte_tip']

    [mean_all, std_all] = mean_std_beta_all(filename_trial)

    ax[1].hlines(mean_all, 0, 1, color='green',
                 label=r'Bow $C$, \textit{forte}')
    ax[1].errorbar(0.4, mean_all,
                   yerr=std_all, capsize=2, color='green')

    filename = '120bpm-semiquavers_A-forte_clamp'
    plot_beta(filename, color='blue')
    [mean_clamp, std_clamp] = mean_std_beta(filename)
    print(filename)
    print(mean_clamp)
    print(std_clamp)

    filename = '120bpm-semiquavers_A-forte_tip'
    plot_beta(filename, color='blue')
    [mean_tip, std_tip] = mean_std_beta(filename)
    print(filename)
    print(mean_tip)
    print(std_tip)

    filename_trial = ['120bpm-semiquavers_A-forte_clamp',
     '120bpm-semiquavers_A-forte_tip']

    [mean_all, std_all] = mean_std_beta_all(filename_trial)

    ax[1].hlines(mean_all, 0, 1, color='blue',
                 label=r'Bow $A$, \textit{forte}')
    ax[1].errorbar(0.2, mean_all,
                   yerr=std_all, capsize=2, color='blue')

    ax[0].set_xlabel('Time (s)', fontname="Times New Roman", fontsize=18)
    ax[1].set_xlabel('Averages', fontname="Times New Roman", fontsize=16)
    ax[0].set_ylabel(r'Bowing point-bridge distance $x_H$ (mm)', fontname="Times New Roman",
                 fontsize=18)
    ax[0].set_xlim([0, 51])
    ax[0].set_ylim([15, 70])
    ax[0].yaxis.set_major_locator(MultipleLocator(5))
    ax[0].yaxis.set_minor_locator(MultipleLocator(1))
    ax[0].set_yticklabels(ax[0].get_yticks(), FontProperties=font)
    ax[0].yaxis.set_major_formatter(ticker.FormatStrFormatter('%i'.format(0)))
    ax[0].xaxis.set_major_locator(MultipleLocator(10))
    ax[0].xaxis.set_minor_locator(MultipleLocator(5))
    ax[0].set_xticklabels(ax[0].get_xticks(), FontProperties=font)
    ax[0].xaxis.set_major_formatter(ticker.FormatStrFormatter('%i'.format(0)))
    ax[1].tick_params(
        axis='x',           # changes apply to the x-axis
        which='both',       # both major and minor ticks are affected
        bottom=False,       # ticks along the bottom edge are off
        top=False,          # ticks along the top edge are off
        labelbottom=False)  # labels along the bottom edge are off
    plt.legend(bbox_to_anchor=(-1.7, 0.65, 1.6, 1), loc="lower left",
                    mode="expand", borderaxespad=0, ncol=1,
                    prop=font, framealpha=1)
    ax[0].grid(axis='y')
    ax[1].grid(axis='y')

    # plt.savefig('figure/Bowing point-bridge distance.png', dpi=600)

def plot_dist_hairstring_all():
    # size of the time segment from which hair-string distances are plotted, starting from the end of the spiccato
    time_segment = 2

    font = font_manager.FontProperties(family='Times New Roman',
                                       style='normal', size=12)

    fig, ax = plt.subplots(figsize=(8,5.5))
    # plt.title('Hair-string distances', fontname="Times New Roman", fontsize=12)
    filename = '120bpm-semiquavers_C-piano_clamp'
    [time, dist_hairstring, spiccato_ini, spiccato_end] = calculate_dist_hairstring(filename)
    plt.plot(time[spiccato_end[0]-time_segment*sf:spiccato_end[0]]-time[spiccato_end[0]-time_segment*sf],
             60+dist_hairstring[spiccato_end[0]-time_segment*sf:spiccato_end[0]]*1000,
             color='red', label=r'Bow $C$, \textit{piano}')
    # plt.hlines(6, 0, time_segment, color='black', alpha = 0.2)
    filename = '120bpm-semiquavers_A-piano_clamp'
    [time, dist_hairstring, spiccato_ini, spiccato_end] = calculate_dist_hairstring(filename)
    plt.plot(time[spiccato_end[1]-time_segment*sf:spiccato_end[1]]-time[spiccato_end[1]-time_segment*sf],
             40+dist_hairstring[spiccato_end[1]-time_segment*sf:spiccato_end[1]]*1000,
             color='orange', label=r'Bow $A$, \textit{piano}')
    # plt.hlines(4, 0, time_segment, color='black', alpha = 0.2)
    filename = '120bpm-semiquavers_C-forte_clamp'
    [time, dist_hairstring, spiccato_ini, spiccato_end] = calculate_dist_hairstring(filename)
    plt.plot(time[spiccato_end[0]-time_segment*sf:spiccato_end[0]]-time[spiccato_end[0]-time_segment*sf],
             20+dist_hairstring[spiccato_end[0]-time_segment*sf:spiccato_end[0]]*1000,
             color='green', label=r'Bow $C$, \textit{forte}')
    # plt.hlines(2, 0, time_segment, color='black', alpha = 0.2)
    filename = '120bpm-semiquavers_A-forte_clamp'
    [time, dist_hairstring, spiccato_ini, spiccato_end] = calculate_dist_hairstring(filename)
    ini = int(spiccato_end[2]-9.5*sf)
    end = int(spiccato_end[2]-7.5*sf)
    plt.plot(time[ini:end]-time[ini],
             dist_hairstring[ini:end]*1000,
             color='blue', label=r'Bow $A$, \textit{forte}')
    # plt.hlines(0, 0, time_segment, color='black', alpha = 0.2)

    # ax.set_yticks(np.linspace(-1,7,17))
    plt.ylim([-10,70])
    ax.yaxis.set_major_locator(MultipleLocator(5))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%i'.format(0)))
    ax.yaxis.set_minor_locator(MultipleLocator(1))
    ax.set_yticklabels([' ', ' ',
                        '-5', '0', '+5', ' ',
                        '-5', '0', '+5', ' ',
                        '-5', '0', '+5', ' ',
                        '-5', '0', '+5', ' '],
                       fontname="Times New Roman", fontsize=12)
    plt.xlim([0,2])
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax.set_xticklabels(ax.get_xticks(), FontProperties=font)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%i'.format(0)))
    plt.xlabel('Time (s)', fontname="Times New Roman", fontsize=12)
    plt.ylabel('Hair-string distance (mm)', fontname="Times New Roman", fontsize=12)
    plt.grid(axis='y')

    plt.legend(bbox_to_anchor=(0.25, 1.02, 0.5, 0.2), loc="lower left",
                    mode="expand", borderaxespad=0, ncol=2,
                    prop=font, framealpha=1)
    
    # plt.savefig('figure/Hair-string distance.png', dpi=600)

if __name__ == "__main__":
    # plot_x_s_all()
    # plot_beta_all()
    plot_dist_hairstring_all()