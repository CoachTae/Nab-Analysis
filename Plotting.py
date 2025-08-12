import numpy as np
import matplotlib.pyplot as plt

def plot_ADC_counts_WFFile(waveformFile, title=''):
    '''
    Plots counts vs ADC channels.

    Parameters:
        data: waveformFile objects like File.noiseWaves()

    '''

    fig, ax = plt.subplots()
    
    counts = {}
    for i in range(10000):
        try:
            data = waveformFile.wave(i)
            for reading in data:
                if reading in counts.keys():
                    counts[reading] += 1
                else:
                    counts[reading] = 1
        except:
            break

    print(f'{i} waveforms recorded.')

    x = list(counts.keys())
    y = list(counts.values())

    ax.set_xlabel('ADC Channel')
    ax.set_ylabel('Counts')
    ax.set_title(title)
    plt.plot(x,y, 'o')
    plt.show()


def plot_ADC_counts_lists(WF_List, title=''):
    '''
    Same as above, but takes in a list of waveforms (lists).
    Effectively a list of lists. Each entry of WF_List should be a waveform (also a list) itself.

    Parameters:
        WF_List: A list of waveforms (also lists).
        title: Title for the plot
    '''
    fig, ax = plt.subplots()
    
    counts = {}
    for waveform in WF_List:
        for reading in waveform:
            if reading in counts.keys():
                counts[reading] += 1
            else:
                counts[reading] = 1

    #print(f'{len(WF_List)} waveforms recorded.')

    x = list(counts.keys())
    y = list(counts.values())

    ax.set_xlabel('ADC Channel')
    ax.set_ylabel('Counts')
    ax.set_title(title)
    plt.plot(x, y, 'o')
    plt.show()




def plot_stacked_waveforms(waveform1, waveform2, x1=None, x2=None, 
                          label1="Waveform 1", label2="Waveform 2", 
                          xlabel="Time", ylabel="Amplitude", title=None):
    """
    Plots two waveforms in vertically stacked subplots.
    Optionally specify x1 and x2 for x-axis values (defaults to sample number).
    """
    if x1 is None:
        x1 = np.arange(len(waveform1))
    if x2 is None:
        x2 = np.arange(len(waveform2))
        
    fig, axs = plt.subplots(2, 1, sharex=False, figsize=(10, 6))
    axs[0].plot(x1, waveform1)
    axs[0].set_ylabel(ylabel)
    axs[0].set_title(label1)
    axs[0].grid(True)
    
    axs[1].plot(x2, waveform2)
    axs[1].set_xlabel(xlabel)
    axs[1].set_ylabel(ylabel)
    axs[1].set_title(label2)
    axs[1].grid(True)
    
    if title:
        fig.suptitle(title)
    
    plt.tight_layout()
    plt.show()
