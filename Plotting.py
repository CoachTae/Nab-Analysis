import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

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


def plot_wf(
        data, 
        title=None, 
        xlabel="Time (4 ns)", 
        ylabel="Detector Reading (ADC)", 
        color="blue",
        marker=None, 
        linestyle="-", 
        grid=True, 
        figsize=(8, 6), 
        save_path=None
    ):
        """
        Plots a 1D array with optional customization.

        Parameters:
        - data (array-like): The 1D data array to plot.
        - title (str, optional): Title of the plot.
        - xlabel (str, optional): Label for the x-axis.
        - ylabel (str, optional): Label for the y-axis.
        - color (str, optional): Color of the line/markers.
        - marker (str, optional): Marker style (e.g., 'o', 'x').
        - linestyle (str, optional): Style of the line (e.g., '-', '--', ':').
        - grid (bool, optional): Whether to show a grid.
        - figsize (tuple, optional): Size of the figure in inches.
        - save_path (str, optional): If provided, saves the plot to this file path.
        """
        data = np.array(data)
        x = np.arange(len(data))  # equally spaced x values

        plt.figure(figsize=figsize)
        plt.plot(x, data, color=color, marker=marker, linestyle=linestyle)

        if title:
            plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        if grid:
            plt.grid(True)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()




def histogram(
        data,
        bins=10,
        range=None,
        density=False,
        color="blue",
        edgecolor="black",
        alpha=0.7,
        title=None,
        xlabel="Value",
        ylabel="Counts",
        grid=True,
        figsize=(8, 6),
        save_path=None
    ):
        """
        Plots a histogram for a 1D array with optional customization.

        Parameters:
        - data (array-like): Input 1D array.
        - bins (int or sequence, optional): Number of bins or bin edges.
        - range (tuple, optional): Lower and upper range of the bins.
        - density (bool, optional): If True, normalize histogram to show probability density.
        - color (str, optional): Fill color of the bars.
        - edgecolor (str, optional): Color of bin edges.
        - alpha (float, optional): Opacity of the bars.
        - title (str, optional): Title of the plot.
        - xlabel (str, optional): X-axis label.
        - ylabel (str, optional): Y-axis label.
        - grid (bool, optional): Show grid if True.
        - figsize (tuple, optional): Figure size in inches.
        - save_path (str, optional): If provided, saves the plot to this file path.
        """
        data = np.array(data)

        plt.figure(figsize=figsize)
        plt.hist(data, bins=bins, range=range, density=density,
                 color=color, edgecolor=edgecolor, alpha=alpha)

        if title:
            plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel if not density else "Probability Density")

        if grid:
            plt.grid(True, linestyle="--", alpha=0.6)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()
