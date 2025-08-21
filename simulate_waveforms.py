import math
import numpy as np
import random

#%%
def sim_wf(t0, amplitude, rise_time, decay):
    """
    Simulate a waveform with fixed time base from 0 to 7000.
    The waveform rises linearly to a set energy then decays exponentially.

    Parameters:
        t0 (float): Time of start of waveform (in samples)
        amplitude (float): Peak height
        rise_time (float): Time to reach the peak (in samples)
        decay (float): Decay time constant (in samples)

    Returns:
        t (np.ndarray): Time array from 0 to 7000
        wf (np.ndarray): Waveform values, length 7001
        energy (float): Integral (sum) of waveform
    """
    t = np.arange(0, 7000)  # fixed time base
    wf = np.zeros_like(t, dtype=np.float64)

    t0 = int(t0)
    rise_end = t0 + int(rise_time)

    # Linear rise
    for i in range(t0, min(rise_end, len(wf))):
        wf[i] = amplitude * (i - t0) / rise_time

    # Exponential decay
    for i in range(rise_end, len(wf)):
        wf[i] = amplitude * np.exp(-(i - rise_end) / decay)

    # Energy = sum since dt = 1
    energy = np.sum(wf)

    return t, wf, energy



def batch_wfs(t0, amplitude, rise_time, decay,
              t0min=None, t0max=None, t0mean=None, t0var=None,
              ampmin=None, ampmax=None, ampmean=None, ampvar=None,
              rise_timemin=None, rise_timemax=None, rise_timemean=None, rise_timevar=None,
              decaymin=None, decaymax=None, decaymean=None, decayvar=None,
              N=None):
    '''
    Creates a batch of waveforms in a 2D numpy matrix.

    This method gives 3 options for choosing non-static parameters:
        - Create N waveforms with non-static parameters equally spaced out (doing a scan)
        - Create N waveforms with non-static parameters randomly chosen
        - Same as option 2 but the randomness pulls from a normal distribution if given a mean and variance

    To select the 1st method, set the desired variable to the string "'linear'" and provide a min and max for that parameter.
    To select the 2nd method, set the desired variable to the string "'random'"
    To select the 3rd method, set the desired variable to the string "'gaussian'"

    Parameters:
        t0 - Time that the waveform begins between 0 and 7000
        amplitude - Highest value that the waveform reaches before decaying
        rise_time - Amount of time it takes to reach the highest value
        decay - The tau exponential decay constant

        (Parameter)min - Minimum allowed value for the given parameter
        (Parameter)max - Maximum allowed value for the given parameter
        (Parameter)mean - Mean value for given parameter (assuming it's normally distributed)
        (Parameter)var - Variance for given parameter (assuming it's normally distributed)

        N - Number of waveforms to be created.
    '''

    def get_values(x, xmin, xmax, xmean, xvar, N):
        # For parameter scans
        if x.lower() == 'linear':
            return np.linspace(xmin, xmax, N).tolist()

        elif x.lower() == 'random':
            xvals = np.random.uniform(xmin, xmax, N)
            
        elif x.lower() == 'gaussian':
            xvals = []
            # Safeguard against infinite loops
            max_trials = 100000 * N

            trials = 0
            while len(xvals) < N and trials < max_trials:
                sample = random.gauss(xmean, math.sqrt(xstd))
                trials += 1
                if xmin <= sample <= xmax:
                    xvals.append(sample)
            xvals = np.array(xvals)

            if len(xvals) < N:
                raise RuntimeError(f'Could not generate {N} valid samples after {trials} trials.')
        return xvals


    if type(t0) is str:
        t0vals = get_values(t0, t0min, t0max, t0mean, t0var, N)
    else:
        t0vals = [t0]*N

    if type(amplitude) is str:
        ampvals = get_values(amplitude, ampmin, ampmax, ampmean, ampvar, N)
    else:
        ampvals = [amplitude]*N
        
    if type(rise_time) is str:
        rise_timevals = get_values(rise_time, rise_timemin, rise_timemax, rise_timemean, rise_timevar, N)
    else:
        rise_timevals = [rise_time]*N
        
    if type(decay) is str:
        decayvals = get_values(decay, decaymin, decaymax, decaymean, decayvar, N)
    else:
        decayvals = [decay]*N

    wfs = []
    wfs = [sim_wf(t0vals[i], ampvals[i], rise_timevals[i], decayvals[i])[1] for i in range(N)]

    wfs = np.array(wfs)
    return wfs

def gaussian_noise(mean=0.0, std=1.0):
    """
    Generate Gaussian (normal) noise.

    Parameters:
        length (int): Number of samples (typically 14001 for your waveform)
        mean (float): Mean of the noise
        std (float): Standard deviation of the noise

    Returns:
        noise (np.ndarray): Noise array of length `length`
    """
    return np.random.normal(loc=mean, scale=std, size=7000)
#%%

if __name__ == '__main__':
    import sys
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    import random
    import time
    sys.path.append('/Users/akannan/Downloads/Lab/Nab-Analysis')
    import Paths
    from scipy.interpolate import interp1d

    start_time = time.time()
    User = 'arush'

    if User.lower() == 'skylar':
        paths = Paths.Skylar_Paths

    elif User.lower() == 'arush':
        paths = Paths.Arush_Paths

    # Path for PyNab package
    sys.path.append(paths[0])
    # Path for deltarice (package created by David Matthews for Nab)
    sys.path.append(paths[1])
    import nabPy as Nab
    import h5py
    
    dataLoc = '/Users/akannan/Downloads/'
    num = 7616
    dataFile = Nab.File(f'{dataLoc}Run{num}_1.h5')
    run = Nab.DataRun(paths[2], 7616)
    parameters = run.parameterFile()


    coinc = run.coincWaves()
    noise = run.noiseWaves()



    kwargs = {'alpha': 0.75,'cmap': 'plasma','logNorm': True}
    kwargsPlot = {'labels':np.asarray(np.arange(1,128),dtype=str), 'labelValues': True}
    fig = dataFile.plotHitLocations(plot = True, sourceFile='noise', kwargsFig = kwargs, kwargsPlot=kwargsPlot)

    fig



    #%%
    headers = noise.headers()
    indices = headers.index.tolist()

    noise.resetCuts()
    i = random.choice(indices)
    s1 =  noise.wave(i)
    i = random.choice(indices)
    s2 = noise.wave(i)
    i = random.choice(indices)
    s3 = noise.wave(i)

    st = np.append(s1, s2)
    stt = np.append(st, s3)


    avg = np.mean(stt)
    std = np.std(stt)

    print(std)


    #%%
    i=random.randint(0, 200)
    w1=  coinc.wave(i)
    t1 = np.arange(len(w1))
    plt.figure(figsize=(12, 6))
    plt.plot(t1, w1, label="Raw Waveform")
    plt.title(f"Coinc {i}")
    plt.xlabel("Time")
    plt.legend()
    plt.tight_layout()
    plt.show()


    noise = gaussian_noise(mean=avg, std=std)

    t0=6000
    amp = 40
    risetime = 10
    exp_decay_param = 2000

    t, wf, energy = sim_wf(t0=t0, amplitude=amp, rise_time=risetime, decay=exp_decay_param)

    waveform = wf + noise



    plt.figure(figsize=(12, 6))
    plt.plot(t, waveform)
    plt.xlim(t[0], t[-1])
    plt.xlabel("Time")
    plt.ylabel("Energy")
    plt.title("Simulated Waveform")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()








   
