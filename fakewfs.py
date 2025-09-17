import math
import numpy as np
import random
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import time
import dask.array as da
import tensorflow
#%% 1
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
                sample = random.gauss(xmean, math.sqrt(xvar))
                trials += 1
                if xmin <= sample <= xmax:
                    xvals.append(sample)

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
        length (int): Number of samples (typically 7001 for your waveform)
        mean (float): Mean of the noise
        std (float): Standard deviation of the noise

    Returns:
        noise (np.ndarray): Noise array of length `length`
    """
    return np.random.normal(loc=mean, scale=std, size=7000)


def generate_training_data(noise, pixel = None, ampmin = 0, ampmax = 80, ampmean = 35, ampvar = 225):
    '''
    Generates training data for use in a logistic regression model.
    Maximizes the training waves based on avaliable noise waveforms from datarun.
    Note the following "specilized" modules must be imported and initialized prior to running this function:
        from dask_ml.preprocessing import StandardScaler, Nabpy.
    Also note, the waveforms are generated to simulate proton waveforms from the entire energy spectra.
    i.e decay, rise time, 
    
    Paramaters:
        noise (Nab waveformFileClass object): noise from a specified datarun. 
        pixel (int): pixel number from which noise will be pulled. If none is specified, run will take sample from all pixels
        ampmin (int): minimum amplitude of waveforms (initialized to 0, the min physical proton energy)
        ampmax (int): maximum amplitude of waveforms (initialized to 80, the max physical proton energy)
        ampmean (int): mean amplitude (initialized to 35, amplitude mean physical proton energy)
        ampvar (int): mean variance (initialized to 225, amplitude max physical proton energy)
    Return:
        X (dask.array): Standardized waveforms (Dask array). 
        Y (dask.array): Labels (1 for proton, 0 for noise).
        scalar (dask_ml.preprocessing StandardScaler): scaler for future testing use
                                                     
    '''
    
    if pixel != None:
        noise.resetCuts()
        noise.defineCut("pixel", "=", pixel)
        numwaves = noise.headers().shape[0]
        simwfs = batch_wfs(t0="random", amplitude="gaussian", rise_time="random", decay="gaussian",
                      t0min=3420, t0max=3500,
                      ampmin = ampmean, ampmax = ampmax, ampmean=ampmean, ampvar=ampvar,
                      rise_timemin=8, rise_timemax=22,
                      decaymin = 1200, decaymax = 1300, decaymean=1259.2, decayvar=2421.9,
                      N = int(numwaves/2))
    else:
        noise.resetCuts()
        numwaves = noise.headers().shape[0]
        print(numwaves)
        simwfs = batch_wfs(t0="random", amplitude="gaussian", rise_time="random", decay="gaussian",
                      t0min=3420, t0max=3500,
                      ampmin = ampmean, ampmax = ampmax, ampmean=ampmean, ampvar=ampvar,
                      rise_timemin=8, rise_timemax=22,
                      decaymin = 1200, decaymax = 1300, decaymean=1259.2, decayvar=2421.9,
                      N = 400)
        
    protonwfs = []
    noisewfs = []
    j  = 0
    for i in range(len(simwfs)):
        n = noise.wave(j)[:7000]
        protonwfs.append(simwfs[i] + n)
        noisewfs.append(noise.wave(j)[7000:])
        j+=1
    
    noisewfs = np.array(noisewfs)
    protonwfs = np.array(protonwfs)
    
    # Stack data
    X_np = np.vstack((protonwfs, noisewfs))
    Y_np = np.hstack((
        np.ones(len(protonwfs)),    # Label 1 for protons
        np.zeros(len(noisewfs))    # Label 0 for noise
        ))
    
    # Convert to Dask arrays
    X = da.from_array(X_np, chunks="auto")
    Y = da.from_array(Y_np, chunks="auto")
    
    # Scale using Dask's StandardScaler
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    return X , Y , scaler 
def test_model(noise, pixel = None, ampmin = 0, ampmax = 80, ampmean = 35, ampvar = 225):
    '''
    This function is mostly intended as a sanity check. It will generate simulated waveforms with experimental noise.
    Then it will train and test a dask-ml logistic regression model. It will print a confusion matrix and a classification report.
    Note: if you intend to apply a model on real data, do not use this function. Use _____ instead, as test_model() 
    automatically partitions 20% of the data away for testing, which could limit already sparse data. 
    
    Paramaters:
        noise (Nab waveformFileClass object): noise from a specified datarun. 
        pixel (int): pixel number from which noise will be pulled. If none is specified, run will take sample from all pixels
        ampmin (int): minimum amplitude of waveforms (initialized to 0, the min physical proton energy)
        ampmax (int): maximum amplitude of waveforms (initialized to 80, the max physical proton energy)
        ampmean (int): mean amplitude (initialized to 35, amplitude mean physical proton energy)
        ampvar (int): mean variance (initialized to 225, amplitude max physical proton energy)
    Return:
        model (LogisticRegression): The trained logistic regression model

    '''
    X, Y = generate_training_data(n1)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    y_test_np = y_test.compute()
    y_pred_np = y_pred.compute()
    cm = confusion_matrix(y_test_np, y_pred_np)
    print("Confusion Matrix:")
    print(cm)
    print(classification_report(y_test_np, y_pred_np, target_names=["Noise", "Proton"]))    
    
    return model

def get_waveform_by_index(idx):
    '''
    retrives specific waveform by the index
    Parameters
    ----------
    idx : int
        index of waveform
    Returns
    -------
    waveform: 
        waveform
    '''
    return coinc.wave(idx)
    
def plot_waveform(waveform, wf = "proton"):
    '''
    Plots a given waveform by index.
    Parameters
    ----------
    index : int
        Index of waveform.
    Returns
    ------
    None.
    '''
    sample = waveform
    t = np.arange(len(sample))
    plt.figure(figsize=(12, 6))
    plt.plot(t, sample, label="Raw Waveform")
    if wf == "proton":
        plt.title(f"Proton")
    else:
        plt.title(f"Noise")
    plt.xlabel("Time")
    plt.legend()
    plt.tight_layout()
    plt.show()

#%% 2

if __name__ == '__main__':
    
    sys.path.append('/Users/akannan/Downloads/Lab/Nab-Analysis')
    import Paths

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
    from dask_ml.preprocessing import StandardScaler
    from dask_ml.linear_model import LogisticRegression
    from dask_ml.model_selection import train_test_split
    from dask_ml.metrics import accuracy_score
    from sklearn.metrics import confusion_matrix, classification_report
    sys.path.append('/Users/akannan/Downloads/Lab/NabWork/NabPy/src/nabPy')
    import basicFunctions as bf
#%% 3
    run = Nab.DataRun(paths[2], 7616)
    parameters = run.parameterFile()
#    nr = Nab.DataRun(paths[2], 7834)
    n1 = run.noiseWaves()
    coinc = run.coincWaves()    
    filter_settings = [1250, 50, 1250]
#%% 4
    X, Y, scaler = generate_training_data(n1)
    model = LogisticRegression()
    model.fit(X, Y)
#%% 5
    coinc.resetCuts()
    coinc.defineCut("hit type", "=", 0)
    indices = coinc.headers().index.tolist()
    print(len(indices))
    coinc.defineCut("custom", indices[0:49000]) # Hopefully this will take a smaller sample of 5000 protons as opposed to the 150,000
    Xtest = coinc.waves()
    print(Xtest)  
    initial_energies, t = bf.applyTrapFilter(Xtest, *filter_settings)
#%% 5.5
    plt.hist(initial_energies, bins=np.arange(0,100), color='skyblue', edgecolor='black')
    plt.xlabel('Energy')
    plt.ylabel('Frequency')
    plt.ylim(0,600)
    plt.title('protons hits vs energy (DAQ)')
    plt.show()

#    plt.figure(figsize=(10, 6))
#    plt.title("protons hits vs energy (DAQ)")
#    protonE.hist('energy', bins = np.arange(0,100), color='skyblue', edgecolor='black')
#%% 6  
    
  
#%% 7
    X_data_scaled = scaler.transform(Xtest)
#%% 8
    y_pred = model.predict(X_data_scaled)
    y_pred_result = y_pred.compute()
    
#%% 9      
    y_pred_da = da.from_array(y_pred_result, chunks=(1000,))

    # Boolean masks
    proton_mask = y_pred_da == 1
    noise_mask  = y_pred_da == 0
    
    # Split arrays
    proton_hits = Xtest[proton_mask]
    noise_hits   = Xtest[noise_mask]
    
    
    
    
#%% 10    
    proton_hits = proton_hits.compute_chunk_sizes()
    filtered_energies, timing = bf.applyTrapFilter(proton_hits, *filter_settings)
#%% 11
    plt.hist(filtered_energies, bins=np.arange(0,100), color='skyblue', edgecolor='black')
    plt.xlabel('Energy')
    plt.ylabel('Frequency')
    plt.ylim(0,3000)
    plt.title('protons hits (by LR model) vs energy (DAQ)')
    plt.show()



#%% 12
    noise_hits = noise_hits.compute_chunk_sizes()
    noise_energies, timing = bf.applyTrapFilter(noise_hits, *filter_settings)    
    plt.hist(noise_energies, bins=np.arange(0,100), color='orange', edgecolor='black')
    plt.xlabel('Energy')
    plt.ylabel('Frequency')
    plt.title('noise hits (by LR model) vs energy (DAQ)')
    plt.show()

#%% 
    i = np.random.randint(0,noise_hits.shape[0])
    plot_waveform(noise_hits[i], wf = "n")
    print(i)
    


   
