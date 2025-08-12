import numpy as np
import matplotlib.pyplot as plt

def create_waveform(
    amplitude,      # Peak value of spike
    rise_time,      # Time over which signal rises linearly to amplitude
    tau,            # Decay time constant (exponential)
    t0=3500,        # Pulse start time
):
    """
    Generates a waveform:
    - Baseline 0
    - Linear rise starting at t0, reaching 'amplitude' after 'rise_time'
    - Exponential decay from 'amplitude' with time constant 'decay_tau'
    """
    t = np.linspace(0, 7000, 7000)
    waveform = np.zeros_like(t)
    
    # Indices for different regions (Lists of bools. idx_rise indicates a rise wherever the index is True. idx_decay indicates a decay wherever the index is True.
    idx_rise = (t >= t0) & (t < t0 + rise_time)
    idx_decay = (t >= t0 + rise_time)
    
    # Linear rise: y = amplitude * (t - t0) / rise_time
    waveform[idx_rise] = amplitude * (t[idx_rise] - t0) / rise_time
    
    # Exponential decay: y = amplitude * exp(- (t - (t0 + rise_time)) / decay_tau )
    waveform[idx_decay] = amplitude * np.exp( - (t[idx_decay] - (t0 + rise_time)) / tau )
    
    return waveform

# Example usage:
if __name__ == "__main__":
    amp = 1.0
    t0 = 10
    rise_time = 5   # 5 units rise time
    tau = 15   # 15 units decay constant
    
    wf = create_waveform(amp, rise_time, tau)
    plt.plot(wf)
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.title("Simulated Particle Waveform")
    plt.show()
