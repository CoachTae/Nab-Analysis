# nab_noise_reduction.py

import sys
import numpy as np
import matplotlib.pyplot as plt
import bisect
import Paths
import pandas as pd


class NabFalseProtons:
    def __init__(self, user='arush', run_number=7616, filter_settings=None, nabpy_path=None):
        self.user = user.lower()
        self.run_number = run_number
        self.filter_settings = filter_settings or [1250, 50, 1250]
        self.paths = self._get_paths()
        self._setup_paths()

        # Dynamically import nabPy if path provided
        if nabpy_path:
            sys.path.append(nabpy_path)
        import nabPy as Nab
        self.Nab = Nab

        self.run = self.Nab.DataRun(self.paths[2], self.run_number)
        self.coinc = self.run.coincWaves()

        self.parameterNames = ['amp1', 'mean1', 'sigma1', 'offset1', 'amp2', 'mean2', 'sigma2', 'offset2']
        self.Etimestamp = {}
        self.Ptimestamp = {}
        self.not_in_range = {}
        self._sorted_electron_ts = None
    '''
    -----------------------------------------------------------------------------
    File handling functions
    -----------------------------------------------------------------------------
    '''
    def _get_paths(self):
        return Paths.Skylar_Paths if self.user == 'skylar' else Paths.Arush_Paths

    def _setup_paths(self):
        sys.path.append(self.paths[0])
        sys.path.append(self.paths[1])
    '''
    -----------------------------------------------------------------------------
    Timestamp based Filtering Functions
    -----------------------------------------------------------------------------
    '''
    def extract_electrons(self):
        '''
        This function extracts the timestamps of all electron hits as recognized by the DAQ,
        and adds them to a dictionary: Etimestamp.
        
        Returns
        -------
        resultFile: class or None
			Returns a resultFileClass object of the electron hits.
			If the code fails for some reason, such as no waveforms being present, this returns None
        '''
        self.coinc.resetCuts()
        self.coinc.defineCut("hit type", "=", 2)
        elecE = self.coinc.determineEnergyTiming(method='trap', params=self.filter_settings)
        headers = self.coinc.headers()
        indices = headers.index.tolist()
        for i in range(len(indices)):
            h = np.array(self.coinc.head(i, pandas=True))
            self.Etimestamp[indices[i]] = h[2]
        return elecE

    def extract_protons(self):
        '''
        This function extracts the timestamps of all proton hits as recognized by the DAQ,
        and adds them to a dictionary: Ptimestamp. 
        It then filters out protons with nonphysical energies.
        
        Returns
        -------
        resultFile: class or None
			Returns a resultFileClass object of the proton hits (with energies > 0).
			If the code fails for some reason, such as no waveforms being present, this returns None
        '''
        self.coinc.resetCuts()
        self.coinc.defineCut("hit type", "=", 0)
        protonE = self.coinc.determineEnergyTiming(method='trap', params=self.filter_settings)
        protonE.defineCut("energy", '>', 0)
        cut = protonE.returnCut()
        self.coinc.defineCut("custom", cut)
        protonE = self.coinc.determineEnergyTiming(method='trap', params=self.filter_settings)
        indices = self.coinc.headers().index.tolist()
        for i in range(len(indices)):
            h = np.array(self.coinc.head(i, pandas=True))
            self.Ptimestamp[indices[i]] = h[2]
        return protonE
    
    
    
    def filter_protons_by_time_range(self, window=10000, lower_bound=0):
        '''
        Any proton must be within a specified timestamp window or physically 40 us 
        of any electron. This function removes all protons that do not fall within 
        the time range of any electron, from the Ptimetamps dictionary.
        
        Parameters
        ----------
        window: int, defaults to 10000
            The timestamp window of 10,000 timestamps or physically 40 us. 
            Can be adjusted to any potential time range.
        
        lower_bound: int, defaults to 0
            The lower bound for the time range. Any proton's timestamp must 
            be greater than or equal to this value to be considered for filtering.

        Returns
        -------
        sizes: dict
            Returns the number of protons:
            - 'initial': The number of electrons in the system.
            - 'filtered': The number of protons that pass the filtering criteria.
            - 'removed': The number of protons that were removed based on the filtering.
        '''
    
        # Sort the electron timestamps
        sorted_Ets = sorted(self.Etimestamp.items(), key=lambda x: x[1])
        sorted_ts = [ts for _, ts in sorted_Ets]
        
        # Iterate through protons
        for Pidx, Pts in list(self.Ptimestamp.items()):
            # Apply the lower bound check
            if Pts < lower_bound:
                self.not_in_range[Pidx] = Pts
                del self.Ptimestamp[Pidx]
                continue
        
            # Bisect to find the closest electrons' timestamps within the window
            pos = bisect.bisect_left(sorted_ts, Pts - window)
            in_range = False
            
            # Check if the proton is within the upper and lower bound time range
            while pos < len(sorted_ts) and sorted_ts[pos] <= Pts + window:
                if abs(sorted_ts[pos] - Pts) <= window:
                    in_range = True
                    break
                pos += 1
        
            # If not in range, remove proton
            if not in_range:
                self.not_in_range[Pidx] = Pts
                del self.Ptimestamp[Pidx]

        # Return the counts of protons before and after filtering
        return {
            "initial": len(sorted_Ets),  # Initial number of electrons
            "filtered": len(self.Ptimestamp),  # Remaining protons after filtering
            "removed": len(self.not_in_range)  # Protons removed based on the filter
        }
    
    def real_and_false_protons(self):
        '''
        This function essentially organizes all the protons calssified by the DAQ 
        (with physical energies) into those in and out of the time range. In order to 
        function as intended, this must be called after extract_electron_timestamps(),
        extract_proton_timestamps() and filter_protons_by_time_range().

        Returns
        -------
        false_protons : Returns a resultFileClass object of the proton hits by 
            index from the not_in_range dictionary.(protons not in the time range)
        real_protons : Returns a resultFileClass object of the proton hits by 
            index from the Ptimestamp dictionary.(protons not in the time range)

        '''
        self.coinc.resetCuts()
        self.coinc.defineCut("custom", list(self.not_in_range.keys()))
        false_protons = self.coinc.determineEnergyTiming(method='trap', params=self.filter_settings)

        self.coinc.resetCuts()
        self.coinc.defineCut("custom", list(self.Ptimestamp.keys()))
        real_protons = self.coinc.determineEnergyTiming(method='trap', params=self.filter_settings)

        return false_protons, real_protons

    '''
    -----------------------------------------------------------------------------
    Useful Data Access and Plotting Functions
    -----------------------------------------------------------------------------
    ''' 
    
    def get_waveform_by_index(self, idx):
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
        return self.coinc.wave(idx)
    
    def plot_waveform(self, index):
        '''
        Plots a given waveform by index.
        Parameters
        ----------
        index : int
            Index of waveform.
        Returns
        -------
        None.
        '''
        self.coinc.resetCuts()
        sample = self.get_waveform_by_index(index)
        t = np.arange(len(sample))
        plt.figure(figsize=(12, 6))
        plt.plot(t, sample, label="Raw Waveform")
        plt.title(f"Proton {index}")
        plt.xlabel("Time")
        plt.legend()
        plt.tight_layout()
        plt.show()
        
    def double_gaussian(self, x, amp1, mean1, sigma1, offset1, amp2, mean2, sigma2, offset2):
        gaussian1 = (amp1 * np.exp(-1.0 * ((x - mean1) / sigma1) ** 2.0) + offset1)
        gaussian2 = (amp2 * np.exp(-1.0 * ((x - mean2) / sigma2) ** 2.0) + offset2)
        return gaussian1 + gaussian2

    def double_gaussian_derivative(self, x, amp1, mean1, sigma1, offset1, amp2, mean2, sigma2, offset2):
        term1 = -(x - mean1) / (sigma1**2) * np.exp(-((x - mean1) / sigma1) ** 2)
        term2 = -(x - mean2) / (sigma2**2) * np.exp(-((x - mean2) / sigma2) ** 2)
        return amp1 * term1 + amp2 * term2

    def double_gaussian_second_derivative(self, x, amp1, mean1, sigma1, offset1, amp2, mean2, sigma2, offset2):
        term1 = ((x - mean1) ** 2 / (sigma1 ** 4) - 1 / (sigma1 ** 2)) * np.exp(-((x - mean1) / sigma1) ** 2)
        term2 = ((x - mean2) ** 2 / (sigma2 ** 4) - 1 / (sigma2 ** 2)) * np.exp(-((x - mean2) / sigma2) ** 2)
        return amp1 * term1 + amp2 * term2



    def double_gaussian_turning_point(self, params, search_range=4):
        '''
        Identifies the local minima, the turning point, between the 2 modes of 
        a double Gaussian by analyzing where the second derivative changes sign.
        
        Parameters
        ----------
        params : list or tuple
            Expected format: [amp1, mean1, sigma1, offset1, amp2, mean2, sigma2, offset2]
        search_range : int, optional, default=4
            Interval around the initial guess (between the two means) to search for the inflection point.
            
        Returns
        -------
        turning_point : float or None
            The location of the turning point between the two modes of the double Gaussian. If no
            turning point is found, returns string saying no turning points.
        '''

        try:
            amp1, mean1, sigma1, offset1, amp2, mean2, sigma2, offset2 = params
        except ValueError:
            raise ValueError("Expected params to be a list or tuple of 8 values.")

        init = (mean1 + mean2) / 2  
    
        distance = abs(mean1 - mean2)
        if distance < 2 * search_range:
            search_range = max(2, distance // 2)
    

        lower_bound = init - search_range
        upper_bound = init + search_range
        step_size = 0.1 
        x_values = np.arange(lower_bound, upper_bound, step_size)

        prev_derivative = None
        for i, x in enumerate(x_values[1:], 1): 
            current_derivative = self.double_gaussian_second_derivative(
                x, amp1, mean1, sigma1, offset1, amp2, mean2, sigma2, offset2
            )
            if prev_derivative is not None and prev_derivative * current_derivative < 0:
                x1, x2 = x_values[i-1], x
                y1, y2 = prev_derivative, current_derivative
                if abs(y2 - y1) > 1e-10:
                    turning_point = x1 - y1 * (x2 - x1) / (y2 - y1)
                    return turning_point

            prev_derivative = current_derivative
    
        return "No turning points in given range"
    
    
    
    def noise_gaussian_SD_bound(self, params, n):
        '''
        This function returns the right hand value of the noise gaussian (in double gaussian fit)
        n standard deviations away from the peak. 
        Since gaussian, 1 std dev away from the peak (in both directions) will encompass 68% of the noise,
        n=2 ==> 95%, n=3 ==> 99.7% of the noise measured.
        Parameters
        ----------
        params : list or tuple
            Expected format: [amp1, mean1, sigma1, offset1, amp2, mean2, sigma2, offset2]
        n : int
            expects the number of std deviations away from the mean you want to measure
            
        Raises
        ------
        ValueError
            if params in unexpected format

        Returns
        -------
        mark : double
            value 'n' standard deviations from the mean (peak) on the right hand side of the curve.

        '''
        try:
            amp1, mean1, sigma1, offset1, amp2, mean2, sigma2, offset2 = params
        except ValueError:
            raise ValueError("Expected params to be a list or tuple of 8 values.")
        
        mark = 1
        
        if mean1 <= mean2:
            mark = mean1 + (n * abs(sigma1))
        else:
            mark = mean2 + (n * abs(sigma2))
        
        
        return mark
    '''
    -----------------------------------------------------------------------------
    name soon to be determined fitlering method
    -----------------------------------------------------------------------------
    '''
    
    














        
      
