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

        self.Etimestamp = {}
        self.Ptimestamp = {}
        self.not_in_range = {}
        self._sorted_electron_ts = None

    def _get_paths(self):
        return Paths.Skylar_Paths if self.user == 'skylar' else Paths.Arush_Paths

    def _setup_paths(self):
        sys.path.append(self.paths[0])
        sys.path.append(self.paths[1])

    def extract_electron_timestamps(self):
        self.coinc.resetCuts()
        self.coinc.defineCut("hit type", "=", 2)
        elecE = self.coinc.determineEnergyTiming(method='trap', params=self.filter_settings)
        headers = self.coinc.headers()
        indices = headers.index.tolist()
        for i in range(len(indices)):
            h = np.array(self.coinc.head(i, pandas=True))
            self.Etimestamp[indices[i]] = h[2]
        return elecE

    def extract_proton_timestamps(self):
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

    def filter_protons_by_time_range(self, window=10000):
        sorted_Ets = sorted(self.Etimestamp.items(), key=lambda x: x[1])
        sorted_ts = [ts for _, ts in sorted_Ets]

        for Pidx, Pts in list(self.Ptimestamp.items()):
            pos = bisect.bisect_left(sorted_ts, Pts - window)
            in_range = False
            while pos < len(sorted_ts) and sorted_ts[pos] <= Pts + window:
                if abs(sorted_ts[pos] - Pts) <= window:
                    in_range = True
                    break
                pos += 1
            if not in_range:
                self.not_in_range[Pidx] = Pts
                del self.Ptimestamp[Pidx]

        return {
            "initial": len(sorted_Ets),
            "filtered": len(self.Ptimestamp),
            "removed": len(self.not_in_range)
        }

    def get_waveform_by_index(self, idx):
        return self.coinc.wave(idx)

    def determine_real_and_false_protons(self):
        self.coinc.resetCuts()
        self.coinc.defineCut("custom", list(self.not_in_range.keys()))
        false_protons = self.coinc.determineEnergyTiming(method='trap', params=self.filter_settings)

        self.coinc.resetCuts()
        self.coinc.defineCut("custom", list(self.Ptimestamp.keys()))
        real_protons = self.coinc.determineEnergyTiming(method='trap', params=self.filter_settings)

        return false_protons, real_protons

    def plot_waveform(self, index):
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
