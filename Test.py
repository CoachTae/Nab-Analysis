import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import time
# Path for PyNab package
sys.path.append("C:/Users/ricardo/Downloads/pyNab-master/pyNab-master/src")
# Path for deltarice (package created by David Matthew for Nab)
sys.path.append("C:/Users/ricardo/Downloads/deltarice-master/deltarice-master/build/lib.linux-x86_64-3.10")
import nabPy as Nab
import h5py

data = Nab.File("/mnt/c/Users/ricardo/Downloads/Run5320_0.h5")
print(data)


# Print something to show that the code has resolved completely.
print("Test successful!")
