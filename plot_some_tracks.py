import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

ds = xr.open_dataset("build/scatter.nc")

plt.ion()
ax = plt.figure().add_subplot(projection="3d")
for i in range(100):
    ax.plot(ds.rx[:, i], ds.ry[:, i], ds.rz[:, i], 'o-')
