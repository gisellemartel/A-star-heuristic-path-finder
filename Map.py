# -------------------------------------------------------
# Assignment 1
# Written by Giselle Martel 26352936
# For COMP 472 Section JX – Summer 2020
# --------------------------------------------------------

import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


# entry point of app
from matplotlib.colors import ListedColormap


def init():
    plt.ion()
    plt.rcParams.update({'figure.dpi': 200})
    matplotlib.use("TkAgg")
    plotFileData("./Shape/crime_dt.shp", 0.002)
    plt.ioff()
    plt.show()


def plotFileData(file, step_size):
    data = gpd.read_file(file)
    values = data.values

    crime_points = [v[3] for v in values]

    min_x = data.total_bounds[0]
    max_x = data.total_bounds[1]

    min_y = data.total_bounds[2]
    max_y = data.total_bounds[3]

    print(min_x)
    print(max_x)
    print(min_y)
    print(max_y)

    x_grid_steps = np.arange(min_x, max_x, step_size)
    y_grid_steps = np.arange(min_y, max_y, step_size)

    flat_grid = []

    for i in range(0, len(x_grid_steps) - 1):
        p1 = (x_grid_steps[i], y_grid_steps[i + 1])
        p2 = (x_grid_steps[i], y_grid_steps[i])
        p3 = (x_grid_steps[i + 1], y_grid_steps[i])
        p4 = (x_grid_steps[i + 1], y_grid_steps[i + 1])
        grid_cell = (p1, p2, p3, p4)
        flat_grid.append(grid_cell)


    # print(flat_grid)




init()