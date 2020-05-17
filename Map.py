# -------------------------------------------------------
# Assignment 1
# Written by Giselle Martel 26352936
# For COMP 472 Section JX – Summer 2020
# --------------------------------------------------------

import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.widgets import Slider, Button, RadioButtons
import numpy as np


# create the plot figure
fig, ax = plt.subplots(figsize=(7, 11))
t = np.arange(-75.589, -7.550, 0.001)
s = t
l, = plt.plot(t, s, lw=2)

# read data from shp file and fetch attribute containing crime point data
data = gpd.read_file("./Shape/crime_dt.shp")
values = data.values

# get all the crime points on the map
crime_points = [v[3] for v in values]

x_values = [point.x for point in crime_points]
y_values = [point.y for point in crime_points]

# generate a color map to represent the crime areas
cmap = matplotlib.colors.LinearSegmentedColormap.from_list('Crime Points MTL', ['purple', 'yellow'])

# get the bounds of the crime map from shapefile data
total_bounds = data.total_bounds


def init():
    plt.ion()
    plt.rcParams.update({'figure.dpi': 200})
    matplotlib.use("TkAgg")

    axmap = plt.axes([0.1, 0.25, 0.7, 0.7])

    updateMap(0.002, 50, axmap)

    # set layout of figures
    axthreshold = plt.axes([0.2, 0.05, 0.5, 0.03])
    axcell = plt.axes([0.2, 0.1, 0.5, 0.03])

    # sliders to change threshold and cell size interactively
    sthreshold = Slider(axthreshold, 'Threshold', 0, 100, valinit=50, valstep=5)
    scell = Slider(axcell, 'Cell Size', 0.001, 0.003, valinit=0.002, valstep=0.001)

    def update(val):
        threshold = sthreshold.val
        cell_size = scell.val
        l.set_ydata(1)
        updateMap(cell_size, threshold, axmap)
        # fig.canvas.draw_idle()
        # plt.draw()

    sthreshold.on_changed(update)
    scell.on_changed(update)

    plt.ioff()
    plt.title('Crime Data MTL')
    plt.show()


def updateMap(step_size, threshold, axmap):
    # get the horizontal-vertical size of grid (X*Y)
    grid_dimensions = calcGridDimensions(total_bounds, step_size)
    # use matplotlib hist2d function to plot grid with given step size
    crime_map = axmap.hist2d(x_values, y_values, bins=grid_dimensions, cmap=cmap)

    # obtain the # of crimes per grid cell, and sort from least to most crimes
    crimes_per_cell_sorted = np.array(crime_map[0]).flatten()
    crimes_per_cell_sorted.sort()

    # calculate the threshold index based on given threshold percentage
    # and calculate the threshold_val based on the sorted list of # crimes
    # for example, if array = [0, 3, 5, 9] and threshold percentage is 50%
    # then the index would be 4 * 0.5 - 1 = 1 and the value would be 3
    threshold_index = int(len(crimes_per_cell_sorted) * (1 - threshold * 0.01)) - 1
    threshold_val = crimes_per_cell_sorted[threshold_index]

    # determine the norm of the grid based on the threshhold value
    grid_norm = BoundaryNorm([threshold_val], ncolors=cmap.N)

    # use matplotlib hist2d function to plot grid with given step size and threshold value
    crime_map = axmap.hist2d(x_values, y_values, bins=grid_dimensions, cmap=cmap, norm=grid_norm)

    crimes_per_cell = np.array(crime_map[0])
    grid_x_ticks = np.array(crime_map[1])
    grid_y_ticks = np.array(crime_map[2])

    generateCrimeMapDataForEachCell(crimes_per_cell, grid_x_ticks, grid_y_ticks, step_size)



def calcGridDimensions(total_bounds, step_size):
    # get the bounds of the whole crime area
    min_x = total_bounds[0]
    min_y = total_bounds[1]
    max_x = total_bounds[2]
    max_y = total_bounds[3]

    # get num cells on x and y axis of grid based on bounds of crime area and step size
    x_grid_steps = np.ceil((max_x - min_x) / step_size)
    y_grid_steps = np.ceil((max_y - min_y) / step_size)

    return [int(x_grid_steps), int(y_grid_steps)]


def generateCrimeMapDataForEachCell(crimes_per_cell, grid_x_ticks, grid_y_ticks, step_size):
    for x in range(0, len(grid_x_ticks) - 1):
        for y in range(0, len(grid_y_ticks) - 1):
            curr_cell_val = crimes_per_cell[x][y]
            display_val = str(curr_cell_val)
            text_padding = step_size / 2
            margin_horizontal = grid_x_ticks[x] + text_padding
            margin_vertical = grid_y_ticks[y] + text_padding
            plt.text(margin_horizontal, margin_vertical, display_val, fontdict=dict(fontsize=3, ha='center', va='center'))

    vals = crimes_per_cell.flatten()
    # plt.legend(str(np.std(vals)))
    # plt.legend(str(np.mean(vals)))


init()