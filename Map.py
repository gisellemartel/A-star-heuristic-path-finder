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


class CrimeMap:

    def __init__(self):
        # create the plot figure
        self.fig, self.ax = plt.subplots(figsize=(7, 11))

        # read data from shp file and fetch attribute containing crime point data
        self.data = gpd.read_file("./Shape/crime_dt.shp")
        values = self.data.values

        # get all the crime points on the map
        self.crime_points = [v[3] for v in values]

        self.x_values = [point.x for point in self.crime_points]
        self.y_values = [point.y for point in self.crime_points]

        # generate a color map to represent the crime areas
        self.cmap = matplotlib.colors.LinearSegmentedColormap.from_list('Crime Points MTL', ['purple', 'yellow'])

        # get the bounds of the crime map from shapefile data
        self.total_bounds = self.data.total_bounds

        self.grid_text = []
        self.crimes_per_cell = []
        self.grid_x_ticks = []
        self.grid_y_ticks = []

    def plotCrimeMap(self):
        plt.ion()
        plt.rcParams.update({'figure.dpi': 200})
        matplotlib.use("TkAgg")

        # set layout of figures
        axthreshold = plt.axes([0.2, 0.05, 0.5, 0.03])
        axcell = plt.axes([0.2, 0.1, 0.5, 0.03])
        axmap = plt.axes([0.1, 0.25, 0.7, 0.7])

        self.updateMap(0.002, 50, axmap)

        # sliders to change threshold and cell size interactively
        sthreshold = Slider(axthreshold, 'Threshold %', 0, 100, valinit=50, valstep=5, valfmt='%0.0f')
        scell = Slider(axcell, 'Cell Size', 0.001, 0.005, valinit=0.002, valstep=0.001, valfmt='%0.3f')

        def update(val):
            threshold = sthreshold.val
            cell_size = scell.val
            self.updateMap(cell_size, threshold, axmap)

        sthreshold.on_changed(update)
        scell.on_changed(update)

        plt.ioff()
        plt.title('Crime Data MTL')

        # Place a legend above this subplot, expanding itself to
        # fully use the given bounding box.
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

        plt.show()

    def updateMap(self, step_size, threshold, axmap):
        # get the horizontal-vertical size of grid (X*Y)
        grid_dimensions = self.calcGridDimensions(step_size)
        # use matplotlib hist2d function to plot grid with given step size
        crime_map = axmap.hist2d(self.x_values, self.y_values, bins=grid_dimensions, cmap=self.cmap)

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
        grid_norm = BoundaryNorm([threshold_val], ncolors=self.cmap.N)

        # use matplotlib hist2d function to plot grid with given step size and threshold value
        crime_map = axmap.hist2d(self.x_values, self.y_values, bins=grid_dimensions, cmap=self.cmap, norm=grid_norm)

        # update the tick values and # of crimes per cell
        self.crimes_per_cell = np.array(crime_map[0])
        self.grid_x_ticks = np.array(crime_map[1])
        self.grid_y_ticks = np.array(crime_map[2])

        self.generateCrimeMapDataForEachCell(step_size, axmap)
        self.displayStDevAndMean(axmap)

    def calcGridDimensions(self, step_size):
        # get the bounds of the whole crime area
        min_x = self.total_bounds[0]
        min_y = self.total_bounds[1]
        max_x = self.total_bounds[2]
        max_y = self.total_bounds[3]

        # get num cells on x and y axis of grid based on bounds of crime area and step size
        x_grid_steps = np.ceil((max_x - min_x) / step_size)
        y_grid_steps = np.ceil((max_y - min_y) / step_size)

        return [int(x_grid_steps), int(y_grid_steps)]

    def generateCrimeMapDataForEachCell(self, step_size, axmap):
        # make sure to remove old cell labels before updating
        if len(self.grid_text) > 0:
            for t in range(0, len(self.grid_text)):
                self.grid_text[t].remove()
                self.grid_text[t] = None
            self.grid_text = []

        # display crime per grid cell in UI
        for x in range(0, len(self.grid_x_ticks) - 1):
            for y in range(0, len(self.grid_y_ticks) - 1):
                curr_cell_val = self.crimes_per_cell[x][y]
                display_val = str(curr_cell_val)
                text_padding = step_size / 2
                margin_horizontal = self.grid_x_ticks[x] + text_padding
                margin_vertical = self.grid_y_ticks[y] + text_padding
                text = axmap.text(margin_horizontal, margin_vertical, display_val, fontdict=dict(fontsize=3, ha='center', va='center'))
                self.grid_text.append(text)

    def displayStDevAndMean(self, axmap):
        stdev = self.crimes_per_cell.std()
        mean = self.crimes_per_cell.mean()
        median = np.median(self.crimes_per_cell)
        textstr = '\n'.join((
            r'$\mu=%.2f$' % (mean,),
            r'$\mathrm{median}=%.2f$' % (median,),
            r'$\sigma=%.2f$' % (stdev,)))

        # these are matplotlib.patch.Patch properties
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

        # place a text box in upper left in axes coords
        axmap.text(-5, -5, textstr, fontsize=14, verticalalignment='bottom', bbox=props)


crimes_map = CrimeMap()
crimes_map.plotCrimeMap()
