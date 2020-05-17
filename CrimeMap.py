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

        # get all the crime points on the map
        self.crime_points = [v[3] for v in self.data.values]
        self.x_values = [point.x for point in self.crime_points]
        self.y_values = [point.y for point in self.crime_points]

        # generate a color map to represent the crime areas
        self.cmap = matplotlib.colors.LinearSegmentedColormap.from_list('Crime Points MTL', ['purple', 'yellow'])

        # get the bounds of the crime map from shapefile data
        self.total_bounds = self.data.total_bounds

        self.grid_display_text = []
        self.crimes_per_cell = []
        self.grid_x_ticks = []
        self.grid_y_ticks = []

        self.step_size = 0.002
        self.threshold = 50

        self.axmap = None

        self.show_data = True

    def onPressDataToggle(self, event):
        self.show_data = not self.show_data
        self.setGridDisplayData()

    def plotCrimeMap(self):
        plt.ion()
        plt.rcParams.update({'figure.dpi': 200, 'font.size': 5})
        matplotlib.use("TkAgg")

        # set layout of figures
        axthreshold = plt.axes([0.2, 0.05, 0.5, 0.03])
        axcell = plt.axes([0.2, 0.1, 0.5, 0.03])
        axtoggle = plt.axes([0.77, 0.05, 0.05, 0.05])
        self.axmap = plt.axes([0.1, 0.25, 0.7, 0.7])

        self.updateMap()

        # sliders to change threshold and cell size interactively and button to toggle data view
        sthreshold = Slider(axthreshold, 'Threshold %', 0, 100, valinit=50, valstep=5, valfmt='%0.0f')
        scell = Slider(axcell, 'Cell Size', 0.001, 0.005, valinit=0.002, valstep=0.001, valfmt='%0.3f')
        btoggle = Button(axtoggle, 'hide/show\ndata')
        sthreshold.label.set_fontsize(8)
        scell.label.set_fontsize(8)
        btoggle.label.set_fontsize(4)

        def update(val):
            self.threshold = sthreshold.val
            self.step_size = scell.val
            self.updateMap()

        sthreshold.on_changed(update)
        scell.on_changed(update)
        btoggle.on_clicked(self.onPressDataToggle)

        plt.ioff()

        plt.show()

    def updateMap(self):
        # get the horizontal-vertical size of grid (X*Y)
        grid_dimensions = self.calcGridDimensions()
        # use matplotlib hist2d function to plot grid with given step size
        crime_map = self.axmap.hist2d(self.x_values, self.y_values, bins=grid_dimensions, cmap=self.cmap)

        # obtain the # of crimes per grid cell, and sort from least to most crimes
        crimes_per_cell_sorted = np.array(crime_map[0]).flatten()
        crimes_per_cell_sorted[::-1].sort()

        # calculate the threshold index based on given threshold percentage
        # and calculate the threshold_val based on the sorted list of # crimes
        # for example, if array = [0, 3, 5, 9] and threshold percentage is 50%
        # then the index would be 4 * 0.5 - 1 = 1 and the value would be 3
        threshold_val = 0
        if self.threshold < 100:
            threshold_index = int(len(crimes_per_cell_sorted) * (1 - self.threshold * 0.01)) - 1
            threshold_val = crimes_per_cell_sorted[threshold_index]
        else:
            threshold_index = 0
            threshold_val = crimes_per_cell_sorted[threshold_index] + 1

        # determine the norm of the grid based on the threshhold value
        grid_norm = BoundaryNorm([threshold_val], ncolors=self.cmap.N)

        # use matplotlib hist2d function to plot grid with given step size and threshold value
        crime_map = self.axmap.hist2d(self.x_values, self.y_values, bins=grid_dimensions, cmap=self.cmap, norm=grid_norm)

        # update the tick values and # of crimes per cell
        self.crimes_per_cell = np.array(crime_map[0])
        self.grid_x_ticks = np.array(crime_map[1])
        self.grid_y_ticks = np.array(crime_map[2])

        if self.show_data:
            self.setGridDisplayData()
        self.displayStDevAndMean(threshold_val)

    def calcGridDimensions(self):
        # get the bounds of the whole crime area
        min_x = self.total_bounds[0]
        min_y = self.total_bounds[1]
        max_x = self.total_bounds[2]
        max_y = self.total_bounds[3]

        # get num cells on x and y axis of grid based on bounds of crime area and step size
        x_grid_steps = np.ceil((max_x - min_x) / self.step_size)
        y_grid_steps = np.ceil((max_y - min_y) / self.step_size)

        return [int(x_grid_steps), int(y_grid_steps)]

    def setGridDisplayData(self):
        if len(self.grid_display_text) > 0:
            for t in range(0, len(self.grid_display_text)):
                self.grid_display_text[t].remove()
                self.grid_display_text[t] = None
            self.grid_display_text = []
        if self.show_data:
            # display crime per grid cell in UI
            for x in range(0, len(self.grid_x_ticks) - 1):
                for y in range(0, len(self.grid_y_ticks) - 1):
                    curr_cell_val = self.crimes_per_cell[x][y]
                    display_val = str(int(curr_cell_val))
                    text_padding = self.step_size / 2
                    margin_horizontal = self.grid_x_ticks[x] + text_padding
                    margin_vertical = self.grid_y_ticks[y] + text_padding
                    text = self.axmap.text(margin_horizontal, margin_vertical, display_val,
                                      fontdict=dict(fontsize=3, ha='center', va='center'))
                    self.grid_display_text.append(text)

    def displayStDevAndMean(self, threshold_val):
        stdev = self.crimes_per_cell.std()
        mean = self.crimes_per_cell.mean()
        median = np.median(self.crimes_per_cell)
        textstr = ' '.join((
            r'$\mu=%.2f$' % (mean,),
            r'$\sigma=%.2f$' % (stdev,),
            r'$threshold val=%.0f$ ' % threshold_val))

        # display info
        plt.title(textstr, fontsize=8)

crimes_map = CrimeMap()
crimes_map.plotCrimeMap()
