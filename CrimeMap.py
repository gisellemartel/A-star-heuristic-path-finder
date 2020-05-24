# -------------------------------------------------------
# Assignment 1
# Written by Giselle Martel 26352936
# For COMP 472 Section JX – Summer 2020
# --------------------------------------------------------

import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.widgets import Slider, Button
import matplotlib.ticker as tkr
import numpy as np


#TODO: When parsing info from file, make grid object containing every point (tick) on grid
# this will be used by A*
class Cell:
    def __init__(self):
        self.vertices = [Node(), Node(), Node(), Node()]
        self.is_high_crime_area = False
        self.num_crimes = 0

    def display(self):
        print('\ncell vertices')
        for v in self.vertices:
            v.display()

        print('\nis_obstruction: ' + self.is_obstruction)
        print('\nnum_crimes: ' + self.num_crimes)


class Node:
    def __init__(self):
        self.grid_pos = [0, 0]
        self.lat_long = [-73.58, 45.51]
        self.adjacent_nodes = []
        self.adjacent_cells = []
        self.h = 0
        self.g = 0
        self.f = 0

    def display(self):
        print('adjacent_nodes')
        for node in self.adjacent_nodes:
            node.display()

        print('adjacent_cells')
        for cell in self.adjacent_cells:
            cell.display()

        print('h:' + str(self.h))
        print('\ng:' + str(self.g))
        print('\nf:' + str(self.f))


def findPositionOfTick(grid_ticks, p):
    # if user selects point that is not grid tick, it will select the tick to the left of the point on x-axis
    for i in range(0, len(grid_ticks) - 1):
        if grid_ticks[i] == p:
            return i
        if grid_ticks[i] > p:
            # print(i - 1)
            if i - 1 > 0:
                return i - 1
            else:
                return i


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
        self.cmap = matplotlib.colors.LinearSegmentedColormap.from_list('Crime Points MTL', ['white', 'orange'])

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

        self.threshold_val = 0

        self.start = -1
        self.goal = -1
        self.gridXMarkings = []

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
        sthreshold = Slider(axthreshold, 'Threshold %', 0, 100, valinit=50, valstep=1, valfmt='%0.0f')
        scell = Slider(axcell, 'Cell Size', 0.001, 0.005, valinit=0.002, valstep=0.001, valfmt='%0.3f')
        btoggle = Button(axtoggle, 'hide/show\ndata')
        sthreshold.label.set_fontsize(8)
        scell.label.set_fontsize(8)
        btoggle.label.set_fontsize(4)

        def update(val):
            self.threshold = sthreshold.val
            self.step_size = scell.val
            self.start = -1
            self.goal = -1
            self.clearPointsOnMap()
            self.updateMap()

        sthreshold.on_changed(update)
        scell.on_changed(update)
        btoggle.on_clicked(self.onPressDataToggle)
        plt.ioff()
        self.axmap.set_picker(self.onPickMapCoordinate)

        plt.show()

    def drawPointOnMap(self, i, j, symbol, color):
        x_node = self.grid_x_ticks[i]
        y_node = self.grid_y_ticks[j]
        start = self.axmap.text(x_node, y_node, symbol, fontdict=dict(fontsize=8, ha='center', va='center', color=color))
        self.gridXMarkings.append(start)
        plt.draw()

    def clearPointsOnMap(self):
        for marking in self.gridXMarkings:
            marking.remove()
            marking = None
        self.gridXMarkings = []

    def onPickMapCoordinate(self, artist, mouseevent):
        # get the point the user selected on grid
        x, y = mouseevent.xdata, mouseevent.ydata

        # if the user selected invalid area outside bounds of grid
        if x is None or y is None \
                or x < self.total_bounds[0] or x > self.total_bounds[2] \
                or y < self.total_bounds[1] or y > self.total_bounds[3]:
            self.clearPointsOnMap()

        # if the start point is not set
        if self.start == -1:
            self.start = self.findPosOnGridFromPoint([x, y])
            # if the selected point was somehow invalid, clear markers and do nothing
            if self.start[0] == -1 or self.start[1] == -1:
                self.start = -1
                self.clearPointsOnMap()
            else:
                self.drawPointOnMap(self.start[0], self.start[1], 'O', 'r')

        # if the start point is set but not the destination
        elif self.start != -1 and self.goal == -1:
            # set the destination and then conduct AStarSearch
            self.goal = self.findPosOnGridFromPoint([x, y])
            # if the selected point was somehow invalid, clear markers and do nothing
            if self.goal[0] == -1 or self.goal[1] == -1:
                self.goal = -1
            # if the user chose same end point as start point, then do nothing
            elif self.start == self.goal:
                self.goal = -1
            else:
                self.drawPointOnMap(self.goal[0], self.goal[1], 'X', 'g')
                # call the AStarSearch on the start and goal points
                self.aStarSearch()

        # user is conducting new search
        else:
            self.clearPointsOnMap()
            self.goal = -1
            self.start = self.findPosOnGridFromPoint([x, y])
            # if the selected point was somehow invalid, clear markers and do nothing
            if self.start[0] == -1 or self.start[1] == -1:
                self.start = -1
            else:
                self.drawPointOnMap(self.start[0], self.start[1], 'O', 'r')

        print(self.start)
        print(self.goal)

        return True, {}

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
        if self.threshold < 100:
            threshold_index = int(len(crimes_per_cell_sorted) * (1 - self.threshold * 0.01)) - 1
            self.threshold_val = crimes_per_cell_sorted[threshold_index]
        else:
            threshold_index = 0
            self.threshold_val = crimes_per_cell_sorted[threshold_index] + 1

        # determine the norm of the grid based on the threshhold value
        grid_norm = BoundaryNorm([self.threshold_val], ncolors=self.cmap.N)

        # use matplotlib hist2d function to plot grid with given step size and threshold value
        crime_map = self.axmap.hist2d(
            self.x_values,
            self.y_values,
            bins=grid_dimensions,
            cmap=self.cmap,
            norm=grid_norm,
            )

        padding = .002

        plt.xlim(self.total_bounds[0]-padding, self.total_bounds[2]+padding)
        plt.ylim(self.total_bounds[1]-padding, self.total_bounds[3]+padding)

        self.axmap.xaxis.set_minor_locator(tkr.AutoMinorLocator(n=4))
        self.axmap.xaxis.set_minor_formatter(tkr.FixedFormatter(grid_dimensions))

        # update the tick values and # of crimes per cell
        self.crimes_per_cell = np.array(crime_map[0])
        self.grid_x_ticks = np.array(crime_map[1])
        self.grid_y_ticks = np.array(crime_map[2])

        if self.show_data:
            self.setGridDisplayData()
        self.displayStDevAndMean()

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

    def displayStDevAndMean(self):
        stdev = self.crimes_per_cell.std()
        mean = self.crimes_per_cell.mean()

        if self.start != -1:
            x1 = '%.3f' % self.grid_x_ticks[self.start[0]]
            y1 = '%.3f' % self.grid_y_ticks[self.start[1]]
            start = 'start: (' + str(x1) + ',' + str(y1) + ')'
        else:
            start = ''

        if self.goal != -1:
            x2 = '%.3f' % self.grid_x_ticks[self.goal[0]]
            y2 = '%.3f' % self.grid_y_ticks[self.goal[1]]
            end = 'goal: (' + str(x2) + ',' + str(y2) + ')'
        else:
            end = ''

        textstr = ' '.join((
            r'$\mu=%.2f$' % (mean,),
            r'$\sigma=%.2f$' % (stdev,),
            r'$threshold\ value=%.0f$ ' % (self.threshold_val,),
            start,
            end,
        ))


        # display info
        plt.title(textstr, fontsize=8)

    def findPosOnGridFromPoint(self, point):
        x = point[0]
        y = point[1]

        if x is None or y is None:
            return [-1, -1]
        if self.grid_x_ticks[0] > x or x > self.grid_x_ticks[len(self.grid_x_ticks) - 1]:
            return [-1, -1]
        if self.grid_y_ticks[0] > y or y > self.grid_y_ticks[len(self.grid_y_ticks) - 1]:
            return [-1, -1]

        # if user selects point that is not grid tick, it will select the tick to the left of the point on x-axis
        x_pos = findPositionOfTick(self.grid_x_ticks, x)

        # if user selects point that is not grid tick, it will select the tick below the point on the y-axis
        y_pos = findPositionOfTick(self.grid_y_ticks, y)

        return [x_pos, y_pos]

    def aStarSearch(self):

        if self.start[0] == -1 or self.start[1] == -1 or self.goal[0] == -1 or self.goal[1] == -1:
            print('Invalid value given, point found outside of boundaries of map!')
            return

        print(self.start)
        print(self.goal)
        plt.title('Initiating A* search...', fontsize=8)


        open_list = []
        closed_list = []

        # add start node to open list
        open_list.append(self.start)


        self.axmap.annotate("",
                            xy=(self.start[0], self.start[1]), xycoords='data',
                            xytext=(self.goal[0], self.goal[1]), textcoords='data',
                            arrowprops=dict(arrowstyle="-", connectionstyle="arc3,rad=0."),)
        plt.draw()

        plt.title('Search Done!', fontsize=8)
        self.displayStDevAndMean()

def main():
    crimes_map = CrimeMap()
    crimes_map.plotCrimeMap()


if __name__ == '__main__':
    main()