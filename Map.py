# -------------------------------------------------------
# Assignment 1
# Written by Giselle Martel 26352936
# For COMP 472 Section JX – Summer 2020
# --------------------------------------------------------

import geopandas as gpd
from shapely.geometry import Polygon
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm


# entry point of app
def init():
    plt.ion()
    plt.rcParams.update({'figure.dpi': 200})
    matplotlib.use("TkAgg")

    crime_map = createMapFromShpFile("./Shape/crime_dt.shp", 0.002, 50)

    generateCrimeMapDataForEachCell(crime_map)
    generateStandardDeviation(crime_map)
    generateMean(crime_map)

    plt.ioff()
    plt.show()


def createMapFromShpFile(file, step_size, threshold):
    # read data from shp file and fetch attribute containing crime point data
    data = gpd.read_file(file)
    values = data.values

    # create the plot figure
    plt.figure(figsize=(7, 7))

    # get all the crime points on the map
    crime_points = [v[3] for v in values]

    # generate a color map to represent the crime areas
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list('Crime Points MTL', ['purple', 'yellow'])

    x_values = [point.x for point in crime_points]
    y_values = [point.y for point in crime_points]

    # get the bounds of the crime map from shapefile data
    total_bounds = data.total_bounds

    # get the horizontal-vertical size of grid (X*Y)
    grid_size = calcNumGridCells(total_bounds, step_size)

    # determine the norm of the grid based on the given threshold
    grid_norm = BoundaryNorm([threshold], ncolors=cmap.N)

    # use matplotlib hist2d function to plot grid with given threshold and step size
    crime_map = plt.hist2d(x_values, y_values, bins=grid_size, cmap=cmap, norm=grid_norm)

    return crime_map


def calcNumGridCells(total_bounds, step_size):
    # get the bounds of the whole crime area
    min_x = total_bounds[0]
    min_y = total_bounds[1]
    max_x = total_bounds[2]
    max_y = total_bounds[3]

    # get num cells in grid based on bounds of crime area and step size
    x_grid_steps = np.ceil((max_x - min_x) / step_size)
    y_grid_steps = np.ceil((max_y - min_y) / step_size)

    return [int(x_grid_steps), int(y_grid_steps)]


def generateCrimeMapDataForEachCell(crime_map):
    # DEBUG
    vals = crime_map[0]
    xedges = crime_map[1]
    yedges = crime_map[2]
    xy_vertices = []
    for (x, xi) in zip(xedges, range(0, len(xedges) - 1)):
        for (y, yi) in zip(yedges, range(0, len(yedges) - 1)):
            xy_vertices.append([x, y])
            plt.text(x + 0.002 / 2, y + 0.002 / 2, str(vals[xi][yi]),
                     fontdict=dict(fontsize=3, ha='center', va='center'))
    print(vals)


def generateStandardDeviation(crime_map):
    pass


def generateMean(crime_map):
    pass






init()
