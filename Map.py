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
from matplotlib.colors import ListedColormap
from random import random

# entry point of app
def init():
    plt.ion()
    plt.rcParams.update({'figure.dpi': 200})
    matplotlib.use("TkAgg")

    createMapFromShpFile("./Shape/crime_dt.shp", 0.002)

    plt.ioff()
    plt.show()


def createMapFromShpFile(file, step_size):
    # read data from shp file and fetch attribute containing crime point data
    data = gpd.read_file(file)
    values = data.values

    # get all the crime points on the map
    crime_points = [v[3] for v in values]

    # get the bounds of the whole crime area
    min_x = data.total_bounds[0]
    min_y = data.total_bounds[1]
    max_x = data.total_bounds[2]
    max_y = data.total_bounds[3]

    print(min_x)
    print(max_x)
    print(min_y)
    print(max_y)

    # generate array containing tick of horizontal & vertical grid lines
    # i.e. x-values: [-73.590, -73.588, -73.586, ..... , -73.550]
    x_grid_steps = np.arange(min_x, max_x, step_size)
    y_grid_steps = np.arange(min_y, max_y, step_size)

    print('len of x steps:')
    print(len(x_grid_steps))
    print(len(y_grid_steps))

    # create the grid with desired cell-size
    generateMapGrid(x_grid_steps, y_grid_steps, data, crime_points)


def generateMapGrid(x_steps, y_steps, data, crime_points):
    polygons = []

    # create polygon object for each cell of the map grid
    #
    #       p1-----p4
    #       |      |
    #       |      |
    #       |      |
    #       p2-----p3
    #
    for i in range(0, len(x_steps) - 1):
        for j in range(0, len(y_steps) - 1):
            p1 = (x_steps[i], y_steps[j + 1])
            p2 = (x_steps[i], y_steps[j])
            p3 = (x_steps[i + 1], y_steps[j])
            p4 = (x_steps[i + 1], y_steps[j + 1])
            print(i)
            print(j)
            print(i + 1)
            print(j + 1)
            print('\n')
            grid_cell = [p1, p2, p3, p4]
            grid_cell_polygon = Polygon(grid_cell)
            polygons.append(grid_cell_polygon)

    grid = gpd.GeoDataFrame({'geometry': polygons})
    grid.plot()

init()
