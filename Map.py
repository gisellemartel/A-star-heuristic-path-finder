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

    # print(min_x)
    # print(max_x)
    # print(min_y)
    # print(max_y)

    # generate array containing tick of horizontal & vertical grid lines
    # i.e. x-values: [-73.590, -73.588, -73.586, ..... , -73.550]
    x_grid_steps = np.arange(min_x, max_x, step_size)
    y_grid_steps = np.arange(min_y, max_y, step_size)

    # print('len of x steps:')
    # print(len(x_grid_steps))
    # print(len(y_grid_steps))

    # create the grid with desired cell-size
    grid = generateMapGridCells(x_grid_steps, y_grid_steps)
    crimes_per_cell_map = generateCrimesPerCellMap(grid, crime_points)


def generateMapGridCells(x_steps, y_steps):
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
            # print(i)
            # print(j)
            # print(i + 1)
            # print(j + 1)
            # print('\n')
            grid_cell = [p1, p2, p3, p4]
            grid_cell_polygon = Polygon(grid_cell)
            polygons.append(grid_cell_polygon)

    # grid = gpd.GeoDataFrame({'geometry': polygons})
    # grid.plot()
    return polygons


def generateCrimesPerCellMap(grid, crime_points):
    for point in crime_points:
        p1 = point.x
        p2 = point.y

        for cell in grid:
            vertices = np.asarray(cell.exterior.coords)
            x1 = vertices[0][0]
            y1 = vertices[0][1]
            x2 = vertices[2][0]
            y2 = vertices[2][1]

            if isInGridCell(x1, y1, x2, y2, p1, p2):
                break


# determines if a point is inside of a grid cell given its top-left and bottom-right coordinates
def isInGridCell(x1, y1, x2, y2, p1, p2):
    print(p1 > x1 and p1 < x2 and
        p2 > y1 and p2 < y2)
    return x1 < p1 < x2 and y1 < p2 < y2


init()
