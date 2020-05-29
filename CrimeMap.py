# -------------------------------------------------------
# Assignment 1
# Written by Giselle Martel 26352936
# For COMP 472 Section JX - Summer 2020
# --------------------------------------------------------

import matplotlib

matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.widgets import Slider, Button
import geopandas as gpd
import numpy as np
from queue import PriorityQueue
import time

MAX_SEARCH_TIME = 10

DIAGONAL_EDGE_COST = 1.5
CRIME_EDGE_COST = 1.3
SAFE_EDGE_COST = 1


class Cell:
    def __init__(self, p1, p2, p3, p4, x_pos, y_pos, is_high_crime_area, num_crimes):
        self.vertices = [p1, p2, p3, p4]
        self.grid_pos = [x_pos, y_pos]
        self.is_high_crime_area = is_high_crime_area
        self.num_crimes = num_crimes

    def __eq__(self, cell):
        return self.vertices == cell.vertices \
               and self.is_high_crime_area == cell.is_high_crime_area \
               and self.num_crimes == cell.num_crimes

    def display(self):
        # print('\ncell vertices')
        # for v in self.vertices:
        #     points = v.grid_pos
        #     print(points)
        # for v in self.vertices:
        #     points = v.lat_long
        #     print(points)

        print('is_high_crime_area: ' + str(self.is_high_crime_area))
        print('num_crimes: ' + str(self.num_crimes))


class Node:
    def __init__(self, x_pos, y_pos, x_tick, y_tick):
        self.grid_pos = [x_pos, y_pos]
        self.lat_long = [x_tick, y_tick]
        self.adjacent_nodes = []
        self.adjacent_cells = []
        self.h = 0
        self.g = 0
        self.f = 0
        self.cumulative_g = 0
        self.parent = None

    def __eq__(self, node):
        return self.grid_pos == node.grid_pos

    def __lt__(self, node):
        return self.g < node.g

    def add_adjacent_cell(self, cell):
        self.adjacent_cells.append(cell)

    def display(self):
        print('grid pos: ' + str(self.grid_pos))
        print('h:' + str(self.h))
        print('g:' + str(self.g))
        print('f:' + str(self.f))

        if len(self.adjacent_nodes) > 0:
            print('adjacent_nodes')
            for node in self.adjacent_nodes:
                node.display()

        # print('adjacent_cells')
        # for cell in self.adjacent_cells:
        #     cell.display()

        print('\n')


def find_pos_of_tick(grid_ticks, p, margin):
    #if user select point outside grid boundary, find nearest point:
    distance_from_boundary = abs(grid_ticks[0] - p)
    if distance_from_boundary < margin/2:
        return 0

    distance_from_boundary = abs(grid_ticks[len(grid_ticks) - 1] - p)
    if distance_from_boundary < margin/2:
        return len(grid_ticks) - 1

    # if user selects point that is not grid tick, it will select the tick to the left of the point on x-axis
    for i in range(0, len(grid_ticks)):
        if grid_ticks[i] == p:
            return i
        if grid_ticks[i] > p:
            # print(i - 1)
            if i - 1 > 0:
                return i - 1
            else:
                return i


def is_in_list(el, lst):
    for item in lst:
        if el == item:
            return True
    return False


class CrimeMap:

    def __init__(self):
        # create the plot figure
        # read data from shp file and fetch attribute containing crime point data
        self.data = gpd.read_file("./Shape/crime_dt.shp")

        # get all the crime points on the map
        self.crime_points = [v[3] for v in self.data.values]
        self.x_values = [point.x for point in self.crime_points]
        self.y_values = [point.y for point in self.crime_points]

        # generate a color map to represent the crime areas
        self.cmap = matplotlib.colors.LinearSegmentedColormap.from_list('Crime Points MTL', ['white', 'purple'])

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
        self.plot_stats = ""

        self.threshold_val = 0

        self.start = -1
        self.goal = -1
        self.grid_markings = []
        self.plot_lines = []
        self.grid_size = 0

        # data to be used by A*
        self.cells = []
        self.nodes = {}

    def on_press_data_toggle(self, event):
        self.show_data = not self.show_data
        self.set_grid_display_data()

    def plot_crime_map(self):
        plt.ion()
        plt.rcParams.update({'figure.dpi': 200, 'font.size': 5})
        # set layout of figures
        axthreshold = plt.axes([0.2, 0.02, 0.5, 0.03])
        axcell = plt.axes([0.2, 0.1, 0.4, 0.03])
        axtoggle = plt.axes([0.75, 0.02, 0.08, 0.08])
        axexit = plt.axes([0.1, 0.92, 0.05, 0.05])
        self.axmap = plt.axes([0.1, 0.2, 0.7, 0.7])

        self.update_crime_map()

        # sliders to change threshold and cell size interactively and button to toggle data view
        sthreshold = Slider(axthreshold, 'Threshold %', 0, 100, valinit=50, valstep=1, valfmt='%0.0f')
        scell = Slider(axcell, 'Cell Size', 0.001, 0.005, valinit=0.002, valstep=0.001, valfmt='%0.3f')
        btoggle = Button(axtoggle, 'hide/show\ndata', hovercolor='purple')
        bexit = Button(axexit, 'exit', color='red', hovercolor='grey')
        sthreshold.label.set_fontsize(8)
        scell.label.set_fontsize(8)
        btoggle.label.set_fontsize(6)
        bexit.label.set_fontsize(6)

        def update(val):
            self.threshold = sthreshold.val
            self.step_size = scell.val
            self.start = -1
            self.goal = -1
            self.clear_points_on_map()
            self.update_crime_map()

        def on_press_exit(event):
            quit()

        sthreshold.on_changed(update)
        scell.on_changed(update)
        btoggle.on_clicked(self.on_press_data_toggle)
        bexit.on_clicked(on_press_exit)
        plt.ioff()

        self.axmap.set_picker(self.on_pick_map_coordinate)

        plt.show()

    def draw_point_on_map(self, i, j, symbol, color):
        x_node = self.grid_x_ticks[i]
        y_node = self.grid_y_ticks[j]
        start = self.axmap.text(x_node, y_node, symbol,
                                fontdict=dict(fontsize=8, ha='center', va='center', color=color))
        self.grid_markings.append(start)
        plt.draw()

    def clear_points_on_map(self):
        for marking in self.grid_markings:
            marking.remove()
        self.grid_markings = []

        for line in self.plot_lines:
            line.set_xdata([])
            line.set_ydata([])
            line.remove()
        self.plot_lines = []

    def on_pick_map_coordinate(self, artist, mouseevent):
        # get the point the user selected on grid
        x, y = mouseevent.xdata, mouseevent.ydata

        # if the user selected invalid area outside bounds of grid
        if x is None or y is None:
            return True, {}

        # if the start point is not set
        if self.start == -1:
            self.start = self.find_bottom_left_cell_vertex_from_point([x, y])
            # if the selected point was somehow invalid, clear markers and do nothing
            if self.start[0] == -1 or self.start[1] == -1:
                self.start = -1
                self.clear_points_on_map()
            else:
                self.draw_point_on_map(self.start[0], self.start[1], 'S', 'r')

        # if the start point is set but not the destination
        elif self.start != -1 and self.goal == -1:
            # set the destination and then conduct AStarSearch
            self.goal = self.find_bottom_left_cell_vertex_from_point([x, y])
            # if the selected point was somehow invalid, clear markers and do nothing
            if self.goal[0] == -1 or self.goal[1] == -1:
                self.goal = -1
            # if the user chose same end point as start point, then do nothing
            elif self.start == self.goal:
                self.goal = -1
            else:
                self.draw_point_on_map(self.goal[0], self.goal[1], 'G', 'g')
                # call the AStarSearch on the start and goal points
                self.a_star_search()

        # user is conducting new search
        else:
            self.clear_points_on_map()
            self.goal = -1
            self.start = self.find_bottom_left_cell_vertex_from_point([x, y])
            # if the selected point was somehow invalid, clear markers and do nothing
            if self.start[0] == -1 or self.start[1] == -1:
                self.start = -1
            else:
                self.draw_point_on_map(self.start[0], self.start[1], 'S', 'r')

        return True, {}

    def add_adjacent_cell_to_node(self, idx, cell):
        # add the cells that are next to a node
        # | - - - - | - - - - |
        # |     1   |    2    |
        # |         |         |
        # | - - - - N - - - - |
        # |    3    |    4    |
        # |         |         |
        # | - - - - | - - - - |
        #
        if idx in self.nodes and self.nodes[idx] is not None:
            self.nodes[idx].add_adjacent_cell(cell)

    def parse_corner_node(self, i, cell, pos):
        x = pos[0]
        y = pos[1]

        adj_pos = []

        if x == 0:
            adj_pos.insert(0, 1)
        elif x == self.grid_size - 1:
            adj_pos.insert(0, self.grid_size - 2)

        if y == 0:
            adj_pos.insert(1, 1)
        elif x == self.grid_size - 1:
            adj_pos.insert(1, self.grid_size - 2)

        # only 1 possible path (as long as adjacent cell is not blocked)
        if not cell.is_high_crime_area:
            self.add_node_to_adjacency_lst(i, adj_pos, DIAGONAL_EDGE_COST)

    def add_node_to_adjacency_lst(self, i, pos, cost):
        node = self.get_node_by_pos(pos)
        if node:
            node.g = cost
            self.nodes[i].adjacent_nodes.append((cost, node))

    def get_node_by_pos(self, pos):
        for i in self.nodes:
            if self.nodes[i] and self.nodes[i].grid_pos == pos:
                return self.nodes[i]
        return None

    def parse_grid_nodes(self):
        # find all the adjacent nodes for each node in grid and place them in an adjacency list
        #
        #   X - - X - - X
        #   |     |     |
        #   |     |     |
        #   X - - N - - X
        #   |     |     |
        #   |     |     |
        #   X - - X - - X
        #
        for i in self.nodes:
            # get the adj cells of the current node
            cells = self.nodes[i].adjacent_cells
            node_x, node_y = self.nodes[i].grid_pos

            # node is one of 4 grid corners
            if len(cells) == 1:
                cell = cells[0]

                # bottom left
                if cell.grid_pos == [0, 0]:
                    self.parse_corner_node(i, cell, [0, 0])

                # top left
                elif cell.grid_pos == [0, self.grid_size - 1]:
                    self.parse_corner_node(i, cell, [0, self.grid_size - 1])

                # bottom right
                elif cell.grid_pos == [self.grid_size - 1, 0]:
                    self.parse_corner_node(i, cell, [self.grid_size - 1, 0])

                # top right
                elif cell.grid_pos == [self.grid_size - 1, self.grid_size - 1]:
                    self.parse_corner_node(i, cell, [self.grid_size - 1, self.grid_size - 1])

            # node is a non-corner boundary node
            elif len(cells) == 2:
                # both adjacent cells are high crime and therefore no path possible from this node
                if cells[0].is_high_crime_area and cells[1].is_high_crime_area:
                    continue

                # find the possible paths from boundary node
                # note that paths along boundary edges of graph not permitted
                else:
                    x0, y0 = cells[0].grid_pos
                    x1, y1 = cells[1].grid_pos

                    # cell 1 is located lower-left quadrant and cell 2 is located in upper-left quadrant
                    if x0 < node_x and y0 < node_y and x1 < node_x and y1 == node_y:
                        if cells[0].is_high_crime_area:
                            # path along left horizontal possible
                            self.add_node_to_adjacency_lst(i, [node_x - 1, node_y], CRIME_EDGE_COST)
                            # path along upper left diagonal possible
                            self.add_node_to_adjacency_lst(i, [node_x - 1, node_y + 1], DIAGONAL_EDGE_COST)
                        elif cells[1].is_high_crime_area:
                            # path along left horizontal possible
                            self.add_node_to_adjacency_lst(i, [node_x - 1, node_y], CRIME_EDGE_COST)
                            # path along lower left diagonal possible
                            self.add_node_to_adjacency_lst(i, [node_x - 1, node_y - 1], DIAGONAL_EDGE_COST)
                        else:
                            # path along left horizontal possible
                            self.add_node_to_adjacency_lst(i, [node_x - 1, node_y], SAFE_EDGE_COST)
                            # path along both left diagonals possible
                            self.add_node_to_adjacency_lst(i, [node_x - 1, node_y + 1], DIAGONAL_EDGE_COST)
                            self.add_node_to_adjacency_lst(i, [node_x - 1, node_y - 1], DIAGONAL_EDGE_COST)

                    # cell 1 is located lower-left quadrant and cell 2 is located in lower right quadrant
                    elif x0 < node_x and y0 < node_y and x1 == node_x and y1 < node_y:
                        if cells[0].is_high_crime_area:
                            # path along lower vertical possible
                            self.add_node_to_adjacency_lst(i, [node_x, node_y - 1], CRIME_EDGE_COST)
                            # path along lower left diagonal possible
                            self.add_node_to_adjacency_lst(i, [node_x - 1, node_y - 1], DIAGONAL_EDGE_COST)
                        elif cells[1].is_high_crime_area:
                            # path along lower vertical possible
                            self.add_node_to_adjacency_lst(i, [node_x, node_y - 1], CRIME_EDGE_COST)
                            # path along lower right diagonal possible
                            self.add_node_to_adjacency_lst(i, [node_x + 1, node_y - 1], DIAGONAL_EDGE_COST)
                        else:
                            # path along lower vertical possible
                            self.add_node_to_adjacency_lst(i, [node_x, node_y - 1], SAFE_EDGE_COST)
                            # path along both lower diagonals possible
                            self.add_node_to_adjacency_lst(i, [node_x - 1, node_y - 1], DIAGONAL_EDGE_COST)
                            self.add_node_to_adjacency_lst(i, [node_x + 1, node_y - 1], DIAGONAL_EDGE_COST)

                    # cell 1 is located in lower-right quadrant and cell 2 is located in upper-right quadrant
                    elif x0 == node_x and y0 < node_y and x1 == node_x and y1 == node_y:
                        if cells[0].is_high_crime_area:
                            # path along right horizontal possible
                            self.add_node_to_adjacency_lst(i, [node_x + 1, node_y], CRIME_EDGE_COST)
                            # path along upper right diagonal possible
                            self.add_node_to_adjacency_lst(i, [node_x + 1, node_y + 1], DIAGONAL_EDGE_COST)
                        elif cells[1].is_high_crime_area:
                            # path along right horizontal possible
                            self.add_node_to_adjacency_lst(i, [node_x + 1, node_y], CRIME_EDGE_COST)
                            # path along lower right diagonal possible
                            self.add_node_to_adjacency_lst(i, [node_x + 1, node_y - 1], DIAGONAL_EDGE_COST)
                        else:
                            # path along right horizontal possible
                            self.add_node_to_adjacency_lst(i, [node_x + 1, node_y], SAFE_EDGE_COST)
                            # path along both right diagonals possible
                            self.add_node_to_adjacency_lst(i, [node_x + 1, node_y + 1], DIAGONAL_EDGE_COST)
                            self.add_node_to_adjacency_lst(i, [node_x + 1, node_y - 1], DIAGONAL_EDGE_COST)

                    # cell 1 is located in upper-left quadrant and cell 2 is located in upper-right quandrant
                    elif x0 < node_x and y0 == node_y and x1 == node_x and y1 == node_y:
                        if cells[0].is_high_crime_area:
                            # path along upper vertical possible
                            self.add_node_to_adjacency_lst(i, [node_x, node_y + 1], CRIME_EDGE_COST)
                            # path along upper left diagonal possible
                            self.add_node_to_adjacency_lst(i, [node_x - 1, node_y + 1], DIAGONAL_EDGE_COST)
                        elif cells[1].is_high_crime_area:
                            # path along upper vertical possible
                            self.add_node_to_adjacency_lst(i, [node_x, node_y + 1], CRIME_EDGE_COST)
                            # path along upper right diagonal possible
                            self.add_node_to_adjacency_lst(i, [node_x + 1, node_y + 1], DIAGONAL_EDGE_COST)
                        else:
                            # path along upper vertical possible
                            self.add_node_to_adjacency_lst(i, [node_x, node_y + 1], SAFE_EDGE_COST)
                            # path along both upper diagonals possible
                            self.add_node_to_adjacency_lst(i, [node_x - 1, node_y + 1], DIAGONAL_EDGE_COST)
                            self.add_node_to_adjacency_lst(i, [node_x + 1, node_y + 1], DIAGONAL_EDGE_COST)

            # non-boundary node
            elif len(cells) == 4:
                # all adjacent cells are high crime and therefore no path possible from this node
                if cells[0].is_high_crime_area and cells[1].is_high_crime_area and cells[2].is_high_crime_area and \
                        cells[3].is_high_crime_area:
                    continue
                else:
                    pass
                    # all diagonals that fall within a low crime area cell are valid paths
                    for j in range(0, len(cells)):
                        if not cells[j].is_high_crime_area:
                            x, y = cells[j].grid_pos
                            # bottom left
                            if x < node_x and y < node_y:
                                self.add_node_to_adjacency_lst(i, [node_x - 1, node_y - 1], DIAGONAL_EDGE_COST)
                            # top left
                            elif x < node_x and y == node_y:
                                self.add_node_to_adjacency_lst(i, [node_x - 1, node_y + 1], DIAGONAL_EDGE_COST)
                            # bottom right
                            elif x == node_x and y < node_y:
                                self.add_node_to_adjacency_lst(i, [node_x + 1, node_y - 1], DIAGONAL_EDGE_COST)
                            # top right
                            elif x == node_x and y == node_y:
                                self.add_node_to_adjacency_lst(i, [node_x + 1, node_y + 1], DIAGONAL_EDGE_COST)
                            else:
                                print('SOMETHING WENT WRONG _ _ _ _ _ _ _ _ _ _ _ _ _ ')

                    # horizontal and vertical paths
                    # left horizontal
                    if cells[0].is_high_crime_area and cells[1].is_high_crime_area:
                        pass
                    elif not cells[0].is_high_crime_area and not cells[1].is_high_crime_area:
                        self.add_node_to_adjacency_lst(i, [node_x - 1, node_y], SAFE_EDGE_COST)
                    elif not cells[0].is_high_crime_area or not cells[1].is_high_crime_area:
                        self.add_node_to_adjacency_lst(i, [node_x - 1, node_y], CRIME_EDGE_COST)

                    # right horizontal
                    if cells[2].is_high_crime_area and cells[3].is_high_crime_area:
                        pass
                    elif not cells[2].is_high_crime_area and not cells[3].is_high_crime_area:
                        self.add_node_to_adjacency_lst(i, [node_x + 1, node_y], SAFE_EDGE_COST)
                    elif not cells[2].is_high_crime_area or not cells[3].is_high_crime_area:
                        self.add_node_to_adjacency_lst(i, [node_x + 1, node_y], CRIME_EDGE_COST)

                    # north vertical
                    if cells[1].is_high_crime_area and cells[3].is_high_crime_area:
                        pass
                    elif not cells[1].is_high_crime_area and not cells[3].is_high_crime_area:
                        self.add_node_to_adjacency_lst(i, [node_x, node_y + 1], SAFE_EDGE_COST)
                    elif not cells[1].is_high_crime_area or not cells[3].is_high_crime_area:
                        self.add_node_to_adjacency_lst(i, [node_x, node_y + 1], CRIME_EDGE_COST)

                    # south vertical
                    if cells[0].is_high_crime_area and cells[2].is_high_crime_area:
                        pass
                    elif not cells[0].is_high_crime_area and not cells[2].is_high_crime_area:
                        self.add_node_to_adjacency_lst(i, [node_x, node_y - 1], SAFE_EDGE_COST)
                    elif not cells[0].is_high_crime_area or not cells[2].is_high_crime_area:
                        self.add_node_to_adjacency_lst(i, [node_x, node_y - 1], CRIME_EDGE_COST)

    def parse_grid_cells(self, crime_map):
        self.cells = []
        self.nodes = {}
        crimes = crime_map[0]
        n = len(crimes)

        # create node object for each vertex in grid
        # create cell object for each cell in grid. each cell has 4 vertices (nodes) to be used in A* path finder
        #
        #
        #   top_left - - - - - - top_right
        #   |                            |
        #   |                            |
        #   |                            |
        #   |             GRID           |
        #   |             CELL           |
        #   |                            |
        #   |                            |
        #   |                            |
        #   bottom_left - - - bottom_right
        #
        # bottom_left, bottom_right, top_left, top_right represent nodes and the cube represents a cell
        for i in range(0, n + 1):
            for j in range(0, n + 1):

                x1 = i % (n + 1)
                y1 = j % (n + 1)
                x2 = (i + 1) % (n + 1)
                y2 = (j + 1) % (n + 1)

                bottom_left = Node(x1, y1, self.grid_x_ticks[x1], self.grid_y_ticks[y1])
                bottom_right = Node(x2, y1, self.grid_x_ticks[x2], self.grid_y_ticks[y1])
                top_left = Node(x1, y2, self.grid_x_ticks[x1], self.grid_y_ticks[y2])
                top_right = Node(x2, y2, self.grid_x_ticks[x2], self.grid_y_ticks[y2])

                if i < n and j < n:
                    num_crimes = crimes[i][j]
                    global cell
                    # determine if the cell is an obstruction if it has crimes that meet or exceed threshold
                    if  num_crimes >= self.threshold_val:
                        cell = Cell(bottom_left, bottom_right, top_left, top_right, i, j, True, num_crimes)
                        self.cells.append(cell)
                    else:
                        cell = Cell(bottom_left, bottom_right, top_left, top_right, i, j, False, num_crimes)
                        self.cells.append(cell)

                    # 1d representation of 2d data
                    max_cols = n + 1
                    pos1 = x1 * max_cols + y1
                    pos2 = x2 * max_cols + y1
                    pos3 = x1 * max_cols + y2
                    pos4 = x2 * max_cols + y2

                    max_num_nodes = (n+1)*(n+1)

                    # add nodes to nodes dictionary
                    if pos1 not in self.nodes and pos1 < max_num_nodes:
                        self.nodes[pos1] = bottom_left
                    if pos2 not in self.nodes and pos2 < max_num_nodes:
                        self.nodes[pos2] = bottom_right
                    if pos3 not in self.nodes and pos3 < max_num_nodes:
                        self.nodes[pos3] = top_left
                    if pos4 not in self.nodes and pos4 < max_num_nodes:
                        self.nodes[pos4] = top_right

                    self.add_adjacent_cell_to_node(pos1, cell)
                    self.add_adjacent_cell_to_node(pos2, cell)
                    self.add_adjacent_cell_to_node(pos3, cell)
                    self.add_adjacent_cell_to_node(pos4, cell)

        # debug
        # for i in sorted(self.nodes.keys()):
        #     print(i, end=': ')
        #     print(self.nodes[i].grid_pos)
        #     cells = self.nodes[i].adjacent_cells
        #     for c in cells:
        #         x1, y1 = c.grid_pos
        #         print('     [' + str(x1) + ',' + str(y1) + ']')

        # parse each node to make the adjacency_list of neighbour nodes
        self.parse_grid_nodes()

    def update_crime_map(self):
        # get the horizontal-vertical size of grid (X*Y)
        grid_dimensions = self.calc_grid_dimensions()

        self.grid_size = grid_dimensions[0]
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

        self.axmap.set_xticks(np.arange(self.total_bounds[0], self.total_bounds[2], self.step_size))
        self.axmap.set_yticks(np.arange(self.total_bounds[1], self.total_bounds[3], self.step_size))
        plt.setp(self.axmap.get_xticklabels()[1::2], visible=False)

        for tick in self.axmap.xaxis.get_minor_ticks():
            tick.tick1line.set_markersize(0)

        # update the tick values and # of crimes per cell
        self.crimes_per_cell = np.array(crime_map[0])
        self.grid_x_ticks = np.array(crime_map[1])
        self.grid_y_ticks = np.array(crime_map[2])

        self.parse_grid_cells(crime_map)

        if self.show_data:
            self.set_grid_display_data()
        self.gen_plot_stats()
        # display info
        plt.title(str(self.plot_stats) + '\nClick on grid to select start and goal', fontsize=8)

    def calc_grid_dimensions(self):
        # get the bounds of the whole crime area
        min_x, min_y, max_x, max_y = self.total_bounds
        # get num cells on x and y axis of grid based on bounds of crime area and step size
        x_grid_steps = np.ceil((max_x - min_x) / self.step_size)
        y_grid_steps = np.ceil((max_y - min_y) / self.step_size)

        return [int(x_grid_steps), int(y_grid_steps)]

    def set_grid_display_data(self):
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

    def gen_plot_stats(self):
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

        self.plot_stats = ' '.join((
            r'$\mu=%.2f$' % (mean,),
            r'$\sigma=%.2f$' % (stdev,),
            r'$threshold\ value=%.0f$ ' % (self.threshold_val,),
            start,
            end,
        ))

    def find_bottom_left_cell_vertex_from_point(self, point):
        x = point[0]
        y = point[1]

        if x is None or y is None:
            return [-1, -1]
        if self.grid_x_ticks[0] > x or x > self.grid_x_ticks[len(self.grid_x_ticks) - 1]:
            return [-1, -1]
        if self.grid_y_ticks[0] > y or y > self.grid_y_ticks[len(self.grid_y_ticks) - 1]:
            return [-1, -1]

        # if user selects point that is not grid tick, it will select the tick to the left of the point on x-axis
        x_pos = find_pos_of_tick(self.grid_x_ticks, x, self.step_size)

        # if user selects point that is not grid tick, it will select the tick below the point on the y-axis
        y_pos = find_pos_of_tick(self.grid_y_ticks, y, self.step_size)

        return [x_pos, y_pos]

    def show_search_admissibility(self, path, path_cost):
        # retrace back the costs of the shortest path
        path[0].cumulative_g = 0
        for i in range(1,len(path)):
            path[i].cumulative_g = path[i - 1].cumulative_g + path[i].g

        # do the admissibility check for each step
        for i in range(0, len(path)-1):
            h_star = round(path_cost - path[i].cumulative_g, 1)
            print('g=' + str(round(path[i].cumulative_g,1)) + '    ', end='')
            print('h=' + str(round(path[i].h, 1)) + '    ', end='')
            print('h*=' + str(round(h_star,1))+ '    ', end='')
            print('h <= h* ? - ' + str(path[i].h < h_star))

    def is_crime_edge(self, n1, n2):
        # determines if line formed by 2 nodes goes through a crime edge
        cells_1 = n1.adjacent_cells
        cells_2 = n2.adjacent_cells

        common_cells = [c for c in cells_1 if c in cells_2]

        return common_cells[0].is_high_crime_area ^ common_cells[1].is_high_crime_area

    def calc_step_cost(self, n1, n2):
        if abs(n1.grid_pos[0] - n2.grid_pos[0]) > 0 and abs(n1.grid_pos[1] - n2.grid_pos[1]):
            return DIAGONAL_EDGE_COST
        elif self.is_crime_edge(n1, n2):
            return CRIME_EDGE_COST
        else:
            return SAFE_EDGE_COST

    def print_path(self, path):
        # print the path
        for i in range(0, len(path) - 1):
            x1 = path[i].lat_long[0]
            y1 = path[i].lat_long[1]
            x2 = path[i + 1].lat_long[0]
            y2 = path[i + 1].lat_long[1]
            self.draw_path_line(x1, y1, x2, y2)

    def draw_path_line(self, x1, y1, x2, y2):
        line, =self.axmap.plot([x1, x2], [y1, y2], color='black')
        plt.pause(0.00001)
        self.plot_lines.append(line)

    def search_heuristic(self, curr_node, goal_node):
        # use a heuristic that calculates the diagonal distance, and then
        x1, y1 = curr_node.grid_pos
        x2, y2 = goal_node.grid_pos
        dx = abs(x1 - x2)
        dy = abs(y1 - y2)

        # the diagonal distance between the 2 nodes
        diagonal_distance = dx + dy
        # the amount of distance we save by using a diagonal over 2 straight lines to move to next cell
        # i.e. diagonal cost = 1.5 and two straight lines = 1 + 1 = 2
        savings_from_taking_diagonal = (DIAGONAL_EDGE_COST - 2 * SAFE_EDGE_COST)
        estimate_diagonal_cost_savings = savings_from_taking_diagonal * min(dx,dy)
        estimate_remaing_straight_steps_cost = SAFE_EDGE_COST * diagonal_distance

        # an estimation of how many diagonal and straight path steps we have left to take until the goal
        t_ratio = self.threshold * 0.01
        h = (estimate_remaing_straight_steps_cost + estimate_diagonal_cost_savings) * t_ratio
        return 0

    def a_star_search(self):
        plt.title(str(self.plot_stats) + "\nSearching for Goal...", fontsize=8)

        # Get start and goals nodes from parsed node dictionary
        start_node = self.get_node_by_pos(self.start)
        goal_node = self.get_node_by_pos(self.goal)

        # set timer for A* search time
        start_time = time.time()
        time_elapsed = 0.0

        # create empty open and closed lists
        open_list = PriorityQueue()
        closed_list = []

        # add start node to open list
        open_list.put((0.0, start_node))

        goal_found = False

        shortest_path = 0

        # Loop goal is found or all possible nodes visited
        while not open_list.empty() and time_elapsed < MAX_SEARCH_TIME:
            # Get the current node from open list
            current_cost, current_node = open_list.get()

            # We have found the goal
            if current_node == goal_node:
                # backtrack to get shortest path
                goal_node.cumulative_g = 0
                curr = goal_node
                path = []
                while curr != start_node:
                    step_cost = self.calc_step_cost(curr, curr.parent)
                    shortest_path = shortest_path + step_cost
                    path.append(curr)
                    curr.parent.g = step_cost
                    curr = curr.parent
                path.append(start_node)
                path.reverse()
                # calculate the elapsed time
                time_elapsed = time.time() - start_time
                print('A* search found shortest path of ' + str(shortest_path)
                      + ' successfully in ' + str(time_elapsed) + " seconds. Drawing path...")
                # print the path on the grid UI
                self.print_path(path)
                plt.title(str(self.plot_stats) + "\nSuccess! A* search found the goal in "
                          + str(round(time_elapsed,5)) + "s.\nTotal Cost: " +  str(round(shortest_path, 2)), fontsize=8)
                goal_found = True
                # debug
                # self.show_search_admissibility(path, shortest_path)
                break

            # add node to closed list once it has been visited
            closed_list.append(current_node)

            # iterate over neighbour nodes of the current node to populate the open list
            for cost, neighbour in current_node.adjacent_nodes:
                # debug code
                # x1, y1 = child_node.lat_long, x2, y2 = current_node.lat_long
                # self.draw_path_line(x1, y1, x2, y2)

                # set the cost from start to current node
                # neighbour.g = cost
                neighbour.cumulative_g = cost + current_node.cumulative_g
                # the estimates cost from current node to goal node
                neighbour.h = self.search_heuristic(current_node, goal_node)
                # f = g + h
                neighbour.f =  neighbour.cumulative_g + neighbour.h

                queue_nodes = [n for _,n in open_list.queue]
                # check to see if neighbour is already in open or closed list
                if neighbour not in closed_list and neighbour not in queue_nodes:
                    # if we have not yet visited or placed this node in the queue, then set its parent
                    neighbour.parent = current_node
                    open_list.put((neighbour.f, neighbour))
                # neighbour was found in open list, we need to see if there is cheaper path to it
                elif neighbour in queue_nodes:
                    new_pq_items = []
                    # if any nodes are replaced with an updated cost
                    # we need to set a flag to replace the priority queue with updated costs
                    should_replace = False
                    for f, n in open_list.queue:
                        # if same node is already in queue but with higher cost,
                        # we replace the higher cost node with the cheaper one
                        # we need to also make sure to set its new parent to current node
                        if neighbour == n and neighbour.f < f:
                            n.g = neighbour.g
                            n.cumulative_g = neighbour.cumulative_g
                            n.parent = current_node
                            new_pq_items.append((neighbour.f, n))
                            should_replace = True
                        # otherwise keep the neighbour node
                        else:
                            new_pq_items.append((f, n))

                    if should_replace:
                        # we have to reorder the priority queue open list if changes were made to costs
                        open_list = PriorityQueue()
                        for f_node_pair in new_pq_items:
                            open_list.put(f_node_pair)

                # update timer
                time_elapsed = time.time() - start_time

        # A* search either ran out of time or no valid path was possible due to obstacles
        if not goal_found and time_elapsed < MAX_SEARCH_TIME:
            print('Due to blocks, no path is found. Please change the map and try again')
            plt.title(str(self.plot_stats) + "\nDue to blocks, no path is found. Please change the map and try again", fontsize=7)
        elif not goal_found and time_elapsed >= MAX_SEARCH_TIME:
            print('Time is up. The optimal path is not found.')
            plt.title(str(self.plot_stats) + "\nTime is up. The optimal path is not found.", fontsize=8)
        plt.show()

def main():
    crimes_map = CrimeMap()
    crimes_map.plot_crime_map()


if __name__ == '__main__':
    main()
