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
import matplotlib.ticker as tkr
import geopandas as gpd
import numpy as np
from queue import PriorityQueue

DIAGONAL_EDGE_COST = 1.5
CRIME_EDGE_COST = 1.2
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
        self.adjacent_nodes = PriorityQueue()
        self.adjacent_cells = []
        self.h = 0
        self.g = 0
        self.f = 0
        self.parent = None

    def __eq__(self, node):
        return self.grid_pos == node.grid_pos

    def __lt__(self, node):
        return self.g < node.g

    def addAdjacentCell(self, cell):
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

    def displayPriorityQ(self):
        for i in range(self.adjacent_nodes.qsize()):
            temp = self.adjacent_nodes.get()
            self.adjacent_nodes.put(temp)
            print(temp[0])
            print('h: ' + str(temp[1].h) + ', g: ' + str(temp[1].g) + ', f: ' + str(temp[1].f))
        print('\n')


def findPositionOfTick(grid_ticks, p):
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


def isInList(el, lst):
    for item in lst:
        if el == item:
            return True
    return False


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

        self.threshold_val = 0

        self.start = -1
        self.goal = -1
        self.gridMarkings = []
        self.grid_size = 0

        # data to be used by A*
        self.cells = []
        self.nodes = {}

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
        start = self.axmap.text(x_node, y_node, symbol,
                                fontdict=dict(fontsize=8, ha='center', va='center', color=color))
        self.gridMarkings.append(start)
        plt.draw()

    def clearPointsOnMap(self):
        for marking in self.gridMarkings:
            marking.remove()
            marking = None
        self.gridMarkings = []

    def onPickMapCoordinate(self, artist, mouseevent):
        # get the point the user selected on grid
        x, y = mouseevent.xdata, mouseevent.ydata

        # if the user selected invalid area outside bounds of grid
        if x is None or y is None \
                or x < self.total_bounds[0] or x > self.total_bounds[2] \
                or y < self.total_bounds[1] or y > self.total_bounds[3]:
            return True, {}

        # if the start point is not set
        if self.start == -1:
            self.start = self.findPosOnGridFromPoint([x, y])
            # if the selected point was somehow invalid, clear markers and do nothing
            if self.start[0] == -1 or self.start[1] == -1:
                self.start = -1
                self.clearPointsOnMap()
            else:
                self.drawPointOnMap(self.start[0], self.start[1], 'S', 'r')

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
                self.drawPointOnMap(self.goal[0], self.goal[1], 'G', 'g')
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
                self.drawPointOnMap(self.start[0], self.start[1], 'S', 'r')

        return True, {}

    def addAdjacentCellToNode(self, idx, cell):
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
            self.nodes[idx].addAdjacentCell(cell)

    def parseCornerNode(self, i, cell, pos):
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
            self.addNodeToPriorityQ(i, adj_pos, DIAGONAL_EDGE_COST)

    def addNodeToPriorityQ(self, i, pos, cost):
        node = self.getNodeByPos(pos)
        if node:
            node.g = cost
            self.nodes[i].adjacent_nodes.put((cost, node))

    def parseNodes(self):
        # find all the adjacent nodes for each node in grid and place them in priority queue based on actual cost
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
                    self.parseCornerNode(i, cell, [0, 0])

                # top left
                elif cell.grid_pos == [0, self.grid_size - 1]:
                    self.parseCornerNode(i, cell, [0, self.grid_size - 1])

                # bottom right
                elif cell.grid_pos == [self.grid_size - 1, 0]:
                    self.parseCornerNode(i, cell, [self.grid_size - 1, 0])

                # top right
                elif cell.grid_pos == [self.grid_size - 1, self.grid_size - 1]:
                    self.parseCornerNode(i, cell, [self.grid_size - 1, self.grid_size - 1])

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
                            self.addNodeToPriorityQ(i, [node_x - 1, node_y], CRIME_EDGE_COST)
                            # path along upper left diagonal possible
                            self.addNodeToPriorityQ(i, [node_x - 1, node_y + 1], DIAGONAL_EDGE_COST)
                        elif cells[1].is_high_crime_area:
                            # path along left horizontal possible
                            self.addNodeToPriorityQ(i, [node_x - 1, node_y], CRIME_EDGE_COST)
                            # path along lower left diagonal possible
                            self.addNodeToPriorityQ(i, [node_x - 1, node_y - 1], DIAGONAL_EDGE_COST)
                        else:
                            # path along left horizontal possible
                            self.addNodeToPriorityQ(i, [node_x - 1, node_y], SAFE_EDGE_COST)
                            # path along both left diagonals possible
                            self.addNodeToPriorityQ(i, [node_x - 1, node_y + 1], DIAGONAL_EDGE_COST)
                            self.addNodeToPriorityQ(i, [node_x - 1, node_y - 1], DIAGONAL_EDGE_COST)

                    # cell 1 is located lower-left quadrant and cell 2 is located in lower right quadrant
                    elif x0 < node_x and y0 < node_y and x1 == node_x and y1 < node_y:
                        if cells[0].is_high_crime_area:
                            # path along lower vertical possible
                            self.addNodeToPriorityQ(i, [node_x, node_y - 1], CRIME_EDGE_COST)
                            # path along lower left diagonal possible
                            self.addNodeToPriorityQ(i, [node_x - 1, node_y - 1], DIAGONAL_EDGE_COST)
                        elif cells[1].is_high_crime_area:
                            # path along lower vertical possible
                            self.addNodeToPriorityQ(i, [node_x, node_y - 1], CRIME_EDGE_COST)
                            # path along lower right diagonal possible
                            self.addNodeToPriorityQ(i, [node_x + 1, node_y - 1], DIAGONAL_EDGE_COST)
                        else:
                            # path along lower vertical possible
                            self.addNodeToPriorityQ(i, [node_x, node_y - 1], SAFE_EDGE_COST)
                            # path along both lower diagonals possible
                            self.addNodeToPriorityQ(i, [node_x - 1, node_y - 1], DIAGONAL_EDGE_COST)
                            self.addNodeToPriorityQ(i, [node_x + 1, node_y - 1], DIAGONAL_EDGE_COST)

                    # cell 1 is located in lower-right quadrant and cell 2 is located in upper-right quadrant
                    elif x0 == node_x and y0 < node_y and x1 == node_x and y1 == node_y:
                        if cells[0].is_high_crime_area:
                            # path along right horizontal possible
                            self.addNodeToPriorityQ(i, [node_x + 1, node_y], CRIME_EDGE_COST)
                            # path along upper right diagonal possible
                            self.addNodeToPriorityQ(i, [node_x + 1, node_y + 1], DIAGONAL_EDGE_COST)
                        elif cells[1].is_high_crime_area:
                            # path along right horizontal possible
                            self.addNodeToPriorityQ(i, [node_x + 1, node_y], CRIME_EDGE_COST)
                            # path along lower right diagonal possible
                            self.addNodeToPriorityQ(i, [node_x + 1, node_y - 1], DIAGONAL_EDGE_COST)
                        else:
                            # path along right horizontal possible
                            self.addNodeToPriorityQ(i, [node_x + 1, node_y], SAFE_EDGE_COST)
                            # path along both right diagonals possible
                            self.addNodeToPriorityQ(i, [node_x + 1, node_y + 1], DIAGONAL_EDGE_COST)
                            self.addNodeToPriorityQ(i, [node_x + 1, node_y - 1], DIAGONAL_EDGE_COST)

                    # cell 1 is located in upper-left quadrant and cell 2 is located in upper-right quandrant
                    elif x0 < node_x and y0 == node_y and x1 == node_x and y1 == node_y:
                        if cells[0].is_high_crime_area:
                            # path along upper vertical possible
                            self.addNodeToPriorityQ(i, [node_x, node_y + 1], CRIME_EDGE_COST)
                            # path along upper left diagonal possible
                            self.addNodeToPriorityQ(i, [node_x - 1, node_y + 1], DIAGONAL_EDGE_COST)
                        elif cells[1].is_high_crime_area:
                            # path along upper vertical possible
                            self.addNodeToPriorityQ(i, [node_x, node_y + 1], CRIME_EDGE_COST)
                            # path along upper right diagonal possible
                            self.addNodeToPriorityQ(i, [node_x + 1, node_y + 1], DIAGONAL_EDGE_COST)
                        else:
                            # path along upper vertical possible
                            self.addNodeToPriorityQ(i, [node_x, node_y + 1], SAFE_EDGE_COST)
                            # path along both upper diagonals possible
                            self.addNodeToPriorityQ(i, [node_x - 1, node_y + 1], DIAGONAL_EDGE_COST)
                            self.addNodeToPriorityQ(i, [node_x + 1, node_y + 1], DIAGONAL_EDGE_COST)

            # non-boundary node
            elif len(cells) == 4:
                # all adjacent cells are high crime and therefore no path possible from this node
                if cells[0].is_high_crime_area and cells[1].is_high_crime_area and cells[2].is_high_crime_area and cells[3].is_high_crime_area:
                    continue
                else:
                    pass
                    # all diagonals that fall within a low crime area cell are valid paths
                    for j in range(0, len(cells)):
                        if not cells[j].is_high_crime_area:
                            x, y = cells[j].grid_pos
                            # bottom left
                            if x < node_x and y < node_y:
                                self.addNodeToPriorityQ(i, [node_x - 1, node_y - 1], DIAGONAL_EDGE_COST)
                            # top left
                            elif x < node_x and y == node_y:
                                self.addNodeToPriorityQ(i, [node_x - 1, node_y + 1], DIAGONAL_EDGE_COST)
                            # bottom right
                            elif x == node_x and y < node_y:
                                self.addNodeToPriorityQ(i, [node_x + 1, node_y - 1], DIAGONAL_EDGE_COST)
                            # top right
                            elif x == node_x and y == node_y:
                                self.addNodeToPriorityQ(i, [node_x + 1, node_y + 1], DIAGONAL_EDGE_COST)
                            else:
                                print('SOMETHING WENT WRONG _ _ _ _ _ _ _ _ _ _ _ _ _ ')

                    # horizontal and vertical paths
                    # left horizontal
                    if cells[0].is_high_crime_area and cells[1].is_high_crime_area:
                        pass
                    elif not cells[0].is_high_crime_area and not cells[1].is_high_crime_area:
                        self.addNodeToPriorityQ(i, [node_x - 1, node_y], SAFE_EDGE_COST)
                    elif not cells[0].is_high_crime_area or not cells[1].is_high_crime_area:
                        self.addNodeToPriorityQ(i, [node_x - 1, node_y], CRIME_EDGE_COST)

                    # right horizontal
                    if cells[2].is_high_crime_area and cells[3].is_high_crime_area:
                        pass
                    elif not cells[2].is_high_crime_area and not cells[3].is_high_crime_area:
                        self.addNodeToPriorityQ(i, [node_x + 1, node_y], SAFE_EDGE_COST)
                    elif not cells[2].is_high_crime_area or not cells[3].is_high_crime_area:
                        self.addNodeToPriorityQ(i, [node_x + 1, node_y], CRIME_EDGE_COST)

                    # north vertical
                    if cells[1].is_high_crime_area and cells[3].is_high_crime_area:
                        pass
                    elif not cells[1].is_high_crime_area and not cells[3].is_high_crime_area:
                        self.addNodeToPriorityQ(i, [node_x, node_y + 1], SAFE_EDGE_COST)
                    elif not cells[1].is_high_crime_area or not cells[3].is_high_crime_area:
                        self.addNodeToPriorityQ(i, [node_x, node_y + 1], CRIME_EDGE_COST)

                    # south vertical
                    if cells[0].is_high_crime_area and cells[2].is_high_crime_area:
                        pass
                    elif not cells[0].is_high_crime_area and not cells[2].is_high_crime_area:
                        self.addNodeToPriorityQ(i, [node_x, node_y - 1], SAFE_EDGE_COST)
                    elif not cells[0].is_high_crime_area or not cells[2].is_high_crime_area:
                        self.addNodeToPriorityQ(i, [node_x, node_y - 1], CRIME_EDGE_COST)

    def getNodeByPos(self, pos):
        for i in self.nodes:
            if self.nodes[i] and self.nodes[i].grid_pos == pos:
                return self.nodes[i]
        return None

    def parseCrimeMap(self, crime_map):
        self.cells = []
        self.nodes = {}
        crimes = crime_map[0]

        # create node object for each vertex in grid
        # create cell object for each cell in grid. each cell has 4 vertices (nodes) to be used in A* path finder
        #
        #
        # p3 - - - p4
        # |  GRID  |
        # |  CELL  |
        # p1 - - - p2
        #
        # p1, p2, p3, p4 represent nodes and the cube represents a cell
        for i in range(0, len(crimes) + 1):
            for j in range(0, len(crimes) + 1):

                p1 = Node(i, j, crime_map[1][i], crime_map[2][j])

                # boundary checks at edges of graph
                if j + 1 < len(crime_map[2]):
                    p3 = Node(i, j + 1, crime_map[1][i], crime_map[2][j + 1])
                else:
                    p3 = Node(i, j + 1, crime_map[1][i], crime_map[2][j])

                # boundary checks at edges of graph
                if i + 1 < len(crime_map[1]) and j + 1 < len(crime_map[2]):
                    p2 = Node(i + 1, j, crime_map[1][i + 1], crime_map[2][j])
                    p4 = Node(i + 1, j + 1, crime_map[1][i + 1], crime_map[2][j + 1])
                elif i + 1 < len(crime_map[1]) and j + 1 == len(crime_map[2]):
                    p2 = Node(i + 1, j, crime_map[1][i + 1], crime_map[2][j])
                    p4 = Node(i + 2, 0, crime_map[1][i + 1], crime_map[2][j])
                elif i + 1 == len(crime_map[1]) and j + 1 < len(crime_map[2]):
                    p2 = Node(i + 1, j, crime_map[1][i], crime_map[2][j])
                    p4 = Node(i + 1, j + 1, crime_map[1][i], crime_map[2][j + 1])
                else:
                    p2 = Node(i + 1, j, crime_map[1][i], crime_map[2][j])
                    p4 = Node(i + 2, 0, crime_map[1][i], crime_map[2][j])

                # 1d representation of 2d data
                pos1 = i * len(crime_map[1]) + j
                pos2 = (i + 1) * len(crime_map[1]) + j
                pos3 = i * len(crime_map[1]) + j + 1
                pos4 = (i + 1) * len(crime_map[1]) + j + 1

                max_num_nodes = len(crime_map[1]) * len(crime_map[2])

                # add nodes to nodes dictionary
                if pos1 not in self.nodes and pos1 < max_num_nodes:
                    self.nodes[pos1] = p1
                if pos2 not in self.nodes and pos2 < max_num_nodes:
                    self.nodes[pos2] = p2
                if pos3 not in self.nodes and pos3 < max_num_nodes:
                    self.nodes[pos3] = p3
                if pos4 not in self.nodes and pos4 < max_num_nodes:
                    self.nodes[pos4] = p4

                # parse the cell data if we are not at the boundary of graph
                if i < len(crimes) and j < len(crimes) and j < len(crimes[i]) and i < len(crimes[i]):
                    num_crimes = crimes[i][j]

                    # determine if the cell is an obstruction if it has crimes that meet or exceed threshold
                    if num_crimes >= self.threshold_val:
                        cell = Cell(p1, p2, p3, p4, i, j, True, num_crimes)
                    else:
                        cell = Cell(p1, p2, p3, p4, i, j, False, num_crimes)

                    # add the newly created cell object to adjacency list of each of its vertices (nodes)
                    self.addAdjacentCellToNode(pos1, cell)
                    self.addAdjacentCellToNode(pos2, cell)
                    self.addAdjacentCellToNode(pos3, cell)
                    self.addAdjacentCellToNode(pos4, cell)

                    self.cells.append(cell)
        # parse each node to make the priorityQueue adjacencyList
        self.parseNodes()

        # for node in self.nodes:
        #     print(node)
        #     self.nodes[node].displayPriorityQ()

    def updateMap(self):
        # get the horizontal-vertical size of grid (X*Y)
        grid_dimensions = self.calcGridDimensions()

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

        self.parseCrimeMap(crime_map)

        padding = .001

        plt.xlim(self.total_bounds[0] - padding, self.total_bounds[2] + padding)
        plt.ylim(self.total_bounds[1] - padding, self.total_bounds[3] + padding)

        self.axmap.xaxis.set_minor_locator(tkr.AutoMinorLocator(n=5))
        self.axmap.yaxis.set_minor_locator(tkr.AutoMinorLocator(n=5))

        # update the tick values and # of crimes per cell
        self.crimes_per_cell = np.array(crime_map[0])
        self.grid_x_ticks = np.array(crime_map[1])
        self.grid_y_ticks = np.array(crime_map[2])

        if self.show_data:
            self.setGridDisplayData()
        self.displayPlotInfoTitle()

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

    def displayPlotInfoTitle(self):
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

    def drawLine(self, x1, y1, x2, y2):
        annotation = self.axmap.annotate("",
                                         xy=(x1, y1), xycoords='data',
                                         xytext=(x2, y2), textcoords='data',
                                         arrowprops=dict(arrowstyle="-", connectionstyle="arc3,rad=0."), )
        self.gridMarkings.append(annotation)

    def searchHeuristic(self):
        pass

    def aStarSearch(self):

        # sanity check
        if self.start[0] == -1 or self.start[1] == -1 or self.goal[0] == -1 or self.goal[1] == -1:
            print('Invalid value given, point found outside of boundaries of map!')
            return

        # Get start and goals nodes from parsed node dictionary
        start_node = self.getNodeByPos(self.start)
        goal_node = self.getNodeByPos(self.goal)

        # create empty open and closed lists
        open_list = []
        closed_list = []

        # add start node to open list
        open_list.append(start_node)

        # Loop until you find the end
        while len(open_list) > 0:
            # Get the current node
            current_node = open_list[0]
            current_index = 0
            for index, item in enumerate(open_list):
                if item.f < current_node.f:
                    temp = current_node
                    current_node = item
                    item.parent = temp
                    current_index = index

            # Pop current off open list, add to closed list
            open_list.pop(current_index)
            closed_list.append(current_node)

            # Found the goal
            if current_node == goal_node:
                path = []
                current = current_node
                while current is not None:
                    path.append(current)
                    current = current.parent
                path = path[::-1]
                for i in range(0, len(path) - 1):
                    x1 = path[i].lat_long[0]
                    y1 = path[i].lat_long[1]
                    x2 = path[i + 1].lat_long[0]
                    y2 = path[i + 1].lat_long[1]
                    self.drawLine(x1, y1, x2, y2)
                plt.show()

            # # Generate children
            # children = []
            # for adj_node in current_node.adjacent_nodes:  # Adjacent squares
            #
            #     # Get node position
            #     node_position = (current_node.grid_pos[0] + adj_node.grid_pos[0], current_node.grid_pos[1] + adj_node.grid_pos[1])
            #
            #     # # Make sure within range
            #     # if node_position[0] > (len(self.cells) - 1) or node_position[0] < 0 or node_position[1] > (
            #     #         len(self.cells[len(self.cells) - 1]) - 1) or node_position[1] < 0:
            #     #     continue
            #
            #     # Make sure walkable terrain
            #     if self.cells[node_position[0]][node_position[1]] != 0:
            #         continue
            #
            #     # Create new node
            #     new_node = Node(current_node, node_position)
            #
            #     # Append
            #     children.append(new_node)

            # Loop through children
            q = current_node.adjacent_nodes.queue
            for i in range(0, len(q)):

                child = q[i][1]

                # Child is on the closed list
                for closed_child in closed_list:
                    if child == closed_child:
                        continue

                # Create the f, g, and h values
                child.g = current_node.g + 1
                child.h = ((child.grid_pos[0] - goal_node.grid_pos[0]) ** 2) + (
                            (child.grid_pos[1] - goal_node.grid_pos[1]) ** 2)
                child.f = child.g + child.h

                # Child is already in the open list
                for open_node in open_list:
                    if child == open_node and child.g > open_node.g:
                        continue

                # Add the child to the open list
                open_list.append(child)

        # x1 = self.grid_x_ticks[self.start[0]]
        # y1 = self.grid_y_ticks[self.start[1]]
        # x2 = self.grid_x_ticks[self.goal[0]]
        # y2 = self.grid_y_ticks[self.goal[1]]
        # self.drawLine(x1, y1, x2, y2)


def main():
    crimes_map = CrimeMap()
    crimes_map.plotCrimeMap()


if __name__ == '__main__':
    main()
