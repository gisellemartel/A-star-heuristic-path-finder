# heuristic_search_map

Custom implementation of A* heuristic search algorithm for 2D grid representing
low and high crime areas of downtown Montreal. Calculates shortest path heuristcally between 2 points
on the grid

Lower threshold setting --> more obstacles
Higher threshold setting --> less obstacles

## Instructions to run program

- please install Python 3.8.1 in order to run the program
- Once python is installed, please install the following libraries in terminal or commandline in the root folder of the project
  - geopandas: `python -m pip install geopandas`
  - matplotlib: `python -m pip install matplotlib`
- Next, run the program by executing the following command in terminal or commandline in the root folder of the project:
  - `python CrimeMap.py`
- To test the path search algorithm, simply click on the grid to select your start point S and end/goal point G
- press the "hide/show data" button in order to toggle the view of grid data
- when executing the search, the animation will be faster when you toggle the data view off 